"""
COINSENTRA TRADING BRAIN V8 - WHALE INTELLIGENCE
Bybit USDT perpetuals | Android / Pydroid compatible

Install:
    pip install pybit

Run:
    python CoinSentra_Trading_Brain_V8.py

IMPORTANT:
- PAPER TRADING ONLY. This version does not submit, amend, or cancel real orders.
- Public Bybit market data is used; API keys are not required.
- TP/SL are 2% price moves from entry. At 8x leverage, a 2% favorable
  underlying move is approximately 16% gross position ROI before fees.
- Whale detection is an estimate based on public trades/order book. It cannot
  identify wallet owners or prove that a trade is a "whale".
"""

import json
import os
import time
import math
import logging
from datetime import datetime, timezone, date

try:
    from pybit.unified_trading import HTTP
except ImportError:
    raise SystemExit("Missing pybit. Install it with: pip install pybit")

# ======================== SETTINGS ========================

APP_NAME = "CoinSentra Trading Brain V8"
TESTNET = False                 # Public market data only
CATEGORY = "linear"
SYMBOLS = ["ETHUSDT", "XRPUSDT", "SOLUSDT", "SUIUSDT"]  # BTC removed

LEVERAGE = 8                    # Simulation leverage; no real leverage is set
TIMEFRAMES = ["5", "15"]        # 5-minute and 15-minute candles
SCAN_SECONDS = 60
KLINE_LIMIT = 220

TAKE_PROFIT_PCT = 0.02          # 2% underlying price move
STOP_LOSS_PCT = 0.02            # 2% underlying price move
DAILY_LOSS_LIMIT_USD = 6.0
STARTING_PAPER_BALANCE = 100.0
RISK_PER_TRADE_PCT = 0.005      # 0.5% of paper equity
MAX_POSITION_VALUE_PCT = 0.10   # Max notional = 10% of paper equity
MAX_OPEN_POSITIONS = 2

# Whale thresholds: trade notional is price * size.
# Lower thresholds may be useful for smaller-cap symbols like SUI.
WHALE_USD_THRESHOLDS = {
    "ETHUSDT": 100000,
    "XRPUSDT": 50000,
    "SOLUSDT": 50000,
    "SUIUSDT": 25000,
}
ORDERBOOK_DEPTH = 50
ORDERBOOK_TOP_LEVELS = 20
MIN_SIGNAL_SCORE = 70
DATA_DIR = os.environ.get("DATA_DIR", "/var/data")
os.makedirs(DATA_DIR, exist_ok=True)
STATE_FILE = os.path.join(DATA_DIR, "coinsentra_v8_state.json")
LOG_FILE = os.path.join(DATA_DIR, "coinsentra_v8.log")

# ======================== LOGGING / CLIENT ========================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

session = HTTP(testnet=TESTNET)


def now_text():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_state():
    default = {
        "balance": STARTING_PAPER_BALANCE,
        "day": str(date.today()),
        "daily_realized": 0.0,
        "positions": {},
        "closed_trades": [],
        "last_oi": {},
        "last_whale": {},
        "scan_count": 0,
    }
    if not os.path.exists(STATE_FILE):
        return default
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        for key, value in default.items():
            if key not in saved:
                saved[key] = value
        return saved
    except Exception:
        logging.exception("Could not read state file; starting fresh.")
        return default


state = load_state()


def save_state():
    temp = STATE_FILE + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(temp, STATE_FILE)


def api_result(response):
    if not isinstance(response, dict) or response.get("retCode") != 0:
        raise RuntimeError("Bybit API error: " + str(response))
    return response.get("result", {})


# ======================== MARKET DATA ========================

def get_klines(symbol, interval):
    result = api_result(session.get_kline(
        category=CATEGORY, symbol=symbol, interval=interval, limit=KLINE_LIMIT
    ))
    rows = result.get("list", [])
    rows = list(reversed(rows))
    candles = []
    for row in rows:
        # Bybit: start, open, high, low, close, volume, turnover
        candles.append({
            "time": int(row[0]),
            "open": safe_float(row[1]),
            "high": safe_float(row[2]),
            "low": safe_float(row[3]),
            "close": safe_float(row[4]),
            "volume": safe_float(row[5]),
            "turnover": safe_float(row[6]),
        })
    return candles


def get_orderbook(symbol):
    result = api_result(session.get_orderbook(
        category=CATEGORY, symbol=symbol, limit=ORDERBOOK_DEPTH
    ))
    bids = [(safe_float(x[0]), safe_float(x[1])) for x in result.get("b", [])]
    asks = [(safe_float(x[0]), safe_float(x[1])) for x in result.get("a", [])]
    return bids, asks


def get_recent_trades(symbol):
    result = api_result(session.get_public_trade_history(
        category=CATEGORY, symbol=symbol, limit=100
    ))
    return result.get("list", [])


def get_ticker(symbol):
    result = api_result(session.get_tickers(category=CATEGORY, symbol=symbol))
    rows = result.get("list", [])
    if not rows:
        raise RuntimeError("No ticker returned for " + symbol)
    return rows[0]


def get_open_interest(symbol):
    result = api_result(session.get_open_interest(
        category=CATEGORY, symbol=symbol, intervalTime="5min", limit=2
    ))
    rows = result.get("list", [])
    if not rows:
        return 0.0
    return safe_float(rows[0].get("openInterest"))


# ======================== INDICATORS ========================

def ema(values, period):
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1.0)
    current = sum(values[:period]) / period
    for value in values[period:]:
        current = alpha * value + (1 - alpha) * current
    return current


def rsi(values, period=14):
    if len(values) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(values):
    fast = ema(values, 12)
    slow = ema(values, 26)
    if fast is None or slow is None:
        return None
    return fast - slow


def bollinger(values, period=20):
    if len(values) < period:
        return None, None, None
    sample = values[-period:]
    mid = sum(sample) / period
    variance = sum((x - mid) ** 2 for x in sample) / period
    sd = math.sqrt(variance)
    return mid + 2 * sd, mid, mid - 2 * sd


def adx(candles, period=14):
    if len(candles) < period * 2 + 1:
        return None
    trs, plus_dm, minus_dm = [], [], []
    for i in range(1, len(candles)):
        cur, prev = candles[i], candles[i - 1]
        up = cur["high"] - prev["high"]
        down = prev["low"] - cur["low"]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        trs.append(max(
            cur["high"] - cur["low"],
            abs(cur["high"] - prev["close"]),
            abs(cur["low"] - prev["close"])
        ))
    dx_values = []
    for i in range(period, len(trs)):
        tr_sum = sum(trs[i - period + 1:i + 1])
        if tr_sum <= 0:
            continue
        pdi = 100 * sum(plus_dm[i - period + 1:i + 1]) / tr_sum
        mdi = 100 * sum(minus_dm[i - period + 1:i + 1]) / tr_sum
        denom = pdi + mdi
        dx_values.append(0.0 if denom == 0 else 100 * abs(pdi - mdi) / denom)
    if len(dx_values) < period:
        return None
    return sum(dx_values[-period:]) / period


def analyze_timeframe(candles):
    closes = [c["close"] for c in candles]
    if len(closes) < 60:
        raise RuntimeError("Not enough candle history")
    price = closes[-1]
    e9, e21, e50 = ema(closes, 9), ema(closes, 21), ema(closes, 50)
    r = rsi(closes)
    m = macd(closes)
    upper, middle, lower = bollinger(closes)
    a = adx(candles)
    recent_vol = [c["volume"] for c in candles[-21:-1]]
    avg_vol = sum(recent_vol) / len(recent_vol) if recent_vol else 0
    vol_ratio = candles[-1]["volume"] / avg_vol if avg_vol > 0 else 1
    return {
        "price": price, "ema9": e9, "ema21": e21, "ema50": e50,
        "rsi": r, "macd": m, "bb_upper": upper, "bb_mid": middle,
        "bb_lower": lower, "adx": a, "volume_ratio": vol_ratio,
        "bull": e9 > e21 > e50,
        "bear": e9 < e21 < e50,
    }


# ======================== WHALE INTELLIGENCE ========================

def whale_analysis(symbol, trades, bids, asks, open_interest):
    threshold = WHALE_USD_THRESHOLDS.get(symbol, 50000)
    buy_usd = 0.0
    sell_usd = 0.0
    large_trades = []
    for trade in trades:
        price = safe_float(trade.get("price"))
        size = safe_float(trade.get("size"))
        value = price * size
        side = str(trade.get("side", "")).lower()
        if value >= threshold:
            large_trades.append((value, side, price))
            if side == "buy":
                buy_usd += value
            elif side == "sell":
                sell_usd += value

    total_depth = sum(p * q for p, q in bids[:ORDERBOOK_TOP_LEVELS]) + \
                  sum(p * q for p, q in asks[:ORDERBOOK_TOP_LEVELS])
    bid_depth = sum(p * q for p, q in bids[:ORDERBOOK_TOP_LEVELS])
    ask_depth = sum(p * q for p, q in asks[:ORDERBOOK_TOP_LEVELS])
    imbalance = (bid_depth / total_depth) if total_depth > 0 else 0.5

    # Largest visible bid/ask level compared with the average top-level notional.
    bid_notionals = [p * q for p, q in bids[:ORDERBOOK_TOP_LEVELS]]
    ask_notionals = [p * q for p, q in asks[:ORDERBOOK_TOP_LEVELS]]
    avg_bid = sum(bid_notionals) / len(bid_notionals) if bid_notionals else 0
    avg_ask = sum(ask_notionals) / len(ask_notionals) if ask_notionals else 0
    bid_wall = max(bid_notionals) if bid_notionals else 0
    ask_wall = max(ask_notionals) if ask_notionals else 0

    previous_oi = safe_float(state["last_oi"].get(symbol))
    oi_change_pct = ((open_interest - previous_oi) / previous_oi * 100
                     if previous_oi > 0 else 0.0)
    state["last_oi"][symbol] = open_interest

    whale_bias = buy_usd - sell_usd
    if whale_bias > threshold:
        whale_signal = "BUY PRESSURE"
    elif whale_bias < -threshold:
        whale_signal = "SELL PRESSURE"
    else:
        whale_signal = "NO CLEAR WHALE FLOW"

    return {
        "buy_usd": buy_usd, "sell_usd": sell_usd,
        "count": len(large_trades), "imbalance": imbalance,
        "bid_wall": bid_wall, "ask_wall": ask_wall,
        "avg_bid": avg_bid, "avg_ask": avg_ask,
        "oi": open_interest, "oi_change_pct": oi_change_pct,
        "signal": whale_signal, "threshold": threshold,
    }


# ======================== SIGNAL ENGINE ========================

def build_signal(tf5, tf15, whale, funding):
    score = 50
    reasons = []

    if tf5["bull"]:
        score += 12
        reasons.append("5m EMA bullish")
    elif tf5["bear"]:
        score -= 12
        reasons.append("5m EMA bearish")

    if tf15["bull"]:
        score += 16
        reasons.append("15m EMA bullish")
    elif tf15["bear"]:
        score -= 16
        reasons.append("15m EMA bearish")

    if tf5["rsi"] is not None:
        if 52 <= tf5["rsi"] <= 70:
            score += 6
            reasons.append("5m RSI momentum positive")
        elif tf5["rsi"] < 35:
            score += 2
            reasons.append("5m RSI oversold zone")
        elif tf5["rsi"] > 75:
            score -= 7
            reasons.append("5m RSI overheated")

    if tf15["rsi"] is not None:
        if 50 <= tf15["rsi"] <= 68:
            score += 5
        elif tf15["rsi"] < 30:
            score += 2
        elif tf15["rsi"] > 75:
            score -= 5

    if tf5["macd"] is not None and tf5["macd"] > 0:
        score += 5
        reasons.append("MACD positive")
    elif tf5["macd"] is not None and tf5["macd"] < 0:
        score -= 5
        reasons.append("MACD negative")

    if tf5["adx"] is not None and tf5["adx"] >= 20:
        score += 4 if tf5["bull"] else (-4 if tf5["bear"] else 0)
        reasons.append("ADX trend strength")

    if tf5["volume_ratio"] >= 1.8:
        score += 5 if tf5["bull"] else (-5 if tf5["bear"] else 0)
        reasons.append("Volume spike")

    if whale["signal"] == "BUY PRESSURE":
        score += 10
        reasons.append("Large trades favor buyers")
    elif whale["signal"] == "SELL PRESSURE":
        score -= 10
        reasons.append("Large trades favor sellers")

    if whale["imbalance"] >= 0.60:
        score += 6
        reasons.append("Order book bid imbalance")
    elif whale["imbalance"] <= 0.40:
        score -= 6
        reasons.append("Order book ask imbalance")

    if whale["oi_change_pct"] >= 1:
        reasons.append("Open interest rising")
        if score > 55:
            score += 3
        elif score < 45:
            score -= 3
    elif whale["oi_change_pct"] <= -1:
        reasons.append("Open interest falling")

    if funding > 0.0008:
        score -= 3
        reasons.append("Crowded long funding risk")
    elif funding < -0.0008:
        score += 3
        reasons.append("Negative funding")

    score = max(0, min(100, score))
    if score >= MIN_SIGNAL_SCORE:
        direction = "LONG"
    elif score <= 100 - MIN_SIGNAL_SCORE:
        direction = "SHORT"
    else:
        direction = "WAIT"

    return score, direction, reasons


# ======================== PAPER TRADING ========================

def unrealized_pnl(position, price):
    direction = 1 if position["side"] == "LONG" else -1
    return (price - position["entry"]) * direction * position["qty"]


def close_position(symbol, price, reason):
    position = state["positions"].pop(symbol, None)
    if not position:
        return
    pnl = unrealized_pnl(position, price)
    # Simple estimated round-trip fee at 0.055% per side on notional.
    fees = position["entry"] * position["qty"] * 0.0011
    net = pnl - fees
    state["balance"] += net
    state["daily_realized"] += net
    trade = {
        "symbol": symbol, "side": position["side"],
        "entry": position["entry"], "exit": price,
        "qty": position["qty"], "gross_pnl": pnl,
        "estimated_fees": fees, "net_pnl": net,
        "reason": reason, "closed_at": now_text(),
    }
    state["closed_trades"].append(trade)
    state["closed_trades"] = state["closed_trades"][-200:]
    logging.info("PAPER CLOSE %s %s net=%.4f reason=%s",
                 symbol, position["side"], net, reason)
    print("CLOSED", symbol, position["side"], reason,
          "net PnL $%.3f" % net)


def manage_positions(prices):
    for symbol in list(state["positions"].keys()):
        price = prices.get(symbol)
        if not price:
            continue
        p = state["positions"][symbol]
        if p["side"] == "LONG":
            if price >= p["tp"]:
                close_position(symbol, price, "TAKE PROFIT")
            elif price <= p["sl"]:
                close_position(symbol, price, "STOP LOSS")
        else:
            if price <= p["tp"]:
                close_position(symbol, price, "TAKE PROFIT")
            elif price >= p["sl"]:
                close_position(symbol, price, "STOP LOSS")


def open_paper_position(symbol, side, price, score):
    if symbol in state["positions"] or len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return
    if state["daily_realized"] <= -DAILY_LOSS_LIMIT_USD:
        print("DAILY LOSS LIMIT reached. New entries disabled.")
        return

    equity = state["balance"]
    risk_usd = max(0.0, equity * RISK_PER_TRADE_PCT)
    stop_distance = price * STOP_LOSS_PCT
    qty_by_risk = risk_usd / stop_distance if stop_distance > 0 else 0
    max_notional = equity * MAX_POSITION_VALUE_PCT * LEVERAGE
    qty_by_cap = max_notional / price if price > 0 else 0
    qty = min(qty_by_risk, qty_by_cap)
    if qty <= 0:
        return

    if side == "LONG":
        tp, sl = price * (1 + TAKE_PROFIT_PCT), price * (1 - STOP_LOSS_PCT)
    else:
        tp, sl = price * (1 - TAKE_PROFIT_PCT), price * (1 + STOP_LOSS_PCT)

    state["positions"][symbol] = {
        "side": side, "entry": price, "qty": qty,
        "tp": tp, "sl": sl, "score": score,
        "leverage": LEVERAGE, "opened_at": now_text(),
    }
    logging.info("PAPER OPEN %s %s entry=%s qty=%s score=%s",
                 symbol, side, price, qty, score)
    print("PAPER OPEN:", symbol, side, "entry", price,
          "qty", round(qty, 6), "TP", tp, "SL", sl)


# ======================== DISPLAY / MAIN LOOP ========================

def print_dashboard(rows):
    print("\n" + "=" * 72)
    print(APP_NAME, "|", now_text())
    print("PAPER ONLY | 8x simulated | TP 2% | SL 2% | daily loss limit $6")
    print("Symbols:", ", ".join(SYMBOLS), "| Timeframes: 5m + 15m")
    print("-" * 72)
    print("Paper balance: $%.2f | Daily realized: $%.2f | Open: %d/%d" %
          (state["balance"], state["daily_realized"],
           len(state["positions"]), MAX_OPEN_POSITIONS))
    print("-" * 72)
    for row in rows:
        print("%-8s $%-12.6g SCORE %3d %-5s | 5m RSI %5.1f | 15m RSI %5.1f" %
              (row["symbol"], row["price"], row["score"], row["direction"],
               row["rsi5"], row["rsi15"]))
        print("  Whale: %-18s buys $%-10.0f sells $%-10.0f book %.2f OI %+.2f%%" %
              (row["whale"]["signal"], row["whale"]["buy_usd"],
               row["whale"]["sell_usd"], row["whale"]["imbalance"],
               row["whale"]["oi_change_pct"]))
        if row["reasons"]:
            print("  Signals:", ", ".join(row["reasons"]))
    print("-" * 72)
    if state["positions"]:
        print("OPEN PAPER POSITIONS:")
        for symbol, p in state["positions"].items():
            print(" ", symbol, p["side"], "entry", p["entry"],
                  "TP", p["tp"], "SL", p["sl"], "qty", round(p["qty"], 6))
    else:
        print("No open paper positions.")
    print("=" * 72)


def run_scan():
    rows = []
    prices = {}
    for symbol in SYMBOLS:
        try:
            candles5 = get_klines(symbol, "5")
            candles15 = get_klines(symbol, "15")
            tf5 = analyze_timeframe(candles5)
            tf15 = analyze_timeframe(candles15)
            ticker = get_ticker(symbol)
            price = safe_float(ticker.get("lastPrice"), tf5["price"])
            prices[symbol] = price

            bids, asks = get_orderbook(symbol)
            trades = get_recent_trades(symbol)
            oi = get_open_interest(symbol)
            whale = whale_analysis(symbol, trades, bids, asks, oi)
            funding = safe_float(ticker.get("fundingRate"))

            score, direction, reasons = build_signal(tf5, tf15, whale, funding)
            rows.append({
                "symbol": symbol, "price": price, "score": score,
                "direction": direction, "rsi5": tf5["rsi"],
                "rsi15": tf15["rsi"], "whale": whale, "reasons": reasons,
            })
        except Exception as exc:
            logging.exception("Scan failed for %s", symbol)
            print("Data error", symbol, ":", exc)

    manage_positions(prices)

    # Entries only when both timeframes and combined signals agree strongly.
    for row in rows:
        if row["direction"] in ("LONG", "SHORT"):
            open_paper_position(row["symbol"], row["direction"],
                                row["price"], row["score"])

    state["latest_rows"] = rows
    state["last_scan_at"] = now_text()
    print_dashboard(rows)
    state["scan_count"] += 1
    save_state()


def main():
    print(APP_NAME)
    print("Starting public Bybit market-data scanner in PAPER mode.")
    print("No API keys needed. No real orders can be placed by this file.")
    print("Press CTRL+C to stop. State is saved in", STATE_FILE)

    while True:
        today = str(date.today())
        if state["day"] != today:
            state["day"] = today
            state["daily_realized"] = 0.0
            save_state()

        try:
            run_scan()
        except KeyboardInterrupt:
            print("\nStopped by user.")
            save_state()
            break
        except Exception as exc:
            logging.exception("Main scan error")
            print("Main scan error:", exc)

        time.sleep(SCAN_SECONDS)


if __name__ == "__main__":
    main()
