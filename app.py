
import os
import json
import time
import math
import threading
import logging
from datetime import datetime, timezone, date

from flask import Flask, jsonify, render_template_string
from pybit.unified_trading import HTTP

# ============================================================
# COINSENTRA TRADING BRAIN V14.1
# BYBIT | RENDER | PAPER TRADING ONLY
# ============================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("CoinSentra")

# ---------------- TRADING SETTINGS ----------------

SYMBOLS = [
    "ETHUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "SUIUSDT"
]

CATEGORY = "linear"

LEVERAGE = 8
TAKE_PROFIT_PCT = 2.0
STOP_LOSS_PCT = 2.0

MAX_OPEN_POSITIONS = 2
DAILY_LOSS_LIMIT = 6.0

START_BALANCE = 10000.0
RISK_PER_TRADE = 3.0
MAX_EQUITY_EXPOSURE = 10.0

SCAN_INTERVAL = 30
ENTRY_SCORE = 68
FEE_RATE = 0.00055

DATA_DIR = os.getenv("DATA_DIR", "/tmp")
os.makedirs(DATA_DIR, exist_ok=True)

STATE_FILE = os.path.join(
    DATA_DIR,
    "coinsentra_v14.json"
)

# KEEPING YOUR EXISTING API CONFIGURATION
API_KEY = os.getenv("vA9UkYkaoBHPySdPT9", "").strip()
API_SECRET = os.getenv(
    "8uZSZDZdn8ra8xX4ephBuI7yOkyLpGKyot7h",
    ""
).strip()

# ---------------- BYBIT CONNECTIONS ----------------

public_api = HTTP(
    testnet=False,
    timeout=15
)

private_api = None

if API_KEY and API_SECRET:
    private_api = HTTP(
        api_key=API_KEY,
        api_secret=API_SECRET,
        testnet=False,
        timeout=15
    )

# ---------------- GLOBAL STATE ----------------

lock = threading.RLock()
worker_started = False

state = {
    "running": True,
    "balance": START_BALANCE,
    "equity": START_BALANCE,
    "realized_pnl": 0.0,
    "unrealized_pnl": 0.0,
    "total_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0,
    "open_positions": [],
    "completed_trades": [],
    "market": {},
    "last_scan": None,
    "last_error": "",
    "status": "STARTING",
    "daily_date": str(date.today()),
    "daily_pnl": 0.0,
    "daily_locked": False,
    "private_account": "NOT CONNECTED",
    "version": "14.1"
}


# ============================================================
# PERSISTENCE
# ============================================================

def save_state():
    try:
        with lock:
            snapshot = json.loads(json.dumps(state))

        temp_file = STATE_FILE + ".tmp"

        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)

        os.replace(temp_file, STATE_FILE)

    except Exception as e:
        log.exception("State save failed: %s", e)


def load_state():
    global state

    if not os.path.exists(STATE_FILE):
        return

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)

        with lock:
            for key in state:
                if key in saved:
                    state[key] = saved[key]

            state["running"] = True

            if not isinstance(state["open_positions"], list):
                state["open_positions"] = []

            if not isinstance(state["completed_trades"], list):
                state["completed_trades"] = []

            if not isinstance(state["market"], dict):
                state["market"] = {}

        log.info("Saved trading state loaded")

    except Exception as e:
        log.exception("Could not load saved state: %s", e)


# ============================================================
# GENERAL HELPERS
# ============================================================

def now_text():
    return datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def safe_float(value, default=0.0):
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return default


def clamp(value, low, high):
    return max(low, min(high, value))


def api_result(response):
    if not isinstance(response, dict):
        raise RuntimeError("Invalid Bybit response")

    if response.get("retCode", 0) != 0:
        raise RuntimeError(
            str(response.get("retMsg", "Bybit API error"))
        )

    return response.get("result", {})


def reset_daily_if_needed():
    today = str(date.today())

    with lock:
        if state["daily_date"] != today:
            state["daily_date"] = today
            state["daily_pnl"] = 0.0
            state["daily_locked"] = False

    save_state()


# ============================================================
# BYBIT MARKET DATA
# ============================================================

def get_ticker(symbol):
    response = public_api.get_tickers(
        category=CATEGORY,
        symbol=symbol
    )

    result = api_result(response)
    items = result.get("list", [])

    if not items:
        raise RuntimeError("No ticker data for " + symbol)

    return items[0]


def get_price(symbol):
    ticker = get_ticker(symbol)
    price = safe_float(ticker.get("lastPrice"))

    if price <= 0:
        raise RuntimeError("Invalid price for " + symbol)

    return price


def get_candles(symbol, interval="5", limit=150):
    response = public_api.get_kline(
        category=CATEGORY,
        symbol=symbol,
        interval=interval,
        limit=limit
    )

    result = api_result(response)
    rows = result.get("list", [])

    # Bybit returns newest candles first.
    rows = list(reversed(rows))

    candles = []

    for row in rows:
        if len(row) < 6:
            continue

        candles.append({
            "time": int(row[0]),
            "open": safe_float(row[1]),
            "high": safe_float(row[2]),
            "low": safe_float(row[3]),
            "close": safe_float(row[4]),
            "volume": safe_float(row[5])
        })

    if len(candles) < 30:
        raise RuntimeError("Not enough candle data")

    return candles


def get_orderbook(symbol):
    response = public_api.get_orderbook(
        category=CATEGORY,
        symbol=symbol,
        limit=25
    )

    result = api_result(response)

    bids = result.get("b", [])
    asks = result.get("a", [])

    bid_volume = sum(
        safe_float(x[1]) for x in bids
    )
    ask_volume = sum(
        safe_float(x[1]) for x in asks
    )

    total = bid_volume + ask_volume

    imbalance = 0.0

    if total > 0:
        imbalance = (
            (bid_volume - ask_volume) / total
        ) * 100

    return {
        "bid_volume": bid_volume,
        "ask_volume": ask_volume,
        "imbalance": imbalance
    }


def get_funding(symbol):
    try:
        response = public_api.get_tickers(
            category=CATEGORY,
            symbol=symbol
        )

        items = api_result(response).get("list", [])

        if not items:
            return 0.0

        return safe_float(items[0].get("fundingRate"))

    except Exception:
        return 0.0


def get_open_interest(symbol):
    try:
        response = public_api.get_open_interest(
            category=CATEGORY,
            symbol=symbol,
            intervalTime="5min",
            limit=2
        )

        items = api_result(response).get("list", [])

        if not items:
            return 0.0

        return safe_float(items[0].get("openInterest"))

    except Exception:
        return 0.0


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def ema(values, period):
    if len(values) < period:
        return []

    multiplier = 2 / (period + 1)

    result = [sum(values[:period]) / period]

    for value in values[period:]:
        result.append(
            (value - result[-1]) * multiplier + result[-1]
        )

    return result


def calculate_rsi(values, period=14):
    if len(values) <= period:
        return 50.0

    gains = []
    losses = []

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (
            avg_gain * (period - 1) + gains[i]
        ) / period

        avg_loss = (
            avg_loss * (period - 1) + losses[i]
        ) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def calculate_macd(values):
    fast = ema(values, 12)
    slow = ema(values, 26)

    if not fast or not slow:
        return 0.0, 0.0

    offset = len(values) - len(fast)
    macd_values = []

    for i in range(len(slow)):
        macd_values.append(
            fast[i + offset] - slow[i]
        )

    signal = ema(macd_values, 9)

    if not signal:
        return macd_values[-1], 0.0

    return macd_values[-1], signal[-1]


def calculate_bollinger(values, period=20):
    if len(values) < period:
        return 0.0, 0.0, 0.0

    recent = values[-period:]
    middle = sum(recent) / period

    variance = sum(
        (x - middle) ** 2 for x in recent
    ) / period

    deviation = math.sqrt(variance)

    return (
        middle + 2 * deviation,
        middle,
        middle - 2 * deviation
    )


def calculate_adx(candles, period=14):
    if len(candles) < period + 2:
        return 0.0

    plus_dm = []
    minus_dm = []
    true_ranges = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        up_move = current["high"] - previous["high"]
        down_move = previous["low"] - current["low"]

        plus_dm.append(
            up_move if up_move > down_move and up_move > 0 else 0
        )

        minus_dm.append(
            down_move if down_move > up_move and down_move > 0 else 0
        )

        true_ranges.append(max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"])
        ))

    tr = sum(true_ranges[-period:])

    if tr <= 0:
        return 0.0

    plus_di = 100 * sum(plus_dm[-period:]) / tr
    minus_di = 100 * sum(minus_dm[-period:]) / tr

    denominator = plus_di + minus_di

    if denominator <= 0:
        return 0.0

    return abs(plus_di - minus_di) / denominator * 100


def calculate_indicators(candles):
    closes = [x["close"] for x in candles]
    volumes = [x["volume"] for x in candles]

    e9 = ema(closes, 9)
    e21 = ema(closes, 21)
    e50 = ema(closes, 50)

    macd_value, macd_signal = calculate_macd(closes)

    upper, middle, lower = calculate_bollinger(closes)

    avg_volume = (
        sum(volumes[-20:]) / min(20, len(volumes))
        if volumes else 0
    )

    current_volume = volumes[-1] if volumes else 0

    volume_ratio = (
        current_volume / avg_volume
        if avg_volume > 0 else 1
    )

    return {
        "price": closes[-1],
        "rsi": calculate_rsi(closes),
        "ema9": e9[-1] if e9 else 0,
        "ema21": e21[-1] if e21 else 0,
        "ema50": e50[-1] if e50 else 0,
        "macd": macd_value,
        "macd_signal": macd_signal,
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "adx": calculate_adx(candles),
        "volume_ratio": volume_ratio
    }


# ============================================================
# SIGNAL ENGINE
# ============================================================

def score_signal(ind, confirmation, orderbook, funding):
    long_score = 0
    short_score = 0

    price = ind["price"]

    # EMA trend
    if ind["ema9"] > ind["ema21"]:
        long_score += 15
    elif ind["ema9"] < ind["ema21"]:
        short_score += 15

    if ind["ema21"] > ind["ema50"]:
        long_score += 10
    elif ind["ema21"] < ind["ema50"]:
        short_score += 10

    # RSI
    if 50 <= ind["rsi"] <= 68:
        long_score += 12
    elif 32 <= ind["rsi"] < 50:
        short_score += 12

    # MACD
    if ind["macd"] > ind["macd_signal"]:
        long_score += 12
    elif ind["macd"] < ind["macd_signal"]:
        short_score += 12

    # Volume confirmation
    if ind["volume_ratio"] >= 1.2:
        if price >= ind["ema21"]:
            long_score += 8
        else:
            short_score += 8

    # ADX trend strength
    if ind["adx"] >= 20:
        if ind["ema9"] > ind["ema21"]:
            long_score += 8
        elif ind["ema9"] < ind["ema21"]:
            short_score += 8

    # 15-minute confirmation
    if confirmation["ema9"] > confirmation["ema21"]:
        long_score += 15
    elif confirmation["ema9"] < confirmation["ema21"]:
        short_score += 15

    # Order book
    imbalance = orderbook["imbalance"]

    if imbalance > 10:
        long_score += 10
    elif imbalance < -10:
        short_score += 10

    # Funding extreme penalty
    if funding > 0.001:
        long_score -= 5

    if funding < -0.001:
        short_score -= 5

    long_score = int(clamp(long_score, 0, 100))
    short_score = int(clamp(short_score, 0, 100))

    if long_score >= ENTRY_SCORE and long_score > short_score:
        return "LONG", long_score

    if short_score >= ENTRY_SCORE and short_score > long_score:
        return "SHORT", short_score

    return "WAIT", max(long_score, short_score)


# ============================================================
# PAPER TRADING
# ============================================================

def calculate_equity():
    with lock:
        balance = state["balance"]
        positions = list(state["open_positions"])

    unrealized = sum(
        safe_float(p.get("unrealized_pnl"))
        for p in positions
    )

    with lock:
        state["unrealized_pnl"] = unrealized
        state["equity"] = balance + unrealized

    return balance + unrealized


def current_exposure():
    with lock:
        positions = list(state["open_positions"])

    return sum(
        safe_float(p.get("notional"))
        for p in positions
    )


def open_paper_position(symbol, side, score, price, indicators):
    if side not in ("LONG", "SHORT") or price <= 0:
        return False

    with lock:
        if not state["running"] or state["daily_locked"]:
            return False

        if len(state["open_positions"]) >= MAX_OPEN_POSITIONS:
            return False

        if any(
            p["symbol"] == symbol
            for p in state["open_positions"]
        ):
            return False

        equity = state["equity"]

    # Position notional is sized so a 2% adverse move
    # is approximately the configured $3 risk before fees.
    risk_notional = RISK_PER_TRADE / (
        STOP_LOSS_PCT / 100
    )

    exposure_limit = equity * MAX_EQUITY_EXPOSURE / 100
    remaining_exposure = max(
        0,
        exposure_limit - current_exposure()
    )

    notional = min(risk_notional, remaining_exposure)

    if notional <= 0:
        return False

    quantity = notional / price

    if side == "LONG":
        stop = price * (1 - STOP_LOSS_PCT / 100)
        target = price * (1 + TAKE_PROFIT_PCT / 100)
    else:
        stop = price * (1 + STOP_LOSS_PCT / 100)
        target = price * (1 - TAKE_PROFIT_PCT / 100)

    position = {
        "id": str(int(time.time() * 1000)) + symbol,
        "symbol": symbol,
        "side": side,
        "entry_price": price,
        "current_price": price,
        "quantity": quantity,
        "notional": notional,
        "leverage": LEVERAGE,
        "margin": notional / LEVERAGE,
        "stop_loss": stop,
        "take_profit": target,
        "score": score,
        "entry_rsi": indicators["rsi"],
        "entry_time": now_text(),
        "unrealized_pnl": 0.0
    }

    with lock:
        state["open_positions"].append(position)

    save_state()

    log.info(
        "PAPER OPEN %s %s at %.8f score=%s",
        side, symbol, price, score
    )

    return True


def close_paper_position(position_id, exit_price, reason):
    with lock:
        position = next(
            (
                p for p in state["open_positions"]
                if p["id"] == position_id
            ),
            None
        )

        if position is None:
            return False

        state["open_positions"].remove(position)

    entry = position["entry_price"]
    quantity = position["quantity"]
    notional = position["notional"]

    if position["side"] == "LONG":
        gross_pnl = (exit_price - entry) * quantity
    else:
        gross_pnl = (entry - exit_price) * quantity

    fees = notional * FEE_RATE
    fees += (quantity * exit_price) * FEE_RATE

    net_pnl = gross_pnl - fees

    trade = {
        "id": position["id"],
        "symbol": position["symbol"],
        "side": position["side"],
        "entry_price": entry,
        "exit_price": exit_price,
        "quantity": quantity,
        "notional": notional,
        "leverage": position["leverage"],
        "score": position["score"],
        "gross_pnl": gross_pnl,
        "fees": fees,
        "pnl": net_pnl,
        "result": "WIN" if net_pnl > 0 else "LOSS",
        "reason": reason,
        "entry_time": position["entry_time"],
        "exit_time": now_text()
    }

    with lock:
        state["balance"] += net_pnl
        state["realized_pnl"] += net_pnl
        state["daily_pnl"] += net_pnl
        state["total_trades"] += 1

        if net_pnl > 0:
            state["winning_trades"] += 1
        else:
            state["losing_trades"] += 1

        state["completed_trades"].insert(0, trade)

        # Keep the latest 500 completed trades.
        state["completed_trades"] = (
            state["completed_trades"][:500]
        )

        if state["daily_pnl"] <= -DAILY_LOSS_LIMIT:
            state["daily_locked"] = True
            state["status"] = "DAILY LOSS LIMIT REACHED"

    log.info(
        "PAPER CLOSE %s %s PNL=%.4f reason=%s",
        position["side"],
        position["symbol"],
        net_pnl,
        reason
    )

    calculate_equity()
    save_state()

    return True


def monitor_positions():
    with lock:
        positions = list(state["open_positions"])

    for position in positions:
        try:
            price = get_price(position["symbol"])

            entry = position["entry_price"]
            quantity = position["quantity"]
            notional = position["notional"]

            if position["side"] == "LONG":
                gross = (price - entry) * quantity
                stop_hit = price <= position["stop_loss"]
                target_hit = price >= position["take_profit"]
            else:
                gross = (entry - price) * quantity
                stop_hit = price >= position["stop_loss"]
                target_hit = price <= position["take_profit"]

            fees = notional * FEE_RATE
            fees += quantity * price * FEE_RATE

            unrealized = gross - fees

            with lock:
                for p in state["open_positions"]:
                    if p["id"] == position["id"]:
                        p["current_price"] = price
                        p["unrealized_pnl"] = unrealized

            if stop_hit:
                close_paper_position(
                    position["id"],
                    price,
                    "STOP LOSS"
                )

            elif target_hit:
                close_paper_position(
                    position["id"],
                    price,
                    "TAKE PROFIT"
                )

        except Exception as e:
            log.warning(
                "Position monitor %s: %s",
                position["symbol"],
                e
            )

    calculate_equity()


# ============================================================
# PRIVATE ACCOUNT STATUS - READ ONLY
# ============================================================

def check_private_account():
    if private_api is None:
        with lock:
            state["private_account"] = "NOT CONFIGURED"
        return

    try:
        response = private_api.get_wallet_balance(
            accountType="UNIFIED"
        )

        api_result(response)

        with lock:
            state["private_account"] = "CONNECTED - READ ONLY"

    except Exception as e:
        with lock:
            state["private_account"] = "CHECK FAILED"

        log.warning("Private account check failed: %s", e)


# ============================================================
# SCANNER
# ============================================================

def scan_symbol(symbol):
    candles_5m = get_candles(symbol, "5", 150)
    candles_15m = get_candles(symbol, "15", 150)

    ind = calculate_indicators(candles_5m)
    confirmation = calculate_indicators(candles_15m)

    orderbook = get_orderbook(symbol)
    funding = get_funding(symbol)
    open_interest = get_open_interest(symbol)

    side, score = score_signal(
        ind,
        confirmation,
        orderbook,
        funding
    )

    ticker = get_ticker(symbol)

    result = {
        "symbol": symbol,
        "price": ind["price"],
        "change24h": safe_float(
            ticker.get("price24hPcnt")
        ) * 100,
        "volume24h": safe_float(
            ticker.get("volume24h")
        ),
        "turnover24h": safe_float(
            ticker.get("turnover24h")
        ),
        "rsi": ind["rsi"],
        "ema9": ind["ema9"],
        "ema21": ind["ema21"],
        "ema50": ind["ema50"],
        "macd": ind["macd"],
        "macd_signal": ind["macd_signal"],
        "adx": ind["adx"],
        "volume_ratio": ind["volume_ratio"],
        "bb_upper": ind["bb_upper"],
        "bb_middle": ind["bb_middle"],
        "bb_lower": ind["bb_lower"],
        "orderbook_imbalance": orderbook["imbalance"],
        "bid_volume": orderbook["bid_volume"],
        "ask_volume": orderbook["ask_volume"],
        "funding": funding,
        "open_interest": open_interest,
        "signal": side,
        "score": score,
        "confirmation": (
            "BULLISH"
            if confirmation["ema9"] > confirmation["ema21"]
            else "BEARISH"
        ),
        "updated": now_text()
    }

    with lock:
        state["market"][symbol] = result

    if side in ("LONG", "SHORT"):
        open_paper_position(
            symbol,
            side,
            score,
            ind["price"],
            ind
        )

    return result


def scanner_loop():
    log.info("Scanner worker started")

    while True:
        try:
            reset_daily_if_needed()

            with lock:
                running = state["running"]

            if not running:
                with lock:
                    state["status"] = "STOPPED"
                time.sleep(2)
                continue

            with lock:
                if state["daily_locked"]:
                    state["status"] = "DAILY LOSS LIMIT REACHED"
                    time.sleep(2)
                    continue

            monitor_positions()

            for symbol in SYMBOLS:
                with lock:
                    if not state["running"]:
                        break

                try:
                    scan_symbol(symbol)

                except Exception as e:
                    log.exception(
                        "Scan failed for %s: %s",
                        symbol,
                        e
                    )

                    with lock:
                        state["last_error"] = (
                            symbol + ": " + str(e)
                        )

                time.sleep(1)

            with lock:
                state["last_scan"] = now_text()

                if state["running"] and not state["daily_locked"]:
                    state["status"] = "SCANNING"

                state["last_error"] = ""

            calculate_equity()
            save_state()

            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            log.exception("Scanner loop error: %s", e)

            with lock:
                state["last_error"] = str(e)
                state["status"] = "ERROR"

            time.sleep(5)


def start_worker():
    global worker_started

    with lock:
        if worker_started:
            return

        worker_started = True

    thread = threading.Thread(
        target=scanner_loop,
        daemon=True,
        name="CoinSentraScanner"
    )
    thread.start()

    # Read-only wallet status check.
    threading.Thread(
        target=check_private_account,
        daemon=True
    ).start()


# ============================================================
# DASHBOARD
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoinSentra Trading Brain</title>

<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    padding: 14px;
    background: #080d18;
    color: #e7edf7;
    font-family: Arial, sans-serif;
}

header {
    padding: 15px 5px;
    margin-bottom: 14px;
}

h1 {
    color: #44d6bb;
    font-size: 25px;
    margin: 0 0 6px;
}

h2 {
    font-size: 16px;
    color: #44d6bb;
    margin-top: 0;
}

.small {
    color: #8e9bb1;
    font-size: 12px;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 10px;
    margin-bottom: 12px;
}

.card {
    background: #121b2b;
    border: 1px solid #25334a;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
    min-width: 0;
}

.metric {
    color: #8e9bb1;
    font-size: 12px;
}

.big {
    font-size: 22px;
    font-weight: bold;
    margin-top: 8px;
    overflow-wrap: anywhere;
}

.green { color: #39dba9; }
.red { color: #ff657a; }
.yellow { color: #f3c969; }

button {
    border: 0;
    border-radius: 8px;
    padding: 12px 17px;
    font-weight: bold;
    margin: 4px;
    cursor: pointer;
}

.start { background: #21b98f; color: #071b16; }
.stop { background: #e65469; color: white; }
.refresh { background: #263b59; color: white; }

.table-wrap {
    width: 100%;
    overflow-x: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
    min-width: 680px;
    font-size: 12px;
}

th, td {
    padding: 10px 8px;
    border-bottom: 1px solid #27354a;
    text-align: left;
    white-space: nowrap;
}

th { color: #8e9bb1; }

.signal {
    font-weight: bold;
}

input {
    display: none;
}

#scanner {
    margin: 8px 0 12px;
    color: #44d6bb;
    font-size: 13px;
}

footer {
    color: #77859a;
    text-align: center;
    padding: 20px;
    font-size: 11px;
}

@media(max-width: 480px) {
    body { padding: 9px; }
    .big { font-size: 19px; }
}
</style>
</head>

<body>

<header>
    <h1>CoinSentra AI</h1>
    <div class="small">TRADING BRAIN V14.1 · BYBIT · PAPER TRADING</div>
    <div id="scanner">WAITING FOR MARKET DATA</div>

    <button class="start" onclick="control('start')">START</button>
    <button class="stop" onclick="control('stop')">STOP</button>
    <button class="refresh" onclick="update()">REFRESH</button>
</header>

<div class="grid">
    <div class="card">
        <div class="metric">PAPER BALANCE</div>
        <div class="big" id="balance">$10,000</div>
    </div>

    <div class="card">
        <div class="metric">EQUITY</div>
        <div class="big" id="equity">$10,000</div>
    </div>

    <div class="card">
        <div class="metric">REALIZED PNL</div>
        <div class="big" id="realized">$0</div>
    </div>

    <div class="card">
        <div class="metric">UNREALIZED PNL</div>
        <div class="big" id="unrealized">$0</div>
    </div>

    <div class="card">
        <div class="metric">WIN RATE</div>
        <div class="big" id="winrate">0%</div>
    </div>

    <div class="card">
        <div class="metric">COMPLETED TRADES</div>
        <div class="big" id="total">0</div>
    </div>

    <div class="card">
        <div class="metric">DAILY PNL</div>
        <div class="big" id="daily">$0</div>
    </div>

    <div class="card">
        <div class="metric">OPEN POSITIONS</div>
        <div class="big" id="openCount">0</div>
    </div>
</div>

<div class="card">
    <h2>MARKET SCANNER</h2>
    <div id="scanner">WAITING FOR MARKET DATA</div>

    <div class="table-wrap">
    <table>
        <thead>
        <tr>
            <th>SYMBOL</th>
            <th>PRICE</th>
            <th>24H %</th>
            <th>RSI</th>
            <th>ADX</th>
            <th>VOLUME</th>
            <th>ORDER BOOK</th>
            <th>FUNDING</th>
            <th>SIGNAL</th>
            <th>AI SCORE</th>
        </tr>
        </thead>
        <tbody id="marketRows">
        <tr><td colspan="10">Waiting for first scan...</td></tr>
        </tbody>
    </table>
    </div>
</div>

<div class="card">
    <h2>OPEN POSITIONS</h2>
    <div class="table-wrap">
    <table>
        <thead>
        <tr>
            <th>SYMBOL</th>
            <th>SIDE</th>
            <th>ENTRY</th>
            <th>PRICE</th>
            <th>STOP</th>
            <th>TARGET</th>
            <th>LEVERAGE</th>
            <th>PNL</th>
        </tr>
        </thead>
        <tbody id="positions">
        <tr><td colspan="8">No open positions</td></tr>
        </tbody>
    </table>
    </div>
</div>

<div class="card">
    <h2>COMPLETED TRADE HISTORY</h2>
    <div class="small" id="historySummary">No completed trades yet</div>
    <div class="table-wrap">
    <table>
        <thead>
        <tr>
            <th>TIME</th>
            <th>SYMBOL</th>
            <th>SIDE</th>
            <th>ENTRY</th>
            <th>EXIT</th>
            <th>PNL</th>
            <th>RESULT</th>
            <th>REASON</th>
        </tr>
        </thead>
        <tbody id="history">
        <tr><td colspan="8">No completed trades yet</td></tr>
        </tbody>
    </table>
    </div>
</div>

<div class="card">
    <h2>AI DIAGNOSTICS</h2>
    <div class="small">Engine status: <span id="engine">STARTING</span></div>
    <div class="small">Last scan: <span id="lastScan">Waiting</span></div>
    <div class="small">Private account: <span id="privateAccount">Not checked</span></div>
    <div class="small">Daily loss limit: <span id="dailyLimit">$6</span></div>
    <div class="small">Last error: <span id="lastError">None</span></div>
</div>

<footer>
    COINSENTRA TRADING BRAIN V14.1 · PAPER TRADING ONLY
</footer>

<script>
function money(value) {
    return "$" + Number(value || 0).toFixed(2);
}

function price(value) {
    return Number(value || 0).toPrecision(7);
}

function pnlClass(value) {
    return Number(value) >= 0 ? "green" : "red";
}

function safe(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

async function control(action) {
    try {
        const response = await fetch("/control/" + action, {
            method: "POST"
        });

        const data = await response.json();
        document.getElementById("engine").textContent =
            data.status || "UPDATED";

        update();
    } catch (error) {
        document.getElementById("lastError").textContent =
            error.message;
    }
}

async function update() {
    try {
        const response = await fetch("/api/status", {
            cache: "no-store"
        });

        if (!response.ok) {
            throw new Error("Server response " + response.status);
        }

        const d = await response.json();

        document.getElementById("balance").textContent =
            money(d.balance);

        document.getElementById("equity").textContent =
            money(d.equity);

        document.getElementById("realized").textContent =
            money(d.realized_pnl);

        document.getElementById("unrealized").textContent =
            money(d.unrealized_pnl);

        document.getElementById("daily").textContent =
            money(d.daily_pnl);

        document.getElementById("total").textContent =
            d.total_trades;

        document.getElementById("openCount").textContent =
            d.open_positions.length;

        const wins = Number(d.winning_trades || 0);
        const total = Number(d.total_trades || 0);

        document.getElementById("winrate").textContent =
            total ? (wins / total * 100).toFixed(1) + "%" : "0%";

        document.getElementById("engine").textContent =
            d.status;

        document.getElementById("lastScan").textContent =
            d.last_scan || "Waiting";

        document.getElementById("privateAccount").textContent =
            d.private_account;

        document.getElementById("lastError").textContent =
            d.last_error || "None";

        const scanner = document.getElementById("scanner");

        if (scanner) {
            scanner.textContent = d.last_scan
                ? "SCANNER ACTIVE · " + d.status
                : "WAITING FOR FIRST SCAN";
        }

        // MARKET ROWS
        const marketRows = document.getElementById("marketRows");
        const markets = Object.values(d.market || {});

        if (!markets.length) {
            marketRows.innerHTML =
                '<tr><td colspan="10">Waiting for market data...</td></tr>';
        } else {
            marketRows.innerHTML = markets.map(m => {
                const signalColor =
                    m.signal === "LONG" ? "green" :
                    m.signal === "SHORT" ? "red" : "yellow";

                const changeColor =
                    Number(m.change24h) >= 0 ? "green" : "red";

                return `
                <tr>
                    <td>${safe(m.symbol)}</td>
                    <td>${price(m.price)}</td>
                    <td class="${changeColor}">
                        ${Number(m.change24h).toFixed(2)}%
                    </td>
                    <td>${Number(m.rsi).toFixed(1)}</td>
                    <td>${Number(m.adx).toFixed(1)}</td>
                    <td>${Number(m.volume_ratio).toFixed(2)}x</td>
                    <td>${Number(m.orderbook_imbalance).toFixed(1)}%</td>
                    <td>${(Number(m.funding) * 100).toFixed(4)}%</td>
                    <td class="signal ${signalColor}">
                        ${safe(m.signal)}
                    </td>
                    <td>${m.score}/100</td>
                </tr>`;
            }).join("");
        }

        // OPEN POSITIONS
        const positions = document.getElementById("positions");

        if (!d.open_positions.length) {
            positions.innerHTML =
                '<tr><td colspan="8">No open positions</td></tr>';
        } else {
            positions.innerHTML = d.open_positions.map(p => `
                <tr>
                    <td>${safe(p.symbol)}</td>
                    <td>${safe(p.side)}</td>
                    <td>${price(p.entry_price)}</td>
                    <td>${price(p.current_price)}</td>
                    <td>${price(p.stop_loss)}</td>
                    <td>${price(p.take_profit)}</td>
                    <td>${p.leverage}x</td>
                    <td class="${pnlClass(p.unrealized_pnl)}">
                        ${money(p.unrealized_pnl)}
                    </td>
                </tr>
            `).join("");
        }

        // COMPLETED TRADE HISTORY
        const history = document.getElementById("history");

        if (!d.completed_trades.length) {
            history.innerHTML =
                '<tr><td colspan="8">No completed trades yet</td></tr>';
        } else {
            history.innerHTML = d.completed_trades.map(t => `
                <tr>
                    <td>${safe(t.exit_time)}</td>
                    <td>${safe(t.symbol)}</td>
                    <td>${safe(t.side)}</td>
                    <td>${price(t.entry_price)}</td>
                    <td>${price(t.exit_price)}</td>
                    <td class="${pnlClass(t.pnl)}">${money(t.pnl)}</td>
                    <td class="${t.result === "WIN" ? "green" : "red"}">
                        ${safe(t.result)}
                    </td>
                    <td>${safe(t.reason)}</td>
                </tr>
            `).join("");
        }

        document.getElementById("historySummary").textContent =
            d.total_trades + " completed trades · " +
            d.winning_trades + " wins · " +
            d.losing_trades + " losses";

    } catch (error) {
        document.getElementById("lastError").textContent =
            error.message;

        const scanner = document.getElementById("scanner");

        if (scanner) {
            scanner.textContent = "DASHBOARD CONNECTION ERROR";
        }
    }
}

update();
setInterval(update, 5000);
</script>

</body>
</html>
"""


# ============================================================
# WEB ROUTES
# ============================================================

@app.route("/")
def dashboard():
    return render_template_string(HTML)


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "app": "CoinSentra Trading Brain",
        "version": "14.1",
        "mode": "PAPER TRADING ONLY",
        "time": now_text()
    })


@app.route("/api/status")
@app.route("/status")
@app.route("/api/data")
def api_status():
    calculate_equity()

    with lock:
        snapshot = json.loads(json.dumps(state))

    return jsonify(snapshot)


@app.route("/control/stop", methods=["POST"])
def stop_bot():
    with lock:
        state["running"] = False
        state["status"] = "STOPPED"

    save_state()

    return jsonify({
        "ok": True,
        "status": "STOPPED"
    })


@app.route("/control/start", methods=["POST"])
def start_bot():
    with lock:
        state["running"] = True
        state["status"] = "STARTING"

        if state["daily_locked"]:
            state["status"] = "DAILY LOSS LIMIT REACHED"

    save_state()

    return jsonify({
        "ok": True,
        "status": state["status"]
    })


# ============================================================
# STARTUP
# ============================================================

load_state()
calculate_equity()
start_worker()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )
