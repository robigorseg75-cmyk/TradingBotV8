
import os
import json
import time
import threading
import logging
from datetime import datetime, timezone, date

from flask import Flask, jsonify, render_template_string
from pybit.unified_trading import HTTP

# ============================================================
# COINSENTRA TRADING BRAIN V13
# BYBIT | RENDER | PAPER TRADING ONLY
# ============================================================

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("CoinSentra")

# ---------------- SETTINGS ----------------

SYMBOLS = ["ETHUSDT", "XRPUSDT", "SOLUSDT", "SUIUSDT"]
CATEGORY = "linear"

LEVERAGE = 8
TAKE_PROFIT = 0.02
STOP_LOSS = 0.02

MAX_POSITIONS = 2
DAILY_LOSS_LIMIT = 6.0

START_BALANCE = 10000.0
RISK_PER_TRADE = 3.0
MAX_EQUITY_EXPOSURE = 0.10

SCAN_SECONDS = 30
ENTRY_SCORE = 68
CONFIRMATION_INTERVAL = "15"

# Simulated Bybit taker fee per side: 0.055%.
# This is an estimate for paper-trading performance.
FEE_RATE = 0.00055

PAPER_TRADING = True

API_KEY = os.getenv("vA9UkYkaoBHPySdPT9", "").strip()
API_SECRET = os.getenv("8uZSZDZdn8ra8xX4ephBuI7yOkyLpGKyot7h ", "").strip()

# Use Render persistent disk if DATA_DIR is configured.
DATA_DIR = os.getenv("DATA_DIR", "/tmp")
STATE_FILE = os.path.join(DATA_DIR, "coinsentra_v13.json")

# ---------------- CONNECTIONS ----------------

public_api = HTTP(testnet=False, timeout=15)

private_api = None

if API_KEY and API_SECRET:
    private_api = HTTP(
        testnet=False,
        api_key=API_KEY,
        api_secret=API_SECRET,
        timeout=15
    )

# ---------------- STATE ----------------

lock = threading.RLock()

state = {
    "balance": START_BALANCE,
    "positions": [],
    "history": [],
    "daily_pnl": 0.0,
    "daily_date": str(date.today()),
    "running": True,
    "last_scan": None,
    "last_error": None,
    "private_status": "NOT CONFIGURED",
    "account_balance": None,
    "symbols": {},
    "scan_count": 0,
    "version": "13.0"
}


def now_text():
    return datetime.now(timezone.utc).isoformat()


def number(value, default=0.0):
    try:
        result = float(value)
        if result != result or abs(result) == float("inf"):
            return default
        return result
    except (TypeError, ValueError):
        return default


def save_state():
    """Save a consistent state snapshot using atomic replacement."""
    try:
        with lock:
            snapshot = json.loads(json.dumps(state))

        os.makedirs(DATA_DIR, exist_ok=True)

        temp = STATE_FILE + ".tmp"

        with open(temp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)

        os.replace(temp, STATE_FILE)

    except Exception as e:
        log.exception("State save failed: %s", e)


def load_state():
    try:
        if not os.path.exists(STATE_FILE):
            log.info("No previous state file. Starting fresh.")
            return

        with open(STATE_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)

        with lock:
            for key in (
                "balance",
                "positions",
                "history",
                "daily_pnl",
                "daily_date",
                "running",
                "last_scan",
                "last_error",
                "private_status",
                "account_balance",
                "symbols",
                "scan_count"
            ):
                if key in saved:
                    state[key] = saved[key]

            state["version"] = "13.0"

            if not isinstance(state["positions"], list):
                state["positions"] = []

            if not isinstance(state["history"], list):
                state["history"] = []

            # Preserve existing V12 completed history.
            # Remove only non-completed opening-event rows.
            migrated = []

            for trade in state["history"]:
                if not isinstance(trade, dict):
                    continue

                if trade.get("exit") is not None and trade.get("pnl") is not None:
                    trade.setdefault("status", "COMPLETED")
                    trade.setdefault("trade_id", "OLD-" + str(len(migrated)))
                    trade.setdefault("fees", 0.0)
                    trade.setdefault("gross_pnl", trade.get("pnl", 0.0))
                    trade.setdefault("net_pnl", trade.get("pnl", 0.0))
                    migrated.append(trade)

            state["history"] = migrated[-500:]

            # Add missing fields to positions from older versions.
            for p in state["positions"]:
                p.setdefault("opened", now_text())
                p.setdefault("unrealized_pnl", 0.0)

        log.info(
            "Previous state loaded. Positions: %s | Completed trades: %s",
            len(state["positions"]),
            len(state["history"])
        )

    except Exception as e:
        log.exception("State load failed: %s", e)


def reset_daily():
    today = str(date.today())

    with lock:
        if state["daily_date"] != today:
            state["daily_date"] = today
            state["daily_pnl"] = 0.0

    save_state()


# ---------------- MARKET DATA ----------------

def get_price(symbol):
    result = public_api.get_tickers(
        category=CATEGORY,
        symbol=symbol
    )

    items = result.get("result", {}).get("list", [])

    if not items:
        raise ValueError("No ticker data: " + symbol)

    return number(items[0].get("lastPrice"))


def get_candles(symbol, interval, limit=200):
    result = public_api.get_kline(
        category=CATEGORY,
        symbol=symbol,
        interval=interval,
        limit=limit
    )

    rows = result.get("result", {}).get("list", [])

    if not rows:
        raise ValueError("No candle data: " + symbol)

    rows = list(reversed(rows))

    candles = []

    for x in rows:
        candles.append({
            "time": int(x[0]),
            "open": number(x[1]),
            "high": number(x[2]),
            "low": number(x[3]),
            "close": number(x[4]),
            "volume": number(x[5]),
            "turnover": number(x[6])
        })

    return candles


def get_orderbook(symbol):
    result = public_api.get_orderbook(
        category=CATEGORY,
        symbol=symbol,
        limit=25
    )

    data = result.get("result", {})

    bids = data.get("b", [])
    asks = data.get("a", [])

    bid_volume = sum(number(x[1]) for x in bids)
    ask_volume = sum(number(x[1]) for x in asks)

    total = bid_volume + ask_volume

    if total <= 0:
        return 0.0

    return (bid_volume - ask_volume) / total


def get_funding(symbol):
    try:
        result = public_api.get_funding_rate_history(
            category=CATEGORY,
            symbol=symbol,
            limit=1
        )

        rows = result.get("result", {}).get("list", [])

        if rows:
            return number(rows[0].get("fundingRate"))

    except Exception as e:
        log.warning("Funding unavailable %s: %s", symbol, e)

    return 0.0


def get_open_interest(symbol):
    try:
        result = public_api.get_open_interest(
            category=CATEGORY,
            symbol=symbol,
            intervalTime="5min",
            limit=2
        )

        rows = result.get("result", {}).get("list", [])

        if len(rows) >= 2:
            older = number(rows[1].get("openInterest"))
            newer = number(rows[0].get("openInterest"))

            if older > 0:
                return (newer - older) / older * 100

    except Exception as e:
        log.warning("OI unavailable %s: %s", symbol, e)

    return 0.0


# ---------------- INDICATORS ----------------

def ema(values, period):
    if len(values) < period:
        return None

    multiplier = 2 / (period + 1)
    result = sum(values[:period]) / period

    for value in values[period:]:
        result = value * multiplier + result * (1 - multiplier)

    return result


def rsi(values, period=14):
    if len(values) < period + 1:
        return 50.0

    changes = [
        values[i] - values[i - 1]
        for i in range(1, len(values))
    ]

    gains = [max(x, 0) for x in changes]
    losses = [max(-x, 0) for x in changes]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(changes)):
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


def macd(values):
    fast = ema(values, 12)
    slow = ema(values, 26)

    if fast is None or slow is None:
        return 0.0

    return fast - slow


def bollinger(values, period=20):
    sample = values[-period:]

    if len(sample) < period:
        return 0.0, 0.0, 0.0

    middle = sum(sample) / period
    variance = sum((x - middle) ** 2 for x in sample) / period
    deviation = variance ** 0.5

    return (
        middle + 2 * deviation,
        middle,
        middle - 2 * deviation
    )


def adx_approx(candles, period=14):
    if len(candles) < period + 2:
        return 0.0

    tr_values = []
    plus_values = []
    minus_values = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        tr = max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"])
        )

        up = current["high"] - previous["high"]
        down = previous["low"] - current["low"]

        tr_values.append(tr)
        plus_values.append(up if up > down and up > 0 else 0)
        minus_values.append(down if down > up and down > 0 else 0)

    tr_avg = sum(tr_values[-period:]) / period

    if tr_avg <= 0:
        return 0.0

    plus_di = sum(plus_values[-period:]) / period / tr_avg * 100
    minus_di = sum(minus_values[-period:]) / period / tr_avg * 100

    denominator = plus_di + minus_di

    if denominator <= 0:
        return 0.0

    return abs(plus_di - minus_di) / denominator * 100


def analyze_candles(candles):
    closes = [x["close"] for x in candles]
    volumes = [x["volume"] for x in candles]

    if len(closes) < 60:
        raise ValueError("Insufficient candles")

    current = closes[-1]

    e9 = ema(closes, 9)
    e21 = ema(closes, 21)
    e50 = ema(closes, 50)

    current_rsi = rsi(closes)
    current_macd = macd(closes)

    upper, middle, lower = bollinger(closes)

    average_volume = sum(volumes[-21:-1]) / 20
    volume_ratio = (
        volumes[-1] / average_volume
        if average_volume > 0 else 0
    )

    adx = adx_approx(candles)

    return {
        "price": current,
        "ema9": e9,
        "ema21": e21,
        "ema50": e50,
        "rsi": current_rsi,
        "macd": current_macd,
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "volume_ratio": volume_ratio,
        "adx": adx
    }


# ---------------- SIGNAL ENGINE ----------------

def calculate_signal(five, fifteen, imbalance, funding, oi_change):
    long_score = 0.0
    short_score = 0.0

    price = five["price"]

    if five["ema9"] > five["ema21"]:
        long_score += 14
    elif five["ema9"] < five["ema21"]:
        short_score += 14

    if five["ema21"] > five["ema50"]:
        long_score += 12
    elif five["ema21"] < five["ema50"]:
        short_score += 12

    if 52 <= five["rsi"] <= 70:
        long_score += 12
    elif 30 <= five["rsi"] < 48:
        short_score += 12

    if five["macd"] > 0:
        long_score += 10
    elif five["macd"] < 0:
        short_score += 10

    if five["volume_ratio"] >= 1.3:
        if five["ema9"] > five["ema21"]:
            long_score += 10
        elif five["ema9"] < five["ema21"]:
            short_score += 10

    if imbalance > 0.15:
        long_score += 12
    elif imbalance < -0.15:
        short_score += 12

    if fifteen["ema9"] > fifteen["ema21"]:
        long_score += 18
    elif fifteen["ema9"] < fifteen["ema21"]:
        short_score += 18

    if five["adx"] >= 20:
        if five["ema9"] > five["ema21"]:
            long_score += 6
        elif five["ema9"] < five["ema21"]:
            short_score += 6

    if funding > 0.0005:
        short_score += 3
    elif funding < -0.0005:
        long_score += 3

    if oi_change > 0.2:
        if five["ema9"] > five["ema21"]:
            long_score += 5
        elif five["ema9"] < five["ema21"]:
            short_score += 5

    long_score = min(100, round(long_score))
    short_score = min(100, round(short_score))

    signal = "WAIT"

    if (
        long_score >= ENTRY_SCORE
        and long_score - short_score >= 12
        and price > five["ema21"]
        and fifteen["ema9"] > fifteen["ema21"]
    ):
        signal = "LONG"

    elif (
        short_score >= ENTRY_SCORE
        and short_score - long_score >= 12
        and price < five["ema21"]
        and fifteen["ema9"] < fifteen["ema21"]
    ):
        signal = "SHORT"

    return {
        "long_score": long_score,
        "short_score": short_score,
        "signal": signal
    }


# ---------------- PRIVATE ACCOUNT READ ----------------

def check_private_account():
    if private_api is None:
        with lock:
            state["private_status"] = "NOT CONFIGURED"
        return

    try:
        result = private_api.get_wallet_balance(
            accountType="UNIFIED",
            coin="USDT"
        )

        accounts = result.get("result", {}).get("list", [])

        wallet = None

        if accounts:
            coins = accounts[0].get("coin", [])
            wallet = next(
                (x for x in coins if x.get("coin") == "USDT"),
                None
            )

        with lock:
            state["account_balance"] = (
                number(wallet.get("walletBalance"))
                if wallet else 0.0
            )
            state["private_status"] = "CONNECTED - READ ONLY"

    except Exception as e:
        with lock:
            state["private_status"] = "CONNECTION FAILED"
        log.error("Private account check failed: %s", e)


# ---------------- PAPER TRADING ----------------

def calculate_unrealized(position, price):
    entry = number(position.get("entry"))
    quantity = number(position.get("quantity"))

    direction = 1 if position.get("side") == "LONG" else -1

    gross = (price - entry) * quantity * direction

    entry_fee = entry * quantity * FEE_RATE
    exit_fee = price * quantity * FEE_RATE

    return gross - entry_fee - exit_fee


def open_paper_position(symbol, side, price):
    with lock:
        if not state["running"]:
            return

        if len(state["positions"]) >= MAX_POSITIONS:
            return

        if any(p["symbol"] == symbol for p in state["positions"]):
            return

        if state["daily_pnl"] <= -DAILY_LOSS_LIMIT:
            state["running"] = False
            return

        balance = number(state["balance"])

        risk = min(
            RISK_PER_TRADE,
            balance * 0.005
        )

        stop_distance = price * STOP_LOSS

        if stop_distance <= 0 or price <= 0:
            return

        quantity = risk / stop_distance

        max_notional = balance * MAX_EQUITY_EXPOSURE
        quantity = min(quantity, max_notional / price)

        if quantity <= 0:
            return

        position = {
            "symbol": symbol,
            "side": side,
            "entry": price,
            "quantity": quantity,
            "notional": quantity * price,
            "leverage": LEVERAGE,
            "take_profit": (
                price * (1 + TAKE_PROFIT)
                if side == "LONG"
                else price * (1 - TAKE_PROFIT)
            ),
            "stop_loss": (
                price * (1 - STOP_LOSS)
                if side == "LONG"
                else price * (1 + STOP_LOSS)
            ),
            "opened": now_text(),
            "unrealized_pnl": 0.0,
            "fee_rate": FEE_RATE
        }

        state["positions"].append(position)

        log.info("PAPER OPEN %s %s at %s", side, symbol, price)

    save_state()


def close_paper_position(position, price, reason):
    with lock:
        # Find the live position again; avoid closing a stale copy.
        live = next(
            (
                p for p in state["positions"]
                if p.get("symbol") == position.get("symbol")
                and p.get("opened") == position.get("opened")
            ),
            None
        )

        if live is None:
            return

        entry = number(live["entry"])
        quantity = number(live["quantity"])

        direction = 1 if live["side"] == "LONG" else -1

        gross_pnl = (price - entry) * quantity * direction

        entry_fee = entry * quantity * FEE_RATE
        exit_fee = price * quantity * FEE_RATE
        total_fees = entry_fee + exit_fee

        net_pnl = gross_pnl - total_fees

        state["balance"] += net_pnl
        state["daily_pnl"] += net_pnl

        trade = {
            "trade_id": (
                live["symbol"] + "-" +
                str(int(time.time() * 1000))
            ),
            "status": "COMPLETED",
            "symbol": live["symbol"],
            "side": live["side"],
            "entry": entry,
            "exit": price,
            "quantity": quantity,
            "notional": entry * quantity,
            "gross_pnl": round(gross_pnl, 6),
            "fees": round(total_fees, 6),
            "pnl": round(net_pnl, 6),
            "net_pnl": round(net_pnl, 6),
            "reason": reason,
            "opened": live.get("opened", ""),
            "closed": now_text()
        }

        # Newest completed trades are stored first.
        state["history"].insert(0, trade)
        state["history"] = state["history"][:500]

        state["positions"].remove(live)

        if state["daily_pnl"] <= -DAILY_LOSS_LIMIT:
            state["running"] = False
            log.warning("Daily loss limit reached. New entries stopped.")

        log.info(
            "PAPER CLOSE %s | NET PNL %.4f | FEES %.4f | %s",
            live["symbol"], net_pnl, total_fees, reason
        )

    save_state()


def monitor_positions():
    with lock:
        positions = [dict(p) for p in state["positions"]]

    for position in positions:
        try:
            price = get_price(position["symbol"])

            entry = number(position["entry"])

            if entry <= 0:
                continue

            direction = 1 if position["side"] == "LONG" else -1

            gross_change = (
                (price - entry) / entry
            ) * direction

            net_pnl = calculate_unrealized(position, price)

            with lock:
                for live in state["positions"]:
                    if (
                        live.get("symbol") == position.get("symbol")
                        and live.get("opened") == position.get("opened")
                    ):
                        live["unrealized_pnl"] = round(net_pnl, 4)
                        live["current_price"] = price
                        break

            if gross_change >= TAKE_PROFIT:
                close_paper_position(
                    position, price, "TAKE PROFIT"
                )

            elif gross_change <= -STOP_LOSS:
                close_paper_position(
                    position, price, "STOP LOSS"
                )

        except Exception as e:
            log.warning(
                "Position monitor %s: %s",
                position["symbol"], e
            )


# ---------------- DAILY RISK ----------------

def risk_check():
    with lock:
        unrealized = sum(
            number(p.get("unrealized_pnl"))
            for p in state["positions"]
        )

        daily_equity_pnl = state["daily_pnl"] + unrealized

        if daily_equity_pnl <= -DAILY_LOSS_LIMIT:
            state["running"] = False
            log.warning(
                "Daily loss limit reached including unrealized PNL."
            )
            return False

    return True


# ---------------- SCANNER ----------------

def scan_symbol(symbol):
    five_candles = get_candles(symbol, "5", 200)
    fifteen_candles = get_candles(
        symbol, CONFIRMATION_INTERVAL, 100
    )

    five = analyze_candles(five_candles)
    fifteen = analyze_candles(fifteen_candles)

    imbalance = get_orderbook(symbol)
    funding = get_funding(symbol)
    oi_change = get_open_interest(symbol)

    signal = calculate_signal(
        five,
        fifteen,
        imbalance,
        funding,
        oi_change
    )

    data = {
        "symbol": symbol,
        "price": five["price"],
        "rsi": round(five["rsi"], 2),
        "macd": round(five["macd"], 8),
        "adx": round(five["adx"], 2),
        "ema9": round(five["ema9"], 8),
        "ema21": round(five["ema21"], 8),
        "ema50": round(five["ema50"], 8),
        "volume_ratio": round(five["volume_ratio"], 2),
        "orderbook_imbalance": round(imbalance, 3),
        "funding": funding,
        "oi_change": round(oi_change, 3),
        **signal,
        "updated": now_text()
    }

    with lock:
        state["symbols"][symbol] = data
        should_open = (
            state["running"]
            and signal["signal"] in ("LONG", "SHORT")
        )

    if should_open:
        open_paper_position(
            symbol,
            signal["signal"],
            five["price"]
        )

    return data


def scanner_loop():
    log.info("CoinSentra V13 scanner started")

    while True:
        try:
            reset_daily()

            # Monitor existing positions before scanning new signals.
            monitor_positions()

            for symbol in SYMBOLS:
                try:
                    scan_symbol(symbol)

                except Exception as e:
                    log.exception("Scan failed for %s", symbol)

                    with lock:
                        existing = state["symbols"].get(symbol, {})
                        existing["data_error"] = str(e)[:180]
                        state["symbols"][symbol] = existing

            risk_check()

            with lock:
                state["last_scan"] = now_text()
                state["scan_count"] += 1

            save_state()

        except Exception as e:
            log.exception("Scanner loop error")

            with lock:
                state["last_error"] = str(e)[:250]

            save_state()

        time.sleep(SCAN_SECONDS)


# ---------------- TRADE STATISTICS ----------------

def trade_statistics():
    with lock:
        trades = list(state["history"])

        wins = sum(
            1 for t in trades
            if number(t.get("pnl", t.get("net_pnl"))) > 0
        )

        losses = sum(
            1 for t in trades
            if number(t.get("pnl", t.get("net_pnl"))) < 0
        )

        count = len(trades)

        win_rate = (
            wins / count * 100
            if count else 0.0
        )

        realized = sum(
            number(t.get("pnl", t.get("net_pnl")))
            for t in trades
        )

        fees = sum(number(t.get("fees")) for t in trades)

        return {
            "completed_count": count,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 2),
            "history_realized_pnl": round(realized, 4),
            "total_fees": round(fees, 4)
        }


# ---------------- DASHBOARD ----------------

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta charset="UTF-8">
<title>CoinSentra Trading Brain V13</title>
<style>
*{box-sizing:border-box}
body{background:#080d18;color:#e9efff;font-family:Arial;margin:0;padding:15px}
h1{color:#36d9ff;font-size:25px}
h2{color:#36d9ff;font-size:18px}
.card{background:#121d30;border:1px solid #263a58;border-radius:12px;padding:14px;margin:10px 0}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.big{font-size:22px;font-weight:bold;color:#35e0b0;margin-top:7px}
.small{font-size:12px;color:#9bb0ce;line-height:1.6}
button{padding:13px;border:0;border-radius:8px;font-weight:bold;margin:5px 2px}
.start{background:#16b981;color:white}
.stop{background:#dc4545;color:white}
table{width:100%;border-collapse:collapse;font-size:12px;min-width:650px}
td,th{padding:9px 6px;border-bottom:1px solid #293a50;text-align:left;white-space:nowrap}
th{color:#9bb0ce}
.green{color:#39df9a}
.red{color:#ff7070}
.yellow{color:#ffcf65}
.scroll{overflow-x:auto}
.statgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.stat{background:#0c1728;border-radius:8px;padding:10px}
.stat strong{display:block;font-size:18px;margin-top:5px}
</style>
</head>
<body>

<h1>COINSENTRA AI</h1>
<div class="small">TRADING BRAIN V13 | BYBIT | PAPER MODE ONLY</div>

<div class="grid">
<div class="card">Virtual Balance<div class="big" id="balance">$10,000</div></div>
<div class="card">Daily Net PNL<div class="big" id="daily">$0.00</div></div>
<div class="card">Open Positions<div class="big" id="count">0</div></div>
<div class="card">Scanner<div class="big" id="scanner">STARTING</div></div>
</div>

<div class="card">
<h2>CONTROL CENTER</h2>
<button class="stop" onclick="control('stop')">STOP NEW TRADES (C)</button>
<button class="start" onclick="control('start')">START TRADING (V)</button>
<p id="control" class="small">Loading...</p>
<p class="small">Stopping prevents new entries. Existing paper positions remain monitored.</p>
</div>

<div class="card">
<h2>TRADE PERFORMANCE</h2>
<div class="statgrid">
<div class="stat"><span class="small">Completed trades</span><strong id="completed">0</strong></div>
<div class="stat"><span class="small">Win rate</span><strong id="winrate">0%</strong></div>
<div class="stat"><span class="small">Winning trades</span><strong class="green" id="wins">0</strong></div>
<div class="stat"><span class="small">Losing trades</span><strong class="red" id="losses">0</strong></div>
<div class="stat"><span class="small">Realized net PNL</span><strong id="realized">$0.00</strong></div>
<div class="stat"><span class="small">Simulated fees</span><strong id="fees">$0.00</strong></div>
</div>
</div>

<div class="card">
<h2>BYBIT CONNECTION</h2>
<div id="private">Checking...</div>
<div id="account" class="small"></div>
<div class="small">Read-only wallet connection. Paper trading only.</div>
</div>

<div class="card">
<h2>MARKET SCANNER</h2>
<div class="scroll">
<table>
<thead><tr>
<th>COIN</th><th>PRICE</th><th>LONG</th><th>SHORT</th><th>SIGNAL</th>
</tr></thead>
<tbody id="markets"></tbody>
</table>
</div>
</div>

<div class="card">
<h2>INDICATORS</h2>
<div id="indicators" class="small">Waiting for market data...</div>
</div>

<div class="card">
<h2>OPEN PAPER POSITIONS</h2>
<div class="scroll">
<table>
<thead><tr>
<th>COIN</th><th>SIDE</th><th>ENTRY</th><th>LIVE PRICE</th><th>NET PNL</th><th>TP</th><th>SL</th>
</tr></thead>
<tbody id="positions"></tbody>
</table>
</div>
</div>

<div class="card">
<h2>COMPLETED TRADE HISTORY</h2>
<div class="small">A trade is recorded here only after it has actually closed at its take-profit or stop-loss level.</div>
<div class="scroll">
<table>
<thead><tr>
<th>CLOSED</th><th>COIN</th><th>SIDE</th><th>ENTRY</th><th>EXIT</th><th>GROSS PNL</th><th>FEES</th><th>NET PNL</th><th>REASON</th>
</tr></thead>
<tbody id="history">
<tr><td colspan="9">No completed trades yet.</td></tr>
</tbody>
</table>
</div>
</div>

<div class="card small">
Version 13.0 | 5m + 15m confirmation | 8x simulated leverage<br>
Take Profit 2% | Stop Loss 2% | Daily loss limit $6<br>
Simulated fee estimate: 0.055% per side<br>
Last scan: <span id="last">Waiting</span><br>
Last error: <span id="error">None</span>
</div>

<script>
async function control(action){
 const message=document.getElementById('control');
 message.textContent='Sending command...';

 try{
  const response=await fetch('/control/'+action,{method:'POST'});
  const data=await response.json();
  message.textContent=data.message||data.status||data.error||'Command received';
  update();
 }catch(e){
  message.textContent='Could not contact bot.';
 }
}

document.addEventListener('keydown',function(e){
 if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;
 if(e.key.toLowerCase()==='c')control('stop');
 if(e.key.toLowerCase()==='v')control('start');
});

function money(x){
 return '$'+Number(x||0).toFixed(2);
}

function price(x){
 let n=Number(x||0);
 return n?n.toPrecision(8):'-';
}

function safe(x){
 return String(x??'').replace(/[&<>"']/g,c=>({
  '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
 }[c]));
}

function color(x){
 return x==='LONG'?'green':x==='SHORT'?'red':'yellow';
}

async function update(){
 try{
  let r=await fetch('/api/data',{cache:'no-store'});
  if(!r.ok)throw new Error('API response error');

  let d=await r.json();

  document.getElementById('balance').textContent=money(d.balance);
  document.getElementById('daily').textContent=money(d.daily_pnl);
  document.getElementById('count').textContent=d.positions.length;
  document.getElementById('scanner').textContent=d.last_scan?'ACTIVE':'WAITING';

  document.getElementById('control').textContent=
   d.running?'NEW ENTRIES ENABLED':'STOPPED - MONITORING POSITIONS';

  document.getElementById('private').textContent=d.private_status;

  document.getElementById('account').textContent=
   d.account_balance===null
   ?'Account balance unavailable'
   :'Bybit USDT wallet: '+money(d.account_balance);

  document.getElementById('last').textContent=d.last_scan||'Waiting';
  document.getElementById('error').textContent=d.last_error||'None';

  document.getElementById('completed').textContent=d.stats.completed_count;
  document.getElementById('winrate').textContent=d.stats.win_rate+'%';
  document.getElementById('wins').textContent=d.stats.wins;
  document.getElementById('losses').textContent=d.stats.losses;
  document.getElementById('realized').textContent=money(d.stats.history_realized_pnl);
  document.getElementById('fees').textContent=money(d.stats.total_fees);

  let rows='';
  let details='';

  for(let s of Object.values(d.symbols)){
   rows+=`<tr>
   <td>${safe(s.symbol)}</td>
   <td>${price(s.price)}</td>
   <td class="green">${s.long_score??0}</td>
   <td class="red">${s.short_score??0}</td>
   <td class="${color(s.signal)}">${safe(s.signal||'WAIT')}</td>
   </tr>`;

   details+=`<p><b>${safe(s.symbol)}</b> |
   RSI ${s.rsi??'-'} |
   ADX ${s.adx??'-'} |
   Volume ${s.volume_ratio??'-'}x |
   Orderbook ${s.orderbook_imbalance??'-'} |
   Funding ${s.funding??'-'} |
   OI ${s.oi_change??'-'}%
   ${s.data_error?' | ERROR: '+safe(s.data_error):''}</p>`;
  }

  document.getElementById('markets').innerHTML=
   rows||'<tr><td colspan="5">Waiting for first market scan</td></tr>';

  document.getElementById('indicators').innerHTML=
   details||'Waiting for market data';

  let positionRows='';

  for(let p of d.positions){
   positionRows+=`<tr>
    <td>${safe(p.symbol)}</td>
    <td>${safe(p.side)}</td>
    <td>${price(p.entry)}</td>
    <td>${price(p.current_price)}</td>
    <td class="${Number(p.unrealized_pnl)>=0?'green':'red'}">${money(p.unrealized_pnl)}</td>
    <td>${price(p.take_profit)}</td>
    <td>${price(p.stop_loss)}</td>
   </tr>`;
  }

  document.getElementById('positions').innerHTML=
   positionRows||'<tr><td colspan="7">No open positions</td></tr>';

  let historyRows='';

  for(let p of d.history){
   historyRows+=`<tr>
    <td>${safe(p.closed||'-')}</td>
    <td>${safe(p.symbol)}</td>
    <td>${safe(p.side)}</td>
    <td>${price(p.entry)}</td>
    <td>${price(p.exit)}</td>
    <td>${money(p.gross_pnl)}</td>
    <td>${money(p.fees)}</td>
    <td class="${Number(p.pnl)>=0?'green':'red'}">${money(p.pnl)}</td>
    <td>${safe(p.reason)}</td>
   </tr>`;
  }

  document.getElementById('history').innerHTML=
   historyRows||'<tr><td colspan="9">No completed trades yet. Waiting for a position to close.</td></tr>';

 }catch(e){
  document.getElementById('scanner').textContent='CONNECTION ERROR';
 }
}

update();
setInterval(update,5000);
</script>
</body>
</html>
"""


# ---------------- WEB ROUTES ----------------

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "version": "13.0",
        "paper_trading": True
    })


@app.route("/api/status")
@app.route("/status")
def status():
    with lock:
        return jsonify({
            "status": "ok",
            "version": state["version"],
            "running": state["running"],
            "last_scan": state["last_scan"],
            "last_error": state["last_error"],
            "symbols": state["symbols"],
            "scan_count": state["scan_count"]
        })


@app.route("/api/data")
def api_data():
    with lock:
        positions = [dict(p) for p in state["positions"]]
        history = [dict(t) for t in state["history"]]
        symbols = dict(state["symbols"])

        data = {
            "balance": round(number(state["balance"]), 4),
            "daily_pnl": round(number(state["daily_pnl"]), 4),
            "positions": positions,
            "history": history[:100],
            "symbols": symbols,
            "running": state["running"],
            "last_scan": state["last_scan"],
            "private_status": state["private_status"],
            "account_balance": state["account_balance"],
            "last_error": state["last_error"],
            "stats": trade_statistics()
        }

    return jsonify(data)


@app.route("/control/stop", methods=["POST"])
def stop():
    with lock:
        state["running"] = False

    save_state()

    return jsonify({
        "status": "stopped",
        "message": "STOPPED: new paper entries disabled. Existing positions are still monitored."
    })


@app.route("/control/start", methods=["POST"])
def start():
    with lock:
        if state["daily_pnl"] <= -DAILY_LOSS_LIMIT:
            return jsonify({
                "status": "blocked",
                "error": "Daily loss limit reached. Trading remains stopped."
            }), 403

        state["running"] = True

    save_state()

    return jsonify({
        "status": "started",
        "message": "STARTED: new paper entries enabled."
    })


# ---------------- STARTUP ----------------

load_state()

worker_started = False
worker_lock = threading.Lock()


def background_worker():
    log.info("CoinSentra V13 background worker started")

    check_private_account()

    # Recheck read-only wallet balance periodically.
    last_private_check = time.time()

    while True:
        try:
            scanner_loop()

        except Exception:
            log.exception("Background worker error")
            time.sleep(10)

        if time.time() - last_private_check >= 300:
            check_private_account()
            last_private_check = time.time()


@app.before_request
def start_worker_once():
    global worker_started

    with worker_lock:
        if not worker_started:
            thread = threading.Thread(
                target=background_worker,
                name="CoinSentraBackground",
                daemon=True
            )
            thread.start()
            worker_started = True


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
