
# ============================================================
# COINSENTRA TRADING BRAIN V8 - RENDER EDITION
# BYBIT | WHALE INTELLIGENCE | PAPER TRADING
# ============================================================

import os
import json
import time
import math
import logging
import threading
import secrets
from copy import deepcopy
from datetime import datetime, date
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, Response, request
from pybit.unified_trading import HTTP

# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "CoinSentra Trading Brain V8"

TESTNET = False
CATEGORY = "linear"

SYMBOLS = [
    "ETHUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "SUIUSDT"
]

LEVERAGE = 8

SCAN_SECONDS = 60
DASHBOARD_REFRESH = 15

TAKE_PROFIT = 0.02
STOP_LOSS = 0.02

DAILY_LOSS_LIMIT = 6.0

STARTING_BALANCE = 100.0

RISK_PER_TRADE = 0.005
MAX_POSITION_VALUE = 0.10

MAX_OPEN_POSITIONS = 2
MIN_SIGNAL_SCORE = 70

ORDERBOOK_DEPTH = 50

FEE_RATE_PER_SIDE = 0.00055

LOCAL_TZ = ZoneInfo("Europe/Ljubljana")

DATA_DIR = os.environ.get("DATA_DIR", "/var/data")

if not os.path.isdir(DATA_DIR):
    DATA_DIR = "."

STATE_FILE = os.path.join(
    DATA_DIR,
    "coinsentra_v8_state.json"
)

DASHBOARD_PASSWORD = os.environ.get(
    "DASHBOARD_PASSWORD",
    "change-this-password"
)

SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "change-this-secret"
)

WHALE_THRESHOLDS = {
    "ETHUSDT": 100000,
    "XRPUSDT": 50000,
    "SOLUSDT": 50000,
    "SUIUSDT": 25000
}

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

log = logging.getLogger("CoinSentra")

# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)
app.secret_key = SECRET_KEY

# ============================================================
# BYBIT PUBLIC MARKET DATA
# ============================================================

session = HTTP(testnet=TESTNET)

# ============================================================
# STATE
# ============================================================

state_lock = threading.RLock()

def new_state():
    return {
        "balance": STARTING_BALANCE,
        "day": str(datetime.now(LOCAL_TZ).date()),
        "daily_realized": 0.0,
        "positions": {},
        "closed_trades": [],
        "last_oi": {},
        "scan_count": 0
    }


def load_state():

    if os.path.exists(STATE_FILE):

        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)

            default = new_state()

            for key, value in default.items():

                if key not in data:
                    data[key] = value

            return data

        except Exception:
            log.exception("State loading failed")

    return new_state()


state = load_state()

market_status = {
    "running": False,
    "last_scan": None,
    "last_error": None,
    "rows": [],
    "scan_count": 0
}


def save_state():

    try:

        with state_lock:

            temporary = STATE_FILE + ".tmp"

            with open(temporary, "w") as f:
                json.dump(state, f, indent=2)

            os.replace(temporary, STATE_FILE)

    except Exception:
        log.exception("State saving failed")


# ============================================================
# UTILITIES
# ============================================================

def number(value, default=0.0):

    try:
        return float(value)

    except (ValueError, TypeError):
        return default


def timestamp():

    return datetime.now(LOCAL_TZ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def money(value):
    return round(number(value), 4)


def api_result(response):

    if response.get("retCode") != 0:
        raise RuntimeError(str(response))

    return response.get("result", {})


# ============================================================
# BYBIT MARKET DATA
# ============================================================

def get_candles(symbol, interval):

    result = api_result(
        session.get_kline(
            category=CATEGORY,
            symbol=symbol,
            interval=interval,
            limit=220
        )
    )

    rows = list(reversed(result["list"]))

    candles = []

    for row in rows:

        candles.append({
            "open": number(row[1]),
            "high": number(row[2]),
            "low": number(row[3]),
            "close": number(row[4]),
            "volume": number(row[5]),
            "turnover": number(row[6])
        })

    return candles


def get_ticker(symbol):

    result = api_result(
        session.get_tickers(
            category=CATEGORY,
            symbol=symbol
        )
    )

    return result["list"][0]


def get_orderbook(symbol):

    result = api_result(
        session.get_orderbook(
            category=CATEGORY,
            symbol=symbol,
            limit=ORDERBOOK_DEPTH
        )
    )

    bids = [
        (number(x[0]), number(x[1]))
        for x in result["b"]
    ]

    asks = [
        (number(x[0]), number(x[1]))
        for x in result["a"]
    ]

    return bids, asks


def get_trades(symbol):

    result = api_result(
        session.get_public_trade_history(
            category=CATEGORY,
            symbol=symbol,
            limit=100
        )
    )

    return result.get("list", [])


def get_open_interest(symbol):

    result = api_result(
        session.get_open_interest(
            category=CATEGORY,
            symbol=symbol,
            intervalTime="5min",
            limit=2
        )
    )

    rows = result.get("list", [])

    if not rows:
        return 0.0

    return number(rows[0].get("openInterest"))


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def EMA(values, period):

    if len(values) < period:
        return None

    alpha = 2 / (period + 1)

    result = sum(values[:period]) / period

    for value in values[period:]:
        result = alpha * value + (1 - alpha) * result

    return result


def RSI(values, period=14):

    if len(values) < period + 1:
        return None

    changes = [
        values[i] - values[i - 1]
        for i in range(1, len(values))
    ]

    gains = [max(x, 0) for x in changes]
    losses = [max(-x, 0) for x in changes]

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


def MACD(values):

    fast = EMA(values, 12)
    slow = EMA(values, 26)

    if fast is None or slow is None:
        return None

    return fast - slow


def BOLLINGER(values, period=20):

    if len(values) < period:
        return None, None, None

    sample = values[-period:]

    middle = sum(sample) / period

    variance = sum(
        (x - middle) ** 2 for x in sample
    ) / period

    deviation = math.sqrt(variance)

    return (
        middle + 2 * deviation,
        middle,
        middle - 2 * deviation
    )


def ADX(candles, period=14):

    if len(candles) < period * 2 + 1:
        return None

    true_ranges = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(candles)):

        current = candles[i]
        previous = candles[i - 1]

        up = current["high"] - previous["high"]
        down = previous["low"] - current["low"]

        plus_dm.append(
            up if up > down and up > 0 else 0
        )

        minus_dm.append(
            down if down > up and down > 0 else 0
        )

        tr = max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"])
        )

        true_ranges.append(tr)

    dx_values = []

    for i in range(period, len(true_ranges)):

        start = i - period + 1

        tr = sum(true_ranges[start:i + 1])

        if tr <= 0:
            continue

        positive = 100 * sum(
            plus_dm[start:i + 1]
        ) / tr

        negative = 100 * sum(
            minus_dm[start:i + 1]
        ) / tr

        total = positive + negative

        dx = (
            0 if total == 0
            else 100 * abs(positive - negative) / total
        )

        dx_values.append(dx)

    if len(dx_values) < period:
        return None

    return sum(dx_values[-period:]) / period


# ============================================================
# TIMEFRAME ANALYSIS
# ============================================================

def analyze(candles):

    closes = [x["close"] for x in candles]

    if len(closes) < 60:
        raise ValueError("Not enough candles")

    e9 = EMA(closes, 9)
    e21 = EMA(closes, 21)
    e50 = EMA(closes, 50)

    rsi = RSI(closes)
    macd = MACD(closes)

    upper, middle, lower = BOLLINGER(closes)

    adx = ADX(candles)

    recent_volume = [
        x["volume"] for x in candles[-21:-1]
    ]

    average_volume = (
        sum(recent_volume) / len(recent_volume)
        if recent_volume else 0
    )

    volume_ratio = (
        candles[-1]["volume"] / average_volume
        if average_volume > 0 else 1
    )

    return {
        "price": closes[-1],
        "ema9": e9,
        "ema21": e21,
        "ema50": e50,
        "rsi": rsi,
        "macd": macd,
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "adx": adx,
        "volume_ratio": volume_ratio,
        "bull": e9 > e21 > e50,
        "bear": e9 < e21 < e50
    }


# ============================================================
# WHALE INTELLIGENCE
# ============================================================

def whale_tracker(symbol, trades, bids, asks, oi):

    threshold = WHALE_THRESHOLDS[symbol]

    buy_volume = 0.0
    sell_volume = 0.0
    whale_count = 0

    for trade in trades:

        price = number(trade.get("price"))
        quantity = number(trade.get("size"))

        value = price * quantity

        side = str(
            trade.get("side", "")
        ).lower()

        if value >= threshold:

            whale_count += 1

            if side == "buy":
                buy_volume += value

            elif side == "sell":
                sell_volume += value

    bids = bids[:20]
    asks = asks[:20]

    bid_depth = sum(p * q for p, q in bids)
    ask_depth = sum(p * q for p, q in asks)

    total_depth = bid_depth + ask_depth

    imbalance = (
        bid_depth / total_depth
        if total_depth > 0 else 0.5
    )

    biggest_bid_wall = max(
        (p * q for p, q in bids),
        default=0
    )

    biggest_ask_wall = max(
        (p * q for p, q in asks),
        default=0
    )

    previous_oi = number(
        state["last_oi"].get(symbol)
    )

    oi_change = 0.0

    if previous_oi > 0 and oi > 0:

        oi_change = (
            (oi - previous_oi) / previous_oi
        ) * 100

    state["last_oi"][symbol] = oi

    difference = buy_volume - sell_volume

    if difference > threshold:
        signal = "BUY PRESSURE"

    elif difference < -threshold:
        signal = "SELL PRESSURE"

    else:
        signal = "NEUTRAL"

    return {
        "buys": buy_volume,
        "sells": sell_volume,
        "count": whale_count,
        "imbalance": imbalance,
        "bid_wall": biggest_bid_wall,
        "ask_wall": biggest_ask_wall,
        "oi_change": oi_change,
        "signal": signal
    }


# ============================================================
# AI SIGNAL ENGINE
# ============================================================

def AI_SCORE(tf5, tf15, whale, funding):

    score = 50
    reasons = []

    if tf5["bull"]:
        score += 12
        reasons.append("5m bullish EMA")

    elif tf5["bear"]:
        score -= 12
        reasons.append("5m bearish EMA")

    if tf15["bull"]:
        score += 16
        reasons.append("15m bullish EMA")

    elif tf15["bear"]:
        score -= 16
        reasons.append("15m bearish EMA")

    if tf5["rsi"] is not None:

        if 52 <= tf5["rsi"] <= 70:
            score += 6

        elif tf5["rsi"] < 35:
            score += 2

        elif tf5["rsi"] > 75:
            score -= 7

    if tf15["rsi"] is not None:

        if 50 <= tf15["rsi"] <= 68:
            score += 5

        elif tf15["rsi"] > 75:
            score -= 5

    if tf5["macd"] is not None:

        if tf5["macd"] > 0:
            score += 5
            reasons.append("MACD positive")

        else:
            score -= 5
            reasons.append("MACD negative")

    if tf5["adx"] is not None and tf5["adx"] >= 20:

        if tf5["bull"]:
            score += 4

        elif tf5["bear"]:
            score -= 4

        reasons.append("Strong trend")

    if tf5["volume_ratio"] >= 1.8:

        if tf5["bull"]:
            score += 5

        elif tf5["bear"]:
            score -= 5

        reasons.append("Volume spike")

    if whale["signal"] == "BUY PRESSURE":
        score += 10
        reasons.append("Whale buying")

    elif whale["signal"] == "SELL PRESSURE":
        score -= 10
        reasons.append("Whale selling")

    if whale["imbalance"] >= 0.60:
        score += 6
        reasons.append("Buy order book pressure")

    elif whale["imbalance"] <= 0.40:
        score -= 6
        reasons.append("Sell order book pressure")

    if whale["oi_change"] >= 1:

        if score > 55:
            score += 3

        elif score < 45:
            score -= 3

        reasons.append("Open interest increasing")

    elif whale["oi_change"] <= -1:
        reasons.append("Open interest decreasing")

    if funding > 0.0008:
        score -= 3
        reasons.append("Crowded longs")

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


# ============================================================
# PAPER TRADING
# ============================================================

def calculate_pnl(position, price):

    direction = (
        1 if position["side"] == "LONG"
        else -1
    )

    return (
        (price - position["entry"])
        * direction
        * position["quantity"]
    )


def calculate_equity():

    unrealized = 0.0

    for symbol, position in state["positions"].items():

        price = number(
            market_prices.get(symbol),
            position["entry"]
        )

        unrealized += calculate_pnl(
            position, price
        )

    return state["balance"] + unrealized


market_prices = {}


def close_position(symbol, price, reason):

    position = state["positions"].pop(
        symbol, None
    )

    if not position:
        return

    gross = calculate_pnl(position, price)

    entry_fee = (
        position["entry"]
        * position["quantity"]
        * FEE_RATE_PER_SIDE
    )

    exit_fee = (
        price
        * position["quantity"]
        * FEE_RATE_PER_SIDE
    )

    fees = entry_fee + exit_fee

    net = gross - fees

    state["balance"] += net
    state["daily_realized"] += net

    trade = {
        "symbol": symbol,
        "side": position["side"],
        "entry": position["entry"],
        "exit": price,
        "quantity": position["quantity"],
        "gross_pnl": gross,
        "fees": fees,
        "net_pnl": net,
        "reason": reason,
        "time": timestamp()
    }

    state["closed_trades"].append(trade)

    state["closed_trades"] = (
        state["closed_trades"][-200:]
    )

    log.info(
        "POSITION CLOSED %s %s net=%s",
        symbol,
        reason,
        net
    )

    save_state()


def open_position(symbol, side, price, score):

    if symbol in state["positions"]:
        return

    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return

    if state["daily_realized"] <= -DAILY_LOSS_LIMIT:
        log.warning("Daily loss limit reached")
        return

    equity = calculate_equity()

    if equity <= 0:
        return

    risk_amount = equity * RISK_PER_TRADE

    stop_distance = price * STOP_LOSS

    if stop_distance <= 0:
        return

    quantity_by_risk = (
        risk_amount / stop_distance
    )

    maximum_notional = (
        equity * MAX_POSITION_VALUE * LEVERAGE
    )

    quantity_by_cap = (
        maximum_notional / price
    )

    quantity = min(
        quantity_by_risk,
        quantity_by_cap
    )

    if quantity <= 0:
        return

    if side == "LONG":

        tp = price * (1 + TAKE_PROFIT)
        sl = price * (1 - STOP_LOSS)

    else:

        tp = price * (1 - TAKE_PROFIT)
        sl = price * (1 + STOP_LOSS)

    state["positions"][symbol] = {
        "side": side,
        "entry": price,
        "quantity": quantity,
        "tp": tp,
        "sl": sl,
        "score": score,
        "leverage": LEVERAGE,
        "opened": timestamp()
    }

    log.info(
        "NEW PAPER POSITION %s %s entry=%s TP=%s SL=%s",
        symbol, side, price, tp, sl
    )

    save_state()


def manage_positions(prices):

    for symbol in list(state["positions"].keys()):

        if symbol not in prices:
            continue

        price = prices[symbol]

        position = state["positions"][symbol]

        if position["side"] == "LONG":

            if price >= position["tp"]:

                close_position(
                    symbol, price, "TAKE PROFIT"
                )

            elif price <= position["sl"]:

                close_position(
                    symbol, price, "STOP LOSS"
                )

        else:

            if price <= position["tp"]:

                close_position(
                    symbol, price, "TAKE PROFIT"
                )

            elif price >= position["sl"]:

                close_position(
                    symbol, price, "STOP LOSS"
                )


# ============================================================
# SCAN ENGINE
# ============================================================

def run_scan():

    rows = []
    prices = {}

    for symbol in SYMBOLS:

        try:

            candles5 = get_candles(symbol, "5")
            candles15 = get_candles(symbol, "15")

            tf5 = analyze(candles5)
            tf15 = analyze(candles15)

            ticker = get_ticker(symbol)

            price = number(
                ticker.get("lastPrice"),
                tf5["price"]
            )

            prices[symbol] = price

            bids, asks = get_orderbook(symbol)

            trades = get_trades(symbol)

            oi = get_open_interest(symbol)

            whale = whale_tracker(
                symbol,
                trades,
                bids,
                asks,
                oi
            )

            funding = number(
                ticker.get("fundingRate")
            )

            score, direction, reasons = AI_SCORE(
                tf5,
                tf15,
                whale,
                funding
            )

            rows.append({
                "symbol": symbol,
                "price": price,
                "score": score,
                "direction": direction,
                "rsi5": tf5["rsi"],
                "rsi15": tf15["rsi"],
                "adx": tf5["adx"],
                "volume_ratio": tf5["volume_ratio"],
                "funding": funding,
                "whale": whale,
                "reasons": reasons
            })

        except Exception as error:

            log.exception("Scan error for %s", symbol)

    # Update current prices

    market_prices.update(prices)

    # Always manage existing positions first

    manage_positions(prices)

    # Open new simulated positions

    for row in rows:

        if row["direction"] in ("LONG", "SHORT"):

            open_position(
                row["symbol"],
                row["direction"],
                row["price"],
                row["score"]
            )

    state["scan_count"] += 1

    market_status["rows"] = rows
    market_status["last_scan"] = timestamp()
    market_status["scan_count"] = state["scan_count"]
    market_status["last_error"] = None

    save_state()

    log.info(
        "Scan completed. Symbols=%s positions=%s",
        len(rows),
        len(state["positions"])
    )


# ============================================================
# BACKGROUND ENGINE
# ============================================================

def trading_engine():

    log.info("CoinSentra background engine started")

    market_status["running"] = True

    while True:

        started = time.time()

        try:

            today = str(datetime.now(LOCAL_TZ).date())

            if state["day"] != today:

                state["day"] = today
                state["daily_realized"] = 0.0

                save_state()

                log.info("Daily counters reset")

            run_scan()

        except Exception as error:

            market_status["last_error"] = str(error)

            log.exception("Trading engine error")

        elapsed = time.time() - started

        time.sleep(
            max(1, SCAN_SECONDS - elapsed)
        )


# Start exactly one engine in the single Gunicorn worker.

engine_lock = threading.Lock()

def start_engine():

    with engine_lock:

        if not any(
            t.name == "coinsentra-engine" and t.is_alive()
            for t in threading.enumerate()
        ):

            thread = threading.Thread(
                target=trading_engine,
                name="coinsentra-engine",
                daemon=True
            )

            thread.start()


# ============================================================
# DASHBOARD AUTHENTICATION
# ============================================================

@app.before_request
def protect_dashboard():

    if request.path == "/health":
        return None

    auth = request.authorization

    if (
        not auth
        or not secrets.compare_digest(
            auth.password or "",
            DASHBOARD_PASSWORD
        )
    ):

        return Response(
            "Authentication required",
            401,
            {
                "WWW-Authenticate":
                'Basic realm="CoinSentra Private Dashboard"'
            }
        )


# ============================================================
# API STATUS
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "app": APP_NAME
    })


@app.route("/api/status")
def api_status():

    with state_lock:

        current_state = deepcopy(state)
        rows = deepcopy(market_status["rows"])

    positions = []

    for symbol, position in current_state["positions"].items():

        price = number(
            market_prices.get(symbol),
            position["entry"]
        )

        pnl = calculate_pnl(position, price)

        positions.append({
            "symbol": symbol,
            "side": position["side"],
            "entry": position["entry"],
            "price": price,
            "quantity": position["quantity"],
            "tp": position["tp"],
            "sl": position["sl"],
            "score": position["score"],
            "leverage": position["leverage"],
            "pnl": pnl,
            "opened": position["opened"]
        })

    unrealized = sum(
        p["pnl"] for p in positions
    )

    wins = sum(
        1 for t in current_state["closed_trades"]
        if t["net_pnl"] > 0
    )

    losses = sum(
        1 for t in current_state["closed_trades"]
        if t["net_pnl"] < 0
    )

    total_closed = len(current_state["closed_trades"])

    win_rate = (
        wins / total_closed * 100
        if total_closed else 0
    )

    return jsonify({
        "app": APP_NAME,
        "time": timestamp(),
        "running": market_status["running"],
        "last_scan": market_status["last_scan"],
        "last_error": market_status["last_error"],
        "scan_count": current_state["scan_count"],
        "balance": current_state["balance"],
        "equity": current_state["balance"] + unrealized,
        "daily_realized": current_state["daily_realized"],
        "unrealized": unrealized,
        "daily_limit": DAILY_LOSS_LIMIT,
        "leverage": LEVERAGE,
        "take_profit": TAKE_PROFIT * 100,
        "stop_loss": STOP_LOSS * 100,
        "positions": positions,
        "position_count": len(positions),
        "max_positions": MAX_OPEN_POSITIONS,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 2),
        "rows": rows,
        "trades": current_state["closed_trades"][-30:]
    })


# ============================================================
# WEB DASHBOARD
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">

<title>CoinSentra Trading Brain V8</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #080d18;
    color: #e9f0ff;
    font-family: Arial, sans-serif;
}

header {
    background: #101a2d;
    padding: 22px;
    border-bottom: 1px solid #263650;
}

header h1 {
    margin: 0;
    color: #48d8ff;
    font-size: 23px;
}

header p {
    color: #91a6c5;
    margin-bottom: 0;
}

.container {
    padding: 16px;
    max-width: 1250px;
    margin: auto;
}

.grid {
    display: grid;
    grid-template-columns: repeat(
        auto-fit, minmax(145px, 1fr)
    );
    gap: 12px;
}

.card {
    background: #111c30;
    border: 1px solid #253650;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}

.card small {
    color: #92a8c8;
}

.value {
    font-size: 23px;
    font-weight: bold;
    margin-top: 9px;
}

h2 {
    color: #48d8ff;
    font-size: 18px;
    margin-top: 10px;
}

table {
    width: 100%;
    border-collapse: collapse;
    min-width: 720px;
}

th, td {
    padding: 11px 9px;
    border-bottom: 1px solid #263650;
    text-align: left;
    font-size: 12px;
}

th {
    color: #48d8ff;
}

.table-wrap {
    overflow-x: auto;
}

.green {
    color: #31e6a0;
}

.red {
    color: #ff6478;
}

.yellow {
    color: #ffd166;
}

.muted {
    color: #91a6c5;
}

.pill {
    padding: 5px 9px;
    border-radius: 20px;
    background: #263650;
    display: inline-block;
    font-size: 11px;
}

.status {
    color: #31e6a0;
}

.footer {
    text-align: center;
    padding: 25px;
    color: #7286a6;
    font-size: 12px;
}

button {
    background: #163d5a;
    border: 1px solid #48d8ff;
    color: #48d8ff;
    padding: 10px 17px;
    border-radius: 7px;
    cursor: pointer;
}

</style>
</head>

<body>

<header>
    <h1>COINSENTRA AI</h1>
    <p>Trading Brain V8 | Whale Intelligence</p>
    <span class="pill status" id="status">CONNECTING</span>
    <span class="pill">PAPER TRADING ONLY</span>
</header>

<div class="container">

    <div class="grid">

        <div class="card">
            <small>Balance</small>
            <div class="value" id="balance">$100.00</div>
        </div>

        <div class="card">
            <small>Total Equity</small>
            <div class="value" id="equity">$100.00</div>
        </div>

        <div class="card">
            <small>Realized Daily PnL</small>
            <div class="value" id="daily">$0.00</div>
        </div>

        <div class="card">
            <small>Unrealized PnL</small>
            <div class="value" id="unrealized">$0.00</div>
        </div>

        <div class="card">
            <small>Open Positions</small>
            <div class="value" id="open">0 / 2</div>
        </div>

        <div class="card">
            <small>Win Rate</small>
            <div class="value" id="winrate">0%</div>
        </div>

    </div>

    <div class="card">
        <h2>Trading Configuration</h2>
        <p id="config" class="muted"></p>
        <p class="muted">
            Exchange: Bybit | Timeframes: 5m / 15m
        </p>
        <p class="muted">
            Market data: Public API | Execution: Simulated
        </p>
    </div>

    <div class="card">
        <h2>AI Market Scanner</h2>
        <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Price</th>
                    <th>AI Score</th>
                    <th>Signal</th>
                    <th>5m RSI</th>
                    <th>15m RSI</th>
                    <th>ADX</th>
                    <th>Volume</th>
                </tr>
            </thead>
            <tbody id="markets">
                <tr><td colspan="8">Waiting for market scan...</td></tr>
            </tbody>
        </table>
        </div>
    </div>

    <div class="card">
        <h2>Whale Intelligence</h2>
        <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Whale Signal</th>
                    <th>Whale Buys</th>
                    <th>Whale Sells</th>
                    <th>Large Trades</th>
                    <th>Book Imbalance</th>
                    <th>Bid Wall</th>
                    <th>Ask Wall</th>
                    <th>OI Change</th>
                </tr>
            </thead>
            <tbody id="whales">
                <tr><td colspan="9">Waiting...</td></tr>
            </tbody>
        </table>
        </div>
    </div>

    <div class="card">
        <h2>Open Positions</h2>
        <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Entry</th>
                    <th>Current</th>
                    <th>Quantity</th>
                    <th>Take Profit</th>
                    <th>Stop Loss</th>
                    <th>Unrealized PnL</th>
                </tr>
            </thead>
            <tbody id="positions">
                <tr><td colspan="8">No open positions</td></tr>
            </tbody>
        </table>
        </div>
    </div>

    <div class="card">
        <h2>Closed Trades</h2>
        <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>Reason</th>
                    <th>Net PnL</th>
                </tr>
            </thead>
            <tbody id="trades">
                <tr><td colspan="7">No closed trades</td></tr>
            </tbody>
        </table>
        </div>
    </div>

    <div class="card">
        <h2>System Information</h2>
        <p>Last scan: <span id="lastscan">Waiting...</span></p>
        <p>Scan count: <span id="scans">0</span></p>
        <p>Server time: <span id="clock">--</span></p>
        <p id="error" class="red"></p>
        <button onclick="refreshData()">REFRESH DASHBOARD</button>
    </div>

</div>

<div class="footer">
    COINSENTRA TRADING BRAIN V8 |
    PRIVATE PAPER TRADING SYSTEM
</div>

<script>

function fmt(value, digits=4) {
    let n = Number(value);
    return Number.isFinite(n) ? n.toFixed(digits) : "--";
}

function pnlClass(value) {
    return Number(value) >= 0 ? "green" : "red";
}

function money(value) {
    return "$" + fmt(value, 2);
}

function safe(value) {
    return String(value ?? "").replace(
        /[&<>"']/g,
        c => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        })[c]
    );
}

async function refreshData() {

    try {

        const response = await fetch("/api/status", {
            cache: "no-store"
        });

        if (!response.ok) {
            throw new Error("Dashboard connection failed");
        }

        const d = await response.json();

        document.getElementById("status").textContent =
            d.running ? "ENGINE RUNNING" : "ENGINE STARTING";

        document.getElementById("balance").textContent =
            money(d.balance);

        document.getElementById("equity").textContent =
            money(d.equity);

        document.getElementById("daily").textContent =
            money(d.daily_realized);

        document.getElementById("daily").className =
            "value " + pnlClass(d.daily_realized);

        document.getElementById("unrealized").textContent =
            money(d.unrealized);

        document.getElementById("unrealized").className =
            "value " + pnlClass(d.unrealized);

        document.getElementById("open").textContent =
            d.position_count + " / " + d.max_positions;

        document.getElementById("winrate").textContent =
            fmt(d.win_rate, 1) + "%";

        document.getElementById("config").textContent =
            "Leverage: " + d.leverage +
            "x | TP: " + d.take_profit +
            "% | SL: " + d.stop_loss +
            "% | Daily limit: $" + d.daily_limit;

        document.getElementById("lastscan").textContent =
            d.last_scan || "Waiting";

        document.getElementById("scans").textContent =
            d.scan_count;

        document.getElementById("clock").textContent =
            d.time;

        document.getElementById("error").textContent =
            d.last_error ? "ERROR: " + d.last_error : "";

        // MARKET SCANNER

        document.getElementById("markets").innerHTML =
            d.rows.map(r => {

                let cls =
                    r.direction === "LONG" ? "green" :
                    r.direction === "SHORT" ? "red" :
                    "yellow";

                return `
                <tr>
                    <td>${safe(r.symbol)}</td>
                    <td>${fmt(r.price, 6)}</td>
                    <td><b>${fmt(r.score, 0)}</b></td>
                    <td class="${cls}">${safe(r.direction)}</td>
                    <td>${fmt(r.rsi5, 1)}</td>
                    <td>${fmt(r.rsi15, 1)}</td>
                    <td>${fmt(r.adx, 1)}</td>
                    <td>${fmt(r.volume_ratio, 2)}x</td>
                </tr>`;
            }).join("") ||
            '<tr><td colspan="8">Waiting for market scan</td></tr>';

        // WHALE TRACKER

        document.getElementById("whales").innerHTML =
            d.rows.map(r => {

                const w = r.whale;

                let cls =
                    w.signal === "BUY PRESSURE" ? "green" :
                    w.signal === "SELL PRESSURE" ? "red" :
                    "yellow";

                return `
                <tr>
                    <td>${safe(r.symbol)}</td>
                    <td class="${cls}">${safe(w.signal)}</td>
                    <td>${money(w.buys)}</td>
                    <td>${money(w.sells)}</td>
                    <td>${w.count}</td>
                    <td>${fmt(w.imbalance, 3)}</td>
                    <td>${money(w.bid_wall)}</td>
                    <td>${money(w.ask_wall)}</td>
                    <td>${fmt(w.oi_change, 2)}%</td>
                </tr>`;
            }).join("") ||
            '<tr><td colspan="9">Waiting</td></tr>';

        // OPEN POSITIONS

        document.getElementById("positions").innerHTML =
            d.positions.map(p => `
            <tr>
                <td>${safe(p.symbol)}</td>
                <td>${safe(p.side)}</td>
                <td>${fmt(p.entry, 6)}</td>
                <td>${fmt(p.price, 6)}</td>
                <td>${fmt(p.quantity, 6)}</td>
                <td>${fmt(p.tp, 6)}</td>
                <td>${fmt(p.sl, 6)}</td>
                <td class="${pnlClass(p.pnl)}">${money(p.pnl)}</td>
            </tr>
            `).join("") ||
            '<tr><td colspan="8">No open positions</td></tr>';

        // CLOSED TRADES

        document.getElementById("trades").innerHTML =
            d.trades.slice().reverse().map(t => `
            <tr>
                <td>${safe(t.time)}</td>
                <td>${safe(t.symbol)}</td>
                <td>${safe(t.side)}</td>
                <td>${fmt(t.entry, 6)}</td>
                <td>${fmt(t.exit, 6)}</td>
                <td>${safe(t.reason)}</td>
                <td class="${pnlClass(t.net_pnl)}">${money(t.net_pnl)}</td>
            </tr>
            `).join("") ||
            '<tr><td colspan="7">No closed trades</td></tr>';

    } catch(error) {

        document.getElementById("status").textContent =
            "CONNECTION ERROR";

        document.getElementById("error").textContent =
            error.message;
    }
}

refreshData();

setInterval(refreshData, 15000);

</script>

</body>
</html>
"""


@app.route("/")
def home():

    start_engine()

    return HTML


# ============================================================
# STARTUP
# ============================================================

start_engine()


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False
    )
