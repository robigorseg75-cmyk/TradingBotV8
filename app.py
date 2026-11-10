
# ============================================================
# COINSENTRA TRADING BRAIN V9
# RENDER EDITION - REMOTE START / STOP
# BYBIT PUBLIC DATA - PAPER TRADING ONLY
# ============================================================

import os
import json
import time
import math
import logging
import threading
from datetime import datetime, date
from functools import wraps

from flask import Flask, jsonify, request, render_template_string
from pybit.unified_trading import HTTP


# ============================================================
# SETTINGS
# ============================================================

APP_NAME = "CoinSentra Trading Brain V9"

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 10000))

TESTNET = False
CATEGORY = "linear"

SYMBOLS = ["ETHUSDT", "XRPUSDT", "SOLUSDT", "SUIUSDT"]

LEVERAGE = 8
SCAN_SECONDS = 60

TAKE_PROFIT = 0.02
STOP_LOSS = 0.02

DAILY_LOSS_LIMIT = 6.0
STARTING_BALANCE = 100.0

RISK_PER_TRADE = 0.005
MAX_POSITION_VALUE = 0.10
MAX_OPEN_POSITIONS = 2

MIN_SIGNAL_SCORE = 70
ORDERBOOK_DEPTH = 50

STATE_FILE = "coinsentra_v9_state.json"

CONTROL_PIN = os.environ.get("CONTROL_PIN", "")

WHALE_THRESHOLDS = {
    "ETHUSDT": 100000,
    "XRPUSDT": 50000,
    "SOLUSDT": 50000,
    "SUIUSDT": 25000
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = Flask(__name__)

session = HTTP(testnet=TESTNET)

state_lock = threading.RLock()
engine_lock = threading.Lock()

trading_enabled = False
engine_started = False
last_scan = "Not scanned yet"
last_error = ""
engine_status = "STARTING"

stop_event = threading.Event()


# ============================================================
# STATE
# ============================================================

def new_state():
    return {
        "balance": STARTING_BALANCE,
        "day": str(date.today()),
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

            defaults = new_state()

            for key, value in defaults.items():
                if key not in data:
                    data[key] = value

            return data

        except Exception:
            logging.exception("State loading failed")

    return new_state()


state = load_state()


def save_state():

    with state_lock:

        temporary = STATE_FILE + ".tmp"

        with open(temporary, "w") as f:
            json.dump(state, f, indent=2)

        os.replace(temporary, STATE_FILE)


def number(value, default=0.0):

    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def timestamp():

    return datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# AUTHENTICATION
# ============================================================

def authorized():

    if not CONTROL_PIN:
        return False

    supplied = request.headers.get(
        "X-Control-PIN",
        request.form.get("pin", "")
    )

    return supplied == CONTROL_PIN


def require_pin(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if not authorized():
            return jsonify({
                "success": False,
                "error": "Incorrect PIN or CONTROL_PIN not configured"
            }), 403

        return function(*args, **kwargs)

    return wrapper


# ============================================================
# BYBIT PUBLIC DATA
# ============================================================

def api_result(response):

    if response.get("retCode") != 0:
        raise RuntimeError(str(response))

    return response.get("result", {})


def get_candles(symbol, interval):

    result = api_result(
        session.get_kline(
            category=CATEGORY,
            symbol=symbol,
            interval=interval,
            limit=220
        )
    )

    rows = list(reversed(result.get("list", [])))

    return [{
        "open": number(x[1]),
        "high": number(x[2]),
        "low": number(x[3]),
        "close": number(x[4]),
        "volume": number(x[5]),
        "turnover": number(x[6])
    } for x in rows]


def get_ticker(symbol):

    result = api_result(
        session.get_tickers(
            category=CATEGORY,
            symbol=symbol
        )
    )

    rows = result.get("list", [])

    if not rows:
        raise RuntimeError("Ticker unavailable")

    return rows[0]


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
        for x in result.get("b", [])
    ]

    asks = [
        (number(x[0]), number(x[1]))
        for x in result.get("a", [])
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

    return (
        number(rows[0].get("openInterest"))
        if rows else 0.0
    )


# ============================================================
# INDICATORS
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

    return 100 - 100 / (1 + avg_gain / avg_loss)


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

    tr_values = []
    plus_values = []
    minus_values = []

    for i in range(1, len(candles)):

        current = candles[i]
        previous = candles[i - 1]

        up = current["high"] - previous["high"]
        down = previous["low"] - current["low"]

        plus_values.append(
            up if up > down and up > 0 else 0
        )

        minus_values.append(
            down if down > up and down > 0 else 0
        )

        tr_values.append(max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"])
        ))

    dx_values = []

    for i in range(period, len(tr_values)):

        start = i - period + 1

        tr = sum(tr_values[start:i + 1])

        if tr <= 0:
            continue

        plus = 100 * sum(
            plus_values[start:i + 1]
        ) / tr

        minus = 100 * sum(
            minus_values[start:i + 1]
        ) / tr

        total = plus + minus

        dx = (
            100 * abs(plus - minus) / total
            if total > 0 else 0
        )

        dx_values.append(dx)

    if len(dx_values) < period:
        return None

    return sum(dx_values[-period:]) / period


def analyze(candles):

    closes = [x["close"] for x in candles]

    if len(closes) < 60:
        raise ValueError("Not enough candles")

    e9 = EMA(closes, 9)
    e21 = EMA(closes, 21)
    e50 = EMA(closes, 50)

    upper, middle, lower = BOLLINGER(closes)

    volumes = [x["volume"] for x in candles[-21:-1]]
    average_volume = sum(volumes) / len(volumes) if volumes else 0

    volume_ratio = (
        candles[-1]["volume"] / average_volume
        if average_volume > 0 else 1
    )

    return {
        "price": closes[-1],
        "ema9": e9,
        "ema21": e21,
        "ema50": e50,
        "rsi": RSI(closes),
        "macd": MACD(closes),
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "adx": ADX(candles),
        "volume_ratio": volume_ratio,
        "bull": e9 > e21 > e50,
        "bear": e9 < e21 < e50
    }


# ============================================================
# WHALE TRACKER
# ============================================================

def whale_tracker(symbol, trades, bids, asks, oi):

    threshold = WHALE_THRESHOLDS[symbol]

    buys = 0.0
    sells = 0.0
    count = 0

    for trade in trades:

        value = (
            number(trade.get("price"))
            * number(trade.get("size"))
        )

        side = str(trade.get("side", "")).lower()

        if value >= threshold:

            count += 1

            if side == "buy":
                buys += value

            elif side == "sell":
                sells += value

    bid_depth = sum(p * q for p, q in bids[:20])
    ask_depth = sum(p * q for p, q in asks[:20])

    total = bid_depth + ask_depth

    imbalance = bid_depth / total if total > 0 else 0.5

    bid_wall = max(
        [p * q for p, q in bids[:20]],
        default=0
    )

    ask_wall = max(
        [p * q for p, q in asks[:20]],
        default=0
    )

    previous = number(state["last_oi"].get(symbol))

    oi_change = 0.0

    if previous > 0 and oi > 0:
        oi_change = (oi - previous) / previous * 100

    if oi > 0:
        state["last_oi"][symbol] = oi

    difference = buys - sells

    if difference > threshold:
        signal = "BUY PRESSURE"
    elif difference < -threshold:
        signal = "SELL PRESSURE"
    else:
        signal = "NEUTRAL"

    return {
        "buys": buys,
        "sells": sells,
        "count": count,
        "imbalance": imbalance,
        "bid_wall": bid_wall,
        "ask_wall": ask_wall,
        "oi_change": oi_change,
        "signal": signal
    }


# ============================================================
# AI SCORE
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
# PAPER POSITION MANAGEMENT
# ============================================================

def calculate_pnl(position, price):

    direction = 1 if position["side"] == "LONG" else -1

    return (
        (price - position["entry"])
        * direction
        * position["quantity"]
    )


def close_position(symbol, price, reason):

    with state_lock:

        position = state["positions"].pop(symbol, None)

        if not position:
            return

        gross = calculate_pnl(position, price)

        fees = (
            position["entry"]
            * position["quantity"]
            * 0.0011
        )

        net = gross - fees

        state["balance"] += net
        state["daily_realized"] += net

        state["closed_trades"].append({
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
        })

        state["closed_trades"] = state["closed_trades"][-200:]

    logging.info(
        "POSITION CLOSED %s %s net=%s",
        symbol, reason, net
    )

    print("CLOSED:", symbol, reason, "NET:", round(net, 3))

    save_state()


def open_position(symbol, side, price, score):

    global trading_enabled

    with state_lock:

        if not trading_enabled:
            return

        if symbol in state["positions"]:
            return

        if len(state["positions"]) >= MAX_OPEN_POSITIONS:
            return

        if state["daily_realized"] <= -DAILY_LOSS_LIMIT:
            return

        equity = state["balance"]

        risk_amount = equity * RISK_PER_TRADE
        stop_distance = price * STOP_LOSS

        if stop_distance <= 0 or equity <= 0:
            return

        quantity_by_risk = risk_amount / stop_distance

        maximum_notional = (
            equity * MAX_POSITION_VALUE * LEVERAGE
        )

        quantity_by_cap = maximum_notional / price

        quantity = min(quantity_by_risk, quantity_by_cap)

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

    print("NEW PAPER POSITION:", symbol, side, price)

    save_state()


def manage_positions(prices):

    for symbol in list(state["positions"].keys()):

        if symbol not in prices:
            continue

        price = prices[symbol]
        position = state["positions"].get(symbol)

        if not position:
            continue

        if position["side"] == "LONG":

            if price >= position["tp"]:
                close_position(symbol, price, "TAKE PROFIT")

            elif price <= position["sl"]:
                close_position(symbol, price, "STOP LOSS")

        else:

            if price <= position["tp"]:
                close_position(symbol, price, "TAKE PROFIT")

            elif price >= position["sl"]:
                close_position(symbol, price, "STOP LOSS")


# ============================================================
# MARKET SCANNER
# ============================================================

def run_scan():

    global last_scan, last_error

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
                symbol, trades, bids, asks, oi
            )

            funding = number(ticker.get("fundingRate"))

            score, direction, reasons = AI_SCORE(
                tf5, tf15, whale, funding
            )

            rows.append({
                "symbol": symbol,
                "price": price,
                "score": score,
                "direction": direction,
                "rsi5": tf5["rsi"],
                "rsi15": tf15["rsi"],
                "whale": whale,
                "reasons": reasons
            })

        except Exception as error:

            last_error = f"{symbol}: {error}"

            logging.exception("Scan error: %s", symbol)

            print("DATA ERROR:", symbol, error)

    # Existing positions are monitored even when stopped.
    manage_positions(prices)

    # New positions only when START is enabled.
    if trading_enabled:

        for row in rows:

            if row["direction"] in ("LONG", "SHORT"):

                open_position(
                    row["symbol"],
                    row["direction"],
                    row["price"],
                    row["score"]
                )

    with state_lock:
        state["scan_count"] += 1

    save_state()

    last_scan = timestamp()
    last_error = ""

    print("\n" + "=" * 60)
    print(APP_NAME)
    print("SCAN:", last_scan)
    print("TRADING ENABLED:", trading_enabled)
    print("BALANCE:", round(state["balance"], 2))
    print("DAILY PNL:", round(state["daily_realized"], 2))
    print("OPEN POSITIONS:", len(state["positions"]))

    for row in rows:

        print(
            row["symbol"],
            "PRICE:", row["price"],
            "SCORE:", row["score"],
            "SIGNAL:", row["direction"],
            "RSI5:", row["rsi5"],
            "RSI15:", row["rsi15"],
            "WHALE:", row["whale"]["signal"]
        )

    print("=" * 60)


# ============================================================
# BACKGROUND ENGINE
# ============================================================

def engine_loop():

    global engine_status

    engine_status = "RUNNING"

    print("BACKGROUND ENGINE STARTED")

    while not stop_event.is_set():

        if not engine_lock.acquire(blocking=False):
            stop_event.wait(2)
            continue

        try:

            today = str(date.today())

            if state["day"] != today:

                state["day"] = today
                state["daily_realized"] = 0.0

                save_state()

            run_scan()

        except Exception as error:

            last_message = str(error)

            logging.exception("Engine error")

            print("ENGINE ERROR:", last_message)

            engine_status = "ERROR"

        finally:
            engine_lock.release()

        engine_status = "RUNNING"

        stop_event.wait(SCAN_SECONDS)

    engine_status = "STOPPED"


def start_engine():

    global engine_started

    with engine_lock:

        if engine_started:
            return

        stop_event.clear()

        thread = threading.Thread(
            target=engine_loop,
            daemon=True,
            name="CoinSentraEngine"
        )

        engine_started = True
        thread.start()


# ============================================================
# WEB DASHBOARD
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="15">
<title>CoinSentra Trading Brain V9</title>

<style>
body {
    background: #101827;
    color: white;
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 18px;
}
.container {
    max-width: 850px;
    margin: auto;
}
h1 {
    color: #42d9ff;
    font-size: 25px;
}
.card {
    background: #1b293b;
    border-radius: 13px;
    padding: 17px;
    margin: 12px 0;
}
.status {
    font-size: 20px;
    font-weight: bold;
}
button {
    border: 0;
    border-radius: 10px;
    padding: 17px;
    width: 48%;
    font-size: 18px;
    font-weight: bold;
    color: white;
    margin: 5px 1%;
}
.stop { background: #d83c4d; }
.start { background: #159b67; }
input {
    width: 95%;
    padding: 13px;
    border-radius: 8px;
    border: 0;
    margin: 8px 0;
    font-size: 16px;
}
.small { color: #9fb2c9; }
.green { color: #39dc99; }
.red { color: #ff6476; }
.row {
    display: flex;
    justify-content: space-between;
    gap: 10px;
}
</style>
</head>

<body>
<div class="container">

<h1>COINSENTRA TRADING BRAIN V9</h1>
<p class="small">Bybit Whale Intelligence | Render Edition</p>

<div class="card">
    <div>TRADING STATUS</div>
    <div class="status {{ 'green' if enabled else 'red' }}">
        {{ 'ACTIVE - NEW PAPER TRADES ENABLED' if enabled else 'STOPPED - NEW TRADES DISABLED' }}
    </div>
    <p class="small">Engine: {{ engine_status }}</p>
    <p class="small">Last scan: {{ last_scan }}</p>
    <p class="small">Last error: {{ last_error or 'None' }}</p>
</div>

<div class="card">
    <h3>REMOTE CONTROL</h3>
    <input id="pin" type="password" placeholder="Enter your control PIN">
    <div class="row">
        <button class="stop" onclick="control('stop')">C - STOP</button>
        <button class="start" onclick="control('start')">V - START</button>
    </div>
    <p id="message" class="small">Enter PIN before controlling.</p>
    <p class="small">Keyboard shortcuts: C = Stop, V = Start.</p>
</div>

<div class="card">
    <h3>ACCOUNT</h3>
    <p>Paper balance: <b>$ {{ "%.2f"|format(balance) }}</b></p>
    <p>Daily realized PNL: <b>$ {{ "%.2f"|format(daily_pnl) }}</b></p>
    <p>Open positions: <b>{{ positions|length }} / 2</b></p>
    <p>Completed scans: <b>{{ scan_count }}</b></p>
</div>

<div class="card">
    <h3>SETTINGS</h3>
    <p>Leverage simulation: {{ leverage }}x</p>
    <p>Take profit: {{ tp }}%</p>
    <p>Stop loss: {{ sl }}%</p>
    <p>Daily loss limit: ${{ loss_limit }}</p>
    <p class="small">Paper trading only. No live orders.</p>
</div>

<div class="card">
    <h3>OPEN PAPER POSITIONS</h3>
    {% if positions %}
        {% for symbol, p in positions.items() %}
            <p>
            <b>{{ symbol }} - {{ p.side }}</b><br>
            Entry: {{ p.entry }}<br>
            TP: {{ p.tp }}<br>
            SL: {{ p.sl }}<br>
            Quantity: {{ "%.6f"|format(p.quantity) }}
            </p>
            <hr>
        {% endfor %}
    {% else %}
        <p class="small">No open positions.</p>
    {% endif %}
</div>

<div class="card">
    <h3>RECENT CLOSED TRADES</h3>
    {% for trade in trades %}
        <p>
        {{ trade.time }}<br>
        {{ trade.symbol }} {{ trade.side }} |
        {{ trade.reason }} |
        Net: ${{ "%.3f"|format(trade.net_pnl) }}
        </p>
        <hr>
    {% else %}
        <p class="small">No completed trades yet.</p>
    {% endfor %}
</div>

<p class="small">
CoinSentra V9 | Public Bybit market data | Paper trading
</p>

</div>

<script>
async function control(action) {
    const pin = document.getElementById("pin").value;
    const message = document.getElementById("message");

    message.textContent = "Sending command...";

    try {
        const response = await fetch("/control/" + action, {
            method: "POST",
            headers: {
                "X-Control-PIN": pin
            }
        });

        const result = await response.json();

        message.textContent = result.success
            ? result.message
            : result.error;

        if (result.success) {
            setTimeout(() => location.reload(), 700);
        }

    } catch (error) {
        message.textContent = "Connection error: " + error;
    }
}

document.addEventListener("keydown", function(event) {
    if (event.target.tagName === "INPUT") return;

    if (event.key.toLowerCase() === "c") {
        control("stop");
    }

    if (event.key.toLowerCase() === "v") {
        control("start");
    }
});
</script>

</body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def dashboard():

    with state_lock:

        positions = {
            symbol: dict(position)
            for symbol, position in state["positions"].items()
        }

        trades = list(reversed(state["closed_trades"][-10:]))

        return render_template_string(
            HTML,
            enabled=trading_enabled,
            engine_status=engine_status,
            last_scan=last_scan,
            last_error=last_error,
            balance=state["balance"],
            daily_pnl=state["daily_realized"],
            positions=positions,
            trades=trades,
            scan_count=state["scan_count"],
            leverage=LEVERAGE,
            tp=TAKE_PROFIT * 100,
            sl=STOP_LOSS * 100,
            loss_limit=DAILY_LOSS_LIMIT
        )


@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "engine": engine_status,
        "trading_enabled": trading_enabled
    })


@app.route("/status")
def status():

    return jsonify({
        "engine": engine_status,
        "trading_enabled": trading_enabled,
        "last_scan": last_scan,
        "last_error": last_error,
        "balance": state["balance"],
        "daily_pnl": state["daily_realized"],
        "open_positions": len(state["positions"]),
        "scan_count": state["scan_count"]
    })


@app.route("/control/stop", methods=["POST"])
@require_pin
def stop_trading():

    global trading_enabled

    with state_lock:
        trading_enabled = False

    save_state()

    logging.warning("REMOTE STOP COMMAND")

    return jsonify({
        "success": True,
        "message": "STOPPED: No new paper positions will open. Existing positions remain monitored."
    })


@app.route("/control/start", methods=["POST"])
@require_pin
def start_trading():

    global trading_enabled

    with state_lock:

        if state["daily_realized"] <= -DAILY_LOSS_LIMIT:

            return jsonify({
                "success": False,
                "error": "Daily loss limit reached. Cannot restart trading today."
            }), 403

        trading_enabled = True

    logging.warning("REMOTE START COMMAND")

    return jsonify({
        "success": True,
        "message": "STARTED: New paper trading positions are enabled."
    })


# ============================================================
# START WEB SERVER ENGINE
# ============================================================

start_engine()


if __name__ == "__main__":

    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        use_reloader=False
    )
