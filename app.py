
import os
import json
import time
import threading
import logging
from datetime import datetime, timezone

from flask import Flask, jsonify, request, render_template_string
from pybit.unified_trading import HTTP

# ============================================================
# COINSENTRA TRADING BRAIN V10 - RENDER EDITION
# PAPER TRADING ONLY - NO REAL ORDERS
# ============================================================

APP_NAME = "CoinSentra Trading Brain V10"

SYMBOLS = ["ETHUSDT", "XRPUSDT", "SOLUSDT", "SUIUSDT"]

LEVERAGE = 8
TAKE_PROFIT_PCT = 0.02
STOP_LOSS_PCT = 0.02
DAILY_LOSS_LIMIT = 6.0

STARTING_BALANCE = 100.0
RISK_PER_TRADE = 0.005
MAX_POSITION_VALUE_PCT = 0.10
MAX_OPEN_POSITIONS = 2
MIN_ENTRY_SCORE = 70

SCAN_INTERVAL = 60
CATEGORY = "linear"

WHALE_THRESHOLDS = {
    "ETHUSDT": 100000,
    "XRPUSDT": 50000,
    "SOLUSDT": 50000,
    "SUIUSDT": 25000
}

DATA_DIR = os.environ.get("DATA_DIR", "/var/data")
if os.path.isdir(DATA_DIR) and os.access(DATA_DIR, os.W_OK):
    STATE_FILE = os.path.join(DATA_DIR, "coinsentra_v10_state.json")
else:
    STATE_FILE = os.environ.get(
        "STATE_FILE", "coinsentra_v10_state.json"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = Flask(__name__)

# Public Bybit market data only. No API keys are used.
session = HTTP(testnet=False)

lock = threading.RLock()

state = {
    "balance": STARTING_BALANCE,
    "realized_pnl": 0.0,
    "daily_pnl": 0.0,
    "daily_date": datetime.now(timezone.utc).date().isoformat(),
    "positions": {},
    "history": [],
    "signals": {},
    "last_scan": "Not scanned yet",
    "last_error": "",
    "trading_enabled": False,
    "engine_started": False,
    "scan_count": 0
}


def now_text():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def save_state():
    try:
        with lock:
            data = {
                "balance": state["balance"],
                "realized_pnl": state["realized_pnl"],
                "daily_pnl": state["daily_pnl"],
                "daily_date": state["daily_date"],
                "positions": state["positions"],
                "history": state["history"][-200:],
                "signals": state["signals"],
                "last_scan": state["last_scan"],
                "trading_enabled": state["trading_enabled"],
                "scan_count": state["scan_count"]
            }

        folder = os.path.dirname(STATE_FILE)
        if folder:
            os.makedirs(folder, exist_ok=True)

        temp_file = STATE_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        os.replace(temp_file, STATE_FILE)

    except Exception as exc:
        logging.exception("State save failed: %s", exc)


def load_state():
    try:
        if not os.path.exists(STATE_FILE):
            return

        with open(STATE_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)

        with lock:
            for key in (
                "balance", "realized_pnl", "daily_pnl",
                "daily_date", "positions", "history",
                "signals", "last_scan", "scan_count"
            ):
                if key in saved:
                    state[key] = saved[key]

            # Always require the user to press START after a restart.
            state["trading_enabled"] = False

        logging.info("Previous paper-trading state loaded.")

    except Exception as exc:
        logging.exception("State load failed: %s", exc)


def number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ema(values, period):
    if len(values) < period:
        return None

    alpha = 2.0 / (period + 1.0)
    result = sum(values[:period]) / period

    for value in values[period:]:
        result = value * alpha + result * (1.0 - alpha)

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
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_macd(values):
    fast = ema(values, 12)
    slow = ema(values, 26)

    if fast is None or slow is None:
        return 0.0

    return fast - slow


def calculate_bollinger(values, period=20):
    if len(values) < period:
        return None, None, None

    recent = values[-period:]
    middle = sum(recent) / period
    variance = sum((x - middle) ** 2 for x in recent) / period
    deviation = variance ** 0.5

    return middle + 2 * deviation, middle, middle - 2 * deviation


def calculate_adx(highs, lows, closes, period=14):
    if len(closes) < period * 2 + 1:
        return 0.0

    plus_dm = []
    minus_dm = []
    true_ranges = []

    for i in range(1, len(closes)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]

        plus_dm.append(up if up > down and up > 0 else 0)
        minus_dm.append(down if down > up and down > 0 else 0)

        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        true_ranges.append(tr)

    start = len(true_ranges) - period
    atr = sum(true_ranges[start:]) / period

    if atr <= 0:
        return 0.0

    plus = sum(plus_dm[start:]) / period
    minus = sum(minus_dm[start:]) / period

    plus_di = 100 * plus / atr
    minus_di = 100 * minus / atr

    total = plus_di + minus_di
    if total == 0:
        return 0.0

    return 100 * abs(plus_di - minus_di) / total


def get_candles(symbol, interval, limit=100):
    response = session.get_kline(
        category=CATEGORY,
        symbol=symbol,
        interval=interval,
        limit=limit
    )

    rows = response.get("result", {}).get("list", [])
    rows = list(reversed(rows))

    candles = []
    for row in rows:
        candles.append({
            "time": int(row[0]),
            "open": number(row[1]),
            "high": number(row[2]),
            "low": number(row[3]),
            "close": number(row[4]),
            "volume": number(row[5])
        })

    return candles


def get_orderbook_imbalance(symbol):
    response = session.get_orderbook(
        category=CATEGORY,
        symbol=symbol,
        limit=50
    )

    result = response.get("result", {})
    bids = result.get("b", [])
    asks = result.get("a", [])

    bid_volume = sum(number(x[1]) for x in bids)
    ask_volume = sum(number(x[1]) for x in asks)

    total = bid_volume + ask_volume
    if total == 0:
        return 0.0

    return (bid_volume - ask_volume) / total


def get_whale_activity(symbol):
    response = session.get_public_trade_history(
        category=CATEGORY,
        symbol=symbol,
        limit=100
    )

    trades = response.get("result", {}).get("list", [])
    threshold = WHALE_THRESHOLDS.get(symbol, 50000)

    buy_value = 0.0
    sell_value = 0.0
    whale_count = 0

    for trade in trades:
        price = number(trade.get("price"))
        qty = number(trade.get("size"))
        value = price * qty

        if value >= threshold:
            whale_count += 1
            if trade.get("side") == "Buy":
                buy_value += value
            else:
                sell_value += value

    return {
        "count": whale_count,
        "buy_value": buy_value,
        "sell_value": sell_value
    
    
def get_funding_rate(symbol):
    response = session.get_funding_rate_history(
        category="linear",
        symbol=symbol,
        limit=1
    )

    rows = response.get("result", {}).get("list", [])

    if not rows:
        return 0.0

    return number(rows[0].get("fundingRate"))


    return number(rows[0].get("fundingRate"))


def get_open_interest(symbol):
    response = session.get_open_interest(
        category=CATEGORY,
        symbol=symbol,
        intervalTime="5min",
        limit=2
    )

    rows = response.get("result", {}).get("list", [])
    if len(rows) < 2:
        return 0.0

    old_value = number(rows[1].get("openInterest"))
    new_value = number(rows[0].get("openInterest"))

    if old_value == 0:
        return 0.0

    return ((new_value - old_value) / old_value) * 100


def analyze_symbol(symbol):
    candles5 = get_candles(symbol, "5", 100)
    candles15 = get_candles(symbol, "15", 100)

    if len(candles5) < 60 or len(candles15) < 60:
        raise RuntimeError("Not enough candle data")

    closes = [c["close"] for c in candles5]
    highs = [c["high"] for c in candles5]
    lows = [c["low"] for c in candles5]
    volumes = [c["volume"] for c in candles5]

    price = closes[-1]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema50 = ema(closes, 50)

    rsi = calculate_rsi(closes)
    macd = calculate_macd(closes)
    macd_previous = calculate_macd(closes[:-1])

    upper, middle, lower = calculate_bollinger(closes)
    adx = calculate_adx(highs, lows, closes)

    average_volume = sum(volumes[-21:-1]) / 20
    relative_volume = (
        volumes[-1] / average_volume if average_volume > 0 else 1
    )

    closes15 = [c["close"] for c in candles15]
    ema9_15 = ema(closes15, 9)
    ema21_15 = ema(closes15, 21)

    orderbook = get_orderbook_imbalance(symbol)
    whales = get_whale_activity(symbol)
    funding = get_funding_rate(symbol)
    oi_change = get_open_interest(symbol)

    long_score = 0
    short_score = 0

    if ema9 > ema21:
        long_score += 15
    else:
        short_score += 15

    if ema21 > ema50:
        long_score += 10
    else:
        short_score += 10

    if ema9_15 > ema21_15:
        long_score += 15
    else:
        short_score += 15

    if 50 <= rsi <= 68:
        long_score += 10
    elif 32 <= rsi < 50:
        short_score += 10
    elif rsi > 70:
        short_score -= 5
    elif rsi < 30:
        long_score -= 5

    if macd > macd_previous:
        long_score += 10
    else:
        short_score += 10

    if relative_volume > 1.5:
        if closes[-1] > closes[-2]:
            long_score += 10
        else:
            short_score += 10

    if orderbook > 0.15:
        long_score += 10
    elif orderbook < -0.15:
        short_score += 10

    if whales["buy_value"] > whales["sell_value"]:
        long_score += 10
    elif whales["sell_value"] > whales["buy_value"]:
        short_score += 10

    if oi_change > 0.5:
        if price > closes[-2]:
            long_score += 5
        else:
            short_score += 5

    if funding > 0.001:
        long_score -= 3
    elif funding < -0.001:
        short_score -= 3

    long_score = max(0, min(100, long_score))
    short_score = max(0, min(100, short_score))

    if adx < 15:
        long_score = min(long_score, 65)
        short_score = min(short_score, 65)

    if long_score >= short_score:
        direction = "LONG"
        score = long_score
    else:
        direction = "SHORT"
        score = short_score

    if score < MIN_ENTRY_SCORE:
        direction = "WAIT"

    return {
        "symbol": symbol,
        "price": price,
        "direction": direction,
        "score": int(score),
        "long_score": int(long_score),
        "short_score": int(short_score),
        "rsi": round(rsi, 2),
        "macd": round(macd, 6),
        "ema9": round(ema9, 6),
        "ema21": round(ema21, 6),
        "ema50": round(ema50, 6),
        "adx": round(adx, 2),
        "relative_volume": round(relative_volume, 2),
        "orderbook_imbalance": round(orderbook, 3),
        "whale_count": whales["count"],
        "whale_buy": round(whales["buy_value"], 2),
        "whale_sell": round(whales["sell_value"], 2),
        "funding": funding,
        "oi_change": round(oi_change, 3),
        "bollinger_upper": upper,
        "bollinger_middle": middle,
        "bollinger_lower": lower,
        "updated": now_text()
    }


def close_position(symbol, price, reason):
    with lock:
        position = state["positions"].get(symbol)
        if not position:
            return

        entry = number(position["entry"])
        qty = number(position["qty"])
        side = position["side"]

        if side == "LONG":
            pnl = (price - entry) * qty
        else:
            pnl = (entry - price) * qty

        state["balance"] += pnl
        state["realized_pnl"] += pnl
        state["daily_pnl"] += pnl

        record = {
            "symbol": symbol,
            "side": side,
            "entry": entry,
            "exit": price,
            "pnl": round(pnl, 4),
            "reason": reason,
            "closed": now_text()
        }

        state["history"].append(record)
        state["history"] = state["history"][-200:]
        del state["positions"][symbol]

    save_state()


def manage_positions(prices):
    with lock:
        positions = list(state["positions"].items())

    for symbol, position in positions:
        price = prices.get(symbol)
        if not price:
            continue

        entry = number(position["entry"])
        side = position["side"]

        if side == "LONG":
            change = (price - entry) / entry
        else:
            change = (entry - price) / entry

        if change >= TAKE_PROFIT_PCT:
            close_position(symbol, price, "TAKE PROFIT")
        elif change <= -STOP_LOSS_PCT:
            close_position(symbol, price, "STOP LOSS")


def open_position(signal):
    symbol = signal["symbol"]
    side = signal["direction"]
    price = number(signal["price"])

    with lock:
        if not state["trading_enabled"]:
            return

        if symbol in state["positions"]:
            return

        if len(state["positions"]) >= MAX_OPEN_POSITIONS:
            return

        if state["daily_pnl"] <= -DAILY_LOSS_LIMIT:
            return

        balance = number(state["balance"])

        risk_amount = balance * RISK_PER_TRADE
        risk_based_notional = risk_amount / STOP_LOSS_PCT
        max_notional = balance * MAX_POSITION_VALUE_PCT * LEVERAGE
        notional = min(risk_based_notional, max_notional)

        if notional <= 0 or price <= 0:
            return

        qty = notional / price

        state["positions"][symbol] = {
            "symbol": symbol,
            "side": side,
            "entry": price,
            "qty": qty,
            "notional": notional,
            "leverage": LEVERAGE,
            "opened": now_text()
        }

        state["history"].append({
            "symbol": symbol,
            "side": side,
            "entry": price,
            "event": "PAPER OPEN",
            "score": signal["score"],
            "opened": now_text()
        })

        state["history"] = state["history"][-200:]

    save_state()


def run_scan():
    today = datetime.now(timezone.utc).date().isoformat()

    with lock:
        if state["daily_date"] != today:
            state["daily_date"] = today
            state["daily_pnl"] = 0.0

    prices = {}
    scan_errors = []

    for symbol in SYMBOLS:
        try:
            signal = analyze_symbol(symbol)
            prices[symbol] = signal["price"]

            with lock:
                state["signals"][symbol] = signal

            if signal["direction"] in ("LONG", "SHORT"):
                open_position(signal)

        except Exception as exc:
            logging.exception("Scan failed for %s", symbol)
            scan_errors.append(symbol + ": " + str(exc))

    manage_positions(prices)

    with lock:
        state["last_scan"] = now_text()
        state["scan_count"] += 1
        state["last_error"] = "; ".join(scan_errors)

    save_state()


def engine_loop():
    logging.info("%s engine started", APP_NAME)

    while True:
        try:
            run_scan()
        except Exception as exc:
            logging.exception("Engine error")
            with lock:
                state["last_error"] = str(exc)

        time.sleep(SCAN_INTERVAL)


def start_engine():
    with lock:
        if state["engine_started"]:
            return

        state["engine_started"] = True

    thread = threading.Thread(
        target=engine_loop,
        name="CoinSentraEngine",
        daemon=True
    )
    thread.start()


# ============================================================
# DASHBOARD
# ============================================================

PAGE = r"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta charset="UTF-8">
<title>CoinSentra Trading Brain</title>
<style>
body {
    margin:0;
    padding:16px;
    background:#08111f;
    color:#edf4ff;
    font-family:Arial,sans-serif;
}
.container {max-width:1050px;margin:auto}
h1 {font-size:25px;margin-bottom:5px}
.sub {color:#91a4bd;margin-bottom:18px}
.grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(145px,1fr));
    gap:10px;
}
.card {
    background:#111e30;
    border:1px solid #263b55;
    border-radius:12px;
    padding:14px;
    margin-bottom:12px;
}
.label {color:#91a4bd;font-size:12px}
.value {font-size:23px;font-weight:bold;margin-top:7px}
button {
    border:0;
    border-radius:10px;
    padding:16px;
    font-size:16px;
    font-weight:bold;
    width:100%;
    cursor:pointer;
}
.buttons {
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:12px;
}
.stop {background:#d94343;color:white}
.start {background:#13a36e;color:white}
table {width:100%;border-collapse:collapse;font-size:13px}
th,td {padding:10px 6px;border-bottom:1px solid #263b55;text-align:left}
th {color:#91a4bd}
.green {color:#38d996}
.red {color:#ff7777}
.yellow {color:#ffd166}
.small {font-size:12px;color:#91a4bd}
.scroll {overflow-x:auto}
.status {font-weight:bold}
</style>
</head>
<body>
<div class="container">
<h1>🧠 CoinSentra Trading Brain V10</h1>
<div class="sub">Bybit public market data • Paper trading only</div>

<div class="card">
  <div class="label">REMOTE TRADING CONTROL</div>
  <div class="buttons" style="margin-top:12px">
    <button class="stop" onclick="control('stop')">C — STOP TRADING</button>
    <button class="start" onclick="control('start')">V — START TRADING</button>
  </div>
  <p id="message" class="small">Use the buttons to control new paper trades.</p>
  <p class="small">STOP prevents new entries. Existing paper positions continue to be monitored.</p>
</div>

<div class="grid">
  <div class="card"><div class="label">TRADING STATUS</div><div class="value" id="enabled">STOPPED</div></div>
  <div class="card"><div class="label">PAPER BALANCE</div><div class="value" id="balance">$100.00</div></div>
  <div class="card"><div class="label">REALIZED PNL</div><div class="value" id="pnl">$0.00</div></div>
  <div class="card"><div class="label">TODAY PNL</div><div class="value" id="daily">$0.00</div></div>
  <div class="card"><div class="label">OPEN POSITIONS</div><div class="value" id="count">0</div></div>
  <div class="card"><div class="label">SCAN COUNT</div><div class="value" id="scans">0</div></div>
</div>

<div class="card">
<h3>Market signals</h3>
<div class="scroll">
<table>
<thead><tr>
<th>Symbol</th><th>Price</th><th>Signal</th><th>Score</th>
<th>RSI</th><th>ADX</th><th>Volume</th><th>Order book</th><th>Whales</th>
</tr></thead>
<tbody id="signals"></tbody>
</table>
</div>
</div>

<div class="card">
<h3>Open paper positions</h3>
<div class="scroll">
<table>
<thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Quantity</th><th>Notional</th><th>Opened</th></tr></thead>
<tbody id="positions"></tbody>
</table>
</div>
</div>

<div class="card">
<h3>Trade history</h3>
<div class="scroll">
<table>
<thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit / Event</th><th>PnL</th><th>Reason</th></tr></thead>
<tbody id="history"></tbody>
</table>
</div>
</div>

<div class="card">
<h3>System information</h3>
<p class="small">Last scan: <span id="lastscan">Waiting</span></p>
<p class="small">Last error: <span id="error">None</span></p>
<p class="small">Leverage setting: 8x simulated | TP: 2% | SL: 2%</p>
<p class="small">Daily loss limit: $6 | Maximum open positions: 2</p>
<p class="small">This dashboard simulates trades. It does not send orders to Bybit.</p>
</div>
</div>

<script>
async function control(action) {
    const message = document.getElementById("message");
    message.textContent = "Sending command...";
    try {
        const response = await fetch("/control/" + action, {
            method: "POST"
        });
        const data = await response.json();
        message.textContent = data.message || data.error || "Command received";
        await refresh();
    } catch (e) {
        message.textContent = "Could not contact the bot.";
    }
}

document.addEventListener("keydown", function(event) {
    if (event.target.tagName === "INPUT" ||
        event.target.tagName === "TEXTAREA") return;

    if (event.key.toLowerCase() === "c") control("stop");
    if (event.key.toLowerCase() === "v") control("start");
});

function money(value) {
    return "$" + Number(value || 0).toFixed(2);
}

function safe(value) {
    return String(value ?? "").replace(/[&<>"']/g, function(c) {
        return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];
    });
}

async function refresh() {
    try {
        const response = await fetch("/status");
        const data = await response.json();

        document.getElementById("enabled").textContent =
            data.trading_enabled ? "RUNNING" : "STOPPED";
        document.getElementById("enabled").className =
            "value " + (data.trading_enabled ? "green" : "red");

        document.getElementById("balance").textContent = money(data.balance);
        document.getElementById("pnl").textContent = money(data.realized_pnl);
        document.getElementById("daily").textContent = money(data.daily_pnl);
        document.getElementById("count").textContent = data.positions.length;
        document.getElementById("scans").textContent = data.scan_count;
        document.getElementById("lastscan").textContent = safe(data.last_scan);
        document.getElementById("error").textContent = safe(data.last_error || "None");

        document.getElementById("signals").innerHTML =
            data.signals.map(s => `
            <tr>
            <td>${safe(s.symbol)}</td>
            <td>${Number(s.price).toPrecision(7)}</td>
            <td>${safe(s.direction)}</td>
            <td>${safe(s.score)}</td>
            <td>${safe(s.rsi)}</td>
            <td>${safe(s.adx)}</td>
            <td>${safe(s.relative_volume)}x</td>
            <td>${safe(s.orderbook_imbalance)}</td>
            <td>${safe(s.whale_count)}</td>
            </tr>`).join("");

        document.getElementById("positions").innerHTML =
            data.positions.map(p => `
            <tr>
            <td>${safe(p.symbol)}</td>
            <td>${safe(p.side)}</td>
            <td>${Number(p.entry).toPrecision(7)}</td>
            <td>${Number(p.qty).toFixed(5)}</td>
            <td>${money(p.notional)}</td>
            <td>${safe(p.opened)}</td>
            </tr>`).join("");

        document.getElementById("history").innerHTML =
            data.history.slice().reverse().map(h => `
            <tr>
            <td>${safe(h.symbol)}</td>
            <td>${safe(h.side || "")}</td>
            <td>${h.entry ? Number(h.entry).toPrecision(7) : "-"}</td>
            <td>${h.exit ? Number(h.exit).toPrecision(7) : safe(h.event || "-")}</td>
            <td>${h.pnl !== undefined ? money(h.pnl) : "-"}</td>
            <td>${safe(h.reason || (h.event || ""))}</td>
            </tr>`).join("");

    } catch (e) {
        document.getElementById("error").textContent = "Dashboard refresh error";
    }
}

refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>
"""


# ============================================================
# WEB ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "app": APP_NAME,
        "time": now_text()
    })


@app.route("/status")
def status():
    with lock:
        return jsonify({
            "app": APP_NAME,
            "trading_enabled": state["trading_enabled"],
            "balance": round(state["balance"], 4),
            "realized_pnl": round(state["realized_pnl"], 4),
            "daily_pnl": round(state["daily_pnl"], 4),
            "positions": list(state["positions"].values()),
            "history": state["history"][-50:],
            "signals": list(state["signals"].values()),
            "last_scan": state["last_scan"],
            "last_error": state["last_error"],
            "scan_count": state["scan_count"]
        })


@app.route("/control/stop", methods=["POST"])
def stop_trading():
    with lock:
        state["trading_enabled"] = False

    save_state()
    return jsonify({
        "ok": True,
        "message": "STOPPED: no new paper trades will be opened."
    })


@app.route("/control/start", methods=["POST"])
def start_trading():
    with lock:
        if state["daily_pnl"] <= -DAILY_LOSS_LIMIT:
            return jsonify({
                "ok": False,
                "error": "Daily loss limit reached. Trading remains stopped."
            }), 403

        state["trading_enabled"] = True

    save_state()
    return jsonify({
        "ok": True,
        "message": "STARTED: new paper trades are enabled."
    })


# ============================================================
# STARTUP
# ============================================================

load_state()
start_engine()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
