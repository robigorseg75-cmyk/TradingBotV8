import os
import json
import time
import threading
import logging
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template_string
from pybit.unified_trading import HTTP

# ============================================================
# COINSENTRA TRADING BRAIN V11 - RENDER EDITION
# PAPER TRADING ONLY. NO REAL BYBIT ORDERS.
# ============================================================

APP_NAME = "CoinSentra Trading Brain V11"

SYMBOLS = ["ETHUSDT", "XRPUSDT", "SOLUSDT", "SUIUSDT"]
CATEGORY = "linear"

LEVERAGE = 8                  # Simulation setting only
TAKE_PROFIT_PCT = 0.02
STOP_LOSS_PCT = 0.02
DAILY_LOSS_LIMIT = 6.0
STARTING_BALANCE = 100.0
RISK_PER_TRADE = 0.005
MAX_POSITION_VALUE_PCT = 0.10
MAX_OPEN_POSITIONS = 2
MIN_ENTRY_SCORE = 70
SCAN_INTERVAL = 60

WHALE_THRESHOLDS = {
    "ETHUSDT": 100000,
    "XRPUSDT": 50000,
    "SOLUSDT": 50000,
    "SUIUSDT": 25000,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = Flask(__name__)

# Public market data only. No API keys and no order-placement methods.
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
    "scan_count": 0,
}

# Render persistent disk is optional. If mounted at /var/data, it is used.
DATA_DIR = os.environ.get("DATA_DIR", "/var/data")
if os.path.isdir(DATA_DIR) and os.access(DATA_DIR, os.W_OK):
    STATE_FILE = os.path.join(DATA_DIR, "coinsentra_v11_state.json")
else:
    STATE_FILE = os.environ.get("STATE_FILE", "coinsentra_v11_state.json")


def now_text():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def save_state():
    try:
        with lock:
            saved = {
                "balance": state["balance"],
                "realized_pnl": state["realized_pnl"],
                "daily_pnl": state["daily_pnl"],
                "daily_date": state["daily_date"],
                "positions": state["positions"],
                "history": state["history"][-200:],
                "signals": state["signals"],
                "last_scan": state["last_scan"],
                "scan_count": state["scan_count"],
            }

        folder = os.path.dirname(STATE_FILE)
        if folder:
            os.makedirs(folder, exist_ok=True)

        temp_path = STATE_FILE + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(saved, file, indent=2)
        os.replace(temp_path, STATE_FILE)

    except Exception:
        logging.exception("Could not save bot state")


def load_state():
    try:
        if not os.path.exists(STATE_FILE):
            return

        with open(STATE_FILE, "r", encoding="utf-8") as file:
            saved = json.load(file)

        with lock:
            for key in (
                "balance", "realized_pnl", "daily_pnl", "daily_date",
                "positions", "history", "signals", "last_scan", "scan_count"
            ):
                if key in saved:
                    state[key] = saved[key]

            # Safety: after a Render restart, user must press START again.
            state["trading_enabled"] = False

        logging.info("Previous paper-trading state loaded")

    except Exception:
        logging.exception("Could not load saved state")


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

    changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]

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
    variance = sum((value - middle) ** 2 for value in recent) / period
    deviation = variance ** 0.5
    return middle + 2 * deviation, middle, middle - 2 * deviation


def calculate_adx(highs, lows, closes, period=14):
    if len(closes) < period * 2 + 1:
        return 0.0

    plus_dm, minus_dm, true_ranges = [], [], []
    for i in range(1, len(closes)):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        true_ranges.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        ))

    start = len(true_ranges) - period
    atr = sum(true_ranges[start:]) / period
    if atr <= 0:
        return 0.0

    plus_di = 100.0 * (sum(plus_dm[start:]) / period) / atr
    minus_di = 100.0 * (sum(minus_dm[start:]) / period) / atr
    total = plus_di + minus_di
    if total == 0:
        return 0.0
    return 100.0 * abs(plus_di - minus_di) / total


def safe_market_metric(name, function, fallback):
    """An optional data source must never stop the whole symbol scan."""
    try:
        return function()
    except Exception as exc:
        logging.warning("%s data unavailable: %s", name, exc)
        return fallback


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
            "volume": number(row[5]),
        })
    return candles


def get_orderbook_imbalance(symbol):
    response = session.get_orderbook(
        category=CATEGORY, symbol=symbol, limit=50
    )
    result = response.get("result", {})
    bids = result.get("b", [])
    asks = result.get("a", [])
    bid_volume = sum(number(row[1]) for row in bids)
    ask_volume = sum(number(row[1]) for row in asks)
    total = bid_volume + ask_volume
    return (bid_volume - ask_volume) / total if total else 0.0


def get_whale_activity(symbol):
    response = session.get_public_trade_history(
        category=CATEGORY, symbol=symbol, limit=100
    )
    trades = response.get("result", {}).get("list", [])
    threshold = WHALE_THRESHOLDS.get(symbol, 50000)

    buy_value = 0.0
    sell_value = 0.0
    whale_count = 0

    for trade in trades:
        value = number(trade.get("price")) * number(trade.get("size"))
        if value >= threshold:
            whale_count += 1
            if trade.get("side") == "Buy":
                buy_value += value
            else:
                sell_value += value

    return {
        "count": whale_count,
        "buy_value": buy_value,
        "sell_value": sell_value,
    }


def get_funding_rate(symbol):
    # FIXED: pybit uses get_funding_rate_history(), not get_funding_rate().
    response = session.get_funding_rate_history(
        category=CATEGORY,
        symbol=symbol,
        limit=1
    )
    rows = response.get("result", {}).get("list", [])
    if not rows:
        return 0.0
    return number(rows[0].get("fundingRate"))


def get_open_interest_change(symbol):
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
    return ((new_value - old_value) / old_value) * 100.0


def analyze_symbol(symbol):
    candles5 = get_candles(symbol, "5", 100)
    candles15 = get_candles(symbol, "15", 100)

    if len(candles5) < 60 or len(candles15) < 60:
        raise RuntimeError("Bybit returned insufficient 5m/15m candle data")

    closes = [candle["close"] for candle in candles5]
    highs = [candle["high"] for candle in candles5]
    lows = [candle["low"] for candle in candles5]
    volumes = [candle["volume"] for candle in candles5]
    closes15 = [candle["close"] for candle in candles15]

    price = closes[-1]
    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema50 = ema(closes, 50)
    ema9_15 = ema(closes15, 9)
    ema21_15 = ema(closes15, 21)

    rsi = calculate_rsi(closes)
    macd_now = calculate_macd(closes)
    macd_previous = calculate_macd(closes[:-1])
    upper, middle, lower = calculate_bollinger(closes)
    adx = calculate_adx(highs, lows, closes)

    average_volume = sum(volumes[-21:-1]) / 20
    relative_volume = volumes[-1] / average_volume if average_volume > 0 else 1.0

    # Each optional endpoint is isolated. A funding/whale error won't stop scanning.
    orderbook = safe_market_metric(
        "order book", lambda: get_orderbook_imbalance(symbol), 0.0
    )
    whales = safe_market_metric(
        "whale trades", lambda: get_whale_activity(symbol),
        {"count": 0, "buy_value": 0.0, "sell_value": 0.0}
    )
    funding = safe_market_metric(
        "funding rate", lambda: get_funding_rate(symbol), 0.0
    )
    oi_change = safe_market_metric(
        "open interest", lambda: get_open_interest_change(symbol), 0.0
    )

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
        long_score -= 5
    elif rsi < 30:
        short_score -= 5

    if macd_now > macd_previous:
        long_score += 10
    else:
        short_score += 10

    if relative_volume > 1.5:
        if closes[-1] > closes[-2]:
            long_score += 10
        elif closes[-1] < closes[-2]:
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
        elif price < closes[-2]:
            short_score += 5

    if funding > 0.001:
        long_score -= 3
    elif funding < -0.001:
        short_score -= 3

    long_score = max(0, min(100, long_score))
    short_score = max(0, min(100, short_score))

    # Avoid treating weak, sideways markets as strong signals.
    if adx < 15:
        long_score = min(long_score, 65)
        short_score = min(short_score, 65)

    if long_score >= short_score:
        direction, score = "LONG", long_score
    else:
        direction, score = "SHORT", short_score

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
        "macd": round(macd_now, 8),
        "ema9": round(ema9, 8),
        "ema21": round(ema21, 8),
        "ema50": round(ema50, 8),
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
        "updated": now_text(),
    }


def close_position(symbol, price, reason):
    with lock:
        position = state["positions"].get(symbol)
        if not position:
            return

        entry = number(position["entry"])
        qty = number(position["qty"])
        side = position["side"]

        pnl = (price - entry) * qty if side == "LONG" else (entry - price) * qty

        state["balance"] += pnl
        state["realized_pnl"] += pnl
        state["daily_pnl"] += pnl

        state["history"].append({
            "symbol": symbol,
            "side": side,
            "entry": entry,
            "exit": price,
            "pnl": round(pnl, 4),
            "reason": reason,
            "closed": now_text(),
        })
        state["history"] = state["history"][-200:]
        del state["positions"][symbol]

    save_state()


def manage_positions(prices):
    with lock:
        open_positions = list(state["positions"].items())

    for symbol, position in open_positions:
        price = prices.get(symbol)
        if price is None:
            continue

        entry = number(position["entry"])
        if entry <= 0:
            continue

        if position["side"] == "LONG":
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
            state["trading_enabled"] = False
            return

        balance = number(state["balance"])
        risk_amount = balance * RISK_PER_TRADE
        risk_based_notional = risk_amount / STOP_LOSS_PCT
        max_notional = balance * MAX_POSITION_VALUE_PCT * LEVERAGE
        notional = min(risk_based_notional, max_notional)

        if price <= 0 or notional <= 0:
            return

        qty = notional / price
        state["positions"][symbol] = {
            "symbol": symbol,
            "side": side,
            "entry": price,
            "qty": qty,
            "notional": notional,
            "leverage": LEVERAGE,
            "opened": now_text(),
        }
        state["history"].append({
            "symbol": symbol,
            "side": side,
            "entry": price,
            "event": "PAPER OPEN",
            "score": signal["score"],
            "opened": now_text(),
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
    errors = []

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
            errors.append(f"{symbol}: {type(exc).__name__}: {exc}")

    manage_positions(prices)

    with lock:
        state["last_scan"] = now_text()
        state["scan_count"] += 1
        state["last_error"] = " | ".join(errors)

    save_state()


def engine_loop():
    logging.info("%s engine started", APP_NAME)
    while True:
        try:
            run_scan()
        except Exception as exc:
            logging.exception("Unexpected engine error")
            with lock:
                state["last_error"] = f"{type(exc).__name__}: {exc}"
        time.sleep(SCAN_INTERVAL)


def start_engine():
    with lock:
        if state["engine_started"]:
            return
        state["engine_started"] = True

    threading.Thread(
        target=engine_loop,
        name="CoinSentraEngine",
        daemon=True
    ).start()


PAGE = r"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta charset="UTF-8">
<title>CoinSentra Trading Brain V11</title>
<style>
body{margin:0;padding:16px;background:#08111f;color:#edf4ff;font-family:Arial,sans-serif}
.container{max-width:1100px;margin:auto}
h1{font-size:25px;margin-bottom:5px}
.sub,.small{color:#91a4bd}
.sub{margin-bottom:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px}
.card{background:#111e30;border:1px solid #263b55;border-radius:12px;padding:14px;margin-bottom:12px}
.label{color:#91a4bd;font-size:12px}
.value{font-size:23px;font-weight:bold;margin-top:7px}
.buttons{display:grid;grid-template-columns:1fr 1fr;gap:12px}
button{border:0;border-radius:10px;padding:16px;font-size:16px;font-weight:bold;width:100%;cursor:pointer}
.stop{background:#d94343;color:white}.start{background:#13a36e;color:white}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:10px 6px;border-bottom:1px solid #263b55;text-align:left}
th{color:#91a4bd}.green{color:#38d996}.red{color:#ff7777}
.scroll{overflow-x:auto}
</style>
</head>
<body>
<div class="container">
<h1>🧠 CoinSentra Trading Brain V11</h1>
<div class="sub">Bybit public market data • Paper trading only</div>

<div class="card">
<div class="label">REMOTE TRADING CONTROL</div>
<div class="buttons" style="margin-top:12px">
<button class="stop" onclick="control('stop')">C — STOP TRADING</button>
<button class="start" onclick="control('start')">V — START TRADING</button>
</div>
<p id="message" class="small">Use buttons to control new simulated trades.</p>
<p class="small">STOP blocks new entries. Existing paper positions continue to be monitored.</p>
</div>

<div class="grid">
<div class="card"><div class="label">TRADING STATUS</div><div class="value" id="enabled">STOPPED</div></div>
<div class="card"><div class="label">PAPER BALANCE</div><div class="value" id="balance">$100.00</div></div>
<div class="card"><div class="label">REALIZED PNL</div><div class="value" id="pnl">$0.00</div></div>
<div class="card"><div class="label">TODAY PNL</div><div class="value" id="daily">$0.00</div></div>
<div class="card"><div class="label">OPEN POSITIONS</div><div class="value" id="count">0</div></div>
<div class="card"><div class="label">SCAN COUNT</div><div class="value" id="scans">0</div></div>
</div>

<div class="card"><h3>Market signals</h3><div class="scroll">
<table><thead><tr><th>Symbol</th><th>Price</th><th>Signal</th><th>Score</th><th>RSI</th><th>ADX</th><th>Volume</th><th>Order book</th><th>Whales</th></tr></thead>
<tbody id="signals"></tbody></table></div></div>

<div class="card"><h3>Open paper positions</h3><div class="scroll">
<table><thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Quantity</th><th>Notional</th><th>Opened</th></tr></thead>
<tbody id="positions"></tbody></table></div></div>

<div class="card"><h3>Trade history</h3><div class="scroll">
<table><thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit / Event</th><th>PnL</th><th>Reason</th></tr></thead>
<tbody id="history"></tbody></table></div></div>

<div class="card">
<h3>System information</h3>
<p class="small">Last scan: <span id="lastscan">Waiting</span></p>
<p class="small">Last error: <span id="error">None</span></p>
<p class="small">Simulated leverage: 8x | Take profit: 2% | Stop loss: 2%</p>
<p class="small">Daily loss limit: $6 | Maximum open positions: 2</p>
<p class="small">No real Bybit orders are sent by this application.</p>
</div>
</div>

<script>
async function control(action){
 const message=document.getElementById("message");
 message.textContent="Sending command...";
 try{
  const response=await fetch("/control/"+action,{method:"POST"});
  const data=await response.json();
  message.textContent=data.message||data.error||"Command received";
  await refresh();
 }catch(e){message.textContent="Could not contact the bot.";}
}
document.addEventListener("keydown",function(event){
 if(event.target.tagName==="INPUT"||event.target.tagName==="TEXTAREA")return;
 if(event.key.toLowerCase()==="c")control("stop");
 if(event.key.toLowerCase()==="v")control("start");
});
function money(v){return "$"+Number(v||0).toFixed(2);}
function safe(v){return String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
async function refresh(){
 try{
  const response=await fetch("/status");
  const d=await response.json();
  const enabled=document.getElementById("enabled");
  enabled.textContent=d.trading_enabled?"RUNNING":"STOPPED";
  enabled.className="value "+(d.trading_enabled?"green":"red");
  document.getElementById("balance").textContent=money(d.balance);
  document.getElementById("pnl").textContent=money(d.realized_pnl);
  document.getElementById("daily").textContent=money(d.daily_pnl);
  document.getElementById("count").textContent=d.positions.length;
  document.getElementById("scans").textContent=d.scan_count;
  document.getElementById("lastscan").textContent=safe(d.last_scan);
  document.getElementById("error").textContent=safe(d.last_error||"None");

  document.getElementById("signals").innerHTML=d.signals.map(s=>`
   <tr><td>${safe(s.symbol)}</td><td>${Number(s.price).toPrecision(7)}</td>
   <td>${safe(s.direction)}</td><td>${safe(s.score)}</td><td>${safe(s.rsi)}</td>
   <td>${safe(s.adx)}</td><td>${safe(s.relative_volume)}x</td>
   <td>${safe(s.orderbook_imbalance)}</td><td>${safe(s.whale_count)}</td></tr>`).join("");

  document.getElementById("positions").innerHTML=d.positions.map(p=>`
   <tr><td>${safe(p.symbol)}</td><td>${safe(p.side)}</td>
   <td>${Number(p.entry).toPrecision(7)}</td><td>${Number(p.qty).toFixed(5)}</td>
   <td>${money(p.notional)}</td><td>${safe(p.opened)}</td></tr>`).join("");

  document.getElementById("history").innerHTML=d.history.slice().reverse().map(h=>`
   <tr><td>${safe(h.symbol)}</td><td>${safe(h.side||"")}</td>
   <td>${h.entry?Number(h.entry).toPrecision(7):"-"}</td>
   <td>${h.exit?Number(h.exit).toPrecision(7):safe(h.event||"-")}</td>
   <td>${h.pnl!==undefined?money(h.pnl):"-"}</td>
   <td>${safe(h.reason||h.event||"")}</td></tr>`).join("");
 }catch(e){document.getElementById("error").textContent="Dashboard refresh error";}
}
refresh();
setInterval(refresh,15000);
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/health")
def health():
    return jsonify({"ok": True, "app": APP_NAME, "time": now_text()})


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
            "scan_count": state["scan_count"],
        })


@app.route("/control/stop", methods=["POST"])
def stop_trading():
    with lock:
        state["trading_enabled"] = False
    save_state()
    return jsonify({
        "ok": True,
        "message": "STOPPED: new paper trades are disabled."
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


# Load state and start exactly one engine thread in this Gunicorn worker.
load_state()
start_engine()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
