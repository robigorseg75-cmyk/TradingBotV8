
import os
import json
import time
import threading
import logging
from datetime import datetime, timezone, date

import requests
from flask import Flask, jsonify, request, render_template_string

# =========================================================
# COINSENTRA TRADING BRAIN V11 - RENDER
# PAPER TRADING - BYBIT PUBLIC MARKET DATA
# =========================================================

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("CoinSentra")

SYMBOLS = ["ETHUSDT", "XRPUSDT", "SOLUSDT", "SUIUSDT"]

LEVERAGE = 8
TAKE_PROFIT = 0.02
STOP_LOSS = 0.02
DAILY_LOSS_LIMIT = 6.0
START_BALANCE = 100.0
MAX_POSITIONS = 2
SCAN_INTERVAL = 30
MIN_SCORE = 65
MAX_MARGIN = 10.0

BASE_URL = "https://api.bybit.com"
DATA_DIR = "/var/data"
if not os.path.isdir(DATA_DIR):
    DATA_DIR = "."

STATE_FILE = os.path.join(DATA_DIR, "coinsentra_v11.json")

lock = threading.RLock()
session = requests.Session()
session.headers.update({"User-Agent": "CoinSentraTradingBrain/11.0"})

state = {
    "balance": START_BALANCE,
    "realized_pnl": 0.0,
    "daily_pnl": 0.0,
    "daily_date": str(date.today()),
    "running": True,
    "positions": [],
    "history": [],
    "signals": {},
    "prices": {},
    "last_scan": "Waiting for first scan",
    "last_error": "",
    "scan_count": 0
}


# =========================================================
# STORAGE
# =========================================================

def save_state():
    try:
        with lock:
            data = dict(state)
            with open(STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)
    except Exception as e:
        log.error("Save error: %s", e)


def load_state():
    global state
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
            with lock:
                for key in state:
                    if key in saved:
                        state[key] = saved[key]
    except Exception as e:
        log.error("Load error: %s", e)


def reset_daily_if_needed():
    today = str(date.today())
    if state["daily_date"] != today:
        state["daily_date"] = today
        state["daily_pnl"] = 0.0


# =========================================================
# BYBIT PUBLIC API
# =========================================================

def api_get(endpoint, params):
    try:
        response = session.get(
            BASE_URL + endpoint,
            params=params,
            timeout=12
        )
        response.raise_for_status()
        data = response.json()

        if data.get("retCode") != 0:
            raise Exception(data.get("retMsg", "Bybit API error"))

        return data.get("result", {})
    except Exception as e:
        log.warning("API error %s: %s", endpoint, e)
        raise


def get_klines(symbol, interval, limit=100):
    result = api_get("/v5/market/kline", {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    })

    rows = result.get("list", [])
    rows.reverse()

    candles = []
    for row in rows:
        candles.append({
            "time": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "turnover": float(row[6])
        })

    if len(candles) < 30:
        raise Exception("Not enough candle data")

    return candles


def get_orderbook(symbol):
    result = api_get("/v5/market/orderbook", {
        "category": "linear",
        "symbol": symbol,
        "limit": 50
    })

    bids = result.get("b", [])
    asks = result.get("a", [])

    bid_volume = sum(float(x[1]) for x in bids[:20])
    ask_volume = sum(float(x[1]) for x in asks[:20])

    total = bid_volume + ask_volume
    imbalance = (bid_volume - ask_volume) / total if total else 0

    return imbalance


def get_whale_activity(symbol):
    result = api_get("/v5/market/recent-trade", {
        "category": "linear",
        "symbol": symbol,
        "limit": 100
    })

    buys = 0.0
    sells = 0.0

    for trade in result.get("list", []):
        value = float(trade["price"]) * float(trade["size"])

        if value >= 10000:
            if trade["side"] == "Buy":
                buys += value
            else:
                sells += value

    total = buys + sells
    ratio = (buys - sells) / total if total else 0

    return ratio, buys, sells


def get_derivatives(symbol):
    oi_result = api_get("/v5/market/open-interest", {
        "category": "linear",
        "symbol": symbol,
        "intervalTime": "5min",
        "limit": 2
    })

    oi_list = oi_result.get("list", [])
    oi_change = 0.0

    if len(oi_list) >= 2:
        old = float(oi_list[1]["openInterest"])
        new = float(oi_list[0]["openInterest"])
        if old:
            oi_change = (new - old) / old

    funding_result = api_get("/v5/market/funding/history", {
        "category": "linear",
        "symbol": symbol,
        "limit": 1
    })

    funding_list = funding_result.get("list", [])
    funding = (
        float(funding_list[0]["fundingRate"])
        if funding_list else 0.0
    )

    return oi_change, funding


# =========================================================
# INDICATORS - PURE PYTHON
# =========================================================

def ema(values, period):
    if len(values) < period:
        return None

    multiplier = 2 / (period + 1)
    result = sum(values[:period]) / period

    for value in values[period:]:
        result = (value * multiplier) + (
            result * (1 - multiplier)
        )

    return result


def rsi(values, period=14):
    if len(values) < period + 1:
        return 50.0

    changes = [
        values[i] - values[i - 1]
        for i in range(1, len(values))
    ]

    gains = [max(x, 0) for x in changes[-period:]]
    losses = [abs(min(x, 0)) for x in changes[-period:]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd_values(values):
    if len(values) < 35:
        return 0.0, 0.0

    fast = ema(values[-35:], 12)
    slow = ema(values[-35:], 26)

    return fast - slow, 0.0


def bollinger(values, period=20):
    sample = values[-period:]
    middle = sum(sample) / period
    variance = sum((x - middle) ** 2 for x in sample) / period
    deviation = variance ** 0.5

    return middle + 2 * deviation, middle - 2 * deviation


def adx_approx(candles, period=14):
    if len(candles) < period + 2:
        return 0.0

    plus = []
    minus = []
    ranges = []

    for i in range(-period, 0):
        current = candles[i]
        previous = candles[i - 1]

        up = current["high"] - previous["high"]
        down = previous["low"] - current["low"]

        plus.append(max(up, 0) if up > down else 0)
        minus.append(max(down, 0) if down > up else 0)

        ranges.append(max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"])
        ))

    tr = sum(ranges) / period
    if tr == 0:
        return 0.0

    pdi = 100 * (sum(plus) / period) / tr
    mdi = 100 * (sum(minus) / period) / tr

    total = pdi + mdi
    return abs(pdi - mdi) / total * 100 if total else 0.0


def volume_ratio(candles):
    previous = candles[-21:-1]
    average = sum(x["volume"] for x in previous) / len(previous)

    if average == 0:
        return 1.0

    return candles[-1]["volume"] / average


# =========================================================
# AI SIGNAL ENGINE
# =========================================================

def analyze_symbol(symbol):
    candles5 = get_klines(symbol, "5", 100)
    candles15 = get_klines(symbol, "15", 100)

    closes = [x["close"] for x in candles5]
    closes15 = [x["close"] for x in candles15]

    price = closes[-1]

    rsi_value = rsi(closes)
    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema50 = ema(closes, 50)

    ema15_fast = ema(closes15, 9)
    ema15_slow = ema(closes15, 21)

    macd, _ = macd_values(closes)
    upper, lower = bollinger(closes)
    adx = adx_approx(candles5)
    vol_ratio = volume_ratio(candles5)

    order_imbalance = get_orderbook(symbol)
    whale_ratio, whale_buys, whale_sells = get_whale_activity(symbol)
    oi_change, funding = get_derivatives(symbol)

    long_score = 0
    short_score = 0

    # 5-minute trend
    if ema9 > ema21:
        long_score += 15
    else:
        short_score += 15

    if ema21 > ema50:
        long_score += 10
    else:
        short_score += 10

    # RSI
    if 50 <= rsi_value <= 68:
        long_score += 12
    elif rsi_value < 35:
        long_score += 7
    elif rsi_value > 65:
        short_score += 12
    elif rsi_value < 50:
        short_score += 8

    # MACD
    if macd > 0:
        long_score += 10
    else:
        short_score += 10

    # 15-minute confirmation
    if ema15_fast > ema15_slow:
        long_score += 15
    else:
        short_score += 15

    # Volume
    if vol_ratio > 1.2:
        if closes[-1] > closes[-2]:
            long_score += 8
        else:
            short_score += 8

    # Order book
    if order_imbalance > 0.10:
        long_score += 10
    elif order_imbalance < -0.10:
        short_score += 10

    # Whale trades
    if whale_ratio > 0.15:
        long_score += 10
    elif whale_ratio < -0.15:
        short_score += 10

    # Open interest confirmation
    if oi_change > 0.005:
        if long_score > short_score:
            long_score += 5
        elif short_score > long_score:
            short_score += 5

    # Extreme funding penalty
    if funding > 0.001:
        long_score -= 5
    elif funding < -0.001:
        short_score -= 5

    # ADX
    if adx < 12:
        long_score -= 5
        short_score -= 5

    direction = "WAIT"
    score = max(long_score, short_score)

    if long_score >= MIN_SCORE and long_score > short_score:
        direction = "LONG"
    elif short_score >= MIN_SCORE and short_score > long_score:
        direction = "SHORT"

    return {
        "symbol": symbol,
        "price": price,
        "direction": direction,
        "score": round(min(score, 100), 1),
        "long_score": round(long_score, 1),
        "short_score": round(short_score, 1),
        "rsi": round(rsi_value, 2),
        "adx": round(adx, 2),
        "volume_ratio": round(vol_ratio, 2),
        "orderbook": round(order_imbalance, 3),
        "whale_ratio": round(whale_ratio, 3),
        "whale_buys": round(whale_buys, 2),
        "whale_sells": round(whale_sells, 2),
        "oi_change": round(oi_change * 100, 3),
        "funding": funding,
        "ema9": round(ema9, 8),
        "ema21": round(ema21, 8),
        "ema50": round(ema50, 8),
        "updated": datetime.now(timezone.utc).isoformat()
    }


# =========================================================
# PAPER TRADING ENGINE
# =========================================================

def open_position(signal):
    symbol = signal["symbol"]
    direction = signal["direction"]
    price = signal["price"]

    if direction not in ("LONG", "SHORT"):
        return

    if any(p["symbol"] == symbol for p in state["positions"]):
        return

    if len(state["positions"]) >= MAX_POSITIONS:
        return

    if state["daily_pnl"] <= -DAILY_LOSS_LIMIT:
        return

    margin = min(MAX_MARGIN, max(0, state["balance"] * 0.10))
    if margin < 2:
        return

    position = {
        "id": str(int(time.time() * 1000)),
        "symbol": symbol,
        "side": direction,
        "entry": price,
        "current": price,
        "margin": margin,
        "leverage": LEVERAGE,
        "quantity": (margin * LEVERAGE) / price,
        "tp": price * (1 + TAKE_PROFIT if direction == "LONG" else 1 - TAKE_PROFIT),
        "sl": price * (1 - STOP_LOSS if direction == "LONG" else 1 + STOP_LOSS),
        "pnl": 0.0,
        "opened": datetime.now(timezone.utc).isoformat()
    }

    state["positions"].append(position)
    log.info("PAPER OPEN: %s %s at %s", direction, symbol, price)


def close_position(position, price, reason):
    if position["side"] == "LONG":
        move = (price - position["entry"]) / position["entry"]
    else:
        move = (position["entry"] - price) / position["entry"]

    pnl = position["margin"] * LEVERAGE * move

    state["balance"] += pnl
    state["realized_pnl"] += pnl
    state["daily_pnl"] += pnl

    record = dict(position)
    record["exit"] = price
    record["pnl"] = round(pnl, 4)
    record["reason"] = reason
    record["closed"] = datetime.now(timezone.utc).isoformat()

    state["history"].insert(0, record)
    state["history"] = state["history"][:100]

    state["positions"].remove(position)

    log.info("PAPER CLOSE: %s PnL %.4f %s", position["symbol"], pnl, reason)


def manage_positions():
    for position in list(state["positions"]):
        price = state["prices"].get(position["symbol"])
        if not price:
            continue

        position["current"] = price

        if position["side"] == "LONG":
            move = (price - position["entry"]) / position["entry"]
            if price >= position["tp"]:
                close_position(position, price, "TAKE PROFIT")
                continue
            if price <= position["sl"]:
                close_position(position, price, "STOP LOSS")
                continue
        else:
            move = (position["entry"] - price) / position["entry"]
            if price <= position["tp"]:
                close_position(position, price, "TAKE PROFIT")
                continue
            if price >= position["sl"]:
                close_position(position, price, "STOP LOSS")
                continue

        position["pnl"] = round(
            position["margin"] * LEVERAGE * move, 4
        )


# =========================================================
# SCANNER THREAD
# =========================================================

def scanner_loop():
    log.info("CoinSentra V11 scanner started")

    while True:
        try:
            with lock:
                reset_daily_if_needed()

                for symbol in SYMBOLS:
                    try:
                        signal = analyze_symbol(symbol)
                        state["signals"][symbol] = signal
                        state["prices"][symbol] = signal["price"]

                    except Exception as e:
                        log.warning("%s scan failed: %s", symbol, e)
                        state["signals"][symbol] = {
                            "symbol": symbol,
                            "direction": "DATA ERROR",
                            "score": 0,
                            "error": str(e)
                        }

                manage_positions()

                if state["running"] and state["daily_pnl"] > -DAILY_LOSS_LIMIT:
                    candidates = sorted(
                        state["signals"].values(),
                        key=lambda x: x.get("score", 0),
                        reverse=True
                    )

                    for signal in candidates:
                        if len(state["positions"]) >= MAX_POSITIONS:
                            break
                        if signal.get("direction") in ("LONG", "SHORT"):
                            open_position(signal)

                state["scan_count"] += 1
                state["last_scan"] = datetime.now(timezone.utc).isoformat()

                if all(
                    x.get("direction") == "DATA ERROR"
                    for x in state["signals"].values()
                ):
                    state["last_error"] = "All market data requests failed"
                else:
                    state["last_error"] = ""

                save_state()

        except Exception as e:
            log.exception("Scanner error")
            with lock:
                state["last_error"] = str(e)

        time.sleep(SCAN_INTERVAL)


# =========================================================
# WEB DASHBOARD
# =========================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoinSentra Trading Brain V11</title>
<style>
body{background:#08111e;color:#e8f0ff;font-family:Arial;margin:0;padding:15px}
h1{color:#38d9ff;font-size:24px}
h2{font-size:18px;color:#38d9ff}
.card{background:#122238;border:1px solid #263d59;border-radius:12px;padding:14px;margin:10px 0}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}
.stat{background:#192e47;padding:12px;border-radius:9px}
.value{font-size:21px;font-weight:bold;margin-top:7px}
button{border:0;border-radius:9px;padding:14px 25px;font-size:17px;font-weight:bold;margin:5px;cursor:pointer}
.start{background:#00c987;color:#061b16}
.stop{background:#ff5264;color:white}
.long{color:#00e6a0}.short{color:#ff6678}.wait{color:#ffc857}
.small{color:#a9bad0;font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{text-align:left;padding:9px 5px;border-bottom:1px solid #29405a}
.scroll{overflow-x:auto}
</style>
</head>
<body>
<h1>⚡ COINSENTRA TRADING BRAIN V11</h1>
<div class="small">BYBIT PUBLIC DATA • PAPER TRADING • RENDER</div>

<div class="card">
<h2>REMOTE CONTROL</h2>
<button class="start" onclick="control('start')">V - START</button>
<button class="stop" onclick="control('stop')">C - STOP</button>
<p id="status">Connecting...</p>
</div>

<div class="grid">
<div class="stat">Balance<div class="value" id="balance">$100</div></div>
<div class="stat">Realized PnL<div class="value" id="realized">$0</div></div>
<div class="stat">Daily PnL<div class="value" id="daily">$0</div></div>
<div class="stat">Open positions<div class="value" id="count">0 / 2</div></div>
</div>

<div class="card">
<h2>MARKET SCANNER</h2>
<div class="scroll">
<table>
<thead><tr><th>Coin</th><th>Price</th><th>Signal</th><th>AI</th><th>RSI</th></tr></thead>
<tbody id="signals"></tbody>
</table>
</div>
</div>

<div class="card">
<h2>OPEN POSITIONS</h2>
<div id="positions">No open positions</div>
</div>

<div class="card">
<h2>TRADE SETTINGS</h2>
<div class="grid">
<div>Leverage: <b>8x</b></div>
<div>TP: <b>2%</b></div>
<div>SL: <b>2%</b></div>
<div>Daily loss: <b>$6</b></div>
<div>Max positions: <b>2</b></div>
<div>Scan: <b>30 sec</b></div>
</div>
</div>

<div class="card">
<h2>TRADE HISTORY</h2>
<div id="history">No closed trades</div>
</div>

<div class="card">
<h2>SYSTEM</h2>
<div id="system" class="small">Loading...</div>
</div>

<script>
async function control(action){
  try{
    const r=await fetch('/control/'+action,{method:'POST'});
    const d=await r.json();
    refresh();
  }catch(e){alert('Connection error');}
}

async function refresh(){
 try{
  const r=await fetch('/api/status');
  const d=await r.json();

  document.getElementById('balance').textContent='$'+d.balance.toFixed(2);
  document.getElementById('realized').textContent='$'+d.realized_pnl.toFixed(2);
  document.getElementById('daily').textContent='$'+d.daily_pnl.toFixed(2);
  document.getElementById('count').textContent=d.positions.length+' / 2';
  document.getElementById('status').textContent=
    d.running?'🟢 TRADING ENABLED':'🔴 TRADING STOPPED';

  let rows='';
  for(const s of Object.values(d.signals)){
    const cls=s.direction==='LONG'?'long':s.direction==='SHORT'?'short':'wait';
    rows+=`<tr><td>${s.symbol||''}</td><td>${s.price?s.price.toPrecision(7):'-'}</td><td class="${cls}">${s.direction||'-'}</td><td>${s.score??0}</td><td>${s.rsi??'-'}</td></tr>`;
  }
  document.getElementById('signals').innerHTML=rows||'<tr><td colspan="5">Waiting for market data...</td></tr>';

  let p='';
  for(const x of d.positions){
    p+=`<div class="stat"><b>${x.symbol} - ${x.side}</b><br>Entry: ${x.entry}<br>Current: ${x.current}<br>TP: ${x.tp}<br>SL: ${x.sl}<br>Margin: $${x.margin}<br>Leverage: ${x.leverage}x<br>Unrealized PnL: $${x.pnl}</div>`;
  }
  document.getElementById('positions').innerHTML=p||'No open positions';

  let h='';
  for(const x of d.history.slice(0,10)){
    h+=`<div class="stat">${x.symbol} ${x.side} | ${x.reason}<br>Exit: ${x.exit}<br>PnL: $${x.pnl}</div>`;
  }
  document.getElementById('history').innerHTML=h||'No closed trades';

  document.getElementById('system').textContent=
    'Last scan: '+d.last_scan+
    ' | Scans: '+d.scan_count+
    ' | Error: '+(d.last_error||'None');
 }catch(e){
  document.getElementById('system').textContent='Dashboard connection error';
 }
}

document.addEventListener('keydown',e=>{
 if(e.key.toLowerCase()==='c') control('stop');
 if(e.key.toLowerCase()==='v') control('start');
});
refresh();
setInterval(refresh,5000);
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": "V11"})


@app.route("/api/status")
def api_status():
    with lock:
        return jsonify(state)


@app.route("/control/<action>", methods=["POST"])
def control(action):
    if action == "stop":
        with lock:
            state["running"] = False
            save_state()
        return jsonify({"ok": True, "running": False})

    if action == "start":
        with lock:
            reset_daily_if_needed()
            if state["daily_pnl"] <= -DAILY_LOSS_LIMIT:
                return jsonify({
                    "ok": False,
                    "message": "Daily loss limit reached"
                }), 403
            state["running"] = True
            save_state()
        return jsonify({"ok": True, "running": True})

    return jsonify({"ok": False, "error": "Unknown action"}), 400


# =========================================================
# STARTUP
# =========================================================

load_state()

scanner_thread = threading.Thread(
    target=scanner_loop,
    daemon=True,
    name="CoinSentraScanner"
)
scanner_thread.start()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
