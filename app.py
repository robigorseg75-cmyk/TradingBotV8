
# ============================================================
# COINSENTRA TRADING BRAIN V12
# RENDER EDITION - PAPER TRADING
# ETH / XRP / SOL / SUI
# ============================================================

import os
import json
import time
import threading
import logging
from datetime import datetime, timezone, date

import requests
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("CoinSentra")

# ---------------- CONFIGURATION ----------------

SYMBOLS = ["ETHUSDT", "XRPUSDT", "SOLUSDT", "SUIUSDT"]

LEVERAGE = 8
TP = 0.02
SL = 0.02
DAILY_LIMIT = 6.0
START_BALANCE = 100.0
MAX_POSITIONS = 2
MARGIN_PER_TRADE = 10.0

MIN_SCORE = 65
SCAN_INTERVAL = 60

BASE_URL = "https://api.bybit.com"

DATA_DIR = "/var/data"
if not os.path.isdir(DATA_DIR):
    DATA_DIR = "."

STATE_FILE = os.path.join(DATA_DIR, "csai_v12.json")

lock = threading.RLock()
http = requests.Session()
http.headers.update({"User-Agent": "CoinSentra-V12"})

state = {
    "balance": START_BALANCE,
    "realized_pnl": 0.0,
    "daily_pnl": 0.0,
    "daily_date": str(date.today()),
    "running": True,
    "positions": [],
    "history": [],
    "signals": {},
    "last_scan": "Waiting",
    "last_error": "",
    "scan_count": 0,
    "scanner_status": "Starting",
    "version": "V12"
}


# ---------------- TIME ----------------

def now():
    return datetime.now(timezone.utc).isoformat()


def reset_daily():
    today = str(date.today())
    if state["daily_date"] != today:
        state["daily_date"] = today
        state["daily_pnl"] = 0.0


# ---------------- STORAGE ----------------

def save_state():
    try:
        with lock:
            temp = STATE_FILE + ".tmp"
            with open(temp, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(temp, STATE_FILE)
    except Exception as e:
        log.error("Storage error: %s", e)


def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                saved = json.load(f)

            with lock:
                for key in state:
                    if key in saved:
                        state[key] = saved[key]

            # Paper positions are restored, but scanner
            # must refresh prices before managing them.
            state["scanner_status"] = "Restarted - refreshing data"

    except Exception as e:
        log.error("Load error: %s", e)


# ---------------- BYBIT API ----------------

def api_get(endpoint, params):
    response = http.get(
        BASE_URL + endpoint,
        params=params,
        timeout=12
    )
    response.raise_for_status()
    data = response.json()

    if data.get("retCode") != 0:
        raise RuntimeError(data.get("retMsg", "Bybit API error"))

    return data.get("result", {})


def candles(symbol, interval):
    result = api_get("/v5/market/kline", {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": 100
    })

    rows = list(reversed(result.get("list", [])))

    data = []
    for r in rows:
        data.append({
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5])
        })

    # Exclude the current incomplete candle.
    if len(data) > 1:
        data = data[:-1]

    if len(data) < 55:
        raise RuntimeError("Insufficient candle data")

    return data


def ticker_price(symbol):
    result = api_get("/v5/market/tickers", {
        "category": "linear",
        "symbol": symbol
    })
    rows = result.get("list", [])
    if not rows:
        raise RuntimeError("Ticker unavailable")
    return float(rows[0]["lastPrice"])


def orderbook_imbalance(symbol):
    result = api_get("/v5/market/orderbook", {
        "category": "linear",
        "symbol": symbol,
        "limit": 50
    })

    bids = result.get("b", [])
    asks = result.get("a", [])

    bv = sum(float(x[1]) for x in bids[:20])
    av = sum(float(x[1]) for x in asks[:20])

    total = bv + av
    return (bv - av) / total if total else 0.0


def whale_flow(symbol):
    result = api_get("/v5/market/recent-trade", {
        "category": "linear",
        "symbol": symbol,
        "limit": 100
    })

    buys = 0.0
    sells = 0.0

    for t in result.get("list", []):
        value = float(t["price"]) * float(t["size"])

        # Large trade filter - $10,000
        if value >= 10000:
            if t["side"] == "Buy":
                buys += value
            else:
                sells += value

    total = buys + sells
    ratio = (buys - sells) / total if total else 0.0

    return ratio, buys, sells


def derivatives(symbol):
    oi = api_get("/v5/market/open-interest", {
        "category": "linear",
        "symbol": symbol,
        "intervalTime": "5min",
        "limit": 2
    })

    rows = oi.get("list", [])
    oi_change = 0.0

    if len(rows) >= 2:
        old = float(rows[1]["openInterest"])
        new = float(rows[0]["openInterest"])
        if old:
            oi_change = (new - old) / old

    funding_data = api_get("/v5/market/funding/history", {
        "category": "linear",
        "symbol": symbol,
        "limit": 1
    })

    fr = funding_data.get("list", [])
    funding = float(fr[0]["fundingRate"]) if fr else 0.0

    return oi_change, funding


# ---------------- INDICATORS ----------------

def ema(values, period):
    if len(values) < period:
        return None

    multiplier = 2 / (period + 1)
    result = sum(values[:period]) / period

    for v in values[period:]:
        result = v * multiplier + result * (1 - multiplier)

    return result


def rsi(values, period=14):
    changes = [
        values[i] - values[i - 1]
        for i in range(1, len(values))
    ]

    recent = changes[-period:]
    gains = sum(max(x, 0) for x in recent) / period
    losses = sum(abs(min(x, 0)) for x in recent) / period

    if losses == 0:
        return 100.0

    return 100 - 100 / (1 + gains / losses)


def macd(values):
    fast = ema(values[-60:], 12)
    slow = ema(values[-60:], 26)
    return fast - slow


def bollinger(values):
    sample = values[-20:]
    middle = sum(sample) / 20
    variance = sum((x - middle) ** 2 for x in sample) / 20
    deviation = variance ** 0.5

    return middle + 2 * deviation, middle - 2 * deviation


def adx_approx(data):
    plus = []
    minus = []
    ranges = []

    for i in range(-14, 0):
        c = data[i]
        p = data[i - 1]

        up = c["high"] - p["high"]
        down = p["low"] - c["low"]

        plus.append(max(up, 0) if up > down else 0)
        minus.append(max(down, 0) if down > up else 0)

        ranges.append(max(
            c["high"] - c["low"],
            abs(c["high"] - p["close"]),
            abs(c["low"] - p["close"])
        ))

    tr = sum(ranges) / 14

    if tr == 0:
        return 0.0

    pdi = 100 * (sum(plus) / 14) / tr
    mdi = 100 * (sum(minus) / 14) / tr

    return 100 * abs(pdi - mdi) / (pdi + mdi) if pdi + mdi else 0


def volume_ratio(data):
    average = sum(x["volume"] for x in data[-21:-1]) / 20
    return data[-1]["volume"] / average if average else 1.0


# ---------------- SIGNAL ENGINE ----------------

def analyze(symbol):
    c5 = candles(symbol, "5")
    c15 = candles(symbol, "15")

    close5 = [x["close"] for x in c5]
    close15 = [x["close"] for x in c15]

    price = ticker_price(symbol)

    e9 = ema(close5, 9)
    e21 = ema(close5, 21)
    e50 = ema(close5, 50)

    e15_9 = ema(close15, 9)
    e15_21 = ema(close15, 21)

    rv = rsi(close5)
    mv = macd(close5)
    av = adx_approx(c5)
    vr = volume_ratio(c5)
    upper, lower = bollinger(close5)

    ob = orderbook_imbalance(symbol)
    whale, whale_buy, whale_sell = whale_flow(symbol)
    oi, funding = derivatives(symbol)

    long = 0
    short = 0
    reasons = []

    if e9 > e21:
        long += 15
    else:
        short += 15
        reasons.append("5m EMA bearish")

    if e21 > e50:
        long += 10
    else:
        short += 10
        reasons.append("EMA50 trend bearish")

    if 50 <= rv <= 68:
        long += 12
    elif rv < 35:
        long += 7
    elif rv > 65:
        short += 12
    elif rv < 50:
        short += 8

    if mv > 0:
        long += 10
    else:
        short += 10

    if e15_9 > e15_21:
        long += 15
    else:
        short += 15
        reasons.append("15m confirmation bearish")

    if vr >= 1.2:
        if close5[-1] > close5[-2]:
            long += 8
        else:
            short += 8
    else:
        reasons.append("Volume below confirmation level")

    if ob > 0.10:
        long += 10
    elif ob < -0.10:
        short += 10
    else:
        reasons.append("Order book neutral")

    if whale > 0.15:
        long += 10
    elif whale < -0.15:
        short += 10
    else:
        reasons.append("No strong large-trade direction")

    if oi > 0.005:
        if long > short:
            long += 5
        elif short > long:
            short += 5

    if funding > 0.001:
        long -= 5
    elif funding < -0.001:
        short -= 5

    if av < 12:
        long -= 5
        short -= 5
        reasons.append("Weak trend strength")

    direction = "WAIT"
    score = max(long, short)

    if long >= MIN_SCORE and long > short:
        direction = "LONG"
    elif short >= MIN_SCORE and short > long:
        direction = "SHORT"

    if direction == "WAIT":
        reasons.append("Entry score not reached or directions conflict")

    return {
        "symbol": symbol,
        "price": price,
        "direction": direction,
        "score": round(max(0, min(100, score)), 1),
        "long_score": round(long, 1),
        "short_score": round(short, 1),
        "rsi": round(rv, 2),
        "macd": round(mv, 8),
        "adx": round(av, 2),
        "volume_ratio": round(vr, 2),
        "orderbook": round(ob, 3),
        "whale_ratio": round(whale, 3),
        "whale_buy": round(whale_buy, 2),
        "whale_sell": round(whale_sell, 2),
        "oi_change": round(oi * 100, 3),
        "funding": funding,
        "bollinger_upper": upper,
        "bollinger_lower": lower,
        "reason": "; ".join(reasons) if reasons else "Conditions confirmed",
        "updated": now(),
        "error": ""
    }


# ---------------- PAPER TRADING ----------------

def open_position(signal):
    if signal["direction"] not in ("LONG", "SHORT"):
        return

    if len(state["positions"]) >= MAX_POSITIONS:
        return

    if any(p["symbol"] == signal["symbol"] for p in state["positions"]):
        return

    if state["daily_pnl"] <= -DAILY_LIMIT:
        return

    margin = min(MARGIN_PER_TRADE, state["balance"] * 0.10)

    if margin < 2:
        return

    price = signal["price"]
    side = signal["direction"]

    position = {
        "id": str(int(time.time() * 1000)),
        "symbol": signal["symbol"],
        "side": side,
        "entry": price,
        "current": price,
        "margin": margin,
        "leverage": LEVERAGE,
        "quantity": margin * LEVERAGE / price,
        "tp": price * (1 + TP if side == "LONG" else 1 - TP),
        "sl": price * (1 - SL if side == "LONG" else 1 + SL),
        "pnl": 0.0,
        "opened": now()
    }

    state["positions"].append(position)
    log.info("PAPER OPEN %s %s at %s", side, signal["symbol"], price)


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
    record.update({
        "exit": price,
        "pnl": round(pnl, 4),
        "reason": reason,
        "closed": now()
    })

    state["history"].insert(0, record)
    state["history"] = state["history"][:100]
    state["positions"].remove(position)

    log.info("PAPER CLOSE %s PnL=%s", position["symbol"], pnl)


def manage_positions():
    for p in list(state["positions"]):
        signal = state["signals"].get(p["symbol"], {})
        price = signal.get("price")

        if not price:
            continue

        p["current"] = price

        if p["side"] == "LONG":
            move = (price - p["entry"]) / p["entry"]

            if price >= p["tp"]:
                close_position(p, price, "TAKE PROFIT")
                continue

            if price <= p["sl"]:
                close_position(p, price, "STOP LOSS")
                continue

        else:
            move = (p["entry"] - price) / p["entry"]

            if price <= p["tp"]:
                close_position(p, price, "TAKE PROFIT")
                continue

            if price >= p["sl"]:
                close_position(p, price, "STOP LOSS")
                continue

        p["pnl"] = round(p["margin"] * LEVERAGE * move, 4)


# ---------------- ENTRY DIAGNOSTICS ----------------

def entry_diagnostics():
    if not state["running"]:
        return "Trading stopped by user"

    if state["daily_pnl"] <= -DAILY_LIMIT:
        return "Daily loss limit reached"

    if len(state["positions"]) >= MAX_POSITIONS:
        return "Maximum open positions reached"

    if state["balance"] < 20:
        return "Insufficient paper balance"

    return "Waiting for a confirmed signal"


# ---------------- SCANNER ----------------

def scanner_loop():
    log.info("CoinSentra V12 scanner started")

    while True:
        scan_errors = []

        with lock:
            state["scanner_status"] = "Scanning"
            reset_daily()

        for symbol in SYMBOLS:
            try:
                result = analyze(symbol)

                with lock:
                    state["signals"][symbol] = result

            except Exception as e:
                message = str(e)
                scan_errors.append(symbol + ": " + message)

                with lock:
                    previous = state["signals"].get(symbol, {})
                    state["signals"][symbol] = {
                        **previous,
                        "symbol": symbol,
                        "direction": "DATA ERROR",
                        "score": 0,
                        "reason": message,
                        "error": message,
                        "updated": now()
                    }

                log.warning("%s scan failed: %s", symbol, message)

        with lock:
            manage_positions()

            if state["running"]:
                candidates = sorted(
                    state["signals"].values(),
                    key=lambda x: x.get("score", 0),
                    reverse=True
                )

                for s in candidates:
                    if len(state["positions"]) >= MAX_POSITIONS:
                        break

                    if s.get("direction") in ("LONG", "SHORT"):
                        open_position(s)

            state["scan_count"] += 1
            state["last_scan"] = now()
            state["scanner_status"] = (
                "Partial data errors" if scan_errors else "Running normally"
            )
            state["last_error"] = " | ".join(scan_errors)
            state["entry_diagnostics"] = entry_diagnostics()

            save_state()

        time.sleep(SCAN_INTERVAL)


# ---------------- DASHBOARD ----------------

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoinSentra V12</title>
<style>
*{box-sizing:border-box}
body{margin:0;padding:15px;background:#07111e;color:#edf5ff;font-family:Arial}
h1{color:#35d9ff;font-size:23px}
h2{font-size:17px;color:#35d9ff}
.card{background:#12243a;border:1px solid #29425d;border-radius:12px;padding:14px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:9px}
.stat{background:#1a304a;border-radius:9px;padding:12px;overflow-wrap:anywhere}
.value{font-size:22px;font-weight:bold;margin-top:7px}
.small{font-size:12px;color:#a9bfd5}
button{border:0;border-radius:9px;padding:14px 23px;font-weight:bold;font-size:16px;margin:5px}
.start{background:#00d69a;color:#06251d}
.stop{background:#ff5367;color:white}
table{width:100%;border-collapse:collapse;font-size:12px}
td,th{text-align:left;padding:9px 5px;border-bottom:1px solid #2a405a}
.scroll{overflow-x:auto}
.long{color:#00e5a0}
.short{color:#ff6478}
.wait{color:#ffd16a}
.error{color:#ff7d7d}
</style>
</head>
<body>

<h1>⚡ COINSENTRA TRADING BRAIN V12</h1>
<div class="small">RENDER • BYBIT PUBLIC DATA • PAPER TRADING</div>

<div class="card">
<h2>REMOTE CONTROL</h2>
<button class="start" onclick="control('start')">V - START</button>
<button class="stop" onclick="control('stop')">C - STOP</button>
<div id="status">Connecting...</div>
<div id="diagnostics" class="small"></div>
</div>

<div class="grid">
<div class="stat">Balance<div class="value" id="balance">$100</div></div>
<div class="stat">Realized PnL<div class="value" id="realized">$0</div></div>
<div class="stat">Daily PnL<div class="value" id="daily">$0</div></div>
<div class="stat">Positions<div class="value" id="count">0/2</div></div>
</div>

<div class="card">
<h2>MARKET SCANNER</h2>
<div class="scroll">
<table>
<thead>
<tr><th>Coin</th><th>Price</th><th>Signal</th><th>AI</th><th>RSI</th></tr>
</thead>
<tbody id="signals"></tbody>
</table>
</div>
</div>

<div class="card">
<h2>AI DIAGNOSTICS</h2>
<div id="details">Waiting for first scan...</div>
</div>

<div class="card">
<h2>OPEN POSITIONS</h2>
<div id="positions">No positions</div>
</div>

<div class="card">
<h2>TRADE SETTINGS</h2>
<div class="grid">
<div class="stat">Leverage<br><b>8x</b></div>
<div class="stat">Take profit<br><b>2%</b></div>
<div class="stat">Stop loss<br><b>2%</b></div>
<div class="stat">Daily limit<br><b>$6</b></div>
<div class="stat">Max positions<br><b>2</b></div>
<div class="stat">Scan interval<br><b>60 sec</b></div>
</div>
</div>

<div class="card">
<h2>TRADE HISTORY</h2>
<div id="history">No closed trades</div>
</div>

<div class="card">
<h2>SYSTEM MONITOR</h2>
<div id="system" class="small">Loading...</div>
</div>

<script>
async function control(action){
 try{
  const r=await fetch('/control/'+action,{method:'POST'});
  const d=await r.json();
  if(!r.ok) alert(d.message||'Action rejected');
  refresh();
 }catch(e){alert('Connection error');}
}

function money(x){return '$'+Number(x||0).toFixed(2)}

async function refresh(){
 try{
  const r=await fetch('/api/status');
  const d=await r.json();

  document.getElementById('balance').textContent=money(d.balance);
  document.getElementById('realized').textContent=money(d.realized_pnl);
  document.getElementById('daily').textContent=money(d.daily_pnl);
  document.getElementById('count').textContent=d.positions.length+'/2';

  document.getElementById('status').textContent=
    d.running?'🟢 TRADING ENABLED':'🔴 TRADING STOPPED';

  document.getElementById('diagnostics').textContent=
    'Entry status: '+(d.entry_diagnostics||'Waiting');

  let rows='';
  let detail='';

  for(const s of Object.values(d.signals||{})){
   let cls=s.direction==='LONG'?'long':
           s.direction==='SHORT'?'short':
           s.direction==='DATA ERROR'?'error':'wait';

   rows+=`<tr>
    <td>${s.symbol||'-'}</td>
    <td>${s.price?Number(s.price).toPrecision(7):'-'}</td>
    <td class="${cls}">${s.direction||'-'}</td>
    <td>${s.score??0}</td>
    <td>${s.rsi??'-'}</td>
   </tr>`;

   detail+=`<div class="stat">
    <b>${s.symbol}</b> - ${s.direction}<br>
    LONG: ${s.long_score??'-'} | SHORT: ${s.short_score??'-'}<br>
    RSI: ${s.rsi??'-'} | ADX: ${s.adx??'-'}<br>
    Volume ratio: ${s.volume_ratio??'-'}<br>
    Order book: ${s.orderbook??'-'}<br>
    Whale flow: ${s.whale_ratio??'-'}<br>
    OI change: ${s.oi_change??'-'}%<br>
    Funding: ${s.funding??'-'}<br>
    <span class="${cls}">${s.reason||s.error||'Waiting'}</span>
   </div>`;
  }

  document.getElementById('signals').innerHTML=
    rows||'<tr><td colspan="5">Waiting for first scan</td></tr>';

  document.getElementById('details').innerHTML=
    detail||'Waiting for first scan';

  let positions='';
  for(const p of d.positions){
   positions+=`<div class="stat">
    <b>${p.symbol} - ${p.side}</b><br>
    Entry: ${p.entry}<br>
    Current: ${p.current}<br>
    TP: ${p.tp}<br>
    SL: ${p.sl}<br>
    Margin: ${money(p.margin)}<br>
    Leverage: ${p.leverage}x<br>
    Unrealized PnL: ${money(p.pnl)}
   </div>`;
  }
  document.getElementById('positions').innerHTML=
    positions||'No open positions';

  let history='';
  for(const p of (d.history||[]).slice(0,10)){
   history+=`<div class="stat">
    ${p.symbol} ${p.side} | ${p.reason}<br>
    Exit: ${p.exit}<br>
    PnL: ${money(p.pnl)}
   </div>`;
  }
  document.getElementById('history').innerHTML=
    history||'No closed trades';

  document.getElementById('system').textContent=
    'Version: '+d.version+
    ' | Scanner: '+d.scanner_status+
    ' | Scans: '+d.scan_count+
    ' | Last scan: '+d.last_scan+
    ' | Errors: '+(d.last_error||'None');

 }catch(e){
  document.getElementById('system').textContent=
    'Dashboard API connection error: '+e;
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


# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "version": "V12"
    })


@app.route("/api/status")
def api_status():
    with lock:
        result = dict(state)
        result["positions"] = [dict(x) for x in state["positions"]]
        result["history"] = [dict(x) for x in state["history"]]
        result["signals"] = {
            k: dict(v) for k, v in state["signals"].items()
        }
        return jsonify(result)


@app.route("/control/<action>", methods=["POST"])
def control(action):
    with lock:
        if action == "stop":
            state["running"] = False
            state["entry_diagnostics"] = "Stopped by user"
            save_state()
            return jsonify({"ok": True, "running": False})

        if action == "start":
            reset_daily()

            if state["daily_pnl"] <= -DAILY_LIMIT:
                return jsonify({
                    "ok": False,
                    "message": "Daily loss limit reached"
                }), 403

            state["running"] = True
            save_state()
            return jsonify({"ok": True, "running": True})

    return jsonify({"ok": False, "message": "Unknown action"}), 400


# ---------------- STARTUP ----------------

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
        port=int(os.environ.get("PORT", "10000"))
    )
