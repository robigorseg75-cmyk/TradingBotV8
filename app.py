
import os
import json
import time
import threading
import logging
from datetime import datetime, timezone, date

from flask import Flask, jsonify, render_template_string
from pybit.unified_trading import HTTP

# ============================================================
# COINSENTRA TRADING BRAIN V12
# BYBIT | RENDER | PAPER TRADING ONLY
# ============================================================

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("CoinSentra")

# ---------------- SETTINGS ----------------

SYMBOLS = ["ETHUSDT", "XRPUSDT", "SOLUSDT", "SUIUSDT"]

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

# Always paper trading. No order endpoint is used.
PAPER_TRADING = True

API_KEY = os.getenv("vA9UkYkaoBHPySdPT9", "").strip()
API_SECRET = os.getenv("8uZSZDZdn8ra8xX4ephBuI7yOkyLpGKyot7h ", "").strip()

DATA_DIR = os.getenv("DATA_DIR", "/tmp")
STATE_FILE = os.path.join(DATA_DIR, "coinsentra_v12.json")

# ---------------- CONNECTIONS ----------------

public_api = HTTP(
    testnet=False,
    timeout=15
)

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
    "version": "12.0"
}


def save_state():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        temp = STATE_FILE + ".tmp"

        with open(temp, "w") as f:
            json.dump(state, f)

        os.replace(temp, STATE_FILE)
    except Exception as e:
        log.warning("State save: %s", e)


def load_state():
    global state
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)

            with lock:
                state.update(saved)
                state["running"] = saved.get("running", True)
                state["positions"] = saved.get("positions", [])
                state["history"] = saved.get("history", [])
    except Exception as e:
        log.warning("State load: %s", e)


def reset_daily():
    today = str(date.today())

    if state["daily_date"] != today:
        state["daily_date"] = today
        state["daily_pnl"] = 0.0
        save_state()


# ---------------- MARKET DATA ----------------

def get_price(symbol):
    result = public_api.get_tickers(
        category="linear",
        symbol=symbol
    )

    items = result["result"]["list"]

    if not items:
        raise ValueError("No ticker data: " + symbol)

    return float(items[0]["lastPrice"])


def get_candles(symbol, interval, limit=200):
    result = public_api.get_kline(
        category="linear",
        symbol=symbol,
        interval=interval,
        limit=limit
    )

    rows = result["result"]["list"]

    if not rows:
        raise ValueError("No candle data: " + symbol)

    rows = list(reversed(rows))

    return [
        {
            "time": int(x[0]),
            "open": float(x[1]),
            "high": float(x[2]),
            "low": float(x[3]),
            "close": float(x[4]),
            "volume": float(x[5]),
            "turnover": float(x[6])
        }
        for x in rows
    ]


def get_orderbook(symbol):
    result = public_api.get_orderbook(
        category="linear",
        symbol=symbol,
        limit=25
    )

    data = result["result"]

    bids = data.get("b", [])
    asks = data.get("a", [])

    bid_volume = sum(float(x[1]) for x in bids)
    ask_volume = sum(float(x[1]) for x in asks)

    total = bid_volume + ask_volume

    if total <= 0:
        return 0.0

    return (bid_volume - ask_volume) / total


def get_funding(symbol):
    try:
        result = public_api.get_funding_rate_history(
            category="linear",
            symbol=symbol,
            limit=1
        )

        rows = result["result"]["list"]

        if rows:
            return float(rows[0]["fundingRate"])

    except Exception as e:
        log.warning("Funding unavailable %s: %s", symbol, e)

    return 0.0


def get_open_interest(symbol):
    try:
        result = public_api.get_open_interest(
            category="linear",
            symbol=symbol,
            intervalTime="5min",
            limit=2
        )

        rows = result["result"]["list"]

        if len(rows) >= 2:
            older = float(rows[1]["openInterest"])
            newer = float(rows[0]["openInterest"])

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

    # EMA trend
    if five["ema9"] > five["ema21"]:
        long_score += 14
    elif five["ema9"] < five["ema21"]:
        short_score += 14

    if five["ema21"] > five["ema50"]:
        long_score += 12
    elif five["ema21"] < five["ema50"]:
        short_score += 12

    # RSI
    if 52 <= five["rsi"] <= 70:
        long_score += 12
    elif 30 <= five["rsi"] < 48:
        short_score += 12

    # MACD
    if five["macd"] > 0:
        long_score += 10
    elif five["macd"] < 0:
        short_score += 10

    # Volume
    if five["volume_ratio"] >= 1.3:
        if five["ema9"] > five["ema21"]:
            long_score += 10
        elif five["ema9"] < five["ema21"]:
            short_score += 10

    # Order book
    if imbalance > 0.15:
        long_score += 12
    elif imbalance < -0.15:
        short_score += 12

    # Higher timeframe confirmation
    if fifteen["ema9"] > fifteen["ema21"]:
        long_score += 18
    elif fifteen["ema9"] < fifteen["ema21"]:
        short_score += 18

    # ADX trend strength
    if five["adx"] >= 20:
        if five["ema9"] > five["ema21"]:
            long_score += 6
        elif five["ema9"] < five["ema21"]:
            short_score += 6

    # Funding: avoid crowded direction
    if funding > 0.0005:
        short_score += 3
    elif funding < -0.0005:
        long_score += 3

    # Open interest confirmation
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
        state["private_status"] = "NOT CONFIGURED"
        return

    try:
        result = private_api.get_wallet_balance(
            accountType="UNIFIED",
            coin="USDT"
        )

        accounts = result["result"]["list"]

        if accounts:
            coins = accounts[0].get("coin", [])

            usdt = next(
                (x for x in coins if x["coin"] == "USDT"),
                None
            )

            state["account_balance"] = (
                float(usdt.get("walletBalance", 0))
                if usdt else 0.0
            )

        state["private_status"] = "CONNECTED - READ ONLY"

    except Exception as e:
        state["private_status"] = "CONNECTION FAILED"
        log.error("Private account check failed: %s", e)


# ---------------- PAPER TRADING ----------------

def open_paper_position(symbol, side, price):

    if len(state["positions"]) >= MAX_POSITIONS:
        return

    if any(p["symbol"] == symbol for p in state["positions"]):
        return

    if state["daily_pnl"] <= -DAILY_LOSS_LIMIT:
        return

    # Keep maximum stop risk near $3 per trade.
    risk = min(
        RISK_PER_TRADE,
        state["balance"] * 0.005
    )

    stop_distance = price * STOP_LOSS

    if stop_distance <= 0:
        return

    quantity = risk / stop_distance

    # Cap exposure to 10% of virtual equity.
    max_notional = state["balance"] * MAX_EQUITY_EXPOSURE
    quantity = min(quantity, max_notional / price)

    if quantity <= 0:
        return

    position = {
        "symbol": symbol,
        "side": side,
        "entry": price,
        "quantity": quantity,
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
        "opened": datetime.now(timezone.utc).isoformat(),
        "unrealized_pnl": 0.0
    }

    state["positions"].append(position)

    log.info(
        "PAPER OPEN %s %s at %s",
        side, symbol, price
    )

    save_state()


def close_paper_position(position, price, reason):

    direction = 1 if position["side"] == "LONG" else -1

    pnl = (
        (price - position["entry"])
        * position["quantity"]
        * direction
    )

    state["balance"] += pnl
    state["daily_pnl"] += pnl

    state["history"].insert(0, {
        "symbol": position["symbol"],
        "side": position["side"],
        "entry": position["entry"],
        "exit": price,
        "pnl": round(pnl, 4),
        "reason": reason,
        "closed": datetime.now(timezone.utc).isoformat()
    })

    state["history"] = state["history"][:100]

    state["positions"].remove(position)

    log.info(
        "PAPER CLOSE %s PNL %.4f %s",
        position["symbol"], pnl, reason
    )

    save_state()


def monitor_positions():

    for position in list(state["positions"]):

        try:
            price = get_price(position["symbol"])

            direction = (
                1 if position["side"] == "LONG" else -1
            )

            pnl = (
                (price - position["entry"])
                * position["quantity"]
                * direction
            )

            position["unrealized_pnl"] = round(pnl, 4)

            if position["side"] == "LONG":
                if price >= position["take_profit"]:
                    close_paper_position(
                        position, price, "TAKE PROFIT"
                    )
                elif price <= position["stop_loss"]:
                    close_paper_position(
                        position, price, "STOP LOSS"
                    )

            else:
                if price <= position["take_profit"]:
                    close_paper_position(
                        position, price, "TAKE PROFIT"
                    )
                elif price >= position["stop_loss"]:
                    close_paper_position(
                        position, price, "STOP LOSS"
                    )

        except Exception as e:
            log.warning(
                "Position monitor %s: %s",
                position["symbol"], e
            )


# ---------------- SCANNER ----------------

def scan_symbol(symbol):

    five_candles = get_candles(symbol, "5", 200)
    fifteen_candles = get_candles(symbol, CONFIRMATION_INTERVAL, 100)

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
        "macd": round(five["macd"], 6),
        "adx": round(five["adx"], 2),
        "ema9": round(five["ema9"], 6),
        "ema21": round(five["ema21"], 6),
        "ema50": round(five["ema50"], 6),
        "volume_ratio": round(five["volume_ratio"], 2),
        "orderbook_imbalance": round(imbalance, 3),
        "funding": funding,
        "oi_change": round(oi_change, 3),
        **signal,
        "updated": datetime.now(timezone.utc).isoformat()
    }

    with lock:
        state["symbols"][symbol] = data

        if state["running"] and signal["signal"] in ("LONG", "SHORT"):
            open_paper_position(
                symbol,
                signal["signal"],
                five["price"]
            )

    return data


def scanner_loop():

    log.info("CoinSentra V12 scanner started")

    while True:
        try:
            with lock:
                reset_daily()

            # Always monitor existing paper positions.
            with lock:
                monitor_positions()

            for symbol in SYMBOLS:
                try:
                    scan_symbol(symbol)
                except Exception as e:
                    log.error("Scan failed %s: %s", symbol, e)

                    with lock:
                        existing = state["symbols"].get(symbol, {})
                        existing["data_error"] = str(e)[:180]
                        state["symbols"][symbol] = existing

            with lock:
                state["last_scan"] = datetime.now(
                    timezone.utc
                ).isoformat()

                state["scan_count"] += 1
                state["last_error"] = None
                save_state()

        except Exception as e:
            log.exception("Scanner loop error")
            state["last_error"] = str(e)[:250]

        time.sleep(SCAN_SECONDS)


# ---------------- DASHBOARD ----------------

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CoinSentra Trading Brain V12</title>
<style>
body{background:#080d18;color:#e9efff;font-family:Arial;margin:0;padding:15px}
h1{color:#36d9ff;font-size:25px}
h2{color:#36d9ff;font-size:18px}
.card{background:#121d30;border:1px solid #263a58;border-radius:12px;padding:14px;margin:10px 0}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.big{font-size:23px;font-weight:bold;color:#35e0b0}
.small{font-size:12px;color:#9bb0ce}
button{padding:13px;border:0;border-radius:8px;font-weight:bold;margin:5px}
.start{background:#16b981;color:white}
.stop{background:#dc4545;color:white}
table{width:100%;border-collapse:collapse;font-size:12px}
td,th{padding:9px 4px;border-bottom:1px solid #293a50;text-align:left}
.green{color:#39df9a}
.red{color:#ff7070}
.yellow{color:#ffcf65}
.scroll{overflow-x:auto}
</style>
</head>
<body>

<h1>COINSENTRA AI</h1>
<div class="small">TRADING BRAIN V12 | BYBIT | PAPER MODE</div>

<div class="grid">
<div class="card">Virtual Balance<div class="big" id="balance">$10,000</div></div>
<div class="card">Daily PNL<div class="big" id="daily">$0</div></div>
<div class="card">Open Positions<div class="big" id="count">0</div></div>
<div class="card">Scanner<div class="big" id="scanner">STARTING</div></div>
</div>

<div class="card">
<h2>CONTROL CENTER</h2>
<button class="stop" onclick="control('stop')">STOP NEW TRADES (C)</button>
<button class="start" onclick="control('start')">START TRADING (V)</button>
<p id="control">Loading...</p>
<p class="small">Stopping prevents new entries. Existing paper positions remain monitored.</p>
</div>

<div class="card">
<h2>BYBIT CONNECTION</h2>
<div id="private">Checking...</div>
<div id="account" class="small"></div>
<div class="small">Paper mode: real orders disabled</div>
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
<div id="positions">None</div>
</div>

<div class="card">
<h2>TRADE HISTORY</h2>
<div id="history">No completed trades</div>
</div>

<div class="card small">
Version 12.0 | 5m + 15m | 8x simulated leverage<br>
Take Profit 2% | Stop Loss 2% | Daily Loss Limit $6<br>
Last scan: <span id="last">Waiting</span>
</div>

<script>
async function control(action){
 await fetch('/control/'+action,{method:'POST'});
 update();
}

document.addEventListener('keydown',function(e){
 if(e.target.tagName==='INPUT')return;
 if(e.key.toLowerCase()==='c')control('stop');
 if(e.key.toLowerCase()==='v')control('start');
});

function money(x){return '$'+Number(x||0).toFixed(2)}
function color(x){return x==='LONG'?'green':x==='SHORT'?'red':'yellow'}

async function update(){
 try{
  let r=await fetch('/api/data');
  let d=await r.json();

  document.getElementById('balance').textContent=money(d.balance);
  document.getElementById('daily').textContent=money(d.daily_pnl);
  document.getElementById('count').textContent=d.positions.length;
  document.getElementById('scanner').textContent=d.last_scan?'ACTIVE':'WAITING';
  document.getElementById('control').textContent=d.running?'NEW ENTRIES ENABLED':'STOPPED - MONITORING POSITIONS';
  document.getElementById('private').textContent=d.private_status;
  document.getElementById('account').textContent=
    d.account_balance===null?'Account balance unavailable': 'Bybit USDT wallet: '+money(d.account_balance);

  document.getElementById('last').textContent=d.last_scan||'Waiting';

  let rows='';
  let details='';

  for(let s of Object.values(d.symbols)){
   rows+=`<tr>
   <td>${s.symbol}</td>
   <td>${Number(s.price||0).toPrecision(7)}</td>
   <td class="green">${s.long_score??0}</td>
   <td class="red">${s.short_score??0}</td>
   <td class="${color(s.signal)}">${s.signal||'WAIT'}</td>
   </tr>`;

   details+=`<p><b>${s.symbol}</b> | RSI ${s.rsi??'-'} | ADX ${s.adx??'-'} | Volume ${s.volume_ratio??'-'}x | Orderbook ${s.orderbook_imbalance??'-'} | OI ${s.oi_change??'-'}% ${s.data_error?' | ERROR: '+s.data_error:''}</p>`;
  }

  document.getElementById('markets').innerHTML=rows||'<tr><td colspan="5">Waiting for first market scan</td></tr>';
  document.getElementById('indicators').innerHTML=details||'Waiting for market data';

  document.getElementById('positions').innerHTML=d.positions.length?
   d.positions.map(p=>`<p>${p.symbol} ${p.side} | Entry ${p.entry} | PNL ${money(p.unrealized_pnl)} | TP ${p.take_profit} | SL ${p.stop_loss}</p>`).join(''):'No open positions';

  document.getElementById('history').innerHTML=d.history.length?
   d.history.slice(0,10).map(p=>`<p>${p.symbol} ${p.side} | ${money(p.pnl)} | ${p.reason}</p>`).join(''):'No completed trades';

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
        "version": "12.0",
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
        return jsonify({
            "balance": round(state["balance"], 2),
            "daily_pnl": round(state["daily_pnl"], 2),
            "positions": state["positions"],
            "history": state["history"],
            "symbols": state["symbols"],
            "running": state["running"],
            "last_scan": state["last_scan"],
            "private_status": state["private_status"],
            "account_balance": state["account_balance"],
            "last_error": state["last_error"]
        })


@app.route("/control/stop", methods=["POST"])
def stop():
    with lock:
        state["running"] = False
        save_state()
    return jsonify({"status": "stopped"})


@app.route("/control/start", methods=["POST"])
def start():
    with lock:
        state["running"] = True
        save_state()
    return jsonify({"status": "started"})


# ---------------- STARTUP ----------------

load_state()


def background_worker():
    # Private account check is read-only.
    check_private_account()

    # Refresh the private account balance periodically.
    while True:
        try:
            scanner_loop()
        except Exception:
            log.exception("Background worker stopped unexpectedly")
            time.sleep(10)


worker_started = False
worker_lock = threading.Lock()


@app.before_request
def start_worker_once():
    global worker_started

    with worker_lock:
        if not worker_started:
            thread = threading.Thread(
                target=background_worker,
                daemon=True
            )
            thread.start()
            worker_started = True


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
