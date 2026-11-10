import os
import json
import time
import threading
import logging
from datetime import datetime, timezone, date
from pathlib import Path

from flask import Flask, jsonify, Response
from pybit.unified_trading import HTTP

# =========================================================
# COINSENTRA TRADING BRAIN V8
# PUBLIC DASHBOARD - PAPER TRADING ONLY
# =========================================================

APP_NAME = "CoinSentra Trading Brain"
VERSION = "V8"

SYMBOLS = ["ETHUSDT", "XRPUSDT", "SOLUSDT", "SUIUSDT"]

LEVERAGE = 8
TAKE_PROFIT = 0.02
STOP_LOSS = 0.02

STARTING_BALANCE = 100.0
RISK_PER_TRADE = 0.005
MAX_POSITION_VALUE = 0.10
MAX_OPEN_POSITIONS = 2
DAILY_LOSS_LIMIT = 6.0

SCAN_INTERVAL = 60
MIN_AI_SCORE = 70

DATA_DIR = Path(os.getenv("DATA_DIR", "."))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "coinsentra_state.json"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("CoinSentra")

app = Flask(__name__)

# Public Bybit market data only. No API keys and no order functions.
exchange = HTTP(testnet=False)

lock = threading.RLock()
engine_started = False

state = {
    "balance": STARTING_BALANCE,
    "realized_pnl": 0.0,
    "daily_pnl": 0.0,
    "daily_date": str(date.today()),
    "positions": [],
    "signals": {},
    "market": {},
    "whales": {},
    "events": [],
    "last_scan": None,
    "running": False,
    "errors": []
}


# =========================================================
# STATE STORAGE
# =========================================================

def load_state():
    global state
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)

            if isinstance(saved, dict):
                for key in state:
                    if key in saved:
                        state[key] = saved[key]

            if not isinstance(state.get("positions"), list):
                state["positions"] = []

            if not isinstance(state.get("signals"), dict):
                state["signals"] = {}

    except Exception as exc:
        log.warning("Could not load saved state: %s", exc)


def save_state():
    try:
        with lock:
            temp_file = STATE_FILE.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            temp_file.replace(STATE_FILE)
    except Exception as exc:
        log.warning("Could not save state: %s", exc)


def add_event(message):
    with lock:
        state["events"].insert(0, {
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            "message": str(message)
        })
        state["events"] = state["events"][:40]


def add_error(message):
    with lock:
        state["errors"].insert(0, str(message))
        state["errors"] = state["errors"][:10]


# =========================================================
# INDICATORS
# =========================================================

def ema(values, period):
    if len(values) < period:
        return None

    multiplier = 2 / (period + 1)
    result = sum(values[:period]) / period

    for value in values[period:]:
        result = (value * multiplier) + (result * (1 - multiplier))

    return result


def rsi(values, period=14):
    if len(values) < period + 1:
        return 50.0

    gains = []
    losses = []

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger(values, period=20):
    if len(values) < period:
        return None, None, None

    sample = values[-period:]
    middle = sum(sample) / period
    variance = sum((x - middle) ** 2 for x in sample) / period
    deviation = variance ** 0.5

    return middle + 2 * deviation, middle, middle - 2 * deviation


def macd(values):
    fast = ema(values, 12)
    slow = ema(values, 26)

    if fast is None or slow is None:
        return 0.0, 0.0

    return fast - slow, 0.0


def adx_estimate(highs, lows, closes, period=14):
    if len(closes) < period + 2:
        return 0.0

    plus_moves = []
    minus_moves = []
    true_ranges = []

    for i in range(1, len(closes)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]

        plus_moves.append(max(up, 0) if up > down else 0)
        minus_moves.append(max(down, 0) if down > up else 0)

        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        true_ranges.append(tr)

    tr_avg = sum(true_ranges[-period:]) / period
    if tr_avg == 0:
        return 0.0

    plus_di = 100 * (sum(plus_moves[-period:]) / period) / tr_avg
    minus_di = 100 * (sum(minus_moves[-period:]) / period) / tr_avg

    total = plus_di + minus_di
    if total == 0:
        return 0.0

    return abs(plus_di - minus_di) / total * 100


# =========================================================
# BYBIT MARKET DATA
# =========================================================

def get_candles(symbol, interval):
    response = exchange.get_kline(
        category="linear",
        symbol=symbol,
        interval=interval,
        limit=100
    )

    rows = response.get("result", {}).get("list", [])
    rows = list(reversed(rows))

    candles = []
    for row in rows:
        candles.append({
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5])
        })

    return candles


def get_orderbook_data(symbol):
    response = exchange.get_orderbook(
        category="linear",
        symbol=symbol,
        limit=50
    )

    result = response.get("result", {})
    bids = result.get("b", [])
    asks = result.get("a", [])

    bid_volume = sum(float(x[1]) for x in bids[:20])
    ask_volume = sum(float(x[1]) for x in asks[:20])

    total = bid_volume + ask_volume
    imbalance = 0.0

    if total > 0:
        imbalance = (bid_volume - ask_volume) / total * 100

    bid_wall = max(
        [float(x[1]) for x in bids],
        default=0
    )
    ask_wall = max(
        [float(x[1]) for x in asks],
        default=0
    )

    return {
        "imbalance": round(imbalance, 2),
        "bid_volume": round(bid_volume, 2),
        "ask_volume": round(ask_volume, 2),
        "bid_wall": round(bid_wall, 2),
        "ask_wall": round(ask_wall, 2)
    }


def get_whale_activity(symbol):
    response = exchange.get_public_trade_history(
        category="linear",
        symbol=symbol,
        limit=100
    )

    trades = response.get("result", {}).get("list", [])

    buy_volume = 0.0
    sell_volume = 0.0
    large_trades = []

    for trade in trades:
        price = float(trade.get("price", 0))
        size = float(trade.get("size", 0))
        value = price * size
        side = trade.get("side", "")

        if side == "Buy":
            buy_volume += value
        else:
            sell_volume += value

        if value >= 10000:
            large_trades.append({
                "side": side,
                "value": round(value, 2)
            })

    return {
        "buy_volume": round(buy_volume, 2),
        "sell_volume": round(sell_volume, 2),
        "large_trades": large_trades[:5],
        "large_count": len(large_trades)
    }


def get_open_interest_change(symbol):
    try:
        response = exchange.get_open_interest(
            category="linear",
            symbol=symbol,
            intervalTime="5min",
            limit=2
        )

        rows = response.get("result", {}).get("list", [])

        if len(rows) >= 2:
            newest = float(rows[0]["openInterest"])
            older = float(rows[1]["openInterest"])

            if older:
                return round((newest - older) / older * 100, 2)

    except Exception:
        pass

    return 0.0


def get_funding_rate(symbol):
    try:
        response = exchange.get_tickers(
            category="linear",
            symbol=symbol
        )

        rows = response.get("result", {}).get("list", [])
        if rows:
            return float(rows[0].get("fundingRate", 0)) * 100

    except Exception:
        pass

    return 0.0


# =========================================================
# ANALYSIS AND AI SCORE
# =========================================================

def analyze_symbol(symbol):
    candles5 = get_candles(symbol, "5")
    candles15 = get_candles(symbol, "15")

    if len(candles5) < 35 or len(candles15) < 35:
        raise ValueError("Not enough candle data")

    closes5 = [x["close"] for x in candles5]
    highs5 = [x["high"] for x in candles5]
    lows5 = [x["low"] for x in candles5]
    volumes5 = [x["volume"] for x in candles5]

    closes15 = [x["close"] for x in candles15]

    price = closes5[-1]

    ema9 = ema(closes5, 9)
    ema21 = ema(closes5, 21)
    ema50_15 = ema(closes15, 50)

    rsi_value = rsi(closes5)
    macd_value, _ = macd(closes5)

    upper, middle, lower = bollinger(closes5)
    adx_value = adx_estimate(highs5, lows5, closes5)

    avg_volume = sum(volumes5[-21:-1]) / 20
    volume_ratio = volumes5[-1] / avg_volume if avg_volume else 0

    orderbook = get_orderbook_data(symbol)
    whales = get_whale_activity(symbol)
    oi_change = get_open_interest_change(symbol)
    funding = get_funding_rate(symbol)

    buy_score = 0
    sell_score = 0

    if ema9 > ema21:
        buy_score += 15
    else:
        sell_score += 15

    if price > ema50_15:
        buy_score += 15
    else:
        sell_score += 15

    if 50 <= rsi_value <= 68:
        buy_score += 12
    elif 32 <= rsi_value <= 50:
        sell_score += 12
    elif rsi_value > 72:
        sell_score += 8
    elif rsi_value < 28:
        buy_score += 8

    if macd_value > 0:
        buy_score += 10
    else:
        sell_score += 10

    if volume_ratio >= 1.5:
        if closes5[-1] > closes5[-2]:
            buy_score += 10
        else:
            sell_score += 10

    if orderbook["imbalance"] > 10:
        buy_score += 12
    elif orderbook["imbalance"] < -10:
        sell_score += 12

    if whales["buy_volume"] > whales["sell_volume"] * 1.2:
        buy_score += 8
    elif whales["sell_volume"] > whales["buy_volume"] * 1.2:
        sell_score += 8

    if adx_value >= 20:
        if ema9 > ema21:
            buy_score += 8
        else:
            sell_score += 8

    if oi_change > 0 and buy_score > sell_score:
        buy_score += 5
    elif oi_change > 0 and sell_score > buy_score:
        sell_score += 5

    buy_score = min(100, buy_score)
    sell_score = min(100, sell_score)

    if buy_score >= MIN_AI_SCORE and buy_score > sell_score:
        signal = "LONG"
        score = buy_score
    elif sell_score >= MIN_AI_SCORE and sell_score > buy_score:
        signal = "SHORT"
        score = sell_score
    else:
        signal = "WAIT"
        score = max(buy_score, sell_score)

    return {
        "symbol": symbol,
        "price": price,
        "signal": signal,
        "score": score,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "rsi": round(rsi_value, 2),
        "ema9": round(ema9, 8),
        "ema21": round(ema21, 8),
        "ema50_15m": round(ema50_15, 8),
        "macd": round(macd_value, 8),
        "adx": round(adx_value, 2),
        "volume_ratio": round(volume_ratio, 2),
        "bollinger_upper": upper,
        "bollinger_middle": middle,
        "bollinger_lower": lower,
        "orderbook": orderbook,
        "whales": whales,
        "open_interest_change": oi_change,
        "funding_rate": round(funding, 5),
        "updated": datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    }


# =========================================================
# PAPER TRADING ENGINE
# =========================================================

def current_equity():
    return float(state["balance"]) + sum(
        float(p.get("unrealized_pnl", 0))
        for p in state["positions"]
    )


def close_position(position, exit_price, reason):
    entry = float(position["entry"])
    notional = float(position["notional"])
    direction = position["side"]

    if direction == "LONG":
        move = (exit_price - entry) / entry
    else:
        move = (entry - exit_price) / entry

    gross_pnl = notional * move
    fees = notional * 0.00055 + notional * (exit_price / entry) * 0.00055
    net_pnl = gross_pnl - fees

    with lock:
        state["balance"] += net_pnl
        state["realized_pnl"] += net_pnl
        state["daily_pnl"] += net_pnl

        position["exit"] = exit_price
        position["pnl"] = round(net_pnl, 4)
        position["reason"] = reason
        position["closed_at"] = datetime.now(timezone.utc).isoformat()

        state["positions"].remove(position)

    add_event(
        f"{position['symbol']} {direction} closed: "
        f"{reason}, PnL ${net_pnl:.2f}"
    )


def manage_positions():
    with lock:
        positions = list(state["positions"])
        market = dict(state["market"])

    for position in positions:
        symbol = position["symbol"]
        info = market.get(symbol)

        if not info:
            continue

        price = float(info["price"])
        entry = float(position["entry"])
        side = position["side"]

        if side == "LONG":
            move = (price - entry) / entry
        else:
            move = (entry - price) / entry

        position["unrealized_pnl"] = round(
            float(position["notional"]) * move, 4
        )

        if move >= TAKE_PROFIT:
            close_position(position, price, "TAKE PROFIT")
        elif move <= -STOP_LOSS:
            close_position(position, price, "STOP LOSS")


def open_paper_position(info):
    symbol = info["symbol"]
    side = info["signal"]
    price = float(info["price"])

    with lock:
        if state["daily_pnl"] <= -DAILY_LOSS_LIMIT:
            return

        if len(state["positions"]) >= MAX_OPEN_POSITIONS:
            return

        if any(p["symbol"] == symbol for p in state["positions"]):
            return

        equity = current_equity()
        risk_amount = max(0, equity * RISK_PER_TRADE)

        # Position sizing based on 2% stop distance.
        notional_by_risk = risk_amount / STOP_LOSS
        notional_cap = equity * MAX_POSITION_VALUE * LEVERAGE
        notional = min(notional_by_risk, notional_cap)

        if notional <= 0:
            return

        position = {
            "id": f"{symbol}-{int(time.time())}",
            "symbol": symbol,
            "side": side,
            "entry": price,
            "notional": round(notional, 2),
            "leverage": LEVERAGE,
            "score": info["score"],
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "unrealized_pnl": 0.0
        }

        state["positions"].append(position)

    add_event(
        f"PAPER {side} opened: {symbol} at {price:.6g} "
        f"(score {info['score']})"
    )


# =========================================================
# ENGINE LOOP
# =========================================================

def scan_once():
    today = str(date.today())

    with lock:
        if state["daily_date"] != today:
            state["daily_date"] = today
            state["daily_pnl"] = 0.0

    for symbol in SYMBOLS:
        try:
            info = analyze_symbol(symbol)

            with lock:
                state["signals"][symbol] = info
                state["market"][symbol] = {
                    "price": info["price"]
                }
                state["whales"][symbol] = info["whales"]

            if info["signal"] in ("LONG", "SHORT"):
                open_paper_position(info)

        except Exception as exc:
            log.exception("Scan failed for %s", symbol)
            add_error(f"{symbol}: {str(exc)[:180]}")

    manage_positions()

    with lock:
        state["last_scan"] = datetime.now(timezone.utc).isoformat()
        state["running"] = True

    save_state()


def engine_loop():
    add_event("CoinSentra Trading Brain V8 started")

    while True:
        try:
            scan_once()
        except Exception as exc:
            log.exception("Engine error")
            add_error(str(exc)[:180])

        time.sleep(SCAN_INTERVAL)


def start_engine():
    global engine_started

    with lock:
        if engine_started:
            return
        engine_started = True

    thread = threading.Thread(
        target=engine_loop,
        name="CoinSentraEngine",
        daemon=True
    )
    thread.start()


# =========================================================
# DASHBOARD
# =========================================================

HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoinSentra Trading Brain V8</title>
<style>
:root {
  color-scheme: dark;
  --bg:#080d18; --panel:#111a2b; --line:#263550;
  --text:#edf3ff; --muted:#93a4bf; --green:#22d39a;
  --red:#ff647c; --blue:#67a9ff; --gold:#f6c86b;
}
* {box-sizing:border-box}
body {
  margin:0; background:var(--bg); color:var(--text);
  font-family:Arial,sans-serif;
}
header {
  padding:22px 16px; border-bottom:1px solid var(--line);
  background:#0d1525;
}
h1 {font-size:23px;margin:0 0 7px}
.sub {color:var(--muted);font-size:13px}
main {max-width:1200px;margin:auto;padding:15px}
.grid {
  display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
  gap:10px;margin-bottom:14px;
}
.card {
  background:var(--panel);border:1px solid var(--line);
  border-radius:13px;padding:14px;min-width:0;
}
.label {font-size:12px;color:var(--muted);margin-bottom:8px}
.value {font-size:22px;font-weight:bold;overflow-wrap:anywhere}
.section {font-size:17px;margin:22px 0 10px}
table {width:100%;border-collapse:collapse;font-size:13px}
th,td {padding:11px 8px;border-bottom:1px solid var(--line);text-align:left}
th {color:var(--muted);font-weight:normal}
.green {color:var(--green)}
.red {color:var(--red)}
.blue {color:var(--blue)}
.gold {color:var(--gold)}
.pill {
  display:inline-block;padding:5px 8px;border-radius:7px;
  background:#1c2a40;font-size:11px;font-weight:bold;
}
button {
  background:#2677d9;color:white;border:0;border-radius:8px;
  padding:11px 15px;font-weight:bold;cursor:pointer;
}
.small {font-size:12px;color:var(--muted)}
.event {padding:9px 0;border-bottom:1px solid var(--line);font-size:13px}
.two {display:grid;grid-template-columns:1fr 1fr;gap:12px}
.scroll {overflow-x:auto}
.warning {color:var(--gold);font-size:12px;line-height:1.5}
@media(max-width:700px) {
 .grid {grid-template-columns:repeat(2,minmax(0,1fr))}
 .two {grid-template-columns:1fr}
 table {min-width:700px}
}
</style>
</head>
<body>
<header>
  <h1>◈ COINSENTRA TRADING BRAIN <span class="blue">V8</span></h1>
  <div class="sub">Bybit market intelligence · Paper trading · 5m + 15m analysis</div>
</header>
<main>
  <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap;margin:14px 0">
    <div class="small" id="status">Connecting to engine...</div>
    <button onclick="refreshData()">↻ Refresh dashboard</button>
  </div>

  <div class="grid">
    <div class="card"><div class="label">Paper balance</div><div class="value" id="balance">$100.00</div></div>
    <div class="card"><div class="label">Realized PnL</div><div class="value" id="realized">$0.00</div></div>
    <div class="card"><div class="label">Today's PnL</div><div class="value" id="daily">$0.00</div></div>
    <div class="card"><div class="label">Open positions</div><div class="value" id="open">0 / 2</div></div>
  </div>

  <div class="card">
    <div class="section" style="margin-top:0">Market scanner</div>
    <div class="scroll">
    <table>
      <thead><tr>
        <th>Symbol</th><th>Price</th><th>Signal</th><th>AI score</th>
        <th>RSI</th><th>ADX</th><th>Volume</th><th>Order book</th>
      </tr></thead>
      <tbody id="scanner"><tr><td colspan="8">Waiting for market data...</td></tr></tbody>
    </table>
    </div>
  </div>

  <div class="section">Open paper positions</div>
  <div class="card scroll">
    <table>
      <thead><tr>
        <th>Symbol</th><th>Side</th><th>Entry</th><th>Notional</th>
        <th>Leverage</th><th>Unrealized PnL</th><th>AI score</th>
      </tr></thead>
      <tbody id="positions"><tr><td colspan="7">No open positions</td></tr></tbody>
    </table>
  </div>

  <div class="section">Whale activity & market intelligence</div>
  <div class="card scroll">
    <table>
      <thead><tr>
        <th>Symbol</th><th>Buy volume</th><th>Sell volume</th>
        <th>Large trades</th><th>Open interest change</th><th>Funding</th>
      </tr></thead>
      <tbody id="whales"><tr><td colspan="6">Waiting for data...</td></tr></tbody>
    </table>
  </div>

  <div class="section">Recent activity</div>
  <div class="two">
    <div class="card"><div id="events" class="small">No events yet.</div></div>
    <div class="card">
      <div class="label">Risk settings</div>
      <p class="small">Simulated leverage: <b>8×</b></p>
      <p class="small">Take profit: <b>2%</b></p>
      <p class="small">Stop loss: <b>2%</b></p>
      <p class="small">Daily realized-loss limit: <b>$6</b></p>
      <p class="small">Maximum open positions: <b>2</b></p>
      <p class="small">Scan interval: <b>60 seconds</b></p>
      <p class="warning">Paper trading only. This dashboard is public and has no login protection. Anyone with its URL can view it. Do not add API keys or private account information.</p>
    </div>
  </div>
  <div class="small" style="text-align:center;padding:22px">
    COINSENTRA AI · V8 · Public market data · <span id="updated">Not updated</span>
  </div>
</main>
<script>
function money(v) {
  return "$" + Number(v || 0).toFixed(2);
}
function cls(v) {
  return Number(v) >= 0 ? "green" : "red";
}
function price(v) {
  return Number(v || 0).toPrecision(7);
}
async function refreshData() {
  try {
    const response = await fetch("/api/status");
    const d = await response.json();

    document.getElementById("status").textContent =
      d.running ? "● Engine running · Last scan: " + (d.last_scan || "starting")
                : "Engine starting...";

    document.getElementById("balance").textContent = money(d.balance);
    document.getElementById("realized").innerHTML =
      '<span class="' + cls(d.realized_pnl) + '">' + money(d.realized_pnl) + '</span>';
    document.getElementById("daily").innerHTML =
      '<span class="' + cls(d.daily_pnl) + '">' + money(d.daily_pnl) + '</span>';
    document.getElementById("open").textContent =
      (d.positions || []).length + " / 2";

    const signals = d.signals || {};
    document.getElementById("scanner").innerHTML =
      Object.values(signals).map(x => {
        let signalClass = x.signal === "LONG" ? "green" :
                          x.signal === "SHORT" ? "red" : "gold";
        return `<tr>
          <td><b>${x.symbol}</b></td>
          <td>${price(x.price)}</td>
          <td><span class="pill ${signalClass}">${x.signal}</span></td>
          <td><b>${x.score}/100</b></td>
          <td>${x.rsi}</td><td>${x.adx}</td>
          <td>${x.volume_ratio}x</td>
          <td class="${Number(x.orderbook.imbalance)>=0?'green':'red'}">${x.orderbook.imbalance}%</td>
        </tr>`;
      }).join("") || '<tr><td colspan="8">Waiting for market data...</td></tr>';

    document.getElementById("positions").innerHTML =
      (d.positions || []).map(p => `<tr>
        <td><b>${p.symbol}</b></td><td>${p.side}</td>
        <td>${price(p.entry)}</td><td>${money(p.notional)}</td>
        <td>${p.leverage}x</td>
        <td class="${cls(p.unrealized_pnl)}">${money(p.unrealized_pnl)}</td>
        <td>${p.score}</td>
      </tr>`).join("") || '<tr><td colspan="7">No open positions</td></tr>';

    document.getElementById("whales").innerHTML =
      Object.entries(signals).map(([symbol,x]) => `<tr>
        <td><b>${symbol}</b></td>
        <td class="green">${money(x.whales.buy_volume)}</td>
        <td class="red">${money(x.whales.sell_volume)}</td>
        <td>${x.whales.large_count}</td>
        <td>${x.open_interest_change}%</td>
        <td>${x.funding_rate}%</td>
      </tr>`).join("") || '<tr><td colspan="6">Waiting for data...</td></tr>';

    document.getElementById("events").innerHTML =
      (d.events || []).map(e =>
        `<div class="event"><span class="blue">${e.time}</span> — ${e.message}</div>`
      ).join("") || "No events yet.";

    document.getElementById("updated").textContent =
      "Dashboard updated " + new Date().toLocaleTimeString();

  } catch (e) {
    document.getElementById("status").textContent =
      "Could not connect to dashboard API. Retrying...";
  }
}
refreshData();
setInterval(refreshData, 10000);
</script>
</body>
</html>
"""


# =========================================================
# WEB ROUTES
# =========================================================

@app.route("/")
def home():
    return Response(HTML, mimetype="text/html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": VERSION})


@app.route("/api/status")
def api_status():
    with lock:
        result = {
            "balance": state["balance"],
            "realized_pnl": state["realized_pnl"],
            "daily_pnl": state["daily_pnl"],
            "positions": state["positions"],
            "signals": state["signals"],
            "whales": state["whales"],
            "events": state["events"],
            "last_scan": state["last_scan"],
            "running": state["running"],
            "errors": state["errors"]
        }

    return jsonify(result)


# =========================================================
# STARTUP
# =========================================================

load_state()
start_engine()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )