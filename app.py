import os
import json
import time
import threading
import logging
import math
import uuid
import hashlib
import copy
import urllib.request
import urllib.error
from datetime import datetime, timezone, date

from flask import Flask, jsonify, render_template_string
from pybit.unified_trading import HTTP

# ============================================================
# COINSENTRA TRADING BRAIN V16.8 FIXED
# BYBIT | RENDER | SUPABASE | PAPER TRADING
# RELIABLE STATE | FRESH SCANNER | EASIER ENTRIES
# CANDLESTICK PATTERN RECOGNITION
# RSI 50 DIRECTION | VOLUME 0.50x ENTRY
# FIXED TRADE LIFECYCLE | ATOMIC CLOSE | STATE RECOVERY
# ============================================================

VERSION = "16.8-FIXED"

SYMBOLS = [
    "ETHUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "SUIUSDT",
    "ADAUSDT",
    "NEARUSDT"
]

CATEGORY = "linear"
LEVERAGE = 8
MAX_POSITIONS = 3
MIN_TRADE_NOTIONAL = 60.0
DAILY_LOSS_LIMIT = 10.0
START_BALANCE = 10000.0
RISK_PER_TRADE = 3.0
MAX_EQUITY_EXPOSURE = 0.10

SCAN_SECONDS = 30
ENTRY_SCORE = 55
RSI_INTERVAL = "5"
TREND_INTERVAL = "15"

STOP_LOSS_PERCENT = 0.014
TAKE_PROFIT_PERCENT = 0.017
COOLDOWN_SECONDS = 60
FEE_RATE = 0.00055
PAPER_TRADING = True
MAX_HISTORY = 500

RSI_BULLISH_LEVEL = 50
RSI_BEARISH_LEVEL = 50
MIN_VOLUME_RATIO = 0.50

LEARNING_FEATURES = [
    "trend_fast", "trend_slow", "rsi", "macd", "volume",
    "orderbook", "confirmation", "higher_timeframe", "adx",
    "funding", "open_interest"
]

LEARNING_MIN_SAMPLES = 10
LEARNING_WINDOW = 30
LEARNING_MIN_WEIGHT = 0.75
LEARNING_MAX_WEIGHT = 1.25
LEARNING_RATE = 0.15

CANDLE_PATTERN_SCORE = 8
CANDLE_STRONG_PATTERN_SCORE = 10

# ============================================================
# ENVIRONMENT
# ============================================================

DATA_DIR = os.environ.get("DATA_DIR", "/tmp")
STATE_FILE = os.path.join(DATA_DIR, "coinsentra_v16.json")

BYBIT_API_KEY = os.environ.get("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.environ.get("BYBIT_API_SECRET", "")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

SUPABASE_ENABLED = bool(
    SUPABASE_URL.startswith("https://") and SUPABASE_KEY
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("CoinSentra")

# ============================================================
# APP
# ============================================================

app = Flask(__name__)

lock = threading.RLock()
worker_lock = threading.Lock()
save_lock = threading.Lock()
worker_started = False

# ============================================================
# BYBIT
# ============================================================

public_api = HTTP(testnet=False, timeout=15)
private_api = None

if BYBIT_API_KEY and BYBIT_API_SECRET:
    try:
        private_api = HTTP(
            testnet=False,
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET,
            timeout=15
        )
    except Exception:
        log.exception("Private Bybit initialization failed")

# ============================================================
# HELPERS
# ============================================================

def now_text():
    return datetime.now(timezone.utc).isoformat()

def number(value, default=0.0):
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except (TypeError, ValueError):
        pass
    return default

def clamp(value, low, high):
    return max(low, min(high, value))

def check_bybit_result(result):
    if not isinstance(result, dict):
        raise RuntimeError("Invalid Bybit response")

    if result.get("retCode", 0) != 0:
        raise RuntimeError(result.get("retMsg", "Bybit API error"))

    return result

def supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

def http_json(url, method="GET", headers=None, body=None, timeout=15):
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers or {},
        method=method
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")

    return json.loads(raw) if raw else {}

# ============================================================
# LEARNING STATE
# ============================================================

def default_weights():
    return {name: 1.0 for name in LEARNING_FEATURES}

def default_learning():
    return {
        "enabled": True,
        "samples": 0,
        "updates": 0,
        "last_update": None,
        "last_result": "Waiting",
        "processed_ids": [],
        "weights": default_weights()
    }

# ============================================================
# STATE
# ============================================================

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
    "last_closed_by_symbol": {},
    "version": VERSION,
    "learning": default_learning()
}

# ============================================================
# STATE SNAPSHOT
# ============================================================

def state_snapshot():
    with lock:
        return copy.deepcopy({
            "balance": state["balance"],
            "positions": state["positions"],
            "history": state["history"][:MAX_HISTORY],
            "daily_pnl": state["daily_pnl"],
            "daily_date": state["daily_date"],
            "running": state["running"],
            "last_scan": state["last_scan"],
            "last_error": state["last_error"],
            "private_status": state["private_status"],
            "account_balance": state["account_balance"],
            "symbols": state["symbols"],
            "scan_count": state["scan_count"],
            "last_closed_by_symbol": state["last_closed_by_symbol"],
            "version": VERSION,
            "learning": state["learning"]
        })

# ============================================================
# SUPABASE
# ============================================================

def supabase_save(snapshot):
    if not SUPABASE_ENABLED:
        return False

    try:
        payload = json.dumps({
            "id": "main",
            "state": snapshot,
            "updated_at": now_text()
        }).encode("utf-8")

        url = SUPABASE_URL + "/rest/v1/coinsentra_state"

        headers = supabase_headers()
        headers["Prefer"] = "resolution=merge-duplicates,return=minimal"

        request = urllib.request.Request(
            url,
            data=payload,
            headers=headers,
            method="POST"
        )

        with urllib.request.urlopen(request, timeout=20) as response:
            return 200 <= response.status < 300

    except Exception:
        log.exception("Supabase state save failed")
        return False

def supabase_load():
    if not SUPABASE_ENABLED:
        return None

    try:
        url = (
            SUPABASE_URL
            + "/rest/v1/coinsentra_state"
            + "?select=state&id=eq.main"
        )

        result = http_json(
            url,
            headers=supabase_headers(),
            timeout=20
        )

        if isinstance(result, list) and result:
            return result[0].get("state")

    except Exception:
        log.exception("Supabase state load failed")

    return None

# ============================================================
# SAVE STATE - LOCAL ATOMIC + REMOTE
# ============================================================

def save_state():
    with save_lock:
        snapshot = state_snapshot()
        local_ok = False

        try:
            os.makedirs(DATA_DIR, exist_ok=True)

            temporary = STATE_FILE + ".tmp"

            with open(temporary, "w", encoding="utf-8") as file:
                json.dump(snapshot, file, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())

            os.replace(temporary, STATE_FILE)
            local_ok = True

        except Exception:
            log.exception("Local state save failed")

        if SUPABASE_ENABLED:
            remote_ok = supabase_save(snapshot)

            if not remote_ok:
                log.error("REMOTE PERSISTENCE FAILED - LOCAL STATE KEPT")

            return local_ok

        log.warning("Supabase not configured; Render local storage is temporary")
        return local_ok

# ============================================================
# LOAD STATE
# ============================================================

def load_state():
    loaded = None

    if SUPABASE_ENABLED:
        loaded = supabase_load()

    if loaded is None:
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as file:
                loaded = json.load(file)
        except Exception:
            loaded = None

    if not isinstance(loaded, dict):
        log.warning("No recoverable state found; initializing new state")
        return

    with lock:
        state["balance"] = number(loaded.get("balance"), START_BALANCE)

        state["positions"] = [
            p for p in loaded.get("positions", []) if isinstance(p, dict)
        ]

        state["history"] = [
            t for t in loaded.get("history", []) if isinstance(t, dict)
        ][:MAX_HISTORY]

        state["daily_pnl"] = number(loaded.get("daily_pnl"))
        state["daily_date"] = loaded.get("daily_date", str(date.today()))
        state["running"] = bool(loaded.get("running", True))
        state["last_scan"] = loaded.get("last_scan")
        state["last_error"] = loaded.get("last_error")
        state["private_status"] = loaded.get("private_status", "NOT CONFIGURED")
        state["account_balance"] = loaded.get("account_balance")
        state["symbols"] = loaded.get("symbols", {})
        state["scan_count"] = int(loaded.get("scan_count", 0))
        state["last_closed_by_symbol"] = loaded.get("last_closed_by_symbol", {})
        state["version"] = VERSION

        learning = loaded.get("learning", default_learning())

        if not isinstance(learning, dict):
            learning = default_learning()

        merged_learning = default_learning()
        merged_learning.update(learning)
        old_weights = learning.get("weights", {})
        merged_learning["weights"] = default_weights()

        for key in LEARNING_FEATURES:
            merged_learning["weights"][key] = clamp(
                number(old_weights.get(key), 1.0),
                LEARNING_MIN_WEIGHT,
                LEARNING_MAX_WEIGHT
            )

        merged_learning["processed_ids"] = list(
            dict.fromkeys(merged_learning.get("processed_ids", []))
        )[-1000:]

        state["learning"] = merged_learning

        # Normalize recovered positions
        normalized = []

        for position in state["positions"]:
            position["entry"] = number(position.get("entry"))
            position["quantity"] = number(position.get("quantity"))
            position["margin"] = number(position.get("margin"))
            position["leverage"] = number(position.get("leverage"), LEVERAGE)
            position["unrealized_pnl"] = number(position.get("unrealized_pnl"))
            position["stop_loss"] = number(position.get("stop_loss"))
            position["take_profit"] = number(position.get("take_profit"))

            if position["entry"] <= 0 or position["quantity"] <= 0:
                continue

            if not position.get("trade_id"):
                position["trade_id"] = uuid.uuid4().hex

            if position["stop_loss"] <= 0:
                position["stop_loss"] = (
                    position["entry"] * (1 - STOP_LOSS_PERCENT)
                    if position.get("side") == "LONG"
                    else position["entry"] * (1 + STOP_LOSS_PERCENT)
                )

            if position["take_profit"] <= 0:
                position["take_profit"] = (
                    position["entry"] * (1 + TAKE_PROFIT_PERCENT)
                    if position.get("side") == "LONG"
                    else position["entry"] * (1 - TAKE_PROFIT_PERCENT)
                )

            normalized.append(position)

        state["positions"] = normalized

    log.info(
        "STATE RECOVERED | Positions %s | History %s | Balance %.2f",
        len(state["positions"]), len(state["history"]), state["balance"]
    )

# ============================================================
# DAILY RESET
# ============================================================

def reset_daily():
    today = str(date.today())
    changed = False

    with lock:
        if state["daily_date"] != today:
            state["daily_date"] = today
            state["daily_pnl"] = 0.0
            changed = True

    if changed:
        save_state()

# ============================================================
# LEARNING WEIGHTS
# ============================================================

def get_learning_weights():
    with lock:
        return dict(state["learning"]["weights"])

def learning_snapshot_for_side(features, side):
    direction = 1 if side == "LONG" else -1
    return {name: number(features.get(name)) * direction for name in LEARNING_FEATURES}

# ============================================================
# TRADE NORMALIZATION
# ============================================================

def clean_trade(trade):
    if not isinstance(trade, dict) or trade.get("status") != "COMPLETED":
        return None

    required = ["entry", "exit", "quantity", "pnl"]

    if any(key not in trade for key in required):
        return None

    entry = number(trade.get("entry"))
    exit_price = number(trade.get("exit"))
    quantity = number(trade.get("quantity"))

    if entry <= 0 or exit_price <= 0 or quantity <= 0:
        return None

    result = dict(trade)
    result["entry"] = entry
    result["exit"] = exit_price
    result["quantity"] = quantity
    result["pnl"] = number(trade.get("pnl"))
    result["net_pnl"] = number(trade.get("net_pnl"), result["pnl"])
    result["gross_pnl"] = number(trade.get("gross_pnl"))
    result["fees"] = number(trade.get("fees"))
    result["trade_id"] = str(trade.get("trade_id") or uuid.uuid4().hex)
    result["status"] = "COMPLETED"
    return result

# ============================================================
# ADAPTIVE LEARNING
# ============================================================

def update_learning(trade):
    trade = clean_trade(trade)

    if trade is None:
        return

    trade_id = trade["trade_id"]

    with lock:
        learning = state["learning"]

        if trade_id in learning["processed_ids"]:
            return

        learning["processed_ids"].append(trade_id)
        learning["processed_ids"] = learning["processed_ids"][-1000:]

        snapshot = trade.get("feature_snapshot", {})
        outcome = 1 if trade["pnl"] > 0 else -1

        for name in LEARNING_FEATURES:
            value = number(snapshot.get(name))

            if value == 0:
                continue

            adjustment = LEARNING_RATE * outcome * value

            learning["weights"][name] = clamp(
                learning["weights"][name] + adjustment,
                LEARNING_MIN_WEIGHT,
                LEARNING_MAX_WEIGHT
            )

        learning["samples"] += 1
        learning["updates"] += 1
        learning["last_update"] = now_text()
        learning["last_result"] = "WIN" if trade["pnl"] > 0 else "LOSS"

    save_state()

def recover_learning():
    with lock:
        trades = list(state["history"])

    for trade in reversed(trades):
        update_learning(trade)

# ============================================================
# MARKET DATA
# ============================================================

def get_price(symbol):
    result = check_bybit_result(
        public_api.get_tickers(category=CATEGORY, symbol=symbol)
    )

    data = result.get("result", {}).get("list", [])

    if not data:
        raise RuntimeError("No ticker returned")

    price = number(data[0].get("lastPrice"))

    if price <= 0:
        raise RuntimeError("Invalid price")

    return price

def get_candles(symbol, interval, limit=200):
    result = check_bybit_result(
        public_api.get_kline(
            category=CATEGORY,
            symbol=symbol,
            interval=interval,
            limit=limit
        )
    )

    raw = result.get("result", {}).get("list", [])
    candles = []

    for row in reversed(raw):
        if len(row) < 6:
            continue

        candle = {
            "time": int(row[0]),
            "open": number(row[1]),
            "high": number(row[2]),
            "low": number(row[3]),
            "close": number(row[4]),
            "volume": number(row[5])
        }

        if all(candle[k] > 0 for k in ("open", "high", "low", "close")):
            candles.append(candle)

    if len(candles) < 60:
        raise RuntimeError("Insufficient candles")

    # Exclude current incomplete candle
    return candles[:-1]

def get_orderbook(symbol):
    try:
        result = check_bybit_result(
            public_api.get_orderbook(category=CATEGORY, symbol=symbol, limit=50)
        )

        book = result.get("result", {})
        bids = book.get("b", [])
        asks = book.get("a", [])

        bid_volume = sum(number(x[1]) for x in bids)
        ask_volume = sum(number(x[1]) for x in asks)
        total = bid_volume + ask_volume

        if total <= 0:
            return 0.0

        return (bid_volume - ask_volume) / total

    except Exception:
        log.exception("Orderbook error for %s", symbol)
        return 0.0

def get_funding(symbol):
    try:
        result = check_bybit_result(
            public_api.get_tickers(category=CATEGORY, symbol=symbol)
        )

        data = result.get("result", {}).get("list", [])

        if data:
            return number(data[0].get("fundingRate"))

    except Exception:
        log.exception("Funding error")

    return 0.0

def get_open_interest(symbol):
    try:
        result = check_bybit_result(
            public_api.get_open_interest(
                category=CATEGORY,
                symbol=symbol,
                intervalTime="5min",
                limit=6
            )
        )

        values = [
            number(x.get("openInterest"))
            for x in result.get("result", {}).get("list", [])
        ]

        if len(values) >= 2 and values[-2] > 0:
            return (values[-1] - values[-2]) / values[-2] * 100

    except Exception:
        log.exception("Open interest error")

    return 0.0

# ============================================================
# INDICATORS
# ============================================================

def ema(values, period):
    if not values:
        return 0.0

    k = 2 / (period + 1)
    result = values[0]

    for value in values[1:]:
        result = value * k + result * (1 - k)

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

    if avg_loss <= 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)

def macd(values):
    if len(values) < 26:
        return 0.0

    return ema(values, 12) - ema(values, 26)

def bollinger(values, period=20, multiplier=2):
    subset = values[-period:]

    if not subset:
        return 0.0, 0.0, 0.0

    middle = sum(subset) / len(subset)
    variance = sum((x - middle) ** 2 for x in subset) / len(subset)
    deviation = math.sqrt(variance)

    return middle + multiplier * deviation, middle, middle - multiplier * deviation

def atr(candles, period=14):
    if len(candles) < period + 1:
        return 0.0

    ranges = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        tr = max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"])
        )

        ranges.append(tr)

    return sum(ranges[-period:]) / period

def adx_approx(candles, period=14):
    if len(candles) < period + 2:
        return 0.0

    plus_values = []
    minus_values = []
    tr_values = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        up_move = current["high"] - previous["high"]
        down_move = previous["low"] - current["low"]

        plus_values.append(
            up_move if up_move > down_move and up_move > 0 else 0
        )
        minus_values.append(
            down_move if down_move > up_move and down_move > 0 else 0
        )

        tr_values.append(max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"])
        ))

    tr_avg = sum(tr_values[-period:]) / period

    if tr_avg <= 0:
        return 0.0

    plus_di = sum(plus_values[-period:]) / period / tr_avg * 100
    minus_di = sum(minus_values[-period:]) / period / tr_avg * 100
    total = plus_di + minus_di

    if total <= 0:
        return 0.0

    return abs(plus_di - minus_di) / total * 100

# ============================================================
# CANDLESTICK PATTERN RECOGNITION
# ============================================================

def candle_parts(candle):
    open_price = candle["open"]
    close_price = candle["close"]
    high = candle["high"]
    low = candle["low"]

    body = abs(close_price - open_price)
    candle_range = max(high - low, 1e-12)
    upper_wick = high - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low

    return {
        "body": body,
        "range": candle_range,
        "upper": max(0, upper_wick),
        "lower": max(0, lower_wick),
        "bullish": close_price > open_price,
        "bearish": close_price < open_price,
        "body_ratio": body / candle_range,
        "upper_ratio": max(0, upper_wick) / candle_range,
        "lower_ratio": max(0, lower_wick) / candle_range
    }

def detect_candlestick_patterns(candles):
    result = {
        "bullish": [],
        "bearish": [],
        "neutral": [],
        "score": 0,
        "signal": "NEUTRAL",
        "display": "No pattern"
    }

    if len(candles) < 5:
        return result

    c1, c2, c3 = candles[-1], candles[-2], candles[-3]
    p1, p2, p3 = candle_parts(c1), candle_parts(c2), candle_parts(c3)
    body1, body2, body3 = p1["body"], p2["body"], p3["body"]

    if (
        p2["bearish"] and p1["bullish"]
        and c1["open"] <= c2["close"]
        and c1["close"] >= c2["open"]
        and body1 >= body2 * 1.05
    ):
        result["bullish"].append("Bullish Engulfing")

    if (
        p1["lower"] >= body1 * 2
        and p1["upper"] <= body1 * 0.6
        and p1["body_ratio"] <= 0.45
        and p1["lower_ratio"] >= 0.50
    ):
        result["bullish"].append("Hammer")

    if (
        p3["bearish"] and p3["body_ratio"] > 0.5
        and p2["body_ratio"] < 0.35
        and p1["bullish"]
        and c1["close"] > (c3["open"] + c3["close"]) / 2
    ):
        result["bullish"].append("Morning Star")

    if len(candles) >= 4:
        b, c, d = candles[-3], candles[-2], candles[-1]
        pb, pc, pd = candle_parts(b), candle_parts(c), candle_parts(d)

        if (
            pb["bullish"] and pc["bullish"] and pd["bullish"]
            and c["close"] > b["close"] and d["close"] > c["close"]
            and pb["body_ratio"] > 0.45
            and pc["body_ratio"] > 0.45
            and pd["body_ratio"] > 0.45
        ):
            result["bullish"].append("Three White Soldiers")

    if (
        p2["bullish"] and p1["bearish"]
        and c1["open"] >= c2["close"]
        and c1["close"] <= c2["open"]
        and body1 >= body2 * 1.05
    ):
        result["bearish"].append("Bearish Engulfing")

    if (
        p1["upper"] >= body1 * 2
        and p1["lower"] <= body1 * 0.6
        and p1["body_ratio"] <= 0.45
        and p1["upper_ratio"] >= 0.50
    ):
        result["bearish"].append("Shooting Star")

    if (
        p3["bullish"] and p3["body_ratio"] > 0.5
        and p2["body_ratio"] < 0.35
        and p1["bearish"]
        and c1["close"] < (c3["open"] + c3["close"]) / 2
    ):
        result["bearish"].append("Evening Star")

    if len(candles) >= 4:
        b, c, d = candles[-3], candles[-2], candles[-1]
        pb, pc, pd = candle_parts(b), candle_parts(c), candle_parts(d)

        if (
            pb["bearish"] and pc["bearish"] and pd["bearish"]
            and c["close"] < b["close"] and d["close"] < c["close"]
            and pb["body_ratio"] > 0.45
            and pc["body_ratio"] > 0.45
            and pd["body_ratio"] > 0.45
        ):
            result["bearish"].append("Three Black Crows")

    if p1["body_ratio"] <= 0.10:
        result["neutral"].append("Doji")

    if p1["lower_ratio"] >= 0.55:
        result["neutral"].append("Long Lower Wick")

    if p1["upper_ratio"] >= 0.55:
        result["neutral"].append("Long Upper Wick")

    bullish_count = len(result["bullish"])
    bearish_count = len(result["bearish"])

    result["score"] = bullish_count * CANDLE_PATTERN_SCORE - bearish_count * CANDLE_PATTERN_SCORE

    if "Three White Soldiers" in result["bullish"]:
        result["score"] += CANDLE_STRONG_PATTERN_SCORE

    if "Three Black Crows" in result["bearish"]:
        result["score"] -= CANDLE_STRONG_PATTERN_SCORE

    if result["score"] > 0:
        result["signal"] = "BULLISH"
    elif result["score"] < 0:
        result["signal"] = "BEARISH"

    all_patterns = result["bullish"] + result["bearish"] + result["neutral"]
    result["display"] = " | ".join(all_patterns) if all_patterns else "No pattern"
    return result

# ============================================================
# CANDLE ANALYSIS
# ============================================================

def analyze_candles(candles):
    closes = [x["close"] for x in candles]
    volumes = [x["volume"] for x in candles]

    if len(closes) < 60:
        raise ValueError("Insufficient candles")

    e9 = ema(closes, 9)
    e21 = ema(closes, 21)
    e50 = ema(closes, 50)

    upper, middle, lower = bollinger(closes)
    previous_volumes = volumes[-21:-1]
    average_volume = sum(previous_volumes) / len(previous_volumes) if previous_volumes else 0

    volume_ratio = volumes[-1] / average_volume if average_volume > 0 else 0.0

    return {
        "price": closes[-1],
        "ema9": e9,
        "ema21": e21,
        "ema50": e50,
        "rsi": rsi(closes),
        "macd": macd(closes),
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "volume_ratio": volume_ratio,
        "adx": adx_approx(candles),
        "atr": atr(candles)
    }

# ============================================================
# SIGNAL ENGINE
# ============================================================

def calculate_signal(five, fifteen, hourly, imbalance, funding, oi_change, candle_patterns=None):
    weights = get_learning_weights()

    candle_patterns = candle_patterns or {
        "bullish": [], "bearish": [], "neutral": [],
        "score": 0, "signal": "NEUTRAL", "display": "No pattern"
    }

    long_score = 0.0
    short_score = 0.0
    features = {name: 0.0 for name in LEARNING_FEATURES}

    if fifteen["ema9"] > fifteen["ema21"]:
        features["trend_fast"] = 1
        long_score += 14 * weights["trend_fast"]
    elif fifteen["ema9"] < fifteen["ema21"]:
        features["trend_fast"] = -1
        short_score += 14 * weights["trend_fast"]

    if fifteen["ema21"] > fifteen["ema50"]:
        features["trend_slow"] = 1
        long_score += 12 * weights["trend_slow"]
    elif fifteen["ema21"] < fifteen["ema50"]:
        features["trend_slow"] = -1
        short_score += 12 * weights["trend_slow"]

    if 50 < five["rsi"] <= 70:
        features["rsi"] = 1
        long_score += 8 * weights["rsi"]
    elif 30 <= five["rsi"] < 50:
        features["rsi"] = -1
        short_score += 8 * weights["rsi"]

    if fifteen["macd"] > 0:
        features["macd"] = 1
        long_score += 8 * weights["macd"]
    elif fifteen["macd"] < 0:
        features["macd"] = -1
        short_score += 8 * weights["macd"]

    if fifteen["volume_ratio"] >= MIN_VOLUME_RATIO:
        if fifteen["ema9"] > fifteen["ema21"]:
            features["volume"] = 1
            long_score += 8 * weights["volume"]
        elif fifteen["ema9"] < fifteen["ema21"]:
            features["volume"] = -1
            short_score += 8 * weights["volume"]

    if imbalance > 0.10:
        features["orderbook"] = 1
        long_score += 5 * weights["orderbook"]
    elif imbalance < -0.10:
        features["orderbook"] = -1
        short_score += 5 * weights["orderbook"]

    if fifteen["ema9"] > fifteen["ema21"]:
        features["confirmation"] = 1
        long_score += 15 * weights["confirmation"]
    elif fifteen["ema9"] < fifteen["ema21"]:
        features["confirmation"] = -1
        short_score += 15 * weights["confirmation"]

    if fifteen["ema9"] > fifteen["ema21"] and fifteen["ema21"] > fifteen["ema50"]:
        features["higher_timeframe"] = 1
        long_score += 20 * weights["higher_timeframe"]
    elif fifteen["ema9"] < fifteen["ema21"] and fifteen["ema21"] < fifteen["ema50"]:
        features["higher_timeframe"] = -1
        short_score += 20 * weights["higher_timeframe"]

    if fifteen["adx"] >= 16:
        if fifteen["ema9"] > fifteen["ema21"]:
            features["adx"] = 1
            long_score += 6 * weights["adx"]
        elif fifteen["ema9"] < fifteen["ema21"]:
            features["adx"] = -1
            short_score += 6 * weights["adx"]

    if funding > 0.0005:
        features["funding"] = -1
        short_score += 2 * weights["funding"]
    elif funding < -0.0005:
        features["funding"] = 1
        long_score += 2 * weights["funding"]

    if oi_change > 0.2:
        if fifteen["ema9"] > fifteen["ema21"]:
            features["open_interest"] = 1
            long_score += 2 * weights["open_interest"]
        elif fifteen["ema9"] < fifteen["ema21"]:
            features["open_interest"] = -1
            short_score += 2 * weights["open_interest"]

    candle_score = number(candle_patterns.get("score"))

    if candle_score > 0:
        long_score += candle_score
    elif candle_score < 0:
        short_score += abs(candle_score)

    long_score = min(100, round(long_score))
    short_score = min(100, round(short_score))

    signal = "WAIT"

    long_confirmed = (
        five["rsi"] > 50 and five["rsi"] <= 70
        and fifteen["ema9"] > fifteen["ema21"]
        and fifteen["ema21"] > fifteen["ema50"]
        and fifteen["adx"] >= 16
        and fifteen["volume_ratio"] >= MIN_VOLUME_RATIO
    )

    short_confirmed = (
        five["rsi"] < 50 and five["rsi"] >= 30
        and fifteen["ema9"] < fifteen["ema21"]
        and fifteen["ema21"] < fifteen["ema50"]
        and fifteen["adx"] >= 16
        and fifteen["volume_ratio"] >= MIN_VOLUME_RATIO
    )

    if (
        long_score >= ENTRY_SCORE
        and long_score - short_score >= 12
        and long_confirmed
        and fifteen["price"] > fifteen["ema21"]
    ):
        signal = "LONG"
    elif (
        short_score >= ENTRY_SCORE
        and short_score - long_score >= 12
        and short_confirmed
        and fifteen["price"] < fifteen["ema21"]
    ):
        signal = "SHORT"

    return {
        "long_score": long_score,
        "short_score": short_score,
        "signal": signal,
        "features": features,
        "candle_pattern": candle_patterns.get("display", "No pattern"),
        "candle_signal": candle_patterns.get("signal", "NEUTRAL"),
        "candle_score": candle_score,
        "bullish_patterns": candle_patterns.get("bullish", []),
        "bearish_patterns": candle_patterns.get("bearish", []),
        "neutral_patterns": candle_patterns.get("neutral", [])
    }

# ============================================================
# ACCOUNTING
# ============================================================

def position_gross_pnl(position, price):
    entry = number(position.get("entry"))
    quantity = number(position.get("quantity"))
    direction = 1 if position.get("side") == "LONG" else -1
    return (price - entry) * quantity * direction

def position_fees(position, price):
    entry = number(position.get("entry"))
    quantity = number(position.get("quantity"))
    return entry * quantity * FEE_RATE + price * quantity * FEE_RATE

def position_net_pnl(position, price):
    return position_gross_pnl(position, price) - position_fees(position, price)

# ============================================================
# ACCOUNT TOTALS
# ============================================================

def account_totals():
    with lock:
        balance = number(state["balance"])
        positions = copy.deepcopy(state["positions"])

    unrealized = sum(number(p.get("unrealized_pnl")) for p in positions)
    used_margin = sum(number(p.get("margin")) for p in positions)

    return {
        "balance": round(balance, 4),
        "unrealized": round(unrealized, 4),
        "equity": round(balance + unrealized, 4),
        "used_margin": round(used_margin, 4),
        "available_balance": round(balance - used_margin, 4)
    }

# ============================================================
# COOLDOWN
# ============================================================

def symbol_in_cooldown(symbol):
    with lock:
        closed_at = number(state["last_closed_by_symbol"].get(symbol))

    return closed_at > 0 and time.time() - closed_at < COOLDOWN_SECONDS

# ============================================================
# OPEN PAPER POSITION
# ============================================================

def open_paper_position(symbol, side, price, features=None, atr_value=0.0):
    if not PAPER_TRADING or side not in ("LONG", "SHORT") or price <= 0:
        return False

    if symbol_in_cooldown(symbol):
        return False

    with lock:
        if not state["running"] or len(state["positions"]) >= MAX_POSITIONS:
            return False

        if any(p.get("symbol") == symbol for p in state["positions"]):
            return False

        if state["daily_pnl"] <= -DAILY_LOSS_LIMIT:
            state["running"] = False
            return False

        balance = number(state["balance"])
        if balance <= 0:
            return False

        used_margin = sum(number(p.get("margin")) for p in state["positions"])
        available_balance = max(0.0, balance - used_margin)

        if available_balance <= 0:
            return False

        stop_distance = price * STOP_LOSS_PERCENT
        risk_amount = min(RISK_PER_TRADE, balance * 0.005)

        quantity = risk_amount / stop_distance
        minimum_quantity = MIN_TRADE_NOTIONAL / price
        quantity = max(quantity, minimum_quantity)

        max_notional = balance * MAX_EQUITY_EXPOSURE
        affordable_notional = available_balance * LEVERAGE
        max_notional = min(max_notional, affordable_notional)

        if max_notional < MIN_TRADE_NOTIONAL:
            return False

        quantity = min(quantity, max_notional / price)
        if quantity <= 0:
            return False

        notional = quantity * price
        if notional < MIN_TRADE_NOTIONAL:
            return False

        margin = notional / LEVERAGE
        if margin > available_balance:
            return False

        if side == "LONG":
            stop_price = price * (1 - STOP_LOSS_PERCENT)
            target_price = price * (1 + TAKE_PROFIT_PERCENT)
        else:
            stop_price = price * (1 + STOP_LOSS_PERCENT)
            target_price = price * (1 - TAKE_PROFIT_PERCENT)

        position = {
            "trade_id": uuid.uuid4().hex,
            "symbol": symbol,
            "side": side,
            "entry": price,
            "current_price": price,
            "quantity": quantity,
            "notional": notional,
            "margin": margin,
            "leverage": LEVERAGE,
            "take_profit": target_price,
            "stop_loss": stop_price,
            "stop_distance": stop_distance,
            "risk_amount": quantity * stop_distance,
            "reward_risk": TAKE_PROFIT_PERCENT / STOP_LOSS_PERCENT,
            "opened": now_text(),
            "unrealized_pnl": 0.0,
            "gross_unrealized_pnl": 0.0,
            "entry_fee": notional * FEE_RATE,
            "fee_rate": FEE_RATE,
            "feature_snapshot": learning_snapshot_for_side(features or {}, side)
        }

        state["positions"].append(position)

    # Save failure must not delete the live in-memory paper position.
    save_state()

    log.info(
        "PAPER OPEN | %s | %s | Entry %.8f | SL %.8f | TP %.8f | Notional %.2f | Margin %.2f",
        side, symbol, price, stop_price, target_price, notional, margin
    )

    return True

# ============================================================
# COMPLETED TRADE ENGINE - ATOMIC / IDEMPOTENT
# ============================================================

def close_paper_position(position, price, reason):
    if not PAPER_TRADING or not isinstance(position, dict):
        return False

    trade_id = position.get("trade_id")

    if not trade_id or price <= 0:
        return False

    with lock:
        live = next(
            (p for p in state["positions"] if p.get("trade_id") == trade_id),
            None
        )

        if live is None:
            return False

        if any(t.get("trade_id") == trade_id for t in state["history"]):
            state["positions"].remove(live)
            return False

        entry = number(live.get("entry"))
        quantity = number(live.get("quantity"))

        if entry <= 0 or quantity <= 0:
            return False

        gross = position_gross_pnl(live, price)
        fees = position_fees(live, price)
        net = gross - fees

        trade = {
            "trade_id": trade_id,
            "status": "COMPLETED",
            "symbol": live.get("symbol", "UNKNOWN"),
            "side": live.get("side", "UNKNOWN"),
            "entry": entry,
            "exit": price,
            "quantity": quantity,
            "notional": number(live.get("notional"), entry * quantity),
            "leverage": number(live.get("leverage"), LEVERAGE),
            "margin": number(live.get("margin")),
            "gross_pnl": round(gross, 6),
            "fees": round(fees, 6),
            "pnl": round(net, 6),
            "net_pnl": round(net, 6),
            "reason": reason,
            "opened": live.get("opened", ""),
            "closed": now_text(),
            "feature_snapshot": dict(live.get("feature_snapshot", {}))
        }

        # Commit trade to in-memory state exactly once.
        state["history"].insert(0, trade)
        state["history"] = state["history"][:MAX_HISTORY]
        state["balance"] += net
        state["daily_pnl"] += net
        state["last_closed_by_symbol"][live.get("symbol")] = time.time()
        state["positions"].remove(live)

        if state["daily_pnl"] <= -DAILY_LOSS_LIMIT:
            state["running"] = False

    # Persistence failure does not reverse completed trade accounting.
    save_state()

    try:
        update_learning(trade)
    except Exception:
        log.exception("Learning update failed")

    log.info(
        "PAPER CLOSE | %s | %s | Exit %.8f | Gross %.6f | Fees %.6f | Net %.6f | %s",
        trade["side"], trade["symbol"], price, gross, fees, net, reason
    )

    return True

# ============================================================
# POSITION MONITOR
# ============================================================

def monitor_positions():
    with lock:
        positions = copy.deepcopy(state["positions"])

    changed = False

    for position in positions:
        try:
            entry = number(position.get("entry"))
            quantity = number(position.get("quantity"))
            side = position.get("side")

            if entry <= 0 or quantity <= 0 or side not in ("LONG", "SHORT"):
                continue

            stop_price = number(position.get("stop_loss"))
            target_price = number(position.get("take_profit"))

            if stop_price <= 0 or target_price <= 0:
                continue

            price = get_price(position["symbol"])
            net = position_net_pnl(position, price)
            gross = position_gross_pnl(position, price)

            with lock:
                for live in state["positions"]:
                    if live.get("trade_id") == position.get("trade_id"):
                        live["current_price"] = price
                        live["unrealized_pnl"] = round(net, 6)
                        live["gross_unrealized_pnl"] = round(gross, 6)
                        live.pop("state_error", None)
                        changed = True
                        break

            if side == "LONG":
                if price <= stop_price:
                    close_paper_position(position, price, "STOP LOSS")
                elif price >= target_price:
                    close_paper_position(position, price, "TAKE PROFIT")
            else:
                if price >= stop_price:
                    close_paper_position(position, price, "STOP LOSS")
                elif price <= target_price:
                    close_paper_position(position, price, "TAKE PROFIT")

        except Exception:
            log.exception("Position monitor failed for %s", position.get("symbol"))

    if changed:
        save_state()

# ============================================================
# DAILY RISK
# ============================================================

def risk_check():
    with lock:
        realized = number(state["daily_pnl"])
        unrealized = sum(number(p.get("unrealized_pnl")) for p in state["positions"])
        total = realized + unrealized

        if total <= -DAILY_LOSS_LIMIT:
            state["running"] = False
            log.warning("DAILY LOSS LIMIT REACHED | %.4f", total)
            return False

    return True

# ============================================================
# PRIVATE ACCOUNT - READ ONLY
# ============================================================

def check_private_account():
    if private_api is None:
        with lock:
            state["private_status"] = "NOT CONFIGURED"
        return

    try:
        result = check_bybit_result(
            private_api.get_wallet_balance(
                accountType="UNIFIED",
                coin="USDT"
            )
        )

        accounts = result.get("result", {}).get("list", [])
        wallet = None

        if accounts:
            wallet = next(
                (item for item in accounts[0].get("coin", []) if item.get("coin") == "USDT"),
                None
            )

        with lock:
            state["account_balance"] = number(wallet.get("walletBalance")) if wallet else 0.0
            state["private_status"] = "CONNECTED - READ ONLY"

    except Exception:
        with lock:
            state["private_status"] = "CONNECTION FAILED"
        log.exception("Private account check failed")

    save_state()

# ============================================================
# SCANNER
# ============================================================

def scan_symbol(symbol):
    five_candles = get_candles(symbol, RSI_INTERVAL, 200)
    five = analyze_candles(five_candles)

    fifteen_candles = get_candles(symbol, TREND_INTERVAL, 200)
    fifteen = analyze_candles(fifteen_candles)

    fifteen["price"] = get_price(symbol)
    candle_patterns = detect_candlestick_patterns(five_candles)

    imbalance = get_orderbook(symbol)
    funding = get_funding(symbol)
    oi_change = get_open_interest(symbol)

    signal = calculate_signal(
        five, fifteen, fifteen, imbalance, funding, oi_change, candle_patterns
    )

    trend = (
        "BULLISH"
        if fifteen["ema9"] > fifteen["ema21"] and fifteen["ema21"] > fifteen["ema50"]
        else "BEARISH"
        if fifteen["ema9"] < fifteen["ema21"] and fifteen["ema21"] < fifteen["ema50"]
        else "MIXED"
    )

    data = {
        "symbol": symbol,
        "price": fifteen["price"],
        "rsi": round(five["rsi"], 2),
        "rsi_5m": round(five["rsi"], 2),
        "rsi_15m": round(fifteen["rsi"], 2),
        "macd": round(fifteen["macd"], 8),
        "adx": round(fifteen["adx"], 2),
        "atr": round(fifteen["atr"], 8),
        "ema9": round(fifteen["ema9"], 8),
        "ema21": round(fifteen["ema21"], 8),
        "ema50": round(fifteen["ema50"], 8),
        "volume_ratio": round(fifteen["volume_ratio"], 2),
        "orderbook_imbalance": round(imbalance, 3),
        "funding": funding,
        "oi_change": round(oi_change, 3),
        "hourly_trend": trend,
        **signal,
        "updated": now_text(),
        "data_error": None
    }

    with lock:
        state["symbols"][symbol] = data
        should_open = state["running"] and signal["signal"] in ("LONG", "SHORT")

    if should_open:
        open_paper_position(
            symbol, signal["signal"], fifteen["price"],
            signal["features"], fifteen["atr"]
        )

    return data

# ============================================================
# BACKGROUND WORKER
# ============================================================

def background_worker():
    log.info("CoinSentra V16.8 worker started")
    last_private_check = 0.0

    while True:
        cycle_start = time.time()

        try:
            reset_daily()
            monitor_positions()

            for symbol in SYMBOLS:
                try:
                    scan_symbol(symbol)
                except Exception as error:
                    log.exception("Scan failed for %s", symbol)

                    with lock:
                        existing = dict(state["symbols"].get(symbol, {}))
                        existing["symbol"] = symbol
                        existing["data_error"] = str(error)[:180]
                        state["symbols"][symbol] = existing

            risk_check()

            with lock:
                state["last_scan"] = now_text()
                state["scan_count"] += 1
                state["last_error"] = None

            save_state()

        except Exception:
            log.exception("Worker cycle failed")

            with lock:
                state["last_error"] = "Worker cycle failed"

            save_state()

        if time.time() - last_private_check >= 300:
            check_private_account()
            last_private_check = time.time()

        elapsed = time.time() - cycle_start
        time.sleep(max(1, SCAN_SECONDS - elapsed))

def start_worker_once():
    global worker_started

    with worker_lock:
        if worker_started:
            return

        thread = threading.Thread(
            target=background_worker,
            name="CoinSentraV16Worker",
            daemon=True
        )

        thread.start()
        worker_started = True

# ============================================================
# PERFORMANCE
# ============================================================

def trade_statistics():
    with lock:
        trades = [dict(t) for t in state["history"] if clean_trade(t)]

    pnls = [number(t.get("pnl")) for t in trades]
    wins_list = [p for p in pnls if p > 0]
    losses_list = [p for p in pnls if p < 0]

    wins = len(wins_list)
    losses = len(losses_list)
    count = len(pnls)

    realized = sum(pnls)
    fees = sum(number(t.get("fees")) for t in trades)
    gross_wins = sum(wins_list)
    gross_losses = abs(sum(losses_list))

    profit_factor = gross_wins / gross_losses if gross_losses else (gross_wins if gross_wins else 0)
    average_win = gross_wins / wins if wins else 0
    average_loss = gross_losses / losses if losses else 0

    chronological = list(reversed(pnls))
    running_equity = START_BALANCE
    peak_equity = START_BALANCE
    max_drawdown = 0.0

    for pnl in chronological:
        running_equity += pnl
        peak_equity = max(peak_equity, running_equity)

        if peak_equity > 0:
            drawdown = (peak_equity - running_equity) / peak_equity * 100
            max_drawdown = max(max_drawdown, drawdown)

    return {
        "completed_count": count,
        "wins": wins,
        "losses": losses,
        "breakeven": count - wins - losses,
        "win_rate": round(wins / count * 100, 2) if count else 0,
        "history_realized_pnl": round(realized, 4),
        "total_fees": round(fees, 4),
        "profit_factor": round(profit_factor, 3),
        "average_win": round(average_win, 4),
        "average_loss": round(average_loss, 4),
        "max_drawdown": round(max_drawdown, 2)
    }

def learning_statistics():
    with lock:
        learning = copy.deepcopy(state["learning"])

        return {
            "enabled": learning["enabled"],
            "samples": learning["samples"],
            "updates": learning["updates"],
            "last_update": learning["last_update"],
            "last_result": learning["last_result"],
            "minimum_samples": LEARNING_MIN_SAMPLES,
            "window": LEARNING_WINDOW,
            "weights": dict(learning["weights"])
        }

# ============================================================
# DASHBOARD
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta charset="UTF-8">
<title>CoinSentra Trading Brain V16.8</title>
<style>
*{box-sizing:border-box}
body{background:#080d18;color:#e9efff;font-family:Arial;margin:0;padding:15px}
h1{color:#36d9ff;font-size:25px}
h2{color:#36d9ff;font-size:18px}
.card{background:#121d30;border:1px solid #263a58;border-radius:12px;padding:14px;margin:10px 0}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.big{font-size:22px;font-weight:bold;color:#35e0b0;margin-top:7px;overflow-wrap:anywhere}
.small{font-size:12px;color:#9bb0ce;line-height:1.6}
button{padding:13px;border:0;border-radius:8px;font-weight:bold;margin:5px 2px;cursor:pointer}
.start{background:#16b981;color:white}
.stop{background:#dc2525;color:white}
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
.weight{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #293a50}
.pattern{font-weight:bold}
</style>
</head>
<body>

<h1>COINSENTRA AI</h1>
<div class="small">TRADING BRAIN V16.8 | BYBIT | PAPER MODE ONLY | 5M RSI 50 DIRECTION + 15M TREND + CANDLE PATTERNS | VOLUME 0.50x | MAX 3 POSITIONS</div>

<div class="grid">
<div class="card">Realized Balance<div class="big" id="balance">$10,000</div></div>
<div class="card">Total Equity<div class="big" id="equity">$10,000</div></div>
<div class="card">Unrealized PNL<div class="big" id="unrealized">$0.00</div></div>
<div class="card">Daily Net PNL<div class="big" id="daily">$0.00</div></div>
<div class="card">Available Balance<div class="big" id="available">$10,000</div></div>
<div class="card">Open Positions<div class="big" id="count">0 / 3</div></div>
</div>

<div class="card">
<h2>CONTROL CENTER</h2>
<button class="stop" onclick="control('stop')">END TRADING (C)</button>
<button class="start" onclick="control('start')">START TRADING (V)</button>
<p id="control" class="small">Loading...</p>
<p class="small">Stopping disables new entries. Existing paper positions remain monitored.</p>
</div>

<div class="card">
<h2>ADAPTIVE AI LEARNING</h2>
<div class="statgrid">
<div class="stat"><span class="small">Learning samples</span><strong id="learnsamples">0</strong></div>
<div class="stat"><span class="small">Weight updates</span><strong id="learnupdates">0</strong></div>
<div class="stat"><span class="small">Last result</span><strong id="learnresult">Waiting</strong></div>
<div class="stat"><span class="small">Learning status</span><strong id="learnstatus">Collecting data</strong></div>
</div>
<p class="small" id="learnmessage">Learning uses completed paper trades.</p>
<h2>INDICATOR WEIGHTS</h2>
<div id="weights" class="small">Waiting...</div>
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
<div class="stat"><span class="small">Profit factor</span><strong id="profitfactor">0</strong></div>
<div class="stat"><span class="small">Average win / loss</span><strong id="averages">0 / 0</strong></div>
<div class="stat"><span class="small">Maximum drawdown</span><strong id="drawdown">0%</strong></div>
</div>
</div>

<div class="card">
<h2>BYBIT CONNECTION</h2>
<div id="private">Checking...</div>
<div id="account" class="small"></div>
<div class="small">Read-only wallet connection. No live orders are sent.</div>
</div>

<div class="card">
<h2>MARKET SCANNER</h2>
<div id="scanner" class="small">WAITING FOR MARKET DATA</div>
<div class="scroll">
<table>
<thead><tr><th>COIN</th><th>PRICE</th><th>LONG</th><th>SHORT</th><th>SIGNAL</th><th>15M TREND</th></tr></thead>
<tbody id="markets"></tbody>
</table>
</div>
</div>

<div class="card">
<h2>INDICATORS (5M RSI / 15M TREND / CANDLE PATTERNS)</h2>
<div id="indicators" class="small">Waiting for market data...</div>
</div>

<div class="card">
<h2>OPEN PAPER POSITIONS</h2>
<div class="scroll">
<table>
<thead><tr><th>COIN</th><th>SIDE</th><th>ENTRY</th><th>LIVE PRICE</th><th>NET PNL</th><th>MARGIN</th><th>TP</th><th>SL</th></tr></thead>
<tbody id="positions"></tbody>
</table>
</div>
</div>

<div class="card">
<h2>COMPLETED TRADE HISTORY</h2>
<div class="small">Closed trades only.</div>
<div class="scroll">
<table>
<thead><tr><th>CLOSED</th><th>COIN</th><th>SIDE</th><th>ENTRY</th><th>EXIT</th><th>GROSS PNL</th><th>FEES</th><th>NET PNL</th><th>REASON</th></tr></thead>
<tbody id="history"></tbody>
</table>
</div>
</div>

<div class="card small">
Version 16.8 | Easier entry scoring + candlestick recognition<br>
5M RSI above 50 bullish, below 50 bearish<br>
Volume gate 0.50x average volume<br>
Patterns: Engulfing, Hammer, Shooting Star, Morning/Evening Star, Three Soldiers/Crows, Doji, Wicks<br>
Maximum 3 positions | Minimum $60 notional<br>
Stop loss 1.4% | Take profit 1.7%<br>
60-second cooldown | Paper trading only<br>
Last scan: <span id="last">Waiting</span><br>
Last error: <span id="error">None</span>
</div>

<script>
async function control(action){
 const message=document.getElementById('control');
 if(action==='stop'){
  if(!confirm('End trading? Existing positions remain monitored.'))return;
 }
 message.textContent='Sending command...';
 try{
  const response=await fetch('/control/'+action,{method:'POST'});
  const data=await response.json();
  message.textContent=data.message||data.error||'Command received';
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
function money(x){return '$'+Number(x||0).toFixed(2);}
function price(x){let n=Number(x||0);return n?n.toPrecision(8):'-';}
function safe(x){
 return String(x??'').replace(/[&<>"']/g,c=>({
  '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
 }[c]));
}
function color(x){return x==='LONG'?'green':x==='SHORT'?'red':'yellow';}
function pnlColor(x){return Number(x)>=0?'green':'red';}
async function update(){
 try{
  let r=await fetch('/api/data',{cache:'no-store'});
  if(!r.ok)throw new Error('API error');
  let d=await r.json();

  document.getElementById('balance').textContent=money(d.balance);
  document.getElementById('equity').textContent=money(d.equity);
  document.getElementById('unrealized').textContent=money(d.unrealized);
  document.getElementById('daily').textContent=money(d.daily_pnl);
  document.getElementById('available').textContent=money(d.available_balance);
  document.getElementById('count').textContent=d.positions.length+' / 3';

  document.getElementById('control').textContent=
   d.running?'TRADING ACTIVE - NEW ENTRIES ENABLED':'TRADING STOPPED - MONITORING POSITIONS';

  document.getElementById('private').textContent=d.private_status;
  document.getElementById('account').textContent=
   d.account_balance===null?'Account balance unavailable':'Bybit USDT wallet: '+money(d.account_balance);

  document.getElementById('last').textContent=d.last_scan||'Waiting';
  document.getElementById('error').textContent=d.last_error||'None';

  let s=d.stats;
  document.getElementById('completed').textContent=s.completed_count;
  document.getElementById('winrate').textContent=s.win_rate+'%';
  document.getElementById('wins').textContent=s.wins;
  document.getElementById('losses').textContent=s.losses;
  document.getElementById('realized').textContent=money(s.history_realized_pnl);
  document.getElementById('fees').textContent=money(s.total_fees);
  document.getElementById('profitfactor').textContent=s.profit_factor;
  document.getElementById('averages').textContent=money(s.average_win)+' / '+money(s.average_loss);
  document.getElementById('drawdown').textContent=s.max_drawdown+'%';

  let learning=d.learning;
  document.getElementById('learnsamples').textContent=learning.samples+'/'+learning.minimum_samples+' minimum';
  document.getElementById('learnupdates').textContent=learning.updates;
  document.getElementById('learnresult').textContent=learning.last_result||'Waiting';
  document.getElementById('learnstatus').textContent=learning.samples>=learning.minimum_samples?'ADAPTIVE':'COLLECTING';

  document.getElementById('learnmessage').textContent=
   learning.samples>=learning.minimum_samples
   ?'Adaptive learning active. Last update: '+(learning.last_update||'N/A')
   :'Waiting for '+Math.max(0,learning.minimum_samples-learning.samples)+' eligible completed trades.';

  let weightRows='';
  for(let [name,value] of Object.entries(learning.weights||{})){
   weightRows+=`<div class="weight"><span>${safe(name.replaceAll('_',' ').toUpperCase())}</span><b>${Number(value).toFixed(3)}x</b></div>`;
  }
  document.getElementById('weights').innerHTML=weightRows||'Waiting...';

  document.getElementById('scanner').textContent=d.last_scan?'ACTIVE':'WAITING';

  let rows='',details='';
  for(let s of Object.values(d.symbols)){
   rows+=`<tr>
   <td>${safe(s.symbol)}</td><td>${price(s.price)}</td>
   <td class="green">${s.long_score??0}</td>
   <td class="red">${s.short_score??0}</td>
   <td class="${color(s.signal)}">${safe(s.signal||'WAIT')}</td>
   <td>${safe(s.hourly_trend||'-')}</td></tr>`;

   details+=`<p><b>${safe(s.symbol)}</b> |
   5M RSI ${s.rsi_5m??'-'} | 15M RSI ${s.rsi_15m??'-'} |
   15M ADX ${s.adx??'-'} |
   ATR ${price(s.atr)} |
   15M Volume ${s.volume_ratio??'-'}x |
   Orderbook ${s.orderbook_imbalance??'-'} |
   Funding ${s.funding??'-'} |
   OI ${s.oi_change??'-'}% |
   <span class="pattern">CANDLE: ${safe(s.candle_pattern||'No pattern')} (${safe(s.candle_signal||'NEUTRAL')})</span>
   ${s.data_error?' | ERROR: '+safe(s.data_error):''}</p>`;
  }

  document.getElementById('markets').innerHTML=rows||'<tr><td colspan="6">Waiting for market scan</td></tr>';
  document.getElementById('indicators').innerHTML=details||'Waiting for data';

  let positionRows='';
  for(let p of d.positions){
   positionRows+=`<tr>
    <td>${safe(p.symbol)}</td><td>${safe(p.side)}</td>
    <td>${price(p.entry)}</td><td>${price(p.current_price)}</td>
    <td class="${pnlColor(p.unrealized_pnl)}">${money(p.unrealized_pnl)}</td>
    <td>${money(p.margin)}</td><td>${price(p.take_profit)}</td>
    <td>${price(p.stop_loss)}</td></tr>`;
  }
  document.getElementById('positions').innerHTML=positionRows||'<tr><td colspan="8">No open positions</td></tr>';

  let historyRows='';
  for(let p of d.history){
   historyRows+=`<tr>
    <td>${safe(p.closed||'-')}</td><td>${safe(p.symbol)}</td>
    <td>${safe(p.side)}</td><td>${price(p.entry)}</td>
    <td>${price(p.exit)}</td><td>${money(p.gross_pnl)}</td>
    <td>${money(p.fees)}</td>
    <td class="${pnlColor(p.pnl)}">${money(p.pnl)}</td>
    <td>${safe(p.reason)}</td></tr>`;
  }
  document.getElementById('history').innerHTML=historyRows||'<tr><td colspan="9">No completed trades yet</td></tr>';

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

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    start_worker_once()
    return render_template_string(HTML)

@app.route("/health")
def health():
    start_worker_once()
    return jsonify({"status": "ok", "version": VERSION, "paper_trading": True})

@app.route("/api/status")
@app.route("/status")
def status():
    start_worker_once()

    with lock:
        return jsonify({
            "status": "ok",
            "version": state["version"],
            "running": state["running"],
            "last_scan": state["last_scan"],
            "last_error": state["last_error"],
            "symbols": state["symbols"],
            "scan_count": state["scan_count"],
            "open_positions": len(state["positions"]),
            "completed_trades": len(state["history"]),
            "learning": state["learning"]
        })

@app.route("/api/data")
def api_data():
    start_worker_once()

    with lock:
        positions = copy.deepcopy(state["positions"])
        history = []

        for item in state["history"]:
            trade = clean_trade(item)
            if trade:
                history.append(trade)

        symbols = copy.deepcopy(state["symbols"])
        balance = number(state["balance"])
        daily_pnl = number(state["daily_pnl"])
        running = state["running"]
        last_scan = state["last_scan"]
        private_status = state["private_status"]
        account_balance = state["account_balance"]
        last_error = state["last_error"]

    unrealized = sum(number(p.get("unrealized_pnl")) for p in positions)
    used_margin = sum(number(p.get("margin")) for p in positions)

    totals = {
        "balance": round(balance, 4),
        "unrealized": round(unrealized, 4),
    +#+#+#+#+#+