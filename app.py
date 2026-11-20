
import os
import json
import time
import uuid
import math
import threading
import logging
from datetime import datetime, timezone, date

from flask import Flask, jsonify, render_template_string
from pybit.unified_trading import HTTP

# ============================================================
# COINSENTRA TRADING BRAIN V16
# BYBIT | RENDER | PAPER TRADING ONLY
# HISTORY RECOVERY | ADAPTIVE LEARNING | ATR STOPS
# ============================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("CoinSentraV16")

# ---------------- SETTINGS ----------------

SYMBOLS = ["ETHUSDT", "XRPUSDT", "SOLUSDT", "SUIUSDT"]
CATEGORY = "linear"

LEVERAGE = 8
MAX_POSITIONS = 2
DAILY_LOSS_LIMIT = 6.0

START_BALANCE = 10000.0
RISK_PER_TRADE = 3.0
MAX_EQUITY_EXPOSURE = 0.10

SCAN_SECONDS = 30
ENTRY_SCORE = 68
CONFIRMATION_INTERVAL = "15"
HIGHER_INTERVAL = "60"

ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 1.5
MIN_STOP_PERCENT = 0.004
MAX_STOP_PERCENT = 0.015
REWARD_RISK = 1.8

COOLDOWN_SECONDS = 600
FEE_RATE = 0.00055
PAPER_TRADING = True

VERSION = "16.0"
MAX_HISTORY = 500

# ---------------- ADAPTIVE LEARNING ----------------

LEARNING_ENABLED = True
LEARNING_MIN_SAMPLES = 10
LEARNING_WINDOW = 30
LEARNING_MIN_WEIGHT = 0.75
LEARNING_MAX_WEIGHT = 1.25
LEARNING_RATE = 0.15

LEARNING_FEATURES = [
    "trend_fast", "trend_slow", "rsi", "macd", "volume",
    "orderbook", "confirmation", "higher_timeframe",
    "adx", "funding", "open_interest"
]

DEFAULT_WEIGHTS = {name: 1.0 for name in LEARNING_FEATURES}

# ---------------- STORAGE ----------------

DATA_DIR = os.getenv("DATA_DIR", "/tmp")
STATE_FILE = os.path.join(DATA_DIR, "coinsentra_v16.json")

API_KEY = os.getenv("BYBIT_API_KEY", "").strip()
API_SECRET = os.getenv("BYBIT_API_SECRET", "").strip()

# ---------------- BYBIT CONNECTION ----------------

public_api = HTTP(testnet=False, timeout=15)
private_api = None

if API_KEY and API_SECRET:
    private_api = HTTP(
        api_key=API_KEY,
        api_secret=API_SECRET,
        testnet=False,
        timeout=15
    )

# ---------------- STATE ----------------

lock = threading.RLock()
storage_lock = threading.Lock()
worker_lock = threading.Lock()
worker_started = False

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
    "learning": {
        "enabled": LEARNING_ENABLED,
        "weights": dict(DEFAULT_WEIGHTS),
        "samples": 0,
        "updates": 0,
        "last_update": None,
        "last_result": "Waiting for completed trades",
        "processed_ids": []
    }
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def now_text():
    return datetime.now(timezone.utc).isoformat()


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def check_bybit_result(result):
    if not isinstance(result, dict):
        raise ValueError("Invalid Bybit response")

    code = result.get("retCode", 0)

    if code not in (0, "0", None):
        raise ValueError(
            "Bybit error " + str(code) + ": "
            + str(result.get("retMsg", "Unknown error"))
        )

    return result


def first_value(data, names, default=None):
    for name in names:
        value = data.get(name)
        if value is not None:
            return value
    return default


# ============================================================
# TRADE CLEANING AND HISTORY MIGRATION
# ============================================================

def clean_trade(trade):
    """
    Normalize completed trades from V16 and older versions.
    Open positions are never converted into completed trades.
    """

    if not isinstance(trade, dict):
        return None

    status = str(trade.get("status", "COMPLETED")).upper()

    if status in ("OPEN", "ACTIVE", "PENDING"):
        return None

    entry = number(first_value(
        trade, ["entry", "entry_price", "avg_entry_price"]
    ))

    exit_price = number(first_value(
        trade, ["exit", "exit_price", "close_price", "avg_exit_price"]
    ))

    quantity = number(first_value(
        trade, ["quantity", "qty", "size", "closed_qty"]
    ))

    pnl_value = first_value(
        trade, ["pnl", "net_pnl", "realized_pnl", "profit"]
    )

    if pnl_value is None:
        return None

    if entry <= 0 or exit_price <= 0 or quantity <= 0:
        return None

    pnl = number(pnl_value)
    fees = number(first_value(
        trade, ["fees", "fee", "total_fee"], 0
    ))

    gross = number(first_value(
        trade, ["gross_pnl", "gross_profit"], pnl + fees
    ))

    snapshot = first_value(
        trade, ["feature_snapshot", "features", "learning_features"], {}
    )

    if not isinstance(snapshot, dict):
        snapshot = {}

    normalized_snapshot = {}

    for name in LEARNING_FEATURES:
        if name in snapshot:
            normalized_snapshot[name] = max(
                -1.0, min(1.0, number(snapshot.get(name)))
            )

    result = dict(trade)
    result.update({
        "trade_id": str(
            first_value(trade, ["trade_id", "id"], "OLD-" + uuid.uuid4().hex[:12])
        ),
        "status": "COMPLETED",
        "symbol": str(trade.get("symbol", "UNKNOWN")),
        "side": str(trade.get("side", "UNKNOWN")).upper(),
        "entry": entry,
        "exit": exit_price,
        "quantity": quantity,
        "gross_pnl": gross,
        "fees": fees,
        "pnl": pnl,
        "net_pnl": pnl,
        "reason": str(trade.get("reason", "CLOSED")),
        "opened": str(trade.get("opened", trade.get("entry_time", ""))),
        "closed": str(trade.get("closed", trade.get("close_time", ""))),
        "feature_snapshot": normalized_snapshot
    })

    return result


def normalize_history(items):
    cleaned = []
    seen = set()

    if not isinstance(items, list):
        return cleaned

    for item in items:
        trade = clean_trade(item)

        if trade is None:
            continue

        trade_id = trade["trade_id"]

        if trade_id in seen:
            continue

        seen.add(trade_id)
        cleaned.append(trade)

    # Newest first when timestamps are available.
    cleaned.sort(
        key=lambda t: str(t.get("closed", "")),
        reverse=True
    )

    return cleaned[:MAX_HISTORY]


# ============================================================
# STATE SAVE / LOAD
# ============================================================

def save_state():
    """
    Atomic JSON save.
    Do not call while holding the state lock.
    """

    temp_file = None

    try:
        with storage_lock:
            os.makedirs(DATA_DIR, exist_ok=True)

            with lock:
                snapshot = json.loads(
                    json.dumps(state, allow_nan=False)
                )

            temp_file = STATE_FILE + "." + uuid.uuid4().hex + ".tmp"

            with open(temp_file, "w", encoding="utf-8") as file:
                json.dump(
                    snapshot,
                    file,
                    indent=2,
                    allow_nan=False
                )
                file.flush()
                os.fsync(file.fileno())

            os.replace(temp_file, STATE_FILE)

        log.info(
            "STATE SAVED | positions=%s | history=%s | learning_samples=%s",
            len(snapshot.get("positions", [])),
            len(snapshot.get("history", [])),
            snapshot.get("learning", {}).get("samples", 0)
        )
        return True

    except Exception:
        log.exception("V16 state save failed")
        return False

    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass


def recover_position(position):
    """
    Recover an older open position without undefined constants.
    Existing valid SL/TP values are never overwritten.
    """

    if not isinstance(position, dict):
        return None

    position = dict(position)

    entry = number(position.get("entry"))
    quantity = number(position.get("quantity"))
    side = str(position.get("side", "")).upper()

    if entry <= 0 or quantity <= 0 or side not in ("LONG", "SHORT"):
        position["state_error"] = "Invalid saved position values"
        return position

    position.setdefault("trade_id", uuid.uuid4().hex)
    position.setdefault("opened", now_text())
    position.setdefault("current_price", entry)
    position.setdefault("unrealized_pnl", 0.0)
    position.setdefault("gross_unrealized_pnl", 0.0)
    position.setdefault("feature_snapshot", {})

    if not isinstance(position["feature_snapshot"], dict):
        position["feature_snapshot"] = {}

    stop = number(position.get("stop_loss"))
    target = number(position.get("take_profit"))

    # Use stored stop distance first, otherwise a conservative
    # minimum-stop fallback for positions from older versions.
    stop_distance = number(position.get("stop_distance"))

    if stop_distance <= 0:
        stop_distance = entry * MIN_STOP_PERCENT

    if stop <= 0:
        stop = (
            entry - stop_distance
            if side == "LONG"
            else entry + stop_distance
        )

    if target <= 0:
        target_distance = stop_distance * REWARD_RISK
        target = (
            entry + target_distance
            if side == "LONG"
            else entry - target_distance
        )

    position["stop_loss"] = stop
    position["take_profit"] = target
    position["stop_distance"] = stop_distance
    position.pop("state_error", None)

    return position


def rebuild_learning_samples(history):
    count = 0

    for trade in history:
        snapshot = trade.get("feature_snapshot", {})

        if not isinstance(snapshot, dict) or not snapshot:
            continue

        if number(trade.get("pnl")) == 0:
            continue

        count += 1

    return count


def load_state():
    if not os.path.exists(STATE_FILE):
        log.info("New V16 paper account")
        return

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            saved = json.load(file)

        if not isinstance(saved, dict):
            raise ValueError("Saved state is not a dictionary")

        history = normalize_history(saved.get("history", []))

        positions = []
        for item in saved.get("positions", []):
            recovered = recover_position(item)
            if recovered is not None:
                positions.append(recovered)

        saved_learning = saved.get("learning", {})
        if not isinstance(saved_learning, dict):
            saved_learning = {}

        saved_weights = saved_learning.get("weights", {})
        if not isinstance(saved_weights, dict):
            saved_weights = {}

        weights = {}

        for name in LEARNING_FEATURES:
            weights[name] = round(
                max(
                    LEARNING_MIN_WEIGHT,
                    min(
                        LEARNING_MAX_WEIGHT,
                        number(saved_weights.get(name), 1.0)
                    )
                ),
                4
            )

        processed_ids = saved_learning.get("processed_ids", [])
        if not isinstance(processed_ids, list):
            processed_ids = []

        learning = {
            "enabled": LEARNING_ENABLED,
            "weights": weights,
            "samples": rebuild_learning_samples(history),
            "updates": max(0, safe_int(saved_learning.get("updates"))),
            "last_update": saved_learning.get("last_update"),
            "last_result": saved_learning.get(
                "last_result", "Waiting for completed trades"
            ),
            "processed_ids": [str(x) for x in processed_ids][-MAX_HISTORY:]
        }

        with lock:
            state["balance"] = number(
                saved.get("balance"), START_BALANCE
            )
            state["positions"] = positions
            state["history"] = history
            state["daily_pnl"] = number(saved.get("daily_pnl"))
            state["daily_date"] = str(
                saved.get("daily_date", str(date.today()))
            )
            state["running"] = bool(saved.get("running", True))
            state["last_scan"] = saved.get("last_scan")
            state["last_error"] = saved.get("last_error")
            state["private_status"] = saved.get(
                "private_status", "NOT CONFIGURED"
            )
            state["account_balance"] = saved.get("account_balance")
            state["symbols"] = (
                saved.get("symbols", {})
                if isinstance(saved.get("symbols", {}), dict)
                else {}
            )
            state["scan_count"] = max(
                0, safe_int(saved.get("scan_count"))
            )
            state["last_closed_by_symbol"] = (
                saved.get("last_closed_by_symbol", {})
                if isinstance(saved.get("last_closed_by_symbol", {}), dict)
                else {}
            )
            state["learning"] = learning
            state["version"] = VERSION

        log.info(
            "V16 RESTORED | positions=%s | completed=%s | eligible_learning_samples=%s",
            len(positions),
            len(history),
            learning["samples"]
        )

        save_state()

    except Exception:
        log.exception("V16 state loading failed")


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
# ADAPTIVE LEARNING
# ============================================================

def learning_snapshot_for_side(features, side):
    if not isinstance(features, dict):
        return {}

    direction = 1 if side == "LONG" else -1
    snapshot = {}

    for name in LEARNING_FEATURES:
        value = number(features.get(name))
        snapshot[name] = max(
            -1.0, min(1.0, value * direction)
        )

    return snapshot


def update_learning(trade):
    if not LEARNING_ENABLED or not isinstance(trade, dict):
        return

    trade = clean_trade(trade)

    if trade is None:
        log.warning("LEARNING SKIPPED: invalid completed trade")
        return

    snapshot = trade.get("feature_snapshot", {})
    pnl = number(trade.get("pnl"))
    trade_id = str(trade.get("trade_id"))

    if not snapshot:
        log.warning(
            "LEARNING SKIPPED: missing feature snapshot | %s",
            trade_id
        )
        return

    if pnl == 0:
        log.info("LEARNING SKIPPED: breakeven trade | %s", trade_id)
        return

    with lock:
        learning = state["learning"]
        processed = learning.get("processed_ids", [])

        if trade_id in processed:
            log.info("LEARNING DUPLICATE IGNORED | %s", trade_id)
            return

        processed.append(trade_id)
        learning["processed_ids"] = processed[-MAX_HISTORY:]

        # Count eligible completed trades from the actual history.
        eligible = []

        for item in state["history"]:
            normalized = clean_trade(item)

            if normalized is None:
                continue

            features = normalized.get("feature_snapshot", {})

            if not features or number(normalized.get("pnl")) == 0:
                continue

            eligible.append({
                "features": features,
                "outcome": 1.0 if number(normalized["pnl"]) > 0 else -1.0
            })

        eligible = eligible[:LEARNING_WINDOW]
        learning["samples"] = rebuild_learning_samples(state["history"])

        if len(eligible) < LEARNING_MIN_SAMPLES:
            remaining = LEARNING_MIN_SAMPLES - len(eligible)
            learning["last_result"] = (
                "Collecting samples: " + str(remaining) + " more needed"
            )
            learning["last_update"] = now_text()

            log.info(
                "LEARNING COLLECTING | eligible=%s | required=%s",
                len(eligible),
                LEARNING_MIN_SAMPLES
            )

        else:
            weights = dict(learning["weights"])

            for name in LEARNING_FEATURES:
                values = []

                for sample in eligible:
                    feature_value = number(
                        sample["features"].get(name)
                    )
                    values.append(
                        max(-1.0, min(1.0, feature_value))
                        * sample["outcome"]
                    )

                correlation = (
                    sum(values) / len(values)
                    if values else 0.0
                )

                new_weight = (
                    number(weights.get(name), 1.0)
                    + correlation * LEARNING_RATE
                )

                weights[name] = round(
                    max(
                        LEARNING_MIN_WEIGHT,
                        min(LEARNING_MAX_WEIGHT, new_weight)
                    ),
                    4
                )

            learning["weights"] = weights
            learning["updates"] = safe_int(
                learning.get("updates")
            ) + 1
            learning["last_update"] = now_text()
            learning["last_result"] = (
                "Weights updated from recent completed trades"
            )

            log.info(
                "LEARNING UPDATED | samples=%s | updates=%s",
                len(eligible),
                learning["updates"]
            )

    save_state()


def get_learning_weights():
    with lock:
        return dict(state["learning"]["weights"])


# ============================================================
# MARKET DATA
# ============================================================

def get_price(symbol):
    result = check_bybit_result(
        public_api.get_tickers(
            category=CATEGORY,
            symbol=symbol
        )
    )

    rows = result.get("result", {}).get("list", [])

    if not rows:
        raise ValueError("No ticker data: " + symbol)

    price = number(rows[0].get("lastPrice"))

    if price <= 0:
        raise ValueError("Invalid ticker price: " + symbol)

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

    rows = result.get("result", {}).get("list", [])

    if not rows:
        raise ValueError("No candle data: " + symbol)

    candles = []

    for row in reversed(rows):
        if len(row) < 7:
            continue

        candles.append({
            "time": safe_int(row[0]),
            "open": number(row[1]),
            "high": number(row[2]),
            "low": number(row[3]),
            "close": number(row[4]),
            "volume": number(row[5]),
            "turnover": number(row[6])
        })

    if len(candles) < 60:
        raise ValueError("Insufficient candle history: " + symbol)

    # Exclude the currently forming candle.
    candles = candles[:-1]

    if len(candles) < 55:
        raise ValueError("Not enough completed candles: " + symbol)

    return candles


def get_orderbook(symbol):
    result = check_bybit_result(
        public_api.get_orderbook(
            category=CATEGORY,
            symbol=symbol,
            limit=25
        )
    )

    data = result.get("result", {})
    bids = data.get("b", [])
    asks = data.get("a", [])

    bid_volume = sum(
        number(x[1]) for x in bids if len(x) > 1
    )
    ask_volume = sum(
        number(x[1]) for x in asks if len(x) > 1
    )

    total = bid_volume + ask_volume

    if total <= 0:
        return 0.0

    return (bid_volume - ask_volume) / total


def get_funding(symbol):
    try:
        result = check_bybit_result(
            public_api.get_funding_rate_history(
                category=CATEGORY,
                symbol=symbol,
                limit=1
            )
        )

        rows = result.get("result", {}).get("list", [])

        if rows:
            return number(rows[0].get("fundingRate"))

    except Exception as error:
        log.warning("Funding error %s: %s", symbol, error)

    return 0.0


def get_open_interest(symbol):
    try:
        result = check_bybit_result(
            public_api.get_open_interest(
                category=CATEGORY,
                symbol=symbol,
                intervalTime="5min",
                limit=2
            )
        )

        rows = result.get("result", {}).get("list", [])

        if len(rows) >= 2:
            older = number(rows[1].get("openInterest"))
            newer = number(rows[0].get("openInterest"))

            if older > 0:
                return (newer - older) / older * 100

    except Exception as error:
        log.warning("OI error %s: %s", symbol, error)

    return 0.0


# ============================================================
# INDICATORS
# ============================================================

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
    variance = sum(
        (x - middle) ** 2 for x in sample
    ) / period

    deviation = math.sqrt(max(0.0, variance))

    return (
        middle + 2 * deviation,
        middle,
        middle - 2 * deviation
    )


def atr(candles, period=ATR_PERIOD):
    if len(candles) < period + 1:
        return 0.0

    true_ranges = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        true_ranges.append(max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"])
        ))

    sample = true_ranges[-period:]

    return sum(sample) / len(sample) if sample else 0.0


def adx_approx(candles, period=14):
    if len(candles) < period + 2:
        return 0.0

    tr_values = []
    plus_values = []
    minus_values = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        tr_values.append(max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"])
        ))

        up = current["high"] - previous["high"]
        down = previous["low"] - current["low"]

        plus_values.append(
            up if up > down and up > 0 else 0
        )
        minus_values.append(
            down if down > up and down > 0 else 0
        )

    tr_avg = sum(tr_values[-period:]) / period

    if tr_avg <= 0:
        return 0.0

    plus_di = (
        sum(plus_values[-period:]) / period
        / tr_avg * 100
    )
    minus_di = (
        sum(minus_values[-period:]) / period
        / tr_avg * 100
    )

    total = plus_di + minus_di

    if total <= 0:
        return 0.0

    return abs(plus_di - minus_di) / total * 100


def analyze_candles(candles):
    closes = [x["close"] for x in candles]
    volumes = [x["volume"] for x in candles]

    if len(closes) < 60:
        raise ValueError("Insufficient candles for indicators")

    upper, middle, lower = bollinger(closes)

    previous_volumes = volumes[-21:-1]
    average_volume = (
        sum(previous_volumes) / len(previous_volumes)
        if previous_volumes else 0.0
    )

    volume_ratio = (
        volumes[-1] / average_volume
        if average_volume > 0 else 0.0
    )

    return {
        "price": closes[-1],
        "ema9": ema(closes, 9),
        "ema21": ema(closes, 21),
        "ema50": ema(closes, 50),
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

def calculate_signal(five, fifteen, hourly, imbalance, funding, oi_change):
    weights = get_learning_weights()

    long_score = 0.0
    short_score = 0.0

    features = {name: 0.0 for name in LEARNING_FEATURES}

    if five["ema9"] > five["ema21"]:
        features["trend_fast"] = 1
        long_score += 14 * weights["trend_fast"]
    elif five["ema9"] < five["ema21"]:
        features["trend_fast"] = -1
        short_score += 14 * weights["trend_fast"]

    if five["ema21"] > five["ema50"]:
        features["trend_slow"] = 1
        long_score += 12 * weights["trend_slow"]
    elif five["ema21"] < five["ema50"]:
        features["trend_slow"] = -1
        short_score += 12 * weights["trend_slow"]

    if 50 <= five["rsi"] <= 68:
        features["rsi"] = 1
        long_score += 8 * weights["rsi"]
    elif 32 <= five["rsi"] <= 50:
        features["rsi"] = -1
        short_score += 8 * weights["rsi"]

    if five["macd"] > 0:
        features["macd"] = 1
        long_score += 8 * weights["macd"]
    elif five["macd"] < 0:
        features["macd"] = -1
        short_score += 8 * weights["macd"]

    if five["volume_ratio"] >= 1.15:
        if five["ema9"] > five["ema21"]:
            features["volume"] = 1
            long_score += 8 * weights["volume"]
        elif five["ema9"] < five["ema21"]:
            features["volume"] = -1
            short_score += 8 * weights["volume"]

    if imbalance > 0.15:
        features["orderbook"] = 1
        long_score += 5 * weights["orderbook"]
    elif imbalance < -0.15:
        features["orderbook"] = -1
        short_score += 5 * weights["orderbook"]

    if fifteen["ema9"] > fifteen["ema21"]:
        features["confirmation"] = 1
        long_score += 15 * weights["confirmation"]
    elif fifteen["ema9"] < fifteen["ema21"]:
        features["confirmation"] = -1
        short_score += 15 * weights["confirmation"]

    if (
        hourly["ema9"] > hourly["ema21"]
        and hourly["ema21"] > hourly["ema50"]
    ):
        features["higher_timeframe"] = 1
        long_score += 20 * weights["higher_timeframe"]

    elif (
        hourly["ema9"] < hourly["ema21"]
        and hourly["ema21"] < hourly["ema50"]
    ):
        features["higher_timeframe"] = -1
        short_score += 20 * weights["higher_timeframe"]

    if five["adx"] >= 18:
        if five["ema9"] > five["ema21"]:
            features["adx"] = 1
            long_score += 6 * weights["adx"]
        elif five["ema9"] < five["ema21"]:
            features["adx"] = -1
            short_score += 6 * weights["adx"]

    if funding > 0.0005:
        features["funding"] = -1
        short_score += 2 * weights["funding"]
    elif funding < -0.0005:
        features["funding"] = 1
        long_score += 2 * weights["funding"]

    if oi_change > 0.2:
        if five["ema9"] > five["ema21"]:
            features["open_interest"] = 1
            long_score += 2 * weights["open_interest"]
        elif five["ema9"] < five["ema21"]:
            features["open_interest"] = -1
            short_score += 2 * weights["open_interest"]

    long_score = min(100, round(long_score))
    short_score = min(100, round(short_score))

    long_confirmed = (
        five["ema9"] > five["ema21"]
        and five["ema21"] > five["ema50"]
        and fifteen["ema9"] > fifteen["ema21"]
        and hourly["ema9"] > hourly["ema21"]
        and hourly["ema21"] > hourly["ema50"]
        and 50 <= five["rsi"] <= 68
        and five["adx"] >= 18
        and five["volume_ratio"] >= 1.15
    )

    short_confirmed = (
        five["ema9"] < five["ema21"]
        and five["ema21"] < five["ema50"]
        and fifteen["ema9"] < fifteen["ema21"]
        and hourly["ema9"] < hourly["ema21"]
        and hourly["ema21"] < hourly["ema50"]
        and 32 <= five["rsi"] <= 50
        and five["adx"] >= 18
        and five["volume_ratio"] >= 1.15
    )

    signal = "WAIT"

    if (
        long_score >= ENTRY_SCORE
        and long_score - short_score >= 15
        and long_confirmed
        and five["price"] > five["ema21"]
    ):
        signal = "LONG"

    elif (
        short_score >= ENTRY_SCORE
        and short_score - long_score >= 15
        and short_confirmed
        and five["price"] < five["ema21"]
    ):
        signal = "SHORT"

    return {
        "long_score": long_score,
        "short_score": short_score,
        "signal": signal,
        "features": features
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

    return (
        entry * quantity * FEE_RATE
        + price * quantity * FEE_RATE
    )


def position_net_pnl(position, price):
    return position_gross_pnl(position, price) - position_fees(
        position, price
    )


def account_totals():
    with lock:
        balance = number(state["balance"])
        positions = [dict(p) for p in state["positions"]]

    unrealized = sum(
        number(p.get("unrealized_pnl")) for p in positions
    )
    used_margin = sum(
        number(p.get("margin")) for p in positions
    )

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
        closed_at = number(
            state["last_closed_by_symbol"].get(symbol)
        )

    return (
        closed_at > 0
        and time.time() - closed_at < COOLDOWN_SECONDS
    )


# ============================================================
# OPEN PAPER POSITION
# ============================================================

def open_paper_position(symbol, side, price, features=None, atr_value=0.0):
    if not PAPER_TRADING or side not in ("LONG", "SHORT"):
        return False

    if price <= 0 or atr_value <= 0:
        return False

    if symbol_in_cooldown(symbol):
        log.info("COOLDOWN ACTIVE | %s", symbol)
        return False

    with lock:
        if not state["running"]:
            return False

        if len(state["positions"]) >= MAX_POSITIONS:
            return False

        if any(p.get("symbol") == symbol for p in state["positions"]):
            return False

        if state["daily_pnl"] <= -DAILY_LOSS_LIMIT:
            state["running"] = False
            return False

        balance = number(state["balance"])

        if balance <= 0:
            return False

        raw_stop = atr_value * ATR_STOP_MULTIPLIER
        minimum_stop = price * MIN_STOP_PERCENT
        maximum_stop = price * MAX_STOP_PERCENT
        stop_distance = max(raw_stop, minimum_stop)

        if stop_distance > maximum_stop:
            log.info("ENTRY SKIPPED | %s | ATR stop too wide", symbol)
            return False

        risk_amount = min(RISK_PER_TRADE, balance * 0.005)
        quantity = risk_amount / stop_distance

        max_notional = balance * MAX_EQUITY_EXPOSURE
        quantity = min(quantity, max_notional / price)

        if quantity <= 0:
            return False

        notional = quantity * price
        margin = notional / LEVERAGE

        if margin > balance:
            return False

        if side == "LONG":
            stop_price = price - stop_distance
            target_price = price + stop_distance * REWARD_RISK
        else:
            stop_price = price + stop_distance
            target_price = price - stop_distance * REWARD_RISK

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
            "reward_risk": REWARD_RISK,
            "opened": now_text(),
            "unrealized_pnl": 0.0,
            "gross_unrealized_pnl": 0.0,
            "entry_fee": notional * FEE_RATE,
            "fee_rate": FEE_RATE,
            "feature_snapshot": learning_snapshot_for_side(
                features or {}, side
            )
        }

        state["positions"].append(position)

    if not save_state():
        with lock:
            state["positions"] = [
                p for p in state["positions"]
                if p.get("trade_id") != position["trade_id"]
            ]
        log.error("POSITION NOT OPENED: state save failed")
        return False

    log.info(
        "PAPER OPEN | %s | %s | Entry %.8f | SL %.8f | TP %.8f | Risk %.4f",
        side, symbol, price, stop_price, target_price,
        position["risk_amount"]
    )

    return True


# ============================================================
# COMPLETED TRADE ENGINE
# ============================================================

def close_paper_position(position, price, reason):
    if not PAPER_TRADING or not isinstance(position, dict):
        return False

    trade_id = position.get("trade_id")

    if not trade_id or price <= 0:
        log.error("CLOSE FAILED: invalid ID or price")
        return False

    trade = None

    with lock:
        live = next(
            (
                p for p in state["positions"]
                if p.get("trade_id") == trade_id
            ),
            None
        )

        if live is None:
            return False

        entry = number(live.get("entry"))
        quantity = number(live.get("quantity"))

        if entry <= 0 or quantity <= 0:
            live["state_error"] = "Invalid entry or quantity"
            log.error("CLOSE FAILED: invalid values | %s", trade_id)
            return False

        # If this ID already exists in history, remove the duplicate
        # position rather than creating a second completed trade.
        existing = next(
            (
                t for t in state["history"]
                if isinstance(t, dict)
                and str(t.get("trade_id")) == str(trade_id)
            ),
            None
        )

        if existing is not None and clean_trade(existing) is not None:
            state["positions"].remove(live)
            log.warning("DUPLICATE CLOSE RECOVERED | %s", trade_id)
            duplicate_recovery = True

        else:
            duplicate_recovery = False

            gross = position_gross_pnl(live, price)
            fees = position_fees(live, price)
            net = gross - fees
            closed = now_text()

            trade = {
                "trade_id": str(trade_id),
                "status": "COMPLETED",
                "symbol": live.get("symbol", "UNKNOWN"),
                "side": live.get("side", "UNKNOWN"),
                "entry": entry,
                "exit": price,
                "quantity": quantity,
                "notional": number(
                    live.get("notional"), entry * quantity
                ),
                "leverage": number(
                    live.get("leverage"), LEVERAGE
                ),
                "margin": number(live.get("margin")),
                "gross_pnl": round(gross, 6),
                "fees": round(fees, 6),
                "pnl": round(net, 6),
                "net_pnl": round(net, 6),
                "reason": str(reason),
                "opened": live.get("opened", ""),
                "closed": closed,
                "feature_snapshot": dict(
                    live.get("feature_snapshot", {})
                )
            }

            state["history"].insert(0, trade)
            state["history"] = normalize_history(state["history"])

            state["balance"] += net
            state["daily_pnl"] += net

            symbol = live.get("symbol")

            if symbol:
                state["last_closed_by_symbol"][symbol] = time.time()

            state["positions"].remove(live)

            if state["daily_pnl"] <= -DAILY_LOSS_LIMIT:
                state["running"] = False

            log.info(
                "TRADE COMPLETED | %s | %s | Gross %.4f | Fees %.4f | Net %.4f | ID %s",
                symbol, reason, gross, fees, net, trade_id
            )

    # History insertion and position removal are saved together.
    if not save_state():
        log.error("CRITICAL: completed trade save failed | %s", trade_id)
        return False

    if duplicate_recovery:
        return True

    try:
        update_learning(trade)
    except Exception:
        log.exception("Learning update failed | %s", trade_id)

    return True


# ============================================================
# POSITION MONITOR
# ============================================================

def monitor_positions():
    with lock:
        positions = [dict(p) for p in state["positions"]]

    changed = False

    for position in positions:
        try:
            entry = number(position.get("entry"))
            quantity = number(position.get("quantity"))
            side = position.get("side")

            if (
                entry <= 0
                or quantity <= 0
                or side not in ("LONG", "SHORT")
            ):
                log.error(
                    "MONITOR SKIPPED INVALID POSITION | %s",
                    position.get("trade_id")
                )
                continue

            stop_price = number(position.get("stop_loss"))
            target_price = number(position.get("take_profit"))

            if stop_price <= 0 or target_price <= 0:
                log.error(
                    "MONITOR SKIPPED: missing SL/TP | %s",
                    position.get("trade_id")
                )
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
            log.exception(
                "Monitor error for %s",
                position.get("symbol")
            )

    if changed:
        save_state()


# ============================================================
# DAILY RISK
# ============================================================

def risk_check():
    with lock:
        realized = number(state["daily_pnl"])
        unrealized = sum(
            number(p.get("unrealized_pnl"))
            for p in state["positions"]
        )
        total_daily_pnl = realized + unrealized

        if total_daily_pnl <= -DAILY_LOSS_LIMIT:
            state["running"] = False
            log.warning(
                "DAILY LOSS LIMIT REACHED | %.4f",
                total_daily_pnl
            )
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
            coins = accounts[0].get("coin", [])
            wallet = next(
                (
                    item for item in coins
                    if item.get("coin") == "USDT"
                ),
                None
            )

        with lock:
            state["account_balance"] = (
                number(wallet.get("walletBalance"))
                if wallet else 0.0
            )
            state["private_status"] = "CONNECTED - READ ONLY"

    except Exception as error:
        with lock:
            state["private_status"] = "CONNECTION FAILED"

        log.error("Private account error: %s", error)

    save_state()


# ============================================================
# SCANNER
# ============================================================

def scan_symbol(symbol):
    five_candles = get_candles(symbol, "5", 200)
    fifteen_candles = get_candles(
        symbol, CONFIRMATION_INTERVAL, 120
    )
    hourly_candles = get_candles(
        symbol, HIGHER_INTERVAL, 120
    )

    five = analyze_candles(five_candles)
    fifteen = analyze_candles(fifteen_candles)
    hourly = analyze_candles(hourly_candles)

    five["price"] = get_price(symbol)

    imbalance = get_orderbook(symbol)
    funding = get_funding(symbol)
    oi_change = get_open_interest(symbol)

    signal = calculate_signal(
        five, fifteen, hourly, imbalance, funding, oi_change
    )

    data = {
        "symbol": symbol,
        "price": five["price"],
        "rsi": round(five["rsi"], 2),
        "macd": round(five["macd"], 8),
        "adx": round(five["adx"], 2),
        "atr": round(five["atr"], 8),
        "ema9": round(five["ema9"], 8),
        "ema21": round(five["ema21"], 8),
        "ema50": round(five["ema50"], 8),
        "volume_ratio": round(five["volume_ratio"], 2),
        "orderbook_imbalance": round(imbalance, 3),
        "funding": funding,
        "oi_change": round(oi_change, 3),
        "hourly_trend": (
            "BULLISH"
            if hourly["ema9"] > hourly["ema21"]
            and hourly["ema21"] > hourly["ema50"]
            else "BEARISH"
            if hourly["ema9"] < hourly["ema21"]
            and hourly["ema21"] < hourly["ema50"]
            else "MIXED"
        ),
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
            five["price"],
            signal["features"],
            five["atr"]
        )

    return data


# ============================================================
# BACKGROUND WORKER
# ============================================================

def background_worker():
    log.info("CoinSentra V16 worker started")
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
                        existing = state["symbols"].get(symbol, {})
                        existing["data_error"] = str(error)[:180]
                        existing["symbol"] = symbol
                        state["symbols"][symbol] = existing

            risk_check()

            with lock:
                state["last_scan"] = now_text()
                state["scan_count"] += 1
                state["last_error"] = None

            save_state()

        except Exception as error:
            log.exception("Worker cycle failed")

            with lock:
                state["last_error"] = str(error)[:250]

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
# PERFORMANCE STATISTICS
# ============================================================

def trade_statistics():
    with lock:
        trades = normalize_history(state["history"])

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

    profit_factor = (
        gross_wins / gross_losses
        if gross_losses > 0
        else gross_wins if gross_wins > 0 else 0.0
    )

    average_win = gross_wins / wins if wins else 0.0
    average_loss = gross_losses / losses if losses else 0.0

    chronological = list(reversed(pnls))
    running_equity = START_BALANCE
    peak_equity = START_BALANCE
    max_drawdown = 0.0

    for pnl in chronological:
        running_equity += pnl
        peak_equity = max(peak_equity, running_equity)

        if peak_equity > 0:
            drawdown = (
                (peak_equity - running_equity)
                / peak_equity * 100
            )
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
        learning = state["learning"]

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
<title>CoinSentra Trading Brain V16</title>
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
.stat strong{display:block;font-size:18px;margin-top:5px;overflow-wrap:anywhere}
.weight{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #293a50}
</style>
</head>
<body>

<h1>COINSENTRA AI</h1>
<div class="small">TRADING BRAIN V16 | BYBIT | PAPER MODE ONLY</div>

<div class="grid">
<div class="card">Realized Balance<div class="big" id="balance">$10,000</div></div>
<div class="card">Total Equity<div class="big" id="equity">$10,000</div></div>
<div class="card">Unrealized PNL<div class="big" id="unrealized">$0.00</div></div>
<div class="card">Daily Net PNL<div class="big" id="daily">$0.00</div></div>
<div class="card">Available Balance<div class="big" id="available">$10,000</div></div>
<div class="card">Open Positions<div class="big" id="count">0</div></div>
</div>

<div class="card">
<h2>CONTROL CENTER</h2>
<button class="stop" onclick="control('stop')">END TRADING (C)</button>
<button class="start" onclick="control('start')">START TRADING (V)</button>
<p id="control" class="small">Loading...</p>
<p class="small">Stopping disables new entries. Existing paper positions remain monitored for TP and SL.</p>
</div>

<div class="card">
<h2>ADAPTIVE AI LEARNING</h2>
<div class="statgrid">
<div class="stat"><span class="small">Eligible samples</span><strong id="learnsamples">0</strong></div>
<div class="stat"><span class="small">Weight updates</span><strong id="learnupdates">0</strong></div>
<div class="stat"><span class="small">Last result</span><strong id="learnresult">Waiting</strong></div>
<div class="stat"><span class="small">Learning status</span><strong id="learnstatus">Collecting</strong></div>
</div>
<p class="small" id="learnmessage">Learning uses completed paper trades with feature data. It needs at least 10 eligible trades before adjusting weights.</p>
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
<thead><tr><th>COIN</th><th>PRICE</th><th>LONG</th><th>SHORT</th><th>SIGNAL</th><th>1H TREND</th></tr></thead>
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
<thead><tr><th>COIN</th><th>SIDE</th><th>ENTRY</th><th>LIVE PRICE</th><th>NET PNL</th><th>MARGIN</th><th>TP</th><th>SL</th></tr></thead>
<tbody id="positions"></tbody>
</table>
</div>
</div>

<div class="card">
<h2>COMPLETED TRADE HISTORY</h2>
<div class="small">Only closed trades appear here. Open trades are shown above.</div>
<div class="scroll">
<table>
<thead><tr><th>CLOSED</th><th>COIN</th><th>SIDE</th><th>ENTRY</th><th>EXIT</th><th>GROSS PNL</th><th>FEES</th><th>NET PNL</th><th>REASON</th></tr></thead>
<tbody id="history"><tr><td colspan="9">No completed trades yet.</td></tr></tbody>
</table>
</div>
</div>

<div class="card small">
Version 16.0 | 5m + 15m + 1h confirmation<br>
ATR stop | 1.8:1 target/risk | 10-minute re-entry cooldown<br>
Paper trading only | Fee estimate 0.055% per side<br>
Last scan: <span id="last">Waiting</span><br>
Last error: <span id="error">None</span>
</div>

<script>
async function control(action){
 const message=document.getElementById('control');

 if(action==='stop'){
  if(!confirm('End trading? New positions stop opening. Existing positions remain monitored.'))return;
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
  if(!r.ok)throw new Error('API response error');
  let d=await r.json();

  document.getElementById('balance').textContent=money(d.balance);
  document.getElementById('equity').textContent=money(d.equity);
  document.getElementById('unrealized').textContent=money(d.unrealized);
  document.getElementById('daily').textContent=money(d.daily_pnl);
  document.getElementById('available').textContent=money(d.available_balance);
  document.getElementById('count').textContent=d.positions.length;

  document.getElementById('control').textContent=
   d.running?'TRADING ACTIVE - NEW ENTRIES ENABLED':'TRADING ENDED - MONITORING OPEN POSITIONS';

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
  document.getElementById('averages').textContent=
   money(s.average_win)+' / '+money(s.average_loss);
  document.getElementById('drawdown').textContent=s.max_drawdown+'%';

  let learning=d.learning;

  document.getElementById('learnsamples').textContent=
   learning.samples+'/'+learning.minimum_samples+' minimum';
  document.getElementById('learnupdates').textContent=learning.updates;
  document.getElementById('learnresult').textContent=learning.last_result||'Waiting';
  document.getElementById('learnstatus').textContent=
   learning.samples>=learning.minimum_samples?'ADAPTIVE':'COLLECTING';

  document.getElementById('learnmessage').textContent=
   learning.samples>=learning.minimum_samples
   ?'Weights adjust using recent eligible completed trades. Last update: '+(learning.last_update||'Not available')
   :'The engine needs '+Math.max(0,learning.minimum_samples-learning.samples)+' more eligible completed trades before its first adjustment.';

  let weightRows='';
  for(let [name,value] of Object.entries(learning.weights||{})){
   weightRows+=`<div class="weight"><span>${safe(name.replaceAll('_',' ').toUpperCase())}</span><b>${Number(value).toFixed(3)}x</b></div>`;
  }
  document.getElementById('weights').innerHTML=weightRows||'Waiting...';

  document.getElementById('scanner').textContent=d.last_scan?'ACTIVE':'WAITING';

  let rows='';
  let details='';

  for(let s of Object.values(d.symbols)){
   rows+=`<tr>
   <td>${safe(s.symbol)}</td><td>${price(s.price)}</td>
   <td class="green">${s.long_score??0}</td>
   <td class="red">${s.short_score??0}</td>
   <td class="${color(s.signal)}">${safe(s.signal||'WAIT')}</td>
   <td>${safe(s.hourly_trend||'-')}</td></tr>`;

   details+=`<p><b>${safe(s.symbol)}</b> |
   RSI ${s.rsi??'-'} | ADX ${s.adx??'-'} |
   ATR ${price(s.atr)} |
   Volume ${s.volume_ratio??'-'}x |
   Orderbook ${s.orderbook_imbalance??'-'} |
   Funding ${s.funding??'-'} |
   OI ${s.oi_change??'-'}%
   ${s.data_error?' | ERROR: '+safe(s.data_error):''}</p>`;
  }

  document.getElementById('markets').innerHTML=
   rows||'<tr><td colspan="6">Waiting for first market scan</td></tr>';

  document.getElementById('indicators').innerHTML=
   details||'Waiting for market data';

  let positionRows='';
  for(let p of d.positions){
   positionRows+=`<tr>
    <td>${safe(p.symbol)}</td><td>${safe(p.side)}</td>
    <td>${price(p.entry)}</td><td>${price(p.current_price)}</td>
    <td class="${pnlColor(p.unrealized_pnl)}">${money(p.unrealized_pnl)}</td>
    <td>${money(p.margin)}</td><td>${price(p.take_profit)}</td>
    <td>${price(p.stop_loss)}</td></tr>`;
  }

  document.getElementById('positions').innerHTML=
   positionRows||'<tr><td colspan="8">No open positions</td></tr>';

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

  document.getElementById('history').innerHTML=
   historyRows||'<tr><td colspan="9">No completed trades yet.</td></tr>';

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
# WEB ROUTES
# ============================================================

@app.route("/")
def home():
    start_worker_once()
    return render_template_string(HTML)


@app.route("/health")
def health():
    start_worker_once()

    return jsonify({
        "status": "ok",
        "version": VERSION,
        "paper_trading": True
    })


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
        positions = [dict(p) for p in state["positions"]]
        history = normalize_history(state["history"])
        symbols = dict(state["symbols"])
        balance = number(state["balance"])
        daily_pnl = number(state["daily_pnl"])
        running = state["running"]
        last_scan = state["last_scan"]
        private_status = state["private_status"]
        account_balance = state["account_balance"]
        last_error = state["last_error"]

    unrealized = sum(
        number(p.get("unrealized_pnl")) for p in positions
    )
    used_margin = sum(
        number(p.get("margin")) for p in positions
    )

    totals = {
        "balance": round(balance, 4),
        "unrealized": round(unrealized, 4),
        "equity": round(balance + unrealized, 4),
        "available_balance": round(balance - used_margin, 4),
        "used_margin": round(used_margin, 4)
    }

    return jsonify({
        **totals,
        "daily_pnl": round(daily_pnl + unrealized, 4),
        "positions": positions,
        "history": history[:100],
        "symbols": symbols,
        "running": running,
        "last_scan": last_scan,
        "private_status": private_status,
        "account_balance": account_balance,
        "last_error": last_error,
        "stats": trade_statistics(),
        "learning": learning_statistics()
    })


@app.route("/api/history")
def api_history():
    with lock:
        history = normalize_history(state["history"])

    return jsonify({
        "count": len(history),
        "history": history[:MAX_HISTORY],
        "stats": trade_statistics()
    })


@app.route("/control/stop", methods=["POST"])
@app.route("/control/end", methods=["POST"])
def stop():
    with lock:
        state["running"] = False

    save_state()

    log.warning(
        "TRADING ENDED BY USER - open positions remain monitored"
    )

    return jsonify({
        "status": "stopped",
        "message": (
            "TRADING ENDED: new entries disabled. "
            "Existing positions are still monitored."
        )
    })


@app.route("/control/start", methods=["POST"])
def start():
    with lock:
        if state["daily_pnl"] <= -DAILY_LOSS_LIMIT:
            return jsonify({
                "status": "blocked",
                "error": (
                    "Daily loss limit reached. "
                    "Trading remains stopped."
                )
            }), 403

        state["running"] = True

    save_state()

    log.info("TRADING STARTED BY USER")

    return jsonify({
        "status": "started",
        "message": "TRADING STARTED: new paper entries enabled."
    })


# ============================================================
# STARTUP
# ============================================================

load_state()

if __name__ == "__main__":
    start_worker_once()

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
