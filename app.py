
import os, json, time, math, threading, logging, hashlib
from datetime import datetime, timezone, date
from flask import Flask, jsonify, render_template_string, request
from pybit.unified_trading import HTTP

# COINSENTRA TRADING BRAIN V16.1 - PAPER TRADING ONLY
# Reconstructed standalone version. No live order methods are used.

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('coinsentra')

SYMBOLS = ['ETHUSDT', 'XRPUSDT', 'SOLUSDT', 'SUIUSDT']
CATEGORY = 'linear'
LEVERAGE = 8
MAX_POSITIONS = 2
DAILY_LOSS_LIMIT = 6.0
START_BALANCE = 10000.0
RISK_PER_TRADE = 3.0
MAX_EQUITY_EXPOSURE = 0.10
SCAN_SECONDS = 30
ENTRY_SCORE = 68
CONFIRMATION_INTERVAL = '15'
HIGHER_INTERVAL = '60'
ATR_PERIOD = 14
ATR_STOP_MULT = 1.5
STOP_MIN = 0.004
STOP_MAX = 0.015
REWARD_RISK = 1.8
COOLDOWN_SECONDS = 600
FEE_RATE = 0.00055
PAPER_TRADING = True

DATA_DIR = os.getenv('DATA_DIR', '/tmp')
os.makedirs(DATA_DIR, exist_ok=True)
STATE_FILE = os.path.join(DATA_DIR, 'coinsentra_v16.json')

MAX_HISTORY = 1000
LEARNING_ENABLED = True
LEARNING_MIN_SAMPLES = 10
LEARNING_WINDOW = 30
LEARNING_MIN_WEIGHT = 0.75
LEARNING_MAX_WEIGHT = 1.25
LEARNING_RATE = 0.15

FEATURES = [
    'trend_fast', 'trend_slow', 'rsi', 'macd', 'volume',
    'orderbook', 'confirmation', 'higher_timeframe', 'adx',
    'funding', 'open_interest'
]

public_api = HTTP(testnet=False, timeout=15)

# Private API is read-only account balance access only.
# No order endpoints are called.
try:
    private_api = HTTP(
        testnet=False,
        api_key=os.getenv('BYBIT_API_KEY', ''),
        api_secret=os.getenv('BYBIT_API_SECRET', ''),
        timeout=15
    ) if os.getenv('BYBIT_API_KEY') and os.getenv('BYBIT_API_SECRET') else None
except Exception:
    private_api = None

state_lock = threading.RLock()
save_lock = threading.Lock()
close_lock = threading.Lock()
worker_lock = threading.Lock()

state = {
    'balance': START_BALANCE,
    'initial_balance': START_BALANCE,
    'positions': {},
    'history': [],
    'daily_pnl': 0.0,
    'daily_date': str(date.today()),
    'running': True,
    'scan_status': 'Starting',
    'last_scan': None,
    'error': '',
    'private_status': 'Not configured',
    'account_balance': None,
    'scan_count': 0,
    'last_closed_by_symbol': {},
    'version': '16.1',
    'learning': {
        'weights': {f: 1.0 for f in FEATURES},
        'samples': [],
        'processed_ids': []
    }
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def now_text():
    return datetime.now(timezone.utc).isoformat()


def num(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def check_result(r):
    if not isinstance(r, dict) or r.get('retCode', 0) != 0:
        raise RuntimeError(
            str(r.get('retMsg', 'API error'))
            if isinstance(r, dict) else 'API response error'
        )
    return r.get('result', {})


def stable_trade_id(t):
    fields = [
        t.get('symbol'), t.get('side'), t.get('opened_at'),
        t.get('closed_at'), t.get('entry'), t.get('exit'),
        t.get('qty'), t.get('pnl')
    ]
    raw = json.dumps(fields, sort_keys=True, default=str).encode()
    return 'T-' + hashlib.sha256(raw).hexdigest()[:16]


def clean_trade(t):
    if not isinstance(t, dict):
        return None

    x = dict(t)

    for k in ('entry', 'exit', 'qty', 'pnl', 'score', 'fees', 'stop', 'target'):
        if k in x:
            x[k] = num(x[k])

    x.setdefault('opened_at', '')
    x.setdefault('closed_at', '')
    x.setdefault('symbol', '?')
    x.setdefault('side', '?')
    x['id'] = str(x.get('id') or stable_trade_id(x))

    return x


def normalize_history(items):
    seen = set()
    out = []

    if not isinstance(items, list):
        items = []

    for item in items:
        t = clean_trade(item)

        if t and t['id'] not in seen:
            seen.add(t['id'])
            out.append(t)

    out.sort(key=lambda t: str(t.get('closed_at', '')), reverse=True)
    return out[:MAX_HISTORY]


def snapshot_state():
    with state_lock:
        return json.loads(json.dumps(state))


def save_state():
    with save_lock:
        snap = snapshot_state()
        tmp = STATE_FILE + '.tmp'

        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(snap, f, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp, STATE_FILE)

        except Exception as e:
            log.exception('State save failed: %s', e)


def load_state():
    if not os.path.exists(STATE_FILE):
        return

    try:
        with open(STATE_FILE, encoding='utf-8') as f:
            data = json.load(f)

        with state_lock:
            for k in [
                'balance', 'initial_balance', 'daily_pnl', 'daily_date',
                'positions', 'running', 'last_scan', 'scan_count',
                'last_closed_by_symbol'
            ]:
                if k in data:
                    state[k] = data[k]

            state['history'] = normalize_history(data.get('history', []))
            state['error'] = ''
            state['version'] = '16.1'

            learn = data.get('learning') or {}

            state['learning']['weights'].update(learn.get('weights', {}))
            state['learning']['processed_ids'] = list(
                set(learn.get('processed_ids', []))
            )
            state['learning']['samples'] = learn.get('samples', [])[-LEARNING_WINDOW:]

        log.info(
            'Loaded state: %s history entries, %s positions',
            len(state['history']), len(state['positions'])
        )

    except Exception:
        log.exception('State load failed')


def daily_reset():
    today = str(date.today())

    with state_lock:
        if state.get('daily_date') != today:
            state['daily_date'] = today
            state['daily_pnl'] = 0.0


# ============================================================
# BYBIT MARKET DATA
# ============================================================

def api_call(method, **kwargs):
    r = method(**kwargs)
    return check_result(r)


def get_price(symbol):
    r = api_call(
        public_api.get_tickers,
        category=CATEGORY,
        symbol=symbol
    )

    rows = r.get('list', [])

    if not rows:
        raise RuntimeError('Ticker not found')

    return num(rows[0].get('lastPrice'))


def candles(symbol, interval, limit=100):
    r = api_call(
        public_api.get_kline,
        category=CATEGORY,
        symbol=symbol,
        interval=interval,
        limit=limit
    )

    rows = r.get('list', [])
    rows = list(reversed(rows))

    return [
        {
            'time': int(x[0]),
            'open': num(x[1]),
            'high': num(x[2]),
            'low': num(x[3]),
            'close': num(x[4]),
            'volume': num(x[5])
        }
        for x in rows
    ]


def orderbook_imbalance(symbol):
    try:
        r = api_call(
            public_api.get_orderbook,
            category=CATEGORY,
            symbol=symbol,
            limit=25
        )

        bids = r.get('b', [])
        asks = r.get('a', [])

        bv = sum(num(x[1]) for x in bids)
        av = sum(num(x[1]) for x in asks)

        return (bv - av) / (bv + av) if bv + av else 0

    except Exception:
        return 0


def funding_rate(symbol):
    try:
        r = api_call(
            public_api.get_tickers,
            category=CATEGORY,
            symbol=symbol
        )

        return num(r.get('list', [{}])[0].get('fundingRate'))

    except Exception:
        return 0


def open_interest(symbol):
    try:
        r = api_call(
            public_api.get_open_interest,
            category=CATEGORY,
            symbol=symbol,
            intervalTime='5min',
            limit=2
        )

        a = r.get('list', [])
        return num(a[0].get('openInterest')) if a else 0

    except Exception:
        return 0


# ============================================================
# INDICATORS
# ============================================================

def ema(values, period):
    if not values:
        return []

    k = 2 / (period + 1)
    out = [values[0]]

    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))

    return out


def rsi(values, period=14):
    if len(values) < period + 1:
        return 50

    gains = []
    losses = []

    for i in range(1, len(values)):
        d = values[i] - values[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))

    ag = sum(gains[-period:]) / period
    al = sum(losses[-period:]) / period

    if al == 0:
        return 100

    return 100 - 100 / (1 + ag / al)


def macd(values):
    if len(values) < 26:
        return 0, 0

    e12 = ema(values, 12)
    e26 = ema(values, 26)
    line = [a - b for a, b in zip(e12, e26)]
    sig = ema(line, 9)

    return line[-1], sig[-1]


def atr(rows, period=14):
    if len(rows) < period + 1:
        return 0

    trs = []

    for i in range(1, len(rows)):
        h = rows[i]['high']
        l = rows[i]['low']
        cprev = rows[i - 1]['close']

        trs.append(max(
            h - l,
            abs(h - cprev),
            abs(l - cprev)
        ))

    return sum(trs[-period:]) / period


def adx_approx(rows, period=14):
    if len(rows) < period + 2:
        return 0

    plus = []
    minus = []
    tr = []

    for i in range(1, len(rows)):
        up = rows[i]['high'] - rows[i - 1]['high']
        dn = rows[i - 1]['low'] - rows[i]['low']

        plus.append(max(up, 0) if up > dn else 0)
        minus.append(max(dn, 0) if dn > up else 0)

        tr.append(max(
            rows[i]['high'] - rows[i]['low'],
            abs(rows[i]['high'] - rows[i - 1]['close']),
            abs(rows[i]['low'] - rows[i - 1]['close'])
        ))

    t = sum(tr[-period:]) or 1
    p = 100 * sum(plus[-period:]) / t
    m = 100 * sum(minus[-period:]) / t

    return abs(p - m) / (p + m) * 100 if p + m else 0


# ============================================================
# SIGNAL SCORING
# ============================================================

def weighted_score(features):
    with state_lock:
        weights = dict(state['learning']['weights'])

    return sum(
        num(v) * num(weights.get(k, 1), 1)
        for k, v in features.items()
    )


def analyze(symbol):
    rows = candles(symbol, '5', 100)
    confirm = candles(symbol, CONFIRMATION_INTERVAL, 100)
    higher = candles(symbol, HIGHER_INTERVAL, 100)

    closes = [r['close'] for r in rows]
    cc = [r['close'] for r in confirm]
    hc = [r['close'] for r in higher]

    price = closes[-1]
    e9 = ema(closes, 9)[-1]
    e21 = ema(closes, 21)[-1]
    e50 = ema(closes, 50)[-1]

    r = rsi(closes)
    m, s = macd(closes)
    a = atr(rows)
    ad = adx_approx(rows)

    vavg = sum(x['volume'] for x in rows[-20:]) / 20
    vr = rows[-1]['volume'] / vavg if vavg else 1

    ob = orderbook_imbalance(symbol)
    fr = funding_rate(symbol)
    oi = open_interest(symbol)

    ce = ema(cc, 20)[-1]
    he = ema(hc, 20)[-1]

    longf = {
        'trend_fast': 1 if e9 > e21 else 0,
        'trend_slow': 1 if e21 > e50 else 0,
        'rsi': 1 if 50 < r < 72 else 0,
        'macd': 1 if m > s else 0,
        'volume': 1 if vr > 1 else 0,
        'orderbook': 1 if ob > 0.05 else 0,
        'confirmation': 1 if cc[-1] > ce else 0,
        'higher_timeframe': 1 if hc[-1] > he else 0,
        'adx': 1 if ad > 18 else 0,
        'funding': 1 if fr < 0.0005 else 0,
        'open_interest': 1 if oi > 0 else 0
    }

    shortf = {k: 0 for k in FEATURES}

    shortf.update({
        'trend_fast': 1 if e9 < e21 else 0,
        'trend_slow': 1 if e21 < e50 else 0,
        'rsi': 1 if 28 < r < 50 else 0,
        'macd': 1 if m < s else 0,
        'volume': 1 if vr > 1 else 0,
        'orderbook': 1 if ob < -0.05 else 0,
        'confirmation': 1 if cc[-1] < ce else 0,
        'higher_timeframe': 1 if hc[-1] < he else 0,
        'adx': 1 if ad > 18 else 0,
        'funding': 1 if fr > -0.0005 else 0,
        'open_interest': 1 if oi > 0 else 0
    })

    ls = weighted_score(longf) * 100 / 11
    ss = weighted_score(shortf) * 100 / 11

    side = 'LONG' if ls >= ss else 'SHORT'
    score = max(ls, ss)
    feats = longf if side == 'LONG' else shortf

    stop_pct = max(
        STOP_MIN,
        min(STOP_MAX, (a * ATR_STOP_MULT / price) if price else STOP_MIN)
    )

    return {
        'symbol': symbol,
        'price': price,
        'side': side,
        'score': round(score, 1),
        'long_score': round(ls, 1),
        'short_score': round(ss, 1),
        'features': feats,
        'rsi': round(r, 1),
        'macd': m,
        'atr': a,
        'adx': round(ad, 1),
        'volume_ratio': round(vr, 2),
        'orderbook': ob,
        'funding': fr,
        'stop_pct': stop_pct
    }


# ============================================================
# PAPER POSITION MANAGEMENT
# ============================================================

def position_size(price, stop_pct):
    with state_lock:
        balance = num(state['balance'])
        exposure = sum(
            num(p.get('qty')) * num(p.get('entry'))
            for p in state['positions'].values()
        )

    risk_cash = max(0, min(RISK_PER_TRADE, balance * 0.01))
    stop_dist = price * stop_pct
    qty = risk_cash / stop_dist if stop_dist > 0 else 0

    qty = min(
        qty,
        (balance * MAX_EQUITY_EXPOSURE - exposure) / price
        if price > 0 else 0
    )

    return max(0, qty)


def cooldown_ok(symbol):
    with state_lock:
        last = num(state['last_closed_by_symbol'].get(symbol, 0))

    return time.time() - last >= COOLDOWN_SECONDS


def open_position(sig):
    symbol = sig['symbol']
    side = sig['side']
    price = sig['price']
    stop_pct = sig['stop_pct']
    qty = position_size(price, stop_pct)

    if qty <= 0:
        return

    stop = price * (1 - stop_pct if side == 'LONG' else 1 + stop_pct)
    target = price * (
        1 + stop_pct * REWARD_RISK
        if side == 'LONG'
        else 1 - stop_pct * REWARD_RISK
    )

    pos = {
        'id': 'P-' + hashlib.sha256(
            f'{symbol}{side}{time.time()}'.encode()
        ).hexdigest()[:12],
        'symbol': symbol,
        'side': side,
        'entry': price,
        'qty': qty,
        'stop': stop,
        'target': target,
        'score': sig['score'],
        'opened_at': now_text(),
        'features': sig['features'],
        'fees': price * qty * FEE_RATE
    }

    with state_lock:
        if symbol in state['positions'] or len(state['positions']) >= MAX_POSITIONS:
            return

        state['positions'][symbol] = pos

    save_state()
    log.info('Paper position opened %s %s @ %.6f', side, symbol, price)


def close_position(symbol, price, reason='Exit'):
    with close_lock:
        with state_lock:
            p = state['positions'].get(symbol)

            if not p:
                return False

            pos = dict(p)

        gross = (
            (price - pos['entry']) * pos['qty']
            * (1 if pos['side'] == 'LONG' else -1)
        )

        fees = (pos['entry'] * pos['qty'] + price * pos['qty']) * FEE_RATE
        pnl = gross - fees

        trade = {
            'id': pos.get('id'),
            'symbol': symbol,
            'side': pos['side'],
            'entry': pos['entry'],
            'exit': price,
            'qty': pos['qty'],
            'pnl': pnl,
            'fees': fees,
            'score': pos.get('score', 0),
            'opened_at': pos.get('opened_at'),
            'closed_at': now_text(),
            'reason': reason,
            'features': pos.get('features', {})
        }

        trade = clean_trade(trade)

        with state_lock:
            if any(x.get('id') == trade['id'] for x in state['history']):
                return False

            state['history'] = normalize_history([trade] + state['history'])
            state['balance'] += pnl
            state['daily_pnl'] += pnl
            state['positions'].pop(symbol, None)
            state['last_closed_by_symbol'][symbol] = time.time()

            learning_update_locked(trade)

        save_state()
        log.info('Closed %s pnl %.4f', symbol, pnl)
        return True


# ============================================================
# ADAPTIVE LEARNING
# ============================================================

def learning_update_locked(trade):
    learn = state['learning']
    tid = trade['id']

    if tid in learn['processed_ids']:
        return

    feats = trade.get('features') or {}
    pnl = num(trade.get('pnl'))

    if not feats:
        return

    sample = {
        'id': tid,
        'pnl': pnl,
        'features': feats,
        'time': trade.get('closed_at')
    }

    learn['samples'] = (learn.get('samples', []) + [sample])[-LEARNING_WINDOW:]
    learn['processed_ids'] = (learn.get('processed_ids', []) + [tid])[-MAX_HISTORY:]

    if len(learn['samples']) >= LEARNING_MIN_SAMPLES:
        for key in FEATURES:
            wins = [
                x for x in learn['samples']
                if num(x['features'].get(key)) > 0
            ]

            if wins:
                wr = sum(
                    1 for x in wins if num(x['pnl']) > 0
                ) / len(wins)

                w = num(learn['weights'].get(key, 1))
                w = max(
                    LEARNING_MIN_WEIGHT,
                    min(
                        LEARNING_MAX_WEIGHT,
                        w + (wr - 0.5) * LEARNING_RATE
                    )
                )

                learn['weights'][key] = w


# ============================================================
# MONITORING AND SCANNING
# ============================================================

def monitor_positions():
    with state_lock:
        positions = list(state['positions'].items())

    for symbol, p in positions:
        try:
            price = get_price(symbol)

            if (
                p['side'] == 'LONG' and price <= p['stop']
            ) or (
                p['side'] == 'SHORT' and price >= p['stop']
            ):
                close_position(symbol, price, 'Stop loss')

            elif (
                p['side'] == 'LONG' and price >= p['target']
            ) or (
                p['side'] == 'SHORT' and price <= p['target']
            ):
                close_position(symbol, price, 'Take profit')

        except Exception as e:
            log.warning('Position monitor %s: %s', symbol, e)


def scan_once():
    daily_reset()
    monitor_positions()

    with state_lock:
        open_symbols = set(state['positions'])
        running = state['running']
        daily_pnl = state['daily_pnl']

    if not running:
        return

    if daily_pnl <= -DAILY_LOSS_LIMIT:
        with state_lock:
            state['scan_status'] = 'Daily loss limit reached'
        return

    for symbol in SYMBOLS:
        try:
            sig = analyze(symbol)

            if (
                sig['score'] >= ENTRY_SCORE
                and symbol not in open_symbols
                and cooldown_ok(symbol)
            ):
                open_position(sig)

        except Exception as e:
            log.warning('Scan %s: %s', symbol, e)

    with state_lock:
        state['scan_count'] += 1
        state['last_scan'] = now_text()
        state['scan_status'] = 'Scanning'
        state['error'] = ''

    save_state()


def worker():
    while True:
        with state_lock:
            running = state['running']

        try:
            scan_once()

        except Exception as e:
            log.exception('Scanner failure')

            with state_lock:
                state['error'] = str(e)
                state['scan_status'] = 'Error'

            save_state()

        time.sleep(SCAN_SECONDS)


# ============================================================
# READ-ONLY ACCOUNT CONNECTION
# ============================================================

def read_account():
    if not private_api:
        return

    try:
        r = check_result(
            private_api.get_wallet_balance(accountType='UNIFIED')
        )

        rows = r.get('list', [])
        coins = rows[0].get('coin', []) if rows else []
        usdt = next((x for x in coins if x.get('coin') == 'USDT'), {})

        with state_lock:
            state['account_balance'] = num(usdt.get('walletBalance'))
            state['private_status'] = 'Read-only account connected'

    except Exception as e:
        with state_lock:
            state['private_status'] = 'Account read error: ' + str(e)


# ============================================================
# STATISTICS
# ============================================================

def stats():
    with state_lock:
        hist = list(state['history'])
        bal = state['balance']
        pnl = state['balance'] - state['initial_balance']
        positions = list(state['positions'].values())

    wins = [t for t in hist if num(t.get('pnl')) > 0]
    losses = [t for t in hist if num(t.get('pnl')) < 0]

    return {
        'balance': bal,
        'total_pnl': pnl,
        'win_rate': 100 * len(wins) / len(hist) if hist else 0,
        'wins': len(wins),
        'losses': len(losses),
        'trades': len(hist),
        'positions': positions,
        'history': hist[:100]
    }


# ============================================================
# DASHBOARD
# ============================================================

load_state()

worker_started = False


def start_worker_once():
    global worker_started

    with worker_lock:
        if worker_started:
            return

        worker_started = True

        threading.Thread(
            target=worker,
            daemon=True,
            name='CSAI-worker'
        ).start()


start_worker_once()

HTML = '''<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CoinSentra Trading Brain</title>
<style>
body{background:#0b1020;color:#e7ecf7;font-family:Arial;margin:0;padding:16px}
h1{color:#65d9ff}
.card{background:#151d31;border:1px solid #26344e;border-radius:12px;padding:14px;margin:10px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.muted{color:#94a3b8}
button{padding:10px;border:0;border-radius:8px;background:#35b8dd;color:#06111b;font-weight:bold}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:8px;border-bottom:1px solid #27334a;text-align:left;word-break:break-word}
.green{color:#63e6a6}
.red{color:#ff7e87}
</style>
</head>
<body>
<h1>CoinSentra Trading Brain V16.1</h1>
<p class="muted">BYBIT • PAPER TRADING ONLY</p>

<div class="grid" id="tiles"></div>

<div class="card">
<h2>Bot Control</h2>
<button onclick="fetch('/control/start')">Start</button>
<button onclick="fetch('/control/stop')">Pause</button>
<button onclick="fetch('/control/end')">End</button>
<p id="status"></p>
</div>

<div class="card">
<h2>Open Positions</h2>
<div id="positions"></div>
</div>

<div class="card">
<h2>Completed Trades</h2>
<div id="history"></div>
</div>

<div class="card">
<h2>Adaptive Learning</h2>
<div id="learn"></div>
</div>

<script>
function table(rows,cols){
    if(!rows.length)return 'No records yet';

    return '<table><tr>' +
        cols.map(c=>'<th>'+c[0]+'</th>').join('') +
        '</tr>' +
        rows.map(r=>'<tr>' +
            cols.map(c=>'<td>'+((r[c[1]]??'')+'')+'</td>').join('') +
            '</tr>').join('') +
        '</table>';
}

async function refresh(){
    let d=await(await fetch('/api/data')).json();

    document.getElementById('tiles').innerHTML=[
        ['Paper Balance',d.balance.toFixed(2)],
        ['Total PnL',d.total_pnl.toFixed(2)],
        ['Win Rate',d.win_rate.toFixed(1)+'%'],
        ['Trades',d.trades],
        ['Open Positions',d.positions.length],
        ['Scanner',d.scan_status]
    ].map(x=>'<div class="card"><div class="muted">'+x[0]+'</div><h2>'+x[1]+'</h2></div>').join('');

    document.getElementById('status').innerText=
        'Status: '+d.scan_status+
        ' • Last scan: '+(d.last_scan||'--')+
        ' • '+(d.error||'');

    document.getElementById('positions').innerHTML=table(
        d.positions,
        [['Symbol','symbol'],['Side','side'],['Entry','entry'],
        ['Qty','qty'],['Stop','stop'],['Target','target']]
    );

    document.getElementById('history').innerHTML=table(
        d.history,
        [['Closed','closed_at'],['Symbol','symbol'],['Side','side'],
        ['Entry','entry'],['Exit','exit'],['PnL','pnl'],['Reason','reason']]
    );

    document.getElementById('learn').innerHTML=
        'Samples: '+d.learning.samples+
        '<br>Weights: '+JSON.stringify(d.learning.weights);
}

refresh();
setInterval(refresh,5000);
</script>
</body>
</html>'''


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def home():
    return render_template_string(HTML)


@app.route('/health')
def health():
    return jsonify({'ok': True, 'version': '16.1'})


@app.route('/api/status')
@app.route('/status')
def status():
    with state_lock:
        return jsonify({
            'running': state['running'],
            'status': state['scan_status'],
            'error': state['error'],
            'last_scan': state['last_scan']
        })


@app.route('/api/data')
def data():
    s = stats()

    with state_lock:
        s.update({
            'scan_status': state['scan_status'],
            'last_scan': state['last_scan'],
            'error': state['error'],
            'scan_count': state['scan_count'],
            'learning': {
                'samples': len(state['learning']['samples']),
                'weights': state['learning']['weights']
            }
        })

    return jsonify(s)


@app.route('/api/history')
def history():
    with state_lock:
        return jsonify(state['history'])


@app.route('/control/start')
def start():
    with state_lock:
        state['running'] = True
        state['scan_status'] = 'Running'

    save_state()
    return jsonify({'ok': True, 'running': True})


@app.route('/control/stop')
def stop():
    with state_lock:
        state['running'] = False
        state['scan_status'] = 'Paused'

    save_state()
    return jsonify({'ok': True, 'running': False})


@app.route('/control/end')
def end():
    return stop()


# ============================================================
# STARTUP
# ============================================================

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', '8080'))
    )
