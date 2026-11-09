import os
import time
import threading
from functools import wraps
from flask import Flask, jsonify, request, Response
import CoinSentra_Trading_Brain_V8 as brain

app = Flask(__name__)
PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")
worker_started = False
worker_lock = threading.Lock()

if not PASSWORD:
    raise RuntimeError("Set DASHBOARD_PASSWORD in Render Environment before starting this private dashboard.")


def require_login(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.username != "coinsentra" or auth.password != PASSWORD:
            return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="CoinSentra Private Dashboard"'})
        return fn(*args, **kwargs)
    return wrapped


def scanner_loop():
    while True:
        try:
            today = str(brain.date.today())
            if brain.state.get("day") != today:
                brain.state["day"] = today
                brain.state["daily_realized"] = 0.0
                brain.save_state()
            brain.run_scan()
        except Exception:
            brain.logging.exception("Render scanner loop error")
        time.sleep(brain.SCAN_SECONDS)


def start_worker():
    global worker_started
    with worker_lock:
        if not worker_started:
            thread = threading.Thread(target=scanner_loop, name="coinsentra-scanner", daemon=True)
            thread.start()
            worker_started = True

@app.before_request
def ensure_worker():
    if request.path != "/health":
        start_worker()

@app.get("/health")
def health():
    return "ok", 200

@app.get("/api/status")
@require_login
def status():
    s = brain.state
    rows = s.get("latest_rows", [])
    positions = s.get("positions", {})
    return jsonify({
        "app": brain.APP_NAME,
        "mode": "PAPER TRADING ONLY",
        "symbols": brain.SYMBOLS,
        "leverage": brain.LEVERAGE,
        "take_profit_pct": brain.TAKE_PROFIT_PCT * 100,
        "stop_loss_pct": brain.STOP_LOSS_PCT * 100,
        "daily_loss_limit": brain.DAILY_LOSS_LIMIT_USD,
        "balance": round(float(s.get("balance", 0)), 2),
        "daily_realized": round(float(s.get("daily_realized", 0)), 3),
        "open_positions": positions,
        "closed_trades": list(reversed(s.get("closed_trades", [])[-20:])),
        "rows": rows,
        "last_scan_at": s.get("last_scan_at", "Waiting for first scan"),
        "scan_count": s.get("scan_count", 0),
    })

@app.get("/")
@require_login
def dashboard():
    return '''<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"><title>CoinSentra Trading Brain V8</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0b1020;color:#e8eefc;font:15px Arial,sans-serif}header{padding:22px 16px;background:#121a30;border-bottom:1px solid #263452}h1{margin:0;font-size:22px}small,.muted{color:#9eacc8}.wrap{max-width:1100px;margin:auto;padding:16px}.pill{display:inline-block;background:#173c35;color:#78e2b5;border-radius:20px;padding:6px 10px;font-size:12px;margin-top:10px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px}.card{background:#121a30;border:1px solid #263452;border-radius:12px;padding:14px;margin-bottom:12px}.k{color:#9eacc8;font-size:12px}.v{font-size:22px;font-weight:bold;margin-top:7px}.tablewrap{overflow-x:auto}table{width:100%;border-collapse:collapse;min-width:680px}th,td{text-align:left;padding:10px;border-bottom:1px solid #263452;font-size:13px}th{color:#9eacc8}.LONG,.BUY{color:#69e0a7}.SHORT,.SELL{color:#ff8585}.WAIT{color:#f4ca72}.section{margin:22px 0 10px;font-size:17px}pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#b9c8e6}button{background:#243b67;color:white;border:0;padding:10px 14px;border-radius:8px}
</style></head><body><header><div class="wrap"><h1>CoinSentra Trading Brain V8</h1><div class="pill">PAPER TRADING ONLY · 8x SIMULATED · WHALE INTELLIGENCE</div></div></header><main class="wrap"><div class="grid">
<div class="card"><div class="k">PAPER BALANCE</div><div class="v" id="balance">—</div></div><div class="card"><div class="k">TODAY REALIZED P&L</div><div class="v" id="pnl">—</div></div><div class="card"><div class="k">OPEN POSITIONS</div><div class="v" id="open">—</div></div><div class="card"><div class="k">SCANS COMPLETED</div><div class="v" id="scans">—</div></div></div>
<p class="muted" id="last">Connecting to scanner…</p><div class="section">Market intelligence</div><div class="card tablewrap"><table><thead><tr><th>Symbol</th><th>Price</th><th>AI score</th><th>Signal</th><th>5m RSI</th><th>15m RSI</th><th>Whale flow</th><th>Buy $</th><th>Sell $</th><th>Book imbalance</th><th>OI change</th></tr></thead><tbody id="markets"><tr><td colspan="11">Waiting for first scan…</td></tr></tbody></table></div>
<div class="section">Open paper positions</div><div class="card tablewrap"><table><thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Take profit</th><th>Stop loss</th><th>Quantity</th></tr></thead><tbody id="positions"><tr><td colspan="6">None</td></tr></tbody></table></div>
<div class="section">Recent closed trades</div><div class="card tablewrap"><table><thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>Net P&L</th><th>Reason</th></tr></thead><tbody id="history"><tr><td colspan="7">No closed trades yet</td></tr></tbody></table></div>
<p class="muted">Symbols: ETH · XRP · SOL · SUI | 5m + 15m | TP 2% / SL 2% | Daily loss stop $6. Data from Bybit public market endpoints. Whale flow is an estimate, not wallet identification.</p><button onclick="loadData()">Refresh dashboard</button></main>
<script>
const money=x=>'$'+Number(x||0).toLocaleString(undefined,{maximumFractionDigits:4});
const price=x=>Number(x||0).toLocaleString(undefined,{maximumFractionDigits:8});
async function loadData(){try{const r=await fetch('/api/status');if(!r.ok)throw Error('Session expired');const d=await r.json();document.getElementById('balance').textContent=money(d.balance);document.getElementById('pnl').textContent=money(d.daily_realized);document.getElementById('open').textContent=Object.keys(d.open_positions).length+'/2';document.getElementById('scans').textContent=d.scan_count;document.getElementById('last').textContent='Last scan: '+d.last_scan_at+' · '+d.mode+' · Daily loss limit: $'+d.daily_loss_limit;
document.getElementById('markets').innerHTML=d.rows.map(x=>`<tr><td><b>${x.symbol}</b></td><td>${price(x.price)}</td><td>${x.score}/100</td><td class="${x.direction}">${x.direction}</td><td>${Number(x.rsi5||0).toFixed(1)}</td><td>${Number(x.rsi15||0).toFixed(1)}</td><td>${x.whale.signal}</td><td>${money(x.whale.buy_usd)}</td><td>${money(x.whale.sell_usd)}</td><td>${Number(x.whale.imbalance||0).toFixed(2)}</td><td>${Number(x.whale.oi_change_pct||0).toFixed(2)}%</td></tr>`).join('')||'<tr><td colspan="11">Waiting for first scan…</td></tr>';
document.getElementById('positions').innerHTML=Object.entries(d.open_positions).map(([s,p])=>`<tr><td>${s}</td><td>${p.side}</td><td>${price(p.entry)}</td><td>${price(p.tp)}</td><td>${price(p.sl)}</td><td>${Number(p.qty).toFixed(6)}</td></tr>`).join('')||'<tr><td colspan="6">No open paper positions</td></tr>';
document.getElementById('history').innerHTML=d.closed_trades.map(t=>`<tr><td>${t.closed_at}</td><td>${t.symbol}</td><td>${t.side}</td><td>${price(t.entry)}</td><td>${price(t.exit)}</td><td>${money(t.net_pnl)}</td><td>${t.reason}</td></tr>`).join('')||'<tr><td colspan="7">No closed trades yet</td></tr>';
}catch(e){document.getElementById('last').textContent='Dashboard error: '+e.message}}
loadData();setInterval(loadData,15000);
</script></body></html>'''

if __name__ == "__main__":
    start_worker()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
