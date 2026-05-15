"""
dashboard.py — Real-time CardioEdge monitoring dashboard.
─────────────────────────────────────────────────────────
Opens a browser dashboard showing:
  • Live ECG waveform (8 seconds scrolling)
  • Heart rate gauge
  • Classification bar chart (live probabilities)
  • Per-class alert history
  • Running accuracy stats (if ground truth available)

Usage:
  python deployment/dashboard.py --port COM3
  # Then open http://localhost:5000 in browser

Requires: pip install flask flask-socketio pyserial
"""
import sys, os, threading, time, argparse, json
sys.path.insert(0, "c:/p3/cardiac-edge-ai")
from collections import deque
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from config import ECG_FS, WINDOW_LEN, TFLITE_DIR, CLASSES

parser = argparse.ArgumentParser()
parser.add_argument("--port",  default="COM3", help="Arduino serial port")
parser.add_argument("--baud",  default=115200, type=int)
parser.add_argument("--host",  default="127.0.0.1")
parser.add_argument("--wport", default=5000, type=int, help="Web server port")
args = parser.parse_args()

# ── Flask + SocketIO ──────────────────────────────────────────────────────────
try:
    from flask import Flask, render_template_string
    from flask_socketio import SocketIO
except ImportError:
    print("[ERR] Missing packages. Run: pip install flask flask-socketio")
    sys.exit(1)

app  = Flask(__name__)
sock = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Shared state ──────────────────────────────────────────────────────────────
ECG_DISPLAY = ECG_FS * 8
ecg_buf      = deque([0] * ECG_DISPLAY, maxlen=ECG_DISPLAY)
filt_buf     = deque([0.0] * ECG_DISPLAY, maxlen=ECG_DISPLAY)
hr_history   = deque([0] * 60, maxlen=60)
beat_history = deque(maxlen=100)   # (timestamp, class, conf, probs)
state = {
    "connected": False, "hr": 0, "beats": 0, "alerts": 0,
    "last_class": -1, "last_conf": 0, "last_probs": [0]*5
}
lock = threading.Lock()

# ── TFLite interpreter ────────────────────────────────────────────────────────
import tensorflow as tf
_model = os.path.join(TFLITE_DIR, "student_kd_int8.tflite")
_interp = tf.lite.Interpreter(model_path=_model)
_interp.allocate_tensors()
_inp = _interp.get_input_details()[0]
_out = _interp.get_output_details()[0]

def tflite_infer(window: np.ndarray):
    x = window[np.newaxis, :, np.newaxis].astype(np.float32)
    _interp.set_tensor(_inp["index"], x); _interp.invoke()
    return _interp.get_tensor(_out["index"])[0].tolist()

# ── Bandpass + Pan-Tompkins ───────────────────────────────────────────────────
_sos = butter(2, [0.5, 40.0], btype="bandpass", fs=ECG_FS, output="sos")
_zi  = sosfilt_zi(_sos) * 0.0
_raw_buf = deque(maxlen=ECG_DISPLAY * 2)
_pt_deriv_prev = 0.0
_pt_mwi = deque([0.0] * 30, maxlen=30)
_pt_sig  = 0.2; _pt_noise = 0.1; _pt_thresh = 0.2; _pt_refrac = 0
_pt_peaks = []; _pt_n = 0

def _pt_step(x):
    global _pt_deriv_prev, _pt_sig, _pt_noise, _pt_thresh, _pt_refrac, _pt_n
    deriv = x - _pt_deriv_prev; _pt_deriv_prev = x
    sq = deriv * deriv
    _pt_mwi.append(sq)
    mwi = sum(_pt_mwi) / len(_pt_mwi)
    detected = False
    if _pt_refrac > 0:
        _pt_refrac -= 1
    elif mwi > _pt_thresh:
        _pt_sig   = 0.125*mwi + 0.875*_pt_sig
        _pt_thresh = _pt_noise + 0.25*(_pt_sig - _pt_noise)
        _pt_refrac = int(0.2 * ECG_FS)
        _pt_peaks.append(_pt_n)
        detected = True
    else:
        _pt_noise  = 0.125*mwi + 0.875*_pt_noise
        _pt_thresh = _pt_noise + 0.25*(_pt_sig - _pt_noise)
    _pt_n += 1
    return detected

def process_sample(raw: int):
    global _zi
    norm = (raw - 512.0) / 512.0
    filt_arr, _zi = sosfilt(_sos, [norm], zi=_zi)
    filt = float(filt_arr[0])
    _raw_buf.append(raw)

    detected = _pt_step(filt)
    with lock:
        ecg_buf.append(raw)
        filt_buf.append(filt)

    if detected and len(_pt_peaks) >= 1:
        pk = _pt_peaks[-1]
        if pk >= WINDOW_LEN // 2 and pk + WINDOW_LEN // 2 < len(_raw_buf):
            raw_arr = list(_raw_buf)
            start = pk - WINDOW_LEN // 2
            window = np.array(raw_arr[start:start + WINDOW_LEN], dtype=float)
            mean = window.mean(); std = window.std() + 1e-8
            window = (window - mean) / std
            probs = tflite_infer(window)
            cls = int(np.argmax(probs)); conf = probs[cls] * 100
            hr = 0
            if len(_pt_peaks) >= 2:
                rr = _pt_peaks[-1] - _pt_peaks[-2]
                hr = round(60 * ECG_FS / max(rr, 1))
            with lock:
                state["last_class"] = cls; state["last_conf"] = conf
                state["last_probs"] = probs; state["hr"] = hr; state["beats"] += 1
                hr_history.append(hr)
                if cls in [1, 2, 3, 4]: state["alerts"] += 1
                beat_history.append({
                    "t": time.time(), "cls": cls,
                    "name": CLASSES[cls], "conf": round(conf,1),
                    "probs": [round(p,3) for p in probs], "hr": hr
                })
            return cls, conf, probs
    return None, None, None

# ── Serial reader thread ──────────────────────────────────────────────────────
def serial_thread():
    try:
        import serial
        ser = serial.Serial(args.port, args.baud, timeout=1)
        with lock: state["connected"] = True
        print(f"[Dashboard] Serial connected: {args.port}")
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line and line.lstrip("-").isdigit():
                process_sample(int(line))
    except Exception as e:
        print(f"[Dashboard] Serial error: {e}")
        with lock: state["connected"] = False

# ── SocketIO event emitter ────────────────────────────────────────────────────
def emit_loop():
    while True:
        time.sleep(0.1)
        with lock:
            snap = dict(state)
            ecg_slice = list(ecg_buf)[-ECG_FS * 4:]   # 4 s of data
            filt_slice = list(filt_buf)[-ECG_FS * 4:]
            hr_hist = list(hr_history)
            alerts  = [b for b in list(beat_history) if b["cls"] != 0][-10:]
        sock.emit("update", {
            "ecg":  ecg_slice[::2],      # downsample for network
            "filt": filt_slice[::2],
            "hr_hist": hr_hist,
            "state": snap,
            "alerts": alerts,
        })

# ── HTML dashboard ────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CardioEdge Live Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:     #0d0f1a; --card:   #151827; --border: #252840;
  --text:   #e2e6ff; --muted:  #6b7296; --accent: #5c6ef8;
  --green:  #2dd4bf; --yellow: #fbbf24; --red:    #f87171;
  --purple: #a78bfa;
}
body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif;
       min-height: 100vh; padding: 1.2rem; }
header { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.4rem; }
.logo { font-size: 1.5rem; font-weight: 700; background: linear-gradient(135deg,#5c6ef8,#2dd4bf);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.badge { padding: .25rem .7rem; border-radius: 99px; font-size: .75rem; font-weight: 600;
         letter-spacing: .04em; }
.badge-ok  { background: rgba(45,212,191,.15); color: var(--green); }
.badge-err { background: rgba(248,113,113,.15); color: var(--red);  }
.grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
@media (max-width:900px){ .grid { grid-template-columns: 1fr; } }
.card { background: var(--card); border: 1px solid var(--border); border-radius: .875rem;
        padding: 1.1rem 1.3rem; }
.card h2 { font-size: .8rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
           color: var(--muted); margin-bottom: .8rem; }
.ecg-wrap { grid-column: 1 / -1; }
.metric-big { font-size: 3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.metric-unit { font-size: 1rem; color: var(--muted); margin-left: .3rem; }
.cls-name { font-size: 1.5rem; font-weight: 700; }
.cls-normal { color: var(--green); }
.cls-af     { color: var(--yellow); }
.cls-vt     { color: var(--red); }
.cls-pvc    { color: var(--yellow); }
.cls-lbbb   { color: var(--purple); }
.prob-bar-wrap { margin-top: .5rem; }
.prob-row { display: flex; align-items: center; gap: .6rem; margin: .25rem 0; font-size: .8rem; }
.prob-label { width: 2.5rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.prob-track { flex: 1; background: var(--border); border-radius: 99px; height: 8px; overflow: hidden; }
.prob-fill  { height: 100%; border-radius: 99px; transition: width .3s ease; }
.prob-val   { width: 3rem; text-align: right; color: var(--muted); }
.alert-list { max-height: 220px; overflow-y: auto; }
.alert-item { display: flex; justify-content: space-between; align-items: center;
              padding: .45rem .6rem; margin: .2rem 0; border-radius: .4rem;
              background: rgba(248,113,113,.08); border-left: 3px solid var(--red);
              font-size: .8rem; }
.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .7rem; margin-top: .3rem; }
.stat-box { background: rgba(255,255,255,.03); border-radius: .5rem; padding: .7rem;
            text-align: center; }
.stat-val { font-size: 1.4rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.stat-lbl { font-size: .7rem; color: var(--muted); margin-top: .2rem; }
canvas { width: 100% !important; }
.no-data { color: var(--muted); font-size: .85rem; padding: 2rem; text-align: center; }
</style>
</head>
<body>
<header>
  <div class="logo">CardioEdge</div>
  <span id="connBadge" class="badge badge-err">● Disconnected</span>
  <span style="margin-left:auto;color:var(--muted);font-size:.8rem">
    5-class Arrhythmia Detector  |  TFLite INT8  |  Arduino Nano 33 BLE
  </span>
</header>

<div class="grid">
  <!-- ECG Row -->
  <div class="card ecg-wrap">
    <h2>Live ECG — Bandpass Filtered (0.5–40 Hz)</h2>
    <canvas id="ecgChart" height="80"></canvas>
  </div>

  <!-- HR -->
  <div class="card">
    <h2>Heart Rate</h2>
    <div class="metric-big" id="hrVal">—</div>
    <span class="metric-unit">bpm</span>
    <canvas id="hrChart" height="80" style="margin-top:.8rem"></canvas>
  </div>

  <!-- Classification -->
  <div class="card">
    <h2>Last Classification</h2>
    <div id="clsName" class="cls-name cls-normal">—</div>
    <div id="clsConf" style="color:var(--muted);font-size:.85rem;margin:.3rem 0">Waiting …</div>
    <div class="prob-bar-wrap" id="probBars"></div>
  </div>

  <!-- Stats -->
  <div class="card">
    <h2>Session Statistics</h2>
    <div class="stats-grid">
      <div class="stat-box"><div class="stat-val" id="statBeats">0</div><div class="stat-lbl">Beats</div></div>
      <div class="stat-box"><div class="stat-val" id="statAlerts">0</div><div class="stat-lbl">Alerts</div></div>
    </div>
  </div>

  <!-- Alert History -->
  <div class="card" style="grid-column: 2 / -1;">
    <h2>Alert History (non-normal beats)</h2>
    <div class="alert-list" id="alertList">
      <div class="no-data">No alerts yet — monitoring …</div>
    </div>
  </div>
</div>

<script>
const CLASSES = ["N","AF","VT","PVC","LBBB"];
const CLS_COLOR = ["#2dd4bf","#fbbf24","#f87171","#fbbf24","#a78bfa"];
const CLS_CSS   = ["cls-normal","cls-af","cls-vt","cls-pvc","cls-lbbb"];

// ECG Chart
const ecgCtx = document.getElementById("ecgChart").getContext("2d");
const ecgData = { labels: [], datasets:[{
  label:"Filtered ECG", data:[], borderColor:"#5c6ef8", borderWidth:1.2,
  pointRadius:0, tension:0.2, fill:false
}]};
const ecgChart = new Chart(ecgCtx, { type:"line", data:ecgData,
  options:{ animation:false, plugins:{legend:{display:false}},
    scales:{ x:{display:false}, y:{min:-2,max:2, grid:{color:"#1e2236"},
    ticks:{color:"#6b7296",font:{size:10}}} } }});

// HR Chart
const hrCtx  = document.getElementById("hrChart").getContext("2d");
const hrData = { labels: Array.from({length:60},(_,i)=>i), datasets:[{
  label:"HR", data:Array(60).fill(0), borderColor:"#2dd4bf", borderWidth:1.5,
  pointRadius:0, tension:0.4, fill:true,
  backgroundColor:"rgba(45,212,191,.08)"
}]};
const hrChart = new Chart(hrCtx, { type:"line", data:hrData,
  options:{ animation:false, plugins:{legend:{display:false}},
    scales:{ x:{display:false}, y:{min:40,max:180, grid:{color:"#1e2236"},
    ticks:{color:"#6b7296",font:{size:10}}} } }});

const socket = io();
socket.on("update", d => {
  const s = d.state;

  // Connection badge
  const badge = document.getElementById("connBadge");
  badge.textContent = s.connected ? "● Connected" : "● Disconnected";
  badge.className   = "badge " + (s.connected ? "badge-ok" : "badge-err");

  // ECG waveform
  const filt = d.filt;
  ecgData.labels = filt.map((_,i)=>i);
  ecgData.datasets[0].data = filt;
  ecgChart.update("none");

  // HR
  document.getElementById("hrVal").textContent = s.hr || "—";
  hrData.datasets[0].data = d.hr_hist;
  hrChart.update("none");

  // Stats
  document.getElementById("statBeats").textContent  = s.beats;
  document.getElementById("statAlerts").textContent = s.alerts;

  // Classification
  if (s.last_class >= 0) {
    const cls = s.last_class;
    const el  = document.getElementById("clsName");
    el.textContent  = CLASSES[cls];
    el.className    = "cls-name " + CLS_CSS[cls];
    document.getElementById("clsConf").textContent =
      `Confidence: ${s.last_conf.toFixed(1)}%`;

    // Probability bars
    const bars = document.getElementById("probBars");
    bars.innerHTML = s.last_probs.map((p,i) => `
      <div class="prob-row">
        <span class="prob-label">${CLASSES[i]}</span>
        <div class="prob-track">
          <div class="prob-fill" style="width:${(p*100).toFixed(1)}%;background:${CLS_COLOR[i]}"></div>
        </div>
        <span class="prob-val">${(p*100).toFixed(0)}%</span>
      </div>`).join("");
  }

  // Alert history
  if (d.alerts && d.alerts.length) {
    const list = document.getElementById("alertList");
    list.innerHTML = d.alerts.slice().reverse().map(a => `
      <div class="alert-item">
        <span style="color:${CLS_COLOR[a.cls]};font-weight:700">${a.name}</span>
        <span>conf ${a.conf}%</span>
        <span style="color:var(--muted)">${a.hr} bpm</span>
        <span style="color:var(--muted)">${new Date(a.t*1000).toLocaleTimeString()}</span>
      </div>`).join("");
  }
});
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

if __name__ == "__main__":
    # Start serial reader
    t_ser = threading.Thread(target=serial_thread, daemon=True)
    t_ser.start()
    # Start emitter
    t_emit = threading.Thread(target=emit_loop, daemon=True)
    t_emit.start()

    print(f"\n{'='*55}")
    print(f"  CardioEdge Dashboard")
    print(f"  Open: http://{args.host}:{args.wport}")
    print(f"  Port: {args.port} @ {args.baud} baud")
    print(f"{'='*55}\n")
    sock.run(app, host=args.host, port=args.wport, debug=False, allow_unsafe_werkzeug=True)
