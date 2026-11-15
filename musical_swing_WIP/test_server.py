"""
Test web server for the Pi. Serves a page that shows live MPU6050
accelerometer/gyro readings, so you can check phone -> Pi networking and
the sensor at the same time. Works without the sensor too (page says so).

It also keeps a Bluetooth speaker connected (swing_sim/speaker.py): pick
one of the Pi's paired devices on the page and it's auto-reconnected
whenever it drops. The choice is saved to test_server_speaker.json, so
it's reconnected on the next start too.

    python test_server.py            # then open http://<pi-ip>:8000 on your phone
    python test_server.py --port 8080
    python test_server.py --speaker XX:XX:XX:XX:XX:XX
"""

import argparse
import asyncio
import json
import shutil
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from swing_sim.speaker import SpeakerManager

try:
    import numpy as np
    import sounddevice as sd
    _SD_ERROR = None
except (ImportError, OSError) as e:  # OSError: PortAudio library missing
    sd = None
    _SD_ERROR = str(e)

try:
    from smbus2 import SMBus
except ImportError:
    SMBus = None

ADDR = 0x68
PWR_MGMT_1, ACCEL_CONFIG, GYRO_CONFIG, ACCEL_XOUT_H = 0x6B, 0x1C, 0x1B, 0x3B
ACCEL_LSB_PER_G, GYRO_LSB_PER_DPS = 16384.0, 131.0


class IMU:
    """
    Reads the sensor on a background thread at ~100 Hz and keeps the
    latest sample, so the motion audio and every open page share one
    reader. Opens the sensor lazily and reconnects after errors.
    """

    def __init__(self, bus_num: int = 1, hz: float = 100.0):
        self.bus_num = bus_num
        self.bus = None
        self.period = 1.0 / hz
        self._lock = threading.Lock()
        self._latest = None
        self._latest_at = 0.0
        self._status = "starting"
        self._thread = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="imu", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            data, status = self.read()
            with self._lock:
                self._status = status
                if data is not None:
                    self._latest, self._latest_at = data, time.monotonic()
            time.sleep(self.period if data is not None else 1.0)  # back off while missing

    def get(self, max_age_s: float = 0.5):
        """(data, status); data is None if the last good sample is stale."""
        with self._lock:
            fresh = time.monotonic() - self._latest_at <= max_age_s
            return (self._latest if fresh else None), self._status

    def read(self):
        try:
            if self.bus is None:
                if SMBus is None:
                    return None, "smbus2 not installed"
                self.bus = SMBus(self.bus_num)
                self.bus.write_byte_data(ADDR, PWR_MGMT_1, 0x00)
                self.bus.write_byte_data(ADDR, ACCEL_CONFIG, 0x00)
                self.bus.write_byte_data(ADDR, GYRO_CONFIG, 0x00)
            raw = bytes(self.bus.read_i2c_block_data(ADDR, ACCEL_XOUT_H, 14))
        except OSError as e:
            if self.bus is not None:
                self.bus.close()
            self.bus = None
            return None, f"sensor not available ({e.strerror or e})"
        ax, ay, az, _temp, gx, gy, gz = struct.unpack(">7h", raw)
        return {
            "accel": [round(v / ACCEL_LSB_PER_G, 3) for v in (ax, ay, az)],
            "gyro": [round(v / GYRO_LSB_PER_DPS, 2) for v in (gx, gy, gz)],
        }, "ok"


SPEAKER_FILE = Path(__file__).with_name("test_server_speaker.json")


def load_saved_mac() -> Optional[str]:
    try:
        return json.loads(SPEAKER_FILE.read_text()).get("mac")
    except (OSError, ValueError):
        return None


def save_mac(mac: Optional[str]) -> None:
    SPEAKER_FILE.write_text(json.dumps({"mac": mac}))


def paired_devices() -> list:
    """[{mac, name}] for devices paired with this Pi, via bluetoothctl."""
    exe = shutil.which("bluetoothctl")
    if exe is None:
        return []
    try:
        out = subprocess.run([exe, "devices", "Paired"], capture_output=True,
                             text=True, timeout=5).stdout
    except (subprocess.TimeoutExpired, OSError):
        return []
    devices = []
    for line in out.splitlines():
        parts = line.split(" ", 2)
        if len(parts) >= 2 and parts[0] == "Device":
            devices.append({"mac": parts[1], "name": parts[2] if len(parts) > 2 else parts[1]})
    return devices


_tone_lock = threading.Lock()


def _resonate(x, freqs, bw, sr):
    """Two-pole resonator with a per-sample centre frequency (one formant)."""
    r = np.exp(-np.pi * bw / sr)
    b = 2 * r * np.cos(2 * np.pi * freqs / sr)
    c = -r * r
    a = 1 - b - c  # unity gain at DC keeps the cascade's level sane
    y = np.zeros_like(x)
    y1 = y2 = 0.0
    for n in range(len(x)):
        y0 = a[n] * x[n] + b[n] * y1 + c * y2
        y[n] = y0
        y2, y1 = y1, y0
    return y


def synth_bruh(sr: int = 16000) -> np.ndarray:
    """
    Says "bruh" with a tiny formant synthesizer (no TTS install needed):
    a buzzy voice source, filtered through three vocal-tract resonances
    that glide from an /r/ shape into an "uh" vowel, with a lip-release
    pop at the start and a breathy "h" tail. Pitch falls, for maximum bruh.
    """
    rng = np.random.default_rng(0)
    dur = 0.75
    t = np.arange(int(sr * dur)) / sr

    f0 = np.interp(t, [0, dur], [130, 82])
    saw = (np.cumsum(f0 / sr) % 1.0) * 2 - 1
    voice_amp = np.interp(t, [0, 0.03, 0.06, 0.45, 0.6, dur], [0, 0.15, 1, 1, 0.1, 0])
    breath_amp = np.interp(t, [0, 0.4, 0.52, 0.65, dur], [0, 0, 0.35, 0.2, 0])
    src = saw * voice_amp + rng.normal(0, 1, t.size) * breath_amp

    pop = (t > 0.028) & (t < 0.036)  # the "b": lips releasing
    src[pop] += rng.normal(0, 1.5, pop.sum())

    keys = [0, 0.06, 0.18, dur]  # /r/ ... gliding into /ʌ/ ("uh")
    out = src
    for track, bw in (([300, 350, 640, 600], 90),
                      ([1000, 1050, 1190, 1150], 110),
                      ([1450, 1550, 2390, 2400], 170)):
        out = _resonate(out, np.interp(t, keys, track), bw, sr)

    out -= out.mean()
    return (0.5 * out / np.max(np.abs(out))).astype(np.float32)


PIPEWIRE_DEVICES = ("pipewire", "pulse", "default")


def pick_output_device() -> str:
    """
    The PipeWire ALSA device if present (it routes to Bluetooth), else
    the system default. Re-scans devices so packages/speakers added after
    startup are seen -- but not while the motion stream is open, since a
    re-scan tears down PortAudio under it.
    """
    if not motion.running:
        sd._terminate()
        sd._initialize()
    outputs = [d for d in sd.query_devices() if d["max_output_channels"] > 0]
    device = next((d["name"] for d in outputs if d["name"] in PIPEWIRE_DEVICES), None)
    return device or sd.query_devices(kind="output")["name"]


class MotionAudio:
    """
    Plays a continuous chord driven by how fast the gyro is turning. The
    turning speed becomes a motion amount from 0 (below dead_dps: sensor
    noise/offset at rest) to 1 (full_dps and above), smoothed with a quick
    attack and slow release so it swells with motion rather than flickering.

    Two modes:
      "volume": volume = motion amount, pitch fixed at root_hz.
      "pitch":  volume fixed at 100%, pitch rises from root_hz (at rest)
                up to PITCH_RANGE_OCT octaves higher (at full_dps).

    If the sensor stops delivering, the motion amount drops to 0: silence
    in volume mode, back to the resting pitch in pitch mode.
    """

    SR = 48000
    CHORD_RATIOS = np.array([1.0, 2 ** (4 / 12), 2 ** (7 / 12), 2.0])  # major triad + octave
    ATTACK_S, RELEASE_S = 0.08, 0.6
    GLIDE_S = 0.05  # pitch changes slide over ~50 ms instead of jumping
    GAIN_SMOOTH_S = 0.03  # smooths the jump in volume when switching modes
    MIN_HZ, MAX_HZ = 40.0, 2000.0
    PITCH_RANGE_OCT = 2.0
    MODES = ("volume", "pitch")

    def __init__(self, imu: "IMU"):
        self.imu = imu
        self.mode = "volume"
        self.dead_dps = 8.0
        self.full_dps = 150.0
        self.root_hz = 220.0     # chord root (A3) -- the resting pitch in pitch mode
        self.stream = None
        self.device = None
        self.level = 0.0         # smoothed motion amount 0..1, owned by the audio callback
        self.gain = 0.0          # output volume 0..1, owned by the audio callback
        self.rate_dps = 0.0      # latest gyro speed, for the page
        self._root_now = self.root_hz  # current (gliding) pitch, owned by the callback
        self._phase = np.zeros(len(self.CHORD_RATIOS))

    @property
    def running(self) -> bool:
        return self.stream is not None

    def target_hz(self, level: float) -> float:
        if self.mode == "pitch":
            return self.root_hz * 2 ** (self.PITCH_RANGE_OCT * level)
        return self.root_hz

    def motion_amount(self) -> float:
        data, _ = self.imu.get()
        if data is None:
            self.rate_dps = 0.0
            return 0.0
        self.rate_dps = float(np.linalg.norm(data["gyro"]))
        x = (self.rate_dps - self.dead_dps) / max(1e-6, self.full_dps - self.dead_dps)
        return float(min(1.0, max(0.0, x)))

    def _callback(self, outdata, frames, time_info, status) -> None:
        dt = frames / self.SR
        target = self.motion_amount()
        tau = self.ATTACK_S if target > self.level else self.RELEASE_S
        self.level += (target - self.level) * (1 - np.exp(-dt / tau))

        g0 = self.gain
        want_gain = self.level if self.mode == "volume" else 1.0
        g1 = g0 + (want_gain - g0) * (1 - np.exp(-dt / self.GAIN_SMOOTH_S))
        self.gain = g1

        r0 = self._root_now
        r1 = r0 + (self.target_hz(self.level) - r0) * (1 - np.exp(-dt / self.GLIDE_S))
        self._root_now = r1
        # Per-sample phase steps, so a pitch change is continuous (no clicks).
        steps = 2 * np.pi * np.outer(self.CHORD_RATIOS, np.linspace(r0, r1, frames)) / self.SR
        phases = self._phase[:, None] + np.cumsum(steps, axis=1) - steps
        tone = np.sin(phases).sum(axis=0) / len(self.CHORD_RATIOS)
        self._phase = (phases[:, -1] + steps[:, -1]) % (2 * np.pi)
        ramp = np.linspace(g0, g1, frames, endpoint=False)  # no zipper noise between blocks
        outdata[:, 0] = (0.35 * ramp * tone).astype(np.float32)

    def start(self) -> None:
        if self.running:
            return
        if sd is None:
            raise RuntimeError(f"sounddevice unavailable: {_SD_ERROR}")
        self.device = pick_output_device()
        try:
            stream = sd.OutputStream(samplerate=self.SR, channels=1, dtype="float32",
                                     blocksize=512, device=self.device, callback=self._callback)
            stream.start()
        except sd.PortAudioError as e:
            raise RuntimeError(f"Could not open {self.device}: {e}")
        self.stream = stream

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.gain = self.level = 0.0  # next start fades in from silence
        self._root_now = self.root_hz

    def state(self) -> dict:
        playing_hz = self._root_now if self.running else self.target_hz(self.level)
        return {"enabled": self.running, "mode": self.mode,
                "gain": round(self.gain, 3), "level": round(self.level, 3),
                "rate_dps": round(self.rate_dps, 1), "full_dps": self.full_dps,
                "root_hz": round(self.root_hz, 2), "playing_hz": round(playing_hz, 1),
                "top_hz": round(self.root_hz * 2 ** self.PITCH_RANGE_OCT, 1),
                "device": self.device}


def play_test_tone() -> str:
    """
    Says "bruh" and returns the output device used. Prefers the PipeWire
    ALSA device, because that is what routes to a Bluetooth speaker; the
    raw hw: devices only reach HDMI. Raises RuntimeError with a readable
    reason on failure.
    """
    if sd is None:
        raise RuntimeError(f"sounddevice unavailable: {_SD_ERROR}")
    if not _tone_lock.acquire(blocking=False):
        raise RuntimeError("A test tone is already playing")
    device = None
    try:
        device = pick_output_device()
        sr = 16000
        sd.play(synth_bruh(sr), samplerate=sr, device=device, blocking=True)
        return device
    except sd.PortAudioError as e:
        hint = ("" if device in PIPEWIRE_DEVICES else
                f" -- '{device}' is a raw hardware output; install pipewire-alsa "
                "so audio goes through PipeWire (and on to Bluetooth)")
        raise RuntimeError(f"Audio output failed: {e}{hint}")
    finally:
        _tone_lock.release()


app = FastAPI()
imu = IMU()
motion = MotionAudio(imu)
speaker = SpeakerManager(speaker_mac=None, auto_start=False)  # started in __main__

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pi test server</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 20px; background: #111; color: #eee; }
  h1 { font-size: 1.3rem; margin: 0 0 4px; }
  #status { font-size: .9rem; color: #aaa; margin-bottom: 20px; }
  .ok { color: #6c6 !important; } .bad { color: #e77 !important; }
  table { border-collapse: collapse; width: 100%; max-width: 420px; font-variant-numeric: tabular-nums; }
  th, td { padding: 10px 8px; text-align: right; border-bottom: 1px solid #333; font-size: 1.1rem; }
  th:first-child, td:first-child { text-align: left; color: #aaa; font-weight: normal; }
</style></head>
<body>
<h1>Pi test server</h1>
<div id="status">connecting...</div>
<table>
  <tr><th></th><th>X</th><th>Y</th><th>Z</th></tr>
  <tr><td>Accel (g)</td><td id="a0">-</td><td id="a1">-</td><td id="a2">-</td></tr>
  <tr><td>Gyro (°/s)</td><td id="g0">-</td><td id="g1">-</td><td id="g2">-</td></tr>
</table>

<h1 style="margin-top:32px">Motion audio</h1>
<div class="seg" role="radiogroup">
  <button id="mo-mode-volume" data-mode="volume" class="on">Movement &rarr; volume</button>
  <button id="mo-mode-pitch" data-mode="pitch">Movement &rarr; pitch</button>
</div>
<div id="mo-desc" style="color:#aaa;font-size:.9rem;max-width:420px;margin-top:8px"></div>
<div class="btns"><button id="mo-toggle">Start sound</button></div>
<div class="meter-row"><span>Gyro speed</span><div class="meter"><div id="mo-rate" style="background:#58a"></div></div><span id="mo-rate-n">0 °/s</span></div>
<div class="meter-row"><span>Volume</span><div class="meter"><div id="mo-gain"></div></div><span id="mo-gain-n">0%</span></div>
<div class="meter-row"><span>Pitch now</span><div class="meter"><div id="mo-pitch" style="background:#c8a"></div></div><span id="mo-pitch-n">220 Hz</span></div>
<label style="display:block;margin-top:14px;max-width:420px"><span id="mo-full-label">Full volume</span> at <b id="mo-full-n">150</b> °/s
  <input id="mo-full" type="range" min="20" max="500" step="10" value="150" style="width:100%"></label>
<label style="display:block;margin-top:14px;max-width:420px"><span id="mo-hz-label">Pitch</span> <b id="mo-hz-n">220 Hz (A3)</b>
  <input id="mo-hz" type="range" min="0" max="1000" value="500" style="width:100%"></label>
<div id="mo-msg" class="bad" style="font-size:.9rem;margin-top:6px"></div>
<style>
  .meter-row { display: flex; align-items: center; gap: 10px; max-width: 420px; margin-top: 12px; }
  .meter-row span:first-child { width: 90px; color: #aaa; }
  .meter-row span:last-child { width: 70px; text-align: right; font-variant-numeric: tabular-nums; }
  .meter { flex: 1; height: 14px; background: #222; border-radius: 7px; overflow: hidden; }
  .meter div { height: 100%; width: 0; background: #6c6; transition: width .05s linear; }
  .seg { display: flex; max-width: 420px; margin-top: 10px; }
  .seg button { flex: 1; border-radius: 0; }
  .seg button:first-child { border-radius: 8px 0 0 8px; }
  .seg button:last-child { border-radius: 0 8px 8px 0; border-left: none; }
  .seg button.on { background: #2d4a2d; border-color: #6c6; color: #fff; }
</style>
<script>
let moEnabled = false, fullDps = 150;
const moToggle = document.getElementById("mo-toggle"), moMsg = document.getElementById("mo-msg");
const moFull = document.getElementById("mo-full");

async function setMotion(body) {
  moMsg.textContent = "";
  const r = await fetch("/api/motion", {method: "POST", headers: {"Content-Type": "application/json"},
                                        body: JSON.stringify(body)});
  const j = await r.json();
  if (!r.ok) { moMsg.textContent = j.detail; return; }
  renderMotion(j);
}
moToggle.onclick = () => setMotion({enabled: !moEnabled});
document.querySelectorAll(".seg button").forEach(b => b.onclick = () => setMotion({mode: b.dataset.mode}));
moFull.oninput = () => document.getElementById("mo-full-n").textContent = moFull.value;
moFull.onchange = () => setMotion({full_dps: Number(moFull.value)});

// Pitch slider is logarithmic (each step is the same musical interval), 55-880 Hz = A1..A5.
const moHz = document.getElementById("mo-hz"), HZ_LO = 55, HZ_HI = 880;
const sliderToHz = v => HZ_LO * Math.pow(HZ_HI / HZ_LO, v / 1000);
const hzToSlider = hz => 1000 * Math.log(hz / HZ_LO) / Math.log(HZ_HI / HZ_LO);
const NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"];
function hzLabel(hz) {
  const semis = Math.round(12 * Math.log2(hz / 440));            // from A4
  const name = NOTES[((semis % 12) + 12) % 12];
  const octave = 4 + Math.floor((semis + 9) / 12);              // octaves start at C
  return `${Math.round(hz)} Hz (≈${name}${octave})`;
}
let rootHz = 220, hzTimer = null;
moHz.oninput = () => {
  const hz = sliderToHz(Number(moHz.value));
  document.getElementById("mo-hz-n").textContent = hzLabel(hz);
  clearTimeout(hzTimer);  // send while dragging (throttled), so you hear it slide
  hzTimer = setTimeout(() => setMotion({root_hz: hz}), 60);
};

function renderMotion(m) {
  moEnabled = m.enabled;
  moToggle.textContent = m.enabled ? "Stop sound" : "Start sound";
  const pitchMode = m.mode === "pitch";
  document.querySelectorAll(".seg button").forEach(b => b.classList.toggle("on", b.dataset.mode === m.mode));
  document.getElementById("mo-desc").textContent = pitchMode
    ? `Always at full volume. Pitch rises from ${Math.round(m.root_hz)} Hz at rest to ${Math.round(m.top_hz)} Hz (two octaves up) at full speed.`
    : "Pitch stays put. The faster the sensor turns, the louder it gets; silent at rest.";
  document.getElementById("mo-full-label").textContent = pitchMode ? "Highest pitch" : "Full volume";
  document.getElementById("mo-hz-label").textContent = pitchMode ? "Resting pitch" : "Pitch";
  document.getElementById("mo-gain").style.width = (m.gain * 100) + "%";
  document.getElementById("mo-gain-n").textContent = Math.round(m.gain * 100) + "%";
  const pitchFrac = Math.log(m.playing_hz / m.root_hz) / Math.log(m.top_hz / m.root_hz);
  document.getElementById("mo-pitch").style.width = Math.max(0, Math.min(1, pitchFrac)) * 100 + "%";
  document.getElementById("mo-pitch-n").textContent = Math.round(m.playing_hz) + " Hz";
  document.getElementById("mo-rate").style.width = Math.min(100, m.rate_dps / m.full_dps * 100) + "%";
  document.getElementById("mo-rate-n").textContent = Math.round(m.rate_dps) + " °/s";
  if (document.activeElement !== moFull && m.full_dps !== fullDps) {
    fullDps = m.full_dps; moFull.value = fullDps;
    document.getElementById("mo-full-n").textContent = Math.round(fullDps);
  }
  if (document.activeElement !== moHz && m.root_hz !== rootHz) {
    rootHz = m.root_hz; moHz.value = hzToSlider(rootHz);
    document.getElementById("mo-hz-n").textContent = hzLabel(rootHz);
  }
}
</script>

<h1 style="margin-top:32px">Bluetooth speaker</h1>
<div id="spk-status" style="margin-bottom:12px;color:#aaa">checking...</div>
<select id="spk-select"></select>
<div class="btns">
  <button id="spk-use">Use this speaker</button>
  <button id="spk-connect">Connect</button>
  <button id="spk-disconnect">Disconnect</button>
  <button id="spk-tone">Say "bruh"</button>
</div>
<p style="font-size:.8rem;color:#888;max-width:420px">Only devices already paired with the Pi are listed.
Pair a new speaker once on the Pi: <code>bluetoothctl</code> &rarr; <code>scan on</code> &rarr;
<code>pair MAC</code> &rarr; <code>trust MAC</code>.</p>
<style>
  select, button { font-size: 1rem; padding: 10px 12px; border-radius: 8px; border: 1px solid #444;
                   background: #222; color: #eee; }
  select { width: 100%; max-width: 420px; }
  .btns { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
  button:disabled { opacity: .5; }
</style>
<script>
const spkStatus = document.getElementById("spk-status");
const spkSelect = document.getElementById("spk-select");
let busy = false;

function renderSpeaker(s) {
  spkStatus.textContent = s.message;
  spkStatus.className = s.connected ? "ok" : (s.mac ? "bad" : "");
  const current = [...spkSelect.options].map(o => o.value).join();
  const wanted = s.paired.map(d => d.mac).join();
  if (current !== wanted) {
    spkSelect.innerHTML = s.paired.length ? "" : "<option value=''>no paired devices</option>";
    s.paired.forEach(d => spkSelect.add(new Option(`${d.name} (${d.mac})`, d.mac)));
  }
  if (s.mac && document.activeElement !== spkSelect) spkSelect.value = s.mac;
}

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} :
    {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail || r.statusText);
  return j;
}

async function action(path, body, holdMs = 0) {
  if (busy) return;
  busy = true;
  document.querySelectorAll(".btns button").forEach(b => b.disabled = true);
  spkStatus.textContent = "working... (can take up to 15 s)"; spkStatus.className = "";
  try { renderSpeaker(await api(path, body)); }
  catch (e) { spkStatus.textContent = "Error: " + e.message; spkStatus.className = "bad"; }
  // keep a result message on screen briefly before polling overwrites it
  if (holdMs) await new Promise(r => setTimeout(r, holdMs));
  busy = false;
  document.querySelectorAll(".btns button").forEach(b => b.disabled = false);
}

document.getElementById("spk-use").onclick = () => action("/api/speaker", {mac: spkSelect.value || null});
document.getElementById("spk-connect").onclick = () => action("/api/speaker/connect", {});
document.getElementById("spk-disconnect").onclick = () => action("/api/speaker/disconnect", {});
document.getElementById("spk-tone").onclick = () => action("/api/speaker/test-tone", {}, 4000);

async function poll() {
  if (!busy) { try { renderSpeaker(await api("/api/speaker")); } catch (e) {} }
  setTimeout(poll, 2000);
}
poll();
</script>
<script>
const st = document.getElementById("status");
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.motion) renderMotion(m.motion);
    if (m.data) {
      st.textContent = "connected - live"; st.className = "ok";
      m.data.accel.forEach((v, i) => document.getElementById("a" + i).textContent = v.toFixed(3));
      m.data.gyro.forEach((v, i) => document.getElementById("g" + i).textContent = v.toFixed(2));
    } else {
      st.textContent = "connected - " + m.status; st.className = "bad";
    }
  };
  ws.onclose = () => { st.textContent = "disconnected, retrying..."; st.className = "bad"; setTimeout(connect, 1000); };
}
connect();
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


class SpeakerChoice(BaseModel):
    mac: Optional[str] = None


def speaker_state() -> dict:
    s = speaker.status()
    return {"mac": s.mac, "connected": s.connected, "message": s.message,
            "auto_reconnect": speaker.auto_reconnect, "paired": paired_devices()}


# Plain `def` routes run in FastAPI's threadpool, so the blocking
# bluetoothctl calls inside SpeakerManager don't stall the event loop.
@app.get("/api/speaker")
def get_speaker():
    return speaker_state()


@app.post("/api/speaker")
def set_speaker(choice: SpeakerChoice):
    try:
        speaker.set_speaker_mac(choice.mac)
    except ValueError as e:
        raise HTTPException(400, str(e))
    save_mac(speaker.status().mac)
    if speaker.status().mac:
        speaker.connect()  # connect right away instead of waiting for the next poll
    return speaker_state()


@app.post("/api/speaker/connect")
def connect_speaker():
    if not speaker.status().mac:
        raise HTTPException(400, "Choose a speaker first")
    speaker.connect()
    return speaker_state()


@app.post("/api/speaker/disconnect")
def disconnect_speaker():
    if not speaker.status().mac:
        raise HTTPException(400, "Choose a speaker first")
    speaker.disconnect()  # also pauses auto-reconnect until Connect is pressed
    return speaker_state()


@app.post("/api/speaker/test-tone")
def test_tone():
    try:
        device = play_test_tone()
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    state = speaker_state()
    if device not in PIPEWIRE_DEVICES:
        state["message"] = (f"Played on {device}, which can't reach Bluetooth. "
                            "Install pipewire-alsa, then try again.")
    elif not state["connected"]:
        state["message"] = f"Played on {device}, but no speaker is connected (likely HDMI)."
    else:
        state["message"] = f"Played on {device} -> {state['mac']}. Did you hear 'bruh'?"
    return state


class MotionSettings(BaseModel):
    enabled: Optional[bool] = None
    full_dps: Optional[float] = None
    root_hz: Optional[float] = None
    mode: Optional[str] = None


@app.post("/api/motion")
def set_motion(s: MotionSettings):
    if s.mode is not None:
        if s.mode not in motion.MODES:
            raise HTTPException(400, f"mode must be one of {motion.MODES}")
        motion.mode = s.mode
    if s.full_dps is not None:
        motion.full_dps = min(1000.0, max(motion.dead_dps + 5, s.full_dps))
    if s.root_hz is not None:
        motion.root_hz = min(motion.MAX_HZ, max(motion.MIN_HZ, s.root_hz))
    try:
        if s.enabled is True:
            motion.start()
        elif s.enabled is False:
            motion.stop()
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return motion.state()


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data, status = imu.get()
            if not motion.running:
                motion.motion_amount()  # keep the speed readout live while the sound is off
            await websocket.send_json({"data": data, "status": status, "motion": motion.state()})
            await asyncio.sleep(0.05)  # 20 updates/s
    except WebSocketDisconnect:
        pass


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))  # no packets sent; just picks the LAN interface
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--speaker", help="Bluetooth MAC to keep connected (saved for next time)")
    args = p.parse_args()

    mac = args.speaker or load_saved_mac()
    speaker.set_speaker_mac(mac)
    if args.speaker:
        save_mac(speaker.status().mac)
    speaker.start()
    imu.start()
    print(f"\n  Speaker: {speaker.status().mac or 'none chosen yet (pick one on the page)'}")
    print(f"  Open on your phone (same Wi-Fi): http://{lan_ip()}:{args.port}\n")
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
    finally:
        motion.stop()
        speaker.stop()
