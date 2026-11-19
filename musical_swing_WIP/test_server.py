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

# TODO: master_cap (front and backend), calibration (front and backend), 
# gain sliders for individual layers, upload for audio files, audio selector, 
# "timeline" for audio where you can select at what point the layers turn on (Peak of envelope?),
# multispeaker connection?


import subprocess
import shutil
import argparse
import asyncio
import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from swing_sim.imu_hw import MPU6050Reader
from swing_sim.speaker import SpeakerManager
from swing_sim.audio import AudioEngine, AudioConfig, AudioStatus
from swing_sim.control import IntensityEstimator, Controller, EnvelopeConfig, ControlConfig
# TODO: implement the API calls for these
from swing_sim.calibration import CalibrationManager, CalibrationResult

try:
    import numpy as np
    import sounddevice as sd
    _SD_ERROR = None
except (ImportError, OSError) as e:  # OSError: PortAudio library missing
    sd = None
    _SD_ERROR = str(e)

STALE_S = 0.5  # a sample older than this counts as "sensor lost"


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
    DEBUG ONLY: a test tone for checking the sensor -> sound path, not the
    real audio pipeline (that's the layer engine: AudioEngine + Controller).
    Driven from the page's "Debug: test tone" section via /api/debug/tone.

    Plays a continuous chord driven by how fast the gyro is turning. The
    turning speed becomes a motion amount from 0 (below dead_dps: sensor
    noise/offset at rest) to 1 (full_dps and above), smoothed with a quick
    attack and slow release so it swells with motion rather than flickering.
    Volume = motion amount; the chord's pitch is fixed at ROOT_HZ.

    If the sensor stops delivering, the motion amount drops to 0: silence.
    """

    # TODO: most of audio is for testing
    SR = 48000
    CHORD_RATIOS = np.array([1.0, 2 ** (4 / 12), 2 ** (7 / 12), 2.0])  # major triad + octave
    ROOT_HZ = 220.0  # chord root (A3)
    ATTACK_S, RELEASE_S = 0.08, 0.6

    def __init__(self, imu: MPU6050Reader):
        self.imu = imu
        self.dead_dps = 8.0
        self.full_dps = 150.0
        self.stream = None
        self.device = None
        self.level = 0.0         # smoothed motion amount 0..1, owned by the audio callback
        self.gain = 0.0          # output volume 0..1, owned by the audio callback
        self.rate_dps = 0.0      # latest gyro speed, for the page
        self._phase = np.zeros(len(self.CHORD_RATIOS))

    @property
    def running(self) -> bool:
        return self.stream is not None

    def motion_amount(self) -> float:
        sample = self.imu.get_latest()
        if sample is None or self.imu.seconds_since_last_sample() > STALE_S:
            self.rate_dps = 0.0
            return 0.0
        self.rate_dps = float(np.degrees(self.imu.raw_feature_from(sample)))
        x = (self.rate_dps - self.dead_dps) / max(1e-6, self.full_dps - self.dead_dps)
        return float(min(1.0, max(0.0, x)))

    def _callback(self, outdata, frames, time_info, status) -> None:
        dt = frames / self.SR
        target = self.motion_amount()
        tau = self.ATTACK_S if target > self.level else self.RELEASE_S
        self.level += (target - self.level) * (1 - np.exp(-dt / tau))

        g0, g1 = self.gain, self.level
        self.gain = g1

        # Phase carried across blocks, so the tone is continuous (no clicks).
        steps = 2 * np.pi * self.ROOT_HZ * self.CHORD_RATIOS / self.SR
        phases = self._phase[:, None] + np.outer(steps, np.arange(frames))
        tone = np.sin(phases).sum(axis=0) / len(self.CHORD_RATIOS)
        self._phase = (self._phase + steps * frames) % (2 * np.pi)
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

    def state(self) -> dict:
        return {"enabled": self.running,
                "gain": round(self.gain, 3), "level": round(self.level, 3),
                "rate_dps": round(self.rate_dps, 1), "full_dps": self.full_dps,
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


# setting up all the components we need
app = FastAPI()
imu = MPU6050Reader(sample_hz=100, feature_mode="magnitude")
motion = MotionAudio(imu)
speaker = SpeakerManager(speaker_mac=None, auto_start=False)  # started in __main__
audio_cfg = AudioConfig()
audio = AudioEngine(audio_cfg)
env_cfg = EnvelopeConfig()
estimator = IntensityEstimator(env_cfg)
control_cfg = ControlConfig()
controller = Controller(control_cfg)

INDEX_HTML = Path(__file__).with_name("static") / "index.html"


@app.get("/")
def index():
    return FileResponse(INDEX_HTML)  # read from disk each time: edit, then just reload


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


"""
Methods related to scanning for Bluetooth devices from the web app
"""
_scan = {"scanning": False, "error": None}
_scan_lock = threading.Lock()

def _scan_worker(seconds: float) -> None:
    """
    Runs blocking bluetoothctl scan off request thread
    """
    error = None
    try:
        speaker.scan(seconds)
    except RuntimeError as e:
        error = str(e)
    except Exception as e:
        error = f"Bluetooth scan failed: {e}"
    with _scan_lock:
        _scan.update(scanning=False, error=error)        

@app.post("/api/speaker/scan")
def start_scan():

    with _scan_lock:
        if not _scan["scanning"]:
            _scan.update(scanning=True, error=None)
            threading.Thread(target=_scan_worker, args=(15,), name="bt-scan", daemon=True).start()
    return get_scan()

@app.get("/api/speaker/scan")
def get_scan():
    """{scanning, devices}; devices fill in while the scan runs."""
    with _scan_lock:
        scanning, error = _scan["scanning"], _scan["error"]
    if error:
        raise HTTPException(500, error)
    paired = {d["mac"] for d in paired_devices()}
    devices = [dict(d, paired=d["mac"] in paired) for d in speaker._discovered_devices()]
    return {"scanning": scanning, "devices": devices}

"""
Method to pair speaker in "Pair new speaker" menu
"""
@app.post("/api/speaker/pair")
def pair_speaker(choice: SpeakerChoice):
    ok = speaker.pair(choice.mac)
    speaker.set_speaker_mac(mac); save_mac(mac); speaker.connect()
    return speaker_state()   # page passes this to renderSpeaker()



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


class CalibrateSettings(BaseModel):
    seconds: float = 2.0

# button to calibrate the IMU
@app.post("/api/imu/calibrate")
def calibrate_imu(s: Optional[CalibrateSettings] = None):
    """
    Averages the gyro for `seconds` (sensor still) as the new zero offset.
    The body is optional: an empty POST calibrates for 2 s.
    """
    seconds = min(10.0, max(0.5, (s or CalibrateSettings()).seconds))
    if not imu.calibrate_bias(seconds):
        raise HTTPException(503, imu.message)
    return {"message": imu.message}


# ---- Intensity range calibration (IntensityEstimator floor / max_val) ----
# Same rule as swing_sim/calibration.py: 95th percentile of the raw feature
# (gyro magnitude, rad/s), so one bump or gust doesn't set the value. Done
# directly rather than via CalibrationManager, because that only produces a
# result once BOTH phases have run, and here each button applies on its own.
CAL_PERCENTILE = 95
_range_cal_lock = threading.Lock()


def _collect_raw_feature(seconds: float) -> np.ndarray:
    """Raw intensity feature for every new IMU sample over `seconds` (blocks)."""
    vals, last_t = [], None
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        sample = imu.get_latest()
        if sample is not None and sample.t != last_t and imu.seconds_since_last_sample() <= STALE_S:
            vals.append(imu.raw_feature_from(sample))
            last_t = sample.t
        time.sleep(0.005)
    return np.array(vals, dtype=float)


def range_state() -> dict:
    f, m = estimator.cfg.floor, estimator.cfg.max_val
    return {"floor": round(f, 4), "max_val": round(m, 4),
            "floor_dps": round(float(np.degrees(f)), 1), "max_dps": round(float(np.degrees(m)), 1)}


@app.get("/api/imu/range")
def get_range():
    """Current IntensityEstimator floor/max_val (rad/s, plus °/s for display)."""
    return range_state()


@app.post("/api/imu/calibrate-rest")
def calibrate_rest(s: Optional[CalibrateSettings] = None):
    """
    Swing hanging still. First half of `seconds`: gyro bias (zero offset);
    second half: floor = 95th percentile of what's left (noise + wobble).
    Bias goes first because it changes the resting magnitude.
    """
    # TODO: Implement
    raise HTTPException(500, "calibrate rest not implemented")


@app.post("/api/imu/calibrate-swing")
def calibrate_swing(s: Optional[CalibrateSettings] = None):
    """Someone swinging hard: max_val = 95th percentile of the raw feature over `seconds`."""
    # TODO: implement from CalibrationManager
    raise HTTPException(500, "calibrate swing not implemented")

class ToneSettings(BaseModel):
    enabled: Optional[bool] = None
    full_dps: Optional[float] = None

"""
Layer mode related classes and API defs
"""

class LayerRunner:
    """
    runs step loop for IMU -> IntensityEstimator -> Controller -> AudioEngine gains
    on background thread while layer mode is enabled
    """
    def __init__(self, imu:MPU6050Reader, audio:AudioEngine, estimator:IntensityEstimator, controller:Controller, hz: float = 50):
        self.imu, self.audio = imu, audio
        self.estimator, self.controller = estimator, controller
        self.period = 1.0 / hz
        self.gains = (0.0, 0.0, 0.0)
        self._stop = threading.Event()
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None

    def start(self) -> None:
        if self.running:
            return
        self.controller.reset()
        self.estimator.state.envelope = 0.0
        self.audio.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="layer-step", daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            self._thread = None
        self.audio.stop()
        self.gains = (0.0,0.0,0.0)

    def _run(self) -> None:
        last = time.monotonic()
        while not self._stop.wait(self.period):   # sleeps one period; exits as soon as stop() is called
            now = time.monotonic()
            dt = min(now - last, 0.1)             # same cap as ui.py, so a stall doesn't jump the envelope
            last = now
            self.step(dt)

    def step(self, dt: float):
        # TODO: calibration manager update
        sample = self.imu.get_latest()
        if sample is None or self.imu.seconds_since_last_sample() > STALE_S:
            self.audio.set_fault_mute("imu", True)
            return
        self.audio.set_fault_mute("imu", False)
        raw_feature = self.imu.raw_feature_from(sample)
        state = self.estimator.update(raw_feature, dt)
        ctrl_out = self.controller.update(state.x_norm, dt)
        #what hsould it do if it chatters
        if ctrl_out.chatter_flag:
            self.audio.set_fault_mute("chatter", True)
            return
        self.gains = ctrl_out.gains
        self.audio.set_master_cap(ctrl_out.master_cap)
        self.audio.set_gains(self.gains)


class LayerAudioSettings(BaseModel):
    enabled: bool


class LayerChoice(BaseModel):
    layer: int                  # 0, 1 or 2
    file: Optional[str] = None  # filename in the library folder; None clears the layer


@app.get("/api/layers")
def get_layers():
    return {"files": audio.library_files(),
            "layers": [{"file": f} for f in audio.current_assignments()]}



@app.post("/api/layers")
def set_layer(choice: LayerChoice):
    # TODO: put print in verbose mode
    print(f"choice idx: {choice.layer}\nchoice file: {choice.file}")
    try:
        audio.assign_layer(choice.layer, choice.file)
    except (ValueError, FileNotFoundError) as e: # wrong index/name 
        raise HTTPException(400, str(e))
    except (RuntimeError, OSError) as e:           # file there but unusable
        raise HTTPException(500, f"Couldn't load {choice.file}: {e}")
    return get_layers()


def layer_audio_state() -> dict:
    return {"enabled": audio.status.running, "gains": [0.0, 0.0, 0.0]}

layers = LayerRunner(imu, audio, estimator, controller)

@app.post("/api/layers/audio")
def set_layer_audio(s: LayerAudioSettings):
    """Starts/stops the layer engine (the real audio). Returns layer_audio_state()."""
    # TODO: verbose mode
    motion.stop()  # the debug tone and the layer engine shouldn't play at once
    if audio.load_issues:
        for issue in audio.load_issues:
            print(f"load issues:{issue}")
    if s.enabled:
        layers.start()
        print(f"Layers audio started.\nmessage: {audio.status.message}\nrunning: {audio.status.running}")
    else:
        layers.stop()
        print(f"Layers audio stopped.\nmessage: {audio.status.message}\nrunning: {audio.status.running}")

    return layer_audio_state()

def layer_audio_state() -> dict:
    return {"enabled": layers.running, "gains": [round(g, 3) for g in layers.gains]}


@app.post("/api/debug/tone")
def set_debug_tone(s: ToneSettings):
    """DEBUG: the MotionAudio test tone. Not part of the real audio pipeline."""
    if s.enabled is True and audio.status.running:
        raise HTTPException(409, "Stop the layer audio first")
    if s.full_dps is not None:
        motion.full_dps = min(1000.0, max(motion.dead_dps + 5, s.full_dps))
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
            sample = imu.get_latest()
            data, status = None, imu.message
            if sample is not None and imu.seconds_since_last_sample() <= STALE_S:
                accel = sample.accel_xyz if sample.accel_xyz is not None else np.zeros(3)
                data = {"accel": [round(float(v), 3) for v in accel],
                        "gyro": [round(float(v), 2) for v in np.degrees(sample.gyro_xyz)]}
            if not motion.running:
                motion.motion_amount()  # keep the speed readout live while the sound is off
            await websocket.send_json({"data": data, "status": status,
                                       "tone": motion.state(), "layers": layer_audio_state()})
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
    imu.start()  # gyro bias starts at zero; calibrate from the page's button
    print(f"\n  Speaker: {speaker.status().mac or 'none chosen yet (pick one on the page)'}")
    print(f"  Open on your phone (same Wi-Fi): http://{lan_ip()}:{args.port}\n")
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
    finally:
        motion.stop()
        layers.stop()
        speaker.stop()
        imu.stop()
