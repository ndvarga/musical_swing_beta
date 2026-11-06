from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

try:
    import soundfile as sf
except ImportError:  # lets this module be imported/edited off the Pi too
    sf = None

try:
    import sounddevice as sd
except (ImportError, OSError):
    # sounddevice raises OSError (not ImportError) when the system
    # PortAudio library isn't installed yet -- both cases should leave
    # the module importable, with playback simply reporting unavailable.
    sd = None

from .config import AudioConfig


@dataclass
class AudioStatus:
    running: bool
    message: str


def _resample_linear(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Basic linear-interpolation resample. This is adequate for looping
    background/enrichment clips, not intended as an audiophile-grade
    resampler -- for best fidelity, encode library files at the engine's
    sample_rate to begin with so this is a no-op.
    """
    if orig_sr == target_sr or len(data) == 0:
        return data.astype(np.float32)
    duration = len(data) / float(orig_sr)
    n_target = max(1, int(round(duration * target_sr)))
    x_orig = np.linspace(0.0, duration, num=len(data), endpoint=False)
    x_target = np.linspace(0.0, duration, num=n_target, endpoint=False)
    return np.interp(x_target, x_orig, data).astype(np.float32)


def load_library_file(path: str, target_sr: int) -> np.ndarray:
    """
    Loads an audio file, downmixes to mono, and resamples to target_sr.
    Returns a 1-D float32 array. Raises RuntimeError if soundfile isn't
    installed, and whatever soundfile/OSError the underlying read raises
    for a corrupt/unsupported file -- callers should expect and handle
    both since keeper-uploaded files aren't guaranteed clean.
    """
    if sf is None:
        raise RuntimeError("soundfile is not installed. Run: pip install soundfile")
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    mono = data.mean(axis=1)
    return _resample_linear(mono, sr, target_sr)


def read_looped(arr: np.ndarray, pos: int, frames: int) -> np.ndarray:
    """
    Reads `frames` samples starting at pos, looping around arr's end as
    many times as needed. Returns silence if arr is empty (unassigned
    layer) rather than raising, so callers don't need a special case.
    """
    n = len(arr)
    if n == 0:
        return np.zeros(frames, dtype=np.float32)
    pos = pos % n
    end = pos + frames
    if end <= n:
        return arr[pos:end]
    first = arr[pos:n]
    remaining = end - n
    reps = remaining // n
    tail = remaining % n
    parts = [first] + [arr] * reps + [arr[0:tail]]
    return np.concatenate(parts)


@dataclass
class LibraryLayer:
    """
    One of the 3 enrichment layers: a looping clip plus its own read
    position. Deliberately immutable-by-replacement -- assign_layer()
    always builds a brand new LibraryLayer rather than mutating an
    existing one, so the realtime callback can hold a reference across
    a whole callback without it changing underneath it.
    """
    filename: Optional[str] = None
    data: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    position: int = 0


class AudioEngine:
    """
    Plays up to 3 independently-looping library clips, mixed together
    with per-layer smoothed gains driven by control.py's outputs -- same
    role the original sine-synthesis engine played, but with real audio
    files instead of tones. control.py, calibration.py, and everything
    upstream of gains/master_cap is unchanged.

    Library files live under cfg.library_dir. Layer assignment (which
    file plays for layer 1/2/3) can be changed live via assign_layer(),
    safe to call from another thread (e.g. an API request handler) while
    the realtime audio callback is running.
    """

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self._lock = threading.Lock()

        # Fail-safe mutes, keyed by reason ("paused", "imu", "speaker", ...),
        # kept separate from cfg.safe_mute so a fault clearing never
        # overrides the user's own Safe mute setting, and vice versa.
        self._fault_mutes: set = set()

        self._layers: List[LibraryLayer] = [LibraryLayer() for _ in range(3)]
        self._target_gains = np.zeros(3, dtype=float)
        self._current_gains = np.zeros(3, dtype=float)  # audio-thread-only, no lock needed
        self._master_cap = 0.35

        self._stream: Optional["sd.OutputStream"] = None
        self.status = AudioStatus(running=False, message="Audio not started")

        Path(self.cfg.library_dir).mkdir(parents=True, exist_ok=True)
        self._load_configured_layers()

    # ---------------------------
    # Library / layer management
    # ---------------------------

    def library_files(self) -> List[str]:
        """Filenames currently sitting in the library directory."""
        d = Path(self.cfg.library_dir)
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_file())

    def _load_configured_layers(self) -> None:
        for i, filename in enumerate(self.cfg.layer_files):
            if not filename:
                continue
            try:
                self.assign_layer(i, filename)
            except Exception as e:
                # Don't let one missing/corrupt file at startup take down
                # the whole engine on an unattended device -- leave that
                # layer silent and surface the problem in status instead.
                self.status = AudioStatus(running=False, message=f"Layer {i + 1} load issue: {e}")

    def assign_layer(self, layer_idx: int, filename: Optional[str]) -> None:
        """
        Assigns (or clears, if filename is None) the clip for one layer.
        Decodes the file BEFORE taking the lock, so the realtime callback
        is only ever blocked for the instant it takes to swap a reference,
        never for the duration of a file load.
        """
        if not (0 <= layer_idx < 3):
            raise ValueError("layer_idx must be 0, 1, or 2")

        if not filename:
            new_layer = LibraryLayer(filename=None)
        else:
            path = Path(self.cfg.library_dir) / filename
            if not path.is_file():
                raise FileNotFoundError(f"No such library file: {filename}")
            data = load_library_file(str(path), self.cfg.sample_rate)
            new_layer = LibraryLayer(filename=filename, data=data)

        with self._lock:
            self._layers[layer_idx] = new_layer

        self.cfg.layer_files[layer_idx] = filename

    def current_assignments(self) -> List[Optional[str]]:
        with self._lock:
            return [layer.filename for layer in self._layers]

    # ---------------------------
    # Runtime controls (same interface as the original engine)
    # ---------------------------

    def set_master_cap(self, cap: float) -> None:
        with self._lock:
            self._master_cap = float(cap)

    def set_safe_mute(self, mute: bool) -> None:
        """The user's own mute (UI checkbox / keeper app)."""
        self.cfg.safe_mute = bool(mute)

    def set_fault_mute(self, reason: str, mute: bool) -> None:
        """
        Mutes (or releases) for one fail-safe reason. Output stays silent
        while any reason is set or the user's safe mute is on.
        """
        with self._lock:
            if mute:
                self._fault_mutes.add(reason)
            else:
                self._fault_mutes.discard(reason)

    def mute_reasons(self) -> List[str]:
        """Everything currently muting output, for status displays."""
        with self._lock:
            reasons = sorted(self._fault_mutes)
        if self.cfg.safe_mute:
            reasons.insert(0, "safe_mute")
        return reasons

    def set_gains(self, gains: Tuple[float, float, float]) -> None:
        with self._lock:
            self._target_gains[:] = np.clip(np.array(gains, dtype=float), 0.0, 1.0)

    def start(self) -> None:
        if not self.cfg.enabled:
            self.status = AudioStatus(running=False, message="Audio disabled in config")
            return
        if sd is None:
            self.status = AudioStatus(running=False, message="sounddevice not installed")
            return
        if self._stream is not None:
            return
        try:
            self._stream = sd.OutputStream(
                samplerate=self.cfg.sample_rate,
                blocksize=self.cfg.block_size,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            self.status = AudioStatus(running=True, message="Audio running")
        except Exception as e:
            self.status = AudioStatus(running=False, message=f"Audio error: {e}")
            self._stream = None

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
            self.status = AudioStatus(running=False, message="Audio stopped")

    # ---------------------------
    # Realtime callback
    # ---------------------------

    def _callback(self, outdata, frames, time_info, status) -> None:
        with self._lock:
            muted = self.cfg.safe_mute or bool(self._fault_mutes)
            target = self._target_gains.copy()
            master = float(self._master_cap)
            # Snapshot the layer *references* -- if assign_layer() swaps
            # self._layers[i] mid-callback from another thread, this
            # callback keeps using the object it already grabbed, and the
            # new clip takes effect cleanly on the next callback. No torn
            # reads, no locking needed for the actual mixing work below.
            layers = list(self._layers)

        if muted:
            # Mute is immediate, but unmuting fades in from zero via the
            # gain smoothing below instead of jumping back to the old gains.
            self._current_gains[:] = 0.0
            outdata[:] = np.zeros((frames, 1), dtype=np.float32)
            return

        tau = max(self.cfg.gain_smoothing_tau_s, 1e-4)
        alpha = 1.0 - math.exp(-(frames / self.cfg.sample_rate) / tau)
        self._current_gains += alpha * (target - self._current_gains)
        gains = self._current_gains

        y = np.zeros(frames, dtype=float)
        for i, layer in enumerate(layers):
            chunk = read_looped(layer.data, layer.position, frames)
            y += gains[i] * chunk
            # Position advances every callback regardless of current gain,
            # so a layer fading back in resumes mid-loop instead of
            # popping back in at sample 0 (mirrors how the original sine
            # engine kept phase continuous even while a layer was silent).
            layer.position = (layer.position + frames) % max(len(layer.data), 1)

        y *= master
        y = np.clip(y, -0.95, 0.95)
        outdata[:] = y.reshape(-1, 1).astype(np.float32)
