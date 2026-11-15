from __future__ import annotations

import math
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from smbus2 import SMBus
except ImportError:  # lets this module be imported/edited off the Pi too
    SMBus = None

_PWR_MGMT_1 = 0x6B
_SMPLRT_DIV = 0x19
_CONFIG = 0x1A
_GYRO_CONFIG = 0x1B
_ACCEL_CONFIG = 0x1C
_ACCEL_XOUT_H = 0x3B  # accel(6) + temp(2) + gyro(6), read in one burst
_GYRO_XOUT_H = 0x43
_WHO_AM_I = 0x75

_GYRO_LSB_PER_DPS = 131.0  # at the ±250 °/s range
_ACCEL_LSB_PER_G = 16384.0  # at the ±2 g range

FEATURE_MODES = ("magnitude", "x", "y", "z")


@dataclass
class IMUSample:
    t: float
    gyro_xyz: np.ndarray  # rad/s, bias-corrected
    accel_xyz: Optional[np.ndarray] = None  # g, uncorrected


class MPU6050:
    """
    Minimal MPU6050 driver over I2C. Opening wakes the chip and sets
    ±250 °/s gyro, ±2 g accel, ~44 Hz low-pass and a 1 kHz internal
    sample rate. Raises OSError (or RuntimeError if smbus2 is missing)
    when the sensor can't be reached.
    """

    def __init__(self, bus: int = 1, address: int = 0x68):
        if SMBus is None:
            raise RuntimeError("smbus2 not installed")
        self.address = address
        self._bus = SMBus(bus)
        try:
            self._bus.read_byte_data(address, _WHO_AM_I)  # fails fast if nothing answers
            self._bus.write_byte_data(address, _PWR_MGMT_1, 0x00)    # wake up
            self._bus.write_byte_data(address, _CONFIG, 0x03)        # DLPF ~44 Hz
            self._bus.write_byte_data(address, _SMPLRT_DIV, 0x00)    # 1 kHz / (1 + 0)
            self._bus.write_byte_data(address, _GYRO_CONFIG, 0x00)   # ±250 °/s
            self._bus.write_byte_data(address, _ACCEL_CONFIG, 0x00)  # ±2 g
            time.sleep(0.05)  # let the first conversion finish after wake-up
        except OSError:
            self._bus.close()
            raise

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """(accel in g, gyro in rad/s), raw (no bias correction)."""
        raw = bytes(self._bus.read_i2c_block_data(self.address, _ACCEL_XOUT_H, 14))
        ax, ay, az, _temp, gx, gy, gz = struct.unpack(">7h", raw)
        accel = np.array([ax, ay, az], dtype=float) / _ACCEL_LSB_PER_G
        gyro = np.radians(np.array([gx, gy, gz], dtype=float) / _GYRO_LSB_PER_DPS)
        return accel, gyro

    def read_gyro(self) -> np.ndarray:
        """Gyro in rad/s, raw (no bias correction)."""
        raw = bytes(self._bus.read_i2c_block_data(self.address, _GYRO_XOUT_H, 6))
        return np.radians(np.array(struct.unpack(">3h", raw), dtype=float) / _GYRO_LSB_PER_DPS)

    def close(self) -> None:
        try:
            self._bus.close()
        except OSError:
            pass


class MPU6050Reader:
    """
    Samples an MPU6050 at sample_hz on its own background thread and
    keeps only the most recent reading, which AppRuntime.tick() picks up
    once per UI frame via get_latest().

    Never raises out of the thread: if the sensor isn't there at startup
    (e.g. running off the Pi) or drops off the bus mid-session (a loose
    wire on a moving swing), the thread keeps retrying the connection
    and simply stops publishing samples meanwhile. That makes
    seconds_since_last_sample() grow, which is what tick()'s stale check
    uses to force-mute audio -- so a lost sensor fails safe to silence
    rather than to a frozen gain.
    """

    def __init__(
        self,
        sample_hz: int = 200,
        feature_mode: str = "magnitude",
        bus: int = 1,
        address: int = 0x68,
        reconnect_interval_s: float = 1.0,
    ):
        if feature_mode not in FEATURE_MODES:
            raise ValueError(f"feature_mode must be one of {FEATURE_MODES}")
        self.sample_hz = int(sample_hz)
        self.feature_mode = feature_mode
        self.bus = bus
        self.address = address
        self.reconnect_interval_s = float(reconnect_interval_s)

        self._lock = threading.Lock()
        # Stored raw, with the bias subtracted on read, so a new bias from
        # calibrate_bias() applies immediately, even to the last sample.
        self._latest_raw: Optional[np.ndarray] = None
        self._latest_accel: Optional[np.ndarray] = None
        self._latest_t = 0.0
        self._latest_wall: Optional[float] = None
        self._bias = np.zeros(3)

        # Bias calibration accumulator, filled by the reader thread while
        # calibrate_bias() waits.
        self._cal_active = False
        self._cal_sum = np.zeros(3)
        self._cal_count = 0

        self._t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.message = "IMU not started"

    # ---------------------------
    # Lifecycle
    # ---------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="imu-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    # ---------------------------
    # Reads (safe from any thread)
    # ---------------------------

    def get_latest(self) -> Optional[IMUSample]:
        with self._lock:
            if self._latest_raw is None:
                return None
            return IMUSample(t=self._latest_t, gyro_xyz=self._latest_raw - self._bias,
                             accel_xyz=self._latest_accel)

    def seconds_since_last_sample(self) -> float:
        with self._lock:
            last = self._latest_wall
        return math.inf if last is None else time.monotonic() - last

    def raw_feature_from(self, sample: IMUSample) -> float:
        """
        Turns one gyro sample into the non-negative intensity feature the
        envelope follower expects (same units as the simulator's |gyro_y|,
        rad/s). "magnitude" is the length of the full rate vector, so it
        doesn't care how the board is mounted; "x"/"y"/"z" use the
        absolute value of a single axis, for when the board's mounting is
        known and you want to reject off-axis wobble.
        """
        g = sample.gyro_xyz
        if self.feature_mode == "magnitude":
            return float(np.linalg.norm(g))
        return abs(float(g["xyz".index(self.feature_mode)]))

    def calibrate_bias(self, seconds: float = 2.0) -> bool:
        """
        Averages the gyro over `seconds` (swing still/hanging) and uses
        the result as the zero offset from then on. Blocks the caller for
        `seconds`. Returns False and keeps the previous bias if no samples
        arrived in that window (sensor missing), rather than waiting
        forever.
        """
        with self._lock:
            self._cal_sum = np.zeros(3)
            self._cal_count = 0
            self._cal_active = True

        time.sleep(seconds)

        with self._lock:
            self._cal_active = False
            if self._cal_count == 0:
                self.message = "IMU bias calibration skipped: no samples"
                return False
            self._bias = self._cal_sum / self._cal_count
            self.message = f"IMU bias calibrated from {self._cal_count} samples"
            return True

    # ---------------------------
    # Reader thread
    # ---------------------------

    def _run(self) -> None:
        period = 1.0 / max(self.sample_hz, 1)
        dev: Optional[MPU6050] = None
        next_t = time.monotonic()

        while not self._stop.is_set():
            if dev is None:
                try:
                    dev = MPU6050(self.bus, self.address)
                    self.message = "IMU connected"
                    next_t = time.monotonic()
                except Exception as e:
                    self.message = f"IMU unavailable: {e}"
                    self._stop.wait(self.reconnect_interval_s)
                    continue

            try:
                accel, raw = dev.read()
            except OSError as e:
                # Don't publish anything for a failed read -- let the
                # sample go stale so tick() mutes, and reconnect.
                self.message = f"IMU read error, reconnecting: {e}"
                dev.close()
                dev = None
                continue

            now = time.monotonic()
            with self._lock:
                if self._cal_active:
                    self._cal_sum += raw
                    self._cal_count += 1
                self._latest_raw = raw
                self._latest_accel = accel
                self._latest_t = now - self._t0
                self._latest_wall = now

            # Fixed-rate pacing; if we've fallen more than a period behind
            # (GC pause, bus stall), resync instead of bursting to catch up.
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            elif delay < -period:
                next_t = time.monotonic()

        if dev is not None:
            dev.close()
