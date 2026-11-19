"""
One-shot intensity calibration.

Replaces the live CalibrationManager approach (calibration.py, still used
by ui.py and left untouched). Nothing here runs inside the control loop:
each calibration is a single blocking call that reads the IMU for N
seconds, computes a value, and returns it. Normal running only uses the
numbers saved by the last calibration.

Typical use (e.g. from test_server.py):

    cal = load() or SavedCalibration()      # startup: last saved calibration
    apply(cal, imu, estimator)

    cal = calibrate_rest(imu, 4.0, cal)     # "Calibrate rest" button
    apply(cal, imu, estimator); save(cal)

    cal = calibrate_swing(imu, 8.0, cal)    # "Calibrate swinging" button
    apply(cal, imu, estimator); save(cal)

Units: everything is rad/s, the same units as raw_feature_from().
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from .config import EnvelopeConfig, project_path

if TYPE_CHECKING:  # type hints only; avoids import side effects at runtime
    from .control import IntensityEstimator
    from .imu_hw import MPU6050Reader


# Floor: high percentile, so the floor sits above almost all resting
# noise/wobble -> more margin against layers turning on with no one on
# the swing.
FLOOR_PERCENTILE = 99

# Max: lower percentile, because the smoothed envelope never reaches the
# raw peaks of a swing; a lower max makes the top of the 0..1 range
# reachable. Lower it further (e.g. 80) if x_norm still tops out well
# below 1.0 on a hard swing.
MAX_PERCENTILE = 90

# A sample older than this counts as "sensor lost" (same as test_server.py).
STALE_S = 0.5

CAL_FILE = project_path("calibration.json")


@dataclass
class SavedCalibration:
    """
    One complete calibration. floor/max_val and bias are saved together
    because the floor was measured on bias-corrected readings: loading
    one without the other makes them disagree.
    """
    floor: float = EnvelopeConfig.floor        # rad/s, resting level (0 on the x_norm scale)
    max_val: float = EnvelopeConfig.max_val    # rad/s, hard-swing level (1 on the x_norm scale)
    bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # gyro zero offset per axis, rad/s
    calibrated_at: Optional[str] = None        # local time of the last change, for display


# ---------------------------
# Reading the sensor
# ---------------------------

def collect_raw_feature(imu: "MPU6050Reader", seconds: float) -> np.ndarray:
    """
    Blocks for `seconds`, returning the raw intensity feature of every
    new IMU sample in that window. Stale samples (sensor lost) are
    skipped, so a missing sensor gives an empty array.
    """
    vals = []
    last_t = None
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        sample = imu.get_latest()
        if (sample is not None and sample.t != last_t
                and imu.seconds_since_last_sample() <= STALE_S):
            vals.append(imu.raw_feature_from(sample))
            last_t = sample.t
        time.sleep(0.005)  # faster than any sample rate we use; just avoids a busy loop
    return np.array(vals, dtype=float)


# ---------------------------
# Maths (no sensor, easy to test)
# ---------------------------

def floor_from(samples: np.ndarray, pct: float = FLOOR_PERCENTILE) -> float:
    """Floor from resting data. Raises ValueError if there's no data."""
    if len(samples) == 0:
        raise ValueError("no IMU samples during rest calibration (sensor missing?)")
    return float(np.percentile(samples, pct))


def max_from(samples: np.ndarray, floor: float, pct: float = MAX_PERCENTILE) -> float:
    """
    Max from hard-swing data. Raises ValueError if there's no data or it
    isn't above the floor (nobody swung, or the floor is wrong).
    """
    if len(samples) == 0:
        raise ValueError("no IMU samples during swing calibration (sensor missing?)")
    max_val = float(np.percentile(samples, pct))
    if max_val <= floor:
        raise ValueError(
            f"swing reading ({np.degrees(max_val):.1f} °/s) is not above the floor "
            f"({np.degrees(floor):.1f} °/s); swing harder or recalibrate rest")
    return max_val


# ---------------------------
# One-shot calibrations
# ---------------------------

def calibrate_rest(imu: "MPU6050Reader", seconds: float,
                   current: SavedCalibration) -> SavedCalibration:
    """
    Swing hanging still. First half of `seconds`: gyro bias. Second half:
    floor. Bias goes first because the floor is measured on
    bias-corrected readings.

    Returns a new SavedCalibration (current max_val kept). Raises
    ValueError on failure, with the IMU's bias put back as it was, so a
    failed calibration changes nothing.
    """
    old_bias = _get_bias(imu)
    try:
        if not imu.calibrate_bias(seconds / 2):
            raise ValueError(imu.message)
        floor = floor_from(collect_raw_feature(imu, seconds / 2))
        if floor >= current.max_val:
            raise ValueError(
                f"rest reading ({np.degrees(floor):.1f} °/s) is not below the current max "
                f"({np.degrees(current.max_val):.1f} °/s); keep the swing still, "
                "or recalibrate swinging afterwards")
    except ValueError:
        _set_bias(imu, old_bias)
        raise
    return replace(current, floor=floor, bias=_get_bias(imu), calibrated_at=_now())


def calibrate_swing(imu: "MPU6050Reader", seconds: float,
                    current: SavedCalibration) -> SavedCalibration:
    """
    Someone swinging hard for `seconds`. Returns a new SavedCalibration
    with max_val set (floor and bias kept). Raises ValueError on failure.
    """
    max_val = max_from(collect_raw_feature(imu, seconds), current.floor)
    return replace(current, max_val=max_val, calibrated_at=_now())


# ---------------------------
# Using / saving a calibration
# ---------------------------

def apply(cal: SavedCalibration, imu: "MPU6050Reader", estimator: "IntensityEstimator") -> None:
    """Puts a calibration into effect: gyro bias on the IMU, floor/max on the estimator."""
    _set_bias(imu, cal.bias)
    estimator.update_floor_max(cal.floor, cal.max_val)


def save(cal: SavedCalibration, path: Path = CAL_FILE) -> None:
    Path(path).write_text(json.dumps(asdict(cal), indent=2))


def load(path: Path = CAL_FILE) -> Optional[SavedCalibration]:
    """The saved calibration, or None if there isn't one (never calibrated) or it's unreadable."""
    try:
        d = json.loads(Path(path).read_text())
        return SavedCalibration(
            floor=float(d["floor"]),
            max_val=float(d["max_val"]),
            bias=tuple(float(b) for b in d["bias"]),
            calibrated_at=d.get("calibrated_at"),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


# ---------------------------
# Helpers
# ---------------------------
# MPU6050Reader has no public way to read or set its bias yet, so these
# reach into its private fields (under its lock, as its own methods do).
# TODO: move into MPU6050Reader as get_bias()/set_bias() methods.

def _get_bias(imu: "MPU6050Reader") -> Tuple[float, float, float]:
    with imu._lock:
        return tuple(float(b) for b in imu._bias)


def _set_bias(imu: "MPU6050Reader", xyz) -> None:
    with imu._lock:
        imu._bias = np.asarray(xyz, dtype=float)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
