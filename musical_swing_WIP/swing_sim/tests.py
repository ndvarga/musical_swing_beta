"""
Headless validation checks for the signal chain (envelope + controller),
driven by synthetic gyro-feature traces instead of the real IMU.

Run:
  python -m swing_sim.tests
"""
from __future__ import annotations
import math
import random
from swing_sim.config import AppConfig
from swing_sim.control import IntensityEstimator, Controller


class Harness:
    def __init__(self):
        self.cfg = AppConfig()
        self.est = IntensityEstimator(self.cfg.env)
        self.ctrl = Controller(self.cfg.ctrl)
        self.control_dt = 1.0 / self.cfg.ui.ui_fps  # control runs once per UI tick
        self.t = 0.0
        self.last_out = None


def run_for_seconds(h: Harness, seconds: float, feature) -> None:
    """feature(t) -> raw intensity feature (|gyro| in rad/s) at time t."""
    for _ in range(int(seconds / h.control_dt)):
        env = h.est.update(feature(h.t), h.control_dt)
        h.last_out = h.ctrl.update(env.x_norm, h.control_dt)
        h.t += h.control_dt


def test_still_swing_is_quiet() -> None:
    # Hanging swing nudged by wind: gyro noise well under the envelope floor.
    rng = random.Random(12345)
    h = Harness()
    run_for_seconds(h, 10.0, lambda t: abs(rng.gauss(0.0, 0.03)) + 0.03 * abs(math.sin(2 * math.pi * 0.35 * t)))
    assert h.last_out is not None
    assert max(h.last_out.gains) < 1e-3


def test_strong_swing_activates_layer1() -> None:
    # Real swinging: |omega| follows |sin| of the ~0.45 Hz swing period.
    h = Harness()
    run_for_seconds(h, 12.0, lambda t: 2.0 * abs(math.sin(2 * math.pi * 0.45 * t)))
    assert h.last_out is not None
    assert h.last_out.gains[0] > 0.1


def test_old_config_with_simulator_sections_loads() -> None:
    old = '{"physics": {"damping": 0.2}, "imu": {"sample_hz": 1000}, "ctrl": {"T1": 0.3}, "audio": {"f1_hz": 220.0}}'
    cfg = AppConfig.from_json(old)
    assert cfg.ctrl.T1 == 0.3
    assert not hasattr(cfg, "physics")


if __name__ == "__main__":
    test_still_swing_is_quiet()
    test_strong_swing_activates_layer1()
    test_old_config_with_simulator_sections_loads()
    print("All tests passed.")
