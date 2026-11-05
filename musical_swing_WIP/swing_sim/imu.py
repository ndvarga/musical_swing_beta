#Hello
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .config import IMUConfig


@dataclass
class IMUSample:
    t: float
    gyro_xyz: np.ndarray


class IMUSim:
    def __init__(self, cfg: IMUConfig, seed: int = 12345):
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(self, t: float, omega: float) -> IMUSample:
        if self.cfg.dropouts_enabled and self._rng.random() < (self.cfg.dropout_prob_per_s / max(self.cfg.sample_hz, 1)):
            return IMUSample(t=t, gyro_xyz=np.array([np.nan, np.nan, np.nan], dtype=float))

        gx = float(self._rng.normal(0.0, self.cfg.cross_axis_noise_std))
        gy = float(omega) + self.cfg.gyro_bias + float(self._rng.normal(0.0, self.cfg.gyro_noise_std))
        gz = float(self._rng.normal(0.0, self.cfg.cross_axis_noise_std))

        if self.cfg.spikes_enabled and self._rng.random() < (self.cfg.spike_prob_per_s / max(self.cfg.sample_hz, 1)):
            gy += float(self._rng.choice([-1.0, 1.0])) * self.cfg.spike_mag

        return IMUSample(t=t, gyro_xyz=np.array([gx, gy, gz], dtype=float))
