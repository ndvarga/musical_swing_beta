from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple, List
from .config import EnvelopeConfig, ControlConfig, Mode


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def smoothstep(x: float) -> float:
    return x * x * (3.0 - 2.0 * x)


@dataclass
class EnvelopeState:
    raw_feature: float = 0.0
    envelope: float = 0.0
    x_norm: float = 0.0


class IntensityEstimator:
    def __init__(self, cfg: EnvelopeConfig):
        self.cfg = cfg
        self.state = EnvelopeState()

    def update_floor_max(self, floor: float, max_val: float) -> None:
        self.cfg.floor = float(floor)
        self.cfg.max_val = float(max_val)

    def update(self, raw_feature: float, dt: float) -> EnvelopeState:
        if raw_feature is None or (isinstance(raw_feature, float) and math.isnan(raw_feature)):
            raw_feature = 0.0

        self.state.raw_feature = float(raw_feature)
        env = self.state.envelope
        target = float(raw_feature)

        tau = self.cfg.attack_tau_s if target > env else self.cfg.release_tau_s
        tau = max(tau, 1e-4)
        alpha = 1.0 - math.exp(-dt / tau)

        env_new = env + alpha * (target - env)
        self.state.envelope = env_new

        denom = max(self.cfg.max_val - self.cfg.floor, 1e-6)
        x = (env_new - self.cfg.floor) / denom
        self.state.x_norm = clamp(x, 0.0, 1.0)
        return self.state


@dataclass
class LayerState:
    active: bool = False
    on_timer: float = 0.0
    off_timer: float = 0.0
    transitions: int = 0


@dataclass
class ControlOutput:
    mode: Mode
    gains: Tuple[float, float, float]
    master_cap: float
    x_norm: float
    quiet: bool
    chatter_flag: bool
    status: str


class Controller:
    def __init__(self, cfg: ControlConfig):
        self.cfg = cfg
        self.layers: List[LayerState] = [LayerState() for _ in range(3)]
        self._transition_times: List[float] = []
        self._t = 0.0

    def reset(self) -> None:
        self.layers = [LayerState() for _ in range(3)]
        self._transition_times.clear()
        self._t = 0.0

    def _ramp(self, x: float, x0: float, x1: float) -> float:
        if x <= x0:
            return 0.0
        if x >= x1:
            return 1.0
        u = (x - x0) / max(x1 - x0, 1e-6)
        return smoothstep(u) if self.cfg.curve == "smoothstep" else u

    def _update_layer_state(self, idx: int, x: float, T_on: float, T_off: float, dt: float) -> None:
        layer = self.layers[idx]
        desired_on = x >= T_on
        desired_off = x <= T_off

        if not layer.active:
            if desired_on:
                layer.on_timer += dt
                if layer.on_timer >= self.cfg.on_dwell_s:
                    layer.active = True
                    layer.on_timer = 0.0
                    layer.off_timer = 0.0
                    layer.transitions += 1
                    self._transition_times.append(self._t)
            else:
                layer.on_timer = 0.0
        else:
            if desired_off:
                layer.off_timer += dt
                if layer.off_timer >= self.cfg.off_dwell_s:
                    layer.active = False
                    layer.off_timer = 0.0
                    layer.on_timer = 0.0
                    layer.transitions += 1
                    self._transition_times.append(self._t)
            else:
                layer.off_timer = 0.0

    def _chatter_detect(self) -> bool:
        window_s = 6.0
        self._transition_times = [tt for tt in self._transition_times if (self._t - tt) <= window_s]
        return len(self._transition_times) >= 10

    def update(self, x: float, dt: float) -> ControlOutput:
        self._t += dt
        quiet = self.cfg.quiet_below_T1 and x < self.cfg.T1

        if self.cfg.mode == "single":
            gain = 0.0 if quiet else self._ramp(x, self.cfg.T1, 1.0)
            chatter = self._chatter_detect()
            status = "Audio muted (below T1)" if quiet else "Single-track active"
            if chatter:
                status += " | Chatter detected"
            return ControlOutput("single", (gain, 0.0, 0.0), self.cfg.master_cap, x, quiet, chatter, status)

        T_on = [self.cfg.T1, self.cfg.T2, self.cfg.T3]
        T_off = [self.cfg.T1 - self.cfg.hysteresis, self.cfg.T2 - self.cfg.hysteresis, self.cfg.T3 - self.cfg.hysteresis]

        for i in range(3):
            self._update_layer_state(i, x, T_on[i], T_off[i], dt)

        if quiet:
            gains = (0.0, 0.0, 0.0)
        else:
            g1 = self._ramp(x, self.cfg.T1, self.cfg.T2) if self.layers[0].active else 0.0
            g2 = self._ramp(x, self.cfg.T2, self.cfg.T3) if self.layers[1].active else 0.0
            g3 = self._ramp(x, self.cfg.T3, 1.0) if self.layers[2].active else 0.0
            gains = (g1, g2, g3)

        chatter = self._chatter_detect()
        status = "Audio muted (below T1)" if quiet else f"Layered active: {','.join([str(i+1) for i,L in enumerate(self.layers) if L.active]) or 'None'}"
        if chatter:
            status += " | Chatter detected"

        return ControlOutput("layered", gains, self.cfg.master_cap, x, quiet, chatter, status)
