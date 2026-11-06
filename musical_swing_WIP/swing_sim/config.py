from __future__ import annotations

from dataclasses import dataclass, asdict, field, fields
from typing import Literal, Dict, List, Any, Optional
import json


Mode = Literal["layered", "single"]


@dataclass
class EnvelopeConfig:
    # Asymmetric EMA time constants (seconds)
    attack_tau_s: float = 0.35
    release_tau_s: float = 3.0

    # Normalization parameters (calibrated or manual)
    floor: float = 0.10
    max_val: float = 2.0


@dataclass
class ControlConfig:
    mode: Mode = "layered"

    # Thresholds on normalized x in [0,1]
    T1: float = 0.20
    T2: float = 0.45
    T3: float = 0.70

    hysteresis: float = 0.05          # off-threshold is (T - hysteresis)
    on_dwell_s: float = 0.35          # require above T_on for this long
    off_dwell_s: float = 1.25         # require below T_off for this long

    # Master cap (overall multiplier applied after mixing)
    master_cap: float = 0.35

    # Gain mapping curve shaping
    curve: Literal["linear", "smoothstep"] = "smoothstep"

    # Quiet regime: mute below T1
    quiet_below_T1: bool = True


@dataclass
class AudioConfig:
    enabled: bool = True
    sample_rate: int = 48000
    block_size: int = 512

    # Audio fade smoothing inside audio engine (seconds)
    gain_smoothing_tau_s: float = 0.08

    library_dir: str = "layers" # folder with layer files

    layer_files: List[Optional[str]] = field(default_factory=lambda: [None, None, None])

    noise_level: float = 0.08        # third layer can include noise feel

    safe_mute: bool = False


@dataclass
class UIConfig:
    ui_fps: int = 60
    plot_seconds: float = 12.0        # window length for envelope visualization


@dataclass
class AppConfig:
    env: EnvelopeConfig = field(default_factory=EnvelopeConfig)
    ctrl: ControlConfig = field(default_factory=ControlConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "AppConfig":
        """
        Tolerant of configs saved by older/newer versions: unknown keys
        (e.g. the old sine engine's f1_hz) are ignored, and missing
        sections or fields fall back to their defaults.
        """
        d: Dict[str, Any] = json.loads(s)
        cfg = AppConfig()
        for f in fields(AppConfig):
            section = d.get(f.name)
            if isinstance(section, dict):
                section_cls = type(getattr(cfg, f.name))
                setattr(cfg, f.name, section_cls(**_known_fields(section_cls, section)))
        return cfg


def _known_fields(cls: type, d: Dict[str, Any]) -> Dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}
