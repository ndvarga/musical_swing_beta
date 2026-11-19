from __future__ import annotations

import csv
import os
import threading
import time
from typing import Optional, Dict, Any
from .config import AppConfig, project_path

COLUMNS = [
    "t", "mode", "gyro_y",
    "raw_feature", "envelope", "x_norm", "gain1", "gain2", "gain3",
    "master_cap", "quiet", "status", "calibration_state",
]

FLUSH_EVERY_S = 1.0  # so a power cut on the swing loses at most ~1 s of rows


class DataLogger:
    """
    Writes one CSV row per control step, plus a snapshot of the config
    the run started with. Safe to call log_row() from a control thread
    while start()/stop() come from another (e.g. a web request).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._fp = None
        self._writer = None
        self._last_flush = 0.0
        self._t0 = 0.0
        self.path: Optional[str] = None
        self.rows = 0

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def elapsed(self) -> float:
        """Seconds since start(), 0 when not recording."""
        return time.monotonic() - self._t0 if self.recording else 0.0

    def start(self, cfg: AppConfig, out_dir: str = "logs") -> str:
        out_dir = str(project_path(out_dir))
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"run_{ts}.csv")
        cfg_path = os.path.join(out_dir, f"run_{ts}_config.json")
        with open(cfg_path, "w") as f:
            f.write(cfg.to_json())
        with self._lock:
            self._close()
            self._fp = open(path, "w", newline="")
            self._writer = csv.writer(self._fp)
            self._writer.writerow(COLUMNS)
            self._t0 = self._last_flush = time.monotonic()
            self.path = path
            self.rows = 0
        return path

    def log_row(self, row: Dict[str, Any]) -> None:
        with self._lock:
            if self._writer is None:
                return
            self._writer.writerow([
                row.get("t", 0.0),
                row.get("mode", ""),
                row.get("gyro_y", 0.0),
                row.get("raw_feature", 0.0),
                row.get("envelope", 0.0),
                row.get("x_norm", 0.0),
                row.get("gain1", 0.0),
                row.get("gain2", 0.0),
                row.get("gain3", 0.0),
                row.get("master_cap", 0.0),
                int(bool(row.get("quiet", False))),
                row.get("status", ""),
                row.get("calibration_state", ""),
            ])
            self.rows += 1
            now = time.monotonic()
            if now - self._last_flush >= FLUSH_EVERY_S:
                self._fp.flush()
                self._last_flush = now

    def stop(self) -> None:
        with self._lock:
            self._close()
            self.path = None

    def _close(self) -> None:
        if self._fp:
            self._fp.close()
        self._fp = None
        self._writer = None
