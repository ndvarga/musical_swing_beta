from __future__ import annotations

import csv
import os
import time
from typing import Optional, Dict, Any
from .config import AppConfig


class DataLogger:
    def __init__(self):
        self._fp = None
        self._writer = None
        self.path: Optional[str] = None

    def start(self, cfg: AppConfig, out_dir: str = "logs") -> str:
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"run_{ts}.csv")
        self._fp = open(path, "w", newline="")
        self._writer = csv.writer(self._fp)
        self._writer.writerow([
            "t","mode","gyro_y",
            "raw_feature","envelope","x_norm","gain1","gain2","gain3",
            "master_cap","quiet","status","calibration_state"
        ])
        cfg_path = os.path.join(out_dir, f"run_{ts}_config.json")
        with open(cfg_path, "w") as f:
            f.write(cfg.to_json())
        self.path = path
        return path

    def log_row(self, row: Dict[str, Any]) -> None:
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

    def stop(self) -> None:
        if self._fp:
            self._fp.close()
        self._fp = None
        self._writer = None
        self.path = None
