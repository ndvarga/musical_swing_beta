from __future__ import annotations

import time
import math
import os
import json
from collections import deque
from typing import Deque, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .config import AppConfig
from .imu_hw import MPU6050Reader
from .control import IntensityEstimator, Controller
from .audio import AudioEngine
from .calibration import CalibrationManager, CalibrationResult
from .logging_utils import DataLogger
from .swing_widget import SwingWidget
# TODO: api servier not implemented
"""from .api_server import (
    create_app,
    start_api_server_thread,
    TelemetryHub,
    TelemetrySnapshot,
)"""
from .speaker import SpeakerManager


# ---------------------------
# Utility helpers
# ---------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# ---------------------------
# Runtime: multi-rate loop
# ---------------------------

class AppRuntime(QtCore.QObject):
    """
    Owns the live hardware/audio objects and runs the control loop:

      - imu_hw.py samples the real IMU continuously on its own background
        thread, independent of everything below.
      - tick(), called once per Qt UI timer interval (~ui_fps Hz), reads
        whatever the latest IMU sample is and runs exactly one control +
        audio update per call, using the actually-measured wall-clock dt
        -- there's no fixed-step simulated loop anymore.
      - A stale IMU sample (sensor lost/disconnected) force-mutes audio
        rather than letting it keep playing a frozen gain.
      - Also starts the local API server (api_server.py) the keeper's
        phone app talks to, and feeds it a live telemetry snapshot once
        per tick.

    The Envelope tab visualizes:
      raw_feature = IMU intensity feature (see imu_hw.py's raw_feature_from)
      envelope    = asymmetric EMA(raw_feature)
      x_norm      = normalized intensity in [0,1]
    """

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg

        # Core components
        self.imu_hw = MPU6050Reader(sample_hz=200, feature_mode="magnitude")
        self.estimator = IntensityEstimator(cfg.env)
        self.controller = Controller(cfg.ctrl)
        self.audio = AudioEngine(cfg.audio)
        self.cal = CalibrationManager()
        self.logger = DataLogger()

        # Keeper-app bridge: live telemetry feed + the local API the phone
        # app talks to over WiFi. Replace the placeholder MAC address
        # below with your enclosure speaker's real address -- find it
        # with `bluetoothctl devices` after pairing.
        
        #  TODO: not implemented
        #  self.telemetry = TelemetryHub()
        self.speaker = SpeakerManager(speaker_mac="AA:BB:CC:DD:EE:FF", audio=self.audio)

        # TODO: not implemented
        """start_api_server_thread(
            create_app(self.cfg, self.audio, self.controller, self.telemetry, self.speaker),
            port=8000,
        )"""

        self.imu_hw.start()
        self.imu_hw.calibrate_bias(seconds=2.0)  # keep the swing still/hanging during this

        # Vestigial simulator-only controls (impulse slider/dropdown in
        # MainWindow still read/write these two; nothing acts on them
        # anymore now that real motion comes from imu_hw.py).
        self.impulse_strength = 1.2
        self.impulse_mode = "assist"

        # Recording
        self.recording = False

        # Pause
        self._paused = False

        # Timing bookkeeping. Unlike the simulator, there's no fixed-step
        # physics loop to catch up on here -- real hardware already runs
        # in real time. Control now runs once per UI tick, using the
        # actually-measured wall-clock dt each time (see tick()).
        self._last_wall = time.perf_counter()
        self._stale_timeout_s = 1.5  # force-mute if the IMU goes silent this long
        self.last_t = 0.0  # most recent real IMU sample's timestamp; calibration buttons read this

        self.control_hz = cfg.ui.ui_fps  # control now runs at the UI tick rate, not a fixed 200 Hz

        # Debug/timing
        self.last_dt_wall = 0.0
        self.ui_fps_est = 0.0
        self._fps_smooth = 0.0

        # Exported state (for UI). theta/omega have no real equivalent
        # with a gyro-only IMU -- getting an absolute angle would need
        # sensor fusion with the accelerometer, which this doesn't do --
        # so they stay at 0 and the on-screen pendulum just sits still
        # rather than drift from integrating noisy gyro data alone.
        self.theta = 0.0
        self.omega = 0.0
        self.gyro_y = 0.0

        self.raw_feature = 0.0
        self.envelope = 0.0
        self.x_norm = 0.0

        self.gains = (0.0, 0.0, 0.0)
        self.status = ""
        self.chatter = False

        # Plot buffers at CONTROL rate
        self.plot_seconds = cfg.ui.plot_seconds
        self.buf_len = int(self.plot_seconds * self.control_hz)

        self.t_buf: Deque[float] = deque(maxlen=self.buf_len)
        self.raw_buf: Deque[float] = deque(maxlen=self.buf_len)
        self.env_buf: Deque[float] = deque(maxlen=self.buf_len)
        self.x_buf: Deque[float] = deque(maxlen=self.buf_len)

        self.g1_buf: Deque[float] = deque(maxlen=self.buf_len)
        self.g2_buf: Deque[float] = deque(maxlen=self.buf_len)
        self.g3_buf: Deque[float] = deque(maxlen=self.buf_len)

        self.l1_active: Deque[float] = deque(maxlen=self.buf_len)
        self.l2_active: Deque[float] = deque(maxlen=self.buf_len)
        self.l3_active: Deque[float] = deque(maxlen=self.buf_len)

        # Start audio stream
        self.audio.start()

    @property
    def paused(self) -> bool:
        return self._paused

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)
        self.audio.set_fault_mute("paused", self._paused)

    def reset(self) -> None:
        # No simulated physics state to reset with real hardware -- this
        # now just clears the controller/estimator back to baseline.
        self.controller.reset()
        self.estimator.state.envelope = 0.0
        self.estimator.state.x_norm = 0.0

    def set_mode(self, mode: str) -> None:
        self.cfg.ctrl.mode = "single" if mode == "single" else "layered"

    def set_seed(self, seed: int) -> None:
        # Seeded randomness was a simulator-only concept (reproducible
        # fake wind/sensor noise). Left as a harmless no-op, since the
        # existing seed control in the UI still calls this -- safe to
        # remove that control later.
        self.cfg.physics.seed = int(seed)

    def set_drag_theta(self, theta: float) -> None:
        pass  # drag-to-swing was a simulator-only interaction; no real-hardware equivalent

    def set_dragging(self, dragging: bool) -> None:
        pass  # see set_drag_theta

    def apply_impulse(self) -> None:
        pass  # "kick the swing" was a simulator-only interaction; no real-hardware equivalent

    def update_parameters(self) -> None:
        """
        Sync live-edited config objects into running components.
        This is called every UI tick so changes take effect immediately.
        """
        self.controller.cfg = self.cfg.ctrl
        self.estimator.cfg = self.cfg.env
        # physics/imu config sync removed -- cfg.physics and cfg.imu now
        # only matter to the simulator path used by tests.py, not live
        # hardware.

        self.audio.set_master_cap(self.cfg.ctrl.master_cap)
        self.audio.set_safe_mute(self.cfg.audio.safe_mute)

    def start_recording(self) -> None:
        if not self.recording:
            path = self.logger.start(self.cfg)
            self.recording = True
            print(f"[LOG] Recording started: {path}")

    def stop_recording(self) -> None:
        if self.recording:
            self.logger.stop()
            self.recording = False
            print("[LOG] Recording stopped")

    def tick(self) -> None:
        """
        Called by UI timer. Reads the latest real IMU sample and runs one
        control/audio update. Unlike the simulator, real hardware already
        advances in real time on its own background thread (imu_hw.py),
        so there's no fixed-step catch-up loop here -- control simply
        runs once per UI tick, using the actually-measured wall-clock dt
        (dt_wall) as the time step, rather than a fixed simulated one.
        """
        now = time.perf_counter()
        dt_wall = now - self._last_wall
        self._last_wall = now

        # Cap wall dt to avoid a distorted dt if the OS pauses the app
        dt_wall = min(dt_wall, 0.10)

        self.last_dt_wall = dt_wall
        # FPS estimate (smoothed)
        inst_fps = 1.0 / max(dt_wall, 1e-6)
        self._fps_smooth = 0.9 * self._fps_smooth + 0.1 * inst_fps
        self.ui_fps_est = self._fps_smooth

        if self._paused:
            return

        sample = self.imu_hw.get_latest()
        stale = sample is None or self.imu_hw.seconds_since_last_sample() > self._stale_timeout_s
        if stale:
            # Fail-safe: a lost/disconnected sensor mutes rather than lets
            # audio keep playing whatever gain was last computed.
            self.audio.set_fault_mute("imu", True)
            self.status = "Waiting for first IMU sample..." if sample is None else "IMU signal lost -- audio muted"
            return

        self.audio.set_fault_mute("imu", False)

        raw_feature = self.imu_hw.raw_feature_from(sample)
        t_now = sample.t
        self.last_t = t_now

        # Kept for CSV/status continuity; with feature_mode="magnitude"
        # this is no longer literally the Y-axis gyro reading -- see
        # imu_hw.py's raw_feature_from().
        self.gyro_y = raw_feature

        # Calibration collection
        self.cal.update(t_now, raw_feature)

        env_state = self.estimator.update(raw_feature, dt_wall)
        self.raw_feature = env_state.raw_feature
        self.envelope = env_state.envelope
        self.x_norm = env_state.x_norm

        out = self.controller.update(self.x_norm, dt_wall, wind_on=False)
        self.gains = out.gains
        self.status = out.status
        self.chatter = out.chatter_flag

        # Push to audio
        self.audio.set_master_cap(out.master_cap)
        self.audio.set_gains(out.gains)

        # Push to the keeper app's live telemetry feed
        # TODO: not implemented
        """self.telemetry.push(TelemetrySnapshot(
            t=t_now,
            x_norm=self.x_norm,
            raw_feature=self.raw_feature,
            envelope=self.envelope,
            gains=self.gains,
            status=self.status,
            quiet=out.quiet,
            chatter=self.chatter,
        ))"""

        # Plot buffers
        self.t_buf.append(t_now)
        self.raw_buf.append(self.raw_feature)
        self.env_buf.append(self.envelope)
        self.x_buf.append(self.x_norm)

        self.g1_buf.append(self.gains[0])
        self.g2_buf.append(self.gains[1])
        self.g3_buf.append(self.gains[2])

        self.l1_active.append(1.0 if self.controller.layers[0].active else 0.0)
        self.l2_active.append(1.0 if self.controller.layers[1].active else 0.0)
        self.l3_active.append(1.0 if self.controller.layers[2].active else 0.0)

        # CSV logging at control rate
        if self.recording:
            self.logger.log_row({
                "t": t_now,
                "mode": self.cfg.ctrl.mode,
                "wind_on": False,
                "theta": self.theta,
                "omega": self.omega,
                "gyro_y": self.gyro_y,
                "raw_feature": self.raw_feature,
                "envelope": self.envelope,
                "x_norm": self.x_norm,
                "gain1": self.gains[0],
                "gain2": self.gains[1],
                "gain3": self.gains[2],
                "master_cap": self.cfg.ctrl.master_cap,
                "quiet": out.quiet,
                "status": out.status,
                "calibration_state": self.cal.state,
            })


# ---------------------------
# Main window (full UI)
# ---------------------------

class MainWindow(QtWidgets.QMainWindow):
    """
    Full UI panels:
      - Controls: mode, wind, pause/reset, impulse, damping/length
      - Envelope: attack/release, thresholds, hysteresis, dwell, master cap, safe mute
      - Sensor realism: spikes/dropouts, seed
      - Calibration helper: baseline capture, active capture, apply/cancel
      - Recording: CSV toggle
      - Config export/import
      - Tabs: Swing view and Envelope visualization + layer timeline
      - Keyboard shortcuts: Space=pause/resume, R=reset, I=impulse, W=toggle wind
    """

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.setWindowTitle("Musical Swing Simulator (Local)")
        self.cfg = cfg
        self.rt = AppRuntime(cfg)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Horizontal)
        self.setCentralWidget(splitter)

        # Left column (controls) inside a scroll area
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(8, 8, 8, 8)
        
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_scroll.setWidget(left)
        
        # Usually better to allow horizontal bar "as needed" (in case labels are long)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        splitter.addWidget(left_scroll)


        # Right column (tabs)
        right = QtWidgets.QTabWidget()
        splitter.addWidget(right)
        
        # Make the left panel wider by default
        left_scroll.setMinimumWidth(360)          # try 320–450 to taste
        splitter.setSizes([380, 1000])            # [left_px, right_px] initial
        splitter.setCollapsible(0, False)         # prevent fully collapsing the left pane
        splitter.setStretchFactor(0, 0)           # left keeps its width reasonably
        splitter.setStretchFactor(1, 1)           # right takes resizing priority

        splitter.setStretchFactor(0, 0)  # left (scroll) stays reasonable
        splitter.setStretchFactor(1, 1)  # right expands/contracts more


        # --- Swing tab ---
        swing_tab = QtWidgets.QWidget()
        swing_layout = QtWidgets.QVBoxLayout(swing_tab)
        self.swing_widget = SwingWidget()
        swing_layout.addWidget(self.swing_widget)
        right.addTab(swing_tab, "Swing")

        self.swing_widget.thetaChanged.connect(self._on_drag_theta)
        self.swing_widget.dragStateChanged.connect(self._on_drag_state)

        # --- Envelope tab ---
        env_tab = QtWidgets.QWidget()
        env_layout = QtWidgets.QVBoxLayout(env_tab)

        self.plot = pg.PlotWidget(title="Envelope Visualization (Raw / Envelope / x)")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend()

        self.cur_raw = self.plot.plot([], [], pen=pg.mkPen(width=2), name="raw feature |gyro|")
        self.cur_env = self.plot.plot([], [], pen=pg.mkPen(width=2, style=QtCore.Qt.DashLine), name="envelope")
        self.cur_x = self.plot.plot([], [], pen=pg.mkPen(width=2, style=QtCore.Qt.DotLine), name="x (scaled)")

        # Threshold overlays (On thresholds)
        self.th1_on = pg.InfiniteLine(angle=0, pen=pg.mkPen((180, 50, 50), width=2))
        self.th2_on = pg.InfiniteLine(angle=0, pen=pg.mkPen((180, 50, 50), width=2))
        self.th3_on = pg.InfiniteLine(angle=0, pen=pg.mkPen((180, 50, 50), width=2))
        self.plot.addItem(self.th1_on)
        self.plot.addItem(self.th2_on)
        self.plot.addItem(self.th3_on)

        # Off-threshold overlays (hysteresis)
        self.th1_off = pg.InfiniteLine(angle=0, pen=pg.mkPen((180, 50, 50), width=1, style=QtCore.Qt.DashDotLine))
        self.th2_off = pg.InfiniteLine(angle=0, pen=pg.mkPen((180, 50, 50), width=1, style=QtCore.Qt.DashDotLine))
        self.th3_off = pg.InfiniteLine(angle=0, pen=pg.mkPen((180, 50, 50), width=1, style=QtCore.Qt.DashDotLine))
        self.plot.addItem(self.th1_off)
        self.plot.addItem(self.th2_off)
        self.plot.addItem(self.th3_off)

        env_layout.addWidget(self.plot)

        # Layer active timeline
        self.plot_state = pg.PlotWidget(title="Layer Active Timeline (last window)")
        self.plot_state.showGrid(x=True, y=True, alpha=0.3)
        self.plot_state.setYRange(-0.2, 3.6)
        self.cur_l1 = self.plot_state.plot([], [], pen=pg.mkPen(width=2), name="L1")
        self.cur_l2 = self.plot_state.plot([], [], pen=pg.mkPen(width=2), name="L2")
        self.cur_l3 = self.plot_state.plot([], [], pen=pg.mkPen(width=2), name="L3")
        env_layout.addWidget(self.plot_state)

        # Status labels
        self.status_label = QtWidgets.QLabel("Status: —")
        self.audio_label = QtWidgets.QLabel("Audio: —")
        self.timing_label = QtWidgets.QLabel("Timing: —")
        env_layout.addWidget(self.status_label)
        env_layout.addWidget(self.audio_label)
        env_layout.addWidget(self.timing_label)

        right.addTab(env_tab, "Envelope")

        # Build panels
        left_layout.addWidget(self._panel_controls())
        left_layout.addWidget(self._panel_signal_controls())
        left_layout.addWidget(self._panel_calibration())
        left_layout.addWidget(self._panel_recording())
        left_layout.addWidget(self._panel_config_io())
        left_layout.addStretch(1)

        # Timer (UI rate)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(int(1000 / self.cfg.ui.ui_fps))

        # Initial sync
        self.rt.update_parameters()

    # ---------------------------
    # Panels
    # ---------------------------

    def _panel_controls(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Controls")
        layout = QtWidgets.QGridLayout(g)
        row = 0

        # Mode
        layout.addWidget(QtWidgets.QLabel("Mode:"), row, 0)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["layered", "single"])
        self.mode_combo.setCurrentText(self.cfg.ctrl.mode)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.mode_combo.setToolTip("layered: 3 layers with thresholds; single: 1 track ramps with intensity.")
        layout.addWidget(self.mode_combo, row, 1)
        row += 1

        # Wind
        self.wind_check = QtWidgets.QCheckBox("Wind mode")
        self.wind_check.setChecked(self.cfg.physics.wind_enabled)
        self.wind_check.stateChanged.connect(lambda s: self._set_wind(bool(s)))
        self.wind_check.setToolTip("Adds small perturbations. Default thresholds should remain quiet in wind-only.")
        layout.addWidget(self.wind_check, row, 0, 1, 2)
        row += 1

        # Pause / Reset
        self.pause_btn = QtWidgets.QPushButton("Pause (Space)")
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.reset_btn = QtWidgets.QPushButton("Reset (R)")
        self.reset_btn.clicked.connect(self._reset)
        layout.addWidget(self.pause_btn, row, 0)
        layout.addWidget(self.reset_btn, row, 1)
        row += 1

        # Impulse
        self.impulse_btn = QtWidgets.QPushButton("Impulse / Actuate (I)")
        self.impulse_btn.clicked.connect(self._impulse)
        self.impulse_btn.setToolTip("Injects energy by changing omega instantly (delta-omega).")
        layout.addWidget(self.impulse_btn, row, 0, 1, 2)
        row += 1

        # Impulse strength
        layout.addWidget(QtWidgets.QLabel("Impulse strength:"), row, 0)
        self.impulse_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.impulse_slider.setRange(1, 400)
        self.impulse_slider.setValue(int(self.rt.impulse_strength * 100))
        self.impulse_slider.valueChanged.connect(self._impulse_strength_changed)
        self.impulse_slider.setToolTip("Scales how much omega increases per impulse.")
        layout.addWidget(self.impulse_slider, row, 1)
        row += 1

        # Impulse direction
        layout.addWidget(QtWidgets.QLabel("Impulse direction:"), row, 0)
        self.impulse_dir = QtWidgets.QComboBox()
        self.impulse_dir.addItems(["assist", "positive", "negative"])
        self.impulse_dir.setCurrentText(self.rt.impulse_mode)
        self.impulse_dir.currentTextChanged.connect(self._impulse_dir_changed)
        self.impulse_dir.setToolTip("assist: push along current motion; positive/negative: fixed direction.")
        layout.addWidget(self.impulse_dir, row, 1)
        row += 1

        # Damping
        layout.addWidget(QtWidgets.QLabel("Damping:"), row, 0)
        self.damping_spin = QtWidgets.QDoubleSpinBox()
        self.damping_spin.setRange(0.0, 2.0)
        self.damping_spin.setSingleStep(0.02)
        self.damping_spin.setValue(self.cfg.physics.damping)
        self.damping_spin.valueChanged.connect(self._damping_changed)
        self.damping_spin.setToolTip("Higher damping -> swing dies out faster.")
        layout.addWidget(self.damping_spin, row, 1)
        row += 1

        # Length
        layout.addWidget(QtWidgets.QLabel("Length (m):"), row, 0)
        self.length_spin = QtWidgets.QDoubleSpinBox()
        self.length_spin.setRange(0.3, 4.0)
        self.length_spin.setSingleStep(0.05)
        self.length_spin.setValue(self.cfg.physics.length_m)
        self.length_spin.valueChanged.connect(self._length_changed)
        self.length_spin.setToolTip("Longer length -> slower natural period.")
        layout.addWidget(self.length_spin, row, 1)
        row += 1

        return g

    def _panel_signal_controls(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Signal + Controller Parameters")
        layout = QtWidgets.QGridLayout(g)
        row = 0

        # Attack / release
        layout.addWidget(QtWidgets.QLabel("Attack τ (s):"), row, 0)
        self.attack_spin = QtWidgets.QDoubleSpinBox()
        self.attack_spin.setRange(0.05, 5.0)
        self.attack_spin.setSingleStep(0.05)
        self.attack_spin.setValue(self.cfg.env.attack_tau_s)
        self.attack_spin.valueChanged.connect(self._attack_changed)
        self.attack_spin.setToolTip("How fast envelope rises when motion increases.")
        layout.addWidget(self.attack_spin, row, 1)
        row += 1

        layout.addWidget(QtWidgets.QLabel("Release τ (s):"), row, 0)
        self.release_spin = QtWidgets.QDoubleSpinBox()
        self.release_spin.setRange(0.1, 12.0)
        self.release_spin.setSingleStep(0.1)
        self.release_spin.setValue(self.cfg.env.release_tau_s)
        self.release_spin.valueChanged.connect(self._release_changed)
        self.release_spin.setToolTip("How slow envelope falls. Larger prevents dropouts at arc endpoints.")
        layout.addWidget(self.release_spin, row, 1)
        row += 1

        # Thresholds
        self.T1_spin = self._add_thr_spin(layout, row, "T1:", self.cfg.ctrl.T1, self._T1_changed, "Layer 1 ON threshold in normalized x.")
        row += 1
        self.T2_spin = self._add_thr_spin(layout, row, "T2:", self.cfg.ctrl.T2, self._T2_changed, "Layer 2 ON threshold.")
        row += 1
        self.T3_spin = self._add_thr_spin(layout, row, "T3:", self.cfg.ctrl.T3, self._T3_changed, "Layer 3 ON threshold.")
        row += 1

        # Hysteresis
        layout.addWidget(QtWidgets.QLabel("Hysteresis:"), row, 0)
        self.hys_spin = QtWidgets.QDoubleSpinBox()
        self.hys_spin.setRange(0.0, 0.30)
        self.hys_spin.setSingleStep(0.01)
        self.hys_spin.setValue(self.cfg.ctrl.hysteresis)
        self.hys_spin.valueChanged.connect(self._hys_changed)
        self.hys_spin.setToolTip("OFF threshold = ON threshold - hysteresis.")
        layout.addWidget(self.hys_spin, row, 1)
        row += 1

        # Dwell times
        layout.addWidget(QtWidgets.QLabel("On dwell (s):"), row, 0)
        self.on_dwell = QtWidgets.QDoubleSpinBox()
        self.on_dwell.setRange(0.0, 3.0)
        self.on_dwell.setSingleStep(0.05)
        self.on_dwell.setValue(self.cfg.ctrl.on_dwell_s)
        self.on_dwell.valueChanged.connect(self._on_dwell_changed)
        self.on_dwell.setToolTip("Require x above ON threshold for this duration before turning layer on.")
        layout.addWidget(self.on_dwell, row, 1)
        row += 1

        layout.addWidget(QtWidgets.QLabel("Off dwell (s):"), row, 0)
        self.off_dwell = QtWidgets.QDoubleSpinBox()
        self.off_dwell.setRange(0.0, 8.0)
        self.off_dwell.setSingleStep(0.1)
        self.off_dwell.setValue(self.cfg.ctrl.off_dwell_s)
        self.off_dwell.valueChanged.connect(self._off_dwell_changed)
        self.off_dwell.setToolTip("Require x below OFF threshold for this duration before turning layer off.")
        layout.addWidget(self.off_dwell, row, 1)
        row += 1

        # Master cap
        layout.addWidget(QtWidgets.QLabel("Master cap:"), row, 0)
        self.cap_spin = QtWidgets.QDoubleSpinBox()
        self.cap_spin.setRange(0.0, 1.0)
        self.cap_spin.setSingleStep(0.02)
        self.cap_spin.setValue(self.cfg.ctrl.master_cap)
        self.cap_spin.valueChanged.connect(self._cap_changed)
        self.cap_spin.setToolTip("Overall volume ceiling after mixing layers.")
        layout.addWidget(self.cap_spin, row, 1)
        row += 1

        # Safe mute
        self.safe_mute = QtWidgets.QCheckBox("Safe mute (immediate)")
        self.safe_mute.setChecked(self.cfg.audio.safe_mute)
        self.safe_mute.stateChanged.connect(lambda s: self._safe_mute_changed(bool(s)))
        self.safe_mute.setToolTip("Mute audio output immediately regardless of intensity.")
        layout.addWidget(self.safe_mute, row, 0, 1, 2)
        row += 1

        # Seed (deterministic wind + sensor noise)
        layout.addWidget(QtWidgets.QLabel("Seed:"), row, 0)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 999999999)
        self.seed_spin.setValue(self.cfg.physics.seed)
        self.seed_spin.valueChanged.connect(self._seed_changed)
        self.seed_spin.setToolTip("Deterministic seed for repeatable wind + noise patterns.")
        layout.addWidget(self.seed_spin, row, 1)
        row += 1

        # Sensor realism toggles
        self.spikes_check = QtWidgets.QCheckBox("Sensor spikes")
        self.spikes_check.setChecked(self.cfg.imu.spikes_enabled)
        self.spikes_check.stateChanged.connect(lambda s: self._spikes_changed(bool(s)))
        self.spikes_check.setToolTip("Inject occasional gyro spikes to test robustness.")
        layout.addWidget(self.spikes_check, row, 0)

        self.drop_check = QtWidgets.QCheckBox("Sensor dropouts")
        self.drop_check.setChecked(self.cfg.imu.dropouts_enabled)
        self.drop_check.stateChanged.connect(lambda s: self._dropouts_changed(bool(s)))
        self.drop_check.setToolTip("Inject occasional missing samples (NaNs).")
        layout.addWidget(self.drop_check, row, 1)
        row += 1

        return g

    def _panel_calibration(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Calibration Helper")
        layout = QtWidgets.QGridLayout(g)
        row = 0

        layout.addWidget(QtWidgets.QLabel("Baseline seconds:"), row, 0)
        self.base_secs = QtWidgets.QDoubleSpinBox()
        self.base_secs.setRange(1.0, 30.0)
        self.base_secs.setSingleStep(1.0)
        self.base_secs.setValue(6.0)
        self.base_secs.setToolTip("Collect near-still or wind-only for N seconds; sets floor robustly.")
        layout.addWidget(self.base_secs, row, 1)
        row += 1

        self.base_btn = QtWidgets.QPushButton("Start baseline capture")
        self.base_btn.clicked.connect(self._start_baseline)
        layout.addWidget(self.base_btn, row, 0, 1, 2)
        row += 1

        layout.addWidget(QtWidgets.QLabel("Active seconds:"), row, 0)
        self.act_secs = QtWidgets.QDoubleSpinBox()
        self.act_secs.setRange(1.0, 45.0)
        self.act_secs.setSingleStep(1.0)
        self.act_secs.setValue(8.0)
        self.act_secs.setToolTip("Collect vigorous motion for N seconds; sets max robustly.")
        layout.addWidget(self.act_secs, row, 1)
        row += 1

        self.act_btn = QtWidgets.QPushButton("Start active capture")
        self.act_btn.clicked.connect(self._start_active)
        layout.addWidget(self.act_btn, row, 0, 1, 2)
        row += 1

        self.cal_apply = QtWidgets.QPushButton("Apply suggested params")
        self.cal_apply.clicked.connect(self._apply_calibration)
        self.cal_cancel = QtWidgets.QPushButton("Cancel calibration")
        self.cal_cancel.clicked.connect(self._cancel_calibration)
        layout.addWidget(self.cal_apply, row, 0)
        layout.addWidget(self.cal_cancel, row, 1)
        row += 1

        self.cal_text = QtWidgets.QPlainTextEdit()
        self.cal_text.setReadOnly(True)
        self.cal_text.setPlaceholderText("Calibration results will appear here.")
        layout.addWidget(self.cal_text, row, 0, 1, 2)

        return g

    def _panel_recording(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Recording")
        layout = QtWidgets.QVBoxLayout(g)

        self.rec_check = QtWidgets.QCheckBox("Record CSV time series to ./logs/")
        self.rec_check.stateChanged.connect(lambda s: self._toggle_recording(bool(s)))
        self.rec_check.setToolTip("Logs theta/omega/gyro/envelope/x/gains/state at control rate.")
        layout.addWidget(self.rec_check)

        return g

    def _panel_config_io(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Config I/O")
        layout = QtWidgets.QGridLayout(g)

        self.export_btn = QtWidgets.QPushButton("Export config…")
        self.export_btn.clicked.connect(self._export_config)
        self.import_btn = QtWidgets.QPushButton("Import config…")
        self.import_btn.clicked.connect(self._import_config)

        layout.addWidget(self.export_btn, 0, 0)
        layout.addWidget(self.import_btn, 0, 1)

        return g

    def _add_thr_spin(self, layout, row, label, val, cb, tip):
        layout.addWidget(QtWidgets.QLabel(label), row, 0)
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.01)
        spin.setValue(val)
        spin.valueChanged.connect(cb)
        spin.setToolTip(tip)
        layout.addWidget(spin, row, 1)
        return spin

    # ---------------------------
    # UI loop
    # ---------------------------

    def _on_timer(self) -> None:
        self.rt.update_parameters()
        self.rt.tick()

        # Update swing view
        self.swing_widget.set_state(self.rt.theta, self.rt.omega)

        # Update plots
        self._update_plots()

        # Status labels
        self.status_label.setText(f"Status: {self.rt.status}  |  x={self.rt.x_norm:.3f}")
        muted = self.rt.audio.mute_reasons()
        mute_text = f"  |  muted: {', '.join(muted)}" if muted else ""
        self.audio_label.setText(f"Audio: {self.rt.audio.status.message}{mute_text}  |  gains={tuple(round(g,3) for g in self.rt.gains)}")
        self.timing_label.setText(f"Timing: UI fps~{self.rt.ui_fps_est:.1f} | last frame dt={self.rt.last_dt_wall*1000:.1f} ms")

        # If calibration finished, show it
        if self.rt.cal.result is not None:
            self.cal_text.setPlainText(self.rt.cal.result.explanation)

    def _update_plots(self) -> None:
        if len(self.rt.t_buf) < 2:
            return

        t = np.array(self.rt.t_buf)
        t0 = t[-1]
        tt = t - t0  # relative time window [-plot_seconds, 0]

        raw = np.array(self.rt.raw_buf)
        env = np.array(self.rt.env_buf)
        x = np.array(self.rt.x_buf)

        # Scale x into envelope space for overlay visibility
        scale = max(1.0, float(np.nanmax(env) + 1e-6))
        x_scaled = x * scale

        self.cur_raw.setData(tt, raw)
        self.cur_env.setData(tt, env)
        self.cur_x.setData(tt, x_scaled)

        # Threshold lines (ON/OFF) in x-space scaled into envelope plotting space
        T1_on = _clamp01(self.cfg.ctrl.T1)
        T2_on = _clamp01(self.cfg.ctrl.T2)
        T3_on = _clamp01(self.cfg.ctrl.T3)

        h = max(0.0, float(self.cfg.ctrl.hysteresis))
        T1_off = _clamp01(T1_on - h)
        T2_off = _clamp01(T2_on - h)
        T3_off = _clamp01(T3_on - h)

        self.th1_on.setPos(T1_on * scale)
        self.th2_on.setPos(T2_on * scale)
        self.th3_on.setPos(T3_on * scale)

        self.th1_off.setPos(T1_off * scale)
        self.th2_off.setPos(T2_off * scale)
        self.th3_off.setPos(T3_off * scale)

        # Layer active timeline: baseline offsets + 1.0 when active
        l1 = np.array(self.rt.l1_active) * 1.0
        l2 = np.array(self.rt.l2_active) * 1.0 + 1.2
        l3 = np.array(self.rt.l3_active) * 1.0 + 2.4
        self.cur_l1.setData(tt, l1)
        self.cur_l2.setData(tt, l2)
        self.cur_l3.setData(tt, l3)

    # ---------------------------
    # Callbacks: swing interaction
    # ---------------------------

    def _on_drag_theta(self, theta: float) -> None:
        self.rt.set_drag_theta(theta)

    def _on_drag_state(self, dragging: bool) -> None:
        self.rt.set_dragging(dragging)

    # ---------------------------
    # Callbacks: controls
    # ---------------------------

    def _on_mode_changed(self, mode: str) -> None:
        self.cfg.ctrl.mode = mode
        self.rt.set_mode(mode)

    def _set_wind(self, on: bool) -> None:
        self.cfg.physics.wind_enabled = bool(on)

    def _toggle_pause(self) -> None:
        self.rt.set_paused(not self.rt.paused)
        self.pause_btn.setText("Resume (Space)" if self.rt.paused else "Pause (Space)")

    def _reset(self) -> None:
        self.rt.reset()

    def _impulse(self) -> None:
        self.rt.apply_impulse()

    def _impulse_strength_changed(self, v: int) -> None:
        self.rt.impulse_strength = v / 100.0

    def _impulse_dir_changed(self, s: str) -> None:
        self.rt.impulse_mode = s

    def _damping_changed(self, v: float) -> None:
        self.cfg.physics.damping = float(v)

    def _length_changed(self, v: float) -> None:
        self.cfg.physics.length_m = float(v)

    def _attack_changed(self, v: float) -> None:
        self.cfg.env.attack_tau_s = float(v)

    def _release_changed(self, v: float) -> None:
        self.cfg.env.release_tau_s = float(v)

    def _T1_changed(self, v: float) -> None:
        self.cfg.ctrl.T1 = float(v)

    def _T2_changed(self, v: float) -> None:
        self.cfg.ctrl.T2 = float(v)

    def _T3_changed(self, v: float) -> None:
        self.cfg.ctrl.T3 = float(v)

    def _hys_changed(self, v: float) -> None:
        self.cfg.ctrl.hysteresis = float(v)

    def _on_dwell_changed(self, v: float) -> None:
        self.cfg.ctrl.on_dwell_s = float(v)

    def _off_dwell_changed(self, v: float) -> None:
        self.cfg.ctrl.off_dwell_s = float(v)

    def _cap_changed(self, v: float) -> None:
        self.cfg.ctrl.master_cap = float(v)

    def _safe_mute_changed(self, on: bool) -> None:
        self.cfg.audio.safe_mute = bool(on)

    def _seed_changed(self, v: int) -> None:
        self.cfg.physics.seed = int(v)
        self.rt.set_seed(int(v))

    def _spikes_changed(self, on: bool) -> None:
        self.cfg.imu.spikes_enabled = bool(on)

    def _dropouts_changed(self, on: bool) -> None:
        self.cfg.imu.dropouts_enabled = bool(on)

    def _toggle_recording(self, on: bool) -> None:
        if on:
            self.rt.start_recording()
        else:
            self.rt.stop_recording()

    # ---------------------------
    # Calibration helper
    # ---------------------------

    def _start_baseline(self) -> None:
        self.cal_text.setPlainText("Baseline capture running… keep the swing still.")
        self.rt.cal.start_baseline(self.rt.last_t, float(self.base_secs.value()))

    def _start_active(self) -> None:
        self.cal_text.setPlainText("Active capture running… swing it vigorously now.")
        self.rt.cal.start_active(self.rt.last_t, float(self.act_secs.value()))

    def _apply_calibration(self) -> None:
        res: Optional[CalibrationResult] = self.rt.cal.result
        if res is None:
            self.cal_text.setPlainText("No calibration result yet. Run baseline and active capture first.")
            return

        # Apply floor/max to envelope normalization
        self.cfg.env.floor = res.floor
        self.cfg.env.max_val = res.max_val

        # Convert suggested raw thresholds into normalized x thresholds:
        denom = max(res.max_val - res.floor, 1e-6)
        self.cfg.ctrl.T1 = _clamp01((res.T1 - res.floor) / denom)
        self.cfg.ctrl.T2 = _clamp01((res.T2 - res.floor) / denom)
        self.cfg.ctrl.T3 = _clamp01((res.T3 - res.floor) / denom)

        # Update UI controls
        self.T1_spin.setValue(self.cfg.ctrl.T1)
        self.T2_spin.setValue(self.cfg.ctrl.T2)
        self.T3_spin.setValue(self.cfg.ctrl.T3)

        # Persist calibration result locally
        _ensure_dir("logs")
        out_path = os.path.join("logs", f"calibration_{_timestamp()}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "floor": res.floor,
                "max_val": res.max_val,
                "suggested_raw_T1": res.T1,
                "suggested_raw_T2": res.T2,
                "suggested_raw_T3": res.T3,
                "applied_norm_T1": self.cfg.ctrl.T1,
                "applied_norm_T2": self.cfg.ctrl.T2,
                "applied_norm_T3": self.cfg.ctrl.T3,
                "explanation": res.explanation
            }, f, indent=2)

        self.cal_text.setPlainText(res.explanation + f"\n\nApplied. Saved calibration to {out_path}")

    def _cancel_calibration(self) -> None:
        self.rt.cal.cancel()
        self.cal_text.setPlainText("Calibration canceled.")

    # ---------------------------
    # Config export/import
    # ---------------------------

    def _export_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Config", "config.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.cfg.to_json())

    def _import_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import Config", "", "JSON (*.json)")
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
        cfg2 = AppConfig.from_json(s)

        # Replace config and update runtime pointer
        self.cfg = cfg2
        self.rt.cfg = cfg2

        # Update visible controls
        self.mode_combo.setCurrentText(cfg2.ctrl.mode)
        self.wind_check.setChecked(cfg2.physics.wind_enabled)
        self.damping_spin.setValue(cfg2.physics.damping)
        self.length_spin.setValue(cfg2.physics.length_m)

        self.attack_spin.setValue(cfg2.env.attack_tau_s)
        self.release_spin.setValue(cfg2.env.release_tau_s)

        self.T1_spin.setValue(cfg2.ctrl.T1)
        self.T2_spin.setValue(cfg2.ctrl.T2)
        self.T3_spin.setValue(cfg2.ctrl.T3)
        self.hys_spin.setValue(cfg2.ctrl.hysteresis)
        self.on_dwell.setValue(cfg2.ctrl.on_dwell_s)
        self.off_dwell.setValue(cfg2.ctrl.off_dwell_s)

        self.cap_spin.setValue(cfg2.ctrl.master_cap)
        self.safe_mute.setChecked(cfg2.audio.safe_mute)

        self.seed_spin.setValue(cfg2.physics.seed)
        self.spikes_check.setChecked(cfg2.imu.spikes_enabled)
        self.drop_check.setChecked(cfg2.imu.dropouts_enabled)

        # Ensure deterministic seed is applied to sim
        self.rt.set_seed(cfg2.physics.seed)

    # ---------------------------
    # Keyboard shortcuts
    # ---------------------------

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == QtCore.Qt.Key_Space:
            self._toggle_pause()
        elif key == QtCore.Qt.Key_R:
            self._reset()
        elif key == QtCore.Qt.Key_I:
            self._impulse()
        elif key == QtCore.Qt.Key_W:
            self.wind_check.setChecked(not self.wind_check.isChecked())
        else:
            super().keyPressEvent(event)


def run_app() -> None:
    app = QtWidgets.QApplication([])
    cfg = AppConfig()
    win = MainWindow(cfg)
    win.resize(1250, 820)
    win.show()
    app.exec()
