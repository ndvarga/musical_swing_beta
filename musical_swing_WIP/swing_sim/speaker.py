from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:  # type-only, so speaker.py and audio.py never import each other at runtime
    from .audio import AudioEngine

_MAC_RE = re.compile(r"^([0-9A-F]{2}:){5}[0-9A-F]{2}$")
_PLACEHOLDER_MAC = "AA:BB:CC:DD:EE:FF"


@dataclass
class SpeakerStatus:
    mac: Optional[str]
    connected: bool
    message: str
    last_checked: float  # time.time() of the last connection check, 0 if never


class SpeakerManager:
    """
    Keeps the enclosure's Bluetooth speaker connected, by shelling out to
    bluetoothctl (BlueZ). A background thread checks the connection every
    poll_interval_s and, if auto_reconnect is on, reconnects when the
    speaker has dropped (power blip, out of range, speaker auto-off).

    When the speaker (re)connects, the audio stream is restarted so
    PortAudio picks up the Bluetooth sink fresh, rather than staying
    attached to a sink that disappeared while the speaker was gone.

    The speaker must already be paired and trusted once, by hand:
        bluetoothctl
        > scan on
        > pair XX:XX:XX:XX:XX:XX
        > trust XX:XX:XX:XX:XX:XX
    After that, this class only ever connects/disconnects.

    Like audio.py, this stays importable and harmless off the Pi: if
    bluetoothctl isn't installed, status() just reports that, and nothing
    raises.
    """

    def __init__(
        self,
        speaker_mac: Optional[str],
        audio: Optional["AudioEngine"] = None,
        poll_interval_s: float = 5.0,
        auto_reconnect: bool = True,
        command_timeout_s: float = 15.0,
        auto_start: bool = True,
    ):
        self.audio = audio
        self.poll_interval_s = float(poll_interval_s)
        self.auto_reconnect = bool(auto_reconnect)
        self.command_timeout_s = float(command_timeout_s)

        self._lock = threading.Lock()      # guards status fields below
        self._bt_lock = threading.Lock()   # serializes bluetoothctl calls across threads
        self._mac: Optional[str] = None
        self._connected = False
        self._message = "Speaker not checked yet"
        self._last_checked = 0.0

        self._bluetoothctl = shutil.which("bluetoothctl")
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.set_speaker_mac(speaker_mac)
        if auto_start:
            self.start()

    # ---------------------------
    # Lifecycle
    # ---------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="speaker-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=self.command_timeout_s + 1.0)
        self._thread = None

    # ---------------------------
    # Controls (safe from any thread, e.g. API request handlers)
    # ---------------------------

    def set_speaker_mac(self, mac: Optional[str]) -> None:
        """
        Sets (or clears, if mac is None/empty) which speaker to manage.
        Raises ValueError for a malformed address. The placeholder
        address from ui.py is treated as "no speaker configured".
        """
        if mac:
            mac = mac.strip().upper()
            if not _MAC_RE.match(mac):
                raise ValueError(f"Not a Bluetooth MAC address: {mac!r}")
            if mac == _PLACEHOLDER_MAC:
                mac = None
        with self._lock:
            self._mac = mac or None
            self._connected = False
            self._message = "No speaker configured" if self._mac is None else "Speaker not checked yet"
        if mac is None:
            self._set_audio_mute(False)  # nothing to wait for; play through whatever output exists
        self._wake.set()  # check the new speaker now rather than at the next poll

    def status(self) -> SpeakerStatus:
        with self._lock:
            return SpeakerStatus(
                mac=self._mac,
                connected=self._connected,
                message=self._message,
                last_checked=self._last_checked,
            )

    def connect(self) -> bool:
        """
        Connects now. Blocks for up to command_timeout_s. Returns success.
        Re-enables auto_reconnect, undoing a previous disconnect().
        """
        self.auto_reconnect = True
        mac = self._current_mac()
        if mac is None or not self._check_available():
            return False
        ok, out = self._bt("connect", mac)
        self._refresh(mac, note=None if ok else f"Connect failed: {self._last_line(out)}")
        return self.status().connected

    def disconnect(self) -> bool:
        """
        Disconnects now. Also turns auto_reconnect off, otherwise the
        monitor thread would just reconnect it at the next poll.
        """
        self.auto_reconnect = False
        mac = self._current_mac()
        if mac is None or not self._check_available():
            return False
        ok, out = self._bt("disconnect", mac)
        self._refresh(mac, note=None if ok else f"Disconnect failed: {self._last_line(out)}")
        return ok

    # ---------------------------
    # Internals
    # ---------------------------

    def _current_mac(self) -> Optional[str]:
        with self._lock:
            return self._mac

    def _set_status(self, connected: bool, message: str) -> None:
        with self._lock:
            self._connected = connected
            self._message = message
            self._last_checked = time.time()

    def _check_available(self) -> bool:
        if self._bluetoothctl is None:
            self._set_status(False, "bluetoothctl not found (install bluez)")
            return False
        return True

    def _bt(self, *args: str) -> Tuple[bool, str]:
        """Runs one bluetoothctl command. Returns (ok, full combined output)."""
        with self._bt_lock:
            try:
                p = subprocess.run(
                    [self._bluetoothctl, *args],
                    capture_output=True,
                    text=True,
                    timeout=self.command_timeout_s,
                )
            except subprocess.TimeoutExpired:
                return False, f"bluetoothctl {args[0]} timed out"
            except OSError as e:
                return False, str(e)
        return p.returncode == 0, (p.stdout + p.stderr).strip()

    @staticmethod
    def _last_line(out: str) -> str:
        return out.splitlines()[-1] if out else ""

    def _is_connected(self, mac: str) -> bool:
        ok, out = self._bt("info", mac)
        return ok and "Connected: yes" in out

    def _refresh(self, mac: str, note: Optional[str] = None) -> None:
        """Re-reads the connection state, restarting audio on a fresh connect."""
        was_connected = self.status().connected
        connected = self._is_connected(mac)
        if mac != self._current_mac():
            return  # speaker changed mid-check; the next poll handles the new one
        if connected:
            self._set_status(True, f"Speaker {mac} connected")
            if not was_connected:
                self._restart_audio()
        else:
            self._set_status(False, note or f"Speaker {mac} not connected")
        # Without its speaker, the system may route output to the Pi's own
        # jack/HDMI instead, so stay silent until the speaker is back.
        self._set_audio_mute(not connected)

    def _set_audio_mute(self, mute: bool) -> None:
        if self.audio is not None:
            self.audio.set_fault_mute("speaker", mute)

    def _restart_audio(self) -> None:
        if self.audio is None:
            return
        try:
            self.audio.stop()
            self.audio.start()
        except Exception as e:
            self._set_status(True, f"Speaker connected, but audio restart failed: {e}")

    def _run(self) -> None:
        while not self._stop.is_set():
            mac = self._current_mac()
            if mac is not None and self._check_available():
                self._refresh(mac)
                if not self.status().connected and self.auto_reconnect:
                    ok, out = self._bt("connect", mac)
                    self._refresh(mac, note=None if ok else f"Reconnect failed: {self._last_line(out)}")
            self._wake.wait(self.poll_interval_s)
            self._wake.clear()
