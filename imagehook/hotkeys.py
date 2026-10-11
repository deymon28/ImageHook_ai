"""Global key polling (works while the game has focus) and short confirmation beeps. Windows only."""
from __future__ import annotations

import ctypes
import re
import sys
import threading

_NAMED = {
    "mouse3": 0x04, "mmb": 0x04, "mouse4": 0x05, "xbutton1": 0x05, "mouse5": 0x06, "xbutton2": 0x06,
    "pause": 0x13, "capslock": 0x14, "space": 0x20, "pageup": 0x21, "pagedown": 0x22, "end": 0x23, "home": 0x24,
    "insert": 0x2D, "delete": 0x2E, "scrolllock": 0x91, "tab": 0x09, "backquote": 0xC0, "grave": 0xC0,
}


def parse_key(name: str) -> int:
    """Key name -> Windows virtual-key code: f1..f24, a..z, 0..9, numpad0..9, mouse3/4/5, named keys, or 0xNN."""
    n = name.strip().lower()
    if n in _NAMED:
        return _NAMED[n]
    if m := re.fullmatch(r"f(\d{1,2})", n):
        if 1 <= int(m[1]) <= 24:
            return 0x6F + int(m[1])
    if re.fullmatch(r"[a-z0-9]", n):
        return ord(n.upper())
    if m := re.fullmatch(r"numpad(\d)", n):
        return 0x60 + int(m[1])
    if re.fullmatch(r"0x[0-9a-f]{1,2}", n):
        return int(n, 16)
    raise ValueError(f"unknown key name: {name!r}")


class Keys:
    """Edge-triggered checks of several keys."""

    def __init__(self) -> None:
        self._state: dict[int, bool] = {}
        self._user32 = ctypes.windll.user32 if sys.platform == "win32" else None

    def pressed(self, vk: int) -> bool:
        if self._user32 is None:
            return False
        down = bool(self._user32.GetAsyncKeyState(vk) & 0x8000)
        fired = down and not self._state.get(vk, False)
        self._state[vk] = down
        return fired


def beep(freq: int, ms: int, enabled: bool = True, times: int = 1) -> None:
    if not enabled or sys.platform != "win32":
        return
    import winsound

    def run() -> None:
        for _ in range(times):
            winsound.Beep(freq, ms)

    threading.Thread(target=run, daemon=True).start()
