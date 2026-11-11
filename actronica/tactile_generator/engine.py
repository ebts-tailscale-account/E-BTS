"""Audio engine: owns the sounddevice stream, mixes generators + effects into N channels."""
from __future__ import annotations

import collections

import numpy as np

from .audio_in import AudioCapture, AudioFollower
from .dsp import GenParams, Generator, Spatial, preview, render_effect

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover
    sd = None
    SD_ERROR = exc
else:
    SD_ERROR = None

ROUTE_OFF, ROUTE_A, ROUTE_B, ROUTE_AB = 0, 1, 2, 3

_API_PRIORITY = ["Windows WASAPI", "Windows DirectSound", "MME", "Windows WDM-KS"]


def list_output_devices() -> list[dict]:
    if sd is None:
        return []
    apis = sd.query_hostapis()
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            out.append({
                "index": i,
                "name": d["name"],
                "hostapi": apis[d["hostapi"]]["name"],
                "channels": int(d["max_output_channels"]),
                "sr": int(d["default_samplerate"]),
            })
    return out


def pick_default_device(devs: list[dict], name: str | None = None, hostapi: str | None = None):
    """Prefer a remembered device, then Actronica HSD (WASAPI first), then the system default."""
    if name:
        for d in devs:
            if d["name"] == name and (hostapi is None or d["hostapi"] == hostapi):
                return d
    hsd = [d for d in devs if "HSD" in d["name"].upper()]
    if hsd:
        hsd.sort(key=lambda d: _API_PRIORITY.index(d["hostapi"]) if d["hostapi"] in _API_PRIORITY else 99)
        return hsd[0]
    try:
        default_out = sd.default.device[1]
        for d in devs:
            if d["index"] == default_out:
                return d
    except Exception:
        pass
    return devs[0] if devs else None


def refresh_portaudio() -> None:
    """Re-scan devices (PortAudio only sees hot-plugged hardware after a re-init)."""
    if sd is None:
        return
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        pass


class AudioEngine:
    SCOPE_LEN = 1 << 17

    def __init__(self, n_channels: int = 20):
        self.gens = [
            Generator(GenParams(), seed=1),
            Generator(GenParams(enabled=False, waveform="square", freq=40.0, amp=0.3), seed=2),
        ]
        self.capture = AudioCapture()
        for g in self.gens:
            g.audio = AudioFollower(self.capture)
        self.spatial = Spatial()
        self.master = 0.8
        self.limit = 0.95
        self.fx_gain = 0.8
        self.output_on = False
        self._master_now = 0.0

        self.sr = 48000
        self.blocksize = 1024
        self.stream = None
        self.device: dict | None = None
        self.exclusive_active = False

        self._fx_queue: collections.deque = collections.deque()
        self._fx_active: list = []
        self._fx_clear = False

        self.monitor_ch = 0
        self.scope = np.zeros((3, self.SCOPE_LEN), np.float32)
        self.scope_pos = 0

        self.clips = 0
        self.xruns = 0
        self.last_error: str | None = None
        self.peak = 0.0
        self.n_channels = 0
        self.set_channel_count(n_channels)

    # ------------------------------------------------------------------ routing
    def set_channel_count(self, n: int) -> None:
        route = np.full(n, ROUTE_A, np.int8)
        gain = np.ones(n)
        if self.n_channels:
            k = min(n, self.n_channels)
            route[:k] = self.route[:k]
            gain[:k] = self.gain[:k]
        self.route, self.gain = route, gain
        self._g_prev = np.zeros((3, n))
        self.meters = np.zeros(n)
        self.n_channels = n
        self.monitor_ch = min(self.monitor_ch, n - 1)

    def set_route(self, ch: int, code: int) -> None:
        r = self.route.copy()
        r[ch] = code
        self.route = r

    def set_routes(self, codes) -> None:
        r = np.asarray(codes, np.int8)[: self.n_channels]
        full = self.route.copy()
        full[: len(r)] = r
        self.route = full

    def set_gain(self, ch: int, g: float) -> None:
        a = self.gain.copy()
        a[ch] = min(1.0, max(0.0, g))
        self.gain = a

    # ------------------------------------------------------------------ stream
    @property
    def running(self) -> bool:
        return self.stream is not None and self.stream.active

    def start(self, device: dict, blocksize: int = 1024, exclusive: bool = True) -> str:
        if sd is None:
            raise RuntimeError(f"sounddevice is not available: {SD_ERROR}")
        self.stop()
        info = sd.query_devices(device["index"])
        sr = int(info["default_samplerate"])
        ch = int(info["max_output_channels"])
        self.sr, self.blocksize = sr, blocksize
        if ch != self.n_channels:
            self.set_channel_count(ch)
        self._g_prev[:] = 0.0
        self._master_now = 0.0
        self._fx_active = []
        self.xruns = self.clips = 0
        self.last_error = None

        attempts = []
        if "WASAPI" in device["hostapi"] and exclusive:
            attempts.append(("exclusive", sd.WasapiSettings(exclusive=True)))
        attempts.append(("shared", None))
        errors = []
        for label, extra in attempts:
            try:
                s = sd.OutputStream(device=device["index"], samplerate=sr, channels=ch,
                                    dtype="float32", blocksize=blocksize,
                                    callback=self._callback, extra_settings=extra)
                s.start()
            except Exception as exc:
                errors.append(f"{label}: {exc}")
                continue
            self.stream = s
            self.device = device
            self.exclusive_active = extra is not None
            note = "" if not errors else "  (exclusive failed → shared)"
            return f"Connected · {device['name']} · {device['hostapi']}{note}"
        raise RuntimeError("; ".join(errors))

    def stop(self) -> None:
        s, self.stream = self.stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass

    def stats(self) -> dict:
        s = self.stream
        if s is None:
            return {}
        try:
            return {"latency": float(s.latency), "cpu": float(s.cpu_load)}
        except Exception:
            return {}

    # ------------------------------------------------------------------ effects
    def trigger_fx(self, name: str, gain: float = 1.0) -> None:
        self._fx_queue.append(render_effect(name, self.sr) * gain)

    def clear_fx(self) -> None:
        self._fx_clear = True

    def _render_fx(self, n: int) -> np.ndarray:
        if self._fx_clear:
            self._fx_clear = False
            self._fx_active = []
            self._fx_queue.clear()
        while self._fx_queue:
            self._fx_active.append([self._fx_queue.popleft(), 0])
        out = np.zeros(n)
        keep = []
        for item in self._fx_active:
            buf, pos = item
            k = min(n, len(buf) - pos)
            out[:k] += buf[pos:pos + k]
            item[1] = pos + k
            if item[1] < len(buf):
                keep.append(item)
        self._fx_active = keep
        return out * self.fx_gain

    # ------------------------------------------------------------------ callback
    def _mix(self, n: int, a, b, fx):
        """Shared by the live callback; returns (n, C) mix before master/limit."""
        C = self.n_channels
        route, gain = self.route, self.gain
        tgt = np.stack([((route & ROUTE_A) > 0) * gain,
                        ((route & ROUTE_B) > 0) * gain,
                        (route > 0) * gain])
        if self._g_prev.shape != tgt.shape:
            self._g_prev = np.zeros_like(tgt)
        r = (np.arange(1, n + 1) / n)[:, None]
        prev = self._g_prev
        ga = prev[0] + (tgt[0] - prev[0]) * r
        gb = prev[1] + (tgt[1] - prev[1]) * r
        gf = prev[2] + (tgt[2] - prev[2]) * r
        self._g_prev = tgt
        mix = a[:, None] * ga + b[:, None] * gb + fx[:, None] * gf
        W = self.spatial.render(n, self.sr, np.flatnonzero(route > 0), C)
        if W is not None:
            mix *= W
        return mix

    def _callback(self, outdata, frames, time_info, status):
        if status and status.output_underflow:
            self.xruns += 1
        try:
            a = self.gens[0].render(frames, self.sr)
            b = self.gens[1].render(frames, self.sr)
            fx = self._render_fx(frames)
            mix = self._mix(frames, a, b, fx)
            if mix.shape[1] != outdata.shape[1]:
                outdata.fill(0)
                return
            target = self.master if self.output_on else 0.0
            m = np.linspace(self._master_now, target, frames + 1)[1:]
            self._master_now = target
            mix *= m[:, None]
            pk = float(np.abs(mix).max()) if mix.size else 0.0
            lim = self.limit
            if pk > lim:
                self.clips += 1
                np.clip(mix, -lim, lim, out=mix)
            self.peak = max(self.peak, min(pk, lim))
            outdata[:] = mix
            np.maximum(self.meters, np.abs(mix).max(axis=0), out=self.meters)
            mon = min(self.monitor_ch, mix.shape[1] - 1)
            self._scope_write(a, b, mix[:, mon], fx)
        except Exception as exc:  # never let the audio thread die
            outdata.fill(0)
            self.last_error = repr(exc)

    def _scope_write(self, a, b, o, fx):
        L = self.SCOPE_LEN
        n = len(a)
        pos = self.scope_pos % L
        end = pos + n
        rows = (a, b, o)
        if end <= L:
            for i, r in enumerate(rows):
                self.scope[i, pos:end] = r
        else:
            k = L - pos
            for i, r in enumerate(rows):
                self.scope[i, pos:] = r[:k]
                self.scope[i, : n - k] = r[k:]
        self.scope_pos += n

    def scope_snapshot(self, n: int) -> np.ndarray:
        n = min(n, self.SCOPE_LEN)
        idx = (self.scope_pos - n + np.arange(n)) % self.SCOPE_LEN
        return self.scope[:, idx].astype(np.float64)

    def preview(self, n: int) -> np.ndarray:
        """What the output would look like (no stream running): rows A, B, OUT[monitor]."""
        a = preview(self.gens[0].params, self.sr, n)
        b = preview(self.gens[1].params, self.sr, n)
        ch = min(self.monitor_ch, self.n_channels - 1)
        code, g = int(self.route[ch]), float(self.gain[ch])
        o = (a * bool(code & ROUTE_A) + b * bool(code & ROUTE_B)) * g * self.master
        return np.stack([a, b, np.clip(o, -self.limit, self.limit)])

    # ------------------------------------------------------------------ state
    def to_dict(self) -> dict:
        return {
            "A": self.gens[0].params.to_dict(),
            "B": self.gens[1].params.to_dict(),
            "route": self.route.tolist(),
            "gain": self.gain.tolist(),
            "spatial": self.spatial.to_dict(),
            "master": self.master,
            "limit": self.limit,
            "fx_gain": self.fx_gain,
            "monitor_ch": self.monitor_ch,
        }

    def load_dict(self, d: dict, routing: bool = True) -> None:
        for i, key in enumerate(("A", "B")):
            if key in d and d[key] is not None:
                self.gens[i].params.update(d[key])
        if routing and "route" in d:
            self.set_routes(d["route"])
        if routing and "gain" in d:
            g = np.asarray(d["gain"], float)[: self.n_channels]
            full = self.gain.copy()
            full[: len(g)] = g
            self.gain = full
        if "spatial" in d:
            self.spatial.update(d["spatial"])
        for k in ("master", "limit", "fx_gain"):
            if k in d:
                setattr(self, k, float(d[k]))
        if "monitor_ch" in d:
            self.monitor_ch = min(int(d["monitor_ch"]), self.n_channels - 1)
