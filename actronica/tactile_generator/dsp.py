"""DSP core: oscillators, modulation, burst gating, spatial patterns, one-shot effects.

Everything here is pure numpy and runs inside the audio callback, so it is fully
vectorised per block and keeps its own phase state for click-free playback.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, fields

import numpy as np

TWO_PI = 2.0 * math.pi

WAVEFORMS = ["sine", "square", "triangle", "saw", "noise", "rumble", "audio"]
WAVE_LABELS = {
    "sine": "Sine",
    "square": "Square",
    "triangle": "Triangle",
    "saw": "Saw",
    "noise": "Noise",
    "rumble": "Rumble",
    "audio": "Audio in",
}
MODULATIONS = ["none", "am", "fm", "sweep"]
SPATIAL_MODES = ["static", "chase", "pingpong", "random"]


# --------------------------------------------------------------------------- params
@dataclass
class GenParams:
    enabled: bool = True
    waveform: str = "sine"
    freq: float = 65.0          # Hz (for "rumble": rate of random points)
    amp: float = 0.4            # 0..1
    duty: float = 0.5           # square: duty cycle
    softness: float = 0.3       # square / saw: edge rounding (0 = hard)
    symmetry: float = 0.5       # triangle: rise fraction
    phase: float = 0.0          # degrees
    mod: str = "none"           # none | am | fm | sweep
    mod_rate: float = 2.0       # Hz (AM / FM)
    am_depth: float = 1.0       # 0..1
    fm_dev: float = 20.0        # Hz
    sweep_to: float = 300.0     # Hz (sweep goes freq -> sweep_to)
    sweep_time: float = 5.0     # s
    sweep_pingpong: bool = False
    gate: bool = False          # burst / pulse-train gating
    gate_on_ms: float = 60.0
    gate_off_ms: float = 140.0
    gate_sync: bool = True      # restart waveform phase at each burst
    # audio in (music -> haptics); freq is the carrier for envelope / beats
    audio_mode: str = "envelope"  # bass | envelope | beats
    audio_gain: float = 0.0       # dB
    audio_auto: bool = True       # normalise loud / quiet tracks
    audio_cutoff: float = 120.0   # Hz, low-pass before analysis
    audio_smooth: float = 80.0    # ms, envelope release / hit decay
    audio_floor: float = 0.06     # gate, 0..1
    audio_beat: float = 1.6       # onset threshold (fast / slow energy)

    def to_dict(self) -> dict:
        return asdict(self)

    def update(self, d: dict | None) -> None:
        types = {f.name: type(getattr(self, f.name)) for f in fields(self)}
        for k, v in (d or {}).items():
            if k in types and v is not None:
                setattr(self, k, types[k](v))

    @classmethod
    def from_dict(cls, d: dict | None) -> "GenParams":
        p = cls()
        p.update(d)
        return p


# --------------------------------------------------------------------------- shapes
_pulse_cache: dict = {}


def _soft_k(softness: float):
    """Edge steepness for the tanh-shaped square. None = perfectly hard edges."""
    if softness <= 0.005:
        return None
    return 10.0 ** (2.2 * (1.0 - softness))


def _pulse_raw(p, duty, k):
    # cos(2pi(p - d/2)) > cos(pi d)  <=>  p in (0, d)   (mod 1)
    x = np.cos(TWO_PI * (p - duty * 0.5)) - math.cos(math.pi * duty)
    if k is None:
        return np.where(x > 0.0, 1.0, -1.0)
    return np.tanh(k * x)


def pulse(p, duty=0.5, softness=0.0):
    """Square / pulse wave, DC-free (important for voice-coil actuators), peak-normalised."""
    duty = min(0.98, max(0.02, float(duty)))
    k = _soft_k(float(softness))
    key = (round(duty, 3), None if k is None else round(k, 2))
    norm = _pulse_cache.get(key)
    if norm is None:
        g = np.arange(4096) / 4096.0
        y = _pulse_raw(g, duty, k)
        m = float(y.mean())
        pk = float(np.abs(y - m).max()) or 1.0
        if len(_pulse_cache) > 1024:
            _pulse_cache.clear()
        norm = _pulse_cache[key] = (m, pk)
    m, pk = norm
    return (_pulse_raw(p, duty, k) - m) / pk


def triangle(p, symmetry=0.5):
    s = min(0.995, max(0.005, float(symmetry)))
    return np.where(p < s, -1.0 + 2.0 * p / s, 1.0 - 2.0 * (p - s) / (1.0 - s))


def saw(p, softness=0.0):
    fall = 0.01 + 0.49 * min(1.0, max(0.0, float(softness)))
    return triangle(p, 1.0 - fall)


def shape(waveform: str, p, params: GenParams):
    """Periodic waveform evaluated at phase p in [0, 1)."""
    if waveform == "square":
        return pulse(p, params.duty, params.softness)
    if waveform == "triangle":
        return triangle(p, params.symmetry)
    if waveform == "saw":
        return saw(p, params.softness)
    return np.sin(TWO_PI * p)


# --------------------------------------------------------------------------- generator
class Generator:
    """Stateful block generator. `params` may be edited live from the GUI thread."""

    def __init__(self, params: GenParams | None = None, seed: int | None = None):
        self.params = params or GenParams()
        self.rng = np.random.default_rng(seed)
        self.audio = None  # AudioFollower, attached by the engine
        self.reset()

    def reset(self) -> None:
        self.phase = 0.0
        self.mod_phase = 0.0
        self.sweep_t = 0.0
        self.gate_t = 0.0
        self.amp_now = 0.0
        self.inst_freq = float(self.params.freq)
        self.gate_open = True
        self._rv = self.rng.uniform(-1.0, 1.0, 2)

    def restart(self) -> None:
        """Restart sweep / burst / modulation timing (keeps amplitude to avoid clicks)."""
        self.sweep_t = 0.0
        self.gate_t = 0.0
        self.mod_phase = 0.0

    # value noise: random points at `freq` Hz, smoothstep-interpolated
    def _rumble(self, ph, ph_end):
        K = int(math.floor(ph_end))
        v = np.concatenate([self._rv, self.rng.uniform(-1.0, 1.0, K)]) if K > 0 else self._rv
        k = np.floor(ph).astype(np.int64)
        fr = ph - k
        fr = fr * fr * (3.0 - 2.0 * fr)
        y = v[k] + (v[k + 1] - v[k]) * fr
        self._rv = v[K:K + 2].copy()
        return y

    def render(self, n: int, sr: float) -> np.ndarray:
        p = self.params
        dt = 1.0 / sr
        idx = np.arange(n, dtype=np.float64)
        f0 = max(0.1, float(p.freq))
        is_audio = p.waveform == "audio"
        mod = "none" if is_audio else p.mod  # Audio in has no modulation / burst of its own

        # ---- instantaneous frequency
        freq = None
        if mod == "fm":
            rate = max(0.01, float(p.mod_rate))
            mp = self.mod_phase + idx * (rate * dt)
            self.mod_phase = (self.mod_phase + n * rate * dt) % 1.0
            freq = np.maximum(f0 + float(p.fm_dev) * np.sin(TWO_PI * mp), 0.0)
        elif mod == "sweep":
            T = max(0.05, float(p.sweep_time))
            tt = self.sweep_t + idx * dt
            if p.sweep_pingpong:
                x = np.mod(tt, 2.0 * T) / T
                pos = np.where(x <= 1.0, x, 2.0 - x)
            else:
                pos = np.mod(tt, T) / T
            f2 = max(0.1, float(p.sweep_to))
            freq = f0 * (f2 / f0) ** pos
            self.sweep_t = (self.sweep_t + n * dt) % (2.0 * T)

        # ---- phase accumulation (in cycles)
        ph0 = self.phase
        if freq is None:
            inc = f0 * dt
            ph = ph0 + idx * inc
            ph_end = ph0 + n * inc
            self.inst_freq = f0
        else:
            incs = freq * dt
            cs = np.cumsum(incs)
            ph = ph0 + (cs - incs)
            ph_end = ph0 + float(cs[-1])
            self.inst_freq = float(freq[-1])
        self.phase = ph_end % 1.0

        # ---- burst gate
        env = None
        sync_ph = None
        if p.gate and p.gate_off_ms > 0.1 and not is_audio:
            on = max(1.0, float(p.gate_on_ms)) / 1000.0
            off = float(p.gate_off_ms) / 1000.0
            per = on + off
            gt = np.mod(self.gate_t + idx * dt, per)
            self.gate_t = (self.gate_t + n * dt) % per
            fade = min(0.003, on * 0.25)
            e = np.clip(np.minimum(gt, on - gt) / fade, 0.0, 1.0)
            env = 0.5 - 0.5 * np.cos(math.pi * e)
            self.gate_open = bool(self.gate_t < on)
            if p.gate_sync and freq is None:
                sync_ph = gt * f0
        else:
            self.gate_open = True

        # ---- waveform
        wf = p.waveform
        if wf == "noise":
            y = self.rng.uniform(-1.0, 1.0, n)
        elif wf == "rumble":
            y = self._rumble(ph, ph_end)
        elif wf == "audio":
            ctrl = self.audio.render(n, sr, p) if self.audio is not None else np.zeros(n)
            if p.audio_mode == "bass":
                y = ctrl
            else:
                y = np.sin(TWO_PI * np.mod(ph, 1.0)) * ctrl
        else:
            base = sync_ph if sync_ph is not None else ph
            y = shape(wf, np.mod(base + float(p.phase) / 360.0, 1.0), p)

        # ---- amplitude modulation
        if mod == "am":
            rate = max(0.01, float(p.mod_rate))
            mp = self.mod_phase + idx * (rate * dt)
            self.mod_phase = (self.mod_phase + n * rate * dt) % 1.0
            y = y * (1.0 - float(p.am_depth) * 0.5 * (1.0 + np.cos(TWO_PI * mp)))

        if env is not None:
            y = y * env

        # ---- smoothed amplitude (no zipper noise / clicks)
        target = float(p.amp) if p.enabled else 0.0
        if abs(target - self.amp_now) > 1e-7:
            y = y * np.linspace(self.amp_now, target, n + 1)[1:]
            self.amp_now = target
        else:
            y = y * target
        return y


def preview(params: GenParams, sr: float, n: int) -> np.ndarray:
    g = Generator(copy.copy(params), seed=1)
    g.amp_now = float(params.amp) if params.enabled else 0.0
    return g.render(n, sr)


# --------------------------------------------------------------------------- spatial
class Spatial:
    """Moves the signal across the routed channels (chase / ping-pong / random)."""

    def __init__(self):
        self.mode = "static"
        self.step_ms = 200.0
        self.overlap = 0.3
        self.t = 0.0
        self._rng = np.random.default_rng(7)
        self._rk = None
        self._rcur = 0
        self._rprev = 0
        self.position = 0.0  # for GUI

    def to_dict(self):
        return {"mode": self.mode, "step_ms": self.step_ms, "overlap": self.overlap}

    def update(self, d):
        d = d or {}
        if d.get("mode") in SPATIAL_MODES:
            self.mode = d["mode"]
        if "step_ms" in d:
            self.step_ms = float(d["step_ms"])
        if "overlap" in d:
            self.overlap = float(d["overlap"])

    @staticmethod
    def _bump(d, overlap):
        x = max(0.02, min(1.0, overlap))
        w = np.clip((0.5 + 0.5 * x - d) / x, 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(math.pi * w)

    def render(self, n: int, sr: float, active: np.ndarray, n_ch: int):
        """Returns (n, n_ch) per-sample weights or None for static routing."""
        m = len(active)
        if self.mode == "static" or m < 2:
            self.t += n / sr
            return None
        step = max(0.005, self.step_ms / 1000.0)
        pos = (self.t + np.arange(n) / sr) / step
        self.t = (self.t + n / sr) % 36000.0
        centers = np.arange(m) + 0.5

        if self.mode == "chase":
            c = np.mod(pos, m)
            d = np.abs(c[:, None] - centers[None, :])
            d = np.minimum(d, m - d)
            W = self._bump(d, self.overlap)
            self.position = float(c[-1])
        elif self.mode == "pingpong":
            L = m - 1
            q = np.mod(pos, 2 * L)
            c = np.where(q <= L, q, 2 * L - q) + 0.5
            W = self._bump(np.abs(c[:, None] - centers[None, :]), self.overlap)
            self.position = float(c[-1])
        else:  # random
            ks = np.floor(pos).astype(np.int64)
            uniq = np.unique(ks)
            if self._rk is None or uniq[0] - self._rk > 4 or uniq[0] < self._rk:
                self._rk = int(uniq[0]) - 1
            cur_t, prev_t = [], []
            for k in uniq:
                while self._rk < k:
                    self._rprev = self._rcur % m
                    self._rcur = (self._rprev + 1 + int(self._rng.integers(m - 1))) % m
                    self._rk += 1
                cur_t.append(self._rcur % m)
                prev_t.append(self._rprev % m)
            sel = np.searchsorted(uniq, ks)
            cur = np.asarray(cur_t)[sel]
            prev = np.asarray(prev_t)[sel]
            x = max(0.02, min(1.0, self.overlap))
            r = np.clip((pos - ks) / x, 0.0, 1.0)
            r = 0.5 - 0.5 * np.cos(math.pi * r)
            W = np.zeros((n, m))
            rows = np.arange(n)
            W[rows, prev] += 1.0 - r
            W[rows, cur] += r
            self.position = float(cur[-1]) + 0.5

        out = np.zeros((n, n_ch))
        out[:, active] = W
        return out


# --------------------------------------------------------------------------- effects
def _t(sr, dur):
    return np.arange(int(round(dur * sr))) / sr


def _fade(y, sr, a=0.0015, r=0.004):
    na, nr = max(1, int(a * sr)), max(1, int(r * sr))
    y = y.copy()
    if len(y) > na + nr:
        y[:na] *= 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, na))
        y[-nr:] *= 0.5 + 0.5 * np.cos(np.linspace(0, math.pi, nr))
    return y


def _decay(sr, f, dur, tau):
    t = _t(sr, dur)
    return _fade(np.sin(TWO_PI * f * t) * np.exp(-t / tau), sr, 0.0008, 0.003)


def _tone(sr, f, dur, wf="sine", softness=0.4):
    t = _t(sr, dur)
    p = np.mod(f * t, 1.0)
    y = pulse(p, 0.5, softness) if wf == "square" else np.sin(TWO_PI * p)
    return _fade(y, sr, 0.003, 0.004)


def _gap(sr, dur):
    return np.zeros(int(round(dur * sr)))


def _chirp(sr, f1, f2, dur):
    t = _t(sr, dur)
    r = f2 / f1
    ph = f1 * dur / math.log(r) * (r ** (t / dur) - 1.0)
    return np.sin(TWO_PI * ph)


def _value_noise(sr, rate, dur, seed=3):
    rng = np.random.default_rng(seed)
    t = _t(sr, dur)
    u = t * rate
    v = rng.uniform(-1, 1, int(u[-1]) + 3)
    k = np.floor(u).astype(int)
    fr = u - k
    fr = fr * fr * (3 - 2 * fr)
    return v[k] + (v[k + 1] - v[k]) * fr


def _norm(y):
    pk = float(np.abs(y).max()) or 1.0
    return y / pk


def _hann(y):
    return y * np.hanning(len(y))


def _bounce(sr):
    parts, amp = [], 1.0
    for gap in (0.26, 0.19, 0.13, 0.09, 0.06, 0.04):
        parts += [amp * _decay(sr, 85, 0.04, 0.01), _gap(sr, gap)]
        amp *= 0.78
    return np.concatenate(parts)


def _ramp(sr, up=True):
    t = _t(sr, 0.7)
    env = (t / t[-1]) ** 2 if up else (1 - t / t[-1]) ** 2
    return _fade(np.sin(TWO_PI * 70 * t) * env, sr, 0.003, 0.006)


def _rev(sr):
    y = _chirp(sr, 30, 150, 0.9)
    t = _t(sr, 0.9)
    return _fade(y * np.sin(math.pi * t / 0.9) ** 0.6, sr)


# name -> (builder(sr) -> np.ndarray, keyboard key)
EFFECTS = {
    "Click": (lambda sr: _decay(sr, 90, 0.045, 0.010), "1"),
    "Tick": (lambda sr: _decay(sr, 180, 0.025, 0.004), "2"),
    "Double": (lambda sr: np.concatenate([_decay(sr, 90, 0.045, 0.01), _gap(sr, 0.07),
                                          _decay(sr, 90, 0.045, 0.01)]), "3"),
    "Thud": (lambda sr: _decay(sr, 50, 0.22, 0.06), "4"),
    "Heartbeat": (lambda sr: np.concatenate([_decay(sr, 55, 0.15, 0.04), _gap(sr, 0.1),
                                             0.65 * _decay(sr, 50, 0.18, 0.05)]), "5"),
    "Bounce": (_bounce, "6"),
    "Buzz": (lambda sr: _tone(sr, 150, 0.25, "square"), "7"),
    "Alert": (lambda sr: np.concatenate([_tone(sr, 120, 0.08), _gap(sr, 0.06),
                                         _tone(sr, 120, 0.08), _gap(sr, 0.06),
                                         _tone(sr, 120, 0.08)]), "8"),
    "Ramp up": (lambda sr: _ramp(sr, True), "9"),
    "Ramp down": (lambda sr: _ramp(sr, False), "0"),
    "Rumble": (lambda sr: _fade(_hann(_value_noise(sr, 32, 0.7)), sr), "-"),
    "Rev": (_rev, "="),
}


def render_effect(name: str, sr: float) -> np.ndarray:
    return _norm(EFFECTS[name][0](sr)).astype(np.float64)
