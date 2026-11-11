"""Audio in: capture system audio (WASAPI loopback) or a microphone and turn music into haptics.

Pipeline
    capture thread  ->  mono  ->  decimate to ~3 kHz  ->  shared ring buffer
    output callback ->  AudioFollower (one per generator) processes new samples
                        (high-pass, 4th-order low-pass, envelope / beat detection)
                    ->  resampled to the output rate with drift control

The capture and output devices run on different clocks, so every follower keeps its
own read position ~35 ms behind the writer and gently speeds up / slows down.
"""
from __future__ import annotations

import math
import threading
import warnings

import numpy as np

try:
    import soundcard as sc
except Exception as exc:  # optional dependency
    sc = None
    SC_ERROR = exc
else:
    SC_ERROR = None

try:
    import sounddevice as sd
except Exception:
    sd = None

AUDIO_MODES = ["bass", "envelope", "beats"]
AUDIO_MODE_LABELS = {"bass": "Bass", "envelope": "Envelope", "beats": "Beats"}
TARGET_RATE = 3000.0          # Hz, analysis rate after decimation (haptics only need < ~400 Hz)
RING = 1 << 15                # samples at TARGET_RATE (~11 s) in the capture ring buffer
CAPTURE_CHUNK_S = 0.005       # s, how much audio the capture thread reads per call

# analysis time constants (seconds unless noted)
DC_CUTOFF_HZ = 20.0           # high-pass removing DC / sub-sonic rumble before the low-pass
ATTACK_S = 0.004              # envelope attack
AUTO_LEVEL_DECAY_S = 4.0      # auto-level peak tracker forgets quiet/loud parts over ~4 s
BEAT_FAST_RELEASE_S = 0.025   # fast energy follower (transients)
BEAT_SLOW_S = 0.30            # slow energy average the transients are compared against
BEAT_REFRACTORY_S = 0.11      # minimum time between two detected beats
ENVELOPE_DRIVE = 1.3          # tanh drive applied to the gated envelope (0..1 output)

# resampling to the output clock
TARGET_LAG_S = 0.035          # how far the reader stays behind the capture writer
MAX_EXTRA_LAG_S = 0.08        # beyond target + this, jump forward instead of drifting
DRIFT_GAIN = 0.004            # max relative speed-up / slow-down used to stay at the target lag


def sc_missing() -> bool:
    """True when the optional `soundcard` package (system-audio loopback) is not installed."""
    return sc is None


def default_output_is_haptic() -> bool:
    """True when Windows sends ordinary sound (Spotify, browser, ...) to the HSD board.
    Then the music never reaches a speaker we can listen to: the HSD itself is excluded from
    capture, and while the generator holds it in exclusive mode other apps cannot play there."""
    try:
        if sc is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return _excluded(sc.default_speaker().name)
        if sd is not None:
            return _excluded(sd.query_devices(kind="output")["name"])
    except Exception:
        pass
    return False


def _excluded(name: str) -> bool:
    # never listen to the haptic board itself — that would be a feedback loop
    return "HSD" in name.upper()


def list_sources() -> list[dict]:
    """Loopback sources first (system audio), then microphones."""
    out: list[dict] = []
    if sc is not None:
        try:
            default = sc.default_speaker().name
        except Exception:
            default = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for m in sc.all_microphones(include_loopback=True):
                    if _excluded(m.name):
                        continue
                    out.append({"backend": "soundcard", "id": m.id, "name": m.name,
                                "kind": "loopback" if m.isloopback else "mic",
                                "default": bool(m.isloopback and m.name == default)})
        except Exception:
            pass
    elif sd is not None:
        try:
            apis = sd.query_hostapis()
            for i, d in enumerate(sd.query_devices()):
                api = apis[d["hostapi"]]["name"]
                if d["max_input_channels"] > 0 and api in ("Windows WASAPI", "Core Audio", "ALSA") \
                        and not _excluded(d["name"]):
                    out.append({"backend": "sounddevice", "id": i, "name": d["name"],
                                "kind": "loopback" if "stereo mix" in d["name"].lower() else "mic",
                                "default": False})
        except Exception:
            pass
    out.sort(key=lambda s: (s["kind"] != "loopback", not s["default"], s["name"]))
    return out


def source_label(s: dict) -> str:
    return f"{'System audio' if s['kind'] == 'loopback' else 'Mic'} · {s['name']}"


# ============================================================================ capture
class AudioCapture:
    def __init__(self):
        self.ring = np.zeros(RING, np.float32)
        self.write_pos = 0
        self.rate_d = TARGET_RATE
        self.source: dict | None = None
        self.active = False
        self.error: str | None = None
        self.level = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._carry = np.zeros(0, np.float32)
        self._factor = 16

    @property
    def running(self) -> bool:
        """Capture thread exists (it may still be opening the device)."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ control
    def start(self, source: dict, samplerate: int = 48000) -> None:
        self.stop()
        self.source = source
        self.error = None
        self._stop.clear()
        target = self._run_soundcard if source["backend"] == "soundcard" else self._run_sounddevice
        self._thread = threading.Thread(target=target, args=(source, samplerate), daemon=True,
                                        name="audio-in")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None and t.is_alive():
            t.join(timeout=1.5)
        self.active = False

    def _setup_rate(self, sr: float) -> None:
        self._factor = max(1, int(round(sr / TARGET_RATE)))
        self.rate_d = sr / self._factor
        self._carry = np.zeros(0, np.float32)

    def _push(self, mono: np.ndarray) -> None:
        if mono.size == 0:
            return
        self.level = max(self.level, float(np.abs(mono).max()))
        f = self._factor
        x = np.concatenate([self._carry, mono.astype(np.float32, copy=False)])
        k = len(x) // f
        self._carry = x[k * f:]
        if k == 0:
            return
        dec = x[: k * f].reshape(k, f).mean(axis=1)
        pos = self.write_pos % RING
        end = pos + k
        if end <= RING:
            self.ring[pos:end] = dec
        else:
            s = RING - pos
            self.ring[pos:] = dec[:s]
            self.ring[: k - s] = dec[s:]
        self.write_pos += k  # publish after the data is in place

    # ------------------------------------------------------------------ backends
    def _run_soundcard(self, source: dict, sr: int) -> None:
        com = False
        try:
            try:  # WASAPI needs COM initialised in every thread that touches it
                import ctypes
                com = ctypes.windll.ole32.CoInitializeEx(None, 0) in (0, 1)
            except Exception:
                pass
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mic = sc.get_microphone(id=source["id"], include_loopback=True)
                self._setup_rate(sr)
                chunk = max(64, int(sr * CAPTURE_CHUNK_S))
                with mic.recorder(samplerate=sr, blocksize=chunk) as rec:
                    self.active = True
                    while not self._stop.is_set():
                        d = rec.record(numframes=chunk)
                        if d.ndim == 2:
                            d = d.mean(axis=1)
                        self._push(d)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.active = False
            if com:
                try:
                    import ctypes
                    ctypes.windll.ole32.CoUninitialize()
                except Exception:
                    pass

    def _run_sounddevice(self, source: dict, sr: int) -> None:
        try:
            info = sd.query_devices(source["id"])
            dsr = int(info["default_samplerate"])
            self._setup_rate(dsr)

            def cb(indata, frames, t, status):
                self._push(indata.mean(axis=1))

            with sd.InputStream(device=source["id"], channels=min(2, int(info["max_input_channels"])),
                                samplerate=dsr, dtype="float32", blocksize=max(64, int(dsr * CAPTURE_CHUNK_S)),
                                callback=cb):
                self.active = True
                while not self._stop.wait(0.1):
                    pass
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.active = False


# ============================================================================ analysis
def _lowpass_coeffs(fc: float, fs: float):
    fc = min(fc, fs * 0.45)
    w = 2 * math.pi * fc / fs
    alpha = math.sin(w) / (2 * 0.70710678)
    c = math.cos(w)
    a0 = 1 + alpha
    b0 = (1 - c) / 2 / a0
    return b0, 2 * b0, b0, -2 * c / a0, (1 - alpha) / a0


class AudioFollower:
    """Per-generator analysis state. `render` returns n samples at the output rate:
    bass mode → signal in [-1, 1]; envelope / beats → control in [0, 1]."""

    def __init__(self, capture: AudioCapture):
        self.cap = capture
        self.onsets = 0
        self.reset()

    def reset(self) -> None:
        self.proc_pos = None
        self.read_pos = 0.0
        self.out = np.zeros(RING, np.float32)
        self.z = [0.0] * 8          # two biquads: x1 x2 y1 y2 each
        self.hp_x = self.hp_y = 0.0
        self.env = 0.0
        self.fast = 0.0
        self.slow = 1e-4
        self.hit = 0.0
        self.peak = 1e-3
        self.since = 1e9
        self._coef_key = None
        self._coef = None

    # ------------------------------------------------------------------ DSP (decimated rate)
    def _process(self, x: np.ndarray, p) -> np.ndarray:
        """Run new decimated samples through the per-generator chain, sample by sample (filters and
        followers are recursive). Input: mono at cap.rate_d. Output: same length —
        bass: band-limited signal in [-1, 1]; envelope: gated envelope in [0, 1];
        beats: decaying hit envelope in [0, 1] (each detected onset bumps `self.onsets`)."""
        fs = self.cap.rate_d
        key = (round(p.audio_cutoff, 1), fs)
        if key != self._coef_key:
            self._coef_key, self._coef = key, _lowpass_coeffs(p.audio_cutoff, fs)
        b0, b1, b2, a1, a2 = self._coef
        z = self.z
        hp_a = math.exp(-2 * math.pi * DC_CUTOFF_HZ / fs)
        gain = 10 ** (p.audio_gain / 20.0)
        auto = p.audio_auto
        pk_decay = math.exp(-1.0 / (AUTO_LEVEL_DECAY_S * fs))
        att = 1 - math.exp(-1.0 / (ATTACK_S * fs))
        rel = 1 - math.exp(-1.0 / (max(5.0, p.audio_smooth) / 1000.0 * fs))
        floor = p.audio_floor
        mode = p.audio_mode
        # beats
        f_rel = 1 - math.exp(-1.0 / (BEAT_FAST_RELEASE_S * fs))
        s_coef = 1 - math.exp(-1.0 / (BEAT_SLOW_S * fs))
        hit_decay = math.exp(-1.0 / (max(5.0, p.audio_smooth) / 1000.0 * fs))
        thr = p.audio_beat
        refractory = BEAT_REFRACTORY_S * fs

        hx, hy = self.hp_x, self.hp_y
        env, fast, slow, hit, peak, since = self.env, self.fast, self.slow, self.hit, self.peak, self.since
        out = np.empty(len(x), np.float32)
        for i, v in enumerate(x.tolist()):
            # DC / sub-sonic removal
            hy = hp_a * (hy + v - hx)
            hx = v
            # 4th-order Butterworth-ish low-pass (two biquads)
            y = b0 * hy + b1 * z[0] + b2 * z[1] - a1 * z[2] - a2 * z[3]
            z[1], z[0], z[3], z[2] = z[0], hy, z[2], y
            y2 = b0 * y + b1 * z[4] + b2 * z[5] - a1 * z[6] - a2 * z[7]
            z[5], z[4], z[7], z[6] = z[4], y, z[6], y2
            a = abs(y2)
            peak = a if a > peak else peak * pk_decay
            s = y2 * gain / (peak if peak > 1e-4 else 1e-4) if auto else y2 * gain
            m = abs(s)

            if mode == "beats":
                fast += (m - fast) * (att if m > fast else f_rel)
                slow += (fast - slow) * s_coef
                since += 1
                if fast > slow * thr and fast > floor and since > refractory:
                    since = 0
                    hit = max(hit, min(1.0, 0.45 + 0.55 * fast))
                    self.onsets += 1
                hit *= hit_decay
                out[i] = hit
            else:
                env += (m - env) * (att if m > env else rel)
                g = (env - floor) / (1.0 - floor) if env > floor else 0.0
                if mode == "bass":
                    gate = min(1.0, g / max(env, 1e-6)) if floor > 0 else 1.0
                    out[i] = math.tanh(s * gate)
                else:
                    out[i] = math.tanh(ENVELOPE_DRIVE * g)
        self.hp_x, self.hp_y = hx, hy
        self.env, self.fast, self.slow, self.hit, self.peak, self.since = env, fast, slow, hit, peak, since
        return out

    # ------------------------------------------------------------------ output-rate render
    def render(self, n: int, sr_out: float, p) -> np.ndarray:
        """Called from the output audio callback. Analyses whatever the capture thread wrote since
        the last call, then linearly resamples the result to n samples at sr_out, trailing the
        writer by ~TARGET_LAG_S and nudging the playback speed to absorb clock drift."""
        cap = self.cap
        wp = cap.write_pos
        if not cap.active or wp < 8:
            self.proc_pos = None
            return np.zeros(n)
        rd = cap.rate_d
        if self.proc_pos is None or wp - self.proc_pos > RING // 2 or wp < self.proc_pos:
            keep = self.onsets
            self.reset()
            self.onsets = keep
            self.proc_pos = max(0, wp - int(0.05 * rd))
            self.read_pos = float(self.proc_pos)

        # analyse everything captured since last call
        if wp > self.proc_pos:
            idx = np.arange(self.proc_pos, wp)
            y = self._process(cap.ring[idx % RING], p)
            self.out[idx % RING] = y
            self.proc_pos = wp

        # resample to output rate, keeping ~35 ms behind the writer
        step = rd / sr_out
        span = n * step
        target = max(TARGET_LAG_S * rd, 1.5 * span)
        lag_after = wp - 1 - (self.read_pos + span)
        if lag_after < 0 or lag_after > target + MAX_EXTRA_LAG_S * rd:
            self.read_pos = wp - 1 - span - target  # underrun or too far behind: re-centre
        else:
            step *= 1.0 + DRIFT_GAIN * max(-1.0, min(1.0, (lag_after - target) / target))
        pos = self.read_pos + np.arange(1, n + 1) * step
        base = int(math.floor(pos[0]))
        last = int(math.ceil(pos[-1])) + 1
        win = self.out[np.arange(base, last + 1) % RING]
        vals = np.interp(pos - base, np.arange(len(win)), win)
        self.read_pos = float(pos[-1])
        return vals
