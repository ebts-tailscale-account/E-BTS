#!/usr/bin/env python3
"""EVT2 random-access frame extractor -- seek to a timestamp, accumulate 40 ms frames.

WHY THIS EXISTS (HANDOFF section 15.2e)
---------------------------------------
`postprocess.py`'s event slicing does not survive a 648-indent campaign: it buffers
every poke's events in RAM and only writes at the end (~48 GB for this run), which
OOM'd the workstation. And `metavision_core`'s EventsIterator has no true seek -- a
`start_ts` deep in a 45 GB file streams everything before it.

EVT2 is a FIXED 4-BYTE-WORD format, so random access is easy and exact:

    bits 31..28  type       0x0 = CD_OFF, 0x1 = CD_ON, 0x8 = TIME_HIGH
    CD_*  : bits 27..22 ts_low (6 bits, us) | 21..11 x | 10..0 y
    0x8   : bits 27..0  ts_high (28 bits, units of 64 us)
    t_us  = (ts_high << 6) | ts_low

TIME_HIGH words recur at least every 64 us of event time (~368 bytes at this run's
2.3 Mev/s), so ANY 64 KB probe contains many of them. That makes a plain binary
search over byte offsets exact and cheap: O(log n) seeks instead of a 45 GB pass.

WHY FRAMES, NOT EVENTS (measured on pilot_20260807_134855)
----------------------------------------------------------
A 40 ms window holds ~92,000 events over 288,000 px:
    as events  x,y,t (u2,u2,u4) = 8 B  ->  722 KB
    as a dense 640x450 uint8 count frame ->  281 KB   <- 2.6x smaller, uncompressed
Dense wins because the event rate is high. And the frame is LOSSLESS here:
  * counts peak at 26, so uint8 holds them exactly (never rescale -- section 13.1);
  * every event is ONE polarity (bias_diff_off = 0 kills OFF in hardware), so the
    polarity bit carries no information at all. Verified: 923,529 ON, 0 OFF.

TWO TIME BASES -- do not mix them up
------------------------------------
  * the RAW's own TIME_HIGH values start at the sensor's uptime (1377.077 s for this
    run), NOT at zero;
  * metavision's EventsIterator RE-BASES to 0, and postprocess.py maps
    unix -> device as (t_unix - t_ft[0]) * 1e6 on that re-based scale.
`base_us()` reads the file's first TIME_HIGH so callers can work in the re-based
("metavision") scale and have it converted here.

Usage as a library:
    r = Evt2Reader("camera.raw")
    frames, t0s = r.frames(start_us, end_us, accum_us=40000)   # re-based scale
"""

import numpy as np

HEADER_MAX = 1 << 16
TYPE_TIME_HIGH = 0x8
TYPE_CD_OFF, TYPE_CD_ON = 0x0, 0x1
GEOM_W, GEOM_H = 640, 480


class Evt2Reader:
    def __init__(self, path, probe_bytes=1 << 16):
        self.path = path
        self.probe = probe_bytes
        self.f = open(path, "rb")
        head = self.f.read(HEADER_MAX)
        off = 0
        for line in head.split(b"\n"):
            if line.startswith(b"%"):
                off += len(line) + 1
            else:
                break
        self.header_bytes = off
        import os
        self.size = os.path.getsize(path)
        body = self.size - off
        # A torn final word means the writer died mid-event; refuse to guess.
        self.n_words = body // 4
        self.torn = (body % 4) != 0
        self._base = self._first_time_high()

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False

    # ---- time helpers -------------------------------------------------------
    def _read_words(self, word_idx, n):
        self.f.seek(self.header_bytes + word_idx * 4)
        buf = self.f.read(n * 4)
        return np.frombuffer(buf[: (len(buf) // 4) * 4], dtype="<u4")

    @staticmethod
    def _time_highs(w):
        return w[(w >> 28) == TYPE_TIME_HIGH]

    def _first_time_high(self):
        w = self._read_words(0, self.probe // 4)
        th = self._time_highs(w)
        if not len(th):
            raise RuntimeError("no TIME_HIGH in the first %d bytes" % self.probe)
        return (int(th[0]) & 0x0FFFFFFF) << 6      # us, RAW device scale

    def base_us(self):
        """First TIME_HIGH in the file, in the RAW device scale (us)."""
        return self._base

    def _time_at(self, word_idx):
        """Device time (us, RAW scale) of the first TIME_HIGH at or after word_idx."""
        n = self.probe // 4
        while word_idx < self.n_words:
            w = self._read_words(word_idx, n)
            if not len(w):
                break
            th = self._time_highs(w)
            if len(th):
                return (int(th[0]) & 0x0FFFFFFF) << 6
            word_idx += len(w)
        return None

    def seek_word(self, t_us_raw):
        """Binary-search the word index whose TIME_HIGH first reaches t_us_raw."""
        lo, hi = 0, self.n_words
        probe_words = self.probe // 4
        while hi - lo > probe_words:
            mid = (lo + hi) // 2
            t = self._time_at(mid)
            if t is None or t >= t_us_raw:
                hi = mid
            else:
                lo = mid
        return max(0, lo - probe_words)

    # ---- the thing callers want --------------------------------------------
    def frames(self, start_us, end_us, accum_us=40000, rebased=True,
               width=GEOM_W, height=GEOM_H, y0=0, y1=None):
        """Accumulate non-overlapping `accum_us` count-frames over [start, end).

        Times are in metavision's RE-BASED scale by default (0 = recording start),
        matching postprocess.py. Pass rebased=False to use raw device time.
        Returns (frames uint8 [n, h, w], frame_start_times_us).
        """
        y1 = height if y1 is None else y1
        s = int(start_us) + (self._base if rebased else 0)
        e = int(end_us) + (self._base if rebased else 0)
        nf = max(1, int(np.ceil((e - s) / accum_us)))
        out = np.zeros((nf, y1 - y0, width), np.uint16)   # uint16 while summing

        wi = self.seek_word(s)
        cur_high = None
        chunk = 1 << 22                                   # 4 M words = 16 MB
        done = False
        while wi < self.n_words and not done:
            w = self._read_words(wi, chunk)
            if not len(w):
                break
            wi += len(w)
            typ = (w >> 28).astype(np.uint8)
            is_th = typ == TYPE_TIME_HIGH
            # Carry the running TIME_HIGH across the chunk boundary, then
            # forward-fill so every CD event knows its high bits.
            highs = np.where(is_th, (w & 0x0FFFFFFF).astype(np.int64) << 6, -1)
            if cur_high is not None and (not len(highs) or highs[0] < 0):
                highs[0] = cur_high if highs[0] < 0 else highs[0]
            idx = np.maximum.accumulate(np.where(highs >= 0, np.arange(len(highs)), -1))
            valid = idx >= 0
            hi_val = np.where(valid, highs[np.maximum(idx, 0)], -1)
            if cur_high is not None:
                hi_val = np.where(valid, hi_val, cur_high)
            last_th = highs[highs >= 0]
            if len(last_th):
                cur_high = int(last_th[-1])
            if cur_high is None:
                continue
            is_cd = (typ == TYPE_CD_ON) | (typ == TYPE_CD_OFF)
            m = is_cd & (hi_val >= 0)
            if not m.any():
                if cur_high >= e:
                    break
                continue
            t = hi_val[m] + ((w[m] >> 22) & 0x3F).astype(np.int64)
            x = ((w[m] >> 11) & 0x7FF).astype(np.int32)
            y = (w[m] & 0x7FF).astype(np.int32)
            sel = (t >= s) & (t < e) & (y >= y0) & (y < y1) & (x < width)
            if sel.any():
                fi = ((t[sel] - s) // accum_us).astype(np.int32)
                np.add.at(out, (fi, y[sel] - y0, x[sel]), 1)
            if t[-1] >= e:
                done = True
        np.clip(out, 0, 255, out=out)
        return out.astype(np.uint8), np.arange(nf) * accum_us + int(start_us)
