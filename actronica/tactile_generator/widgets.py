"""Small canvas-drawn widget kit with a dark theme (pure tkinter, no extra deps)."""
from __future__ import annotations

import math
import tkinter as tk
import tkinter.font as tkfont

import numpy as np


class T:
    bg = "#0c0e13"
    panel = "#14171f"
    panel2 = "#1a1e28"
    raised = "#232836"
    hover = "#2a3040"
    border = "#252a37"
    grid = "#1f2430"
    text = "#e8eaf0"
    muted = "#8b93a7"
    dim = "#4d5466"
    a = "#22d3ee"
    b = "#fbbf24"
    ab = "#c084fc"
    good = "#34d399"
    bad = "#fb7185"
    warn = "#f59e0b"
    ink = "#0b0d12"


SCALE = 1.0
FAMILY = "Segoe UI"
FAMILY_SEMI = "Segoe UI Semibold"
MONO = "Consolas"


def init_scale(root: tk.Tk, design_w=1480, design_h=900) -> None:
    global SCALE, FAMILY, FAMILY_SEMI, MONO
    dpi = root.winfo_fpixels("1i") / 96.0
    fit = min(root.winfo_screenwidth() / design_w, (root.winfo_screenheight() - 80) / design_h)
    SCALE = max(0.75, min(dpi, fit))
    fams = set(tkfont.families(root))
    for cand in ("Segoe UI Variable Text", "Segoe UI", "Helvetica"):
        if cand in fams:
            FAMILY = cand
            break
    for cand in ("Segoe UI Variable Text Semibold", "Segoe UI Semibold"):
        if cand in fams:
            FAMILY_SEMI = cand
            break
    else:
        FAMILY_SEMI = FAMILY
    for cand in ("Cascadia Mono", "Consolas", "Courier New"):
        if cand in fams:
            MONO = cand
            break


def px(v: float) -> int:
    return int(round(v * SCALE))


def font(size: float, weight: str = "normal", mono: bool = False):
    if mono:
        return (MONO, -px(size), weight)
    if weight == "bold":
        return (FAMILY_SEMI, -px(size))
    return (FAMILY, -px(size))


def mix(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(x + (y - x) * t)):02x}" for x, y in zip(a, b))


def round_rect(cv: tk.Canvas, x1, y1, x2, y2, r, **kw):
    r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return cv.create_polygon(pts, smooth=True, **kw)


def text_width(f, s: str) -> int:
    return tkfont.Font(font=f).measure(s)


# ---------------------------------------------------------------------------- Card
class Card(tk.Frame):
    def __init__(self, master, title: str, subtitle: str | None = None, pad=14):
        super().__init__(master, bg=T.panel, highlightthickness=1,
                         highlightbackground=T.border, highlightcolor=T.border)
        self.header = tk.Frame(self, bg=T.panel)
        self.header.pack(fill="x", padx=px(pad), pady=(px(11), px(6)))
        tk.Label(self.header, text=title.upper(), font=font(11, "bold"), fg=T.muted,
                 bg=T.panel).pack(side="left")
        self.subtitle = tk.Label(self.header, text=subtitle or "", font=font(11), fg=T.dim, bg=T.panel)
        self.subtitle.pack(side="left", padx=(px(8), 0))
        self.body = tk.Frame(self, bg=T.panel)
        self.body.pack(fill="both", expand=True, padx=px(pad), pady=(0, px(pad - 2)))


def section(master, text: str, bg=T.panel):
    f = tk.Frame(master, bg=bg)
    tk.Label(f, text=text.upper(), font=font(10, "bold"), fg=T.dim, bg=bg).pack(side="left")
    line = tk.Frame(f, bg=T.border, height=1)
    line.pack(side="left", fill="x", expand=True, padx=(px(8), 0), pady=(px(2), 0))
    return f


# ---------------------------------------------------------------------------- Slider
class Slider(tk.Canvas):
    """Label + value readout + track. Drag/click to set, wheel to nudge (Shift = fine),
    double-click the number to type a value, right-click to reset."""

    def __init__(self, master, label, from_, to, value, *, unit="", fmt=None, log=False,
                 color=T.a, command=None, step=None, default=None, bg=T.panel, height=40):
        super().__init__(master, height=px(height), bg=bg, highlightthickness=0, bd=0)
        self.label, self.lo, self.hi, self.unit = label, float(from_), float(to), unit
        self.fmt = fmt or (lambda v: f"{v:.0f}")
        self.log, self.color, self.command, self.step = log, color, command, step
        self.default = value if default is None else default
        self.bgc = bg
        self.value = self._clamp(value)
        self._entry = None
        self._drag = False
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<Button-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", False))
        self.bind("<Double-Button-1>", self._edit)
        self.bind("<Button-3>", lambda e: self.set(self.default, notify=True))
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Enter>", lambda e: self.redraw(hover=True))
        self.bind("<Leave>", lambda e: self.redraw(hover=False))

    def _clamp(self, v):
        v = min(self.hi, max(self.lo, float(v)))
        if self.step:
            v = round(v / self.step) * self.step
        return v

    def _v2f(self, v):
        if self.log:
            return math.log(v / self.lo) / math.log(self.hi / self.lo)
        return (v - self.lo) / (self.hi - self.lo)

    def _f2v(self, f):
        f = min(1.0, max(0.0, f))
        if self.log:
            return self.lo * (self.hi / self.lo) ** f
        return self.lo + f * (self.hi - self.lo)

    def _geom(self):
        w, h = self.winfo_width(), self.winfo_height()
        pad = px(8)
        return w, h, pad, w - pad, h - px(12)

    def redraw(self, hover=None):
        if hover is not None:
            self._hover = hover
        hover = getattr(self, "_hover", False)
        self.delete("all")
        w, h, x0, x1, ty = self._geom()
        if w < 10:
            return
        self.create_text(px(1), px(10), text=self.label, anchor="w", font=font(12),
                         fill=T.text if hover else T.muted)
        self.create_text(w - px(1), px(10), text=f"{self.fmt(self.value)}{self.unit}",
                         anchor="e", font=font(12, mono=True), fill=T.text, tags="val")
        f = self._v2f(self.value)
        xv = x0 + f * (x1 - x0)
        th = px(4)
        round_rect(self, x0, ty - th / 2, x1, ty + th / 2, th / 2, fill=T.raised, outline="")
        if xv - x0 > 1:
            round_rect(self, x0, ty - th / 2, xv, ty + th / 2, th / 2, fill=self.color, outline="")
        r = px(7 if hover or self._drag else 6)
        self.create_oval(xv - r, ty - r, xv + r, ty + r, fill=T.text, outline=self.bgc, width=px(2))

    def _press(self, e):
        w, h, x0, x1, ty = self._geom()
        if e.y >= px(18):
            self._drag = True
            self._set_from_x(e.x)

    def _motion(self, e):
        if self._drag:
            self._set_from_x(e.x)

    def _set_from_x(self, x):
        w, h, x0, x1, ty = self._geom()
        self.set(self._f2v((x - x0) / max(1, x1 - x0)), notify=True)

    def _wheel(self, e):
        d = 1 if e.delta > 0 else -1
        fine = bool(e.state & 0x0001)
        if self.step and (self.hi - self.lo) / self.step <= 200 and not self.log:
            nv = self.value + d * self.step
        else:
            nv = self._f2v(self._v2f(self.value) + d * (0.002 if fine else 0.01))
            if self.step and abs(nv - self.value) < self.step:
                nv = self.value + d * self.step
        self.set(nv, notify=True)
        return "break"

    def _edit(self, e=None):
        if self._entry:
            return
        w = self.winfo_width()
        ent = tk.Entry(self, font=font(12, mono=True), bg=T.raised, fg=T.text, insertbackground=T.text,
                       relief="flat", justify="right", width=9, highlightthickness=1,
                       highlightbackground=self.color, highlightcolor=self.color)
        ent.insert(0, self.fmt(self.value).strip())
        ent.select_range(0, "end")
        self._entry = ent
        win = self.create_window(w - px(1), px(10), window=ent, anchor="e")

        def done(commit):
            if self._entry is None:
                return
            txt = ent.get().strip().replace(",", ".")
            self._entry = None
            self.delete(win)
            ent.destroy()
            if commit:
                try:
                    num = float("".join(c for c in txt if c in "0123456789.-eE") or "x")
                    self.set(num, notify=True)
                except ValueError:
                    pass
            self.redraw()

        ent.bind("<Return>", lambda ev: done(True))
        ent.bind("<KP_Enter>", lambda ev: done(True))
        ent.bind("<Escape>", lambda ev: done(False))
        ent.bind("<FocusOut>", lambda ev: done(True))
        ent.focus_set()

    def set(self, v, notify=False):
        nv = self._clamp(v)
        changed = nv != self.value
        self.value = nv
        self.redraw()
        if notify and changed and self.command:
            self.command(nv)

    def get(self):
        return self.value


# ---------------------------------------------------------------------------- Toggle
class Toggle(tk.Canvas):
    def __init__(self, master, text="", value=False, command=None, color=T.a, bg=T.panel, size=12):
        self.text, self.value, self.command, self.color, self.bgc = text, bool(value), command, color, bg
        self._f = font(size)
        tw = text_width(self._f, text) + px(8) if text else 0
        super().__init__(master, width=px(32) + tw, height=px(22), bg=bg, highlightthickness=0,
                         bd=0, cursor="hand2")
        self.bind("<Button-1>", lambda e: self.set(not self.value, notify=True))
        self.redraw()

    def redraw(self):
        self.delete("all")
        h = px(22)
        w, hh = px(30), px(16)
        y0 = (h - hh) / 2
        on = self.value
        round_rect(self, 1, y0, w, y0 + hh, hh / 2, fill=self.color if on else T.raised,
                   outline="" if on else T.dim)
        r = hh / 2 - px(3)
        cx = w - hh / 2 if on else 1 + hh / 2
        self.create_oval(cx - r, h / 2 - r, cx + r, h / 2 + r, fill=T.ink if on else T.muted, outline="")
        if self.text:
            self.create_text(w + px(8), h / 2, text=self.text, anchor="w", font=self._f,
                             fill=T.text if on else T.muted)

    def set(self, v, notify=False):
        v = bool(v)
        changed = v != self.value
        self.value = v
        self.redraw()
        if notify and changed and self.command:
            self.command(v)


# ---------------------------------------------------------------------------- Segmented
class Segmented(tk.Canvas):
    def __init__(self, master, options, value, command=None, color=T.a, bg=T.panel, height=28,
                 size=11, fill=False, seg_pad=12):
        self.options = list(options)  # [(value, label)]
        self.value, self.command, self.color, self.bgc = value, command, color, bg
        self._f = font(size, "bold")
        self._ws = [text_width(self._f, lbl) + px(seg_pad) * 2 for _, lbl in self.options]
        super().__init__(master, height=px(height), width=sum(self._ws) + px(6), bg=bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self._fill = fill
        self._hover = None
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<Button-1>", self._click)
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda e: (setattr(self, "_hover", None), self.redraw()))

    def _layout(self):
        w = max(self.winfo_width(), int(self["width"]))
        inner = w - px(6)
        if self._fill:
            each = inner / len(self.options)
            ws = [each] * len(self.options)
        else:
            ws = self._ws
        xs, x = [], px(3)
        for wi in ws:
            xs.append((x, x + wi))
            x += wi
        return xs

    def _hit(self, x):
        for i, (a, b) in enumerate(self._layout()):
            if a <= x < b:
                return i
        return None

    def _motion(self, e):
        h = self._hit(e.x)
        if h != self._hover:
            self._hover = h
            self.redraw()

    def _click(self, e):
        i = self._hit(e.x)
        if i is not None:
            self.set(self.options[i][0], notify=True)

    def redraw(self):
        self.delete("all")
        w, h = max(self.winfo_width(), int(self["width"])), int(self["height"])
        round_rect(self, 0, 0, w - 1, h - 1, px(7), fill=T.panel2, outline=T.border)
        for i, ((val, lbl), (a, b)) in enumerate(zip(self.options, self._layout())):
            sel = val == self.value
            if sel:
                round_rect(self, a, px(3), b, h - px(3), px(5), fill=mix(T.raised, self.color, 0.18),
                           outline=self.color)
            elif i == self._hover:
                round_rect(self, a, px(3), b, h - px(3), px(5), fill=T.raised, outline="")
            self.create_text((a + b) / 2, h / 2, text=lbl, font=self._f,
                             fill=self.color if sel else (T.text if i == self._hover else T.muted))

    def set(self, v, notify=False):
        changed = v != self.value
        self.value = v
        self.redraw()
        if notify and changed and self.command:
            self.command(v)

    def set_color(self, c):
        self.color = c
        self.redraw()


# ---------------------------------------------------------------------------- Button
class Button(tk.Canvas):
    def __init__(self, master, text, command=None, kind="ghost", color=T.a, width=None, height=30,
                 size=12, bg=T.panel, anchor="center", sub=None):
        self.text, self.command, self.kind, self.color, self.bgc = text, command, kind, color, bg
        self.sub, self.anchor = sub, anchor
        self._f = font(size, "bold" if kind != "row" else "normal")
        w = width if width is not None else (text_width(self._f, text) / SCALE + 28)
        super().__init__(master, width=px(w), height=px(height), bg=bg, highlightthickness=0, bd=0,
                         cursor="hand2")
        self._hover = False
        self._flash = 0.0
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<Enter>", lambda e: (setattr(self, "_hover", True), self.redraw()))
        self.bind("<Leave>", lambda e: (setattr(self, "_hover", False), self.redraw()))
        self.bind("<Button-1>", self._click)

    def _click(self, e=None):
        self.flash()
        if self.command:
            self.command()

    def flash(self):
        self._flash = 1.0
        self._animate()

    def _animate(self):
        self.redraw()
        if self._flash > 0:
            self._flash = max(0.0, self._flash - 0.12)
            self.after(30, self._animate)

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4:
            w, h = int(self["width"]), int(self["height"])
        k = self.kind
        if k == "primary":
            fill, outline, fg = (mix(self.color, "#ffffff", 0.12) if self._hover else self.color), "", T.ink
        elif k == "danger":
            fill, outline, fg = (mix(T.bad, "#ffffff", 0.12) if self._hover else T.bad), "", T.ink
        elif k == "row":
            fill = T.raised if self._hover else T.panel2
            outline, fg = "", T.text
        else:
            fill = T.hover if self._hover else T.panel2
            outline, fg = (self.color if self._hover else T.border), T.text
        if self._flash > 0:
            fill = mix(fill, self.color, 0.55 * self._flash)
        round_rect(self, 1, 1, w - 2, h - 2, px(7), fill=fill, outline=outline)
        if self.anchor == "w":
            ty = h / 2 if not self.sub else h / 2 - px(7)
            self.create_text(px(12), ty, text=self.text, anchor="w", font=self._f, fill=fg)
            if self.sub:
                self.create_text(px(12), h / 2 + px(8), text=self.sub, anchor="w", font=font(10),
                                 fill=T.muted)
        else:
            if self.sub:
                self.create_text(w / 2, h / 2 - px(7), text=self.text, font=self._f, fill=fg)
                self.create_text(w / 2, h / 2 + px(10), text=self.sub, font=font(10, mono=True),
                                 fill=T.dim if k != "primary" else T.ink)
            else:
                self.create_text(w / 2, h / 2, text=self.text, font=self._f, fill=fg)

    def configure_button(self, text=None, kind=None, color=None):
        if text is not None:
            self.text = text
        if kind is not None:
            self.kind = kind
        if color is not None:
            self.color = color
        self.redraw()


# ---------------------------------------------------------------------------- WaveSelector
class WaveSelector(tk.Canvas):
    """Tiles with mini waveform icons."""

    def __init__(self, master, waves, labels, value, command=None, color=T.a, bg=T.panel, icon_fn=None):
        super().__init__(master, height=px(62), bg=bg, highlightthickness=0, bd=0, cursor="hand2")
        self.waves, self.labels, self.value = waves, labels, value
        self.command, self.color, self.icon_fn = command, color, icon_fn
        self._hover = None
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<Button-1>", self._click)
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda e: (setattr(self, "_hover", None), self.redraw()))

    def _cells(self):
        w = self.winfo_width()
        n = len(self.waves)
        gap = px(6)
        cw = (w - gap * (n - 1)) / n
        return [(i * (cw + gap), i * (cw + gap) + cw) for i in range(n)]

    def _hit(self, x):
        for i, (a, b) in enumerate(self._cells()):
            if a <= x <= b:
                return i
        return None

    def _motion(self, e):
        h = self._hit(e.x)
        if h != self._hover:
            self._hover = h
            self.redraw()

    def _click(self, e):
        i = self._hit(e.x)
        if i is not None:
            self.set(self.waves[i], notify=True)

    def redraw(self):
        self.delete("all")
        h = self.winfo_height()
        for i, (wv, (a, b)) in enumerate(zip(self.waves, self._cells())):
            sel = wv == self.value
            fill = mix(T.panel2, self.color, 0.14) if sel else (T.raised if i == self._hover else T.panel2)
            round_rect(self, a + 1, 1, b - 1, h - 2, px(8), fill=fill, outline=self.color if sel else T.border)
            ys = self.icon_fn(wv, 48) if self.icon_fn else np.sin(np.linspace(0, 2 * np.pi, 48))
            ix0, ix1 = a + px(10), b - px(10)
            cy, amp = px(22), px(10)
            pts = []
            for k, y in enumerate(ys):
                pts += [ix0 + (ix1 - ix0) * k / (len(ys) - 1), cy - amp * float(y)]
            self.create_line(*pts, fill=self.color if sel else T.muted, width=px(2) if sel else px(1.5),
                             capstyle="round", joinstyle="round")
            self.create_text((a + b) / 2, h - px(13), text=self.labels[wv], font=font(10, "bold"),
                             fill=T.text if sel else T.muted)

    def set(self, v, notify=False):
        changed = v != self.value
        self.value = v
        self.redraw()
        if notify and changed and self.command:
            self.command(v)

    def set_color(self, c):
        self.color = c
        self.redraw()
