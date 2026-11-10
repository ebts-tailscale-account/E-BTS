"""Tactile Lab Generator — GUI for driving actuators through a multichannel audio interface (Actronica HSD)."""
from __future__ import annotations

import ctypes
import json
import math
import sys
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np

from . import widgets as W
from .dsp import EFFECTS, WAVE_LABELS, WAVEFORMS, GenParams, pulse, shape
from .engine import (ROUTE_A, ROUTE_AB, ROUTE_B, ROUTE_OFF, AudioEngine, list_output_devices,
                     pick_default_device, refresh_portaudio)
from .presets import BUILTIN_PRESETS
from .widgets import (Button, Card, Segmented, Slider, T, Toggle, WaveSelector, font, mix, px,
                      round_rect, section, text_width)

HERE = Path(__file__).resolve().parent
SESSION_FILE = HERE / "session.json"
PRESET_DIR = HERE / "user_presets"

GEN_COLORS = (T.a, T.b)
ROUTE_COLORS = {ROUTE_OFF: T.dim, ROUTE_A: T.a, ROUTE_B: T.b, ROUTE_AB: T.ab}
ROUTE_LABELS = {ROUTE_OFF: "off", ROUTE_A: "A", ROUTE_B: "B", ROUTE_AB: "A+B"}
FX_KEYS = {key: name for name, (_, key) in EFFECTS.items()}


def fmt_hz(v):
    return f"{v:.1f}" if v < 100 else f"{v:.0f}"


def fmt_rate(v):
    return f"{v:.2f}" if v < 10 else f"{v:.1f}"


_ICONS: dict = {}


def icon_wave(wf, n):
    key = (wf, n)
    if key not in _ICONS:
        x = np.arange(n) / (n - 1)
        p = np.mod(x * 2.0, 1.0)
        if wf == "noise":
            y = np.random.default_rng(4).uniform(-0.9, 0.9, n)
        elif wf == "rumble":
            y = 0.55 * np.sin(2 * np.pi * (1.3 * x) + 1.0) + 0.4 * np.sin(2 * np.pi * 3.2 * x + 0.3)
        elif wf == "square":
            y = pulse(p, 0.5, 0.2)
        else:
            y = shape(wf, p, GenParams(symmetry=0.5, softness=0.0))
        _ICONS[key] = y * 0.95
    return _ICONS[key]


# ============================================================================ small widgets
class Chip(tk.Canvas):
    """Multi-select chip (scope trace visibility)."""

    def __init__(self, master, text, color, value=True, command=None, bg=T.panel):
        self.text, self.color, self.value, self.command, self.bgc = text, color, value, command, bg
        self._f = font(10, "bold")
        super().__init__(master, width=text_width(self._f, text) + px(28), height=px(24), bg=bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self.bind("<Button-1>", lambda e: self.set(not self.value, True))
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = int(self["width"]), int(self["height"])
        on = self.value
        round_rect(self, 1, 1, w - 2, h - 2, px(6), fill=mix(T.panel2, self.color, 0.16) if on else T.panel2,
                   outline=self.color if on else T.border)
        r = px(3.5)
        self.create_oval(px(10) - r, h / 2 - r, px(10) + r, h / 2 + r, fill=self.color if on else T.dim, outline="")
        self.create_text(px(18), h / 2, text=self.text, anchor="w", font=self._f, fill=T.text if on else T.muted)

    def set(self, v, notify=False):
        self.value = bool(v)
        self.redraw()
        if notify and self.command:
            self.command(self.value)


class GenTab(tk.Frame):
    def __init__(self, master, idx, color, on_select, on_toggle):
        super().__init__(master, bg=T.panel2, highlightthickness=1, highlightbackground=T.border, cursor="hand2")
        self.idx, self.color = idx, color
        self.top = tk.Frame(self, bg=T.panel2)
        self.top.pack(fill="x", padx=px(10), pady=(px(7), 0))
        self.letter = tk.Label(self.top, text="AB"[idx], font=font(20, "bold"), fg=color, bg=T.panel2)
        self.letter.pack(side="left")
        self.toggle = Toggle(self.top, value=True, command=on_toggle, color=color, bg=T.panel2)
        self.toggle.pack(side="right")
        self.summary = tk.Label(self, text="", font=font(12), fg=T.text, bg=T.panel2, anchor="w")
        self.summary.pack(fill="x", padx=px(11))
        self.flags = tk.Label(self, text="", font=font(10, "bold"), fg=T.muted, bg=T.panel2, anchor="w")
        self.flags.pack(fill="x", padx=px(11), pady=(0, px(8)))
        for w in (self, self.top, self.letter, self.summary, self.flags):
            w.bind("<Button-1>", lambda e: on_select(idx))

    def set_selected(self, sel):
        bg = mix(T.panel2, self.color, 0.10) if sel else T.panel2
        for w in (self, self.top, self.letter, self.summary, self.flags):
            w.configure(bg=bg)
        self.toggle.configure(bg=bg)
        self.toggle.bgc = bg
        self.toggle.redraw()
        self.configure(highlightbackground=self.color if sel else T.border)


# ============================================================================ app
class TactileLabGenerator:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.engine = AudioEngine()
        self.sel = 0
        self.marks = [[], []]
        self.devices: list[dict] = []
        self.blocksize = 1024
        self.exclusive = True
        self.scope_mode = "wave"
        self.scope_ms = 50
        self.traces = {"a": True, "b": True, "out": True}
        self._mdisp = np.zeros(self.engine.n_channels)
        self._peak_disp = 0.0
        self._clip_flash = 0.0
        self._last_clips = 0
        self._last_status = 0.0
        self._msg = ("", T.muted, 0.0)
        self._paint = None
        self._summary_cache = [None, None]
        self._spec_peak = None

        session = self._read_session()
        self._style()
        self._build()
        self._apply_session(session)
        self._sync_ui()
        self._bind_keys()
        self.refresh_devices(auto_connect=True, prefer=session.get("device"))
        self.root.after(50, self._tick)

    # ------------------------------------------------------------------ setup
    def _style(self):
        r = self.root
        r.configure(bg=T.bg)
        st = ttk.Style(r)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("Dark.TCombobox", fieldbackground=T.panel2, background=T.raised, foreground=T.text,
                     arrowcolor=T.muted, bordercolor=T.border, lightcolor=T.panel2, darkcolor=T.panel2,
                     selectbackground=T.panel2, selectforeground=T.text, padding=px(5), arrowsize=px(12))
        st.map("Dark.TCombobox",
               fieldbackground=[("readonly", T.panel2)], foreground=[("readonly", T.text)],
               background=[("active", T.hover), ("readonly", T.raised)],
               bordercolor=[("focus", T.a), ("hover", T.dim)],
               selectbackground=[("readonly", T.panel2)], selectforeground=[("readonly", T.text)])
        r.option_add("*TCombobox*Listbox.background", T.panel2)
        r.option_add("*TCombobox*Listbox.foreground", T.text)
        r.option_add("*TCombobox*Listbox.selectBackground", T.raised)
        r.option_add("*TCombobox*Listbox.selectForeground", T.a)
        r.option_add("*TCombobox*Listbox.font", font(12))
        r.option_add("*TCombobox*Listbox.borderWidth", 0)

    def _build(self):
        self._build_header()
        self._build_footer()
        body = tk.Frame(self.root, bg=T.bg)
        body.pack(fill="both", expand=True, padx=px(16), pady=(0, px(4)))
        body.grid_columnconfigure(0, minsize=px(372))
        body.grid_columnconfigure(1, weight=1)
        body.grid_columnconfigure(2, minsize=px(316))
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=T.bg)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, px(12)))
        self._build_generators(left)

        center = tk.Frame(body, bg=T.bg)
        center.grid(row=0, column=1, sticky="nsew")
        self._build_scope(center)
        self._build_outputs(center)

        right = tk.Frame(body, bg=T.bg, width=px(316))
        right.grid(row=0, column=2, sticky="nsew", padx=(px(12), 0))
        self._build_effects(right)
        self._build_master(right)
        self._build_presets(right)

    # ------------------------------------------------------------------ header
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=T.bg)
        hdr.pack(fill="x", padx=px(16), pady=(px(12), px(10)))

        logo = tk.Canvas(hdr, width=px(38), height=px(38), bg=T.bg, highlightthickness=0)
        logo.pack(side="left")
        round_rect(logo, 1, 1, px(37), px(37), px(10), fill=T.panel2, outline=T.border)
        xs = np.linspace(px(7), px(31), 40)
        ys = px(19) - px(9) * np.sin(np.linspace(0, 4 * np.pi, 40)) * np.hanning(40)
        logo.create_line(*np.column_stack([xs, ys]).ravel().tolist(), fill=T.a, width=px(2.2),
                         smooth=True, capstyle="round")

        tt = tk.Frame(hdr, bg=T.bg)
        tt.pack(side="left", padx=(px(10), 0))
        tk.Label(tt, text="Tactile Lab Generator", font=font(18, "bold"), fg=T.text, bg=T.bg).pack(anchor="w")
        tk.Label(tt, text="Actronica HSD · multichannel haptic signal generator", font=font(11), fg=T.muted,
                 bg=T.bg).pack(anchor="w", pady=(0, 0))

        self.play_btn = Button(hdr, "▶  Play", command=self.toggle_output, kind="primary", color=T.good,
                               width=138, height=40, size=14, bg=T.bg, sub=None)
        self.play_btn.pack(side="right")

        dev = tk.Frame(hdr, bg=T.bg)
        dev.pack(side="right", padx=(0, px(16)))
        self.status_dot = tk.Canvas(dev, width=px(12), height=px(12), bg=T.bg, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(0, px(6)))
        self.status_lbl = tk.Label(dev, text="Not connected", font=font(11), fg=T.muted, bg=T.bg)
        self.status_lbl.pack(side="left", padx=(0, px(14)))

        self.dev_var = tk.StringVar()
        self.dev_combo = ttk.Combobox(dev, textvariable=self.dev_var, state="readonly", width=40,
                                      style="Dark.TCombobox", font=font(12))
        self.dev_combo.pack(side="left")
        self.dev_combo.bind("<<ComboboxSelected>>", lambda e: (self.root.focus_set(), self.connect()))

        tk.Label(dev, text="Block", font=font(11), fg=T.muted, bg=T.bg).pack(side="left", padx=(px(10), px(4)))
        self.bs_var = tk.StringVar(value=str(self.blocksize))
        self.bs_combo = ttk.Combobox(dev, textvariable=self.bs_var, state="readonly", width=5,
                                     values=["256", "512", "1024", "2048"], style="Dark.TCombobox",
                                     font=font(12))
        self.bs_combo.pack(side="left")
        self.bs_combo.bind("<<ComboboxSelected>>", lambda e: self._on_blocksize())

        self.excl_tg = Toggle(dev, "Exclusive", value=True, command=self._on_exclusive, color=T.a, bg=T.bg,
                              size=11)
        self.excl_tg.pack(side="left", padx=(px(10), px(4)))
        Button(dev, "↻", command=lambda: self.refresh_devices(), width=34, height=30, size=14,
               bg=T.bg).pack(side="left", padx=(px(4), px(4)))
        self.conn_btn = Button(dev, "Connect", command=self._toggle_connect, width=96, height=30, size=11,
                               bg=T.bg)
        self.conn_btn.pack(side="left")

    def _build_footer(self):
        ft = tk.Frame(self.root, bg=T.bg)
        ft.pack(side="bottom", fill="x", padx=px(18), pady=(px(2), px(8)))
        self.foot_lbl = tk.Label(ft, text="", font=font(11, mono=True), fg=T.muted, bg=T.bg, anchor="w")
        self.foot_lbl.pack(side="left")
        tk.Label(ft, text="Space play/stop · Esc panic · Tab A/B · 1–0 − = effects · M mark sweep",
                 font=font(11), fg=T.dim, bg=T.bg).pack(side="right")
        self.msg_lbl = tk.Label(ft, text="", font=font(11), fg=T.muted, bg=T.bg)
        self.msg_lbl.pack(side="right", padx=(0, px(20)))

    # ------------------------------------------------------------------ generators
    def _build_generators(self, parent):
        card = Card(parent, "Generators", "two independent sources")
        card.pack(fill="both", expand=True)
        b = card.body

        tabs = tk.Frame(b, bg=T.panel)
        tabs.pack(fill="x")
        tabs.grid_columnconfigure((0, 1), weight=1, uniform="t")
        self.tabs = []
        for i in range(2):
            t = GenTab(tabs, i, GEN_COLORS[i], self.select_gen, lambda v, i=i: self._toggle_gen(i, v))
            t.grid(row=0, column=i, sticky="ew", padx=(0, px(4)) if i == 0 else (px(4), 0))
            self.tabs.append(t)

        self.wave = WaveSelector(b, WAVEFORMS, WAVE_LABELS, "sine", command=lambda v: self._set("waveform", v),
                                 icon_fn=icon_wave)
        self.wave.pack(fill="x", pady=(px(12), px(6)))

        self.sl = {}
        self.shape_box = tk.Frame(b, bg=T.panel)
        self.shape_box.pack(fill="x")
        sb = self.shape_box
        self.sl["freq"] = Slider(sb, "Frequency", 5, 1000, 65, unit=" Hz", fmt=fmt_hz, log=True, step=0.1,
                                 command=lambda v: self._set("freq", v))
        self.sl["amp"] = Slider(sb, "Amplitude", 0, 100, 40, unit=" %", step=1, default=40,
                                command=lambda v: self._set("amp", v / 100))
        self.sl["duty"] = Slider(sb, "Duty cycle", 2, 98, 50, unit=" %", step=1,
                                 command=lambda v: self._set("duty", v / 100))
        self.sl["symmetry"] = Slider(sb, "Symmetry", 0, 100, 50, unit=" %", step=1,
                                     command=lambda v: self._set("symmetry", v / 100))
        self.sl["softness"] = Slider(sb, "Edge softness", 0, 100, 30, unit=" %", step=1,
                                     command=lambda v: self._set("softness", v / 100))
        self.sl["phase"] = Slider(sb, "Phase", 0, 360, 0, unit="°", step=1,
                                  command=lambda v: self._set("phase", v))

        section(b, "Modulation").pack(fill="x", pady=(px(10), px(7)))
        self.mod_seg = Segmented(b, [("none", "Off"), ("am", "AM"), ("fm", "FM"), ("sweep", "Sweep")], "none",
                                 command=self._on_mod, fill=True)
        self.mod_seg.pack(fill="x")
        self.mod_box = tk.Frame(b, bg=T.panel)
        self.mod_box.pack(fill="x", pady=(px(4), 0))
        mb = self.mod_box
        self.sl["mod_rate"] = Slider(mb, "Rate", 0.1, 50, 2, unit=" Hz", fmt=fmt_rate, log=True, step=0.01,
                                     command=lambda v: self._set("mod_rate", v))
        self.sl["am_depth"] = Slider(mb, "Depth", 0, 100, 100, unit=" %", step=1,
                                     command=lambda v: self._set("am_depth", v / 100))
        self.sl["fm_dev"] = Slider(mb, "Deviation", 0, 300, 20, unit=" Hz", fmt=lambda v: f"±{v:.1f}", step=0.5,
                                   command=lambda v: self._set("fm_dev", v))
        self.sl["sweep_to"] = Slider(mb, "Sweep to", 5, 1000, 300, unit=" Hz", fmt=fmt_hz, log=True, step=0.1,
                                     command=lambda v: self._set("sweep_to", v))
        self.sl["sweep_time"] = Slider(mb, "Duration", 0.2, 60, 5, unit=" s", fmt=lambda v: f"{v:.1f}",
                                       log=True, step=0.1, command=lambda v: self._set("sweep_time", v))
        self.sweep_row = tk.Frame(mb, bg=T.panel)
        self.pp_tg = Toggle(self.sweep_row, "Ping-pong", command=lambda v: self._set("sweep_pingpong", v))
        self.pp_tg.pack(side="left")
        self.mark_btn = Button(self.sweep_row, "◆ Mark  M", command=self.mark_sweep, height=26, size=11)
        self.mark_btn.pack(side="right")
        Button(self.sweep_row, "↺ Restart", command=self._restart_gen, height=26, size=11).pack(
            side="right", padx=(0, px(6)))
        self.sweep_now = tk.Label(self.sweep_row, text="", font=font(12, mono=True), fg=T.text, bg=T.panel)
        self.sweep_now.pack(side="left", padx=(px(12), 0))
        self.marks_box = tk.Frame(mb, bg=T.panel)

        bh = tk.Frame(b, bg=T.panel)
        bh.pack(fill="x", pady=(px(12), px(4)))
        self.gate_tg = Toggle(bh, "", command=lambda v: (self._set("gate", v), self._update_visibility()))
        self.gate_tg.pack(side="right", padx=(px(8), 0))
        section(bh, "Burst · pulse train").pack(side="left", fill="x", expand=True)
        self.gate_box = tk.Frame(b, bg=T.panel)
        self.gate_box.pack(fill="x")
        gb = self.gate_box
        self.sl["gate_on_ms"] = Slider(gb, "On time", 2, 2000, 60, unit=" ms", log=True, step=1,
                                       command=lambda v: self._set("gate_on_ms", v))
        self.sl["gate_off_ms"] = Slider(gb, "Off time", 2, 5000, 140, unit=" ms", log=True, step=1,
                                        command=lambda v: self._set("gate_off_ms", v))
        self.sl["gate_on_ms"].pack(fill="x")
        self.sl["gate_off_ms"].pack(fill="x")
        gr = tk.Frame(gb, bg=T.panel)
        gr.pack(fill="x", pady=(px(4), 0))
        self.sync_tg = Toggle(gr, "Restart phase per burst", command=lambda v: self._set("gate_sync", v))
        self.sync_tg.pack(side="left")
        self.gate_info = tk.Label(gr, text="", font=font(11, mono=True), fg=T.muted, bg=T.panel)
        self.gate_info.pack(side="right")

    # ------------------------------------------------------------------ scope
    def _build_scope(self, parent):
        card = Card(parent, "Scope")
        card.pack(fill="both", expand=True)
        h = card.header
        self.trace_chips = {}
        for key, label, color in (("out", "OUT", T.text), ("b", "B", T.b), ("a", "A", T.a)):
            c = Chip(h, label, color, True, command=lambda v, k=key: self.traces.__setitem__(k, v))
            c.pack(side="right", padx=(px(4), 0))
            self.trace_chips[key] = c
        self.win_seg = Segmented(h, [(10, "10"), (25, "25"), (50, "50"), (100, "100"), (250, "250"),
                                     (1000, "1000 ms")], 50, size=10, height=24, seg_pad=7,
                                 command=lambda v: setattr(self, "scope_ms", v))
        self.win_seg.pack(side="right", padx=(0, px(10)))
        self.mode_seg = Segmented(h, [("wave", "Wave"), ("fft", "Spectrum")], "wave", size=10, height=24,
                                  seg_pad=9, command=self._on_scope_mode)
        self.mode_seg.pack(side="right", padx=(0, px(10)))
        self.scope_cv = tk.Canvas(card.body, bg="#0f1219", highlightthickness=1, highlightbackground=T.border,
                                  height=px(240))
        self.scope_cv.pack(fill="both", expand=True)

    def _on_scope_mode(self, v):
        self.scope_mode = v

    # ------------------------------------------------------------------ outputs
    def _build_outputs(self, parent):
        card = Card(parent, "Outputs", "")
        card.pack(fill="x", pady=(px(12), 0))
        self.out_card = card
        h = card.header
        for label, code in (("Clear", ROUTE_OFF), ("All A+B", ROUTE_AB), ("All B", ROUTE_B), ("All A", ROUTE_A)):
            Button(h, label, command=lambda c=code: self._route_all(c), height=24, size=10,
                   color=ROUTE_COLORS[code] if code else T.bad).pack(side="right", padx=(px(4), 0))

        self.matrix = tk.Canvas(card.body, bg=T.panel, highlightthickness=0, height=px(124))
        self.matrix.pack(fill="x")
        self.matrix.bind("<Configure>", lambda e: self._draw_matrix())
        self.matrix.bind("<Button-1>", self._mx_press)
        self.matrix.bind("<B1-Motion>", self._mx_drag)
        self.matrix.bind("<ButtonRelease-1>", lambda e: setattr(self, "_paint", None))
        self.matrix.bind("<MouseWheel>", self._mx_wheel)
        self.matrix.bind("<Button-3>", self._mx_menu)
        tk.Label(card.body, text="click: cycle route  ·  drag: paint  ·  wheel: gain  ·  shift+click: show in "
                                 "scope  ·  right-click: menu", font=font(10), fg=T.dim, bg=T.panel).pack(
            anchor="w", pady=(px(6), px(4)))

        sp = tk.Frame(card.body, bg=T.panel)
        sp.pack(fill="x", pady=(px(4), 0))
        sp.grid_columnconfigure((1, 2), weight=1, uniform="s")
        left = tk.Frame(sp, bg=T.panel)
        left.grid(row=0, column=0, sticky="w", padx=(0, px(14)))
        tk.Label(left, text="SPATIAL PATTERN", font=font(10, "bold"), fg=T.dim, bg=T.panel).pack(anchor="w",
                                                                                                  pady=(0, px(4)))
        self.sp_seg = Segmented(left, [("static", "Static"), ("chase", "Chase"), ("pingpong", "Ping-pong"),
                                       ("random", "Random")], "static", color=T.ab, size=11,
                                command=self._on_spatial)
        self.sp_seg.pack(anchor="w")
        self.sl["step_ms"] = Slider(sp, "Step time", 20, 3000, 200, unit=" ms", log=True, step=1, color=T.ab,
                                    command=lambda v: setattr(self.engine.spatial, "step_ms", v))
        self.sl["step_ms"].grid(row=0, column=1, sticky="ew", padx=(0, px(14)))
        self.sl["overlap"] = Slider(sp, "Crossfade", 0, 100, 30, unit=" %", step=1, color=T.ab,
                                    command=lambda v: setattr(self.engine.spatial, "overlap", v / 100))
        self.sl["overlap"].grid(row=0, column=2, sticky="ew")

    # ------------------------------------------------------------------ right column
    def _build_effects(self, parent):
        card = Card(parent, "Effects", "one-shot")
        card.pack(fill="x")
        grid = tk.Frame(card.body, bg=T.panel)
        grid.pack(fill="x")
        grid.grid_columnconfigure((0, 1, 2), weight=1, uniform="fx")
        self.fx_btns = {}
        for k, (name, (_, key)) in enumerate(EFFECTS.items()):
            btn = Button(grid, name, command=lambda n=name: self.play_fx(n), sub=key, height=44, size=11,
                         width=80, color=T.ab)
            btn.grid(row=k // 3, column=k % 3, sticky="ew", padx=px(2), pady=px(2))
            self.fx_btns[name] = btn
        self.sl["fx_gain"] = Slider(card.body, "Effect level", 0, 100, 80, unit=" %", step=1, color=T.ab,
                                    command=lambda v: setattr(self.engine, "fx_gain", v / 100))
        self.sl["fx_gain"].pack(fill="x", pady=(px(6), 0))

    def _build_master(self, parent):
        card = Card(parent, "Master")
        card.pack(fill="x", pady=(px(12), 0))
        self.sl["master"] = Slider(card.body, "Volume", 0, 100, 80, unit=" %", step=1, color=T.good,
                                   command=lambda v: setattr(self.engine, "master", v / 100))
        self.sl["master"].pack(fill="x")
        self.sl["limit"] = Slider(card.body, "Safety limit", 5, 100, 95, unit=" %", step=1, color=T.bad,
                                  command=lambda v: setattr(self.engine, "limit", v / 100))
        self.sl["limit"].pack(fill="x")
        self.master_meter = tk.Canvas(card.body, height=px(26), bg=T.panel, highlightthickness=0)
        self.master_meter.pack(fill="x", pady=(px(4), 0))

    def _build_presets(self, parent):
        card = Card(parent, "Presets")
        card.pack(fill="both", expand=True, pady=(px(12), 0))
        grid = tk.Frame(card.body, bg=T.panel)
        grid.pack(fill="x")
        grid.grid_columnconfigure((0, 1), weight=1, uniform="p")
        for k, pr in enumerate(BUILTIN_PRESETS):
            Button(grid, pr["name"], command=lambda p=pr: self.apply_preset(p), kind="row", anchor="w",
                   height=30, size=11, width=60).grid(row=k // 2, column=k % 2, sticky="ew", padx=px(2),
                                                      pady=px(2))
        row = tk.Frame(card.body, bg=T.panel)
        row.pack(fill="x", pady=(px(8), 0))
        row.grid_columnconfigure((0, 1), weight=1, uniform="io")
        Button(row, "Save…", command=self.save_preset, height=28, size=11, width=60).grid(
            row=0, column=0, sticky="ew", padx=px(2))
        Button(row, "Load…", command=self.load_preset, height=28, size=11, width=60).grid(
            row=0, column=1, sticky="ew", padx=px(2))

    # ================================================================== generator logic
    @property
    def p(self) -> GenParams:
        return self.engine.gens[self.sel].params

    def _set(self, key, value):
        setattr(self.p, key, value)
        if key == "waveform":
            self._update_visibility()

    def _on_mod(self, v):
        self.p.mod = v
        if v == "sweep":
            self.engine.gens[self.sel].restart()
        self._update_visibility()

    def _restart_gen(self):
        self.engine.gens[self.sel].restart()
        self.flash("Restarted sweep / burst timing")

    def _toggle_gen(self, i, v):
        self.engine.gens[i].params.enabled = v
        if v:
            self._ensure_routed(i)

    def select_gen(self, i):
        self.sel = i
        c = GEN_COLORS[i]
        for t in self.tabs:
            t.set_selected(t.idx == i)
        for key, s in self.sl.items():
            if key in ("step_ms", "overlap", "fx_gain", "master", "limit"):
                continue
            s.color = c
            s.redraw()
        self.wave.set_color(c)
        self.mod_seg.set_color(c)
        for tg in (self.gate_tg, self.pp_tg, self.sync_tg):
            tg.color = c
            tg.redraw()
        self._load_gen_widgets()

    def _load_gen_widgets(self):
        p = self.p
        self.wave.set(p.waveform)
        s = self.sl
        s["freq"].set(p.freq)
        s["amp"].set(p.amp * 100)
        s["duty"].set(p.duty * 100)
        s["symmetry"].set(p.symmetry * 100)
        s["softness"].set(p.softness * 100)
        s["phase"].set(p.phase)
        s["mod_rate"].set(p.mod_rate)
        s["am_depth"].set(p.am_depth * 100)
        s["fm_dev"].set(p.fm_dev)
        s["sweep_to"].set(p.sweep_to)
        s["sweep_time"].set(p.sweep_time)
        s["gate_on_ms"].set(p.gate_on_ms)
        s["gate_off_ms"].set(p.gate_off_ms)
        self.mod_seg.set(p.mod)
        self.pp_tg.set(p.sweep_pingpong)
        self.gate_tg.set(p.gate)
        self.sync_tg.set(p.gate_sync)
        for i, t in enumerate(self.tabs):
            t.toggle.set(self.engine.gens[i].params.enabled)
        self._update_visibility()
        self._rebuild_marks()

    def _update_visibility(self):
        p = self.p
        wf = p.waveform
        order = {
            "sine": ["freq", "amp", "phase"],
            "square": ["freq", "amp", "duty", "softness", "phase"],
            "triangle": ["freq", "amp", "symmetry", "phase"],
            "saw": ["freq", "amp", "softness", "phase"],
            "noise": ["amp"],
            "rumble": ["freq", "amp"],
        }[wf]
        self.sl["freq"].label = "Grain rate" if wf == "rumble" else ("Start freq" if p.mod == "sweep" else "Frequency")
        self.sl["freq"].redraw()
        for k in ("freq", "amp", "duty", "symmetry", "softness", "phase"):
            self.sl[k].pack_forget()
        for k in order:
            self.sl[k].pack(fill="x")

        for w in (self.sl["mod_rate"], self.sl["am_depth"], self.sl["fm_dev"], self.sl["sweep_to"],
                  self.sl["sweep_time"], self.sweep_row, self.marks_box):
            w.pack_forget()
        m = p.mod
        if m == "am":
            self.sl["mod_rate"].pack(fill="x")
            self.sl["am_depth"].pack(fill="x")
        elif m == "fm":
            self.sl["mod_rate"].pack(fill="x")
            self.sl["fm_dev"].pack(fill="x")
        elif m == "sweep":
            self.sl["sweep_to"].pack(fill="x")
            self.sl["sweep_time"].pack(fill="x")
            self.sweep_row.pack(fill="x", pady=(px(2), 0))
            if self.marks[self.sel]:
                self.marks_box.pack(fill="x", pady=(px(6), 0))

        if p.gate:
            self.gate_box.pack(fill="x")
        else:
            self.gate_box.pack_forget()

    # ------------------------------------------------------------------ sweep marks
    def mark_sweep(self):
        g = self.engine.gens[self.sel]
        if g.params.mod != "sweep":
            self.flash("Mark works while a sweep is running", T.warn)
            return
        if not (self.engine.running and self.engine.output_on):
            self.flash("Start output to mark a frequency", T.warn)
            return
        f = round(g.inst_freq, 1)
        mk = self.marks[self.sel]
        mk.append(f)
        del mk[:-6]
        self._rebuild_marks()
        self._update_visibility()
        self.flash(f"Marked {fmt_hz(f)} Hz")

    def _rebuild_marks(self):
        for w in self.marks_box.winfo_children():
            w.destroy()
        mk = self.marks[self.sel]
        if not mk:
            return
        tk.Label(self.marks_box, text="Marks", font=font(10, "bold"), fg=T.dim, bg=T.panel).pack(side="left")
        for f in mk:
            Button(self.marks_box, f"{fmt_hz(f)}", command=lambda f=f: self._use_mark(f), height=24, size=10,
                   color=GEN_COLORS[self.sel]).pack(side="left", padx=(px(4), 0))
        Button(self.marks_box, "×", command=self._clear_marks, height=24, size=11, width=26,
               color=T.bad).pack(side="right")

    def _use_mark(self, f):
        self.p.freq = f
        self.p.mod = "none"
        self._load_gen_widgets()
        self.flash(f"Generator {'AB'[self.sel]} → {fmt_hz(f)} Hz (sweep off)")

    def _clear_marks(self):
        self.marks[self.sel] = []
        self._rebuild_marks()
        self._update_visibility()

    # ================================================================== routing matrix
    def _mx_cells(self):
        C = self.engine.n_channels
        w = max(self.matrix.winfo_width(), 100)
        cols = min(C, 10)
        rows = math.ceil(C / cols)
        gap = px(6)
        cw = (w - gap * (cols - 1)) / cols
        ch = px(56)
        need = rows * ch + (rows - 1) * gap
        if int(self.matrix["height"]) != need:
            self.matrix.configure(height=need)
        return [((i % cols) * (cw + gap), (i // cols) * (ch + gap),
                 (i % cols) * (cw + gap) + cw, (i // cols) * (ch + gap) + ch) for i in range(C)]

    def _draw_matrix(self):
        cv = self.matrix
        cv.delete("all")
        self._meter_items = []
        route, gain = self.engine.route, self.engine.gain
        for i, (x0, y0, x1, y1) in enumerate(self._mx_cells()):
            code = int(route[i])
            col = ROUTE_COLORS[code]
            on = code > 0
            round_rect(cv, x0 + 1, y0 + 1, x1 - 1, y1 - 1, px(8),
                       fill=mix(T.panel2, col, 0.10) if on else T.panel2, outline=col if on else T.border)
            cv.create_text(x0 + px(9), y0 + px(13), text=str(i + 1), anchor="w", font=font(14, "bold"),
                           fill=T.text if on else T.dim)
            cv.create_text(x1 - px(8), y0 + px(13), text=ROUTE_LABELS[code], anchor="e",
                           font=font(10, "bold"), fill=col)
            if i == self.engine.monitor_ch:
                cx, cy, r = x0 + px(9) + text_width(font(14, "bold"), str(i + 1)) + px(8), y0 + px(13), px(3.5)
                cv.create_polygon(cx, cy - r, cx + r, cy, cx, cy + r, cx - r, cy, fill=T.text, outline="")
            if gain[i] < 0.999:
                cv.create_text(x0 + px(9), y1 - px(22), text=f"{gain[i] * 100:.0f}%", anchor="w",
                               font=font(10, mono=True), fill=T.muted)
            mx0, mx1, my0, my1 = x0 + px(8), x1 - px(8), y1 - px(11), y1 - px(7)
            cv.create_rectangle(mx0, my0, mx1, my1, fill=mix(T.raised, col, 0.18) if on else T.raised,
                                outline="")
            item = cv.create_rectangle(mx0, my0, mx0, my1, fill=col if on else T.dim, outline="",
                                       width=0, state="hidden")
            self._meter_items.append((item, mx0, my0, mx1, my1))
        C = self.engine.n_channels
        self.out_card.subtitle.configure(text=f"{C} channel{'s' if C != 1 else ''}")

    def _mx_hit(self, e):
        for i, (x0, y0, x1, y1) in enumerate(self._mx_cells()):
            if x0 <= e.x <= x1 and y0 <= e.y <= y1:
                return i
        return None

    def _mx_press(self, e):
        i = self._mx_hit(e)
        if i is None:
            return
        if e.state & 0x0001:
            self.engine.monitor_ch = i
            self._draw_matrix()
            return
        code = (int(self.engine.route[i]) + 1) % 4
        self.engine.set_route(i, code)
        self._paint = (code, {i})
        self._draw_matrix()

    def _mx_drag(self, e):
        if not self._paint:
            return
        i = self._mx_hit(e)
        code, done = self._paint
        if i is not None and i not in done:
            done.add(i)
            self.engine.set_route(i, code)
            self._draw_matrix()

    def _mx_wheel(self, e):
        i = self._mx_hit(e)
        if i is None:
            return
        g = float(self.engine.gain[i]) + (0.05 if e.delta > 0 else -0.05)
        self.engine.set_gain(i, round(g, 2))
        self._draw_matrix()

    def _mx_menu(self, e):
        i = self._mx_hit(e)
        if i is None:
            return
        m = tk.Menu(self.root, tearoff=0, bg=T.panel2, fg=T.text, activebackground=T.raised,
                    activeforeground=T.a, bd=0, relief="flat", font=font(12))
        m.add_command(label=f"Channel {i + 1}", state="disabled")
        for code in (ROUTE_A, ROUTE_B, ROUTE_AB, ROUTE_OFF):
            m.add_command(label=f"Route → {ROUTE_LABELS[code]}",
                          command=lambda c=code: (self.engine.set_route(i, c), self._draw_matrix()))
        m.add_separator()
        for g in (100, 75, 50, 25):
            m.add_command(label=f"Gain {g} %",
                          command=lambda g=g: (self.engine.set_gain(i, g / 100), self._draw_matrix()))
        m.add_separator()
        m.add_command(label="Show in scope", command=lambda: (setattr(self.engine, "monitor_ch", i),
                                                               self._draw_matrix()))
        m.add_command(label="Solo this channel", command=lambda: self._solo(i))
        m.tk_popup(e.x_root, e.y_root)

    def _solo(self, i):
        codes = np.zeros(self.engine.n_channels, np.int8)
        codes[i] = self.engine.route[i] or ROUTE_A
        self.engine.set_routes(codes)
        self.engine.monitor_ch = i
        self._draw_matrix()

    def _route_all(self, code):
        self.engine.set_routes(np.full(self.engine.n_channels, code, np.int8))
        self._draw_matrix()

    def _ensure_routed(self, i):
        """Turning a generator on with nowhere to go would be silent — route it where the other one is."""
        bit = ROUTE_A if i == 0 else ROUTE_B
        r = self.engine.route.copy()
        if np.any(r & bit):
            return
        r = np.where(r > 0, r | bit, r) if np.any(r > 0) else np.full_like(r, bit)
        self.engine.set_routes(r)
        self._draw_matrix()

    def _on_spatial(self, v):
        self.engine.spatial.mode = v

    # ================================================================== effects / presets
    def play_fx(self, name):
        if not self.engine.running:
            if not self.connect():
                return
        if not self.engine.output_on:
            self.set_output(True)
        self.engine.trigger_fx(name)
        self.fx_btns[name].flash()

    def apply_preset(self, pr):
        for i, key in enumerate("AB"):
            g = self.engine.gens[i]
            if pr.get(key):
                g.params.update(GenParams.from_dict(pr[key]).to_dict())
                g.restart()
            else:
                g.params.enabled = False
        self.engine.spatial.update(pr.get("spatial", {"mode": "static"}))
        for i in range(2):
            if self.engine.gens[i].params.enabled:
                self._ensure_routed(i)
        self._sync_ui()
        self.flash(f"{pr['name']} — {pr.get('hint', '')}")

    def save_preset(self):
        PRESET_DIR.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(parent=self.root, initialdir=PRESET_DIR, defaultextension=".json",
                                            filetypes=[("Haptic preset", "*.json")], title="Save preset")
        if not path:
            return
        d = self.engine.to_dict()
        d["marks"] = self.marks
        Path(path).write_text(json.dumps(d, indent=1), encoding="utf-8")
        self.flash(f"Saved {Path(path).name}")

    def load_preset(self):
        PRESET_DIR.mkdir(exist_ok=True)
        path = filedialog.askopenfilename(parent=self.root, initialdir=PRESET_DIR,
                                          filetypes=[("Haptic preset", "*.json")], title="Load preset")
        if not path:
            return
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            self.engine.load_dict(d)
            if "marks" in d:
                self.marks = [list(d["marks"][0]), list(d["marks"][1])]
        except Exception as exc:
            self.flash(f"Could not load preset: {exc}", T.bad)
            return
        self._sync_ui()
        self.flash(f"Loaded {Path(path).name}")

    # ================================================================== device / output
    def refresh_devices(self, auto_connect=False, prefer=None):
        if not self.engine.running:
            refresh_portaudio()
        self.devices = list_output_devices()
        labels = [self._dev_label(d) for d in self.devices]
        self.dev_combo.configure(values=labels)
        cur = self.engine.device if self.engine.running else None
        pick = cur or pick_default_device(self.devices, *(
            (prefer.get("name"), prefer.get("hostapi")) if prefer else (None, None)))
        if pick:
            self.dev_var.set(self._dev_label(pick))
        if not self.devices:
            self.flash("No audio output devices found — check the USB data cable", T.bad)
        elif auto_connect and pick and "HSD" in pick["name"].upper():
            self.connect()
        self._update_conn_ui()

    @staticmethod
    def _dev_label(d):
        api = d["hostapi"].replace("Windows ", "")
        return f"{d['name']}  ·  {api}  ·  {d['channels']} ch"

    def _selected_device(self):
        lbl = self.dev_var.get()
        for d in self.devices:
            if self._dev_label(d) == lbl:
                return d
        return None

    def connect(self) -> bool:
        dev = self._selected_device()
        if dev is None:
            self.flash("Select an output device first", T.warn)
            return False
        was_on = self.engine.output_on
        try:
            msg = self.engine.start(dev, self.blocksize, self.exclusive)
        except Exception as exc:
            self.engine.output_on = False
            self.flash(f"Could not open {dev['name']}: {exc}", T.bad)
            self._update_conn_ui()
            return False
        self.engine.output_on = was_on
        self._mdisp = np.zeros(self.engine.n_channels)
        self._draw_matrix()
        self.flash(msg, T.good)
        self._update_conn_ui()
        return True

    def disconnect(self):
        self.set_output(False)
        self.engine.stop()
        self._update_conn_ui()
        self.flash("Disconnected")

    def _toggle_connect(self):
        if self.engine.running:
            self.disconnect()
        else:
            self.connect()

    def _on_blocksize(self):
        self.root.focus_set()
        self.blocksize = int(self.bs_var.get())
        if self.engine.running:
            self.connect()

    def _on_exclusive(self, v):
        self.exclusive = v
        if self.engine.running:
            self.connect()

    def set_output(self, on: bool):
        self.engine.output_on = bool(on)
        if on:
            self.play_btn.configure_button("■  Stop", kind="danger")
        else:
            self.play_btn.configure_button("▶  Play", kind="primary")

    def toggle_output(self):
        if self.engine.output_on:
            self.set_output(False)
            return
        if not self.engine.running and not self.connect():
            return
        self.set_output(True)

    def panic(self):
        self.set_output(False)
        self.engine.clear_fx()
        self.flash("Panic — output muted, effects cleared", T.warn)

    def _update_conn_ui(self):
        run = self.engine.running
        self.status_dot.delete("all")
        c = T.good if run else T.dim
        self.status_dot.create_oval(px(2), px(2), px(10), px(10), fill=c, outline="")
        if run:
            d = self.engine.device
            mode = "exclusive" if self.engine.exclusive_active else "shared"
            self.status_lbl.configure(text=f"{self.engine.sr // 1000} kHz · {self.engine.n_channels} ch · {mode}",
                                      fg=T.text)
            self.conn_btn.configure_button("Disconnect")
        else:
            self.status_lbl.configure(text="Not connected", fg=T.muted)
            self.conn_btn.configure_button("Connect")

    # ================================================================== keys
    def _bind_keys(self):
        self.root.bind_all("<KeyPress>", self._on_key)

    def _on_key(self, e):
        w = self.root.focus_get()
        if isinstance(w, (tk.Entry, ttk.Entry)) and not isinstance(w, ttk.Combobox):
            return None
        ks, ch = e.keysym, (e.char or "")
        if ks == "space":
            self.toggle_output()
            return "break"
        if ks == "Escape":
            self.panic()
            return "break"
        if ks in ("Tab", "ISO_Left_Tab"):
            self.select_gen(1 - self.sel)
            return "break"
        if ch.lower() == "m":
            self.mark_sweep()
            return "break"
        if ch in FX_KEYS:
            self.play_fx(FX_KEYS[ch])
            return "break"
        return None

    # ================================================================== live update
    def _tick(self):
        try:
            self._draw_scope()
            self._update_meters()
            self._update_live()
            now = time.monotonic()
            if now - self._last_status > 0.4:
                self._last_status = now
                self._update_status()
        except Exception:
            traceback.print_exc()
        self.root.after(33, self._tick)

    def _update_live(self):
        for i, t in enumerate(self.tabs):
            p = self.engine.gens[i].params
            if p.waveform == "noise":
                s = f"Noise · {p.amp * 100:.0f}%"
            else:
                s = f"{WAVE_LABELS[p.waveform]} · {fmt_hz(p.freq)} Hz · {p.amp * 100:.0f}%"
            flags = []
            if p.mod != "none":
                flags.append({"am": "AM", "fm": "FM", "sweep": f"SWEEP→{fmt_hz(p.sweep_to)}"}[p.mod])
            if p.gate:
                flags.append("BURST")
            if not p.enabled:
                flags = ["OFF"]
            key = (s, tuple(flags))
            if key != self._summary_cache[i]:
                self._summary_cache[i] = key
                t.summary.configure(text=s, fg=T.text if p.enabled else T.dim)
                t.flags.configure(text="  ".join(flags) or "CONTINUOUS")
        p = self.p
        g = self.engine.gens[self.sel]
        if p.mod == "sweep":
            live = self.engine.running and self.engine.output_on
            self.sweep_now.configure(text=f"{fmt_hz(g.inst_freq)} Hz" if live else "— Hz")
        if p.gate:
            per = p.gate_on_ms + p.gate_off_ms
            self.gate_info.configure(text=f"{1000 / per:.1f}/s · {100 * p.gate_on_ms / per:.0f}%")

    def _update_meters(self):
        e = self.engine
        if len(self._mdisp) != e.n_channels:
            self._mdisp = np.zeros(e.n_channels)
        if e.running:
            m = e.meters.copy()
            e.meters[:] = 0.0
        else:
            m = np.zeros(e.n_channels)
        self._mdisp = np.maximum(m, self._mdisp * 0.85)
        for (item, x0, y0, x1, y1), lv in zip(getattr(self, "_meter_items", []), self._mdisp):
            self.matrix.coords(item, x0, y0, x0 + (x1 - x0) * min(1.0, lv), y1)
            self.matrix.itemconfigure(item, state="normal" if lv > 0.003 else "hidden")

        pk = e.peak if e.running else 0.0
        e.peak = 0.0
        self._peak_disp = max(pk, self._peak_disp * 0.88)
        if e.clips != self._last_clips:
            self._last_clips = e.clips
            self._clip_flash = 1.0
        self._clip_flash = max(0.0, self._clip_flash - 0.03)
        cv = self.master_meter
        cv.delete("all")
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 20:
            return
        bw = w - px(52)
        y0, y1 = px(8), px(18)
        round_rect(cv, 0, y0, bw, y1, px(4), fill=T.raised, outline="")
        lv = min(1.0, self._peak_disp)
        if lv > 0.002:
            col = T.good if lv < e.limit * 0.9 else (T.warn if lv < e.limit * 0.999 else T.bad)
            round_rect(cv, 0, y0, max(px(8), bw * lv), y1, px(4), fill=col, outline="")
        xl = bw * e.limit
        cv.create_line(xl, y0 - px(4), xl, y1 + px(4), fill=T.bad, width=px(2))
        clip_col = mix(T.dim, T.bad, self._clip_flash) if self._clip_flash > 0 else T.dim
        cv.create_text(w, (y0 + y1) / 2, text="CLIP", anchor="e", font=font(10, "bold"), fill=clip_col)

    def _update_status(self):
        e = self.engine
        if e.running:
            st = e.stats()
            mode = "exclusive" if e.exclusive_active else "shared"
            txt = (f"{e.sr} Hz · {e.n_channels} ch · {e.device['hostapi'].replace('Windows ', '')} {mode} · "
                   f"block {e.blocksize} · latency {st.get('latency', 0) * 1000:.0f} ms · "
                   f"CPU {st.get('cpu', 0) * 100:.0f}% · xruns {e.xruns}")
        else:
            txt = "offline · scope shows a preview"
        self.foot_lbl.configure(text=txt)
        if e.last_error:
            self.flash(f"Audio error: {e.last_error}", T.bad)
            e.last_error = None
        if e.running is False and e.stream is not None:
            self._update_conn_ui()
        text, col, t0 = self._msg
        if text and time.monotonic() - t0 > 6:
            self._msg = ("", T.muted, 0)
            self.msg_lbl.configure(text="")

    def flash(self, text, color=T.muted):
        self._msg = (text, color, time.monotonic())
        self.msg_lbl.configure(text=text, fg=color)

    # ------------------------------------------------------------------ scope drawing
    def _scope_data(self, n):
        e = self.engine
        if e.running:
            return e.scope_snapshot(n), True
        return e.preview(n), False

    @staticmethod
    def _trace_coords(v, x0, x1, ymid, yscale):
        n = len(v)
        w = max(2, int(x1 - x0))
        if n > 2 * w:
            k = n // w
            vv = v[: k * w].reshape(w, k)
            ys = np.column_stack([vv.max(1), vv.min(1)]).ravel()
            xs = np.repeat(np.linspace(x0, x1, w), 2)
        else:
            ys = v
            xs = np.linspace(x0, x1, n)
        return np.column_stack([xs, ymid - ys * yscale]).ravel().tolist()

    def _draw_scope(self):
        cv = self.scope_cv
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 40 or h < 40:
            return
        cv.delete("all")
        if self.scope_mode == "fft":
            self._draw_spectrum(cv, w, h)
            return
        e = self.engine
        sr = e.sr
        win = max(16, int(sr * self.scope_ms / 1000))
        extra = int(sr * 0.04)
        data, live = self._scope_data(win + extra)

        # trigger: last rising zero-crossing inside the search area
        trig_row = 2 if self.traces["out"] and np.abs(data[2]).max() > 1e-3 else (0 if e.gens[0].params.enabled else 1)
        seg = data[trig_row, : extra + 1]
        idx = np.flatnonzero((seg[:-1] < 0) & (seg[1:] >= 0))
        start = int(idx[-1]) + 1 if len(idx) else extra
        view = data[:, start:start + win]

        pl, pr, pt, pb = px(44), px(12), px(16), px(22)
        x0, x1, y0, y1 = pl, w - pr, pt, h - pb
        ymid = (y0 + y1) / 2
        ys = (y1 - y0) / 2 * 0.92
        for k in range(11):
            x = x0 + (x1 - x0) * k / 10
            cv.create_line(x, y0, x, y1, fill=T.grid)
        for lv in (-1, -0.5, 0, 0.5, 1):
            y = ymid - lv * ys
            cv.create_line(x0, y, x1, y, fill=T.dim if lv == 0 else T.grid, dash=() if lv in (0,) else (2, 4))
            cv.create_text(x0 - px(6), y, text=f"{lv:+.1f}" if lv else "0", anchor="e", font=font(10, mono=True),
                           fill=T.dim)
        lim = e.limit
        for s in (-1, 1):
            y = ymid - s * lim * ys
            cv.create_line(x0, y, x1, y, fill=mix(T.panel2, T.bad, 0.45), dash=(6, 4))
        cv.create_text(x1, h - px(10), text=f"{self.scope_ms} ms / screen", anchor="e", font=font(10, mono=True),
                       fill=T.dim)
        cv.create_text(x0, h - px(10), text="0", anchor="w", font=font(10, mono=True), fill=T.dim)

        rows = []
        if self.traces["a"] and e.gens[0].params.enabled:
            rows.append((0, T.a, px(1.6)))
        if self.traces["b"] and e.gens[1].params.enabled:
            rows.append((1, T.b, px(1.6)))
        if self.traces["out"]:
            rows.append((2, T.text, px(2.2)))
        for r, col, wd in rows:
            v = view[r]
            if len(v) > 1:
                cv.create_line(*self._trace_coords(v, x0, x1, ymid, ys), fill=col, width=wd)

        # legend
        lx = x0 + px(10)
        for i in range(2):
            p = e.gens[i].params
            if not p.enabled:
                continue
            f = e.gens[i].inst_freq if live else p.freq
            txt = f"{'AB'[i]}  {WAVE_LABELS[p.waveform]}  {fmt_hz(f)} Hz" if p.waveform != "noise" else f"{'AB'[i]}  Noise"
            cv.create_text(lx, y0 + px(12), text=txt, anchor="w", font=font(11, mono=True), fill=GEN_COLORS[i])
            lx += text_width(font(11, mono=True), txt) + px(18)
        cv.create_text(x1 - px(8), y0 + px(12), text=f"OUT · ch {e.monitor_ch + 1}", anchor="e",
                       font=font(11, mono=True), fill=T.text)
        if not live:
            cv.create_text((x0 + x1) / 2, h - px(10), text="PREVIEW — not connected", font=font(10, "bold"),
                           fill=T.dim)
        elif not e.output_on:
            cv.create_text((x0 + x1) / 2, h - px(10), text="OUTPUT MUTED — press Space", font=font(10, "bold"),
                           fill=T.warn)

    def _draw_spectrum(self, cv, w, h):
        e = self.engine
        sr = e.sr
        N = 16384
        data, live = self._scope_data(N)
        win = np.hanning(N)
        scale = 2.0 / win.sum()
        freqs = np.fft.rfftfreq(N, 1.0 / sr)
        fmin, fmax = 5.0, min(2000.0, sr / 2)
        pl, pr, pt, pb = px(44), px(12), px(16), px(22)
        x0, x1, y0, y1 = pl, w - pr, pt, h - pb
        lo_db, hi_db = -90.0, 0.0

        def fx(f):
            return x0 + (x1 - x0) * math.log(f / fmin) / math.log(fmax / fmin)

        def dy(db):
            return y0 + (y1 - y0) * (hi_db - db) / (hi_db - lo_db)

        for f in (10, 20, 50, 100, 200, 500, 1000, 2000):
            if f <= fmax:
                x = fx(f)
                cv.create_line(x, y0, x, y1, fill=T.grid)
                cv.create_text(x, h - px(10), text=f"{f if f < 1000 else str(f // 1000) + 'k'}",
                               font=font(10, mono=True), fill=T.dim)
        for db in (0, -20, -40, -60, -80):
            y = dy(db)
            cv.create_line(x0, y, x1, y, fill=T.grid, dash=(2, 4))
            cv.create_text(x0 - px(6), y, text=f"{db}", anchor="e", font=font(10, mono=True), fill=T.dim)

        mask = (freqs >= fmin) & (freqs <= fmax)
        fsel = freqs[mask]
        xpix = (x0 + (x1 - x0) * np.log(fsel / fmin) / math.log(fmax / fmin)).astype(int)
        starts = np.flatnonzero(np.r_[True, np.diff(xpix) > 0])
        rows = []
        if self.traces["a"] and e.gens[0].params.enabled:
            rows.append((0, T.a, px(1.4)))
        if self.traces["b"] and e.gens[1].params.enabled:
            rows.append((1, T.b, px(1.4)))
        if self.traces["out"]:
            rows.append((2, T.text, px(2)))
        peak_txt = None
        for r, col, wd in rows:
            X = np.abs(np.fft.rfft(data[r] * win)) * scale
            db = 20 * np.log10(X[mask] + 1e-9)
            dbp = np.maximum.reduceat(db, starts)
            xp = xpix[starts]
            yp = np.clip(y0 + (y1 - y0) * (hi_db - dbp) / (hi_db - lo_db), y0, y1)
            if len(xp) > 1:
                cv.create_line(*np.column_stack([xp, yp]).ravel().tolist(), fill=col, width=wd)
            if peak_txt is None or r == 2:
                k = int(np.argmax(db))
                if db[k] > lo_db + 10:
                    peak_txt = (fsel[k], db[k], col)
        if peak_txt:
            f, d, col = peak_txt
            x, y = fx(f), dy(d)
            cv.create_oval(x - px(4), y - px(4), x + px(4), y + px(4), outline=col, width=px(2))
            cv.create_text(x1 - px(8), y0 + px(12), text=f"peak {fmt_hz(f)} Hz · {d:.1f} dB", anchor="e",
                           font=font(11, mono=True), fill=col)
        if not live:
            cv.create_text((x0 + x1) / 2, y0 + px(12), text="PREVIEW — not connected", font=font(10, "bold"),
                           fill=T.dim)

    # ================================================================== session
    def _read_session(self) -> dict:
        try:
            return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _apply_session(self, d: dict):
        if not d:
            return
        try:
            self.engine.load_dict(d)
            self.blocksize = int(d.get("blocksize", self.blocksize))
            self.exclusive = bool(d.get("exclusive", self.exclusive))
            self.scope_mode = d.get("scope_mode", self.scope_mode)
            self.scope_ms = int(d.get("scope_ms", self.scope_ms))
            self.traces.update(d.get("traces", {}))
            mk = d.get("marks")
            if mk and len(mk) == 2:
                self.marks = [list(mk[0]), list(mk[1])]
            self.sel = int(d.get("sel", 0)) % 2
        except Exception:
            traceback.print_exc()

    def _save_session(self):
        d = self.engine.to_dict()
        dev = self.engine.device or self._selected_device()
        d.update({
            "device": {"name": dev["name"], "hostapi": dev["hostapi"]} if dev else None,
            "blocksize": self.blocksize, "exclusive": self.exclusive, "scope_mode": self.scope_mode,
            "scope_ms": self.scope_ms, "traces": self.traces, "marks": self.marks, "sel": self.sel,
        })
        try:
            SESSION_FILE.write_text(json.dumps(d, indent=1), encoding="utf-8")
        except Exception:
            traceback.print_exc()

    def _sync_ui(self):
        e = self.engine
        self.bs_var.set(str(self.blocksize))
        self.excl_tg.set(self.exclusive)
        self.mode_seg.set(self.scope_mode)
        self.win_seg.set(self.scope_ms)
        for k, c in self.trace_chips.items():
            c.set(self.traces.get(k, True))
        self.sp_seg.set(e.spatial.mode)
        self.sl["step_ms"].set(e.spatial.step_ms)
        self.sl["overlap"].set(e.spatial.overlap * 100)
        self.sl["fx_gain"].set(e.fx_gain * 100)
        self.sl["master"].set(e.master * 100)
        self.sl["limit"].set(e.limit * 100)
        self._summary_cache = [None, None]
        self.select_gen(self.sel)
        self._draw_matrix()

    def on_close(self):
        self._save_session()
        self.engine.output_on = False
        try:
            self.root.after(60, self._finish_close)
        except Exception:
            self._finish_close()

    def _finish_close(self):
        self.engine.stop()
        self.root.destroy()


# ============================================================================ entry point
def _dark_titlebar(root):
    if sys.platform != "win32":
        return
    try:
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        val = ctypes.c_int(1)
        for attr in (20, 19):
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(val), ctypes.sizeof(val)) == 0:
                break
    except Exception:
        pass


def main():
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    root = tk.Tk()
    root.title("Tactile Lab Generator")
    W.init_scale(root)
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = min(px(1480), sw - 40), min(px(900), sh - 80)
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 2 - 20)}")
    root.minsize(min(px(1220), sw - 40), min(px(760), sh - 80))
    root.configure(bg=T.bg)
    _dark_titlebar(root)
    app = TactileLabGenerator(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
