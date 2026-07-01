#!/usr/bin/env python3
"""Tk (Tcl/Tk) GUI front-end for the slitless/imaging ETC (slitless_etc.py).

tkinter is Python's binding to Tcl/Tk, so this is a Tcl/Tk GUI that drives the
exposure-time calculator directly and embeds the matplotlib plots live.

Modes    : Spectroscopy (slitless emission line)  /  Imaging (broadband point source)
Calc     : Limiting depth (Nsigma)  /  S/N for a source  /  Exposure for target S/N
Also     : telescope aperture, zodi level, telescope-thermal cooling tradeoff,
           saturation (full-well) check, per-pixel noise budget, Roman/Euclid check.

    python etc_gui.py
"""
import io, contextlib
from dataclasses import replace
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import slitless_etc as etc

HR = 3600.0


class ETCGui:
    def __init__(self, root):
        root.title("Space ETC  —  imaging & slitless spectroscopy")
        self.root = root
        panel = ttk.Frame(root, padding=6)
        panel.grid(row=0, column=0, sticky="ns")

        # -- telescope preset (auto-fills the specs) --
        self._obs = 0.15
        self._band = (0.36, 3.00)
        tf = ttk.LabelFrame(panel, text="telescope", padding=4); tf.pack(fill="x")
        self.telescope = tk.StringVar(value="3.5mST")
        ttk.OptionMenu(tf, self.telescope, "3.5mST", *etc.TELESCOPE_PRESETS.keys(),
                       command=lambda _=None: self.apply_preset()).pack(fill="x")
        self.element = tk.StringVar()
        self.elem_cb = ttk.Combobox(tf, textvariable=self.element, state="readonly")
        self.elem_cb.pack(fill="x", pady=2)
        self.elem_cb.bind("<<ComboboxSelected>>", lambda _e: self.apply_element())

        # -- mode & calculation type --
        self.mode = tk.StringVar(value="Spectroscopy")
        self.calc = tk.StringVar(value="Limiting depth")
        mf = ttk.LabelFrame(panel, text="mode", padding=4); mf.pack(fill="x")
        for m in ("Spectroscopy", "Imaging"):
            ttk.Radiobutton(mf, text=m, value=m, variable=self.mode,
                            command=self.on_mode_change).pack(side="left")
        cf = ttk.LabelFrame(panel, text="calculation", padding=4); cf.pack(fill="x", pady=2)
        for cval in ("Limiting depth", "S/N for source", "Exposure for S/N"):
            ttk.Radiobutton(cf, text=cval, value=cval, variable=self.calc,
                            command=self.compute).pack(anchor="w")

        # -- entry fields --
        self.v = {}
        def field(parent, label, key, default, width=9):
            row = ttk.Frame(parent); row.pack(fill="x", pady=0)
            ttk.Label(row, text=label, width=20, anchor="w").pack(side="left")
            var = tk.StringVar(value=default); self.v[key] = var
            ttk.Entry(row, textvariable=var, width=width).pack(side="left")

        tel = ttk.LabelFrame(panel, text="telescope / detector", padding=4); tel.pack(fill="x", pady=2)
        field(tel, "aperture D [cm]", "diam", "350")
        field(tel, "pixel scale [″]", "pix", "0.11")
        field(tel, "peak throughput η", "eta", "0.30")
        field(tel, "read noise [e-]", "read", "8")
        field(tel, "dark [e-/s/pix]", "dark", "0.010")
        field(tel, "N exposures", "nexp", "3")
        field(tel, "full well [e-]", "fw", "100000")
        self.realistic = tk.BooleanVar(value=True)
        ttk.Checkbutton(tel, text="CALSPEC zodi + component throughput",
                        variable=self.realistic, command=self.compute).pack(anchor="w")

        bg = ttk.LabelFrame(panel, text="background", padding=4); bg.pack(fill="x", pady=2)
        row = ttk.Frame(bg); row.pack(fill="x")
        ttk.Label(row, text="zodi level", width=20, anchor="w").pack(side="left")
        self.zlevel = tk.StringVar(value="typical")
        ttk.OptionMenu(row, self.zlevel, "typical", *etc.ZODI_LEVELS.keys(),
                       command=lambda _=None: self._sync_zodi()).pack(side="left")
        field(bg, "  or μ(0.5µm) [AB]", "zodi", "22.1")
        field(bg, "optics T [K]", "ttel", "270")

        obs = ttk.LabelFrame(panel, text="observation / source", padding=4); obs.pack(fill="x", pady=2)
        field(obs, "λ [µm]", "lam", "1.6")
        field(obs, "filter width [µm]", "fw_um", "0.40")
        field(obs, "resolving power R", "R", "1000")
        field(obs, "source FWHM [″]", "src", "0.3")
        field(obs, "exposure [hr]", "thr", "0.75")
        field(obs, "source AB mag", "mag", "24.0")
        field(obs, "line flux [cgs]", "lflux", "1e-16")
        field(obs, "target S/N", "snr", "10")

        bf = ttk.Frame(panel); bf.pack(fill="x", pady=3)
        for text, cmd in [("Compute", self.compute), ("Plot", self.plot_main),
                          ("Cooling", self.plot_cooling), ("Roman/Euclid", self.validate)]:
            ttk.Button(bf, text=text, command=self._guard(cmd)).pack(side="left", expand=True, fill="x")

        self.out = tk.Text(panel, width=42, height=13, font=("Courier", 9))
        self.out.pack(fill="both", pady=4)

        # -- figure --
        right = ttk.Frame(root); right.grid(row=0, column=1, sticky="nsew")
        self.fig = Figure(figsize=(6.3, 5.4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right)
        root.columnconfigure(1, weight=1); root.rowconfigure(0, weight=1)
        # redraw the embedded figure on resize (fixes black-out / stale repaint over remote X)
        self._resize_job = None
        root.bind("<Configure>", self._on_resize)
        self.refresh_elements()
        self.compute()

    def _on_resize(self, _event=None):
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(120, self._redraw)

    def _redraw(self):
        self._resize_job = None
        try:
            self.canvas.draw_idle()
            self.root.update_idletasks()
        except Exception:
            pass

    # ---- helpers ------------------------------------------------------------
    def _guard(self, fn):
        def w():
            try:
                fn()
            except Exception as e:
                self.out.delete("1.0", "end"); self.out.insert("end", f"Error:\n{e}")
        return w

    def _f(self, k):
        return float(self.v[k].get())

    def _sync_zodi(self):
        self.v["zodi"].set(str(etc.ZODI_LEVELS[self.zlevel.get()]))
        self.compute()

    def apply_preset(self):
        p = etc.TELESCOPE_PRESETS[self.telescope.get()]
        for key in ("diam", "pix", "eta", "read", "dark", "nexp", "fw", "ttel", "R", "lam"):
            self.v[key].set(str(p[key]))
        self._obs = p["obstruction"]; self._band = tuple(p["band"])
        self.realistic.set(p["realistic"])
        self.refresh_elements()
        self.compute()

    def on_mode_change(self):
        self.refresh_elements()
        self.compute()

    def refresh_elements(self):
        els = etc.INSTRUMENT_ELEMENTS.get(self.telescope.get(), {}).get(self.mode.get(), {})
        names = list(els.keys())
        self.elem_cb["values"] = names
        if names and self.element.get() not in names:
            self.element.set(names[0])
            self.apply_element(compute=False)
        elif not names:
            self.element.set("")

    def apply_element(self, compute=True):
        els = etc.INSTRUMENT_ELEMENTS.get(self.telescope.get(), {}).get(self.mode.get(), {})
        el = els.get(self.element.get())
        if el:
            self.v["lam"].set(str(el[0])); self.v["fw_um"].set(str(el[1]))
            if self.mode.get() == "Spectroscopy" and len(el) > 2:
                self.v["R"].set(str(el[2]))
        if compute:
            self.compute()

    def cfg(self):
        kw = dict(diameter_cm=self._f("diam"), obstruction=self._obs, R=self._f("R"),
                  pix_scale=self._f("pix"), eta_peak=self._f("eta"), read_noise=self._f("read"),
                  dark_current=self._f("dark"), n_exp=int(self._f("nexp")),
                  full_well=self._f("fw"), zodi_mu_ref=self._f("zodi"), tel_temp=self._f("ttel"),
                  band_min_A=self._band[0]*1e4, band_max_A=self._band[1]*1e4)
        return etc.realistic_cfg(**kw) if self.realistic.get() else etc.InstrumentConfig(**kw)

    # ---- main compute -------------------------------------------------------
    def compute(self, *_):
        c = self.cfg(); lamA = self._f("lam") * 1e4; fwA = self._f("fw_um") * 1e4
        t = self._f("thr") * HR; src = self._f("src"); o = self.out
        o.delete("1.0", "end")
        det = etc.TELESCOPE_PRESETS.get(self.telescope.get(), {}).get("det", "")
        o.insert("end", f"{self.telescope.get()} [{det}]  {self.mode.get()} / {self.calc.get()}\n")
        o.insert("end", f"D={c.diameter_cm/100:.2f} m  pix={c.pix_scale}″  "
                        f"PSF={c.psf_fwhm(lamA):.3f}″\n")
        if self.mode.get() == "Spectroscopy":
            o.insert("end", f"R={c.R:.0f}  Δλ={c.resolution_element_A(lamA):.1f}Å  "
                            f"disp={c.dispersion_A_per_pix(lamA):.1f}Å/pix\n")
        o.insert("end", "-"*40 + "\n")
        imaging = self.mode.get() == "Imaging"

        if self.calc.get() == "Limiting depth":
            if imaging:
                els = etc.INSTRUMENT_ELEMENTS.get(self.telescope.get(), {}).get("Imaging", {})
                o.insert("end", f"5σ limiting AB in {self._f('thr'):.2f} hr, per filter:\n")
                o.insert("end", f"{'filter':>12} {'λ[µm]':>7} {'5σ AB':>8}\n")
                for name, (lam_um, w_um) in els.items():
                    m = etc.imaging_maglimit(c, lam_um*1e4, w_um*1e4, t)
                    mark = "  <" if name == self.element.get() else ""
                    o.insert("end", f"{name:>12} {lam_um:>7.2f} {m:>8.2f}{mark}\n")
            else:
                o.insert("end", f"{'t[hr]':>7} {'F5σ line [cgs]':>16}\n")
                for tt in sorted({self._f("thr"), 0.75, 3.0, 12.0}):
                    f = etc.f_limit(c, lamA, tt*HR, source_fwhm=src)
                    o.insert("end", f"{tt:>7.2f} {f:>14.2e}\n")
        elif self.calc.get() == "S/N for source":
            if imaging:
                sn = etc.imaging_snr(c, self._f("mag"), lamA, fwA, t)
                o.insert("end", f"AB={self._f('mag'):.1f}, {self._f('thr'):.2f} hr\n  S/N = {sn:.1f}\n")
            else:
                sn = etc.line_sn(c, self._f("lflux"), lamA, t, source_fwhm=src)
                o.insert("end", f"line flux={self._f('lflux'):.2e}, {self._f('thr'):.2f} hr\n  S/N = {sn:.1f}\n")
        else:  # Exposure for S/N
            tgt = self._f("snr")
            if imaging:
                fn = lambda ts: etc.imaging_snr(c, self._f("mag"), lamA, fwA, ts)
            else:
                fn = lambda ts: etc.line_sn(c, self._f("lflux"), lamA, ts, source_fwhm=src)
            texp = etc.exposure_for_snr(fn, tgt)
            o.insert("end", f"time for S/N={tgt:.0f} = {texp/HR:.3f} hr ({texp:.0f} s)\n")

        # count rates + saturation (imaging) / noise note
        o.insert("end", "-"*40 + "\n")
        if imaging:
            cr = etc.count_rates(c, self._f("mag"), lamA, fwA)
            o.insert("end", f"source {cr['source_e_s']:.2f} e-/s (peak {cr['peak_e_s']:.2f})\n"
                            f"sky {cr['sky_e_s_pix']:.2f}  dark {cr['dark_e_s_pix']:.3f} e-/s/pix\n")
            tsingle = t / max(1, int(self._f("nexp")))
            msat = etc.saturation_maglimit(c, lamA, fwA, tsingle)
            warn = "  <-- SATURATED" if self._f("mag") < msat else ""
            o.insert("end", f"saturation bright limit ({tsingle:.0f}s): {msat:.2f} AB{warn}\n")
        else:
            nb = etc.noise_breakdown(c, lamA, t, source_fwhm=src, filter_width_A=fwA)
            tot = nb["sky_e"] + nb["dark_e"] + nb["read_e2"]
            o.insert("end", f"n_pix={nb['n_pix']:.0f}  sky {nb['sky_e']:.0f}  "
                            f"dark {nb['dark_e']:.0f}  read² {nb['read_e2']:.0f} e-\n"
                            f"sky-limited fraction {nb['sky_e']/tot*100:.0f}%\n")

    # ---- plots --------------------------------------------------------------
    def plot_main(self):
        c = self.cfg(); fwA = self._f("fw_um") * 1e4; src = self._f("src")
        imaging = self.mode.get() == "Imaging"; self.ax.clear()
        if self.calc.get() == "Limiting depth":
            if imaging:
                els = etc.INSTRUMENT_ELEMENTS.get(self.telescope.get(), {}).get("Imaging", {})
                lam_um = np.array([v[0] for v in els.values()])
                w_um = np.array([v[1] for v in els.values()])
                for mult, col in [(1, "#3898ec"), (4, "#4ec9b0"), (16, "#d97757")]:
                    tt = self._f("thr")*mult
                    m = [etc.imaging_maglimit(c, l*1e4, w*1e4, tt*HR) for l, w in zip(lam_um, w_um)]
                    self.ax.errorbar(lam_um, m, xerr=w_um/2, fmt="o", color=col,
                                     capsize=2, label=f"{tt:.2f} hr")
                self.ax.set_xlabel("filter λ [µm]"); self.ax.set_ylabel("5σ limiting AB mag")
                self.ax.invert_yaxis()
                self.ax.set_title(f"{self.telescope.get()} imaging depth per filter")
            else:
                lam = np.linspace(c.band_min_A+200, c.band_max_A-200, 220)
                for mult, col in [(1, "#3898ec"), (4, "#4ec9b0"), (16, "#d97757")]:
                    tt = self._f("thr")*mult
                    y = [etc.f_limit(c, l, tt*HR, source_fwhm=src) for l in lam]
                    self.ax.plot(lam/1e4, y, color=col, lw=2, label=f"{tt:.2f} hr")
                self.ax.set_yscale("log"); self.ax.axhline(1e-16, ls=":", color="k", lw=1)
                self.ax.set_xlabel("observed λ [µm]"); self.ax.set_ylabel(r"$F_{5\sigma}$ [erg s$^{-1}$ cm$^{-2}$]")
                self.ax.set_title("Limiting depth vs wavelength")
        else:
            ts = np.logspace(np.log10(60), np.log10(48*HR), 120)
            lamA = self._f("lam")*1e4
            if imaging:
                sn = [etc.imaging_snr(c, self._f("mag"), lamA, fwA, t) for t in ts]
                lab = f"AB={self._f('mag'):.1f}"
            else:
                sn = [etc.line_sn(c, self._f("lflux"), lamA, t, source_fwhm=src) for t in ts]
                lab = f"F={self._f('lflux'):.1e}"
            self.ax.plot(ts/HR, sn, color="#3898ec", lw=2, label=lab)
            self.ax.axhline(self._f("snr"), ls=":", color="k", label=f"S/N={self._f('snr'):.0f}")
            self.ax.set_xscale("log"); self.ax.set_yscale("log")
            self.ax.set_xlabel("exposure [hr]"); self.ax.set_ylabel("S/N")
            self.ax.set_title("S/N vs exposure")
        self.ax.legend(fontsize=8); self.ax.grid(alpha=0.2, which="both")
        self.fig.tight_layout(); self.canvas.draw()

    def plot_cooling(self):
        c0 = self.cfg(); src = self._f("src"); fwA = self._f("fw_um")*1e4
        lam = np.linspace(10000, c0.band_max_A, 180); self.ax.clear()
        import matplotlib.cm as cm
        temps = [150, 180, 210, 240, 270, 290]
        imaging = self.mode.get() == "Imaging"
        for T, col in zip(temps, cm.viridis(np.linspace(0, 0.9, len(temps)))):
            c = replace(c0, tel_temp=T)
            if imaging:
                y = [etc.imaging_maglimit(c, l, fwA, 3*HR) for l in lam]
            else:
                y = [etc.f_limit(c, l, 3*HR, source_fwhm=src) for l in lam]
            self.ax.plot(lam/1e4, y, color=col, lw=2, label=f"{T} K")
        self.ax.set_xlabel("observed λ [µm]")
        if imaging:
            self.ax.set_ylabel("5σ AB (3 hr)"); self.ax.invert_yaxis()
        else:
            self.ax.set_yscale("log"); self.ax.set_ylabel(r"$F_{5\sigma}$ (3 hr)")
        self.ax.set_title("Telescope-temperature tradeoff")
        self.ax.legend(fontsize=7, ncol=2, title="optics T"); self.ax.grid(alpha=0.2, which="both")
        self.fig.tight_layout(); self.canvas.draw()

    def validate(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            etc.compare_missions()
        self.out.delete("1.0", "end"); self.out.insert("end", buf.getvalue())


if __name__ == "__main__":
    root = tk.Tk()
    ETCGui(root)
    root.mainloop()
