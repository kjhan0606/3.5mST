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
        self.mode = tk.StringVar(value="Both")
        self.calc = tk.StringVar(value="Limiting depth")
        mf = ttk.LabelFrame(panel, text="mode", padding=4); mf.pack(fill="x")
        for m in ("Spectroscopy", "Imaging", "Both"):
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
        field(tel, "IPC coupling α", "ipc", "0.0")
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
        self.stray_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(bg, text="bright field star stray light",
                        variable=self.stray_on, command=self.compute).pack(anchor="w")
        field(bg, "  star mag [AB]", "straymag", "10")
        field(bg, "  separation [″]", "straysep", "300")

        obs = ttk.LabelFrame(panel, text="observation / source", padding=4); obs.pack(fill="x", pady=2)
        frow = ttk.Frame(obs); frow.pack(fill="x", pady=0)
        ttk.Label(frow, text="imaging filter", width=20, anchor="w").pack(side="left")
        self.imaging_filter = tk.StringVar(value="SDSS g")
        self.filter_cb = ttk.Combobox(frow, textvariable=self.imaging_filter,
                                      state="readonly", width=17)
        self.filter_cb.pack(side="left")
        self.filter_cb.bind("<<ComboboxSelected>>", lambda _e: self.compute())
        field(obs, "spectral λ [µm]", "lam", "1.6")
        field(obs, "spectral band [µm]", "fw_um", "0.40")
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
        self.plot_main()          # show a plot in the default window

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

    def _modes(self):
        """(imaging_on, spectroscopy_on) from the mode selector, both true for Both."""
        m = self.mode.get()
        return (m in ("Imaging", "Both"), m in ("Spectroscopy", "Both"))

    def on_mode_change(self):
        self.refresh_elements()
        self.compute()
        self.plot_main()

    def refresh_elements(self):
        _, spec = self._modes()
        filters = etc.INSTRUMENT_ELEMENTS.get(self.telescope.get(), {}).get("Imaging", {})
        filter_names = list(filters.keys())
        self.filter_cb["values"] = filter_names
        if filter_names and self.imaging_filter.get() not in filter_names:
            self.imaging_filter.set(filter_names[0])

        self._elem_mode = "Spectroscopy"
        els = etc.INSTRUMENT_ELEMENTS.get(self.telescope.get(), {}).get("Spectroscopy", {}) if spec else {}
        names = list(els.keys())
        self.elem_cb["values"] = names
        if names and self.element.get() not in names:
            self.element.set(names[0])
            self.apply_element(compute=False)
        elif not names:
            self.element.set("")

    def apply_element(self, compute=True):
        pm = getattr(self, "_elem_mode", "Spectroscopy")
        els = etc.INSTRUMENT_ELEMENTS.get(self.telescope.get(), {}).get(pm, {})
        el = els.get(self.element.get())
        if el:
            self.v["lam"].set(str(el[0])); self.v["fw_um"].set(str(el[1]))
            if pm == "Spectroscopy" and len(el) > 2:
                self.v["R"].set(str(el[2]))
        if compute:
            self.compute()

    def selected_filter(self):
        """Return selected imaging filter as (name, pivot_um, width_um)."""
        els = etc.INSTRUMENT_ELEMENTS.get(self.telescope.get(), {}).get("Imaging", {})
        name = self.imaging_filter.get()
        if name not in els:
            raise ValueError(f"No imaging filter is available for {self.telescope.get()}")
        lam_um, width_um = els[name]
        return name, float(lam_um), float(width_um)

    def cfg(self):
        kw = dict(diameter_cm=self._f("diam"), obstruction=self._obs, R=self._f("R"),
                  pix_scale=self._f("pix"), eta_peak=self._f("eta"), read_noise=self._f("read"),
                  dark_current=self._f("dark"), n_exp=int(self._f("nexp")),
                  full_well=self._f("fw"), zodi_mu_ref=self._f("zodi"), tel_temp=self._f("ttel"),
                  dichroic_split_A=etc.TELESCOPE_PRESETS.get(self.telescope.get(), {}).get("split", 0.0),
                  band_min_A=self._band[0]*1e4, band_max_A=self._band[1]*1e4,
                  stray_star_mag=self._f("straymag") if self.stray_on.get() else None,
                  stray_star_sep_arcsec=self._f("straysep"),
                  ipc_alpha=self._f("ipc"))
        return etc.realistic_cfg(**kw) if self.realistic.get() else etc.InstrumentConfig(**kw)

    # ---- main compute -------------------------------------------------------
    def compute(self, *_):
        c = self.cfg(); lamA = self._f("lam") * 1e4; fwA = self._f("fw_um") * 1e4
        filter_name, filter_lam_um, filter_width_um = self.selected_filter()
        img_lamA, img_fwA = filter_lam_um * 1e4, filter_width_um * 1e4
        t = self._f("thr") * HR; src = self._f("src"); o = self.out
        o.delete("1.0", "end")
        img, spec = self._modes()
        det = etc.TELESCOPE_PRESETS.get(self.telescope.get(), {}).get("det", "")
        active = "+".join([m for m, on in [("Imaging", img), ("Spectroscopy", spec)] if on])
        o.insert("end", f"{self.telescope.get()} [{det}]  {active} / {self.calc.get()}\n")
        o.insert("end", f"D={c.diameter_cm/100:.2f} m  pix={c.pix_scale}″  "
                        f"PSF={c.psf_fwhm(img_lamA if img else lamA):.3f}″\n")
        if spec:
            o.insert("end", f"R={c.R:.0f}  Δλ={c.resolution_element_A(lamA):.1f}Å  "
                            f"disp={c.dispersion_A_per_pix(lamA):.1f}Å/pix\n")
        if img:
            o.insert("end", "="*40 + "\nIMAGING\n")
            o.insert("end", f"filter={filter_name}  pivot={filter_lam_um:.4f} µm  "
                            f"rect.width={filter_width_um:.4f} µm\n")
            self._out_imaging(o, c, img_lamA, img_fwA, t)
        if spec:
            o.insert("end", "="*40 + "\nSPECTROSCOPY\n")
            self._out_spectrum(o, c, lamA, t, src, fwA)

    def _out_imaging(self, o, c, lamA, fwA, t):
        calc = self.calc.get()
        if calc == "Limiting depth":
            els = etc.INSTRUMENT_ELEMENTS.get(self.telescope.get(), {}).get("Imaging", {})
            o.insert("end", f"5σ AB in {self._f('thr'):.2f} hr, per filter:\n")
            o.insert("end", f"{'filter':>12} {'λ[µm]':>7} {'5σ AB':>8}\n")
            for name, (lam_um, w_um) in els.items():
                m = etc.imaging_maglimit(c, lam_um*1e4, w_um*1e4, t)
                o.insert("end", f"{name:>12} {lam_um:>7.2f} {m:>8.2f}\n")
        elif calc == "S/N for source":
            sn = etc.imaging_snr(c, self._f("mag"), lamA, fwA, t)
            o.insert("end", f"AB={self._f('mag'):.1f}, {self._f('thr'):.2f} hr -> S/N = {sn:.1f}\n")
        else:
            fn = lambda ts: etc.imaging_snr(c, self._f("mag"), lamA, fwA, ts)
            texp = etc.exposure_for_snr(fn, self._f("snr"))
            o.insert("end", f"AB={self._f('mag'):.1f}: S/N={self._f('snr'):.0f} in {texp/HR:.3f} hr\n")
        tsingle = t / max(1, int(self._f("nexp")))
        msat = etc.saturation_maglimit(c, lamA, fwA, tsingle)
        warn = "  <-- SATURATED" if self._f("mag") < msat else ""
        cr = etc.count_rates(c, self._f("mag"), lamA, fwA)
        o.insert("end", f"src {cr['source_e_s']:.2f} e-/s  sat<{msat:.2f} AB ({tsingle:.0f}s){warn}\n")

    def _out_spectrum(self, o, c, lamA, t, src, fwA):
        calc = self.calc.get()
        if calc == "Limiting depth":
            o.insert("end", f"{'t[hr]':>7} {'F5σ line [cgs]':>16}\n")
            for tt in sorted({self._f("thr"), 0.75, 3.0, 12.0}):
                f = etc.f_limit(c, lamA, tt*HR, source_fwhm=src)
                o.insert("end", f"{tt:>7.2f} {f:>14.2e}\n")
        elif calc == "S/N for source":
            sn = etc.line_sn(c, self._f("lflux"), lamA, t, source_fwhm=src)
            o.insert("end", f"F={self._f('lflux'):.2e}, {self._f('thr'):.2f} hr -> S/N = {sn:.1f}\n")
        else:
            fn = lambda ts: etc.line_sn(c, self._f("lflux"), lamA, ts, source_fwhm=src)
            texp = etc.exposure_for_snr(fn, self._f("snr"))
            o.insert("end", f"F={self._f('lflux'):.2e}: S/N={self._f('snr'):.0f} in {texp/HR:.3f} hr\n")
        nb = etc.noise_breakdown(c, lamA, t, source_fwhm=src, filter_width_A=fwA)
        tot = nb["sky_e"] + nb["dark_e"] + nb["read_e2"]
        o.insert("end", f"n_pix={nb['n_pix']:.0f}  sky-limited {nb['sky_e']/tot*100:.0f}%\n")

    # ---- plots --------------------------------------------------------------
    def plot_main(self):
        img, spec = self._modes()
        self.fig.clear()
        if img and spec:
            self._plot_imaging(self.fig.add_subplot(211))
            self._plot_spectrum(self.fig.add_subplot(212))
        elif img:
            self._plot_imaging(self.fig.add_subplot(111))
        else:
            self._plot_spectrum(self.fig.add_subplot(111))
        self.fig.tight_layout(); self.canvas.draw()

    def _plot_imaging(self, ax):
        c = self.cfg()
        filter_name, filter_lam_um, filter_width_um = self.selected_filter()
        fwA = filter_width_um * 1e4
        if self.calc.get() == "Limiting depth":
            els = etc.INSTRUMENT_ELEMENTS.get(self.telescope.get(), {}).get("Imaging", {})
            names = list(els)
            x = np.arange(len(names), dtype=float)
            lam_um = np.array([els[name][0] for name in names])
            w_um = np.array([els[name][1] for name in names])
            offsets = (-0.18, 0.0, 0.18)
            for (mult, col), dx in zip([(1, "#3898ec"), (4, "#4ec9b0"),
                                        (16, "#d97757")], offsets):
                tt = self._f("thr")*mult
                m = [etc.imaging_maglimit(c, l*1e4, w*1e4, tt*HR) for l, w in zip(lam_um, w_um)]
                ax.plot(x + dx, m, "o-", color=col, ms=4, lw=1.2,
                        label=f"{tt:.2f} hr")
            # The bands separate filter systems.  The calculation uses each
            # filter's pivot wavelength and rectangular-equivalent bandwidth.
            if names == list(etc.STANDARD_FILTERS):
                for lo, hi, colour in [(0, 4, "#f3ead8"), (5, 9, "#e4f0f8"),
                                        (10, 12, "#ebe6f4")]:
                    ax.axvspan(lo - 0.48, hi + 0.48, color=colour, alpha=0.65, zorder=0)
                for xpos, label in [(2, "Johnson-Cousins"), (7, "SDSS"), (11, "2MASS")]:
                    ax.text(xpos, 1.02, label, ha="center", va="bottom", fontsize=7,
                            transform=ax.get_xaxis_transform())
            short_names = [name.split()[-1] for name in names]
            ax.set_xticks(x, short_names, fontsize=7)
            ax.set_xlim(-0.55, len(names) - 0.45)
            ax.set_xlabel("standard photometric filter")
            ax.set_ylabel(r"$5\sigma$ point-source depth [AB mag]")
            ax.invert_yaxis()
            ax.set_title(f"{self.telescope.get()} imaging depth by passband")
        else:
            ts = np.logspace(np.log10(60), np.log10(48*HR), 120); lamA = filter_lam_um*1e4
            sn = [etc.imaging_snr(c, self._f("mag"), lamA, fwA, t) for t in ts]
            ax.plot(ts/HR, sn, color="#3898ec", lw=2,
                    label=f"{filter_name}, AB={self._f('mag'):.1f}")
            ax.axhline(self._f("snr"), ls=":", color="k", label=f"S/N={self._f('snr'):.0f}")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("exposure [hr]"); ax.set_ylabel("imaging S/N"); ax.set_title("Imaging S/N vs exposure")
        ax.legend(fontsize=8); ax.grid(alpha=0.2, which="both")

    def _plot_spectrum(self, ax):
        c = self.cfg(); src = self._f("src"); fwA = self._f("fw_um") * 1e4
        if self.calc.get() == "Limiting depth":
            lam = np.linspace(c.band_min_A+200, c.band_max_A-200, 220)
            for mult, col in [(1, "#3898ec"), (4, "#4ec9b0"), (16, "#d97757")]:
                tt = self._f("thr")*mult
                y = [etc.f_limit(c, l, tt*HR, source_fwhm=src) for l in lam]
                ax.plot(lam/1e4, y, color=col, lw=2, label=f"{tt:.2f} hr")
            ax.set_yscale("log"); ax.axhline(1e-16, ls=":", color="k", lw=1)
            ax.set_xlabel("observed λ [µm]"); ax.set_ylabel(r"$F_{5\sigma}$ [cgs]")
            ax.set_title("Slitless depth vs wavelength")
        else:
            ts = np.logspace(np.log10(60), np.log10(48*HR), 120); lamA = self._f("lam")*1e4
            sn = [etc.line_sn(c, self._f("lflux"), lamA, t, source_fwhm=src) for t in ts]
            ax.plot(ts/HR, sn, color="#d97757", lw=2, label=f"F={self._f('lflux'):.1e}")
            ax.axhline(self._f("snr"), ls=":", color="k", label=f"S/N={self._f('snr'):.0f}")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("exposure [hr]"); ax.set_ylabel("line S/N"); ax.set_title("Line S/N vs exposure")
        ax.legend(fontsize=8); ax.grid(alpha=0.2, which="both")

    def plot_cooling(self):
        c0 = self.cfg(); src = self._f("src"); fwA = self._f("fw_um")*1e4
        lam = np.linspace(10000, c0.band_max_A, 180)
        self.fig.clear(); ax = self.fig.add_subplot(111)
        import matplotlib.cm as cm
        temps = [150, 180, 210, 240, 270, 290]
        img, spec = self._modes()
        imaging = img and not spec        # spectroscopy depth is the default cooling view
        for T, col in zip(temps, cm.viridis(np.linspace(0, 0.9, len(temps)))):
            c = replace(c0, tel_temp=T)
            if imaging:
                y = [etc.imaging_maglimit(c, l, fwA, 3*HR) for l in lam]
            else:
                y = [etc.f_limit(c, l, 3*HR, source_fwhm=src) for l in lam]
            ax.plot(lam/1e4, y, color=col, lw=2, label=f"{T} K")
        ax.set_xlabel("observed λ [µm]")
        if imaging:
            ax.set_ylabel("5σ AB (3 hr)"); ax.invert_yaxis()
        else:
            ax.set_yscale("log"); ax.set_ylabel(r"$F_{5\sigma}$ (3 hr)")
        ax.set_title("Telescope-temperature tradeoff")
        ax.legend(fontsize=7, ncol=2, title="optics T"); ax.grid(alpha=0.2, which="both")
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
