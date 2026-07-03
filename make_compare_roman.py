#!/usr/bin/env python3
"""3.5m ST vs Roman WFI: imaging depth per filter (table) and spectroscopic
5-sigma line-flux depth vs wavelength (figure)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import slitless_etc as etc

HR = 3600.0


def cfg_from_preset(name):
    p = etc.TELESCOPE_PRESETS[name]
    kw = dict(diameter_cm=p["diam"], obstruction=p["obstruction"], pix_scale=p["pix"],
              eta_peak=p["eta"], read_noise=p["read"], dark_current=p["dark"],
              n_exp=p["nexp"], full_well=p["fw"], tel_temp=p["ttel"], R=p["R"],
              dichroic_split_A=p.get("split", 0.0),
              band_min_A=p["band"][0]*1e4, band_max_A=p["band"][1]*1e4)
    return etc.realistic_cfg(**kw) if p["realistic"] else etc.InstrumentConfig(**kw)


c35 = cfg_from_preset("3.5mST")
crm = cfg_from_preset("Roman WFI")

# ---- imaging depth per Roman filter, 1 hr point source ----
t_img = 1.0 * HR
print("Imaging 5sigma AB (1 hr, point source), Roman filters:")
print(f"{'filter':>12} {'lam[um]':>7} {'w[um]':>6} {'3.5mST':>8} {'Roman':>8} {'gain':>6}")
rows = []
for name, (lam_um, w_um) in etc.INSTRUMENT_ELEMENTS["Roman WFI"]["Imaging"].items():
    m35 = etc.imaging_maglimit(c35, lam_um*1e4, w_um*1e4, t_img)
    mrm = etc.imaging_maglimit(crm, lam_um*1e4, w_um*1e4, t_img)
    rows.append((name, lam_um, w_um, m35, mrm, m35-mrm))
    print(f"{name:>12} {lam_um:>7.2f} {w_um:>6.2f} {m35:>8.2f} {mrm:>8.2f} {m35-mrm:>+6.2f}")

# emit a LaTeX tabular body for the proposal
with open("compare_imaging_rows.tex", "w") as f:
    for name, lam_um, w_um, m35, mrm, dg in rows:
        f.write(f"{name} & {lam_um:.2f} & {w_um:.2f} & {m35:.2f} & {mrm:.2f} & ${dg:+.2f}$\\\\\n")
print("wrote compare_imaging_rows.tex")

# ---- spectroscopic 5sigma line flux vs wavelength, 0.75 hr ----
t_spec = 0.75 * HR
fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
# 3.5mST: band-limiting filter ~4000 A across the full 0.36-3.0 um band
lam35 = np.linspace(4000, 29000, 260)
f35 = [etc.f_limit(c35, l, t_spec, source_fwhm=0.3, filter_width_A=4000.) for l in lam35]
ax.plot(lam35/1e4, f35, color="#3898ec", lw=2.2, label="3.5 m ST (R=1000, 0.4 µm filters)")
# Roman: grism disperses the whole 1.0-1.93 um band onto each pixel
lamrm = np.linspace(10000, 19300, 140)
bandrm = 19300 - 10000
frm = [etc.f_limit(crm, l, t_spec, source_fwhm=0.3, filter_width_A=bandrm) for l in lamrm]
ax.plot(lamrm/1e4, frm, color="#d97757", lw=2.2, label="Roman WFI grism (1.0–1.93 µm)")
ax.axhline(1e-16, ls=":", color="k", lw=1, label="Roman HLSS depth ($10^{-16}$)")
ax.set_yscale("log"); ax.set_xlabel("observed wavelength [µm]")
ax.set_ylabel(r"$F_{5\sigma}$ [erg s$^{-1}$ cm$^{-2}$]  (0.75 hr)")
ax.set_title("Slitless 5σ line-flux depth: 3.5 m ST vs Roman")
ax.legend(fontsize=8); ax.grid(alpha=0.2, which="both")
fig.savefig("etc_compare_spec.png", bbox_inches="tight")
print("wrote etc_compare_spec.png")

# ============================ multi-telescope comparison ============================
COLORS = {"3.5mST": "#3898ec", "Roman WFI": "#d97757", "Euclid": "#4ec9b0",
          "JWST": "#7d3c98"}
cfgs = {name: cfg_from_preset(name) for name in
        ("3.5mST", "Roman WFI", "Euclid", "JWST NIRCam SW", "JWST NIRCam LW")}


def in_band(name, lam_um, margin=0.06):
    b = etc.TELESCOPE_PRESETS[name]["band"]
    return b[0] + margin <= lam_um <= b[1] - margin


def jwst_cfg(lam_um):
    return cfgs["JWST NIRCam SW"] if lam_um <= 2.35 else cfgs["JWST NIRCam LW"]


# ---- table: imaging 5sigma AB at common wavelengths (0.3 um band),
# ----        for survey-relevant exposures of 1, 5, 10, 24, 48 hr ----
wl = [0.6, 1.0, 1.5, 2.0, 2.5, 3.5]
cols = ["3.5mST", "Roman WFI", "Euclid", "JWST"]
EXPS_HR = (1.0, 5.0, 10.0, 24.0, 48.0)
lines = []
for t_hr in EXPS_HR:
    t_img_e = t_hr * HR
    print(f"\nMulti-telescope imaging 5sigma AB ({t_hr:g} hr, 0.3um band):")
    print(f"{'lam[um]':>7} " + " ".join(f"{c:>10}" for c in cols))
    lines.append("\\addlinespace[2pt]\n\\multicolumn{5}{@{}l}{\\textbf{"
                 f"{t_hr:g}\\,hr" + "}}\\\\[1pt]\n")
    for lam in wl:
        cells = []
        for c in cols:
            if c == "JWST":
                ok = in_band("JWST NIRCam SW", lam) or in_band("JWST NIRCam LW", lam)
                cc = jwst_cfg(lam)
            else:
                ok = in_band(c, lam); cc = cfgs[c]
            cells.append(f"{etc.imaging_maglimit(cc, lam*1e4, 3000., t_img_e):.2f}" if ok else "--")
        print(f"{lam:>7.1f} " + " ".join(f"{x:>10}" for x in cells))
        lines.append(f"{lam:.1f} & " + " & ".join(cells) + "\\\\\n")
with open("compare_multi_rows.tex", "w") as f:
    f.writelines(lines)
print("wrote compare_multi_rows.tex")

# ---- figure: native imaging filters of each telescope, overlaid (1 hr) ----
fig2, ax2 = plt.subplots(figsize=(7.4, 5.0), dpi=150)
for tel, mark in [("3.5mST", "o"), ("Roman WFI", "s"), ("Euclid", "^"),
                  ("JWST NIRCam SW", "D"), ("JWST NIRCam LW", "D")]:
    els = etc.INSTRUMENT_ELEMENTS[tel]["Imaging"]
    lam = np.array([v[0] for v in els.values()])
    w = np.array([v[1] for v in els.values()])
    m = [etc.imaging_maglimit(cfgs[tel], l*1e4, ww*1e4, t_img) for l, ww in zip(lam, w)]
    key = "JWST" if tel.startswith("JWST") else tel
    lab = "JWST NIRCam" if tel == "JWST NIRCam SW" else (None if tel == "JWST NIRCam LW" else tel)
    ax2.errorbar(lam, m, xerr=w/2, fmt=mark, color=COLORS[key], capsize=2,
                 ms=6, label=lab)
ax2.invert_yaxis()
ax2.set_xlabel("filter central wavelength [µm]")
ax2.set_ylabel("imaging 5σ limiting AB mag  (1 hr, point source)")
ax2.set_title("Imaging depth per filter: 3.5 m ST vs Roman, Euclid, JWST")
ax2.legend(fontsize=8); ax2.grid(alpha=0.2)
fig2.savefig("etc_compare_imaging.png", bbox_inches="tight")
print("wrote etc_compare_imaging.png")
