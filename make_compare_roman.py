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
