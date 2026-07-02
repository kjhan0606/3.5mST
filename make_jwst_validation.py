#!/usr/bin/env python3
"""Validate the ETC against documented JWST sensitivities, in both imaging and
spectroscopy modes, for several targets. Reference values from JWST JDox.

Imaging point-source references (NIRCam):
  F200W : AB 29.0 at S/N=10 in 10 ks               (JDox NIRCam imaging)
  F444W : 24.5 nJy (AB 27.9) at S/N=10 in 10 ks    (JDox NIRCam sensitivity)
Spectroscopy point-source reference (NIRSpec):
  R=1000 grating at 2 um : F_line = 5.72e-19 erg/s/cm2 at S/N=10 in 100 ks
"""
import numpy as np
import slitless_etc as etc

KS = 1000.0
nJy_to_AB = lambda f: -2.5*np.log10(f*1e-9/3631.0)


def jwst_img(channel):
    p = etc.TELESCOPE_PRESETS[channel]
    return etc.InstrumentConfig(diameter_cm=p["diam"], obstruction=p["obstruction"],
        pix_scale=p["pix"], eta_peak=p["eta"], read_noise=p["read"], dark_current=p["dark"],
        n_exp=p["nexp"], full_well=p["fw"], tel_temp=p["ttel"], R=p["R"],
        dichroic_split_A=p.get("split", 0.0),
        band_min_A=p["band"][0]*1e4, band_max_A=p["band"][1]*1e4)


# NIRSpec: 6.5 m, cold optics, R=1000, point source; slit throughput ~0.4
jwst_spec = etc.InstrumentConfig(diameter_cm=650.0, obstruction=0.485, pix_scale=0.10,
    eta_peak=0.40, read_noise=6.0, dark_current=0.002, n_exp=2, tel_temp=45.0, R=1000.0,
    dichroic_split_A=0.0, band_min_A=6000.0, band_max_A=53000.0, extraction_eff=1.0)

rows = []
# --- imaging targets ---
snF200 = etc.imaging_snr(jwst_img("JWST NIRCam SW"), 29.0, 2.00e4, 0.46e4, 10*KS)
rows.append(("NIRCam F200W img", "AB 29.0 pt", "10 ks", "S/N 10", f"S/N {snF200:.1f}"))
abF444 = nJy_to_AB(24.5)
snF444 = etc.imaging_snr(jwst_img("JWST NIRCam LW"), abF444, 4.42e4, 1.02e4, 10*KS)
rows.append(("NIRCam F444W img", f"AB {abF444:.1f} pt", "10 ks", "S/N 10", f"S/N {snF444:.1f}"))
# --- spectroscopy target ---
snNS = etc.line_sn(jwst_spec, 5.72e-19, 2.00e4, 100*KS, source_fwhm=0.10)
rows.append(("NIRSpec R=1000 line", "5.7e-19 @2um", "100 ks", "S/N 10", f"S/N {snNS:.1f}"))

print(f"{'target/mode':22} {'source':14} {'t':8} {'JWST ref':10} {'this ETC':10}")
for r in rows:
    print(f"{r[0]:22} {r[1]:14} {r[2]:8} {r[3]:10} {r[4]:10}")

# emit LaTeX rows
with open("jwst_validation_rows.tex", "w") as f:
    tex = [("NIRCam F200W (imaging)", "AB\\,29.0 point", "10\\,ks", "10", f"{snF200:.1f}"),
           ("NIRCam F444W (imaging)", f"AB\\,{abF444:.1f} point", "10\\,ks", "10", f"{snF444:.1f}"),
           ("NIRSpec R=1000 (line)", "$5.7\\times10^{-19}$ @\\,2\\,$\\mu$m", "100\\,ks", "10", f"{snNS:.1f}")]
    for t in tex:
        f.write(" & ".join(t) + "\\\\\n")
print("wrote jwst_validation_rows.tex")
