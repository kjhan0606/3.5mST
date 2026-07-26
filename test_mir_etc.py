import numpy as np

import slitless_etc as etc


def test_existing_optical_imaging_result_is_unchanged():
    cfg = etc.realistic_cfg()
    assert np.isclose(
        etc.imaging_maglimit(cfg, 6166.0, 1111.0, 2700.0),
        28.22610296830107,
        rtol=1e-12,
    )


def test_mir_flux_limit_inverts_snr():
    for channel in ("NC1", "NC2"):
        cfg = etc.mir_imaging_cfg(channel, "nominal")
        band = tuple(x * 1e4 for x in etc.MIR_CHANNELS[channel]["band_um"])
        limit = etc.imaging_flux_limit_jy(cfg, band, 145.0, snr=5.0)
        assert np.isclose(
            etc.imaging_snr_jy(cfg, limit, band, 145.0),
            5.0,
            rtol=1e-10,
        )


def test_mir_limit_improves_with_exposure_and_aperture():
    band = (6.0e4, 10.0e4)
    cfg = etc.mir_imaging_cfg("NC2", "nominal")
    short = etc.imaging_flux_limit_jy(cfg, band, 145.0)
    long = etc.imaging_flux_limit_jy(cfg, band, 4.0 * 145.0)
    neo_surveyor = etc.mir_imaging_cfg(
        "NC2", "nominal", observatory="NEO Surveyor")
    small_aperture = etc.imaging_flux_limit_jy(
        neo_surveyor, band, 145.0)
    assert long < short
    assert short < small_aperture


def test_published_background_levels_degrade_sensitivity():
    for channel in ("NC1", "NC2"):
        band = tuple(x * 1e4 for x in etc.MIR_CHANNELS[channel]["band_um"])
        limits = [
            etc.imaging_flux_limit_jy(
                etc.mir_imaging_cfg(
                    channel, level, observatory="NEO Surveyor"),
                band,
                145.0,
            )
            for level in ("low", "nominal", "high")
        ]
        assert limits[0] < limits[1] < limits[2]


def test_neo_surveyor_validation_preset_matches_published_nesi_midpoint():
    published_ranges_ujy = {"NC1": (65.0, 120.0), "NC2": (110.0, 280.0)}
    for channel, published in published_ranges_ujy.items():
        band = tuple(x * 1e4 for x in etc.MIR_CHANNELS[channel]["band_um"])
        cfg = etc.mir_imaging_cfg(
            channel, "nominal", observatory="NEO Surveyor")
        model_ujy = etc.imaging_flux_limit_jy(
            cfg, band, 145.0) * 1e6
        published_midpoint = np.sqrt(published[0] * published[1])
        assert np.isclose(model_ujy, published_midpoint, rtol=0.08)
