"""Two-dimensional slitless scene simulation and multi-roll extraction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr

import slitless_etc as legacy

from .config import SpectralChannel, build_instrument, get_channel
from .spectroscopy import ObservedSpectrum, simulate_spectrum


@dataclass(frozen=True)
class SceneSource:
    """One direct-image source and its observed spectral template."""

    source_id: str
    spectrum: ObservedSpectrum
    x_arcsec: float
    y_arcsec: float
    source_fwhm_arcsec: float = 0.0

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("A scene source requires a non-empty identifier.")
        if self.source_fwhm_arcsec < 0:
            raise ValueError("Source FWHM cannot be negative.")


@dataclass(frozen=True)
class SlitlessExposure:
    """One roll-angle detector realization and its sparse scene operator."""

    roll_angle_deg: float
    image_e: np.ndarray
    expectation_e: np.ndarray
    variance_e2: np.ndarray
    source_expectation_e: np.ndarray
    background_e_per_pixel: float
    operator: sparse.csr_matrix
    image_shape: tuple[int, int]
    trace_start_x: Mapping[str, float]
    trace_y: Mapping[str, float]


@dataclass(frozen=True)
class MultiRollScene:
    """Shared truth and independent detector images for several rolls."""

    wavelength_A: np.ndarray
    source_ids: tuple[str, ...]
    truth_spectra_e: Mapping[str, np.ndarray]
    exposures: tuple[SlitlessExposure, ...]
    direct_prior: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ExtractedScene:
    """Jointly recovered spectra from one or more roll angles."""

    wavelength_A: np.ndarray
    spectra_e: Mapping[str, np.ndarray]
    uncertainty_e: Mapping[str, np.ndarray]
    source_ids: tuple[str, ...]
    roll_angles_deg: tuple[float, ...]
    solver_stop_code: int
    solver_iterations: int
    weighted_residual_norm: float
    metadata: Mapping[str, Any]


def shift_spectrum_velocity(
    spectrum: ObservedSpectrum,
    velocity_kms: float,
    *,
    name: str | None = None,
) -> ObservedSpectrum:
    """Doppler-shift a template on its original wavelength grid."""

    factor = 1.0 + float(velocity_kms) / 299792.458
    if factor <= 0:
        raise ValueError("Velocity gives a non-positive Doppler factor.")
    wavelength = spectrum.wavelength_A
    shifted_flux = np.interp(
        wavelength / factor,
        wavelength,
        spectrum.flux_lambda,
    )
    shifted_uncertainty = None
    if spectrum.uncertainty_flux_lambda is not None:
        shifted_uncertainty = np.interp(
            wavelength / factor,
            wavelength,
            spectrum.uncertainty_flux_lambda,
        )
    metadata = dict(spectrum.metadata or {})
    metadata["injected_velocity_kms"] = float(velocity_kms)
    return ObservedSpectrum(
        wavelength,
        shifted_flux,
        name=name or f"{spectrum.name}, v={velocity_kms:g} km/s",
        uncertainty_flux_lambda=shifted_uncertainty,
        native_resolving_power=spectrum.native_resolving_power,
        metadata=metadata,
    )


def scale_spectrum(
    spectrum: ObservedSpectrum,
    factor: float,
    *,
    name: str | None = None,
) -> ObservedSpectrum:
    """Scale physical flux and its uncertainty by one positive factor."""

    if factor <= 0:
        raise ValueError("Spectrum scale factor must be positive.")
    return ObservedSpectrum(
        spectrum.wavelength_A,
        spectrum.flux_lambda * factor,
        name=name or f"{spectrum.name}, scale={factor:g}",
        uncertainty_flux_lambda=(
            None
            if spectrum.uncertainty_flux_lambda is None
            else spectrum.uncertainty_flux_lambda * factor
        ),
        native_resolving_power=spectrum.native_resolving_power,
        metadata=spectrum.metadata,
    )


def _source_offsets(
    sources: Sequence[SceneSource],
    target_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    matches = [item for item in sources if item.source_id == target_id]
    if len(matches) != 1:
        raise ValueError("target_id must identify exactly one scene source.")
    target = matches[0]
    return (
        np.asarray([item.x_arcsec - target.x_arcsec for item in sources]),
        np.asarray([item.y_arcsec - target.y_arcsec for item in sources]),
    )


def _trace_operator(
    sources: Sequence[SceneSource],
    wavelength_A: np.ndarray,
    cfg: legacy.InstrumentConfig,
    roll_angle_deg: float,
    x_offsets_arcsec: np.ndarray,
    y_offsets_arcsec: np.ndarray,
    image_shape: tuple[int, int],
    trace_origin_x: float,
    trace_origin_y: float,
) -> tuple[sparse.csr_matrix, dict[str, float], dict[str, float]]:
    """Build the sparse mapping from source spectral bins to detector pixels."""

    height, width = image_shape
    n_wave = wavelength_A.size
    angle = np.radians(roll_angle_deg)
    dispersion_offset = (
        x_offsets_arcsec * np.cos(angle)
        + y_offsets_arcsec * np.sin(angle)
    ) / cfg.pix_scale
    cross_offset = (
        -x_offsets_arcsec * np.sin(angle)
        + y_offsets_arcsec * np.cos(angle)
    ) / cfg.pix_scale

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    trace_start_x: dict[str, float] = {}
    trace_y: dict[str, float] = {}
    psf_fwhm = cfg.psf_fwhm(wavelength_A)
    for source_index, source in enumerate(sources):
        x_start = trace_origin_x + dispersion_offset[source_index]
        y_center = trace_origin_y + cross_offset[source_index]
        trace_start_x[source.source_id] = float(x_start)
        trace_y[source.source_id] = float(y_center)
        delivered_fwhm_pix = np.hypot(
            source.source_fwhm_arcsec,
            psf_fwhm,
        ) / cfg.pix_scale
        sigma_y = np.maximum(delivered_fwhm_pix / 2.354820045, 0.30)
        for wave_index in range(n_wave):
            x_position = x_start + wave_index
            x_left = int(np.floor(x_position))
            x_fraction = x_position - x_left
            x_pixels = (x_left, x_left + 1)
            x_weights = (1.0 - x_fraction, x_fraction)
            radius = max(2, int(np.ceil(4.0 * sigma_y[wave_index])))
            y_min = max(0, int(np.floor(y_center)) - radius)
            y_max = min(height, int(np.floor(y_center)) + radius + 1)
            y_pixels = np.arange(y_min, y_max)
            if y_pixels.size == 0:
                continue
            y_weights = np.exp(
                -0.5 * ((y_pixels + 0.5 - y_center) / sigma_y[wave_index]) ** 2
            )
            y_weights /= np.sum(y_weights)
            column = source_index * n_wave + wave_index
            for x_pixel, x_weight in zip(x_pixels, x_weights):
                if x_weight <= 0 or x_pixel < 0 or x_pixel >= width:
                    continue
                rows.extend((y_pixels * width + x_pixel).tolist())
                columns.extend([column] * y_pixels.size)
                values.extend((x_weight * y_weights).tolist())
    operator = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(height * width, len(sources) * n_wave),
    ).tocsr()
    return operator, trace_start_x, trace_y


def _direct_prior_image(
    sources: Sequence[SceneSource],
    source_counts: Mapping[str, np.ndarray],
    cfg: legacy.InstrumentConfig,
    x_offsets_arcsec: np.ndarray,
    y_offsets_arcsec: np.ndarray,
    size: int,
) -> np.ndarray:
    """Render a normalized undispersed image used only as a position prior."""

    image = np.zeros((size, size), dtype=float)
    center = 0.5 * size
    pivot_A = np.sqrt(cfg.band_min_A * cfg.band_max_A)
    for source, dx, dy in zip(sources, x_offsets_arcsec, y_offsets_arcsec):
        x_center = center + dx / cfg.pix_scale
        y_center = center + dy / cfg.pix_scale
        sigma = max(
            0.30,
            np.hypot(source.source_fwhm_arcsec, cfg.psf_fwhm(pivot_A))
            / cfg.pix_scale
            / 2.354820045,
        )
        radius = max(2, int(np.ceil(4.0 * sigma)))
        x_min = max(0, int(np.floor(x_center)) - radius)
        x_max = min(size, int(np.floor(x_center)) + radius + 1)
        y_min = max(0, int(np.floor(y_center)) - radius)
        y_max = min(size, int(np.floor(y_center)) + radius + 1)
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        kernel = np.exp(
            -0.5
            * (
                ((xx + 0.5 - x_center) / sigma) ** 2
                + ((yy + 0.5 - y_center) / sigma) ** 2
            )
        )
        kernel /= np.sum(kernel)
        image[y_min:y_max, x_min:x_max] += (
            np.sum(source_counts[source.source_id]) * kernel
        )
    maximum = np.max(image)
    return image / maximum if maximum > 0 else image


def simulate_multi_roll_scene(
    sources: Sequence[SceneSource],
    channel: SpectralChannel | str,
    exposure_s: float,
    *,
    roll_angles_deg: Sequence[float] = (0.0, 60.0, 120.0),
    target_id: str,
    random_seed: int | None = None,
    detector_margin_pixels: int = 20,
    minimum_cross_dispersion_pixels: int = 64,
    **instrument_overrides: float | int | bool | str | None,
) -> MultiRollScene:
    """Render a shared source scene at several dispersion orientations."""

    if not sources:
        raise ValueError("At least one source is required.")
    source_ids = tuple(item.source_id for item in sources)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Scene source identifiers must be unique.")
    if exposure_s <= 0:
        raise ValueError("Exposure time must be positive.")
    if len(roll_angles_deg) == 0:
        raise ValueError("At least one roll angle is required.")
    selected = get_channel(channel) if isinstance(channel, str) else channel
    scene_overrides = dict(instrument_overrides)
    scene_overrides["extraction_eff"] = 1.0
    scene_overrides["n_exp"] = 1
    cfg = build_instrument(selected, **scene_overrides)
    x_offsets, y_offsets = _source_offsets(sources, target_id)
    radius_pixels = int(
        np.ceil(np.max(np.hypot(x_offsets, y_offsets)) / cfg.pix_scale)
    )

    detector_results = [
        simulate_spectrum(
            source.spectrum,
            selected,
            exposure_s,
            source_fwhm_arcsec=source.source_fwhm_arcsec,
            **scene_overrides,
        )
        for source in sources
    ]
    wavelength = detector_results[0].wavelength_A
    if any(
        result.wavelength_A.shape != wavelength.shape
        or not np.allclose(result.wavelength_A, wavelength)
        for result in detector_results[1:]
    ):
        raise ValueError("All scene sources must share one detector wavelength grid.")
    truth_spectra = {
        source.source_id: result.source_e
        for source, result in zip(sources, detector_results)
    }
    truth_vector = np.concatenate(
        [truth_spectra[source_id] for source_id in source_ids]
    )

    padding = detector_margin_pixels + radius_pixels
    image_width = wavelength.size + 2 * padding + 2
    image_height = max(
        minimum_cross_dispersion_pixels,
        2 * padding + 3,
    )
    image_shape = (image_height, image_width)
    trace_origin_x = float(padding)
    trace_origin_y = 0.5 * image_height

    sky_rate = legacy.background_per_pixel(
        cfg,
        (selected.band_min_A, selected.band_max_A),
    )
    _, dark_rate = cfg.detector_at(selected.pivot_A)
    background_e = float((sky_rate + dark_rate) * exposure_s)
    read_variance = float(
        cfg.read_noise_variance_total(selected.pivot_A, exposure_s, 1.0)
    )
    rng = np.random.default_rng(random_seed)
    exposures: list[SlitlessExposure] = []
    for roll_angle in roll_angles_deg:
        operator, trace_start_x, trace_y = _trace_operator(
            sources,
            wavelength,
            cfg,
            float(roll_angle),
            x_offsets,
            y_offsets,
            image_shape,
            trace_origin_x,
            trace_origin_y,
        )
        source_expectation = np.asarray(operator @ truth_vector).reshape(image_shape)
        expectation = source_expectation + background_e
        variance = (
            cfg.ramp_shot_noise_factor() * expectation
            + read_variance
        )
        image = rng.normal(expectation, np.sqrt(variance))
        exposures.append(
            SlitlessExposure(
                roll_angle_deg=float(roll_angle),
                image_e=image,
                expectation_e=expectation,
                variance_e2=variance,
                source_expectation_e=source_expectation,
                background_e_per_pixel=background_e,
                operator=operator,
                image_shape=image_shape,
                trace_start_x=trace_start_x,
                trace_y=trace_y,
            )
        )
    direct_size = max(minimum_cross_dispersion_pixels, 2 * padding + 3)
    direct_prior = _direct_prior_image(
        sources,
        truth_spectra,
        cfg,
        x_offsets,
        y_offsets,
        direct_size,
    )
    maximum_expected = max(float(np.max(item.expectation_e)) for item in exposures)
    return MultiRollScene(
        wavelength_A=wavelength,
        source_ids=source_ids,
        truth_spectra_e=truth_spectra,
        exposures=tuple(exposures),
        direct_prior=direct_prior,
        metadata={
            "channel": selected.key,
            "exposure_s_per_roll": float(exposure_s),
            "total_exposure_s": float(exposure_s * len(roll_angles_deg)),
            "roll_angles_deg": tuple(float(item) for item in roll_angles_deg),
            "target_id": target_id,
            "image_shape": image_shape,
            "pixel_scale_arcsec": float(cfg.pix_scale),
            "background_e_per_pixel": background_e,
            "read_variance_e2_per_pixel": read_variance,
            "maximum_expected_e_per_pixel": maximum_expected,
            "full_well_e": float(cfg.full_well),
            "saturated": bool(maximum_expected >= cfg.full_well),
            "noise_model": "Gaussian detector realization about forward expectation",
        },
    )


def realize_scene_noise(
    scene: MultiRollScene,
    *,
    random_seed: int | None = None,
) -> MultiRollScene:
    """Draw new independent detector noise without rebuilding the operators."""

    rng = np.random.default_rng(random_seed)
    exposures = tuple(
        replace(
            exposure,
            image_e=rng.normal(
                exposure.expectation_e,
                np.sqrt(exposure.variance_e2),
            ),
        )
        for exposure in scene.exposures
    )
    return replace(scene, exposures=exposures)


def extract_multi_roll_scene(
    scene: MultiRollScene,
    *,
    roll_indices: Sequence[int] | None = None,
    regularization: float = 0.0,
    maximum_iterations: int | None = None,
) -> ExtractedScene:
    """Jointly invert selected roll images for every source spectrum."""

    selected_indices = (
        tuple(range(len(scene.exposures)))
        if roll_indices is None
        else tuple(int(item) for item in roll_indices)
    )
    if not selected_indices:
        raise ValueError("At least one roll exposure must be selected.")
    if min(selected_indices) < 0 or max(selected_indices) >= len(scene.exposures):
        raise IndexError("Roll index is outside the available exposure list.")
    if regularization < 0:
        raise ValueError("Regularization cannot be negative.")

    weighted_operators: list[sparse.csr_matrix] = []
    weighted_data: list[np.ndarray] = []
    for index in selected_indices:
        exposure = scene.exposures[index]
        sigma = np.sqrt(exposure.variance_e2.ravel())
        inverse_sigma = np.divide(
            1.0,
            sigma,
            out=np.zeros_like(sigma),
            where=sigma > 0,
        )
        weighted_operators.append(
            exposure.operator.multiply(inverse_sigma[:, None]).tocsr()
        )
        weighted_data.append(
            (exposure.image_e.ravel() - exposure.background_e_per_pixel)
            * inverse_sigma
        )
    design = sparse.vstack(weighted_operators, format="csr")
    data = np.concatenate(weighted_data)
    n_wave = scene.wavelength_A.size
    if regularization > 0:
        difference = sparse.diags(
            (-np.ones(n_wave - 1), np.ones(n_wave - 1)),
            (0, 1),
            shape=(n_wave - 1, n_wave),
            format="csr",
        )
        smooth = sparse.block_diag(
            [difference] * len(scene.source_ids),
            format="csr",
        )
        design = sparse.vstack(
            (design, np.sqrt(regularization) * smooth),
            format="csr",
        )
        data = np.concatenate((data, np.zeros(smooth.shape[0])))

    solution = lsmr(
        design,
        data,
        atol=1.0e-8,
        btol=1.0e-8,
        maxiter=maximum_iterations,
    )
    recovered = solution[0]
    fisher_diagonal = np.asarray(design.power(2).sum(axis=0)).ravel()
    uncertainty = np.divide(
        1.0,
        np.sqrt(fisher_diagonal),
        out=np.full_like(fisher_diagonal, np.inf),
        where=fisher_diagonal > 0,
    )
    spectra = {
        source_id: recovered[index * n_wave:(index + 1) * n_wave]
        for index, source_id in enumerate(scene.source_ids)
    }
    uncertainties = {
        source_id: uncertainty[index * n_wave:(index + 1) * n_wave]
        for index, source_id in enumerate(scene.source_ids)
    }
    return ExtractedScene(
        wavelength_A=scene.wavelength_A,
        spectra_e=spectra,
        uncertainty_e=uncertainties,
        source_ids=scene.source_ids,
        roll_angles_deg=tuple(
            scene.exposures[index].roll_angle_deg for index in selected_indices
        ),
        solver_stop_code=int(solution[1]),
        solver_iterations=int(solution[2]),
        weighted_residual_norm=float(solution[3]),
        metadata={
            "regularization": float(regularization),
            "uncertainty_status": (
                "diagonal Fisher approximation; use injection recovery for covariance"
            ),
        },
    )
