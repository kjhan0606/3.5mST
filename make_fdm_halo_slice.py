#!/usr/bin/env python3
"""Extract a halo-centred density slice from a cuRAMSES FDM AMR output.

The script reads the RAMSES AMR geometry and complex wavefunction directly.
It finds the densest leaf cell, then reconstructs the plane through that cell
at the native AMR resolution. Density on wave levels is |psi|^2.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import struct

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path("/gpfs/kjhan/Hydro/FDM/run_3way_wave/output_00170")
FIGURE = ROOT / "fdm_halo_density_slice.png"
OCTANT_OFFSET = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
     [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
    dtype=float,
)
# Successive factors of four from the complete periodic box to the halo centre.
PANELS = ((1.0, 1024), (0.25, 1024), (0.0625, 256), (0.015625, 256))
PEAK_LEVEL = 11


def read_record(handle) -> bytes:
    marker = handle.read(4)
    if len(marker) != 4:
        raise EOFError("unexpected end of Fortran record stream")
    length = struct.unpack("i", marker)[0]
    payload = handle.read(length)
    closing = struct.unpack("i", handle.read(4))[0]
    if closing != length:
        raise RuntimeError("Fortran record markers do not match")
    return payload


def skip_record(handle) -> None:
    length = struct.unpack("i", handle.read(4))[0]
    handle.seek(length, 1)
    closing = struct.unpack("i", handle.read(4))[0]
    if closing != length:
        raise RuntimeError("Fortran record markers do not match")


def read_int(handle) -> int:
    return struct.unpack("i", read_record(handle))[0]


def read_int_array(handle) -> np.ndarray:
    return np.frombuffer(read_record(handle), dtype=np.int32).copy()


def read_float_array(handle) -> np.ndarray:
    return np.frombuffer(read_record(handle), dtype=np.float64).copy()


def read_owned_geometry(amr_path: Path, cpu: int):
    geometry = {}
    with amr_path.open("rb") as handle:
        ncpu = read_int(handle)
        read_int(handle)
        skip_record(handle)
        nlevel = read_int(handle)
        skip_record(handle)
        nboundary = read_int(handle)
        skip_record(handle)
        skip_record(handle)
        for _ in range(11):
            skip_record(handle)
        skip_record(handle)
        skip_record(handle)
        blocks = read_int_array(handle).reshape(ncpu, nlevel, order="F")
        skip_record(handle)
        skip_record(handle)
        ordering = read_record(handle).decode().strip().lower()
        if "hilbert" in ordering:
            skip_record(handle)
        elif "ksection" in ordering:
            for _ in range(10):
                skip_record(handle)
        elif "bisection" in ordering:
            for _ in range(5):
                skip_record(handle)
        skip_record(handle)
        skip_record(handle)
        skip_record(handle)

        for level in range(1, nlevel + 1):
            for domain in range(1, nboundary + ncpu + 1):
                count = int(blocks[domain - 1, level - 1]) if domain <= ncpu else 0
                if count <= 0:
                    continue
                skip_record(handle)
                skip_record(handle)
                skip_record(handle)
                centers = [read_float_array(handle)[:count] for _ in range(3)]
                skip_record(handle)
                for _ in range(6):
                    skip_record(handle)
                children = [read_int_array(handle)[:count] for _ in range(8)]
                for _ in range(8):
                    skip_record(handle)
                for _ in range(8):
                    skip_record(handle)
                if domain == cpu:
                    geometry[(level, domain)] = (centers, children)
    return geometry, ncpu


def iter_owned_wave_cells(output: Path, number: int, cpu: int):
    amr_path = output / f"amr_{number:05d}.out{cpu:05d}"
    fdm_path = output / f"fdm_{number:05d}.out{cpu:05d}"
    geometry, ncpu = read_owned_geometry(amr_path, cpu)

    with fdm_path.open("rb") as handle:
        read_int(handle)
        read_int(handle)
        nlevel = read_int(handle)
        nboundary = read_int(handle)
        for level in range(1, nlevel + 1):
            for domain in range(1, nboundary + ncpu + 1):
                read_int(handle)
                count = read_int(handle)
                if count <= 0:
                    continue
                real = []
                imag = []
                for _ in range(8):
                    real.append(read_float_array(handle)[:count])
                    imag.append(read_float_array(handle)[:count])
                key = (level, domain)
                if key not in geometry:
                    continue
                centers, children = geometry[key]
                cell_size = 0.5**level
                for octant in range(8):
                    leaf = children[octant] == 0
                    if not np.any(leaf):
                        continue
                    density = (real[octant] ** 2 + imag[octant] ** 2)[leaf]
                    xyz = tuple(
                        (centers[axis] + (OCTANT_OFFSET[octant, axis] - 0.5) * cell_size)[leaf] % 1.0
                        for axis in range(3)
                    )
                    yield level, cell_size, density, xyz


@dataclass
class PeakResult:
    density: float
    x: float
    y: float
    z: float
    level: int
    mass: float
    volume: float
    cells: int


def find_cpu_peak(arguments) -> PeakResult:
    output_string, number, cpu = arguments
    output = Path(output_string)
    best = PeakResult(-np.inf, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0)
    for level, cell_size, density, xyz in iter_owned_wave_cells(output, number, cpu):
        index = int(np.argmax(density))
        maximum = float(density[index])
        if level == PEAK_LEVEL and maximum > best.density:
            best.density = maximum
            best.x = float(xyz[0][index])
            best.y = float(xyz[1][index])
            best.z = float(xyz[2][index])
            best.level = level
        volume = cell_size**3
        best.mass += float(np.sum(density, dtype=np.float64)) * volume
        best.volume += density.size * volume
        best.cells += int(density.size)
    return best


def periodic_offset(values: np.ndarray, center: float) -> np.ndarray:
    return (values - center + 0.5) % 1.0 - 0.5


def deposit_cells(total, count, x, y, density, cell_size, width):
    npixels = total.shape[0]
    pixel_size = width / npixels
    repeat = int(round(cell_size / pixel_size))
    if repeat < 1:
        repeat = 1
    left_x = (x - 0.5 * cell_size + 0.5 * width) / pixel_size
    left_y = (y - 0.5 * cell_size + 0.5 * width) / pixel_size
    # A display pixel belongs to the AMR cell that contains its centre. This
    # remains gap-free when the display grid is translated to the halo peak.
    ix0 = np.ceil(left_x - 0.5 - 1.0e-10).astype(np.int64)
    iy0 = np.ceil(left_y - 0.5 - 1.0e-10).astype(np.int64)
    for offset_x in range(repeat):
        ix = ix0 + offset_x
        valid_x = (ix >= 0) & (ix < npixels)
        if not np.any(valid_x):
            continue
        for offset_y in range(repeat):
            iy = iy0 + offset_y
            valid = valid_x & (iy >= 0) & (iy < npixels)
            if not np.any(valid):
                continue
            flat = iy[valid] * npixels + ix[valid]
            np.add.at(total.ravel(), flat, density[valid])
            np.add.at(count.ravel(), flat, 1.0)


def slice_cpu(arguments):
    output_string, number, cpu, center = arguments
    output = Path(output_string)
    totals = [np.zeros((size, size), dtype=np.float64) for _, size in PANELS]
    counts = [np.zeros((size, size), dtype=np.float64) for _, size in PANELS]

    for _, cell_size, density, xyz in iter_owned_wave_cells(output, number, cpu):
        dz = periodic_offset(xyz[2], center[2])
        plane = np.abs(dz) <= 0.5 * cell_size
        if not np.any(plane):
            continue
        x = periodic_offset(xyz[0][plane], center[0])
        y = periodic_offset(xyz[1][plane], center[1])
        values = density[plane]
        for index, (width, _) in enumerate(PANELS):
            inside = (np.abs(x) <= 0.5 * (width + cell_size)) & (np.abs(y) <= 0.5 * (width + cell_size))
            if np.any(inside):
                deposit_cells(
                    totals[index], counts[index], x[inside], y[inside], values[inside], cell_size, width
                )
    return totals, counts


def parse_output_number(output: Path) -> int:
    return int(output.name.split("_")[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--ncpu", type=int, default=96)
    arguments = parser.parse_args()

    number = parse_output_number(arguments.output)
    jobs = [(str(arguments.output), number, cpu) for cpu in range(1, arguments.ncpu + 1)]
    print(f"Scanning {arguments.ncpu} domains for the densest leaf cell", flush=True)
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        peaks = list(pool.map(find_cpu_peak, jobs))

    peak = max(peaks, key=lambda item: item.density)
    mass = sum(item.mass for item in peaks)
    volume = sum(item.volume for item in peaks)
    cells = sum(item.cells for item in peaks)
    mean_density = mass / volume
    center = (peak.x, peak.y, peak.z)
    print(
        f"peak rho/rhobar={peak.density / mean_density:.6g} at {center}, "
        f"level={peak.level}, leaf_cells={cells:,}, tiled_volume={volume:.12f}",
        flush=True,
    )
    if not np.isclose(volume, 1.0, rtol=0.0, atol=2.0e-8):
        raise RuntimeError(f"leaf-cell volume does not tile the periodic box: {volume}")

    print("Reconstructing the halo-centred AMR slice", flush=True)
    slice_jobs = [(str(arguments.output), number, cpu, center) for cpu in range(1, arguments.ncpu + 1)]
    panel_totals = [np.zeros((size, size), dtype=np.float64) for _, size in PANELS]
    panel_counts = [np.zeros((size, size), dtype=np.float64) for _, size in PANELS]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for totals, counts in pool.map(slice_cpu, slice_jobs):
            for index in range(len(PANELS)):
                panel_totals[index] += totals[index]
                panel_counts[index] += counts[index]

    maps = []
    for total, count in zip(panel_totals, panel_counts):
        if np.any(count == 0):
            raise RuntimeError(f"slice reconstruction left {np.count_nonzero(count == 0)} pixels empty")
        maps.append(total / count / mean_density)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
        }
    )
    fig = plt.figure(figsize=(10.8, 9.0), facecolor="white")
    axes = [
        fig.add_axes((0.025, 0.035, 0.750, 0.900), zorder=1),
        fig.add_axes((0.385, 0.505, 0.370, 0.444), zorder=3),
        fig.add_axes((0.065, 0.205, 0.300, 0.360), zorder=4),
        fig.add_axes((0.345, 0.055, 0.330, 0.396), zorder=5),
    ]
    widths_kpc_h = [1200.0 * width for width, _ in PANELS]
    base_peak = (-0.34 * widths_kpc_h[0], 0.34 * widths_kpc_h[0])
    selection_centers = (base_peak, (0.0, 0.0), (0.0, 0.0))
    image = None
    for index, (ax, density, width) in enumerate(zip(axes, maps, widths_kpc_h)):
        if index == 0:
            shift_y = int(round(base_peak[1] / width * density.shape[0]))
            shift_x = int(round(base_peak[0] / width * density.shape[1]))
            density = np.roll(density, shift=(shift_y, shift_x), axis=(0, 1))
        image = ax.imshow(
            np.log10(np.maximum(density, 1.0e-3)),
            origin="lower",
            cmap="magma",
            vmin=-0.8,
            vmax=6.2,
            extent=(-width / 2, width / 2, -width / 2, width / 2),
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(0.9 if index == 0 else 1.15)
        ax.text(
            0.035,
            0.035 if index == 0 else 0.965,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom" if index == 0 else "top",
            color="white",
            fontsize=13 if index == 0 else 11,
            fontweight="bold",
        )

        scale = (300.0, 75.0, 20.0, 5.0)[index]
        ybar = -0.42 * width
        xbar_left = -0.43 * width
        ax.plot(
            [xbar_left, xbar_left + scale],
            [ybar, ybar],
            color="white",
            linewidth=2.2,
            solid_capstyle="butt",
        )
        ax.text(
            xbar_left + 0.5 * scale,
            ybar + 0.035 * width,
            rf"{scale:g} ckpc $h^{{-1}}$",
            ha="center",
            va="bottom",
            color="white",
            fontsize=9 if index == 0 else 7.5,
        )

    highlight = "#F5F7FA"
    for index in range(len(axes) - 1):
        current = axes[index]
        following = axes[index + 1]
        next_width = widths_kpc_h[index + 1]
        half = 0.5 * next_width
        center_x, center_y = selection_centers[index]
        current.add_patch(
            Rectangle(
                (center_x - half, center_y - half),
                next_width,
                next_width,
                fill=False,
                edgecolor=highlight,
                linewidth=1.1,
                zorder=5,
            )
        )
        if index == 0:
            connectors = (
                ((center_x + half, center_y + half), (0.0, 1.0)),
                ((center_x + half, center_y - half), (0.0, 0.0)),
            )
        elif index == 1:
            connectors = (
                ((center_x - half, center_y - half), (0.0, 1.0)),
                ((center_x + half, center_y - half), (1.0, 1.0)),
            )
        else:
            connectors = (
                ((center_x + half, center_y + half), (0.0, 1.0)),
                ((center_x + half, center_y - half), (0.0, 0.0)),
            )
        for source, target in connectors:
            fig.add_artist(
                ConnectionPatch(
                    xyA=source,
                    coordsA=current.transData,
                    xyB=target,
                    coordsB=following.transAxes,
                    color=highlight,
                    linewidth=0.8,
                    alpha=0.85,
                    clip_on=False,
                    zorder=2.0 if index == 0 else index + 2.5,
                )
            )

    colorbar_axis = fig.add_axes((0.825, 0.155, 0.025, 0.690))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(r"$\log_{10}(\rho_{\rm FDM}/\bar\rho_{\rm FDM})$")
    fig.savefig(FIGURE, dpi=220)
    print(FIGURE)


if __name__ == "__main__":
    main()
