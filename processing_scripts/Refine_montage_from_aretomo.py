#!/usr/bin/env python3
from .Utilities import load_array_from_hdf5, save_array_to_hdf5
import numpy as np
import argparse
import os
from pathlib import Path


def parse_commandline():
    parser = argparse.ArgumentParser(
        description=(
            "Refine montage stitching using AreTomo alignment output. "
            "This script applies global tilt shifts from AreTomo's IMOD-compatible .xf output "
            "to tile positions stored in a montage HDF5 file."
        )
    )
    parser.add_argument(
        "-h5",
        "--h5file",
        help="HDF5 file containing montage tile positions and metadata",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-x",
        "--xf_file",
        help="AreTomo IMOD-compatible .xf alignment file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-t",
        "--tlt_file",
        help=(
            "Optional AreTomo .tlt file listing tilt angles in the same order as the .xf lines. "
            "If omitted, the script assumes the .xf lines are already in the same order as the HDF5 tilts."
        ),
        required=False,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory for corrected HDF5 files",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--tolerance",
        help="Tilt-angle matching tolerance in degrees when mapping .xf lines to HDF5 tilts",
        required=False,
        type=float,
        default=0.2,
    )
    return vars(parser.parse_args())


def load_aretomo_xf(xf_file):
    lines = []
    with open(xf_file, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 6:
                raise ValueError(
                    f"Unable to parse .xf line: expected at least 6 values, got {len(parts)}: {line!r}"
                )
            values = [float(x) for x in parts[:6]]
            lines.append(values)
    return np.asarray(lines, dtype=float)


def load_tlt_file(tlt_file):
    tilts = []
    with open(tlt_file, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            tilts.append(float(parts[0]))
    return np.asarray(tilts, dtype=float)


def match_tilt_order(xf_tilts, h5_tilts, tolerance=0.2):
    if len(xf_tilts) == len(h5_tilts):
        return np.arange(len(h5_tilts), dtype=int)

    mapping = np.full(len(h5_tilts), -1, dtype=int)
    for i, tilt in enumerate(h5_tilts):
        diff = np.abs(xf_tilts - tilt)
        best = int(np.argmin(diff))
        if diff[best] > tolerance:
            raise ValueError(
                f"Tilt {tilt:.3f}° from HDF5 could not be matched to any .xf tilt within {tolerance}°"
            )
        mapping[i] = best
    if np.unique(mapping).size != mapping.size:
        raise ValueError("Tilt mapping from .xf to HDF5 is not one-to-one.")
    return mapping


def apply_aretomo_shifts(positions, shifts_microns):
    corrected = positions.copy()
    if corrected.ndim != 3 or corrected.shape[2] != 2:
        raise ValueError("Refined_positions must have shape (ntilts, ntiles, 2)")
    if corrected.shape[0] != shifts_microns.shape[0]:
        raise ValueError("Number of tilt shifts does not match number of tilts in the HDF5 file")
    for iz in range(corrected.shape[0]):
        corrected[iz, :, 0] += shifts_microns[iz, 0]
        corrected[iz, :, 1] += shifts_microns[iz, 1]
    return corrected


def find_matching_tlt_file(xf_file):
    base = os.path.splitext(xf_file)[0]
    candidates = [base + ext for ext in [".tlt", ".txt", "_st.tlt", "_st.txt"]]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    args = parse_commandline()
    output_dir = args["output"] if args["output"] else os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    h5file = args["h5file"]
    xf_file = args["xf_file"]
    tlt_file = args["tlt_file"]

    positions = load_array_from_hdf5(h5file, "Refined_positions")
    index_map = load_array_from_hdf5(h5file, "index_map")
    tilts = load_array_from_hdf5(h5file, "tilts")
    montage_names = load_array_from_hdf5(h5file, "Montage_filenames")
    pixel_size = load_array_from_hdf5(h5file, "binned_pixel_size")
    pixel_size = pixel_size[0] if np.ndim(pixel_size) > 0 else pixel_size

    xf_data = load_aretomo_xf(xf_file)
    if tlt_file is None:
        tlt_file = find_matching_tlt_file(xf_file)

    if tlt_file is not None:
        xf_tilts = load_tlt_file(tlt_file)
    else:
        xf_tilts = None

    if xf_tilts is None:
        if xf_data.shape[0] != len(tilts):
            raise ValueError(
                "The number of .xf lines does not match the number of tilts in the HDF5 file, "
                "and no .tlt file was provided to establish the mapping."
            )
        mapping = np.arange(len(tilts), dtype=int)
    else:
        if xf_data.shape[0] != len(xf_tilts):
            raise ValueError("Number of lines in .xf file does not match number of tilts in .tlt file.")
        mapping = match_tilt_order(xf_tilts, tilts, tolerance=args["tolerance"])

    shifts_pixels = xf_data[mapping, 4:6]
    shifts_microns = shifts_pixels * pixel_size * 1e-4
    positions_corrected = apply_aretomo_shifts(positions, shifts_microns)

    output_h5 = os.path.join(output_dir, os.path.splitext(os.path.basename(h5file))[0] + "_aretomo_corrected.h5")
    save_array_to_hdf5(
        [tilts, positions_corrected, index_map, montage_names, np.asarray([pixel_size]), shifts_microns],
        output_h5,
        ["tilts", "Refined_positions", "index_map", "Montage_filenames", "binned_pixel_size", "Aretomo_shifts_microns"],
    )

    summary_file = os.path.join(output_dir, os.path.splitext(os.path.basename(h5file))[0] + "_aretomo_mapping.txt")
    with open(summary_file, "w") as fh:
        fh.write(f"HDF5 file: {h5file}\n")
        fh.write(f"Aretomo XF file: {xf_file}\n")
        if tlt_file is not None:
            fh.write(f"Aretomo TLT file: {tlt_file}\n")
        fh.write(f"Pixel size (micron): {pixel_size}\n")
        fh.write("Tilt index mapping (HDF5 tilt -> XF line index, shift px -> shift µm):\n")
        for iz, tilt in enumerate(tilts):
            xf_idx = mapping[iz]
            px = shifts_pixels[iz]
            um = shifts_microns[iz]
            fh.write(f"  {iz:03d}: tilt={tilt:.3f}°, xf_line={xf_idx}, shift_px=({px[0]:.3f},{px[1]:.3f}), shift_um=({um[0]:.6f},{um[1]:.6f})\n")

    print(f"Saved corrected positions to {output_h5}")
    print(f"Saved tilt mapping summary to {summary_file}")

    for iz, tilt in enumerate(tilts):
        out_h5 = os.path.join(output_dir, f"tilt_{tilt:02.0f}_aretomo_corrected_positions.h5")
        save_array_to_hdf5([positions_corrected[iz]], out_h5, ["Refined_positions"])

    print(f"Saved per-tilt corrected position files for {len(tilts)} tilts.")


if __name__ == "__main__":
    main()
