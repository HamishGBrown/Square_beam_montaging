#!/usr/bin/env python3
"""Motion correction for square-beam montage raw frames.

Workflow
--------
1. Load a multi-frame TIFF
2. Auto-detect the square beam mask from the mean frame
3. Find the largest axis-aligned square inscribed in the mask
4. Crop all sub-frames to that square, write a temporary TIFF
5. Run MotionCor2 on the cropped TIFF to estimate per-frame shifts
6. Parse per-frame shifts from the MotionCor2 log
7. Apply the same shifts (Fourier shift) to the original full-size frames
8. Sum the shifted frames → write float32 MRC (single 2-D slice)

The crop is used ONLY for motion estimation; the output MRC covers the
full detector area, so downstream stitching is unaffected.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mrcfile
import numpy as np
from PIL import Image
from tqdm import tqdm

from .Utilities import make_mask

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TIFF I/O
# ---------------------------------------------------------------------------

def load_multipage_tiff(path: str) -> np.ndarray:
    """Return (n_frames, H, W) uint8/uint16 array from a multi-page TIFF."""
    with Image.open(path) as img:
        n = getattr(img, "n_frames", 1)
        img.seek(0)
        first = np.asarray(img.copy())
        out = np.empty((n, *first.shape), dtype=first.dtype)
        out[0] = first
        for i in range(1, n):
            img.seek(i)
            out[i] = np.asarray(img.copy())
    return out


def write_multipage_tiff(frames: np.ndarray, path: str) -> None:
    """Write (n_frames, H, W) array as a multi-page TIFF."""
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        compression="tiff_lzw",
    )


# ---------------------------------------------------------------------------
# Beam mask + largest inscribed square
# ---------------------------------------------------------------------------

def largest_inscribed_square(mask: np.ndarray) -> tuple[int, int, int]:
    """Find the largest axis-aligned square with all pixels inside ``mask``.

    Uses the classic DP histogram approach.

    Returns
    -------
    (row_top, col_left, side)  — corner of the square and its side length.
    """
    h, w = mask.shape
    dp = np.zeros((h, w), dtype=np.int32)
    dp[0, :] = mask[0, :].astype(np.int32)
    dp[:, 0] = mask[:, 0].astype(np.int32)

    for r in range(1, h):
        for c in range(1, w):
            if mask[r, c]:
                dp[r, c] = int(min(dp[r - 1, c], dp[r, c - 1], dp[r - 1, c - 1])) + 1

    idx = int(np.argmax(dp))
    r_br, c_br = divmod(idx, w)
    side = int(dp[r_br, c_br])
    row_top = r_br - side + 1
    col_left = c_br - side + 1
    return row_top, col_left, side


def _fft_friendly_size(n: int) -> int:
    """Largest k ≤ n whose prime factors are all in {2, 3, 5, 7} (CUFFT fast path).

    CUFFT_INTERNAL_ERROR occurs when a dimension has large prime factors (e.g.
    2333 is prime).  CUFFT documents 2/3/5/7 as its supported fast-path primes.
    """
    _CUFFT_PRIMES = (2, 3, 5, 7)
    k = n
    while k > 1:
        m = k
        for p in _CUFFT_PRIMES:
            while m % p == 0:
                m //= p
        if m == 1:
            return k
        k -= 1
    return 1


# ---------------------------------------------------------------------------
# MotionCor2 interaction
# ---------------------------------------------------------------------------

def _find_motioncor2() -> str:
    """Return path to MotionCor2 executable, searching common locations."""
    candidates = [
        "MotionCor2",
        "/programs/x86_64-linux/motioncor2/1.6.4/MotionCor2",
        "/programs/x86_64-linux/motioncor2/1.3.2/MotionCor2",
    ]
    for c in candidates:
        if shutil.which(c):
            return c
    raise FileNotFoundError(
        "MotionCor2 not found. Provide --motioncor2 or load it via sbgrid."
    )


def run_motioncor2(
    in_tiff: str,
    out_mrc: str,
    log_dir: str,
    pixel_size: float,
    bfactor: float,
    gpu: int,
    motioncor2_path: str,
    extra_args: list[str] | None = None,
) -> str:
    """Run MotionCor2 on ``in_tiff``, return captured stdout text."""
    cmd = [
        motioncor2_path,
        "-InTiff", in_tiff,
        "-OutMrc", out_mrc,
        "-PixSize", str(pixel_size),
        "-Bft", str(bfactor),
        "-Gpu", str(gpu),
        "-LogDir", log_dir,
        "-OutStack", "0",
    ]
    if extra_args:
        cmd.extend(extra_args)

    log.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        log.error("MotionCor2 stdout:\n%s", result.stdout[-2000:])
        log.error("MotionCor2 stderr:\n%s", result.stderr[-2000:])
        raise RuntimeError(
            f"MotionCor2 failed with exit code {result.returncode}."
        )
    return result.stdout


def parse_motioncor2_shifts(text: str, n_frames: int) -> np.ndarray:
    """Parse per-frame shifts from MotionCor2 stdout or log file text.

    MotionCor2 prints a table like::

        Full-frame alignment
          Frame   x Shift   y Shift
            1      0.000      0.000
            2      0.950      0.030

    Indices may be 0- or 1-based depending on version; both are handled.

    Returns
    -------
    shifts : (n_frames, 2) float32 array, columns = (shift_x, shift_y) in pixels.
    """
    pattern = re.compile(
        r"^\s*(\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s*$"
    )
    found: dict[int, tuple[float, float]] = {}
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            found[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))

    if not found:
        raise ValueError(
            "No shift lines found in MotionCor2 output.\n"
            f"Output was:\n{text[-2000:]}"
        )

    # Handle 0-based or 1-based frame indexing
    offset = min(found.keys())
    shifts = np.zeros((n_frames, 2), dtype=np.float32)
    for i in range(n_frames):
        key = i + offset
        if key in found:
            shifts[i, 0] = found[key][0]
            shifts[i, 1] = found[key][1]
        else:
            log.warning("Frame %d shift not found in output; using 0,0.", i)
    return shifts


# ---------------------------------------------------------------------------
# Shift application
# ---------------------------------------------------------------------------

def _fourier_shift_2d(frame: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Apply subpixel shift (dy, dx) to a 2-D array via the Fourier shift theorem."""
    h, w = frame.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.rfftfreq(w)
    phase = np.exp(-2j * np.pi * (dy * fy[:, None] + dx * fx[None, :]))
    f_fft = np.fft.rfft2(frame.astype(np.float32))
    shifted = np.fft.irfft2(f_fft * phase, s=frame.shape)
    return shifted.astype(np.float32)


def apply_shifts_and_sum(
    frames: np.ndarray,
    shifts_xy: np.ndarray,
) -> np.ndarray:
    """Shift each frame by (dx, dy) and return their sum.

    Parameters
    ----------
    frames : (n_frames, H, W) array
    shifts_xy : (n_frames, 2) — column 0 = shift_x, column 1 = shift_y

    Returns
    -------
    (H, W) float32 sum of aligned frames.
    """
    n = frames.shape[0]
    result = np.zeros(frames.shape[1:], dtype=np.float32)
    for i in range(n):
        dx = float(shifts_xy[i, 0])
        dy = float(shifts_xy[i, 1])
        shifted = _fourier_shift_2d(frames[i].astype(np.float32), dy, dx)
        result += shifted
    return result


def apply_shifts_split(
    frames: np.ndarray,
    shifts_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift frames and return (full_sum, odd_sum, even_sum) in a single pass.

    odd_sum  accumulates frames at 0-based indices 0, 2, 4, …
    even_sum accumulates frames at 0-based indices 1, 3, 5, …

    All three sums use the same per-frame Fourier shifts, so no extra FFTs
    are computed compared to a plain sum.
    """
    n = frames.shape[0]
    full_sum = np.zeros(frames.shape[1:], dtype=np.float32)
    odd_sum  = np.zeros(frames.shape[1:], dtype=np.float32)
    even_sum = np.zeros(frames.shape[1:], dtype=np.float32)
    for i in range(n):
        dx = float(shifts_xy[i, 0])
        dy = float(shifts_xy[i, 1])
        shifted = _fourier_shift_2d(frames[i].astype(np.float32), dy, dx)
        full_sum += shifted
        if i % 2 == 0:
            odd_sum += shifted
        else:
            even_sum += shifted
    return full_sum, odd_sum, even_sum


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------

def save_diagnostic_plot(
    mean_frame: np.ndarray,
    mask: np.ndarray,
    row_top: int,
    col_left: int,
    side: int,
    output_path: str,
    binning: int = 4,
) -> None:
    """Save a PNG showing the mean frame, beam mask, and crop rectangle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    b = max(1, binning)
    img = mean_frame[::b, ::b]
    msk = mask[::b, ::b]
    r, c, s = row_top // b, col_left // b, max(1, side // b)

    vmin = float(np.percentile(img, 1))
    vmax = float(np.percentile(img, 99))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(os.path.basename(output_path).replace("_diagnostic.png", ""), fontsize=10)

    # Left: mask overlay
    axes[0].imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    outside = np.where(~msk, 1.0, np.nan)
    axes[0].imshow(outside, cmap="Reds", alpha=0.5, vmin=0, vmax=1,
                   interpolation="nearest")
    axes[0].set_title("Beam mask  (red = excluded)", fontsize=9)
    axes[0].axis("off")

    # Right: crop rectangle
    axes[1].imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    rect = mpatches.Rectangle(
        (c, r), s, s,
        linewidth=1.5, edgecolor="lime", facecolor="none",
    )
    axes[1].add_patch(rect)
    axes[1].set_title(
        f"Crop region: {side}×{side} px (unbinned)\n"
        f"origin row={row_top} col={col_left}",
        fontsize=9,
    )
    axes[1].axis("off")

    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    log.info("Diagnostic plot: %s", output_path)


# ---------------------------------------------------------------------------
# Per-file orchestration
# ---------------------------------------------------------------------------

def process_one_tiff(
    input_path: str,
    output_dir: str,
    pixel_size: float,
    motioncor2_path: str,
    bfactor: float = 100.0,
    gpu: int = 0,
    mask_shrink: int = 20,
    mask_threshold: float | None = None,
    template_mask: np.ndarray | None = None,
    keep_temp: bool = False,
    save_diagnostic: bool = False,
    diagnostic_binning: int = 4,
    mc2_extra: list[str] | None = None,
    split_frames: bool = False,
) -> str:
    """Process a single multi-frame TIFF; return path to the output MRC.

    When ``split_frames`` is True, additional files are written:
      {output_dir}/odd/{stem}.mrc  — sum of frames at indices 0, 2, 4, …
      {output_dir}/even/{stem}.mrc — sum of frames at indices 1, 3, 5, …
    These half-datasets are ready to use directly with cryoCARE or similar.
    """
    in_stem = Path(input_path).stem
    out_mrc  = os.path.join(output_dir, f"{in_stem}.mrc")
    out_odd  = os.path.join(output_dir, "odd",  f"{in_stem}.mrc")
    out_even = os.path.join(output_dir, "even", f"{in_stem}.mrc")

    skip = os.path.exists(out_mrc)
    if split_frames:
        skip = skip and os.path.exists(out_odd) and os.path.exists(out_even)
    if skip:
        log.info("Output already exists, skipping: %s", out_mrc)
        return out_mrc

    # 1. Load frames
    frames = load_multipage_tiff(input_path)  # (n, H, W)
    if frames.ndim == 2:
        frames = frames[np.newaxis]  # single frame
    n_frames, H, W = frames.shape

    # 2. Beam mask from mean frame
    mean_frame = frames.astype(np.float32).mean(axis=0)
    if template_mask is not None:
        mask = make_mask(mean_frame, template_mask=template_mask, shrinkn=mask_shrink,
                         absolutethreshold=mask_threshold)
    else:
        mask = make_mask(mean_frame, shrinkn=mask_shrink, absolutethreshold=mask_threshold)

    # 3. Largest inscribed square, rounded to an FFT-friendly size
    row_top, col_left, side = largest_inscribed_square(mask)
    if side < 64:
        raise ValueError(
            f"Largest inscribed square is only {side}px — beam mask may have failed."
        )
    fft_side = _fft_friendly_size(side)
    if fft_side < side:
        # Centre the smaller FFT-friendly square within the inscribed square
        pad = (side - fft_side) // 2
        row_top += pad
        col_left += pad
        log.info(
            "%s: crop adjusted %d→%d px (FFT-friendly)  row=%d col=%d  (full frame %dx%d)",
            in_stem, side, fft_side, row_top, col_left, H, W,
        )
        side = fft_side
    else:
        log.info(
            "%s: square crop  row=%d col=%d side=%d  (full frame %dx%d)",
            in_stem, row_top, col_left, side, H, W,
        )

    # 3b. Optional diagnostic plot (saved before MC2 runs so it exists even on failure)
    if save_diagnostic:
        os.makedirs(output_dir, exist_ok=True)
        diag_path = os.path.join(output_dir, f"{in_stem}_diagnostic.png")
        save_diagnostic_plot(
            mean_frame, mask, row_top, col_left, side,
            diag_path, binning=diagnostic_binning,
        )

    # 4. Crop frames and write temp TIFF
    with tempfile.TemporaryDirectory(prefix="mc2_", suffix=f"_{in_stem}") as tmpdir:
        cropped = frames[:, row_top : row_top + side, col_left : col_left + side]
        tmp_tiff = os.path.join(tmpdir, f"{in_stem}_crop.tif")
        write_multipage_tiff(cropped, tmp_tiff)

        # 5. Run MotionCor2 on cropped TIFF
        tmp_mrc = os.path.join(tmpdir, f"{in_stem}_mc2.mrc")
        stdout_text = run_motioncor2(
            in_tiff=tmp_tiff,
            out_mrc=tmp_mrc,
            log_dir=tmpdir,
            pixel_size=pixel_size,
            bfactor=bfactor,
            gpu=gpu,
            motioncor2_path=motioncor2_path,
            extra_args=mc2_extra,
        )

        if keep_temp:
            dest = os.path.join(output_dir, f"{in_stem}.mc2.log")
            with open(dest, "w") as fh:
                fh.write(stdout_text)
            log.info("Saved MC2 output: %s", dest)

        # 6. Parse shifts from captured stdout
        shifts_xy = parse_motioncor2_shifts(stdout_text, n_frames)
        log.info("Shifts (x,y px):\n%s", shifts_xy)

    # 7. Apply shifts to full frames and sum
    os.makedirs(output_dir, exist_ok=True)
    if split_frames:
        aligned_sum, odd_sum, even_sum = apply_shifts_split(frames, shifts_xy)
    else:
        aligned_sum = apply_shifts_and_sum(frames, shifts_xy)

    # 8. Write output MRC(s) (float32, single 2-D slice each)
    with mrcfile.new(out_mrc, overwrite=True) as mrc:
        mrc.set_data(aligned_sum)
        mrc.voxel_size = pixel_size

    if split_frames:
        for label, data in (("odd", odd_sum), ("even", even_sum)):
            sub_dir = os.path.join(output_dir, label)
            os.makedirs(sub_dir, exist_ok=True)
            out_path = os.path.join(sub_dir, f"{in_stem}.mrc")
            with mrcfile.new(out_path, overwrite=True) as mrc:
                mrc.set_data(data)
                mrc.voxel_size = pixel_size
            log.info("Wrote %s frames: %s", label, out_path)

    return out_mrc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=mc2_frames
#SBATCH --output={log_dir}/mc2_%A_%a.out
#SBATCH --error={log_dir}/mc2_%A_%a.err
#SBATCH -p gpu-l40s
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-{n_minus_1}%{max_concurrent}  # %N caps simultaneous tasks

frames_dir="{frames_dir}"
output_dir="{output_dir}"
pixel_size={pixel_size}
bfactor={bfactor}

source /programs/sbgrid.shrc
module load CUDA/12.2.0

MOTIONCOR2=/programs/x86_64-linux/motioncor2/1.6.4/MotionCor2_1.6.4_Cuda121_Mar312023

mapfile -t tiffs < <(ls "${{frames_dir}}"/*.tif | sort)
input="${{tiffs[$SLURM_ARRAY_TASK_ID]}}"

beam_mask_motioncorr \\
    --input "${{input}}" \\
    --output-dir "${{output_dir}}" \\
    --pixel-size ${{pixel_size}} \\
    --bfactor ${{bfactor}} \\
    --motioncor2 ${{MOTIONCOR2}} \\
    --gpu 0
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", nargs="+", required=True,
        help="Input TIFF file(s) or glob pattern(s). Each TIFF is a multi-frame movie.",
    )
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Directory to write motion-corrected MRC files.",
    )
    parser.add_argument(
        "--pixel-size", type=float, required=True,
        help="Pixel size in Ångström (used by MotionCor2 for dose weighting etc.).",
    )
    parser.add_argument(
        "--motioncor2", default=None,
        help="Path to MotionCor2 executable. Auto-detected if not provided.",
    )
    parser.add_argument(
        "--bfactor", type=float, default=100.0,
        help="B-factor for MotionCor2 alignment (default: 100).",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index for MotionCor2 (default: 0).",
    )
    parser.add_argument(
        "--mask-shrink", type=int, default=20,
        help="Erosion radius (px) to shrink beam mask away from edge (default: 20).",
    )
    parser.add_argument(
        "--mask-threshold", type=float, default=None,
        help="Absolute intensity threshold for beam detection. "
             "If omitted, uses 0.4 × image median (same default as make_mask). "
             "Use choose_mask_params to find the right value interactively.",
    )
    parser.add_argument(
        "--template-mask", default=None,
        help="Path to a .npy binary mask to use as beam template (numpy bool array).",
    )
    parser.add_argument(
        "--keep-logs", action="store_true",
        help="Copy the MotionCor2 -Full.log files to the output directory.",
    )
    parser.add_argument(
        "--save-diagnostic", action="store_true",
        help="Save a PNG diagnostic plot ({stem}_diagnostic.png) showing the "
             "mean frame, beam mask, and crop rectangle for each input file.",
    )
    parser.add_argument(
        "--diagnostic-binning", type=int, default=4,
        help="Downsample factor for diagnostic plots (default: 4).",
    )
    parser.add_argument(
        "--split-frames", action="store_true",
        help="Also write odd- and even-frame sums to {output_dir}/odd/ and "
             "{output_dir}/even/. Useful for cryoCARE and similar denoising "
             "tools that require two independent half-datasets.",
    )
    parser.add_argument(
        "--mc2-args", nargs=argparse.REMAINDER, default=[],
        help="Extra arguments forwarded verbatim to MotionCor2 (after --).",
    )
    parser.add_argument(
        "--print-slurm", action="store_true",
        help="Print a SLURM array job script for the matched input files and exit.",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=200,
        help="Max simultaneously active array tasks (default: 200). "
             "Sets the %%N throttle in the SLURM array directive.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Resolve glob patterns
    paths: list[str] = []
    for pat in args.input:
        expanded = sorted(glob.glob(pat))
        if expanded:
            paths.extend(expanded)
        elif os.path.exists(pat):
            paths.append(pat)
        else:
            log.warning("Pattern matched no files: %s", pat)
    paths = sorted(set(paths))

    if not paths:
        sys.exit("No input TIFF files found.")

    if args.print_slurm:
        frames_dir = os.path.dirname(os.path.abspath(paths[0]))
        log_dir = os.path.join(os.path.abspath(args.output_dir), "logs")
        script = DEFAULT_SLURM_TEMPLATE.format(
            frames_dir=frames_dir,
            output_dir=os.path.abspath(args.output_dir),
            log_dir=log_dir,
            pixel_size=args.pixel_size,
            bfactor=args.bfactor,
            n_minus_1=len(paths) - 1,
            max_concurrent=args.max_concurrent,
        )
        print(script)
        return

    mc2_path = args.motioncor2 or _find_motioncor2()

    template_mask = None
    if args.template_mask:
        template_mask = np.load(args.template_mask)

    mc2_extra = args.mc2_args or []
    # Strip a leading '--' separator if present
    if mc2_extra and mc2_extra[0] == "--":
        mc2_extra = mc2_extra[1:]

    os.makedirs(args.output_dir, exist_ok=True)

    n_ok = 0
    for p in tqdm(paths, desc="Motion correcting"):
        try:
            out = process_one_tiff(
                input_path=p,
                output_dir=args.output_dir,
                pixel_size=args.pixel_size,
                motioncor2_path=mc2_path,
                bfactor=args.bfactor,
                gpu=args.gpu,
                mask_shrink=args.mask_shrink,
                mask_threshold=args.mask_threshold,
                template_mask=template_mask,
                keep_temp=args.keep_logs,
                save_diagnostic=args.save_diagnostic,
                diagnostic_binning=args.diagnostic_binning,
                mc2_extra=mc2_extra,
                split_frames=args.split_frames,
            )
            log.info("Wrote: %s", out)
            n_ok += 1
        except Exception as exc:
            log.error("FAILED %s: %s", p, exc)

    print(f"Done: {n_ok}/{len(paths)} files processed successfully.")


if __name__ == "__main__":
    main()
