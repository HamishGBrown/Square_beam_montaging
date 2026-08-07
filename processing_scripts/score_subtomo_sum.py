#!/usr/bin/env python3
"""
Score the sum volume ``relion_tomo_subtomo --sum`` writes, against a decoy sum.

Why a sum, and why a decoy
--------------------------
The particles have no orientations yet, so the sum of N of them is a
rotationally averaged particle: for a ribosome, a compact ball ~300 A across
sitting on a flat background. That is a real signal -- if the geometry were
wrong the contributing boxes would be uncorrelated and the sum would be flat --
but on its own it proves nothing, because crowded cytoplasm averages to a ball
too, and so does a radial artefact of the box itself.

So the same extraction is run on decoy positions displaced sideways within the
same tiles, and the two are compared. The decoy sees the same dose, the same
defocus, the same tiles and the same box; it differs in one respect only.

What is measured
----------------
``|peak| / rms``   central density relative to the shell outside the particle,
                   after low-passing to the particle's own scale. Looking at the
                   unfiltered sum is looking at the wrong frequency band.
``radius``         where the radial profile falls to half its central value,
                   doubled. Should land near the particle diameter. A "blob"
                   that is the size of the box is the box, not a particle.

Verdict: the picks' contrast must clear 2.0 *and* beat the decoy by 1.5x, the
same thresholds ``overlay_picks_on_montage`` uses on the 2D patch average.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def find_sum(path: str) -> str:
    """Locate the sum volume in a ``--sum --no_writing`` output directory."""
    if os.path.isfile(path):
        return path
    cands = sorted(glob.glob(os.path.join(path, "**", "*.mrc"), recursive=True))
    if not cands:
        raise FileNotFoundError(f"No .mrc under {path}")
    # RELION names it *sum*; fall back to the only file if it does not.
    named = [c for c in cands if "sum" in os.path.basename(c).lower()]
    # A CTF/multiplicity volume is not the thing to score.
    named = [c for c in named if "ctf" not in os.path.basename(c).lower()]
    chosen = (named or cands)[0]
    if len(cands) > 1:
        logger.info("scoring %s (of %d volumes in %s)",
                    os.path.basename(chosen), len(cands), path)
    return chosen


def score(vol: np.ndarray, pixel_size: float, diameter: float = 300.0):
    """Return ``(contrast, half_max_diameter_A, radial_profile)``."""
    from scipy import ndimage

    v = vol.astype(np.float64)
    # Low-pass to the particle's own scale: a Gaussian whose FWHM is a quarter
    # of the diameter keeps the particle and drops the per-voxel noise.
    sigma = (diameter / 4.0) / pixel_size / 2.355
    sm = ndimage.gaussian_filter(v, sigma)

    c = np.array(sm.shape) / 2.0
    idx = np.indices(sm.shape)
    r = np.sqrt(sum((idx[k] - c[k] + 0.5) ** 2 for k in range(3))) * pixel_size

    # Background is the shell beyond 1.5 particle diameters, inside the box.
    outer = r > 1.5 * diameter
    if outer.sum() < 100:
        outer = r > r.max() * 0.7
    bg, rms = sm[outer].mean(), sm[outer].std()
    sm = (sm - bg) / (rms + 1e-12)

    # Sign is a convention (RELION inverts contrast by default), so score |.|.
    peak = sm[tuple(int(x) for x in c)]
    contrast = abs(float(peak))

    edges = np.arange(0, r.max(), pixel_size)
    which = np.digitize(r.ravel(), edges) - 1
    prof = np.bincount(which, sm.ravel(), len(edges)) / np.maximum(
        np.bincount(which, minlength=len(edges)), 1
    )
    prof = prof * np.sign(peak) if peak != 0 else prof
    half = np.flatnonzero(prof < prof[0] / 2.0)
    d_half = 2.0 * edges[half[0]] if half.size else float("nan")
    return contrast, d_half, (edges, prof)


def parse_commandline(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--real", required=True, help="subtomo --sum output dir (picks)")
    p.add_argument("--decoy", default=None, help="the same for displaced picks")
    p.add_argument("--pixel-size", type=float, required=True, help="A/px of the sum")
    p.add_argument("--diameter", type=float, default=300.0, help="particle diameter, A")
    p.add_argument("-o", "--out", required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_commandline(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mrcfile

    results = {}
    for label, path in (("picks", args.real), ("decoy", args.decoy)):
        if path is None:
            continue
        with mrcfile.open(find_sum(path), permissive=True) as m:
            vol = np.asarray(m.data, dtype=np.float64)
        results[label] = score(vol, args.pixel_size, args.diameter) + (vol,)
        c, d, _, _ = results[label]
        logger.info("%-6s |peak|/rms = %5.2f   half-max diameter = %.0f A",
                    label, c, d)

    os.makedirs(args.out, exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))
    for ax, (label, (c, d, prof, vol)) in zip(axes, results.items()):
        mid = vol[vol.shape[0] // 2]
        ax.imshow(mid, cmap="gray")
        ax.set_title(f"{label} (central slice)\n|peak|/rms = {c:.2f}, "
                     f"d$_{{1/2}}$ = {d:.0f} A")
        ax.set_axis_off()
    for label, (c, d, (edges, prof), _) in results.items():
        axes[-1].plot(edges, prof, label=f"{label} ({c:.2f})")
    axes[-1].axvline(args.diameter / 2, color="k", ls=":", lw=1,
                     label=f"{args.diameter:.0f} A radius")
    axes[-1].set_xlabel("radius (A)")
    axes[-1].set_ylabel("density (sigma of background shell)")
    axes[-1].legend()
    png = os.path.join(args.out, "subtomo_sum.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", png)

    real = results.get("picks")
    dec = results.get("decoy")
    if real is None:
        return 0
    ok = real[0] >= 2.0 and (dec is None or real[0] > 1.5 * dec[0])
    if ok:
        logger.info(
            "PASS: the picks' sum is %s. The extraction is pulling real density "
            "out of the raw tiles.",
            "clearly stronger than the decoy's" if dec else "strong",
        )
    else:
        logger.warning(
            "FAIL: |peak|/rms = %.2f vs decoy %s. Either the geometry is wrong "
            "or these picks are not particles. Do not spend GPU hours on a "
            "refinement to find out -- go back to overlay_picks_on_montage.",
            real[0], "n/a" if dec is None else f"{dec[0]:.2f}",
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
