#!/usr/bin/env python3
"""
Measure where the reprojected picks actually land on the montage, per tilt, and
fit the residual to the geometry terms that could have caused it.

Why
---
``picks_tiltseries_canvas.mod`` viewed in 3dmod shows the picks drifting
perpendicular to the tilt axis as |tilt| grows, while the per-tile models look
right. That is not a contradiction: ``picks_to_imod_model`` defaults to the
section whose .aln TILT is nearest zero, and at theta = 0 every term that could
produce a perpendicular drift is multiplied by sin(theta). The tile view is
blind to exactly this error.

So measure it. For each pick and each tilt, cut a patch centred on the
*predicted* position, cross-correlate it against the same pick's patch at the
reference tilt, and call the peak offset the residual (actual - predicted).

Reading the answer off
----------------------
Work in the aligned frame, where the tilt axis is along ``row`` and the shear is
along ``col``, so the reprojection is

    col = cx + dx cos(theta) - dz sin(theta)

Each way of getting that wrong leaves its own signature in the col residual, and
the fit separates them because they are different functions of theta and dx:

    constant            c0                  a plain offset -> --extra-shift
    dz origin wrong     c1 sin(theta)       -> the z centre is off by c1
    cx origin wrong     c2 (cos(theta) - 1) -> the x centre is off by c2
    tilt angle wrong    c3 dx sin(theta)    -> theta is off by -c3 radians

Patches are sampled in *specimen* units -- the col step is cos(theta) aligned
pixels -- so the foreshortening is undone before correlating and a feature looks
the same at every tilt. The measured col shift is scaled back by cos(theta) to
give an aligned-frame residual.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .montage_projection import MontageProjector, load_picks
from .refine_montage_projmatch import alignment_matrix

logger = logging.getLogger(__name__)


def ncc_peak(patch: np.ndarray, ref: np.ndarray, max_shift: int):
    """
    Normalised cross-correlation peak of ``patch`` against ``ref``.

    Returns ``(drow, dcol, quality)`` where the shift is of the feature in
    ``patch`` relative to ``ref`` -- both being centred on their own predicted
    position, that is exactly the residual -- and ``quality`` is the peak
    height in units of the correlation's own standard deviation, which is a far
    better match criterion than the raw correlation coefficient.
    """
    from scipy.signal import fftconvolve

    a = patch - patch.mean()
    b = ref - ref.mean()
    a /= a.std() + 1e-9
    b /= b.std() + 1e-9
    cc = fftconvolve(a, b[::-1, ::-1], mode="same") / a.size

    c = np.array(cc.shape) // 2
    win = cc[c[0] - max_shift: c[0] + max_shift + 1,
             c[1] - max_shift: c[1] + max_shift + 1]
    i, j = np.unravel_index(int(np.argmax(win)), win.shape)
    quality = float((win[i, j] - win.mean()) / (win.std() + 1e-12))

    # Parabolic sub-pixel refinement, skipped on the border where it is unsafe.
    di = dj = 0.0
    if 0 < i < win.shape[0] - 1:
        y0, y1, y2 = win[i - 1, j], win[i, j], win[i + 1, j]
        den = y0 - 2 * y1 + y2
        di = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    if 0 < j < win.shape[1] - 1:
        y0, y1, y2 = win[i, j - 1], win[i, j], win[i, j + 1]
        den = y0 - 2 * y1 + y2
        dj = 0.5 * (y0 - y2) / den if den != 0 else 0.0

    on_edge = i in (0, win.shape[0] - 1) or j in (0, win.shape[1] - 1)
    return (i - max_shift + di, j - max_shift + dj,
            0.0 if on_edge else quality)


def cut_patch(canvas_slice, proj, tilt, aligned_rc, size, col_step):
    """
    Sample a patch of the canvas on the aligned frame's own axes.

    ``col_step`` compresses the col axis so the patch is in specimen units:
    passing ``cos(theta)`` undoes the tilt foreshortening, which is what makes
    a patch at 57 deg comparable with one at 0 deg.
    """
    from scipy import ndimage

    M, t = alignment_matrix(
        proj.tilts[tilt].sec, proj.canvas_shape, proj.conv, proj.out_bin
    )
    Minv = np.linalg.inv(M)

    half = size // 2
    g = np.arange(-half, half, dtype=float)
    orow, ocol = np.meshgrid(g, g * col_step, indexing="ij")
    offs = np.stack([orow.ravel(), ocol.ravel()], axis=1)

    canvas_pred = proj.aligned_to_canvas(tilt, aligned_rc[None, :2])[0]
    pts = canvas_pred[None, :] + offs @ Minv.T
    vals = ndimage.map_coordinates(
        canvas_slice, [pts[:, 0], pts[:, 1]], order=1, mode="constant", cval=0.0
    )
    return vals.reshape(size, size)


def fit_residual(theta: np.ndarray, dx: np.ndarray, resid: np.ndarray,
                 labels: Sequence[str]) -> None:
    """Least squares of the col residual on the terms that could explain it."""
    A = np.stack([
        np.ones_like(theta),
        np.sin(theta),
        np.cos(theta) - 1.0,
        dx * np.sin(theta),
    ], axis=1)
    coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
    pred = A @ coef
    dof = max(len(resid) - A.shape[1], 1)
    s2 = float(((resid - pred) ** 2).sum() / dof)
    cov = s2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(cov))

    logger.info("fit of the perpendicular (col) residual, %d observations:", len(resid))
    for name, c, e in zip(labels, coef, se):
        flag = "  <-- significant" if abs(c) > 3 * e else ""
        logger.info("    %-28s %+9.2f +- %6.2f%s", name, c, e, flag)
    logger.info("    rms residual before fit %.2f, after %.2f (recon voxels)",
                resid.std(), (resid - pred).std())
    logger.info("  interpretation:")
    logger.info("    z centre off by      %+8.2f recon voxels (%+.0f A)",
                coef[1], coef[1] * 13.704)
    logger.info("    x centre off by      %+8.2f recon voxels (%+.0f A)",
                coef[2], coef[2] * 13.704)
    logger.info("    tilt angle off by    %+8.4f deg", -np.rad2deg(coef[3]))
    return coef, se


def parse_commandline(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--picks", required=True)
    p.add_argument("--tomogram", required=True)
    p.add_argument("--aln", required=True)
    p.add_argument("--positions-dir", required=True)
    p.add_argument("--tile-dir", required=True)
    p.add_argument("--canvas-stack", required=True,
                   help="the stitched montage the picks are checked against")
    p.add_argument("-o", "--out", required=True)

    p.add_argument("--roi", nargs=4, type=float, default=[-5500, 12000, -6000, 12000])
    p.add_argument("--pixel-size", type=float, default=3.426)
    p.add_argument("--binning", type=int, default=2)
    p.add_argument("--out-bin", type=int, default=2)
    p.add_argument("--rotate", default="auto")
    p.add_argument("--extra-shift", nargs=2, type=float, default=[0.0, 0.0])
    p.add_argument("--handedness", type=int, choices=[1, -1], default=1)

    p.add_argument("--patch", type=int, default=160,
                   help="patch size in recon voxels (13.7 A each)")
    p.add_argument("--max-shift", type=int, default=40,
                   help="search radius, recon voxels")
    p.add_argument("--min-quality", type=float, default=4.0,
                   help="reject correlation peaks below this many sigma")
    p.add_argument("--max-tilt", type=float, default=45.0,
                   help="ignore tilts beyond this; the patches stop matching")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_commandline(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mrcfile

    with mrcfile.open(args.tomogram, header_only=True, permissive=True) as m:
        recon_shape = (int(m.header.nz), int(m.header.ny), int(m.header.nx))

    rotate = args.rotate if args.rotate == "auto" else int(args.rotate)
    proj = MontageProjector(
        aln_path=args.aln, positions_dir=args.positions_dir, recon_shape=recon_shape,
        roi=args.roi, pixel_size=args.pixel_size, binning=args.binning,
        out_bin=args.out_bin, rotate=rotate, tile_dir=args.tile_dir,
        extra_shift=args.extra_shift, handedness=args.handedness,
    )
    picks = load_picks(args.picks)
    tilts = sorted(proj.tilts)
    ref_tilt = min(tilts, key=lambda t: abs(proj.theta(t)))
    logger.info("%d picks, %d tilts, reference tilt %+.1f (theta %+.2f deg)",
                len(picks), len(tilts), ref_tilt, np.rad2deg(proj.theta(ref_tilt)))

    # The canvas stack's z index is the rank of the tilt among ALL tilts,
    # dark frames included -- the same SEC convention the .aln uses.
    slice_of = {t: i for i, t in enumerate(proj.all_tilts)}

    with mrcfile.mmap(args.canvas_stack, mode="r", permissive=True) as mrc:
        refs = {}
        sl = np.asarray(mrc.data[slice_of[ref_tilt]], dtype=np.float32)
        al_ref = proj.tomo_to_aligned(ref_tilt, picks)
        cos_ref = np.cos(proj.theta(ref_tilt))
        for n in range(len(picks)):
            refs[n] = cut_patch(sl, proj, ref_tilt, al_ref[n], args.patch, cos_ref)

        rows: List[Tuple] = []
        for tilt in tilts:
            th = proj.theta(tilt)
            if abs(np.rad2deg(th)) > args.max_tilt or tilt == ref_tilt:
                continue
            sl = np.asarray(mrc.data[slice_of[tilt]], dtype=np.float32)
            al = proj.tomo_to_aligned(tilt, picks)
            cos_t = np.cos(th)
            nkeep = 0
            for n in range(len(picks)):
                patch = cut_patch(sl, proj, tilt, al[n], args.patch, cos_t)
                if patch.std() < 1e-6:
                    continue
                dr, dc, q = ncc_peak(patch, refs[n], args.max_shift)
                if q < args.min_quality:
                    continue
                # dc is in specimen units; scale back to the aligned frame.
                nkeep += 1
                rows.append((tilt, th, n, picks[n, 0] - (recon_shape[2] - 1) / 2.0,
                             picks[n, 2] - (recon_shape[0] - 1) / 2.0,
                             dr, dc * cos_t, q))
            logger.info("tilt %+6.1f (theta %+6.2f): %2d/%d picks matched",
                        tilt, np.rad2deg(th), nkeep, len(picks))

    if len(rows) < 8:
        logger.error("Only %d usable matches -- cannot fit. Loosen --min-quality "
                     "or check that --canvas-stack is the right file.", len(rows))
        return 1

    arr = np.array([(r[1], r[3], r[4], r[5], r[6], r[7]) for r in rows])
    theta, dx, dz, drow, dcol, qual = arr.T
    logger.info("median |col residual| = %.2f recon voxels (%.0f A)",
                np.median(np.abs(dcol)), np.median(np.abs(dcol)) * 13.704)
    logger.info("median |row residual| = %.2f recon voxels (along the tilt axis)",
                np.median(np.abs(drow)))

    coef, se = fit_residual(theta, dx, dcol, [
        "constant", "sin(theta)  [z centre]",
        "cos(theta)-1  [x centre]", "dx sin(theta)  [tilt angle]",
    ])

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "reprojection_residual.tsv"), "w") as fh:
        fh.write("#tilt\ttheta_deg\tpick\tdx\tdz\tdrow\tdcol\tquality\n")
        for r in rows:
            fh.write("%+.1f\t%+.3f\t%d\t%.1f\t%.1f\t%+.2f\t%+.2f\t%.1f\n"
                     % (r[0], np.rad2deg(r[1]), r[2], r[3], r[4], r[5], r[6], r[7]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    sc = axes[0].scatter(np.rad2deg(theta), dcol, c=dx, cmap="coolwarm", s=14)
    axes[0].set_xlabel("theta (deg)")
    axes[0].set_ylabel("col residual (recon voxels)")
    axes[0].set_title("perpendicular to tilt axis")
    axes[0].axhline(0, color="k", lw=0.5)
    fig.colorbar(sc, ax=axes[0], label="dx (recon voxels)")
    axes[1].scatter(np.rad2deg(theta), drow, s=14, color="#8e8e93")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("theta (deg)")
    axes[1].set_ylabel("row residual (recon voxels)")
    axes[1].set_title("along tilt axis (control)")
    A = np.stack([np.ones_like(theta), np.sin(theta),
                  np.cos(theta) - 1.0, dx * np.sin(theta)], axis=1)
    axes[2].scatter(A @ coef, dcol, s=14, color="#0a84ff")
    lim = [min(dcol.min(), (A @ coef).min()), max(dcol.max(), (A @ coef).max())]
    axes[2].plot(lim, lim, "k--", lw=0.8)
    axes[2].set_xlabel("fitted col residual")
    axes[2].set_ylabel("measured col residual")
    axes[2].set_title("fit")
    png = os.path.join(args.out, "reprojection_residual.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
