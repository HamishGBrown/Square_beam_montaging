#!/usr/bin/env python3
"""
Refine montage tile positions by projection matching against a tomogram.

AreTomo refines a tilt series by reprojecting the tomogram and shifting each
whole image to match (see AreTomo2/ProjAlign/CProjAlignMain.cpp). For a square
beam montage each tilt is a stitch of ~20 tiles and the residual error is
largely per-tile, so this script measures a 2D shift per (tilt, tile) instead.

See projection_matching_plan.md for the full design. This file implements the
first two stages:

  v0 -- geometry only (default, with --search-conventions). Builds the
        coordinate chain from tile positions to AreTomo's aligned frame and
        verifies it against AreTomo's own aligned tilt series. No correlation
        against a reconstruction, no reprojection. The point is that every sign
        and axis convention gets pinned down against ground truth before any
        algorithm is layered on top. SETTLED: rot-1 shift-1 at correlation
        +0.9105, unanimous across three tilts (job 28131022), which is also the
        canonical form from AreTomo's own .xf writer.

  v1 -- static reference (--measure). Reprojects a reconstruction and measures a
        2D residual per (tilt, tile) by masked NCC in mode A. Report only:
        nothing is written back to the HDF5.

One caution on v1: its reference volume contains the tile being matched, which
is self-reinforcing and drives residuals towards zero -- the plan calls
leave-one-out (v2) a correctness requirement for this reason, so v1 magnitudes
are indicative, not trustworthy.

The reprojection-angle sign is settled and is no longer searched. It is derived
from AreTomo's reconstruction and forward-projection kernels together with the
depth-axis flip that `clip rotx` applies; see reproject_volume() for the full
derivation and source citations. --settle-tilt-sign is now only a control.
"""

import argparse
import itertools
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import mrcfile
import numpy as np
from scipy.ndimage import affine_transform

logger = logging.getLogger(__name__)

# Tilt angle embedded in both the tile MRCs and the position HDF5 filenames,
# e.g. Montage_9-A_9-A_-12.mrc / Montage_9-A_9-A_-12_refined_positions.h5
_TILT_RE = re.compile(r"_(-?\d+(?:\.\d+)?)(?:_refined_positions)?\.(?:mrc|h5)$")


# ---------------------------------------------------------------------------
# AreTomo .aln parsing
# ---------------------------------------------------------------------------


@dataclass
class SectionAlignment:
    """Global alignment for one section of the tilt series."""

    sec: int
    rot: float  # tilt axis angle, degrees
    gmag: float
    tx: float  # shift in input-stack (montage canvas) pixels
    ty: float
    tilt: float  # degrees, INCLUDING any -TiltCor offset (see map_sections_to_tilts)


@dataclass
class AlnFile:
    raw_size: Tuple[int, int, int]
    sections: List[SectionAlignment]
    dark_tilts: List[float]
    has_local: bool

    def by_sec(self, sec: int) -> Optional[SectionAlignment]:
        """
        Look up a section by SEC, the 0-based raw frame index.

        SEC is the ONLY reliable join key -- see map_sections_to_tilts for why
        the TILT column is not.
        """
        for s in self.sections:
            if s.sec == sec:
                return s
        return None


def parse_aln(path: str) -> AlnFile:
    """
    Parse an AreTomo .aln file.

    Returns the global ROT/GMAG/TX/TY/TILT table, the list of dark-frame tilt
    angles, and a flag for whether a local (patch) alignment table is present.

    The TILT column holds AreTomo's *corrected* angles -- `-TiltCor 1 <offset>`
    is already folded in -- so it does not match the nominal stage tilts in the
    filenames. Join on SEC via map_sections_to_tilts, never on TILT.
    """
    raw_size = None
    sections: List[SectionAlignment] = []
    dark_tilts: List[float] = []
    has_local = False
    in_local = False

    with open(path, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if "RawSize" in stripped:
                    nums = re.findall(r"-?\d+", stripped.split("=", 1)[1])
                    raw_size = tuple(int(x) for x in nums[:3])
                elif "DarkFrame" in stripped:
                    parts = stripped.split("=", 1)[1].split()
                    dark_tilts.append(float(parts[2]))
                elif "Local Alignment" in stripped:
                    has_local = True
                    in_local = True
                continue
            if in_local:
                # Local/patch table: SEC PATCH X Y DX DY GOODNESS. Deliberately
                # not modelled -- per-tile refinement IS the local alignment.
                continue
            parts = stripped.split()
            if len(parts) < 10:
                continue
            sections.append(
                SectionAlignment(
                    sec=int(parts[0]),
                    rot=float(parts[1]),
                    gmag=float(parts[2]),
                    tx=float(parts[3]),
                    ty=float(parts[4]),
                    tilt=float(parts[9]),
                )
            )

    if raw_size is None:
        raise ValueError(f"{path}: no '# RawSize' header found")
    if not sections:
        raise ValueError(f"{path}: no global alignment rows parsed")

    if has_local:
        logger.warning(
            "%s contains a local (patch) alignment table, which this script "
            "deliberately ignores. Per-tile refinement supplies the local "
            "correction; using both would double-correct. Re-run AreTomo "
            "without -Patch (see 04_AreTomo_nopatch.sh).",
            path,
        )

    logger.info(
        "Parsed %s: RawSize=%s, %d sections, %d dark frames",
        os.path.basename(path),
        raw_size,
        len(sections),
        len(dark_tilts),
    )
    return AlnFile(raw_size, sections, dark_tilts, has_local)


def map_sections_to_tilts(
    aln: AlnFile, nominal_tilts: Sequence[float]
) -> Tuple[Dict[float, SectionAlignment], float]:
    """
    Join .aln sections to nominal tilt angles via SEC, and infer the tilt offset.

    **Do not join on the TILT column.** AreTomo's `-TiltCor 1 <offset>` writes
    the *corrected* angles into the .aln, so for the 9-A dataset the TILT column
    runs -45..+72 while the data runs -60..+60 -- a uniform +12 deg shift. Joining
    on angle silently pairs each tilt with the section 12/3 = 4 positions away
    and every measurement comes out wrong while looking entirely plausible.

    SEC is the 0-based raw frame index into the stack, and stitch.py writes
    z-slices in ascending tilt order (stitch.py:1982), so

        SEC == rank of the tilt angle among all tilts, sorted ascending

    verified exactly (0 mismatches / 37 sections) on the 9-A dataset.

    Parameters
    ----------
    nominal_tilts : sequence of float
        Every tilt in the series, including dark frames -- typically the angles
        parsed from the per-tilt HDF5 filenames. Order is irrelevant; they are
        sorted here.

    Returns
    -------
    (mapping, offset) where mapping is nominal tilt -> SectionAlignment for the
    non-dark sections, and offset is the inferred -TiltCor offset in degrees.
    """
    ordered = sorted(nominal_tilts)
    mapping: Dict[float, SectionAlignment] = {}
    offsets = []
    for sec_idx, nominal in enumerate(ordered):
        sec = aln.by_sec(sec_idx)
        if sec is None:
            continue  # dark frame, dropped from the .aln table
        mapping[nominal] = sec
        offsets.append(sec.tilt - nominal)

    if not offsets:
        raise ValueError(
            "No .aln section matched any tilt by SEC index. The .aln probably "
            "belongs to a different stack than these position files."
        )

    offset = float(np.median(offsets))
    spread = float(np.ptp(offsets))
    if spread > 0.5:
        logger.error(
            "Inferred tilt offsets are inconsistent (spread %.2f deg). The SEC "
            "join is not valid for this dataset -- inspect the .aln by hand.",
            spread,
        )
    elif abs(offset) > 0.05:
        logger.info(
            "Inferred AreTomo -TiltCor offset = %+.2f deg (spread %.3f). The "
            ".aln TILT column already includes it, so reprojection angles "
            "should use TILT as written.",
            offset, spread,
        )
    logger.info(
        "Joined %d/%d tilts to .aln sections by SEC (%d dark frames skipped)",
        len(mapping), len(ordered), len(ordered) - len(mapping),
    )
    return mapping, offset


# ---------------------------------------------------------------------------
# Coordinate chain
# ---------------------------------------------------------------------------


@dataclass
class MontageGeometry:
    """
    The stitch.py canvas convention.

    Positions in the HDF5 are in microns, ordered (x=column, y=row) -- opposite
    to numpy's (row, col). stitch.py places tile t at canvas pixel offset

        col0 = (pos[t,0]*1e4 - origin_A[0]) / pixel_size / binning
        row0 = (pos[t,1]*1e4 - origin_A[1]) / pixel_size / binning

    Beware: the 'binned_pixel_size' dataset inside the HDF5 is a misnomer --
    stitch.py stores pixel_size/binning there, while the canvas pixel size is
    pixel_size*binning (a factor of binning^2 apart). It is not used here.
    """

    pixel_size: float  # unbinned tile pixel size, Angstroms (3.426)
    binning: int  # stitch binning (2)
    origin_A: np.ndarray  # canvas origin (x, y), Angstroms
    canvas_shape: Tuple[int, int]  # (rows, cols) of the stitched montage
    tile_shape: Tuple[int, int]  # (rows, cols) of one tile ON the canvas
    rotate: int = 90  # stitch.py --rotate, degrees CCW

    @property
    def canvas_pixel_size(self) -> float:
        """Angstroms per canvas pixel (6.852 for the 9-A dataset)."""
        return self.pixel_size * self.binning

    @classmethod
    def from_stitch_args(
        cls,
        roi: Sequence[float],
        pixel_size: float,
        binning: int,
        raw_tile_shape: Tuple[int, int],
        rotate: int = 90,
    ) -> "MontageGeometry":
        """
        Reproduce stitch.py's canvas from the arguments given to it.

        `roi` is (x0, x1, y0, y1) in UNBINNED tile pixels, matching
        stitch.py --ROI (see stitch.py:1952). `raw_tile_shape` is the tile's
        (rows, cols) as stored in the MRC, before --rotate is applied.
        """
        roi = [float(x) for x in roi]
        origin_A = np.array([roi[0], roi[2]]) * pixel_size
        width_A = np.array([roi[1] - roi[0], roi[3] - roi[2]]) * pixel_size
        # canvas_px comes out as (x, y); numpy wants (rows, cols) = (y, x)
        canvas_px = (width_A / pixel_size / binning).astype(int)
        canvas_shape = (int(canvas_px[1]), int(canvas_px[0]))

        tile = np.asarray(raw_tile_shape, dtype=int) // binning
        if (rotate // 90) % 2:
            # A 90/270 degree rotation swaps every tile's height and width
            # (stitch.py:714).
            tile = tile[::-1]
        return cls(
            pixel_size=pixel_size,
            binning=binning,
            origin_A=origin_A,
            canvas_shape=canvas_shape,
            tile_shape=(int(tile[0]), int(tile[1])),
            rotate=rotate,
        )

    def tile_origin_rc(self, positions_um: np.ndarray) -> np.ndarray:
        """
        Canvas (row, col) of each tile's top-left corner.

        Parameters
        ----------
        positions_um : (N, 2) array
            Tile positions in microns, (x, y) order, as stored in the HDF5.

        Returns
        -------
        (N, 2) float array of (row, col) canvas pixel offsets.
        """
        xy = (positions_um * 1e4 - self.origin_A) / self.pixel_size / self.binning
        return xy[:, ::-1]  # (x, y) -> (row, col)

    def tile_corners_rc(self, positions_um: np.ndarray) -> np.ndarray:
        """Canvas (row, col) of each tile's four corners, shape (N, 4, 2)."""
        origins = self.tile_origin_rc(positions_um)
        h, w = self.tile_shape
        offsets = np.array([[0, 0], [0, w], [h, w], [h, 0]], dtype=float)
        return origins[:, None, :] + offsets[None, :, :]


@dataclass(frozen=True)
class Convention:
    """
    A candidate sign convention for the canvas -> aligned-frame transform.

    The canonical form, read off AreTomo's own .xf writer
    (AreTomo2/ImodUtil/CSaveXF.cpp:60-70), is

        A = R(-ROT),   d = A @ (-[TX, TY])

    i.e. aligned = R(-ROT) @ (p - centre - [TX, TY]) + centre.

    That fixes the maths but not how it lands in numpy's (row, col) ordering,
    where a rotation can pick up a sign flip depending on axis order. Rather
    than reason about it, v0 enumerates the variants and scores each against
    AreTomo's aligned stack -- see search_conventions().
    """

    # VERIFIED DEFAULT. `rot-1 shift-1` won the v0 search unanimously across
    # tilts -3, 0 and +3 at mean correlation +0.9105 (job 28131022, 2026-07-26),
    # margin 0.095 over the runner-up. It is also the canonical form read off
    # AreTomo's .xf writer, so maths and measurement agree. No rot_offset, no
    # flip_rows, no flip_ty.
    rot_sign: int = -1  # multiplies ROT
    shift_sign: int = -1  # multiplies (TX, TY)
    transpose_xy: bool = False  # treat (TX, TY) as (TY, TX)
    rot_offset: float = 0.0  # added to ROT, in degrees; 180 flips the tilt axis
    flip_rows: bool = False  # mirror the row axis about the centre
    flip_ty: bool = False  # negate the row (TY) shift component only

    def label(self) -> str:
        return (
            f"rot{self.rot_sign:+d} shift{self.shift_sign:+d}"
            f"{' swapTXY' if self.transpose_xy else ''}"
            f"{f' rot+{self.rot_offset:g}' if self.rot_offset else ''}"
            f"{' flipR' if self.flip_rows else ''}"
            f"{' flipTY' if self.flip_ty else ''}"
        )


def alignment_matrix(
    sec: SectionAlignment,
    canvas_shape: Tuple[int, int],
    conv: Convention = Convention(),
    out_bin: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the canvas -> aligned-frame affine transform, in (row, col) space.

    Parameters
    ----------
    sec : SectionAlignment
        Global alignment row for this section. TX/TY are in input-stack
        (montage canvas) pixels -- .aln RawSize confirms this.
    canvas_shape : (rows, cols)
        Shape of the stitched montage.
    conv : Convention
        Sign convention to apply.
    out_bin : float
        AreTomo -OutBin. The aligned stack and reconstruction are this much
        coarser than the canvas, so aligned coordinates scale by 1/out_bin.

    Returns
    -------
    (M, t) such that  aligned_rc = M @ canvas_rc + t.
    """
    theta = np.deg2rad(conv.rot_sign * sec.rot + conv.rot_offset)
    c, s = np.cos(theta), np.sin(theta)
    # Rotation in (row, col) space. Rows are y, cols are x.
    rot = np.array([[c, s], [-s, c]]) * sec.gmag
    if conv.flip_rows:
        # A reflection, det < 0. Not reachable by any combination of rot_sign,
        # rot_offset, shift_sign or transpose_xy -- those are all proper
        # rotations. Composed into `rot` here so the centring below mirrors
        # about the canvas centre rather than about row 0.
        rot = np.array([[-1.0, 0.0], [0.0, 1.0]]) @ rot

    shift_xy = conv.shift_sign * np.array([sec.tx, sec.ty], dtype=float)
    if conv.transpose_xy:
        shift_xy = shift_xy[::-1]
    shift_rc = shift_xy[::-1]  # (x, y) -> (row, col)
    if conv.flip_ty:
        # Negate the row component alone. This is what an MRC(y-up) vs
        # numpy(row-down) *relabelling* does to the translation: the matrix
        # merely conjugates to R(-theta) (reachable via rot_sign), but b -> F@b
        # flips TY and leaves TX alone. shift_sign flips both components and
        # transpose_xy swaps them, so neither can express this on its own.
        shift_rc = shift_rc * np.array([-1.0, 1.0])

    centre = np.array(canvas_shape, dtype=float) / 2.0
    # aligned = R @ (p - centre + shift) + centre, then rescale by out_bin
    t = centre - rot @ centre + rot @ shift_rc
    return rot / out_bin, t / out_bin


def warp_canvas_to_aligned(
    image: np.ndarray,
    sec: SectionAlignment,
    out_shape: Tuple[int, int],
    conv: Convention = Convention(),
    out_bin: float = 1.0,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """
    Resample a canvas-frame image into the aligned frame.

    scipy's affine_transform maps OUTPUT coords to INPUT coords, so the inverse
    of alignment_matrix() is what gets passed through.
    """
    M, t = alignment_matrix(sec, image.shape, conv, out_bin)
    Minv = np.linalg.inv(M)
    return affine_transform(
        image,
        Minv,
        offset=-Minv @ t,
        output_shape=out_shape,
        order=order,
        mode="constant",
        cval=cval,
        prefilter=False,
    )


# ---------------------------------------------------------------------------
# Dataset plumbing
# ---------------------------------------------------------------------------


def tilt_from_filename(path: str) -> float:
    """Extract the tilt angle encoded in a tile MRC or position HDF5 filename."""
    m = _TILT_RE.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"Cannot parse a tilt angle from {path!r}")
    return float(m.group(1))


def find_position_files(directory: str, pattern: str = "_refined_positions.h5") -> Dict[float, str]:
    """Map tilt angle -> per-tilt position HDF5 path."""
    out = {}
    for name in sorted(os.listdir(directory)):
        if name.endswith(pattern):
            out[tilt_from_filename(name)] = os.path.join(directory, name)
    if not out:
        raise FileNotFoundError(f"No *{pattern} files found in {directory}")
    return out


def find_tile_files(directory: str) -> Dict[float, str]:
    """Map tilt angle -> per-tilt tile stack MRC (20 tiles per file)."""
    out = {}
    for name in sorted(os.listdir(directory)):
        if name.endswith(".mrc") and "_refined_positions" not in name:
            try:
                out[tilt_from_filename(name)] = os.path.join(directory, name)
            except ValueError:
                continue
    return out


def load_positions(h5path: str) -> Dict[str, np.ndarray]:
    """Read the datasets this script needs out of a per-tilt position HDF5."""
    with h5py.File(h5path, "r") as h:
        return {
            "positions": h["Refined_positions"][:],
            "original": h["Original_positions"][:],
            "overlaps": h["overlaps"][:],
            "index_map_shape": h["index_map"].shape,
        }


def load_index_map(h5path: str) -> np.ndarray:
    with h5py.File(h5path, "r") as h:
        return h["index_map"][:]


# ---------------------------------------------------------------------------
# v0: convention search and overlay
# ---------------------------------------------------------------------------


def _normalised_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over pixels where both images are finite and nonzero."""
    valid = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    if valid.sum() < 100:
        return float("nan")
    av = a[valid].astype(np.float64)
    bv = b[valid].astype(np.float64)
    av -= av.mean()
    bv -= bv.mean()
    denom = np.sqrt((av * av).sum() * (bv * bv).sum())
    return float((av * bv).sum() / denom) if denom > 0 else float("nan")


def search_conventions(
    canvas: np.ndarray,
    reference: np.ndarray,
    sec: SectionAlignment,
    out_bin: float,
    downsample: int = 8,
) -> List[Tuple[Convention, float]]:
    """
    Score every candidate sign convention against AreTomo's own aligned image.

    This is the core of v0. AreTomo emits its aligned tilt series (-VolZ 0),
    which means the forward transform has ground truth: the correct convention
    is the one whose warped canvas correlates with AreTomo's aligned frame.

    Returns [(convention, correlation), ...] sorted best first.
    """
    ref = reference[::downsample, ::downsample]
    results = []
    seen = {}
    for rot_sign, shift_sign, swap, rot_off, flip, fty in itertools.product(
        (-1, 1), (-1, 1), (False, True), (0.0, 180.0), (False, True), (False, True)
    ):
        conv = Convention(rot_sign, shift_sign, swap, rot_off, flip, fty)
        M, t = alignment_matrix(sec, canvas.shape, conv, out_bin)
        # Deduplicate on the actual transform, so the table reports distinct
        # hypotheses rather than the same one several times over. On the
        # current flag set all 64 combinations do turn out to be distinct
        # (checked 2026-07-26), so this is a guard rather than a filter -- it
        # earns its keep if another axis is added later, since the flags are
        # not obviously independent.
        key = (tuple(np.round(M.ravel(), 6)), tuple(np.round(t, 3)))
        if key in seen:
            continue
        seen[key] = conv
        warped = warp_canvas_to_aligned(
            canvas, sec, reference.shape, conv=conv, out_bin=out_bin, order=1
        )
        score = _normalised_correlation(warped[::downsample, ::downsample], ref)
        results.append((conv, score))
        logger.info("  %-36s correlation = %+.4f", conv.label(), score)
    logger.info(
        "  %d distinct transforms from %d flag combinations",
        len(results), 2 * 2 * 2 * 2 * 2 * 2,
    )
    results.sort(key=lambda r: (-r[1] if np.isfinite(r[1]) else 1e9))
    return results


def plot_footprint_overlay(
    reference: np.ndarray,
    corners_rc: np.ndarray,
    outpath: str,
    title: str = "",
    downsample: int = 8,
) -> None:
    """
    Overlay transformed tile footprints on AreTomo's aligned image.

    If the coordinate chain is right, each quadrilateral lands squarely on the
    corresponding tile in the aligned frame.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = reference[::downsample, ::downsample]
    finite = img[np.isfinite(img)]
    vmin, vmax = np.percentile(finite, [1, 99]) if finite.size else (0, 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    for t, quad in enumerate(corners_rc):
        q = np.vstack([quad, quad[:1]]) / downsample
        ax.plot(q[:, 1], q[:, 0], "-", lw=1.0, color="tab:red", alpha=0.9)
        ax.text(
            q[:-1, 1].mean(),
            q[:-1, 0].mean(),
            str(t),
            color="yellow",
            fontsize=7,
            ha="center",
            va="center",
        )
    ax.set_title(title)
    ax.set_xlabel("aligned frame column / %d" % downsample)
    ax.set_ylabel("aligned frame row / %d" % downsample)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", outpath)


def transform_corners(
    corners_rc: np.ndarray,
    sec: SectionAlignment,
    canvas_shape: Tuple[int, int],
    conv: Convention,
    out_bin: float,
) -> np.ndarray:
    """Push tile corners (N, 4, 2) from canvas into the aligned frame."""
    M, t = alignment_matrix(sec, canvas_shape, conv, out_bin)
    flat = corners_rc.reshape(-1, 2)
    return (flat @ M.T + t).reshape(corners_rc.shape)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def select_check_tilts(
    mapping: Dict[float, SectionAlignment],
    n: int = 3,
    strategy: str = "low",
) -> List[float]:
    """
    Choose tilts for the convention search.

    Two strategies, and the choice is not obvious — see the 2026-07-25 run.

    ``low`` (default) picks the tilts nearest zero, ~(-3, 0, +3). ``high`` picks
    the largest |TX, TY|.

    The original argument for ``high`` was that the section nearest zero tilt is
    AreTomo's anchor, with TX = TY = 0.000 *exactly*, so the shift sign and the
    TX/TY swap are unconstrained there — a four-way tie easy to misread as
    confirmation. That reasoning is still correct, but it assumed the high-tilt
    frames carry usable signal, and on dataset 9-A they do not: at +54 and +60
    the lamella is so foreshortened and dose-attenuated that the frame is
    essentially dark, and every convention scored in the noise (|corr| < 0.04,
    winner +0.003). Only -57 had real content. Correlating against an empty
    reference cannot discriminate anything, so ``high`` bought a stronger
    discriminant in principle and no discriminant at all in practice.

    ``low`` inverts the trade: the frames are bright and dense, so rotation --
    including the suspected 180 deg tilt-axis flip -- is measured cleanly, since
    ROT is constant across sections and is therefore determined at *any* tilt.
    The shift sign and swap stay weakly constrained near the anchor and must be
    settled separately (v1 reprojects at both signs and keeps the smaller
    residual). Getting rotation right first is the prerequisite anyway: with the
    axis 180 deg out, no shift convention can score well.

    Takes the nominal-tilt -> section mapping from map_sections_to_tilts.
    """
    if strategy == "high":
        ranked = sorted(mapping.items(), key=lambda kv: -np.hypot(kv[1].tx, kv[1].ty))
    elif strategy == "low":
        ranked = sorted(mapping.items(), key=lambda kv: abs(kv[0]))
    else:
        raise ValueError(f"Unknown tilt-selection strategy {strategy!r}")

    chosen = []
    for nominal, sec in ranked[:n]:
        chosen.append(nominal)
        logger.info(
            "  will check tilt %+6.1f (SEC %d)  |TX,TY| = %8.1f px",
            nominal, sec.sec, np.hypot(sec.tx, sec.ty),
        )
    if strategy == "low":
        logger.info(
            "  low-tilt strategy: rotation is well determined here, but the "
            "shift sign/swap are weakly constrained near AreTomo's anchor"
        )
    return sorted(chosen)


def _check_one_tilt(args, aln: AlnFile, posfiles: Dict[float, str],
                    tilt: float, sec: SectionAlignment):
    """
    Run the v0 geometry check at one tilt.

    Returns a list of (Convention, score) sorted best first, or None if the
    check could not be run. Returns [] when no aligned stack was supplied.
    """
    logger.info("=" * 70)
    logger.info("Checking tilt %.1f deg (%s)", tilt, os.path.basename(posfiles[tilt]))

    logger.info(
        "Section SEC=%d ROT=%.4f GMAG=%.5f TX=%.3f TY=%.3f TILT=%.2f",
        sec.sec, sec.rot, sec.gmag, sec.tx, sec.ty, sec.tilt,
    )
    logger.info(
        "Reprojection angle = %.2f deg (.aln TILT, which already includes the "
        "-TiltCor offset; nominal stage tilt is %.2f)",
        sec.tilt + args.tilt_offset, tilt,
    )

    pos = load_positions(posfiles[tilt])
    canvas_shape = pos["index_map_shape"]
    logger.info("Canvas (from index_map) = %s", canvas_shape)

    # Rebuild stitch.py's canvas from the arguments it was given and check it
    # against the canvas actually produced. A mismatch here means the --roi /
    # --pixel-size / --binning arguments do not describe this dataset.
    tile_files = find_tile_files(args.tiles_dir) if args.tiles_dir else {}
    if tile_files and tilt in tile_files:
        with mrcfile.open(tile_files[tilt], permissive=True, header_only=True) as m:
            raw_tile_shape = (int(m.header.ny), int(m.header.nx))
            n_tiles = int(m.header.nz)
        logger.info("Tile stack %s: %d tiles of %s", os.path.basename(tile_files[tilt]), n_tiles, raw_tile_shape)
    else:
        raw_tile_shape = tuple(args.tile_shape)
        logger.info("No tile stack found; using --tile-shape %s", raw_tile_shape)

    geom = MontageGeometry.from_stitch_args(
        roi=args.roi,
        pixel_size=args.pixel_size,
        binning=args.binning,
        raw_tile_shape=raw_tile_shape,
        rotate=args.rotate,
    )
    logger.info(
        "Reconstructed canvas = %s, canvas pixel size = %.4f A, tile on canvas = %s",
        geom.canvas_shape, geom.canvas_pixel_size, geom.tile_shape,
    )
    if tuple(geom.canvas_shape) != tuple(canvas_shape):
        logger.error(
            "Canvas mismatch: reconstructed %s but index_map is %s. "
            "Check --roi / --pixel-size / --binning against 02_stitch.sh.",
            geom.canvas_shape, canvas_shape,
        )
        return None
    logger.info("Canvas reconstruction matches index_map exactly.")

    if aln.raw_size[:2] != (canvas_shape[1], canvas_shape[0]):
        logger.warning(
            "aln RawSize %s does not match canvas %s (as nx, ny). "
            "The .aln may be from a different stitch run.",
            aln.raw_size[:2], (canvas_shape[1], canvas_shape[0]),
        )

    corners = geom.tile_corners_rc(pos["positions"])
    logger.info(
        "Tile footprints span rows %.0f..%.0f, cols %.0f..%.0f on the canvas",
        corners[..., 0].min(), corners[..., 0].max(),
        corners[..., 1].min(), corners[..., 1].max(),
    )

    if not args.aligned_stack:
        logger.warning(
            "No --aligned-stack given, so the convention cannot be verified. "
            "Run 04_AreTomo_nopatch.sh and pass its aligned_tiltseries_*.mrc."
        )
        return []

    # ---- Ground-truth check against AreTomo's own aligned stack ----
    with mrcfile.mmap(args.aligned_stack, permissive=True, mode="r") as m:
        # AreTomo drops dark frames, so the aligned stack is indexed by
        # position among the surviving sections in SEC order -- not by SEC
        # itself and not by tilt angle.
        surviving = sorted(x.sec for x in aln.sections)
        zi = surviving.index(sec.sec)
        if zi >= m.data.shape[0]:
            logger.error(
                "Aligned stack has %d slices but section SEC=%d is at position "
                "%d. Stack and .aln disagree.", m.data.shape[0], sec.sec, zi,
            )
            return None
        reference = np.asarray(m.data[zi], dtype=np.float32)
        out_bin = args.out_bin
    logger.info("Aligned stack slice %d, shape %s, -OutBin %g", zi, reference.shape, out_bin)

    # What gets correlated has to be actual image content. An earlier version
    # used a binary tile-coverage mask derived from index_map, which cannot
    # discriminate anything: 99.9% of the pixels it scores are exactly 1.0, so
    # the vector is constant, its variance is ~0, and Pearson r comes out in the
    # noise (|r| < 0.01) for *every* convention no matter how right or wrong the
    # geometry is. Three successive searches -- 8 variants, then 32, at high
    # tilt and then low -- all returned noise for this reason alone.
    #
    # The inpainted montage stack is the exact input AreTomo consumed, one slice
    # per tilt on the canvas, in ascending tilt order, so its z-index is SEC.
    # Correlating that against AreTomo's own aligned slice is a true
    # ground-truth test: the right convention scores ~0.92.
    if args.canvas_stack:
        with mrcfile.mmap(args.canvas_stack, permissive=True, mode="r") as mm:
            if sec.sec >= mm.data.shape[0]:
                logger.error(
                    "Canvas stack has %d slices but SEC=%d. Wrong stack?",
                    mm.data.shape[0], sec.sec,
                )
                return None
            canvas_image = np.asarray(mm.data[sec.sec], dtype=np.float32)
        logger.info(
            "Canvas content from %s slice %d (SEC), std %.3f",
            os.path.basename(args.canvas_stack), sec.sec, canvas_image.std(),
        )
    else:
        index_map = load_index_map(posfiles[tilt])
        canvas_image = (index_map < index_map.max()).astype(np.float32)
        logger.warning(
            "No --canvas-stack given, falling back to the binary coverage mask. "
            "This CANNOT discriminate between conventions -- the mask is "
            "constant wherever it is scored, so every correlation lands in the "
            "noise. Pass the inpainted montage stack for a meaningful result."
        )

    if args.search_conventions:
        logger.info("Scoring candidate conventions against the aligned stack:")
        results = search_conventions(canvas_image, reference, sec, out_bin)
        best, score = results[0]
        logger.info("Best convention: %s (correlation %+.4f)", best.label(), score)
        if not np.isfinite(score) or score < 0.3:
            logger.warning(
                "Best correlation is weak. The transform may be wrong in a way "
                "not covered by the enumerated variants, or the aligned stack "
                "may not correspond to this .aln."
            )
        conv = best
    else:
        conv = Convention(args.rot_sign, args.shift_sign, args.swap_txy,
                          args.rot_offset, args.flip_rows, args.flip_ty)
        logger.info("Using convention %s (no search requested)", conv.label())

    os.makedirs(args.output, exist_ok=True)
    aligned_corners = transform_corners(corners, sec, canvas_shape, conv, out_bin)
    overlay = os.path.join(args.output, f"v0_footprints_tilt{tilt:g}.png")
    plot_footprint_overlay(
        reference,
        aligned_corners,
        overlay,
        title=f"Tile footprints in aligned frame, tilt {tilt:g} deg, {conv.label()}",
    )

    warped = warp_canvas_to_aligned(canvas_image, sec, reference.shape, conv, out_bin)
    logger.info(
        "Final check: warped-canvas vs aligned-stack correlation = %+.4f",
        _normalised_correlation(warped, reference),
    )
    return results if args.search_conventions else [(conv, float("nan"))]


# ---------------------------------------------------------------------------
# v1: static reference -- reproject the tomogram, measure per-tile shifts
# ---------------------------------------------------------------------------


def reproject_volume(
    vol: np.ndarray, angle_deg: float, y_chunk: int = 256, order: int = 1
) -> np.ndarray:
    """
    Reproject a reconstructed volume at one tilt angle.

    `vol` is (nz, ny, nx) in the aligned frame, i.e. after `clip rotx`, with the
    tilt axis along y. Global alignment has already put the tilt axis on y, so
    every row y is an independent x-z problem -- the same property AreTomo
    exploits in GReproj.cu:mGBackProj. y is untouched, so chunking over it is
    exact rather than approximate: no interpolation crosses a chunk boundary.
    That keeps peak memory at one chunk instead of a second full copy of the
    volume, which matters here -- the montage canvas is large enough that the
    volume is GBs even at -OutBin 4.

    `angle_deg` is the tilt angle exactly as written in the .aln TILT column.
    The geometry below is a transcription of AreTomo's own kernels, so both the
    sign and the ray parametrisation are derived rather than guessed.

    Returns (ny, nx), the per-ray *mean* over in-volume samples, matching
    Recon/GForProj.cu:47-49.
    """
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {vol.shape}")
    nz, ny, nx = vol.shape

    # ---- AreTomo's projection geometry, transcribed ----
    #
    # AreTomo backprojects (Recon/GBackProj.cu:31-33) by sampling the sinogram
    # at
    #     xp = x*cos(TILT) + z*sin(TILT)
    # with x, z centred on the volume. TILT is the angle straight out of
    # CAlignParam (Recon/CTomoWbp.cpp:47-52) -- the same array CAlignFile writes
    # to the .aln TILT column (MrcUtil/CAlignFile.cpp:165-169). Nothing in that
    # chain negates it.
    #
    # Recon/GForProj.cu:25-37 inverts that relation. Expanding its fZStartp
    # offset, which is the part that is easy to miss, the ray feeding output
    # column xp reduces to
    #     x = xp/cos(TILT) - tan(TILT)*z        z = the slab's own z planes
    # i.e. a SHEAR, not a rotation. The distinction is not cosmetic. AreTomo
    # walks the ray from `fZStartp = -fXp*fSin/fCos - 0.5*iRayLength`, so the
    # sampled span is re-centred per output column onto where that column's ray
    # actually crosses the slab, and its length is iRayLength = VolZ/cos rather
    # than VolZ. Rotating the slab inside its own nz-tall box instead -- the
    # obvious reading of "rotate then sum" -- throws away every ray whose
    # crossing point falls outside |z| < nz/2. For this canvas (nz = 150,
    # nx = 2186) that is almost the entire field beyond |x| ~ (nz/2)/tan(TILT):
    # at 60 deg only |x| < ~87 px of a 1093 px half-width survives, so the
    # reference is essentially blank at high tilt. Measured, not theorised.
    #
    # z is AreTomo's *internal* z. The volume on disk has been through
    # `clip rotx`, which reverses the depth axis: internal z -> (nz - 1 - z),
    # verified directly against IMOD's clip and identical to what AreTomo's own
    # -FlipVol does (CProcessThread.cpp:505-515, `pfDstFrm = GetFrame(iEndOldY
    # - y)`). Substituting z_disk = -z_internal flips the shear term, giving
    #     x = xp/cos(TILT) + tan(TILT)*z_disk
    # which is the shear at *minus* the tilt angle. That negation is the rotx
    # depth flip, not a free parameter. It is the one convention v0 could not
    # reach, because the tilt angle never enters the 2D alignment transform.
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    if abs(cos_t) < 1e-3:
        raise ValueError(f"Tilt angle {angle_deg} deg is too close to 90 deg")
    tan_t = np.tan(theta)

    # Sample one plane of the slab per output row: z is the identity, so the
    # depth axis is not interpolated at all. AreTomo takes VolZ/cos samples at
    # unit steps along the ray, which oversamples the same planes by 1/cos; the
    # per-ray mean makes the two equivalent.
    M = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [tan_t, 0.0, 1.0 / cos_t]])
    centre = np.array([(nz - 1) / 2.0, 0.0, (nx - 1) / 2.0])
    offset = centre - M @ centre

    # Per-ray sample count, for AreTomo's fInt/iCount normalisation. x is the
    # only axis that can leave the volume and the map does not involve y, so
    # the count is the same for every row and is cheaper computed analytically
    # than by transforming a volume of ones.
    kk = np.arange(nz)[:, None]
    jj = np.arange(nx)[None, :]
    x_in = tan_t * (kk - centre[0]) + (jj - centre[2]) / cos_t + centre[2]
    counts = ((x_in >= 0.0) & (x_in <= nx - 1)).sum(axis=0).astype(np.float32)
    np.maximum(counts, 1.0, out=counts)

    out = np.zeros((ny, nx), dtype=np.float32)
    for y0 in range(0, ny, y_chunk):
        y1 = min(y0 + y_chunk, ny)
        sub = np.asarray(vol[:, y0:y1, :], dtype=np.float32)
        sheared = affine_transform(
            sub, M, offset=offset, order=order,
            mode="constant", cval=0.0, prefilter=False,
        )
        out[y0:y1] = sheared.sum(axis=0) / counts
    return out


@dataclass
class TileShift:
    """One per-(tilt, tile) residual measurement."""

    tilt: float
    tile: int
    drow: float  # aligned-frame shift, reprojection pixels
    dcol: float
    dx_um: float  # mapped back to a position update
    dy_um: float
    weight: float  # masked correlation at the fitted shift, in [-1, 1]
    n_px: int


def measure_tile_shifts(
    canvas_image: np.ndarray,
    index_map: np.ndarray,
    reference: np.ndarray,
    sec: SectionAlignment,
    tilt: float,
    conv: Convention,
    recon_bin: float,
    canvas_apix: float,
    min_px: int = 5000,
    upsample: int = 10,
    max_shift: float = 200.0,
) -> List[TileShift]:
    """
    Mode A: measure a 2D residual per tile against a reprojected reference.

    For each tile, the montage pixels belonging to it (`index_map == tile`) are
    warped into the aligned frame along with their mask, and matched against the
    reprojection by masked normalised cross-correlation (Padfield), which is
    what skimage's phase_cross_correlation implements when given masks. The mask
    matters: `index_map` is a *partition* of the canvas, so an interior tile is
    carved down to a Voronoi-like sliver and an unmasked correlation would be
    dominated by neighbouring tiles' pixels.

    The measured shift is mapped back to a position update through the inverse
    of the very same alignment matrix used for the forward warp, rather than by
    reconstructing R(-ROT) by hand. That is deliberate -- it makes the round
    trip self-consistent by construction and removes any opportunity for a
    fresh sign error on the way back.
    """
    from skimage.registration import phase_cross_correlation

    M, _ = alignment_matrix(sec, index_map.shape, conv, recon_bin)
    Minv = np.linalg.inv(M)

    # index_map is a partition whose fill value is n_tiles (stitch.py), so the
    # tile labels are exactly 0 .. max-1.
    n_tiles = int(index_map.max())
    ref = np.asarray(reference, dtype=np.float32)
    ref_mask = np.isfinite(ref) & (ref != 0)

    out: List[TileShift] = []
    n_rejected = [0]
    for tile in range(n_tiles):
        mask = index_map == tile
        n_canvas = int(mask.sum())
        if n_canvas < min_px:
            logger.debug("tile %2d: only %d canvas px, skipped", tile, n_canvas)
            continue

        tile_img = np.where(mask, canvas_image, 0.0).astype(np.float32)
        w_img = warp_canvas_to_aligned(tile_img, sec, ref.shape, conv, recon_bin)
        w_msk = warp_canvas_to_aligned(
            mask.astype(np.float32), sec, ref.shape, conv, recon_bin
        ) > 0.5
        n_px = int(w_msk.sum())
        if n_px < min_px // int(max(recon_bin, 1) ** 2) or n_px < 200:
            logger.debug("tile %2d: only %d px in aligned frame, skipped", tile, n_px)
            continue

        # Bound the search to a window around the tile's own footprint.
        #
        # This is not an optimisation, it is required for correctness. Padfield
        # masked NCC normalises by the overlap area, so as a displacement grows
        # and the overlap shrinks, the normalised score is *inflated* -- the
        # classic failure mode of masked correlation. Searching a small tile
        # mask against the full reference therefore lets the peak land almost
        # anywhere: the first run of v1 returned median residuals of 1101 px
        # (3.0 um) against a tile that is only 1.97 x 1.40 um, i.e. shifts
        # larger than the tile itself.
        #
        # Cropping to footprint + max_shift keeps the overlap large everywhere
        # in the search window, and makes the correlation far cheaper too.
        rows, cols = np.nonzero(w_msk)
        margin = int(round(max_shift / recon_bin)) + 2
        r0 = max(0, int(rows.min()) - margin)
        r1 = min(ref.shape[0], int(rows.max()) + 1 + margin)
        c0 = max(0, int(cols.min()) - margin)
        c1 = min(ref.shape[1], int(cols.max()) + 1 + margin)
        sl = (slice(r0, r1), slice(c0, c1))

        res = phase_cross_correlation(
            ref[sl], w_img[sl],
            reference_mask=ref_mask[sl], moving_mask=w_msk[sl],
            upsample_factor=upsample,
        )
        shift = np.asarray(res[0] if isinstance(res, tuple) else res, dtype=float)

        # Even inside the window, reject anything beyond the physical bound
        # rather than reporting a number that cannot be a per-tile residual.
        mag_px = float(np.hypot(*shift))
        if mag_px > max_shift / recon_bin:
            logger.debug(
                "tile %2d: |shift| %.1f px exceeds --max-shift, rejected",
                tile, mag_px,
            )
            n_rejected[0] += 1
            continue

        # Quality weight: masked correlation after applying the fitted shift.
        shifted = affine_transform(
            w_img, np.eye(2), offset=-shift, order=1,
            mode="constant", cval=0.0, prefilter=False,
        )
        smsk = affine_transform(
            w_msk.astype(np.float32), np.eye(2), offset=-shift, order=1,
            mode="constant", cval=0.0, prefilter=False,
        ) > 0.5
        both = smsk & ref_mask
        weight = (
            _normalised_correlation(np.where(both, shifted, 0.0), np.where(both, ref, 0.0))
            if both.sum() > 200 else float("nan")
        )

        d_canvas = Minv @ shift  # reprojection px -> canvas px, (row, col)
        # Positions in the HDF5 are (x=col, y=row) in microns.
        dx_um = d_canvas[1] * canvas_apix / 1e4
        dy_um = d_canvas[0] * canvas_apix / 1e4

        out.append(TileShift(
            tilt=tilt, tile=tile,
            drow=float(shift[0]), dcol=float(shift[1]),
            dx_um=float(dx_um), dy_um=float(dy_um),
            weight=float(weight), n_px=n_px,
        ))
    if n_rejected[0]:
        logger.info(
            "  %d tile(s) rejected for exceeding --max-shift %.0f canvas px",
            n_rejected[0], max_shift,
        )
    return out


def _shift_summary(shifts: List[TileShift]) -> Tuple[float, float]:
    """(median |shift| in reprojection px, median weight). NaN-safe."""
    if not shifts:
        return float("nan"), float("nan")
    mag = [float(np.hypot(s.drow, s.dcol)) for s in shifts]
    w = [s.weight for s in shifts if np.isfinite(s.weight)]
    return float(np.median(mag)), float(np.median(w)) if w else float("nan")


def run_measure(args) -> int:
    """
    v1 driver: reproject a static reconstruction and measure per-tile shifts.

    Report only -- nothing is written back to the HDF5. The point is to get real
    per-tile numbers out of a static volume before v2's leave-one-out machinery
    exists.

    The reprojection angle sign is NOT decided here. An earlier version tried to
    settle it empirically, by reprojecting at both signs and keeping the smaller
    residuals; that failed twice over. Median |shift| rewards degenerate zeros,
    so whichever sign rejects more tiles wins, and the reprojection it was
    scoring was blank away from the tilt axis anyway. The sign is now derived
    from AreTomo's own kernels plus the `clip rotx` depth flip -- see
    reproject_volume() -- and reproject_volume() takes the .aln TILT column
    verbatim, with the flip already folded into its shear matrix.
    --settle-tilt-sign therefore reprojects at the deliberately wrong sign as a
    control: a "win" for it means this measurement is broken, not the geometry.

    One caveat on everything reported here: the reference volume contains the
    tile being matched, which is self-reinforcing and drives residuals towards
    zero. Magnitudes are indicative, not trustworthy, until v2.
    """
    if not args.recon:
        logger.error("v1 needs --recon (the rotx'd reconstruction).")
        return 2
    if not args.canvas_stack:
        # v0 can fall back to the coverage mask (uselessly, but it warns and
        # runs); v1 has no fallback -- the montage pixels ARE what gets
        # correlated. Fail here rather than several minutes into a reprojection
        # with a TypeError on None.
        logger.error(
            "v1 needs --canvas-stack (the inpainted montage stack AreTomo "
            "consumed, e.g. stitched_motioncorr/Montage_<ds>_inpainted.mrc)."
        )
        return 2

    aln = parse_aln(args.aln)
    posfiles = find_position_files(args.positions_dir)
    mapping, _ = map_sections_to_tilts(aln, list(posfiles))

    if args.tilt:
        tilts = [min(posfiles, key=lambda t: abs(t - w)) for w in args.tilt]
        tilts = [t for t in tilts if t in mapping]
    else:
        tilts = select_check_tilts(
            mapping, n=args.n_check_tilts, strategy=args.check_tilt_strategy
        )
    if not tilts:
        logger.error("No usable tilts selected.")
        return 2

    conv = Convention(args.rot_sign, args.shift_sign, args.swap_txy,
                      args.rot_offset, args.flip_rows, args.flip_ty)
    logger.info("Using convention %s", conv.label())

    canvas_apix = args.pixel_size * args.binning

    with mrcfile.mmap(args.recon, permissive=True, mode="r") as mm:
        vol_shape = mm.data.shape
        logger.info("Reconstruction %s shape %s", os.path.basename(args.recon), vol_shape)
        if len(vol_shape) != 3:
            logger.error("Reconstruction is not 3D: %s", vol_shape)
            return 2
        # After `clip rotx` the order is (nz, ny, nx) with nz the thin axis.
        # AreTomo's raw output is (x, z, y); if rotx was skipped the thin axis
        # lands elsewhere and every reprojection would be silently wrong.
        if vol_shape[0] > min(vol_shape[1], vol_shape[2]):
            logger.error(
                "Volume looks un-rotx'd: shape %s has a thick first axis. "
                "Expected (nz, ny, nx) with nz smallest -- run `clip rotx`.",
                vol_shape,
            )
            return 2

        signs = (1.0, -1.0) if args.settle_tilt_sign else (1.0,)
        per_sign = {}
        for sign in signs:
            all_shifts: List[TileShift] = []
            for tilt in tilts:
                sec = mapping[tilt]
                angle = sign * (sec.tilt + args.tilt_offset)
                logger.info(
                    "Reprojecting tilt %+.1f (SEC %d) at %+.2f deg%s",
                    tilt, sec.sec, angle,
                    "" if sign > 0 else "   [DELIBERATELY WRONG sign, control]",
                )
                ref = reproject_volume(mm.data, angle, y_chunk=args.y_chunk)

                with h5py.File(posfiles[tilt], "r") as h:
                    index_map = h["index_map"][:]
                with mrcfile.mmap(args.canvas_stack, permissive=True, mode="r") as cs:
                    canvas_image = np.asarray(cs.data[sec.sec], dtype=np.float32)

                shifts = measure_tile_shifts(
                    canvas_image, index_map, ref, sec, tilt, conv,
                    args.recon_bin, canvas_apix, upsample=args.upsample,
                    max_shift=args.max_shift,
                )
                med, wmed = _shift_summary(shifts)
                logger.info(
                    "  %2d tiles measured, median |shift| = %.2f px (%.1f A), "
                    "median weight %.3f",
                    len(shifts), med, med * args.recon_bin * canvas_apix, wmed,
                )
                for s in shifts:
                    logger.info(
                        "    tile %2d  d=(%+7.2f,%+7.2f) px  ->  (%+.4f, %+.4f) um  "
                        "w=%+.3f  n=%d",
                        s.tile, s.drow, s.dcol, s.dx_um, s.dy_um, s.weight, s.n_px,
                    )
                all_shifts.extend(shifts)
            per_sign[sign] = all_shifts

    logger.info("=" * 70)
    for sign, shifts in per_sign.items():
        med, wmed = _shift_summary(shifts)
        logger.info(
            "Reprojection sign %+g: %d measurements, median |shift| = %.2f px, "
            "median weight %.3f", sign, len(shifts), med, wmed,
        )

    if args.settle_tilt_sign and len(per_sign) == 2:
        m_pos, _ = _shift_summary(per_sign[1.0])
        m_neg, _ = _shift_summary(per_sign[-1.0])
        if not (np.isfinite(m_pos) and np.isfinite(m_neg)):
            logger.warning("Could not compare reprojection signs -- no valid measurements.")
        else:
            best = "+" if m_pos <= m_neg else "-"
            ratio = max(m_pos, m_neg) / max(min(m_pos, m_neg), 1e-9)
            logger.info(
                "TILT SIGN: %s wins (median |shift| %.2f vs %.2f px, ratio %.2fx)",
                best, min(m_pos, m_neg), max(m_pos, m_neg), ratio,
            )
            if ratio < 1.2:
                logger.warning(
                    "The two signs are within %.0f%% of each other, which is not a "
                    "decision. Most likely the residuals are dominated by noise "
                    "rather than by the reprojection geometry -- check the median "
                    "weight before trusting either.", (ratio - 1) * 100,
                )
    return 0


def run_geometry_check(args) -> int:
    """
    v0 driver: check the coordinate chain, optionally at several tilts.

    When searching conventions, the winner is decided by consensus across tilts
    rather than by a single tilt's score, because no single tilt constrains
    every degree of freedom (see select_check_tilts).
    """
    aln = parse_aln(args.aln)
    posfiles = find_position_files(args.positions_dir)
    logger.info("Found %d per-tilt position files", len(posfiles))

    # Join .aln sections to tilts by SEC. This must happen before anything
    # else -- joining on the TILT column is off by the -TiltCor offset.
    mapping, inferred_offset = map_sections_to_tilts(aln, list(posfiles))

    if args.tilt:
        wanted = [min(posfiles, key=lambda t: abs(t - want)) for want in args.tilt]
        tilts = []
        for t in wanted:
            if t in mapping:
                tilts.append(t)
            else:
                logger.warning("Tilt %+.1f is a dark frame with no alignment; skipping", t)
    elif args.search_conventions:
        logger.info("Selecting tilts for the convention search:")
        tilts = select_check_tilts(
            mapping, n=args.n_check_tilts, strategy=args.check_tilt_strategy
        )
    else:
        tilts = [min(mapping, key=abs)]

    if not tilts:
        logger.error("No usable tilts to check")
        return 1

    per_tilt: Dict[float, List[Tuple[Convention, float]]] = {}
    for tilt in tilts:
        res = _check_one_tilt(args, aln, posfiles, tilt, mapping[tilt])
        if res is None:
            return 1
        if res:
            per_tilt[tilt] = res

    if not args.search_conventions or not per_tilt:
        return 0

    # ---- Consensus across tilts ----
    logger.info("=" * 70)
    logger.info("Convention consensus across %d tilt(s):", len(per_tilt))
    totals: Dict[Convention, List[float]] = {}
    for scores in per_tilt.values():
        for conv, score in scores:
            totals.setdefault(conv, []).append(score if np.isfinite(score) else -1.0)

    ranked = sorted(totals.items(), key=lambda kv: -float(np.mean(kv[1])))
    for conv, scores in ranked:
        logger.info(
            "  %-28s mean %+.4f   per-tilt %s",
            conv.label(),
            float(np.mean(scores)),
            " ".join(f"{s:+.3f}" for s in scores),
        )

    best, best_scores = ranked[0]
    runner_up_mean = float(np.mean(ranked[1][1])) if len(ranked) > 1 else -1.0
    best_mean = float(np.mean(best_scores))
    logger.info("-" * 70)
    logger.info("WINNER: %s   mean correlation %+.4f", best.label(), best_mean)

    if best_mean < 0.3:
        logger.warning(
            "Winning correlation is weak (%+.4f). The transform may be wrong in "
            "a way not covered by the enumerated variants, or the aligned stack "
            "may not correspond to this .aln.", best_mean,
        )
    elif best_mean - runner_up_mean < 0.05:
        logger.warning(
            "Winner (%+.4f) is barely ahead of runner-up %s (%+.4f). The check "
            "is not decisive — inspect the overlay PNGs before trusting it.",
            best_mean, ranked[1][0].label(), runner_up_mean,
        )
    else:
        logger.info(
            "Decisive: runner-up %s scores %+.4f. Hard-code this as the default "
            "Convention and proceed to v1.",
            ranked[1][0].label(), runner_up_mean,
        )
    return 0


def parse_commandline(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--positions-dir", required=True,
                   help="Directory of *_refined_positions.h5 from stitch.py")
    p.add_argument("--aln", required=True,
                   help="AreTomo .aln file (from a run WITHOUT -Patch)")
    p.add_argument("--tiles-dir", default=None,
                   help="Directory of per-tilt tile stacks (motion_corrected)")
    p.add_argument("--canvas-stack", default=None,
                   help="Stitched montage stack AreTomo consumed, typically "
                        "stitched_motioncorr/Montage_<ds>_inpainted.mrc. One "
                        "slice per tilt on the canvas, indexed by SEC. This is "
                        "the image content the convention search correlates; "
                        "without it the search falls back to a coverage mask "
                        "that cannot discriminate anything.")
    p.add_argument("--aligned-stack", default=None,
                   help="AreTomo aligned tilt series (-VolZ 0 output). Ground "
                        "truth for the convention check.")
    p.add_argument("-o", "--output", default="projmatch_v0",
                   help="Output directory for overlays")

    # Must match 02_stitch.sh
    p.add_argument("--roi", nargs=4, type=float, default=[-5500, 12000, -6000, 12000],
                   help="stitch.py --ROI, unbinned tile pixels (x0 x1 y0 y1)")
    p.add_argument("--pixel-size", type=float, default=3.426,
                   help="Unbinned tile pixel size in Angstroms")
    p.add_argument("--binning", type=int, default=2, help="stitch.py binning")
    p.add_argument("--rotate", type=int, default=90, help="stitch.py --rotate")
    p.add_argument("--tile-shape", nargs=2, type=int, default=[4092, 5760],
                   help="Raw tile (rows, cols); only used if --tiles-dir is absent")

    p.add_argument("--out-bin", type=float, default=2.0,
                   help="AreTomo -OutBin used for the aligned stack / volume")
    p.add_argument("--tilt-offset", type=float, default=0.0,
                   help="EXTRA tilt offset in degrees, added to the .aln TILT "
                        "column. Normally 0: AreTomo's -TiltCor offset is "
                        "already folded into TILT (the script infers and "
                        "reports it). Only set this if you know TILT is raw.")
    p.add_argument("--tilt", type=float, nargs="+", default=None,
                   help="Tilt angle(s) to check. Default without "
                        "--search-conventions is the tilt closest to zero; with "
                        "it, --n-check-tilts tilts picked by "
                        "--check-tilt-strategy.")
    p.add_argument("--check-tilt-strategy", choices=("low", "high"), default="low",
                   help="Which tilts the convention search uses. 'low' (default) "
                        "takes those nearest zero tilt, ~(-3, 0, +3): bright, "
                        "dense frames that pin down rotation. 'high' takes the "
                        "largest-shift tilts, which constrain the shift sign "
                        "better in theory but on 9-A were too dark to correlate "
                        "at all (see select_check_tilts).")
    p.add_argument("--n-check-tilts", type=int, default=3,
                   help="How many tilts to search over for the consensus")

    p.add_argument("--search-conventions", action="store_true",
                   help="Enumerate sign conventions and score each against the "
                        "aligned stack")
    p.add_argument("--rot-sign", type=int, default=-1, choices=(-1, 1))
    p.add_argument("--shift-sign", type=int, default=-1, choices=(-1, 1))
    p.add_argument("--swap-txy", action="store_true")
    p.add_argument("--rot-offset", type=float, default=0.0,
                   help="Degrees added to ROT. 180 flips the tilt axis.")
    p.add_argument("--flip-ty", action="store_true",
                   help="Negate the TY shift component only, leaving TX. This "
                        "is what an MRC(y-up) vs numpy(row-down) relabelling "
                        "does to the translation; --shift-sign flips both "
                        "components and --swap-txy swaps them, so neither can "
                        "express it.")
    p.add_argument("--flip-rows", action="store_true",
                   help="Mirror the row axis about the canvas centre. This is a "
                        "reflection (det < 0) and is NOT reachable by any "
                        "combination of the sign/swap/offset flags, which are "
                        "all proper rotations.")

    # ---- v1: static-reference measurement ----
    p.add_argument("--measure", action="store_true",
                   help="Run v1: reproject --recon and measure per-tile shifts. "
                        "Report only; nothing is written back to the HDF5.")
    p.add_argument("--recon", default=None,
                   help="Rotx'd reconstruction to reproject, e.g. "
                        "AreTomo_nopatch/recon_nopatch_bin4.mrc. Must be in "
                        "(nz, ny, nx) order -- v1 asserts the thin axis is first.")
    p.add_argument("--recon-bin", type=float, default=4.0,
                   help="AreTomo -OutBin used for --recon. Distinct from "
                        "--out-bin, which applies to the aligned stack: the "
                        "reconstruction is deliberately coarser (the montage "
                        "canvas makes a bin-2 volume 78 GB).")
    p.add_argument("--settle-tilt-sign", action="store_true",
                   help="Diagnostic only. The sign is no longer searched: it is "
                        "derived from AreTomo's own kernels plus the `clip rotx` "
                        "depth flip, see reproject_volume(). This flag also "
                        "reprojects at -TILT as a deliberately-wrong control. "
                        "Note the comparison is scored by median |shift|, which "
                        "rewards degenerate zeros -- treat a win for the control "
                        "as evidence the measurement is broken, not the sign.")
    p.add_argument("--max-shift", type=float, default=200.0,
                   help="Largest believable per-tile residual, in CANVAS px "
                        "(200 px = 137 nm at 6.852 A). The masked NCC is "
                        "cropped to the tile footprint plus this margin, and "
                        "peaks beyond it are rejected. Required for "
                        "correctness: masked NCC normalises by overlap area, "
                        "so an unbounded search is biased towards large, "
                        "small-overlap displacements.")
    p.add_argument("--upsample", type=int, default=10,
                   help="Subpixel upsampling factor for the masked NCC")
    p.add_argument("--y-chunk", type=int, default=256,
                   help="Rows per reprojection chunk. y is untouched by the "
                        "shear, so chunking is exact, not an approximation.")

    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_commandline(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if args.measure:
        return run_measure(args)
    return run_geometry_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
