#!/usr/bin/env python3
"""
Project 3D tomogram coordinates back onto individual montage tiles.

A square-beam montage tomogram is reconstructed from a *stitched* tilt series,
so a 3D pick has no direct relationship to the raw data: the pixel it came from
lives somewhere in one of ~20 tiles making up one tilt, and which tile that is
changes from tilt to tilt. This module inverts the whole chain, taking a point
(X, Y, Z) in the reconstruction to (tilt, tile, z-slice, x, y) in the
per-tilt tile stacks written by ``beam_mask_motioncorr --stack-output``.

The chain, in order, and where each link came from
--------------------------------------------------
1. **tomogram -> aligned frame.** Inverting the shear in
   ``refine_montage_projmatch.reproject_volume``::

       col_al = cx + (X - cx) cos(theta) - (Z - cz) sin(theta)
       row_al = Y

   ``theta`` is the ``.aln`` TILT column *as written*, which already includes
   AreTomo's ``-TiltCor`` offset, and the sign of the Z term already carries
   the depth-axis flip that ``clip rotx`` applies. Neither is a free parameter;
   see reproject_volume's own derivation for the AreTomo source citations.

2. **aligned -> canvas.** The inverse of
   ``refine_montage_projmatch.alignment_matrix``, whose sign convention
   (``rot-1 shift-1``) was settled against AreTomo's own aligned stack.

3. **canvas -> tile.** ``MontageGeometry`` places tile *t* from its position in
   microns. A point may fall inside several overlapping tiles; ``index_map``
   decides whether it is covered by real data at all, and among the covering
   tiles we take the one that holds the point furthest from any tile edge.

4. **canvas pixel -> raw tile pixel.** Undo ``stitch.py --rotate`` and the
   stitch binning.

5. **tile -> z-slice.** ``index_map`` labels run 0..n_selected-1, *not* over the
   rows of ``Refined_positions``; map through ``flatnonzero(tile_selection)``.
   The result indexes the per-tilt stack directly, because that stack has one
   slice per position row (verified for all 41 tilts of dataset 9-A).

Two traps worth stating explicitly, because both are silent
-----------------------------------------------------------
* ``stitch.py --rotate`` is **not** recorded in the HDF5. Dataset 9-A was
  stitched with ``--rotate 90`` even though the current ``02_stitch.sh`` has
  that flag commented out and stitch.py defaults to 0. Getting it wrong
  displaces every tile by (2880-2046)/2 = 417 canvas pixels. ``rotate="auto"``
  measures it from ``index_map`` instead of trusting an argument.
* ``index_map`` labels index the *selected* tiles. When ``tile_selection`` has
  any False entry (it does, on more than half the tilts of 9-A) label != row of
  ``Refined_positions`` and everything downstream lands on the wrong tile.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from .refine_montage_projmatch import (
    AlnFile,
    Convention,
    MontageGeometry,
    SectionAlignment,
    alignment_matrix,
    find_position_files,
    map_sections_to_tilts,
    parse_aln,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-tilt bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class TiltInfo:
    """Everything needed to place a point onto the tiles of one tilt."""

    tilt: float  # nominal stage tilt, from the filename
    sec: SectionAlignment  # the .aln row joined by SEC
    positions_um: np.ndarray  # (n_tiles, 2), full length, (x, y) microns
    tile_selection: np.ndarray  # (n_tiles,) bool
    h5path: str
    stack_path: Optional[str] = None  # per-tilt tile stack MRC
    _index_map: Optional[np.ndarray] = field(default=None, repr=False)
    _beam_masks: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def selected(self) -> np.ndarray:
        """Rows of ``positions_um`` that ``index_map`` labels 0, 1, 2, ..."""
        return np.flatnonzero(self.tile_selection)

    def index_map(self) -> np.ndarray:
        """Load (and cache) the canvas partition. ~150 MB per tilt."""
        if self._index_map is None:
            with h5py.File(self.h5path, "r") as h:
                self._index_map = h["index_map"][:]
        return self._index_map

    def beam_masks(self) -> Optional[np.ndarray]:
        """
        Per-tile square-beam footprints, ``(n_selected, *tile_shape)`` bool.

        stitch.py stores these packed along the last axis
        (``np.packbits(..., axis=-1)`` at stitch.py:662) in tile-local canvas
        coordinates -- i.e. binned and already rotated -- and in *selected* tile
        order, like ``beam_shifts`` and unlike ``Refined_positions``.

        This matters for extraction, not just for tidiness. The illuminated
        square is smaller than the detector frame and sits off-centre in it, so
        a particle can be comfortably inside the tile rectangle and still have
        the beam edge running through its box. On dataset 9-A that was ~15% of
        boxes, and a beam edge inside a particle box is a far worse artefact
        than a missing observation.
        """
        if self._beam_masks is None:
            with h5py.File(self.h5path, "r") as h:
                if "beam_masks" not in h:
                    return None
                packed = h["beam_masks"][:]
                width = int(h["beam_masks_width"][()])
            self._beam_masks = np.unpackbits(packed, axis=-1)[..., :width].astype(bool)
        return self._beam_masks

    def drop_index_map(self) -> None:
        self._index_map = None
        self._beam_masks = None


# ---------------------------------------------------------------------------
# Assignment result
# ---------------------------------------------------------------------------


@dataclass
class TileHit:
    """One particle seen in one tile of one tilt."""

    tilt: float
    tile: int  # row of Refined_positions == z-slice of the per-tilt stack
    x: float  # raw (unbinned) tile pixel, column
    y: float  # raw (unbinned) tile pixel, row
    canvas_rc: Tuple[float, float]
    depth: float  # along-beam offset from the tomogram centre, unbinned px
    edge_margin: float  # distance to the nearest tile edge, raw px


# ---------------------------------------------------------------------------
# The projector
# ---------------------------------------------------------------------------


class MontageProjector:
    """
    Maps reconstruction coordinates onto the raw tiles of a square-beam montage.

    All 3D input is in *reconstruction voxels*, (X, Y, Z) = (column, row, slice)
    of the AreTomo volume after ``clip rotx`` -- i.e. exactly what
    ``model2point`` prints for a pick made in that volume.
    """

    def __init__(
        self,
        aln_path: str,
        positions_dir: str,
        recon_shape: Sequence[int],  # (nz, ny, nx) of the reconstruction
        roi: Sequence[float] = (-5500, 12000, -6000, 12000),
        pixel_size: float = 3.426,
        binning: int = 2,
        out_bin: int = 2,
        rotate: object = "auto",
        raw_tile_shape: Optional[Tuple[int, int]] = None,
        tile_dir: Optional[str] = None,
        conv: Convention = Convention(),
        extra_shift: Sequence[float] = (0.0, 0.0),
        handedness: int = 1,
        exclude_sections: Sequence[int] = (),
    ):
        self.aln = parse_aln(aln_path)
        self.recon_shape = tuple(int(x) for x in recon_shape)
        self.out_bin = int(out_bin)
        self.binning = int(binning)
        self.pixel_size = float(pixel_size)
        self.conv = conv
        self.extra_shift = np.asarray(extra_shift, dtype=float)
        self.handedness = int(handedness)

        # The tomogram is `binning * out_bin` coarser than a raw tile pixel.
        # Derive it from the binning factors, never from the ratio of recorded
        # pixel sizes: AreTomo was given -PixSize 6.87 for a canvas that is
        # really 6.852 A/px, so the header says 13.74 A when the truth is
        # 13.704 A. The geometry is exact; only the label is wrong.
        self.tomo_bin = self.binning * self.out_bin

        posfiles = find_position_files(positions_dir)
        mapping, offset = map_sections_to_tilts(self.aln, list(posfiles))
        self.tilt_offset = offset
        # Every tilt that has a position file, dark frames included. This is the
        # slice order of the stitched canvas stack and of the .aln SEC index --
        # `self.tilts` below holds only the sections AreTomo kept, so the two
        # must not be confused when indexing into a stack.
        self.all_tilts: List[float] = sorted(posfiles)

        stacks = _find_tile_stacks(tile_dir) if tile_dir else {}
        if raw_tile_shape is None:
            raw_tile_shape = _probe_tile_shape(stacks)
        self.raw_tile_shape = tuple(int(x) for x in raw_tile_shape)

        self.tilts: Dict[float, TiltInfo] = {}
        for tilt, path in sorted(posfiles.items()):
            sec = mapping.get(tilt)
            if sec is None:
                logger.info("tilt %+6.1f: dark frame in the .aln, skipped", tilt)
                continue
            if sec.sec in exclude_sections:
                logger.warning("tilt %+6.1f: SEC %d excluded by request", tilt, sec.sec)
                continue
            with h5py.File(path, "r") as h:
                pos = h["Refined_positions"][:]
                sel = np.asarray(h["tile_selection"][:], dtype=bool)
                canvas_shape = h["index_map"].shape
            self.tilts[tilt] = TiltInfo(
                tilt=tilt, sec=sec, positions_um=pos, tile_selection=sel,
                h5path=path, stack_path=stacks.get(tilt),
            )
            self.canvas_shape = tuple(int(x) for x in canvas_shape)

        if not self.tilts:
            raise ValueError("No tilts survived the .aln join -- check the inputs.")

        if str(rotate) == "auto":
            rotate = self._detect_rotation(roi)
        self.rotate = int(rotate)

        self.geom = MontageGeometry.from_stitch_args(
            roi=roi, pixel_size=pixel_size, binning=binning,
            raw_tile_shape=self.raw_tile_shape, rotate=self.rotate,
        )
        if tuple(self.geom.canvas_shape) != self.canvas_shape:
            raise ValueError(
                f"Canvas mismatch: --roi/--pixel-size/--binning reconstruct "
                f"{self.geom.canvas_shape} but index_map is {self.canvas_shape}. "
                f"Check them against 02_stitch.sh."
            )
        logger.info(
            "canvas %s, tile on canvas %s, rotate %d, tomogram binning %d",
            self.geom.canvas_shape, self.geom.tile_shape, self.rotate, self.tomo_bin,
        )

    # -- rotation detection -------------------------------------------------

    def _detect_rotation(self, roi: Sequence[float]) -> int:
        """
        Measure ``stitch.py --rotate`` from index_map rather than trusting it.

        The HDF5 does not record the rotation, and getting it wrong shifts every
        tile by half the difference of the tile's two side lengths -- 417 px for
        9-A, far more than any residual we care about. Each candidate rotation
        predicts a tile footprint; the right one puts the centroid of each
        index_map label near the centre of its predicted footprint. Only tiles
        whose footprint lies wholly on the canvas are scored, because a clipped
        footprint biases the centroid regardless of the rotation.
        """
        tilt = min(self.tilts, key=abs)
        info = self.tilts[tilt]
        imap = info.index_map()
        full = info.selected
        scores = {}
        for rot in (0, 90, 180, 270):
            geom = MontageGeometry.from_stitch_args(
                roi=roi, pixel_size=self.pixel_size, binning=self.binning,
                raw_tile_shape=self.raw_tile_shape, rotate=rot,
            )
            org = geom.tile_origin_rc(info.positions_um)
            h, w = geom.tile_shape
            resid = []
            for label in range(int(imap.max())):
                r0, c0 = org[full[label]]
                if r0 < 0 or c0 < 0 or r0 + h > imap.shape[0] or c0 + w > imap.shape[1]:
                    continue
                m = imap == label
                if not m.any():
                    continue
                rr, cc = np.nonzero(m)
                resid.append(
                    np.hypot(rr.mean() - (r0 + h / 2), cc.mean() - (c0 + w / 2))
                )
            scores[rot] = float(np.mean(resid)) if resid else np.inf
        info.drop_index_map()
        best = min(scores, key=scores.get)
        logger.info(
            "rotate=auto on tilt %+.1f: %s -> %d deg",
            tilt, {k: round(v, 1) for k, v in scores.items()}, best,
        )
        if scores[best] > 200:
            logger.error(
                "Even the best rotation leaves a %.0f px mean residual. The ROI, "
                "pixel size or binning are probably wrong for this dataset.",
                scores[best],
            )
        # A footprint test can only see the tile's *shape*, and k and k+2 give
        # the same shape in the same place. So this settles 0-or-180 vs
        # 90-or-270 and nothing more. Getting the remaining 180 deg wrong maps
        # every particle to the diametrically opposite point of its tile, which
        # the patch average in overlay_picks_on_montage will show as pure noise
        # -- that check is not optional.
        opposite = (best + 180) % 360
        if abs(scores[opposite] - scores[best]) < 0.05 * max(scores[best], 1.0):
            logger.warning(
                "rotate=%d and rotate=%d are indistinguishable from tile "
                "footprints alone (%.1f vs %.1f px). Taking %d; confirm it with "
                "the patch average, or with --canvas-stack.",
                best, opposite, scores[best], scores[opposite], best,
            )
        return best

    # -- the chain ----------------------------------------------------------

    def theta(self, tilt: float) -> float:
        """Reprojection angle in radians: the .aln TILT column as written."""
        return np.deg2rad(self.tilts[tilt].sec.tilt)

    def tomo_to_aligned(self, tilt: float, xyz: np.ndarray) -> np.ndarray:
        """
        (N, 3) reconstruction voxels -> (N, 3) of (row, col, depth) in the
        aligned frame, still at the reconstruction's own sampling.

        ``depth`` is the along-beam coordinate, zero at the tomogram centre, and
        is what sets a particle's defocus relative to the tilt image's nominal
        value. Its sign is the classic tomography handedness ambiguity, so it is
        multiplied by ``handedness`` (+-1) rather than asserted.
        """
        xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
        nz, ny, nx = self.recon_shape
        cx, cz = (nx - 1) / 2.0, (nz - 1) / 2.0
        th = self.theta(tilt)
        cos_t, sin_t = np.cos(th), np.sin(th)
        dx, dz = xyz[:, 0] - cx, xyz[:, 2] - cz
        col = cx + dx * cos_t - dz * sin_t
        row = xyz[:, 1]
        depth = self.handedness * (dx * sin_t + dz * cos_t)
        return np.stack([row, col, depth], axis=1)

    def aligned_to_canvas(self, tilt: float, aligned_rc: np.ndarray) -> np.ndarray:
        """(N, 2) aligned (row, col) -> (N, 2) canvas (row, col)."""
        M, t = alignment_matrix(
            self.tilts[tilt].sec, self.canvas_shape, self.conv, self.out_bin
        )
        Minv = np.linalg.inv(M)
        aligned_rc = np.atleast_2d(np.asarray(aligned_rc, dtype=float))
        return (aligned_rc - t) @ Minv.T + self.extra_shift

    def tomo_to_canvas(self, tilt: float, xyz: np.ndarray) -> np.ndarray:
        """(N, 3) reconstruction voxels -> (N, 2) canvas (row, col)."""
        al = self.tomo_to_aligned(tilt, xyz)
        return self.aligned_to_canvas(tilt, al[:, :2])

    def canvas_to_raw_tile(self, local_rc: np.ndarray) -> np.ndarray:
        """
        Tile-local canvas pixels -> raw (unbinned, unrotated) tile pixels.

        ``stitch.py`` applies ``np.rot90(tile, k)`` to the *binned* tile, and
        ``np.rot90(A, 1)[i, j] == A[j, W-1-i]`` for A of shape (H, W). A binned
        pixel b covers raw pixels [b*bin, b*bin+bin-1], whose centre is at
        b*bin + (bin-1)/2.
        """
        local_rc = np.atleast_2d(np.asarray(local_rc, dtype=float))
        i, j = local_rc[:, 0], local_rc[:, 1]
        h, w = self.geom.tile_shape  # the tile AS PLACED, i.e. after rotation
        k = (self.rotate // 90) % 4
        if k == 0:
            rb, cb = i, j
        elif k == 1:
            rb, cb = j, (h - 1) - i
        elif k == 2:
            rb, cb = (h - 1) - i, (w - 1) - j
        else:
            rb, cb = (w - 1) - j, i
        off = (self.binning - 1) / 2.0
        return np.stack([self.binning * rb + off, self.binning * cb + off], axis=1)

    # -- tile assignment ----------------------------------------------------

    def assign(
        self,
        tilt: float,
        xyz: np.ndarray,
        margin: float = 0.0,
        require_coverage: bool = True,
    ) -> List[Optional[TileHit]]:
        """
        Assign each 3D point to at most one tile of this tilt.

        A candidate tile is only accepted if the *whole extraction box* lies
        inside that tile's illuminated square -- not merely inside the tile
        rectangle. The beam is smaller than the frame and offset within it, so
        the rectangle test lets the beam edge cut through ~15% of boxes, and a
        beam edge inside a particle box is worse than no observation at all.
        Among the tiles that pass, we keep the one holding the point furthest
        from its beam edge.

        ``index_map`` is consulted only as a coverage test: it is a partition,
        so its winner at a pixel is not necessarily the roomiest tile, but a
        fill value there does mean no tile contributed real data.

        Returns one entry per input point, ``None`` where the point is not
        usable in this tilt.
        """
        info = self.tilts[tilt]
        canvas = self.tomo_to_canvas(tilt, xyz)
        al = self.tomo_to_aligned(tilt, xyz)
        org = self.geom.tile_origin_rc(info.positions_um)
        h, w = self.geom.tile_shape
        sel = info.selected
        imap = info.index_map() if require_coverage else None
        fill = int(imap.max()) if imap is not None else None

        beams = info.beam_masks()
        if beams is None:
            logger.warning(
                "tilt %+.1f has no beam_masks in its HDF5 (pre-dating beam-mask "
                "storage); falling back to the tile rectangle, which does not "
                "keep the beam edge out of the box.", tilt,
            )
        half = int(np.ceil(margin / self.binning))  # box half-width, canvas px
        margin_canvas = margin / self.binning
        out: List[Optional[TileHit]] = []
        for n in range(canvas.shape[0]):
            rc = canvas[n]
            if imap is not None:
                ri, ci = int(round(rc[0])), int(round(rc[1]))
                if not (0 <= ri < imap.shape[0] and 0 <= ci < imap.shape[1]):
                    out.append(None)
                    continue
                if int(imap[ri, ci]) >= fill:
                    out.append(None)  # no tile contributed data here
                    continue
            best, best_margin = None, -np.inf
            for label, tile in enumerate(sel):
                local = rc - org[tile]
                if not (0 <= local[0] < h and 0 <= local[1] < w):
                    continue
                m = min(local[0], h - 1 - local[0], local[1], w - 1 - local[1])
                if m < margin_canvas:
                    continue
                if beams is not None:
                    r0, c0 = int(round(local[0])) - half, int(round(local[1])) - half
                    if r0 < 0 or c0 < 0 or r0 + 2 * half > h or c0 + 2 * half > w:
                        continue
                    if not beams[label, r0:r0 + 2 * half, c0:c0 + 2 * half].all():
                        continue  # the beam edge crosses this box
                if m > best_margin:
                    best, best_margin = (tile, local), m
            if best is None:
                out.append(None)
                continue
            tile, local = best
            raw = self.canvas_to_raw_tile(local)[0]
            out.append(
                TileHit(
                    tilt=tilt, tile=int(tile), x=float(raw[1]), y=float(raw[0]),
                    canvas_rc=(float(rc[0]), float(rc[1])),
                    depth=float(al[n, 2]) * self.tomo_bin,
                    edge_margin=float(best_margin) * self.binning,
                )
            )
        return out

    # -- RELION projection matrix ------------------------------------------

    def projection_matrix(self, tilt: float, tile: int) -> np.ndarray:
        """
        The 4x4 RELION tomogram projection matrix for one (tilt, tile) image.

        Maps a 3D position in *unbinned tilt-series pixels* with the origin at
        the tomogram corner -- which is what RELION reconstructs from
        ``rlnCenteredCoordinate*Angst`` plus ``rlnTomoSize*`` -- to
        ``(x, y, depth)`` in raw tile pixels.

        Every link in the chain is affine, so the matrix is recovered exactly by
        evaluating the chain on the origin and the three unit vectors. That is
        deliberate: composing the rotations, the inverse alignment matrix, the
        rot90 and the binning offsets by hand is four more chances to drop a
        sign, and this cannot disagree with the mapping ``assign()`` uses.
        """
        info = self.tilts[tilt]
        org = self.geom.tile_origin_rc(info.positions_um)[tile]

        def f(p: np.ndarray) -> np.ndarray:
            xyz = np.asarray(p, dtype=float)[None, :] / self.tomo_bin
            al = self.tomo_to_aligned(tilt, xyz)
            rc = self.aligned_to_canvas(tilt, al[:, :2])[0] - org
            raw = self.canvas_to_raw_tile(rc)[0]
            return np.array([raw[1], raw[0], al[0, 2] * self.tomo_bin])

        origin = f(np.zeros(3))
        P = np.zeros((4, 4))
        P[3, 3] = 1.0
        for k in range(3):
            e = np.zeros(3)
            e[k] = 1.0
            P[:3, k] = f(e) - origin
        P[:3, 3] = origin
        return P

    def tomo_size_unbinned(self) -> Tuple[int, int, int]:
        """(X, Y, Z) size of the tomogram in unbinned tilt-series pixels."""
        nz, ny, nx = self.recon_shape
        return (nx * self.tomo_bin, ny * self.tomo_bin, nz * self.tomo_bin)

    def centred_angst(self, xyz: np.ndarray) -> np.ndarray:
        """
        Reconstruction voxels -> ``rlnCenteredCoordinate{X,Y,Z}Angst``.

        RELION undoes this as ``coord_px = centred/apix + rlnTomoSize/2``, so
        as long as ``rlnTomoSize`` is ``recon_shape * tomo_bin`` the size/2 vs
        (size-1)/2 choice cancels and the round trip is exact.
        """
        xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
        nz, ny, nx = self.recon_shape
        centre = np.array([nx, ny, nz], dtype=float) / 2.0
        return (xyz - centre) * (self.pixel_size * self.tomo_bin)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_tile_stacks(directory: str) -> Dict[float, str]:
    """Map tilt angle -> per-tilt tile stack, e.g. Montage_9-A_9-A_-12.mrc."""
    from .refine_montage_projmatch import tilt_from_filename

    out: Dict[float, str] = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".mrc") or "_refined_positions" in name:
            continue
        try:
            out[tilt_from_filename(name)] = os.path.join(directory, name)
        except ValueError:
            continue
    if not out:
        raise FileNotFoundError(f"No per-tilt tile stacks found in {directory}")
    return out


def _probe_tile_shape(stacks: Dict[float, str]) -> Tuple[int, int]:
    import mrcfile

    if not stacks:
        raise ValueError("Pass --tile-dir or --tile-shape so the tile size is known.")
    path = stacks[sorted(stacks)[0]]
    with mrcfile.open(path, header_only=True, permissive=True) as m:
        return (int(m.header.ny), int(m.header.nx))


def load_picks(path: str) -> np.ndarray:
    """
    Read an IMOD ``model2point`` text file into an (N, 3) array of (X, Y, Z).

    Accepts 3-column ``x y z`` and the 4/5-column forms that carry object and
    contour numbers, by taking the last three columns.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append([float(x) for x in parts[-3:]])
    if not rows:
        raise ValueError(f"No coordinates parsed from {path}")
    return np.asarray(rows, dtype=float)


def stack_slice_index(info: TiltInfo, tile: int) -> int:
    """
    z-slice of the per-tilt tile stack holding tile ``tile``.

    ``Refined_positions`` is full length and the stack has one slice per row, so
    this is the identity -- but it is a real assumption (it fails on stacks
    written before the tile-index re-basing fix, see fix_stack_tile_index.py) and
    is worth naming rather than inlining.
    """
    return int(tile)
