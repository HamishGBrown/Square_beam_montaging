import re
import logging
import networkx as nx
from tqdm import tqdm
from PIL import Image
import os

from skimage.registration import phase_cross_correlation

from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import copy
import glob
import mrcfile
import argparse
import numpy as np
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import threading
from .Utilities import *
from .smoothn import smoothn

try:
    from .beam_mask_motioncorr import _create_stack_locked
except ImportError:
    from beam_mask_motioncorr import _create_stack_locked

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-tile MRC directory support
# ---------------------------------------------------------------------------

# Matches filenames like  Name_003_-12.0.mrc  →  groups (tile_idx, tilt_angle)
_TILE_FNAME_RE = re.compile(r"^.+_(\d+)_(-?\d+\.?\d*)\.mrc$")


class TileStack:
    """Read-only adapter that presents a list of per-tile MRC files as a
    virtual (N, H, W) stack, duck-typing the ``tiles`` interface
    expected by :func:`montage`.

    Supports integer indexing, boolean-mask indexing (returns a list), and
    the ``.dtype`` / ``.shape`` attributes used by :func:`montage`.
    """

    def __init__(self, paths: list) -> None:
        self._paths = paths
        with mrcfile.mmap(paths[0], mode="r", permissive=True) as m:
            self.dtype = m.data.dtype
            self.shape = (len(paths),) + tuple(m.data.shape[-2:])

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            with mrcfile.mmap(self._paths[int(idx)], mode="r", permissive=True) as m:
                return np.asarray(m.data)
        # bool mask or index array → list, compatible with executor.map
        if isinstance(idx, np.ndarray) and idx.dtype == bool:
            indices = np.where(idx)[0]
        else:
            indices = np.asarray(idx, dtype=int)
        return [self[i] for i in indices]


def _group_per_tile_mrcs(directory: str):
    """Group per-tile MRC files from a motioncorr output directory by tilt angle.

    Expects filenames of the form ``{name}_{tile_index}_{tilt_angle}.mrc``
    (the naming convention produced by ``beam_mask_motioncorr``).

    Returns
    -------
    sorted_tilts : list of float
    tile_paths   : dict mapping tilt_angle → list of paths sorted by tile index
    """
    groups: dict = {}
    for path in glob.glob(os.path.join(directory, "*.mrc")):
        m = _TILE_FNAME_RE.match(os.path.basename(path))
        if not m:
            continue
        tile_idx = int(m.group(1))
        tilt = float(m.group(2))
        if tilt == -0.0:
            tilt = 0.0
        groups.setdefault(tilt, []).append((tile_idx, path))

    sorted_tilts = sorted(groups)
    tile_paths = {t: [p for _, p in sorted(groups[t])] for t in sorted_tilts}
    return sorted_tilts, tile_paths


def _first_tile_shape(file_or_stack):
    """Return (H, W) of the first tile from either an MRC stack path or a TileStack."""
    if isinstance(file_or_stack, str):
        return mrcfile.mmap(file_or_stack).data.shape[-2:]
    return file_or_stack.shape[-2:]



def parse_commandline() -> Dict[str, Any]:
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(
        description="Stitch square beam montage tomography data."
    )
    parser.add_argument(
        "-i", "--input",
        help="Either a glob pattern matching per-tilt MRC stacks (e.g. '*.mrc'), "
             "or a directory of per-tile MRC files produced by beam_mask_motioncorr "
             "(files named {stem}_{tile_index}_{tilt_angle}.mrc).",
        required=True, type=str,
    )
    parser.add_argument(
        "--pixel-size", dest="pixel_size", type=float, default=None,
        help="Pixel size in Ångström. If omitted, read from the input MRC's "
             "voxel_size header.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Where to write stitched output. Three forms are accepted: "
             "(1) omitted, or a directory — the current per-tilt TIFF (lzw) "
             "behaviour, placed in ./[input_filename]_output if omitted; "
             "(2) 'path/to/stack.mrc:N' — write directly into z-slice N of a "
             "shared memory-mapped MRC stack at that path (created on first "
             "use, flock-guarded, safe to call concurrently), skipping the "
             "separate JointoMRC.py join step entirely; sets --tilt-index to "
             "N unless --tilt-index is also given; (3) 'path/to/stack.mrc' "
             "(no ':N') — write a single stitched TIFF, with the extension "
             "forced to '.tiff', instead of treating the path as a directory. "
             "Forms (2) and (3) require exactly one discovered tilt (use "
             "--tilt-index, or the ':N' suffix, to select it).",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--tilt-index",
        dest="tilt_index",
        help="Restrict this invocation to stitching only the tilt at this "
             "0-based rank (tilt angles sorted ascending) instead of the "
             "whole series — pairs with --output pointing at an MRC stack to "
             "run one SLURM array task per tilt, each writing only its own "
             "slice. If omitted, every discovered tilt is stitched, as "
             "before. Overrides any ':N' suffix on --output.",
        type=int,
        default=None,
        required=False,
    )
    parser.add_argument(
        "-I",
        "--image_shifts",
        help="Path to text file containing the tilts and a list of image shifts at every tilt (requried).",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-b",
        "--binning",
        help="Binning of input data, defaults to 1",
        required=False,
        type=int,
        default=1,
    )
    parser.add_argument(
        "--rotate",
        help="Rotate every input tile (and template mask) counter-clockwise "
             "by this many degrees before masking/alignment/stitching. Must "
             "be a multiple of 90. Useful when the camera's row/col axes are "
             "rotated relative to the stage x/y axes assumed by the image "
             "shift file. Default 0 (no rotation).",
        required=False,
        type=int,
        choices=[0, 90, 180, 270],
        default=0,
    )
    parser.add_argument(
        "-f",
        "--fringe_size",
        help="Size of Fresnel fringes at edge of beam, this will be removed from the gain reference (default 20).",
        required=False,
        type=int,
        default=20,
    )
    parser.add_argument(
        "-s",
        "--skipcrosscorrelation",
        help="Skip cross-correlation alignment of montage tiles and default to using imageshifts to stitch montage.",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--max_allowed_imshift_correction",
        help="Maximum allowed correction to tile alignments by cross-correlation.",
        type=float,
        default=0.05,
        required=False,
    )

    parser.add_argument(
        "-S",
        "--correctimageshiftfilefactor",
        help="Sometimes imageshift file needs to be divided by a factor of 2 since serial-EM uses super-resolution pixels.",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "-mt",
        "--maskthreshold",
        help="threshold (as fraction of raw image median) for masking of beam.",
        type=float,
        default=0.4,
        required=False,
    )

    parser.add_argument(
        "-ma",
        "--maskabsolutethreshold",
        help="Absolute threshold (in number of image counts) for masking of beam.",
        type=float,
        default=None,
        required=False,
    )

    parser.add_argument(
        "-R",
        "--ROI",
        help="Region of interest in the montage to stitch in format x0:x1,y0:y1 (in pixels).",
        type=int,
        default=None,
        required=False,
        nargs=4,
    )

    parser.add_argument(
        "-nt",
        "--nthreads",
        help="Number of threads for parralel implementation.",
        type=int,
        default=1,
        required=False,
    )

    parser.add_argument(
        "--nosmooth",
        help="Disable smoothing of background regions in the stitched montage.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--mark-uncovered",
        help=(
            "Instead of inpainting uncovered (no-tile) regions, mark them with "
            "pixel value -1. Use this when the output will be processed by "
            "mask_and_inpaint, which detects the sentinel automatically and "
            "includes those regions in its initial mask."
        ),
        action="store_true",
        default=False,
        dest="mark_uncovered",
    )

    parser.add_argument(
        "-pf",
        "--positionfile",
        help="Path to an HDF5 position file to use instead of the default (outdir/<image>_refined_positions.h5).",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "-T",
        "--templatemask",
        help="Path to a template mask file (TIFF or MRC) defining the beam shape. When provided, edge-based cross-correlation is used to align the template to each tile instead of threshold-based masking. For multi-frame MRC or TIFF files, append :INDEX to select a specific frame (e.g. mask.mrc:7).",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--min_mean_intensity",
        help="Tiles with a mean pixel value at or below this threshold are excluded from processing. Default is 5.",
        type=float,
        default=5,
        required=False,
    )

    parser.add_argument(
        "--correct_beam_edges",
        help="Enable correction of beam edge darkening caused by inelastic (plasmon) "
             "scattering.  The template provided via --templatemask is used as the "
             "vacuum reference.  With no argument, plasmon parameters are fitted "
             "independently for every tile.  With an integer argument N "
             "(e.g. --correct_beam_edges 42), tile N is used to fit the parameters "
             "once and those values are applied to all tiles — useful when a known "
             "interior tile gives a better fit than edge tiles.  "
             "Pass -1 to automatically use the tile closest to the centre of the montage.",
        nargs="?",
        const=True,
        default=False,
        type=int,
    )

    parser.add_argument(
        "--plasmon_energy",
        help="Plasmon energy in eV used for beam-edge correction (--correct_beam_edges). "
             "Default 21 eV, appropriate for vitreous ice / water.",
        type=float,
        default=21.0,
        required=False,
    )

    parser.add_argument(
        "--voltage",
        help="Accelerating voltage in kV used for beam-edge correction "
             "(--correct_beam_edges).  Default 300 kV.",
        type=float,
        default=300.0,
        required=False,
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose",
        help="Print detailed per-tile progress and diagnostic information.",
        action="store_true",
        default=False,
    )
    verbosity.add_argument(
        "-q", "--quiet",
        help="Suppress all output except warnings and errors.",
        action="store_true",
        default=False,
    )

    return vars(parser.parse_args())



def stitch(ims,positions,msks,pixel_size,binning=1,montagewidth=None,montageorigin=None,smooth=True,nthreads=1,mark_uncovered=False):
    """
    Stitch a set of tiles into a single montage canvas.

    Parameters
    ----------
    ims : sequence of numpy.ndarray
        2D image tiles to place into the montage (shape: N, rows, cols).
    positions : numpy.ndarray
        Array of shape (N, 2) with tile positions in microns (x, y order).
    msks : sequence of numpy.ndarray
        Boolean/0-1 masks matching each tile; masked-out pixels are ignored.
    pixel_size : float
        Pixel size in Angstroms for the unbinned images.
    binning : int, optional
        Binning factor applied to the montage. Default is 1.
    montagewidth : None or (2,) array_like, optional
        Full montage size in Angstroms. If None, inferred from positions.
    montageorigin : None or (2,) array_like, optional
        Montage origin in Angstroms (most negative image shift). If None, inferred.
    smooth : bool, optional
        If True, smooth background (uncovered) regions; otherwise fill with median.

    Returns
    -------
    canvas : numpy.ndarray
        Stitched montage image (overlaps averaged).
    indxmap : np.ndarray
        Map of indicating index of tiles
    """
    # Determine the global range of tiles in Angstroms
    pixels = ims[0].shape
    if montagewidth is None:
        width = np.ptp(positions, axis=0) * 1e4 + np.asarray(pixels[::-1]) * pixel_size * binning
    else:
        width = montagewidth
    if montageorigin is None:
        origin = np.amin(positions, axis=0) * 1e4
    else:
        origin = montageorigin

    # Calculate the size of the global montage canvas in pixels
    size = (
        np.asarray(width / pixel_size / binning, dtype=int)[::-1]
        # + pixels
        # + np.asarray([1, 1])
    )

    logger.debug("Stitch canvas: %d × %d px  |  %d tile(s)", size[0], size[1], len(ims))
    # Initialize the montage canvas and overlap map
    canvas = np.zeros(size)
    overlap = np.zeros(size, dtype=np.uint8)
    # Initialise the index map with n+1, where n is the number of tiles to identify
    # indices that correspond to no tile, indices that correspond to a tile will be
    # overwritten
    indxmap = len(ims)*np.ones(size, dtype=np.uint16)

    # Place each image onto the canvas, applying the masks
    for i, (im, position, msk) in enumerate(
        zip(ims, positions, msks)
    ): 
        # s is shape of tile, size is shape of canvas
        s = im.shape
        # Desired coordinate of upper left of tile
        y0, x0 = [int(x) for x in (position * 1e4 - origin) / pixel_size / binning]
    

        # Skip tiles that have fallen off canvas
        if x0 > size[0] or y0 > size[1]:
            continue
        if x0 +s[0] <0 or y0 +s[1] <0:
            continue

        # Truncate canvas coordinate beginnings to be >= 0
        cy0, cx0 = [max(coord, 0) for coord in [y0, x0]]
        # Truncate canvas coordinate maximum to be <= canvas array limits
        cx1, cy1 = [
            min(coord, limit) for coord, limit in zip([x0 + s[0], y0 + s[1]], size)
        ]

        # Size of tile that will make it onto the montage canvas
        X = cx1 - cx0
        Y = cy1 - cy0

        # Coordinates of tile, if x0 (or y0) < 0 this implies some of the tile
        # falls off the left (or upper) edge of canvas so is not included
        tx0 = -min(x0,0)
        tx1 = tx0 + X
        ty0 = -min(y0,0)
        ty1 = ty0 + Y

        canvas[cx0:cx1, cy0:cy1] += np.where(msk, im, 0)[tx0:tx1,ty0:ty1]
        overlap[cx0:cx1, cy0:cy1] += np.where(msk, np.uint8(1), np.uint8(0))[tx0:tx1,ty0:ty1]
        indxmap[cx0:cx1, cy0:cy1][msk[tx0:tx1,ty0:ty1]] = np.uint16(i)
    
    if mark_uncovered:
        canvas[overlap < 1] = -1  # sentinel: picked up by mask_and_inpaint as uncovered
    elif smooth:
        logger.info("Smoothing background regions")
        smoothed = smoothn(np.ma.masked_array(canvas, overlap < 1), s=1e7, max_iter=100, workers=nthreads)
        canvas = np.where(overlap < 1, smoothed, canvas)
    else:
        canvas[overlap < 1] = 0 #np.median(canvas[overlap == 1])
    
    overlap = np.where(overlap > 1, overlap, 1)
    canvas /= overlap
    
    return canvas,indxmap



def montage(
    image,
    outdir,
    positions,
    pixel_size,
    binning=8,
    skipcrosscorrelation=False,
    montagewidth=None,
    montageorigin=None,
    tiles=None,
    maxshift=0.05,
    fringe_size=20,
    maskthreshold=0.4,
    maskabsolutethreshold=None,
    nthreads=1,
    positionfile=None,
    smooth=True,
    mark_uncovered=False,
    template_mask=None,
    min_mean_intensity=20,
    correct_beam_edges=False,
    E_plasmon_eV=21.0,
    voltage_kV=300.0,
    rotate=0,
    output_mrc=None,
    tilt_z_index=None,
    n_tilts_total=None,
    fileout_path=None,
):
    """
    Creates a montage image from a series of input images by aligning and stitching them together.

    This function takes a set of images, positions them according to provided coordinates,
    and aligns them based on mask images or generated masks. The images are then combined
    into a single montage image.

    Parameters:
    -----------
    image : str
        mrc file containing montage tiles
    outdir : str
        directory for output
    positions : numpy.ndarray
        Array of shape (N, 2) containing the (x, y) coordinates for positioning each image in units of pixels.
        Note that the x,y have opposite convention to standard y,x Python convention
    pixel_size : float
        The pixel size in Angstroms.
    binning : int, optional
        Binning factor to reduce the image size. Default is 8.
    skipcrosscorrelation : bool, optional
        Skip masked cross correlation of tiles and rely on user provided tile positions
        (if supplied through positionfile kwarg) or raw image tilts (if no positionfile supplied)
    montagewidth : None or (2,) array_like
        size in Angstrom of the full montage in both dimensions, useful for consistency
        with other tilts in the tilt series
    montageorigin : None or (2,) array_like
        Origin point (most negative image shift) in Angstrom of the full montage
        in both dimensions, useful for consistency with other tilts in the tilt
        series
    tiles : None or sliceobject, optional
        Slice object indicating tiles that will be stitched (mainly for testing purposes)
    fringe_size : float, optional
        Size of the Fresnel fringe region excluded from the beam mask. Default is 20.
    medianthreshold : float, optional
        Threshold for (as a fraction of the median) for masking the image. Default is 0.4.
    positionfile : string, optional
        hdf5 file containing already determined or refined tile positions, if provided masked
        cross-correlation will be skipped (ie. this overrides)
    correct_beam_edges : bool, optional
        When True, beam edge darkening due to plasmon scattering is corrected
        per-tile using plasmon_beam_correction().  The template_mask image is
        used as the vacuum reference and is aligned to each tile via the same
        edge cross-correlation used by make_mask.
    E_plasmon_eV : float, optional
        Plasmon energy in eV for beam-edge correction. Default 21 eV.
    voltage_kV : float, optional
        Accelerating voltage in kV for beam-edge correction. Default 300 kV.
    rotate : int, optional
        Rotate every tile (and the template mask) counter-clockwise by this
        many degrees before any further processing. Must be a multiple of
        90. Default 0 (no rotation).
    output_mrc : str or None, optional
        When given, write the stitched canvas into z-slice ``tilt_z_index``
        of a shared memory-mapped MRC stack at this path (created under a
        file lock on first use), instead of only writing the per-tilt TIFF.
        Requires ``tilt_z_index`` and ``n_tilts_total`` to also be given.
    tilt_z_index : int or None, optional
        0-based z-slice this tilt occupies in ``output_mrc``.
    n_tilts_total : int or None, optional
        Total number of tilts in the series — sizes ``output_mrc`` on creation.
    fileout_path : str or None, optional
        Explicit path for the stitched TIFF, overriding the default
        ``outdir/<image>.tif`` naming. Ignored when ``output_mrc`` is given
        (the TIFF is skipped entirely in that case).
    Returns:
    --------
    numpy.ndarray
        The resulting stitched montage image.
    """
    if rotate % 90 != 0:
        raise ValueError("rotate must be a multiple of 90 degrees")
    rot_k = (rotate // 90) % 4

    # Load tile data: accept either an MRC stack path or a TileStack adapter.
    tile_selection = tiles  # save before overwriting with actual image data
    tiles = mrcfile.mrcmemmap.MrcMemmap(image).data if isinstance(image, str) else image

    pixels = np.asarray([x // binning for x in tiles.shape[-2:]], dtype=int)
    if rot_k % 2:
        # A 90°/270° rotation swaps height and width for every tile.
        pixels = pixels[::-1]

    # M is the boolean mask describing which tiles to include.
    # None → all tiles; slice or int-index array → convert to bool; bool array → use directly.
    if tile_selection is None:
        M = np.ones(len(tiles), dtype=bool)
    elif isinstance(tile_selection, slice):
        M = np.zeros(len(tiles), dtype=bool)
        M[tile_selection] = True
    else:
        M = np.zeros(len(tiles), dtype=bool)
        M[tile_selection] = True

    # Convert positions from pixels to microns, remove 3rd dimension
    positions = positions[:, :2] * pixel_size * 1e-4

    # Initialize an empty mask list
    msks = []

    ims = []  # List to store the processed images
    beam_posns = []  # List to store the beam positions

    # Identify template mask
    if template_mask is None:
        if correct_beam_edges:
            logger.warning(
                "correct_beam_edges is enabled but no --templatemask was provided; "
                "falling back to tile 0 as the beam reference."
            )
        logger.debug("No template mask provided — using tile 0 as template")
        template_mask = np.asarray(tiles[0])

    if rot_k:
        template_mask = np.ascontiguousarray(np.rot90(template_mask, k=rot_k))

    # Keep the raw (non-binary) template as the beam reference for plasmon correction.
    raw_template = np.asarray(template_mask, dtype=float)

    smooth_template_mask = convolve(template_mask, Gaussian(3, template_mask.shape))
    template_edges = np.hypot(sobel(smooth_template_mask, axis=0), sobel(smooth_template_mask, axis=1))
    template_mask = make_mask(template_mask,shrinkn=0,medianthreshold=maskthreshold,absolutethreshold=maskabsolutethreshold)


    # Plasmon correction parameter state.
    # _plasmon_fixed: when the user passes --correct_beam_edges N, tile N is
    #   fitted once before the thread pool and its (n, q_E) are locked in for
    #   all other tiles, bypassing per-tile optimisation entirely.
    # _plasmon_running: when no reference tile is given, each tile fits its own
    #   parameters but uses the running mean of previous fits as a warm start,
    #   skipping the 25×25 grid and going straight to Nelder-Mead.
    _plasmon_fixed_n   = [None]
    _plasmon_fixed_q_E = [None]

    ref_tile_idx = correct_beam_edges if (isinstance(correct_beam_edges, int) and not isinstance(correct_beam_edges, bool)) else None
    correct_beam_edges = bool(correct_beam_edges)

    if ref_tile_idx == -1:
        # Auto-select the tile whose stage position is closest to the montage centre.
        active_indices = np.where(M)[0]
        active_positions = positions[M]
        centre = active_positions.mean(axis=0)
        ref_tile_idx = int(active_indices[np.argmin(
            np.linalg.norm(active_positions - centre, axis=1)
        )])
        logger.info(f"Auto-selected centre tile {ref_tile_idx} as plasmon reference")

    if ref_tile_idx is not None and correct_beam_edges:
        logger.info(f"Fitting plasmon parameters on reference tile {ref_tile_idx} ...")
        _ref_img = np.asarray(tiles[ref_tile_idx]).copy()
        if rot_k:
            _ref_img = np.ascontiguousarray(np.rot90(_ref_img, k=rot_k))
        if binning > 1:
            _ref_img = fourier_interpolate(_ref_img, [x // binning for x in _ref_img.shape])
        _ref_mask = make_mask(
            _ref_img,
            shrinkn=fringe_size / binning,
            medianthreshold=maskthreshold,
            absolutethreshold=maskabsolutethreshold,
            template_mask=template_mask,
            template_edges=template_edges,
        )
        _, _plasmon_fixed_n[0], _plasmon_fixed_q_E[0] = plasmon_beam_correction(
            _ref_img, raw_template, _ref_mask,
            pixel_size_nm=pixel_size * binning / 10.0,
            template_edges=template_edges,
            E_plasmon_eV=E_plasmon_eV,
            voltage_kV=voltage_kV,
        )
        logger.info(f"Reference tile fit: n={_plasmon_fixed_n[0]:.3f}  "
                    f"q_E={_plasmon_fixed_q_E[0]:.5f} cyc/nm")

    _plasmon_lock  = threading.Lock()
    _plasmon_n_sum = [0.0]
    _plasmon_qE_sum = [0.0]
    _plasmon_count  = [0]

    def _process_tile(img):
        im = np.asarray(img).copy()
        if rot_k:
            im = np.ascontiguousarray(np.rot90(im, k=rot_k))
        if binning > 1:
            im = fourier_interpolate(im, [x // binning for x in im.shape])
        if np.mean(im) <= min_mean_intensity:
            return None
        mask = make_mask(
            im,
            shrinkn=fringe_size / binning,
            medianthreshold=maskthreshold,
            absolutethreshold=maskabsolutethreshold,
            template_mask=template_mask,
            template_edges=template_edges,
        )
        if correct_beam_edges:
            if _plasmon_fixed_n[0] is not None:
                # Fixed-parameter mode: apply the reference tile's fit to every tile.
                im, _, _ = plasmon_beam_correction(
                    im, raw_template, mask,
                    pixel_size_nm=pixel_size * binning / 10.0,
                    template_edges=template_edges,
                    E_plasmon_eV=E_plasmon_eV,
                    voltage_kV=voltage_kV,
                    n_fixed=_plasmon_fixed_n[0],
                    q_E_fixed=_plasmon_fixed_q_E[0],
                )
            else:
                # Per-tile mode: use running mean as warm start for the grid search.
                with _plasmon_lock:
                    count = _plasmon_count[0]
                    n_hint   = _plasmon_n_sum[0]  / count if count else None
                    q_E_hint = _plasmon_qE_sum[0] / count if count else None
                im, n_fit, q_E_fit = plasmon_beam_correction(
                    im, raw_template, mask,
                    pixel_size_nm=pixel_size * binning / 10.0,
                    template_edges=template_edges,
                    E_plasmon_eV=E_plasmon_eV,
                    voltage_kV=voltage_kV,
                    n_hint=n_hint,
                    q_E_hint=q_E_hint,
                )
                if n_fit > 0.0:
                    with _plasmon_lock:
                        _plasmon_n_sum[0]  += n_fit
                        _plasmon_qE_sum[0] += q_E_fit
                        _plasmon_count[0]  += 1
        return im, mask


    _quiet = not logger.isEnabledFor(logging.INFO)
    max_workers = min(os.cpu_count() or 1, nthreads)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(_process_tile, tiles[M]),
                total=len(positions[M]),
                desc="Making masks",
                disable=_quiet,
            )
        )
    # Single threaded implementation for testing/debugging
    # results = []
    # for tile in tiles[M]:
    #     result = _process_tile(tile)
    #     results.append(result)

    M_indices = np.where(M)[0]
    excluded = [M_indices[i] for i, r in enumerate(results) if r is None]
    if excluded:
        logger.warning("Excluded %d low-intensity tile(s): indices %s", len(excluded), excluded)
        M[excluded] = False
    logger.debug("%d tile(s) retained after intensity filtering", sum(M))

    ims = [r[0] for r in results if r is not None]
    msks = [r[1] for r in results if r is not None]


    if positionfile is None:
        positionfile = os.path.join(
            outdir, os.path.split(image)[1].replace(".mrc", "_refined_positions.h5")
        )
    elif os.path.exists(positionfile):
        skipcrosscorrelation = True

    # Calculate the overlaps between adjacent tiles
    overlaps = find_overlaps(
        positions[M], pixels, pixel_size * binning, msks, plot=False
    )

    if not skipcrosscorrelation:
        # Refine the tile positions using cross-correlation between overlapping
        # tiles
        original_positions = copy.deepcopy(positions)
        positions[M], xcorr, deltas = cross_correlate_tiles(
            positions[M],
            ims,
            msks,
            overlaps,
            pixel_size * binning,
            max_correction=maxshift,
            max_workers=nthreads
        )
    elif os.path.exists(positionfile):
        # If requested to skip cross correlation and an older set of tile
        # alignments exist then load these.
        logger.info("Loading existing tile positions from %s", positionfile)
        positions = load_array_from_hdf5(positionfile, "Refined_positions")
        # xcorr = load_array_from_hdf5(positionfile, "cross_correlations")
        original_positions = positions
    else:
        original_positions = positions

    # Recalculate overlaps before mask clipping since some tiles might have migrated
    # since the last time we did this.

    postrefineoverlaps = find_overlaps(
        positions[M], pixels, pixel_size * binning, msks, plot=False,minoverlapfrac=1/np.prod(pixels)
    )

    # Clip masks to ensure overlapping regions are only taken from closest tile
    msks = clip_masks_to_overlaps(msks,positions[M],postrefineoverlaps,pixels,pixel_size * binning)

    # Stitch the images together using the refined tile positions
    canvas,indxmap = stitch(
        ims,
        positions[M],
        msks,
        pixel_size,
        binning=binning,
        montagewidth=montagewidth,
        montageorigin=montageorigin,
        smooth=smooth,
        mark_uncovered=mark_uncovered,
        nthreads=nthreads,

    )
    if output_mrc is None:
        fileout = fileout_path if fileout_path is not None else os.path.join(
            outdir, os.path.splitext(os.path.split(image)[1])[0] + ".tif"
        )
    # if not skipcrosscorrelation:
    ntiles = len(original_positions[M])
    figsize = 2 * (int(np.ceil(np.sqrt(ntiles))),)
    xcorfig, xcorax = plt.subplots(figsize=figsize)
    cmean = np.mean(canvas)
    cstd = np.std(canvas)
    if montageorigin is None or montagewidth is None:
        extent = None
    else:
        extent = [
            montageorigin[0] ,
            (montageorigin[0] + montagewidth[0]),
            montageorigin[1] ,
            (montageorigin[1] + montagewidth[1]),
        ]
        extent = np.asarray(extent) / pixel_size
    xcorax.imshow(
        canvas,
        cmap=plt.get_cmap("gist_gray"),
        origin="lower",
        vmin=cmean - cstd,
        vmax=cmean + cstd,
        extent=extent,
    )
    # s in unbinned pixels so all coordinates share the same unit
    s = np.asarray(ims[0].shape)[::-1] * binning
    posi = (original_positions[M] * 1e4) / pixel_size
    posi += s / 2
    xcorax.plot(*posi.T, "ko", label="Initial tile positions")
    for i, pos in enumerate(posi):
        xcorax.annotate(str(i), pos)
    posi = (positions[M] * 1e4) / pixel_size
    posi += s / 2
    xcorax.plot(*posi.T, "bo", label="Refined tile positions")
    # xcorrmax = np.amax(xcorr)
    cmap = plt.get_cmap("viridis")
    for ind, (i, j) in enumerate(overlaps):
        # Retrieve image shifts for the overlapping tiles
        x1 = (original_positions[M][i] * 1e4) / pixel_size

        x2 = (original_positions[M][j] * 1e4) / pixel_size
        dx = [int(x) for x in (x2 - x1)][::-1]
        x1 += s / 2
        if not skipcrosscorrelation:
            delta = (deltas[ind] * 1e4) / pixel_size

            shifttoolarge = (
                np.linalg.norm(np.asarray(dx) - delta)
                > maxshift / pixel_size * 1e4
            )
            if shifttoolarge:
                linestyle = "--"
            else:
                linestyle = "-"
        
            xcorax.plot(
                [x1[0], x1[0] + delta[1]],
                [x1[1], x1[1] + delta[0]],
                linestyle=linestyle,
                color="b"
            )
    # if skipcrosscorrelation:
    #     xcorax.plot(*(original_positions[M].T * 1e4)/ pixel_size / binning)
    xcorax.set_xlabel("x (unbinned pixels)")
    xcorax.set_ylabel("y (unbinned pixels)")
    plotfile = os.path.join(
        outdir, os.path.split(image)[1].replace(".mrc", "_Plot.pdf")
    )
    xcorfig.savefig(plotfile)
    # Note that Python Image library does not support write 16-bit integer and writes
    # 32-bit real instead ¯\_(ツ)_/¯
    # Image.fromarray(canvas).save(fileout.replace('.tif','float.tiff'),compression="tiff_lzw")
    # Image.fromarray(canvas.astype(np.uint16)).save(fileout.replace('.tif','nolzw.tiff'))
    # Clipping saves underflow errors upon conversion to uint16, sometimes images values can go
    # negative from smoothing algorithm
    if not skipcrosscorrelation:
        save_array_to_hdf5(
                [original_positions, positions, xcorr, overlaps, np.asarray(deltas),indxmap.astype(np.uint16),pixel_size/binning],
                positionfile,
                [
                    "Original_positions",
                    "Refined_positions",
                    "cross_correlations",
                    "overlaps",
                    "relative_shifts",
                    "index_map",
                    "binned_pixel_size"
                ],
            )
    # Image.fromarray(indxmap.astype(np.uint16)).save(fileout.replace('.tif','_index.tif'),compression="tiff_lzw")
    # Clip to signed 16-bit range.  When mark_uncovered is set, uncovered pixels
    # are exactly -1 (no smoothn runs in that path so no other negatives exist).
    # Otherwise clip negatives to 0 so smoothn artefacts don't become false sentinels.
    lo_clip = -1 if mark_uncovered else 0
    clipped = np.clip(canvas, lo_clip, 32767).astype(np.int16)
    if output_mrc is None:
        Image.fromarray(clipped).save(fileout, compression="tiff_lzw")
        if isinstance(image, str):
            # Stamp the source raw montage's mtime onto the stitched output so
            # downstream tools (JointoMRC.py's --acquisition_order) can recover
            # true acquisition order from file mtime, with no mdoc needed.
            src_mtime = os.path.getmtime(image)
            os.utime(fileout, (src_mtime, src_mtime))
        save_array_as_png(canvas, fileout.replace('.tiff','.png'), cmap=plt.get_cmap('Greys'))

    if output_mrc is not None:
        if tilt_z_index is None or n_tilts_total is None:
            raise ValueError("output_mrc requires both tilt_z_index and n_tilts_total")
        # mode 1 = int16, matching JointoMRC.py's stack convention and the
        # clipped int16 range already used for the TIFF above.
        _create_stack_locked(
            output_mrc, (n_tilts_total, *clipped.shape), pixel_size * binning, mrc_mode=1
        )
        with mrcfile.mmap(output_mrc, mode="r+") as out_mrc:
            out_mrc.data[tilt_z_index] = clipped
        logger.info(
            "Wrote tilt directly into %s  [slice %d/%d]",
            output_mrc, tilt_z_index, n_tilts_total,
        )

    return canvas


def plot_overlaps(positions, overlaps, show=True):
    m = len(positions)
    fig, ax = plt.subplots()
    ax.plot(*positions.T, "ko")
    for n, val in enumerate(overlaps):
        if val:
            i, j = condensed_to_square(n, m)
            x1 = positions[i]
            x2 = positions[j]
            ax.plot([x1[0], x2[0]], [x1[1], x2[1]], "r-")
    plt.show(block=show)
    return fig


def array_overlap(dx, n):
    """Overlapping indices for 1d array shifted dx relative to each other"""
    if dx > 0:
        return [dx, n], [0, n - dx]
    else:
        return [0, n + dx], [-dx, n]


def find_overlaps(positions, pixels, pixel_size, masks, minoverlapfrac=0.01, plot=True):
    """
    Identify pairs of images that overlap in a 2D montage, based on their positions
    and field of view, and further refine the overlap based on binary masks.

    Parameters:
    ----------
    positions : ndarray of shape (m, 2)
        The (x, y) coordinates of `m` images in a 2D montage. Note that
        the positions array is ordered (x,y) opposite to python standard (y,x)
    pixels : tuple or list of length 2
        Dimensions of each image in pixels, as (height, width).
    pixel_size : float
        The size of each pixel in microns.
    masks : list of ndarrays
        Binary masks for each image (same shape as image), used to check if
        a specific region of the images overlaps. If None, the function only
        considers the geometric overlap.
    minoverlapfrac : float, optional
        Minimum fraction of overlap (based on pixel area) required for two images
        to be considered overlapping. Default is 0.03 (i.e., 3%).
    plot : float, optional
    Returns:
    -------
    overlapping_inds : list of lists
        List of pairs of indices representing images that overlap.
        Each element is a list of two indices `[i, j]`, where the i-th and j-th
        images overlap.
    """
    m = len(positions)  # Number of images

    from scipy.spatial.distance import pdist

    # Calculate the size of each field of view in microns (height, width)
    # Order of pixels array has to be reversed from python standard to match
    # positions array
    criterion = pixel_size * np.asarray(pixels)[::-1] * 1e-4

    # Calculate (unique) distances between montage pieces
    # with scipy's pdist (pair-wise distance) function and
    # calculate which are within a "field of view" of each other
    # result is stored in a condensed distance matrix

    overlaps = np.logical_and(
        *[pdist(positions[..., i : i + 1]) / criterion[i] < 1.0 for i in range(2)]
    )

    # Visualize the positions and overlap status
    if plot:
        overlapfig = plot_overlaps(positions, overlaps, show=False)
        overlapax = overlapfig.get_axes()[0]

    # List to store the indices of overlapping image pairs
    overlapping_inds = []

    # Update criterion to be the minimum required overlap area in pixels
    criterion = minoverlapfrac * np.prod(pixels)

    # Iterate over the condensed distance matrix
    for n, val in enumerate(overlaps):
        if val:  # If overlap is detected
            # Convert the index in the condensed matrix back to the square form
            i, j = condensed_to_square(n, m)
            x1 = positions[i]
            x2 = positions[j]

            # Calculate the relative shift in position between the two images, in pixels
            dx = [int(x) for x in (x2 - x1) / pixel_size * 1e4]

            # Determine the overlapping pixel regions in both images
            i1, i2 = array_overlap(dx[1], pixels[0])
            j1, j2 = array_overlap(dx[0], pixels[1])

            # Check if the overlap area in the masks exceeds the minimum criterion
            overlap = (
                np.sum(
                    np.logical_and(
                        masks[i][i1[0] : i1[1], j1[0] : j1[1]],
                        masks[j][i2[0] : i2[1], j2[0] : j2[1]],
                    )
                )
                >= criterion
            )
            if overlap:
                overlapping_inds.append([i, j])
                if plot:
                    x1 = positions[i]
                    x2 = positions[j]
                    overlapax.plot([x1[0], x2[0]], [x1[1], x2[1]], "b--")
    if plot:
        plt.show(block=True)
    return overlapping_inds

def clip_masks_to_overlaps(masks, positions, overlaps, pixels, pixel_size):
    """
    Clip binary masks so when there is overlap, pixels are taken from the closest
    tile to the pixel in the overlapping region.

    Parameters:
    -----------
    masks : list of ndarrays
        List of binary masks for each image tile.
    positions : ndarray of shape (N, 2)
        Array of (x, y) coordinates for each image tile in microns.
    overlaps : list of tuples
        List of pairs of indices representing which tiles overlap.
    pixels : tuple or list of length 2
        Dimensions of each image tile in pixels, as (height, width).
    pixel_size : float
        The size of each pixel in microns.

    Returns:
    --------
    clipped_masks : list of ndarrays
        List of modified binary masks with overlapping regions clipped to 
        closest tile.
    """
    clipped_masks = [copy.deepcopy(msk) for msk in masks]

    x = np.arange(masks[0].shape[-1])
    y = np.arange(masks[0].shape[-2])
    distancefromcenter = np.zeros((len(masks),*masks[0].shape[-2:]))
    for i,mask in enumerate(masks):
        y0,x0 = center_of_mass_2d(mask)
        distancefromcenter[i] = (y[:,None]-y0)**2 + (x[None,:]-x0)**2

    # Pre-compute per-tile canvas offsets using the same integer truncation as stitch,
    # so that dx below matches the actual pixel-level placement of each tile.
    # int(A - B) != int(A) - int(B) in general; computing the difference of rounded
    # individual offsets avoids a systematic 1-pixel gap at seams.
    origin = np.amin(positions, axis=0) * 1e4
    canvas_offsets = np.array(
        [[int(v) for v in (pos * 1e4 - origin) / pixel_size] for pos in positions]
    )  # shape (N, 2): [x_offset, y_offset] per tile

    for i, j in overlaps:
        # Calculate relative shift in pixels, consistent with stitch's canvas placement
        dx_raw = canvas_offsets[j] - canvas_offsets[i]  # [x_shift, y_shift]
        dx = list(dx_raw[::-1])  # [row_shift, col_shift]

        # Get the indices for array overlap
        i1, i2 = array_overlap(dx[0], pixels[0])
        j1, j2 = array_overlap(dx[1], pixels[1])

        # Make a mask where pixels closer to center of tile i are True
        closer = np.less_equal(
            distancefromcenter[i][i1[0] : i1[1], j1[0] : j1[1]],
            distancefromcenter[j][i2[0] : i2[1], j2[0] : j2[1]],
        )
        # Only consider pixels that are in the mask of tile j
        closer = np.where(clipped_masks[j][i2[0] : i2[1], j2[0] : j2[1]],closer,True)
        clipped_masks[i][i1[0] : i1[1], j1[0] : j1[1]] = np.logical_and(closer,clipped_masks[i][i1[0] : i1[1], j1[0] : j1[1]])


        closer = np.greater_equal(
            distancefromcenter[i][i1[0] : i1[1], j1[0] : j1[1]],
            distancefromcenter[j][i2[0] : i2[1], j2[0] : j2[1]],
        )
        closer = np.where(clipped_masks[i][i1[0] : i1[1], j1[0] : j1[1]],closer,True)
        clipped_masks[j][i2[0] : i2[1], j2[0] : j2[1]] = np.logical_and(closer,clipped_masks[j][i2[0] : i2[1], j2[0] : j2[1]])

    return clipped_masks


def cross_correlate_tiles(
    positions,
    tiles,
    masks,
    overlaps,
    pixel_size,
    max_correction=0.3,
    generate_plot=False,
    cross_corr_param_file=None,
    parallelize=True,
    max_workers=None,
):
    """
    Perform cross-correlation-based alignment of image tiles and adjust their positions.

    This function aligns overlapping image tiles by calculating the relative shifts between them
    using masked phase cross-correlation. The shifts are then used to adjust the global positions
    of the tiles through least squares minimization. The function supports visualizing the tile
    positions and shift vectors if desired.

    Args:
        positions (ndarray): Nx2 array of the initial x, y coordinates of the tiles in micrometers.
        tiles (list of ndarrays): List of 2D arrays representing the image tiles.
        masks (list of ndarrays): List of binary masks representing the valid regions of the tiles.
        overlaps (list of tuples): Pairs of indices representing which tiles overlap and should be aligned.
        pixel_size (float): The pixel size in microns.
        max_correction (float, optional): Maximum allowed correction (in microns) for shifts between tiles. Defaults to 0.1.
        generate_plot (bool or string, optional): Whether to generate a plot visualizing the tile positions and shift vectors. Defaults to False.
                                                  If a string this will be the filename that the plot will be saved as.
        parallelize (bool, optional): Whether to parallelize cross-correlation work. Defaults to True.
        max_workers (int, optional): Maximum worker threads to use when parallelizing. Defaults to a tuned value.

    Returns:
        ndarray: Updated Nx2 array of the adjusted x, y coordinates of the tiles.

    Notes:
        - The alignment is solved as a least squares problem (Ax = b) to minimize the relative shifts between overlapping tiles.
        - If some tiles are not connected to others through reliable shift determinations, their positions are adjusted
        using the initial relative positions inferred from the microscope.

    References:
        - Masked phase cross-correlation: https://scikit-image.org/docs/stable/auto_examples/registration/plot_masked_register_translation.html
        - Dirk Padfield, "Masked object registration in the Fourier domain", IEEE Transactions on Image Processing, 2011.
    """
    pixels = tiles[0].shape
    cmap = plt.get_cmap("viridis")

    # Convert maximum allowed shift in microns (max_correction) to pixels (max_shift)
    max_shift = max_correction / pixel_size * 1e4

    # We will solve global alignment of tiles by least squares Ax = b
    # matrix problem (https://en.wikipedia.org/wiki/Linear_least_squares).
    # N is the number of tiles times two (for x and y coordinates)
    N = 2 * len(positions)
    # Initialize lists which we will append rows for the A and b matrices
    A = []
    b = []

    genplot = False
    if type(generate_plot) is bool:
        genplot = generate_plot
        show = True
        savefig = False
    elif type(generate_plot) is str:
        genplot = True
        show = False
        savefig = True

    if genplot:
        xcorfig, xcorax = plt.subplots(figsize=(8, 8))
        xcorax.plot(*positions.T, "ko", label="Initial tile positions")
        for i, pos in enumerate(positions):
            xcorax.annotate(str(i), pos)

    G = nx.Graph()
    # Add nodes to graph
    G.add_nodes_from(list(range(len(positions))))

    xcorr = []
    deltas = []

    if cross_corr_param_file is not None:
        xcorr, deltas = [
            load_array_from_hdf5(cross_corr_param_file, x)
            for x in ("cross_correlations", "relative_shifts")
        ]
    # plot_individual_cross_correlation(tiles,masks,positions,overlaps,pixel_size,filename_template='cross_corr_{0}_{1}.pdf')
    import concurrent.futures
    import os

    def _compute_overlap(args):
        ind, (i, j) = args
        # Retrieve image shifts for the overlapping tiles
        x1 = positions[i]
        x2 = positions[j]

        # Calculate relative shift in pixels
        dx = [int(x) for x in (x2 - x1) / pixel_size * 1e4][::-1]
        if cross_corr_param_file is None:
            # Align tiles by masked phase cross correlation, see:
            # https://scikit-image.org/docs/stable/auto_examples/registration/plot_masked_register_translation.html
            # and Padfield, Dirk. "Masked object registration in the Fourier domain." IEEE Transactions on image processing 21.5 (2011): 2706-2718.

            # Get the indices for array overlap
            i1, i2 = array_overlap(dx[0], pixels[0])
            j1, j2 = array_overlap(dx[1], pixels[1])

            # Crop to the overlap region before cross-correlating.
            # phase_cross_correlation internally FFTs at (2H × 2W) in
            # complex128; using the full tiles wastes memory proportional to
            # (full_tile / overlap)^2. Both crops already show the same
            # physical region, so the algorithm returns only the fine
            # correction; dx is added back to recover the total shift in
            # full-tile coordinates.
            overlap_mask = np.logical_and(
                masks[i][i1[0] : i1[1], j1[0] : j1[1]],
                masks[j][i2[0] : i2[1], j2[0] : j2[1]],
            )
            crop_ref = tiles[i][i1[0] : i1[1], j1[0] : j1[1]]
            crop_mov = tiles[j][i2[0] : i2[1], j2[0] : j2[1]]

            # Calculate shift by masked cross correlation with ordering (Y,X)
            fine_shift = phase_cross_correlation(
                crop_ref,
                crop_mov,
                reference_mask=overlap_mask,
                moving_mask=overlap_mask,
            )[0]
            detected_shift = fine_shift + np.array(dx)
            # delta is measured shift in microns
            delta = np.asarray(detected_shift) * pixel_size * 1e-4
        else:
            delta = deltas[ind]
            # xcorrmax = xcorr[ind]
            detected_shift = delta / pixel_size * 1e-4

        return ind, i, j, dx, detected_shift, delta

    if cross_corr_param_file is None:
        deltas = [None] * len(overlaps)
        use_parallel = parallelize and len(overlaps) > 1 and (max_workers is None or max_workers != 1)
        if use_parallel:
            worker_count = max_workers
            if worker_count is None:
                worker_count = min(32, (os.cpu_count() or 1) + 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = list(
                    tqdm(
                        executor.map(_compute_overlap, enumerate(overlaps)),
                        total=len(overlaps),
                        desc="Cross-correlation alignment",
                    )
                )
        else:
            results = [
                _compute_overlap(args)
                for args in enumerate(tqdm(overlaps, desc="Cross-correlation alignment"))
            ]
        for ind, _, _, _, _, delta in results:
            deltas[ind] = delta
    else:
        results = [
            _compute_overlap(args)
            for args in enumerate(tqdm(overlaps, desc="Cross-correlation alignment"))
        ]

    for ind, i, j, dx, detected_shift, delta in results:

        # shifttoolarge=True
        if generate_plot:
            if ind == 0:
                label = "Shift vector from cross-correlation"
            else:
                label = None

        shifttoolarge = (
            np.linalg.norm(np.asarray(dx) * pixel_size * 1e-4 - delta) > max_correction
        )
        # detected_shift is position j (x2) - position i (x1)
        if not shifttoolarge:
            Arow = np.zeros((2, N))
            Arow[0, 2 * j] = 1
            Arow[0, 2 * i] = -1
            Arow[1, 2 * j + 1] = 1
            Arow[1, 2 * i + 1] = -1
            A.append( Arow)
            b[len(b) :] = ( delta).tolist()
            # Add valid connection to graph
            G.add_edge(i, j)
        # Weight by the maximum value of cross-correlation
        # weights.append(xcorrmax)

    # Join rows of A into numpy matrix
    if len(A) > 0:
        A = np.concatenate(A, axis=0)
    else:
        A = np.asarray(A)
    # Normalize by mean weight
    # meanweight = np.mean(weights)
    # A /= meanweight

    # Function to find the closest points using KDTree
    def find_closest_points_with_kdtree(list1, list2):
        # Create KDTree for the second list
        tree = KDTree(list2)

        # Query the tree with all points in the first list
        distances, indices = tree.query(list1)

        # Find the closest pair
        min_distance_index = distances.argmin()
        closest_point_list1 = list1[min_distance_index]
        closest_point_list2 = list2[indices[min_distance_index]]
        return min_distance_index, indices[min_distance_index]

    # For some tiles there will be no reliable shift determinations
    # from cross-correlation linking them to the other tiles,
    # in this case default to using the original
    # shifts implied by the microscope image shifts if groups of tiles
    # TODO use RANSAC algorithm to find the best fit
    if not nx.is_connected(G):
        # if False:
        extraA = []
        islands = sorted(nx.connected_components(G), key=len)
        largest_island = islands.pop()
        largest_island_points = [positions[x] for x in largest_island]
        # anchor = largest_island.pop()
        # for i in np.where(np.all(A == 0, axis=0))[0]:
        for island in islands:
            island_points = [positions[i] for i in island]

            i, j = find_closest_points_with_kdtree(island_points, largest_island_points)

            i = list(island)[i]
            j = list(largest_island)[j]
            x1 = positions[j]
            x2 = positions[i]
            delta = x1 - x2

            # Loop over x and y components
            for jj in range(2):
                # Make new row to add to A matrix
                Arow = np.zeros((N))
                Arow[2 * j + jj] = 1
                Arow[2 * i + jj] = -1
                extraA.append(Arow)

            b[len(b) :] = delta[::-1].tolist()
        A = np.stack(A.tolist() + extraA, axis=0)
    def objective_function(x,A,b,positions,lam):
        obj = np.linalg.norm(A @ x - b)
        obj += lam*np.linalg.norm(positions - x)
        return obj

    def objective_function(x, A, b, positions, lam):
        obj = np.linalg.norm(A @ x - b)
        obj += lam * np.linalg.norm(positions - x)
        return obj

    from scipy.optimize import minimize

    xvec = positions[:, ::-1].ravel()

    x, residuals, rank, s = np.linalg.lstsq(A, np.asarray(b), rcond=-1)
    newy = x[::2]
    newx = x[1::2]

    # The absolute position is unconstrained in least squares minimization
    # Pin the position closest to the origin to its original absolute position
    i = np.argmin(np.linalg.norm(positions, axis=1))
    newx -= newx[i] - positions[i][0]
    newy -= newy[i] - positions[i][1]

    newpositions = np.zeros_like(positions)
    newpositions[:, 0] = newx
    newpositions[:, 1] = newy

    if show:
        plt.show(block=True)
    return newpositions, xcorr, deltas


def plot_individual_cross_correlation(
    images, masks, positions, overlaps, pixel_size, filename_template=None
):
    """
    Generate a plot showing the two masks and the final aligned images for each pair of overlapping tiles.

    Parameters:
    -----------
    images : list of numpy.ndarray
        List of image tiles.
    masks : list of numpy.ndarray
        List of binary masks corresponding to the image tiles.
    positions : numpy.ndarray
        Array of shape (N, 2) containing the (x, y) coordinates for positioning each image in units of microns.
    overlaps : list of tuples
        List of pairs of indices representing which tiles overlap and should be aligned.
    pixel_size : float
        The pixel size in microns.
    binning : int, optional
        Binning factor to reduce the image size. Default is 1.

    Returns:
    --------
    None
    """
    pixels = images[0].shape

    for i, j in overlaps:
        x1 = positions[i]
        x2 = positions[j]

        dx = [int(x) for x in (x2 - x1) / pixel_size * 1e4][::-1]
        i1, i2 = array_overlap(dx[0], pixels[0])
        j1, j2 = array_overlap(dx[1], pixels[1])
        reference_mask = np.zeros_like(masks[i], dtype=bool)
        reference_mask[i1[0] : i1[1], j1[0] : j1[1]] = np.logical_and(masks[i][
            i1[0] : i1[1], j1[0] : j1[1]
        ],masks[j][
            i2[0] : i2[1], j2[0] : j2[1]
        ])
        moving_mask = np.zeros_like(masks[j], dtype=bool)
        moving_mask[i2[0] : i2[1], j2[0] : j2[1]] = np.logical_and(masks[i][
            i1[0] : i1[1], j1[0] : j1[1]
        ],masks[j][
            i2[0] : i2[1], j2[0] : j2[1]
        ])

        detected_shift = np.asarray(phase_cross_correlation(
            images[i], images[j], reference_mask=reference_mask, moving_mask=moving_mask
        )[0])

        fig = plt.figure(figsize=(8, 12))
        axes = fig.subplot_mosaic([["Image1", "Image2"], ["Stitched", "Stitched"]])
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 2], hspace=0.3)

        axes["Image1"].imshow(images[i], cmap="gray",vmin=np.percentile(images[i][masks[i]],1),vmax=np.percentile(images[i][masks[i]],99))
        axes["Image1"].imshow(reference_mask, alpha=0.5, cmap="Reds")
        axes["Image1"].set_title(f"Image {i} with Mask")

        axes["Image2"].imshow(images[j], cmap="gray",vmin=np.percentile(images[j][masks[j]],1),vmax=np.percentile(images[j][masks[j]],99))
        axes["Image2"].imshow(moving_mask, alpha=0.5, cmap="Reds")
        axes["Image2"].set_title(f"Image {j} with Mask")
        stitched_positions = np.array([
            positions[i],
            positions[i] + detected_shift[::-1] * pixel_size * 1e-4,
        ])
        clipped_masks = clip_masks_to_overlaps(
            [masks[i], masks[j]],
            stitched_positions,
            [(0, 1)],
            pixels,
            pixel_size,
        )
        aligned_image, _ = stitch(
            [images[i], images[j]],
            stitched_positions,
            clipped_masks,
            pixel_size,
            smooth=False,
        )
        # aligned_image = np.roll(images[j], shift=(-int(detected_shift[0]), -int(detected_shift[1])), axis=(0, 1))
        # axes[1, 0].imshow(images[i], cmap='gray')
        # axes[1, 0].set_title(f"Image {i}")

        axes["Stitched"].imshow(
            aligned_image,
            vmin=np.percentile(aligned_image, 10),
            vmax=np.percentile(aligned_image, 90),
            cmap="gray",
        )
        axes["Stitched"].set_title(f"Aligned Image {j}")

        for a in axes.values():
            a.axis("off")

        if filename_template is not None:
            fig.savefig(filename_template.format(i, j))
        else:
            plt.tight_layout()
            plt.show()
        fig.clear()


def parse_image_shifts(file_path, superres=1):
    """Parse an image shift file and return image shifts in units of pixels"""
    with open(file_path, "r") as file:
        lines = file.readlines()
    ntilts = int(lines[0].strip())
    result = []
    index = 1
    # while index < len(lines):
    for _ in range(ntilts):
        # Parse the tilt angle
        tilt_angle = float(lines[index].strip())
        index += 1

        # Parse the number of rows
        n = int(lines[index].strip())
        index += 1

        # Parse the n x 3 array
        array = []
        for _ in range(n):
            array.append([float(x) for x in lines[index].strip().split()])
            index += 1
        array = np.asarray(array)
        array[:, :2] /= superres
        # Add the parsed data to the result
        result.append((tilt_angle, array))

    # Sort the list by tilt angle and return
    return sorted(result, key=lambda x: x[0])


def generate_image_file_names_from_template(name, datadir, tilt, positions):
    imagefiles = [
        os.path.join(datadir, "{0}_{1}_{2}.mrc".format(name, tilt, x))
        for x in range(positions.shape[0])
    ]

    return imagefiles


def _resolve_output_target(output_arg):
    """Classify the ``-o/--output`` argument into an output mode.

    Returns
    -------
    mode : {"dir", "mrc", "tiff"}
    path : str or None
        Directory path (mode "dir", or None to derive from --input),
        MRC stack path (mode "mrc"), or single TIFF path with the
        extension forced to ``.tiff`` (mode "tiff").
    slice_index : int or None
        Z-slice parsed from a ``:N`` suffix in mode "mrc"; otherwise None.
    """
    if output_arg is None:
        return "dir", None, None
    base = os.path.basename(output_arg)
    if ":" in base:
        head, idx_str = output_arg.rsplit(":", 1)
        if os.path.basename(head).lower().endswith(".mrc") and idx_str.lstrip("-").isdigit():
            return "mrc", head, int(idx_str)
    if base.lower().endswith(".mrc"):
        return "tiff", os.path.splitext(output_arg)[0] + ".tiff", None
    return "dir", output_arg, None


def setup_outputdir(args):
    if args["output"] is None:
        outputdir = os.path.split(args["input"])[1].replace("*.mrc", "_output")
    else:
        outputdir = args["output"]
    logger.info("Results will be written to: %s", outputdir)
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    return outputdir



def plot_positions(coordinates, fnam=None, fig=None, color="blue"):
    # Extract X and Y coordinates
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]

    # Create a scatter plot
    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        ax = fig.get_axes()[0]
        ax.title("Coordinate Points with Index Annotations")
        ax.xlabel("X Coordinates")
        ax.ylabel("Y Coordinates")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.scatter(x_coords, y_coords, color=color, label="Coordinates")

    # Annotate each point with its index
    for idx, (x, y) in enumerate(coordinates):
        ax.annotate(
            str(idx),
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            ha="center",
            fontsize=8,
        )

    # Set plot title and labels

    ax.legend()

    # Save the plot to a PDF file
    fig.tight_layout()
    if fnam is None:
        plt.show()
    else:
        fig.savefig(fnam)


def plot_positions(coordinates,fnam=None,fig = None,color='blue'):


    # Extract X and Y coordinates
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]

    # Create a scatter plot
    if fig is None:
        fig,ax = plt.subplots(figsize=(10, 8))
    else:
        ax = fig.get_axes()[0]
        ax.title('Coordinate Points with Index Annotations')
        ax.xlabel('X Coordinates')
        ax.ylabel('Y Coordinates')
        ax.axhline(0, color='black',linewidth=0.5)
        ax.axvline(0, color='black',linewidth=0.5)
        ax.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    ax.scatter(x_coords, y_coords, color=color, label='Coordinates')

    # Annotate each point with its index
    for idx, (x, y) in enumerate(coordinates):
        ax.annotate(str(idx), (x, y), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=8)

    # Set plot title and labels

    
    ax.legend()

    # Save the plot to a PDF file
    fig.tight_layout()
    if fnam is None:
        plt.show()
    else:
        fig.savefig(fnam)



def main():
    args = parse_commandline()

    if args["verbose"]:
        log_level = logging.DEBUG
    elif args["quiet"]:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    output_mode, output_target, output_slice = _resolve_output_target(args["output"])

    if output_mode == "dir":
        outdir = setup_outputdir(args)
        mrc_path = None
        tiff_path = None
    else:
        outdir = os.path.dirname(output_target) or "."
        os.makedirs(outdir, exist_ok=True)
        logger.info("Results will be written to: %s", output_target)
        mrc_path = output_target if output_mode == "mrc" else None
        tiff_path = output_target if output_mode == "tiff" else None

    # Binning constant
    binning = int(args["binning"])

    imageshifts = parse_image_shifts(
        args["image_shifts"], superres=args["correctimageshiftfilefactor"]
    )

    if os.path.isdir(args["input"]):
        # ── Directory of per-tile MRC files from beam_mask_motioncorr ──────────
        # Files named  {stem}_{tile_index}_{tilt_angle}.mrc  are grouped by tilt.
        tilts, _tile_paths = _group_per_tile_mrcs(args["input"])
        if not tilts:
            raise FileNotFoundError(
                "No per-tile MRC files matching the expected naming pattern "
                f"found in directory: {args['input']}"
            )
        files = [TileStack(_tile_paths[t]) for t in tilts]
        logger.debug(
            "Directory input: %d tilt(s), %d tile(s) each",
            len(tilts), len(files[0]),
        )

        # Pixel size: prefer explicit --pixel-size, then MRC voxel_size header
        if args["pixel_size"] is not None:
            pixelsize = args["pixel_size"]
        else:
            with mrcfile.mmap(_tile_paths[tilts[0]][0], mode="r", permissive=True) as _m:
                pixelsize = float(_m.voxel_size.x)
            if pixelsize == 0.0:
                raise ValueError(
                    "MRC voxel_size is not set; provide --pixel-size explicitly."
                )
        logger.debug("Pixel size: %.4f Å  |  binning: %d  |  effective: %.4f Å",
                     pixelsize, binning, pixelsize * binning)

    else:
        # ── Glob of per-tilt MRC stacks (original path) ─────────────────────────
        files = sorted(glob.glob(args["input"]))
        if len(files) < 1:
            raise FileNotFoundError("No files matching {0}".format(args["input"]))
        logger.debug("Found %d MRC file(s) matching '%s'", len(files), args["input"])

        # Pixel size: prefer explicit --pixel-size, then MRC voxel_size header
        if args["pixel_size"] is not None:
            pixelsize = args["pixel_size"]
        else:
            with mrcfile.mmap(files[0], mode="r", permissive=True) as _m:
                pixelsize = float(_m.voxel_size.x)
            if pixelsize == 0.0:
                raise ValueError(
                    "MRC voxel_size is not set; provide --pixel-size explicitly."
                )
        logger.debug("Pixel size: %.4f Å  |  binning: %d  |  effective pixel size: %.4f Å",
                     pixelsize, binning, pixelsize * binning)

        # Tilt angle is parsed straight from each filename, e.g.
        # Montage_182-A_-12.0.mrc -> -12.0. -0.0 normalises to 0.0 so tiles
        # at zero tilt aren't treated as a distinct angle.
        tilts = []
        for f in files:
            m = re.search(r"_(-?\d+\.?\d*)\.mrc$", os.path.basename(f))
            if not m:
                raise ValueError(
                    f"Cannot parse tilt angle from filename: {f} "
                    "(expected it to end in '_{tilt_angle}.mrc')"
                )
            t = float(m.group(1))
            tilts.append(0.0 if t == -0.0 else t)

    # Get image shifts from file
    # TODO make SerialEM put this in the mdoc file in microns (not
    # weird TFS units) to obviate this step
    
    tiltsfromfile = [imageshifts[x][0] for x in range(len(imageshifts))]
    imageshifts = [imageshifts[x][1] for x in range(len(imageshifts))]
    reshapedimshifts = np.concatenate(imageshifts)[:, :2] * pixelsize

    if args['ROI'] is None:
        tile_shape = np.asarray(_first_tile_shape(files[0]))
        if (args["rotate"] // 90) % 2:
            # A 90°/270° rotation swaps each tile's height and width.
            tile_shape = tile_shape[::-1]
        tileFOV = pixelsize * tile_shape
        # Global range of tiles in Angstroms
        globalwidth = np.ptp(reshapedimshifts, axis=0)+tileFOV
        globalorigin = np.amin(reshapedimshifts, axis=0)#+tileFOV
    else:
        roi = [float(x) for x in args['ROI']]
        #Convert ROI from pixels (similar to image shifts file) to Angstroms)
        globalorigin = np.array([roi[0],roi[2]])*pixelsize
        globalwidth = np.array([roi[1]-roi[0],roi[3]-roi[2]])*pixelsize

    canvas_px = (globalwidth / pixelsize / binning).astype(int)
    logger.debug("Montage canvas: %d × %d px (binned)  |  %.1f × %.1f µm",
                 canvas_px[0], canvas_px[1],
                 globalwidth[0] * 1e-4, globalwidth[1] * 1e-4)
    logger.info("Processing %d tilt(s)", len(files))

    # --output pointing at an MRC stack (mode "mrc") writes directly into a
    # shared per-series MRC stack instead of the per-tilt TIFFs, skipping
    # JointoMRC.py; mode "tiff" writes a single named TIFF instead of a
    # per-tilt directory. Both require exactly one tilt to be selected.
    # z-slice assignment is independent of files/tilts iteration order — it's
    # always each tilt's 0-based rank when every discovered tilt angle is
    # sorted ascending, matching JointoMRC.py's default tilt-angle ordering.
    tilt_index = args["tilt_index"]
    if output_mode == "mrc" and output_slice is not None and tilt_index is None:
        tilt_index = output_slice

    n_tilts_total = len(tilts)
    z_index_map = {t: z for z, t in enumerate(sorted(tilts))}

    if tilt_index is None and output_mode in ("mrc", "tiff"):
        if n_tilts_total != 1:
            raise ValueError(
                f"--output {args['output']!r} names a single output file "
                f"but {n_tilts_total} tilt(s) were discovered; pass "
                "--tilt-index (or append ':N' to the MRC path) to select one."
            )
        tilt_index = 0

    if tilt_index is not None:
        if not (0 <= tilt_index < n_tilts_total):
            raise ValueError(
                f"--tilt-index {tilt_index} out of range for "
                f"{n_tilts_total} discovered tilt(s)."
            )
        target_tilt = sorted(tilts)[tilt_index]
        keep = [k for k, t in enumerate(tilts) if t == target_tilt]
        files = [files[k] for k in keep]
        tilts = [tilts[k] for k in keep]
        logger.info(
            "Restricting to tilt index %d/%d  (tilt angle %.2f°)",
            tilt_index, n_tilts_total, target_tilt,
        )

    fringe_size = args["fringe_size"]

    # Load template mask if provided
    template_mask = None
    if args["templatemask"] is not None:
        tmask_path = args["templatemask"]
        # Parse optional ":index" suffix, e.g. "file.mrc:7"
        tmask_index = None
        if ":" in os.path.basename(tmask_path):
            tmask_path, idx_str = tmask_path.rsplit(":", 1)
            tmask_index = int(idx_str)
        if tmask_path.lower().endswith(".mrc"):
            with mrcfile.open(tmask_path, "r") as m:
                data = np.asarray(m.data)
                template_mask = data[tmask_index] if tmask_index is not None else data
        else:
            #Need to flip y of tiff files to match mrc conventions
            img = Image.open(tmask_path)
            if tmask_index is not None:
                img.seek(tmask_index)
            template_mask = np.asarray(img)[::-1]
        if binning > 1:
            template_mask = fourier_interpolate(template_mask, [x // binning for x in template_mask.shape])
        # template_mask = make_mask(template_mask,shrinkn=0,medianthreshold=args["maskthreshold"])
        logger.info("Loaded template mask from %s", tmask_path)

    _quiet = not logger.isEnabledFor(logging.INFO)
    for i, (file, tilt) in enumerate(
        tqdm(zip(files, tilts), total=len(files), desc="Stitching montages", disable=_quiet)
    ):
        logger.debug("[%d/%d] %s  (tilt %.1f°)", i + 1, len(files), os.path.basename(file), tilt)
        indx = find_closest_index(tiltsfromfile, tilt)
        positions = imageshifts[indx]

        tiles = None
        logger.debug("  %d tile(s) selected", len(positions))
        # plot_positions(positions[tiles][:,:2],color='k')

        
        mont = montage(
            file,
            outdir,
            positions,
            pixelsize,
            binning=binning,
            skipcrosscorrelation=args["skipcrosscorrelation"],
            montagewidth=globalwidth,
            montageorigin=globalorigin,
            tiles=tiles,
            fringe_size=fringe_size,
            maxshift=args["max_allowed_imshift_correction"],
            maskthreshold=args["maskthreshold"],
            maskabsolutethreshold=args["maskabsolutethreshold"],
            nthreads=args["nthreads"],
            positionfile=args["positionfile"],
            smooth=not args["nosmooth"] and not args["mark_uncovered"],
            mark_uncovered=args["mark_uncovered"],
            template_mask=template_mask,
            min_mean_intensity=args["min_mean_intensity"],
            correct_beam_edges=args["correct_beam_edges"],
            E_plasmon_eV=args["plasmon_energy"],
            voltage_kV=args["voltage"],
            rotate=args["rotate"],
            output_mrc=mrc_path,
            tilt_z_index=z_index_map[tilt] if mrc_path is not None else None,
            n_tilts_total=n_tilts_total if mrc_path is not None else None,
            fileout_path=tiff_path,
        )


if __name__ == "__main__":
    main()
