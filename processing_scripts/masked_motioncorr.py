"""Masked FFT-based motion correction for cryo-TEM movies.

Implements phase-correlation alignment with spatial masking and
sub-pixel Fourier-domain frame shifting.
"""


from __future__ import annotations
import argparse
import mrcfile
from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
import numpy as np
import os
from PIL import Image
from .Utilities import make_mask
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_multipage_tiff_pil(path: str) -> np.ndarray:
    """Load a multi-page TIFF via PIL into an array of shape (n_frames, h, w[, c])."""
    with Image.open(path) as img:
        n_frames = getattr(img, "n_frames", 1)
        img.seek(0)
        first = np.asarray(img)

        out = np.empty((n_frames, *first.shape), dtype=first.dtype)
        out[0] = first

        for i in range(1, n_frames):
            img.seek(i)
            out[i] = np.asarray(img)

    return out


def _parabolic_subpixel(cc: np.ndarray, y: int, x: int) -> tuple[float, float]:
    """Return subpixel (dy, dx) offset around integer peak via 1D parabola fits."""
    h, w = cc.shape

    ym1 = (y - 1) % h
    yp1 = (y + 1) % h
    xm1 = (x - 1) % w
    xp1 = (x + 1) % w

    c0 = cc[y, x]

    cy_m, cy_p = cc[ym1, x], cc[yp1, x]
    denom_y = 2.0 * (cy_m - 2.0 * c0 + cy_p)
    dy = 0.0 if denom_y == 0.0 else (cy_m - cy_p) / denom_y

    cx_m, cx_p = cc[y, xm1], cc[y, xp1]
    denom_x = 2.0 * (cx_m - 2.0 * c0 + cx_p)
    dx = 0.0 if denom_x == 0.0 else (cx_m - cx_p) / denom_x

    # Clamp for robustness on noisy surfaces.
    dy = float(np.clip(dy, -1.0, 1.0))
    dx = float(np.clip(dx, -1.0, 1.0))
    return dy, dx


def _phase_corr_shift(
    frame: np.ndarray,
    ref_fft_conj: np.ndarray,
    mask: np.ndarray,
    eps: float,
) -> tuple[float, float]:
    """Estimate shift of frame relative to reference represented by conj(FFT(ref*mask))."""
    f_fft = np.fft.rfft2(frame * mask)
    cross = f_fft * ref_fft_conj
    cross /= np.maximum(np.abs(cross), eps)
    cc = np.fft.irfft2(cross, s=frame.shape)

    iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
    suby, subx = _parabolic_subpixel(cc, iy, ix)

    h, w = frame.shape
    sy = float(iy if iy <= h // 2 else iy - h) + suby
    sx = float(ix if ix <= w // 2 else ix - w) + subx
    return sy, sx


def _fourier_shift_2d(frame: np.ndarray, shift_yx: tuple[float, float]) -> np.ndarray:
    """Apply subpixel shift (dy, dx) using Fourier shift theorem."""
    dy, dx = shift_yx
    h, w = frame.shape

    fy = np.fft.fftfreq(h)
    fx = np.fft.rfftfreq(w)
    phase = np.exp(-2j * np.pi * (dy * fy[:, None] + dx * fx[None, :]))

    f_fft = np.fft.rfft2(frame)
    shifted = np.fft.irfft2(f_fft * phase, s=frame.shape)
    return shifted.astype(frame.dtype, copy=False)


def _gaussian_blur_fft(frame: np.ndarray, gauss_kernel_rfft: np.ndarray) -> np.ndarray:
    """Apply a Gaussian blur in Fourier space using a precomputed RFFT kernel."""
    return np.fft.irfft2(np.fft.rfft2(frame) * gauss_kernel_rfft, s=frame.shape).astype(
        frame.dtype, copy=False
    )


def _subpixel_shift_from_xcorr_center_window(
    xcorr: np.ndarray,
    window_size: int = 16,
    upsample_size: int = 512,
) -> tuple[float, float]:
    """Estimate subpixel shift from a center-windowed Fourier interpolation of xcorr."""
    h, w = xcorr.shape
    cy = h // 2
    cx = w // 2
    half = window_size // 2
    up_half = upsample_size // 2

    win = xcorr[cy - half : cy + half, cx - half : cx + half]

    win_fft = np.fft.fftshift(np.fft.fft2(win))
    up_fft = np.zeros((upsample_size, upsample_size), dtype=win_fft.dtype)
    up_fft[up_half - half : up_half + half, up_half - half : up_half + half] = win_fft
    interp = np.fft.ifft2(np.fft.ifftshift(up_fft)).real

    py, px = np.unravel_index(np.argmax(interp), interp.shape)
    scale = float(window_size) / float(upsample_size)
    peak_dy = (float(py) - float(up_half)) * scale
    peak_dx = (float(px) - float(up_half)) * scale

    # Convert correlation-peak offset to translation to apply to moving image.
    return -peak_dy, -peak_dx


def motioncorr_masked(
    movie: np.ndarray,
    mask: np.ndarray | None = None,
    n_iter: int = 2,
    reference: str = "mean",
    return_shifts: bool = True,
    eps: float = 1e-9,
    b_factor: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Align cryo-TEM movie frames using masked FFT phase correlation.

    Parameters
    ----------
    movie
        Array of shape (n_frames, height, width).
    mask
        2D multiplicative mask (same height/width). If None, uses all ones.
        Values are used as weights and should usually be in [0, 1].
    n_iter
        Number of refinement iterations.
    reference
        "mean" for iterative mean-template alignment, or "first" for first-frame anchor.
    return_shifts
        If True, return `(aligned_movie, shifts_yx)` where `shifts_yx` has shape (n_frames, 2).
    eps
        Small constant for spectral normalization stability.
    b_factor
        Optional Gaussian B-factor in pixel^2 applied to frames used for shift estimation.
        If None or <= 0, no blur is applied.

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        Aligned movie, optionally with estimated shifts (dy, dx) per frame.
    """
    movie = np.asarray(movie)
    if movie.ndim != 3:
        raise ValueError("movie must have shape (n_frames, height, width)")

    n, h, w = movie.shape
    if n == 0:
        out = movie.copy()
        shifts = np.zeros((0, 2), dtype=np.float32)
        return (out, shifts) if return_shifts else out

    work_dtype = np.float32 if movie.dtype.kind in "uif" else np.float64
    orig = movie.astype(work_dtype, copy=False)

    aligned = orig.copy()
    shifts = np.zeros((n, 2), dtype=np.float32)

    if reference not in {"mean", "first"}:
        raise ValueError("reference must be 'mean' or 'first'")

    if mask is None:
        # m = np.ones((h, w), dtype=work_dtype)
        m = make_mask(np.mean(aligned, axis=0))
        # fig,ax = plt.subplots()
        # ax.imshow(np.where(m,1,0))
        # plt.show(block=True)
        # import sys;sys.exit()
    else:
        m = np.asarray(mask, dtype=work_dtype)
        if m.shape != (h, w):
            raise ValueError("mask must have shape (height, width)")

    blur_for_alignment = b_factor is not None and b_factor > 0
    if blur_for_alignment:
        fy = np.fft.fftfreq(h)
        fx = np.fft.rfftfreq(w)
        # Transfer function equivalent to Gaussian in real space.
        # b_factor is interpreted in pixel^2.
        gauss_kernel_rfft = np.exp(
            -float(b_factor) * (fy[:, None] ** 2 + fx[None, :] ** 2) / 2
        )
        orig_for_align = np.empty_like(orig)
        for i in range(n):
            orig_for_align[i] = _gaussian_blur_fft(orig[i], gauss_kernel_rfft)
    else:
        orig_for_align = orig

    for iter in tqdm(range(max(1, int(n_iter))), desc="motioncor iterations"):
        ref = aligned.mean(axis=0, dtype=work_dtype)
        if blur_for_alignment:
            ref_for_align = _gaussian_blur_fft(ref, gauss_kernel_rfft)
        else:
            ref_for_align = ref
        # fig,ax = plt.subplots(ncols=2)
        # ax[0].imshow(ref_for_align)
        # ax[1].imshow(orig_for_align[0])
        # plt.show()

        # ref_fft_conj = np.conj(np.fft.rfft2(ref * m))
        new_aligned = np.empty_like(aligned)
        for i in range(n):
            ref = np.concatenate((aligned[:i], aligned[i + 1 :])).mean(
                axis=0, dtype=work_dtype
            )
            ref = _gaussian_blur_fft(ref, gauss_kernel_rfft)
            xcorr = cross_correlate_masked(ref, orig_for_align[i], m, m)
            shifts[i, :] = _subpixel_shift_from_xcorr_center_window(xcorr)
            # shifts[i,0] = detected_shift[0]
            # shifts[i,1] = detected_shift[1]
            print(shifts)
            # Align frame to template by inverse of measured displacement.
            new_aligned[i] = _fourier_shift_2d(orig[i], -shifts[i])
        aligned = new_aligned
    if return_shifts:
        return aligned.astype(movie.dtype, copy=False), shifts
    return aligned.astype(movie.dtype, copy=False)


def test_motioncorr_masked_synthetic_rigid_shift(
    poisson_counts: float = 5_000.0, b_factor=100, pixel_size=1.0
) -> None:
    """Synthetic regression test: rigid shifts + masking + noise."""
    rng = np.random.default_rng(0)

    h, w = 512, 512
    n_frames = 4
    yy, xx = np.mgrid[0:h, 0:w]

    # Structured reference with broad + fine features for stable correlation.
    # base = (
    #     0.8 * np.exp(-((yy - 28.0) ** 2 + (xx - 36.0) ** 2) / (2.0 * 7.0**2))
    #     + 1.1 * np.exp(-((yy - 64.0) ** 2 + (xx - 58.0) ** 2) / (2.0 * 10.0**2))
    #     + 0.35 * np.exp(-((yy - 48.0) ** 2 + (xx - 22.0) ** 2) / (2.0 * 5.5**2))
    # ).astype(np.float32)
    from skimage import data

    base = np.sum(data.astronaut(), axis=2).astype(float)

    true_abs_shifts = np.stack(
        [
            np.linspace(-2.4, 2.8, n_frames, dtype=np.float32),
            np.linspace(1.9, -2.1, n_frames, dtype=np.float32),
        ],
        axis=1,
    )

    movie = np.empty((n_frames, h, w), dtype=np.float32)
    for i, (dy, dx) in enumerate(true_abs_shifts):
        movie[i] = _fourier_shift_2d(base, (float(dy), float(dx)))

    cy, cx = h / 2.0, w / 2.0
    r2 = (0.36 * min(h, w)) ** 2
    mask = (((yy - cy) ** 2 + (xx - cx) ** 2) <= r2).astype(np.float32)
    # if poisson_counts <= 0:
    #     raise ValueError("poisson_counts must be > 0")
    movie = np.clip(movie * mask, 0.0, None)
    frame_sums = np.maximum(movie.sum(axis=(1, 2), keepdims=True), 1e-12)
    if poisson_counts > 0:
        movie = rng.poisson(movie * (poisson_counts / frame_sums)).astype(np.float32)

    aligned, shifts = motioncorr_masked(
        movie, mask=mask, n_iter=2, reference="first", return_shifts=True, b_factor=2
    )
    expected_shifts = true_abs_shifts[0] - true_abs_shifts
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    im0 = ax[0, 0].imshow(base, cmap="gray")
    ax[0, 0].set_title("Original Unshifted")
    ax[0, 0].axis("off")
    fig.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    im1 = ax[0, 1].imshow(movie.mean(axis=0), cmap="gray")
    ax[0, 1].set_title("Mean Unaligned")
    ax[0, 1].axis("off")
    fig.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    im2 = ax[1, 0].imshow(aligned.mean(axis=0), cmap="gray")
    ax[1, 0].set_title("Mean Aligned")
    ax[1, 0].axis("off")
    fig.colorbar(im2, ax=ax[1, 0], fraction=0.046, pad=0.04)

    frame_idx = np.arange(n_frames)
    ax[1, 1].plot(frame_idx, expected_shifts[:, 0], "C0-", label="Applied dy")
    ax[1, 1].plot(frame_idx, expected_shifts[:, 1], "C1-", label="Applied dx")
    ax[1, 1].plot(
        frame_idx, -(shifts[:, 0] - shifts[0, 0]), "C0--", label="Reconstructed dy"
    )
    ax[1, 1].plot(
        frame_idx, -(shifts[:, 1] - shifts[0, 1]), "C1--", label="Reconstructed dx"
    )
    ax[1, 1].set_title("Applied vs Reconstructed Shifts")
    ax[1, 1].set_xlabel("Frame")
    ax[1, 1].set_ylabel("Shift (px)")
    ax[1, 1].legend(loc="best")
    plt.show(block=True)
    plt.close(fig)
    # phase_cross_correlation(ref, moving) returns the shift to apply to moving.
    np.testing.assert_allclose(shifts, expected_shifts, atol=0.45, rtol=0.0)

    pre = np.mean((movie[1:] - movie[0]) ** 2 * mask[None, :, :])
    post = np.mean((aligned[1:] - aligned[0]) ** 2 * mask[None, :, :])
    assert post < 0.35 * pre


__all__ = ["motioncorr_masked"]


def _main() -> None:
    # test_motioncorr_masked_synthetic_rigid_shift(1e4)
    # import sys;sys.exit()
    # parser = argparse.ArgumentParser(
    #     description="Run masked motion correction on an input movie and write output movie."
    # )
    # parser.add_argument("input_movie", type=str, help="Input movie file (.mrc)")
    # parser.add_argument("output_movie", type=str, help="Output movie file (.mrc)")
    # parser.add_argument("Bfact", type=float, help="B factor for alignment in pixels^2, default 100.",required=False,
    #     default=100.0,)
    # parser.add_argument("Pixel size", type=float, help="Pixel size in Å, default 1.",required=False,
    #     default=1.0,)
    # args = parser.parse_args()

    # input = args.input_movie
    # output = args.output_movie

    input = "/home/hgbrown/Desktop/project/Hamish/20251112_ApoFerritin_SubTomo/Frames/binned/Montage_13-A_13-A_000_0.0.tif"
    output = "/home/hgbrown/Desktop/project/Hamish/test_align.mrc"
    b_factor = 100.0
    pixel_size = 6.852 / 2

    extension = os.path.splitext(input)[1]
    if extension == ".mrc":
        with mrcfile.mmap(input, mode="r") as mrc_in:
            movie = np.asarray(mrc_in.data)
    elif extension == ".tif" or extension == ".tiff":
        movie = load_multipage_tiff_pil(input)
    else:
        raise ValueError(
            f"Unsupported input format '{extension}'. Please provide a .tif/.tiff or .mrc file."
        )
    aligned = motioncorr_masked(
        movie, b_factor=b_factor , return_shifts=False
    )
    aligned = np.sum(aligned, axis=0)
    extension = os.path.splitext(output)[1]
    if extension == ".mrc":
        with mrcfile.new(aligned, overwrite=True) as mrc_out:
            mrc_out.set_data(aligned)
    elif extension == ".tif" or extension == ".tiff":
        Image.fromarray(aligned.astype(movie.dtype)).save(output)


if __name__ == "__main__":
    _main()
