"""
Interactive tool for choosing beam-mask parameters for beam_mask_motioncorr.

Shows the mean of all sub-frames from a TIFF alongside the masked image with
the largest inscribed square overlaid.  Drag the sliders to tune the erosion
(shrink) and threshold until the square sits cleanly inside the beam.

Pass the chosen values as --mask-shrink and (if needed) --mask-threshold to
beam_mask_motioncorr.
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from PIL import Image

from .Utilities import make_mask
from .beam_mask_motioncorr import largest_inscribed_square


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mean_frame(tiff_path: str) -> np.ndarray:
    """Return the float32 mean over all sub-frames of a multi-page TIFF."""
    with Image.open(tiff_path) as img:
        n = getattr(img, "n_frames", 1)
        img.seek(0)
        acc = np.asarray(img.copy()).astype(np.float32)
        for i in range(1, n):
            img.seek(i)
            acc += np.asarray(img.copy()).astype(np.float32)
    return acc / n


def downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average downsample by integer factor."""
    if factor <= 1:
        return arr
    h, w = arr.shape
    h2, w2 = h // factor * factor, w // factor * factor
    return arr[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_commandline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True, nargs="+",
        help="TIFF file(s). Accepts shell expansion (Frames/*.tif) or a quoted "
             "glob pattern ('Frames/*.tif'). Prev/Next buttons browse all matched files.",
    )
    parser.add_argument(
        "-b", "--binning", type=int, default=4,
        help="Integer display downsample factor (default: 4). "
             "Does not affect the reported parameter values.",
    )
    parser.add_argument(
        "-s", "--mask-shrink", type=int, default=20,
        help="Initial erosion radius in pixels (default: 20).",
    )
    parser.add_argument(
        "-mt", "--mask-threshold", type=float, default=None,
        help="Initial absolute threshold in counts. "
             "If omitted, uses 0.4 × image median.",
    )
    return vars(parser.parse_args())


# ---------------------------------------------------------------------------
# Main interactive tool
# ---------------------------------------------------------------------------

def main():
    args = parse_commandline()

    # Resolve input: shell expansion gives a list; a quoted glob gives a single string
    inputs = args["input"]
    if len(inputs) == 1 and ("*" in inputs[0] or "?" in inputs[0]):
        paths = sorted(glob.glob(inputs[0]))
        if not paths:
            sys.exit(f"No files matched: {inputs[0]}")
    else:
        paths = sorted(inputs)

    state = {"file_idx": 0, "image": None, "median": None}

    # ------------------------------------------------------------------ load
    def load_file(idx: int):
        idx = max(0, min(len(paths) - 1, idx))
        state["file_idx"] = idx
        img = downsample(load_mean_frame(paths[idx]), args["binning"])
        state["image"] = img
        state["median"] = float(np.median(img))
        return img

    image = load_file(0)
    vmin = float(np.percentile(image, 1))
    vmax = float(np.percentile(image, 99))

    initial_shrink = args["mask_shrink"] // args["binning"]
    initial_threshold = (
        args["mask_threshold"]
        if args["mask_threshold"] is not None
        else state["median"] * 0.4
    )

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.42)

    im_raw = axes[0].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(os.path.basename(paths[0]))
    axes[0].axis("off")

    def compute_masked(img, shrink_binned, threshold):
        mask = make_mask(img, shrinkn=shrink_binned, absolutethreshold=threshold)
        r, c, s = largest_inscribed_square(mask)
        return mask, r, c, s

    mask0, r0, c0, s0 = compute_masked(image, initial_shrink, initial_threshold)

    im_masked = axes[1].imshow(np.where(mask0, image, np.nan), cmap="gray",
                               vmin=vmin, vmax=vmax)
    axes[1].set_facecolor("black")
    axes[1].axis("off")

    # Rectangle showing largest inscribed square
    rect_patch = mpatches.Rectangle(
        (c0, r0), s0, s0,
        linewidth=1.5, edgecolor="lime", facecolor="none",
    )
    axes[1].add_patch(rect_patch)

    def make_title(shrink_binned, threshold, median, side_binned, side_full):
        frac = threshold / median if median else 0.0
        return (
            f"shrink = {shrink_binned * args['binning']} px (unbinned)  |  "
            f"threshold = {threshold:.1f}  ({frac:.3f}× median)\n"
            f"Inscribed square: {side_binned} px (display) = "
            f"{side_full} px (unbinned)"
        )

    axes[1].set_title(
        make_title(initial_shrink, initial_threshold,
                   state["median"], s0, s0 * args["binning"]),
        fontsize=9,
    )

    # ------------------------------------------------------------------ sliders
    ax_shrink = plt.axes([0.15, 0.30, 0.70, 0.03])
    shrink_slider = Slider(
        ax_shrink, "Shrink (display px)",
        0, max(200, initial_shrink * 3),
        valinit=initial_shrink, valstep=1, valfmt="%0.0f",
    )

    ax_thresh = plt.axes([0.15, 0.21, 0.70, 0.03])
    threshold_slider = Slider(
        ax_thresh, "Threshold (counts)",
        float(np.percentile(image, 0.5)),
        float(np.percentile(image, 99.5)),
        valinit=initial_threshold,
    )

    cmd_text = fig.text(
        0.15, 0.13, "", fontsize=8, color="navy", family="monospace",
    )
    frac_text = fig.text(
        0.15, 0.09, "", fontsize=8, color="dimgray",
    )

    def update_cmd(shrink_binned, threshold, side_full):
        shrink_unbinned = shrink_binned * args["binning"]
        cmd = (
            f"beam_mask_motioncorr  --mask-shrink {shrink_unbinned}"
            f"  --mask-threshold {threshold:.1f}"
        )
        cmd_text.set_text(cmd)

    def update_frac(threshold, median):
        frac = threshold / median if median else 0.0
        frac_text.set_text(
            f"threshold = {frac:.4f} × median  (median = {median:.1f} counts)"
        )

    def refresh(img, shrink_binned, threshold):
        med = state["median"]
        mask, r, c, s = compute_masked(img, shrink_binned, threshold)
        masked_img = np.where(mask, img, np.nan)
        im_masked.set_data(masked_img)
        rect_patch.set_xy((c, r))
        rect_patch.set_width(s)
        rect_patch.set_height(s)
        axes[1].set_title(
            make_title(shrink_binned, threshold, med, s, s * args["binning"]),
            fontsize=9,
        )
        update_cmd(shrink_binned, threshold, s * args["binning"])
        update_frac(threshold, med)
        fig.canvas.draw_idle()

    update_cmd(initial_shrink, initial_threshold, s0 * args["binning"])
    update_frac(initial_threshold, state["median"])

    def on_slider_change(_val):
        refresh(state["image"], int(shrink_slider.val), threshold_slider.val)

    shrink_slider.on_changed(on_slider_change)
    threshold_slider.on_changed(on_slider_change)

    # ------------------------------------------------------------------ file navigation
    ax_prev = plt.axes([0.25, 0.04, 0.15, 0.05])
    ax_next = plt.axes([0.60, 0.04, 0.15, 0.05])
    btn_prev = Button(ax_prev, "← Prev file")
    btn_next = Button(ax_next, "Next file →")
    file_label = fig.text(
        0.5, 0.055, f"File 1 / {len(paths)}",
        ha="center", fontsize=9,
    )

    def load_and_refresh(idx: int):
        img = load_file(idx)
        new_vmin = float(np.percentile(img, 1))
        new_vmax = float(np.percentile(img, 99))
        im_raw.set_data(img)
        im_raw.set_clim(new_vmin, new_vmax)
        im_masked.set_clim(new_vmin, new_vmax)
        axes[0].set_title(os.path.basename(paths[state["file_idx"]]))
        file_label.set_text(f"File {state['file_idx'] + 1} / {len(paths)}")
        refresh(img, int(shrink_slider.val), threshold_slider.val)

    def on_prev(_e):
        load_and_refresh(state["file_idx"] - 1)

    def on_next(_e):
        load_and_refresh(state["file_idx"] + 1)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    # ------------------------------------------------------------------ initial cmd
    refresh(image, initial_shrink, initial_threshold)

    plt.show()

    # Print final command on close
    shrink_final = int(shrink_slider.val) * args["binning"]
    thresh_final = threshold_slider.val
    print(
        f"\nFinal parameters:\n"
        f"  --mask-shrink {shrink_final}\n"
        f"  --mask-threshold {thresh_final:.1f}\n"
    )


if __name__ == "__main__":
    main()
