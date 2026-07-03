"""
Interactive tool for choosing beam-mask parameters for beam_mask_motioncorr.

Shows the mean of all sub-frames from a TIFF alongside the masked image with
the largest inscribed square overlaid.  Drag the sliders to tune the erosion
(shrink) and threshold until the square sits cleanly inside the beam.

In the masked panel: enable Paint mode, then left-drag to add regions to the
mask or right-drag to remove them.  The brush radius slider controls brush size.

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

    # ------------------------------------------------------------------ paint state
    paint = {
        "active": False,
        "pressing": False,
        "mode": "add",          # "add" or "remove" — set on mouse press
        "add": np.zeros(image.shape, dtype=bool),
        "remove": np.zeros(image.shape, dtype=bool),
        "computed_mask": None,  # stored by refresh() so paint_at can combine
    }

    def apply_paint_overlay(computed_mask):
        return (computed_mask | paint["add"]) & ~paint["remove"]

    def reset_paint_arrays(shape):
        paint["add"] = np.zeros(shape, dtype=bool)
        paint["remove"] = np.zeros(shape, dtype=bool)

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.50)

    im_raw = axes[0].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(os.path.basename(paths[0]))
    axes[0].axis("off")

    def compute_masked(img, shrink_binned, threshold):
        mask = make_mask(img, shrinkn=shrink_binned, absolutethreshold=threshold)
        r, c, s = largest_inscribed_square(mask)
        return mask, r, c, s

    mask0, r0, c0, s0 = compute_masked(image, initial_shrink, initial_threshold)
    paint["computed_mask"] = mask0

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

    # Brush cursor circle (visible in paint mode)
    brush_cursor = plt.Circle((0, 0), 5, fill=False, color="yellow",
                               linestyle="--", linewidth=1.2, visible=False)
    axes[1].add_patch(brush_cursor)

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
    ax_shrink = plt.axes([0.15, 0.40, 0.70, 0.03])
    shrink_slider = Slider(
        ax_shrink, "Shrink (display px)",
        0, max(200, initial_shrink * 3),
        valinit=initial_shrink, valstep=1, valfmt="%0.0f",
    )

    ax_thresh = plt.axes([0.15, 0.31, 0.70, 0.03])
    threshold_slider = Slider(
        ax_thresh, "Threshold (counts)",
        float(np.percentile(image, 0.5)),
        float(np.percentile(image, 99.5)),
        valinit=initial_threshold,
    )

    ax_brush = plt.axes([0.15, 0.22, 0.70, 0.03])
    brush_slider = Slider(
        ax_brush, "Brush radius (px)",
        1, 80,
        valinit=5, valstep=1, valfmt="%0.0f",
    )

    cmd_text = fig.text(
        0.15, 0.15, "", fontsize=8, color="navy", family="monospace",
    )
    frac_text = fig.text(
        0.15, 0.11, "", fontsize=8, color="dimgray",
    )
    paint_hint = fig.text(
        0.15, 0.07,
        "Paint mode: left-drag = add to mask   right-drag = remove from mask",
        fontsize=8, color="dimgray", visible=False,
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
        computed_mask, r, c, s = compute_masked(img, shrink_binned, threshold)
        paint["computed_mask"] = computed_mask
        final_mask = apply_paint_overlay(computed_mask)
        masked_img = np.where(final_mask, img, np.nan)
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

    def on_brush_changed(_val):
        brush_cursor.set_radius(brush_slider.val)
        fig.canvas.draw_idle()

    shrink_slider.on_changed(on_slider_change)
    threshold_slider.on_changed(on_slider_change)
    brush_slider.on_changed(on_brush_changed)

    # ------------------------------------------------------------------ paint buttons
    ax_paint = plt.axes([0.08, 0.04, 0.14, 0.05])
    btn_paint = Button(ax_paint, "Paint: OFF", color="lightgray", hovercolor="silver")

    ax_reset_paint = plt.axes([0.24, 0.04, 0.14, 0.05])
    btn_reset_paint = Button(ax_reset_paint, "Reset paint")

    def on_paint_toggle(_e):
        paint["active"] = not paint["active"]
        if paint["active"]:
            btn_paint.label.set_text("Paint: ON")
            btn_paint.ax.set_facecolor("lightgreen")
            paint_hint.set_visible(True)
            brush_cursor.set_visible(False)  # shown on mouse-enter
        else:
            btn_paint.label.set_text("Paint: OFF")
            btn_paint.ax.set_facecolor("lightgray")
            paint_hint.set_visible(False)
            brush_cursor.set_visible(False)
        fig.canvas.draw_idle()

    def on_reset_paint(_e):
        reset_paint_arrays(state["image"].shape)
        final_mask = apply_paint_overlay(paint["computed_mask"])
        im_masked.set_data(np.where(final_mask, state["image"], np.nan))
        fig.canvas.draw_idle()

    btn_paint.on_clicked(on_paint_toggle)
    btn_reset_paint.on_clicked(on_reset_paint)

    # ------------------------------------------------------------------ file navigation
    ax_prev = plt.axes([0.50, 0.04, 0.15, 0.05])
    ax_next = plt.axes([0.76, 0.04, 0.15, 0.05])
    btn_prev = Button(ax_prev, "← Prev file")
    btn_next = Button(ax_next, "Next file →")
    file_label = fig.text(
        0.5, 0.055, f"File 1 / {len(paths)}",
        ha="center", fontsize=9,
    )

    def load_and_refresh(idx: int):
        img = load_file(idx)
        reset_paint_arrays(img.shape)
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

    # ------------------------------------------------------------------ paint mouse events
    def paint_at(xdata, ydata, button):
        if xdata is None or ydata is None:
            return
        img = state["image"]
        h, w = img.shape
        r = int(brush_slider.val)
        cx, cy = int(round(xdata)), int(round(ydata))
        ys, xs = np.ogrid[:h, :w]
        circle = (ys - cy) ** 2 + (xs - cx) ** 2 <= r ** 2
        if button == 1:  # left click — add to mask
            paint["add"][circle] = True
            paint["remove"][circle] = False
        else:            # right click — remove from mask
            paint["remove"][circle] = True
            paint["add"][circle] = False
        final_mask = apply_paint_overlay(paint["computed_mask"])
        im_masked.set_data(np.where(final_mask, img, np.nan))
        fig.canvas.draw_idle()

    def on_mouse_press(event):
        if not paint["active"] or event.inaxes != axes[1]:
            return
        paint["pressing"] = True
        paint["mode"] = "add" if event.button == 1 else "remove"
        paint_at(event.xdata, event.ydata, event.button)

    def on_mouse_release(event):
        paint["pressing"] = False

    def on_mouse_motion(event):
        if not paint["active"]:
            return
        in_panel = event.inaxes == axes[1] and event.xdata is not None
        brush_cursor.set_visible(in_panel)
        if in_panel:
            brush_cursor.set_center((event.xdata, event.ydata))
            brush_cursor.set_radius(brush_slider.val)
            if paint["pressing"]:
                btn = 1 if paint["mode"] == "add" else 3
                paint_at(event.xdata, event.ydata, btn)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_mouse_press)
    fig.canvas.mpl_connect("button_release_event", on_mouse_release)
    fig.canvas.mpl_connect("motion_notify_event", on_mouse_motion)

    # ------------------------------------------------------------------ initial render
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
