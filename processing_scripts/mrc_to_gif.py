#!/usr/bin/env python3
"""
Convert an MRC tilt series stack to an animated GIF.

Usage:
    python mrc_to_gif.py -i tiltseries.mrc -o tiltseries.gif
    python mrc_to_gif.py -i tiltseries.mrc -o tiltseries.gif -s 0.25 --fps 10
    python mrc_to_gif.py -i tiltseries.mrc -o tiltseries.gif -s 512      # longest edge = 512 px
    python mrc_to_gif.py -i tiltseries.mrc -o tiltseries.gif --rawtlt tiltseries.rawtlt
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import mrcfile
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    p = argparse.ArgumentParser(
        description="Read an MRC tilt series stack, downsample, and write an animated GIF."
    )
    p.add_argument("-i", "--input", required=True, help="Input MRC file.")
    p.add_argument("-o", "--output", default=None,
                   help="Output GIF file. Defaults to the input path with extension replaced by .gif.")
    p.add_argument(
        "-s", "--scale", default=0.25,
        help="Downscale factor (float fraction, e.g. 0.25) or target size for the "
             "longest edge in pixels (int > 1). Default: 0.25",
    )
    p.add_argument(
        "--fps", type=float, default=8,
        help="Frames per second in the output GIF. Default: 8",
    )
    p.add_argument(
        "--loop", type=int, default=0,
        help="Number of GIF loops (0 = loop forever). Default: 0",
    )
    p.add_argument(
        "--contrast", type=float, default=3.0,
        help="Clip data at +/-N standard deviations from the mean before scaling "
             "to 8-bit. Higher = more contrast, fewer saturated pixels. Default: 3",
    )
    p.add_argument(
        "--cmap", choices=["gray", "hot", "plasma"], default="gray",
        help="Colourmap to apply. Default: gray",
    )
    p.add_argument(
        "--stat-frames", type=int, default=5,
        help="Number of evenly-spaced frames used to estimate global mean/std "
             "for contrast normalisation (avoids loading the full stack). Default: 5",
    )
    p.add_argument(
        "--rawtlt", default=None,
        help="Path to a .rawtlt file (one tilt angle per line). When provided, "
             "each frame is labelled with its tilt angle in degrees.",
    )
    p.add_argument(
        "--label-size", type=int, default=None,
        help="Font size for the tilt-angle label in output pixels. "
             "Defaults to ~4%% of the output image height.",
    )
    p.add_argument(
        "--label-color", default="white",
        help="Label text colour (any PIL colour name or hex, e.g. 'white', '#ffff00'). "
             "Default: white",
    )
    p.add_argument(
        "--label-pos", choices=["tl", "tr", "bl", "br"], default="bl",
        help="Label position: tl=top-left, tr=top-right, bl=bottom-left, br=bottom-right. "
             "Default: bl",
    )
    p.add_argument(
        "--axis", choices=["z", "y", "x"], default="z",
        help="Axis to slice along. z=tilt series (default), y=tomogram Y slices, x=tomogram X slices.",
    )
    p.add_argument(
        "--bin", type=int, default=None, dest="bin_factor",
        help="Bin the volume by this integer factor using IMOD's binvol before processing. "
             "For --axis z (tilt series) bins XY only; for y/x bins all 3 dimensions. "
             "When set, --scale is still applied afterwards for any additional resize.",
    )
    p.add_argument(
        "--rotate", type=int, choices=[0, 90, 180, 270], default=0,
        help="Rotate each frame by this many degrees counter-clockwise. Default: 0",
    )
    p.add_argument(
        "--bounce", action="store_true",
        help="Play frames forward then backward (ping-pong) instead of looping from the start.",
    )
    p.add_argument(
        "--aperpix", type=float, default=None,
        help="Input pixel size in Angstroms/pixel. Required for scale bar.",
    )
    p.add_argument(
        "--scalebar", type=float, default=None,
        help="Scale bar length in nm. If --aperpix is given but --scalebar is omitted, "
             "a round value ~15%% of the image width is chosen automatically.",
    )
    p.add_argument(
        "--scalebar-pos", choices=["tl", "tr", "bl", "br"], default="br",
        help="Scale bar position. Default: br",
    )
    p.add_argument(
        "--scalebar-color", default="white",
        help="Scale bar and label colour. Default: white",
    )
    return p.parse_args()


def compute_output_size(h, w, scale_arg):
    try:
        scale = float(scale_arg)
    except ValueError:
        sys.exit("Cannot parse --scale value: {0}".format(scale_arg))

    if 0 < scale <= 1.0:
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
    else:
        target = int(scale)
        ratio = target / max(h, w)
        new_h = max(1, int(round(h * ratio)))
        new_w = max(1, int(round(w * ratio)))

    return new_h, new_w


def load_tilt_angles(rawtlt_path, nframes):
    """Read tilt angles from a .rawtlt file.
    Accepts single-value lines (angle) or two-value lines (angle acquisition_order).
    """
    with open(rawtlt_path) as f:
        angles = [float(line.split()[0]) for line in f if line.strip()]
    if len(angles) != nframes:
        print(
            f"  Warning: {rawtlt_path} has {len(angles)} angles but MRC has "
            f"{nframes} frames. Angles will be truncated or padded with None."
        )
        angles = (angles + [None] * nframes)[:nframes]
    return angles


def _load_font(font_size):
    for font_path in [
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(font_path, size=font_size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def _color_for_mode(color, mode):
    """Convert a PIL color name/hex to the right fill type for the image mode."""
    from PIL import ImageColor
    if mode == "L":
        return ImageColor.getcolor(color, "L")
    return color  # RGB accepts strings directly


def _draw_shadowed_text(draw, xy, text, font, fg, shadow):
    x, y = xy
    for dx, dy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        draw.text((x + dx, y + dy), text, font=font, fill=shadow)
    draw.text((x, y), text, font=font, fill=fg)


def nice_scalebar_nm(image_width_px, aperpix_output):
    """Pick a round scale bar length (nm) that is ~15% of the image width."""
    target_px = image_width_px * 0.15
    target_nm = target_px * aperpix_output / 10.0  # Å → nm
    # Round to a nice value: 1, 2, 5, 10, 20, 50, 100, 200, 500, …
    magnitude = 10 ** np.floor(np.log10(target_nm))
    for nice in [1, 2, 5, 10]:
        candidate = nice * magnitude
        if candidate >= target_nm * 0.5:
            return candidate
    return magnitude * 10


def add_scalebar(img, bar_nm, aperpix_output, font_size, color, position):
    """Draw a scale bar and label directly onto the image (L or RGB, in-place)."""
    bar_px = int(round(bar_nm * 10 / aperpix_output))  # nm→Å, then ÷ Å/px
    if bar_px < 2:
        return img

    label = f"{bar_nm:g} nm" if bar_nm < 1000 else f"{bar_nm/1000:g} µm"
    fg = _color_for_mode(color, img.mode)
    bg = _color_for_mode("black", img.mode)

    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)
    tw, th = _text_size(draw, label, font)

    W, H = img.size
    margin = max(6, font_size // 3)
    bar_h = max(2, font_size // 5)

    block_w = max(bar_px, tw)
    x0 = margin if position in ("tl", "bl") else W - block_w - margin
    y0 = margin if position in ("tl", "tr") else H - (bar_h + 2 + th) - margin

    bar_x0 = x0 + (block_w - bar_px) // 2
    bar_x1 = bar_x0 + bar_px
    bar_y1 = y0 + bar_h

    for dx, dy in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        draw.rectangle([bar_x0 + dx, y0 + dy, bar_x1 + dx, bar_y1 + dy], fill=bg)
    draw.rectangle([bar_x0, y0, bar_x1, bar_y1], fill=fg)

    _draw_shadowed_text(draw, (x0 + (block_w - tw) // 2, y0 + bar_h + 2), label, font, fg, bg)
    return img


def add_label(img, text, font_size, color, position):
    """Draw a tilt-angle label directly onto the image (L or RGB, in-place)."""
    fg = _color_for_mode(color, img.mode)
    bg = _color_for_mode("black", img.mode)

    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)
    tw, th = _text_size(draw, text, font)

    W, H = img.size
    margin = max(4, font_size // 4)
    x = margin if position in ("tl", "bl") else W - tw - margin
    y = margin if position in ("tl", "tr") else H - th - margin

    _draw_shadowed_text(draw, (x, y), text, font, fg, bg)
    return img


def to_gif_frame(img):
    """Convert L or RGB image to GIF palette mode without dithering."""
    if img.mode == "L":
        # Direct copy: pixel value n → palette index n → grey(n, n, n). Lossless.
        img_p = Image.frombytes("P", img.size, img.tobytes())
        img_p.putpalette([i for i in range(256) for _ in range(3)])
        return img_p
    if img.mode == "RGB":
        return img.quantize(colors=256, dither=0)
    return img  # already P


def make_frame_getter(data, axis):
    """Return (nframes, height, width, get_frame_fn) for the chosen slice axis."""
    nz, ny, nx = data.shape
    if axis == "z":
        return nz, ny, nx, lambda i: data[i]
    if axis == "y":
        return ny, nz, nx, lambda i: data[:, i, :]
    # axis == "x"
    return nx, nz, ny, lambda i: data[:, :, i]


def estimate_stats(nframes, get_frame, n_sample, n_std):
    """Sample a few evenly-spaced frames to estimate global clipping limits."""
    indices = np.linspace(0, nframes - 1, min(n_sample, nframes), dtype=int)
    sample = np.concatenate([get_frame(i).ravel() for i in indices]).astype(np.float32)
    mu, sigma = sample.mean(), sample.std()
    return float(mu - n_std * sigma), float(mu + n_std * sigma)


def frame_to_pil(frame, lo, hi, new_w, new_h, cmap_name):
    f = np.clip(frame.astype(np.float32), lo, hi)
    u8 = ((f - lo) / (hi - lo) * 255).astype(np.uint8)
    img = Image.fromarray(u8, mode="L").resize((new_w, new_h), Image.LANCZOS)

    if cmap_name == "gray":
        return img  # stay in L mode — lossless, no palette quantization

    import matplotlib.cm as cm
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(np.array(img) / 255.0)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def run_binvol(input_path, bin_factor, axis):
    """Bin an MRC file using IMOD binvol; returns path to a temporary binned file."""
    if not shutil.which("binvol"):
        sys.exit("binvol not found in PATH — is IMOD installed and on PATH?")

    tmp = tempfile.NamedTemporaryFile(suffix=".mrc", delete=False)
    tmp.close()

    # Tilt series: bin XY only to preserve frame count; tomogram: full 3-D bin.
    if axis == "z":
        cmd = ["binvol", "-binXY", str(bin_factor), input_path, tmp.name]
    else:
        cmd = ["binvol", "-bin", str(bin_factor), input_path, tmp.name]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        Path(tmp.name).unlink(missing_ok=True)
        sys.exit(f"binvol failed:\n{result.stderr or result.stdout}")

    return tmp.name


def main():
    args = parse_args()
    if args.output is None:
        args.output = str(Path(args.input).with_suffix(".gif"))
        print(f"Output: {args.output}")
    duration_ms = int(round(1000 / args.fps))

    binned_tmp = None
    mrc_path = args.input
    if args.bin_factor is not None:
        print(f"Binning with IMOD binvol (factor {args.bin_factor}) ...")
        binned_tmp = run_binvol(args.input, args.bin_factor, args.axis)
        mrc_path = binned_tmp

    print(f"Reading {mrc_path} ...")
    mrc = mrcfile.mmap(mrc_path, mode="r", permissive=True)
    data = mrc.data

    if data.ndim == 2:
        data = data[np.newaxis]

    nframes, h, w, get_frame = make_frame_getter(data, args.axis)
    axis_label = {"z": "Z (tilt series)", "y": "Y slices", "x": "X slices"}[args.axis]
    print(f"  {data.shape}, dtype {data.dtype}  →  {nframes} frames along {axis_label}, {h} × {w} px each")

    new_h, new_w = compute_output_size(h, w, args.scale)
    print(f"  Output size: {new_h} × {new_w} px  ({new_h/h:.1%} of original)")

    tilt_angles = None
    if args.rawtlt:
        print(f"Loading tilt angles from {args.rawtlt} ...")
        tilt_angles = load_tilt_angles(args.rawtlt, nframes)
        print(f"  {len(tilt_angles)} angles: {tilt_angles[0]:.1f}° … {tilt_angles[-1]:.1f}°")

    # Scale bar setup — compute effective pixel size of the output image
    scalebar_nm = None
    aperpix_output = None
    if args.aperpix is not None:
        scale_factor = new_w / w  # output/input pixel ratio
        aperpix_output = args.aperpix / scale_factor  # Å/px in the output image
        scalebar_nm = args.scalebar if args.scalebar is not None else nice_scalebar_nm(new_w, aperpix_output)
        bar_px = int(round(scalebar_nm * 10 / aperpix_output))
        print(f"  Scale bar: {scalebar_nm:g} nm = {bar_px} output pixels  "
              f"(effective pixel size {aperpix_output:.2f} Å/px)")
    elif args.scalebar is not None:
        sys.exit("--scalebar requires --aperpix to be set.")

    font_size = args.label_size or max(12, int(new_h * 0.04))

    print(f"Estimating contrast from {min(args.stat_frames, nframes)} sample frames ...")
    lo, hi = estimate_stats(nframes, get_frame, args.stat_frames, args.contrast)
    print(f"  Clipping range: [{lo:.1f}, {hi:.1f}]")

    print("Converting frames ...")
    pil_frames = []
    for i in range(nframes):
        img = frame_to_pil(get_frame(i), lo, hi, new_w, new_h, args.cmap)
        if args.rotate:
            img = img.rotate(args.rotate, expand=True)  # works for L and RGB
        if tilt_angles is not None and tilt_angles[i] is not None:
            img = add_label(img, f"{tilt_angles[i]:+.1f}°", font_size,
                            args.label_color, args.label_pos)
        if scalebar_nm is not None:
            img = add_scalebar(img, scalebar_nm, aperpix_output, font_size,
                               args.scalebar_color, args.scalebar_pos)
        pil_frames.append(to_gif_frame(img))
        print(f"  frame {i+1}/{nframes}", end="\r")
    print()

    mrc.close()
    if binned_tmp is not None:
        Path(binned_tmp).unlink(missing_ok=True)

    if args.bounce:
        # forward + reversed, excluding duplicated end-points
        pil_frames = pil_frames + pil_frames[-2:0:-1]
        print(f"  Bounce mode: {len(pil_frames)} frames total (forward + reverse)")

    print(f"Writing {args.output}  ({len(pil_frames)} frames @ {args.fps} fps) ...")
    pil_frames[0].save(
        args.output,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=args.loop,
        optimize=False,
    )
    print("Done.")


if __name__ == "__main__":
    main()
