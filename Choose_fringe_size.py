import mrcfile
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import argparse
from Utilities import fourier_interpolate, make_mask



def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(
        description="Stitch square beam montage tomography data."
    )
    parser.add_argument(
        "-i", "--input", help="*.mrc file for raw data", required=True, type=str
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
    return vars(parser.parse_args())


def main():
    args = parse_commandline()

    # Load the image
    image_path = args['input']
    with mrcfile.mrcmemmap.MrcMemmap(args["input"]) as m:
        image = np.asarray(m.data[0])
    image = fourier_interpolate(image, [x // args["binning"] for x in image.shape])

    vmin = np.percentile(image, 10)
    vmax = np.percentile(image, 90)

    # Plot the original image and create space for the eroded image
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    # Display the original image
    ax[0].imshow(image,vmin=vmin,vmax=vmax)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Placeholder for blurred image
    mask = make_mask(image, 20/ args["binning"],medianthreshold=args['maskthreshold'],absolutethreshold=args['maskabsolutethreshold'])  # Initial blur with small sigma
    blurred_display = ax[1].imshow(np.where(mask,image,0),vmin=vmin,vmax=vmax)
    ax[1].set_title("Image with fringes removed")
    ax[1].axis("off")

    # Create slider for adjusting the standard deviation
    ax_sigma = plt.axes([0.2, 0.1, 0.65, 0.03])
    fringe_slider = Slider(ax_sigma, "Fringe size (pixels)", 1, 200, valinit=20, valstep=1,valfmt='%0.0f')

    # Update function for slider
    def update(val):
        fringe_size = fringe_slider.val
        defringed_image = np.where(make_mask(image, fringe_size / args['binning'],medianthreshold=args['maskthreshold'],absolutethreshold=args['maskabsolutethreshold']),image,0)
        blurred_display.set_data(defringed_image)
        ax[1].set_title(f"Fringe size {fringe_size:.0f}")
        fig.canvas.draw_idle()

    # Connect the slider to the update function
    fringe_slider.on_changed(update)

    # Show the interactive plot
    plt.show()

if __name__=='__main__':
    main()
