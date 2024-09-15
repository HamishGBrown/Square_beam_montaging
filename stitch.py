import re
import networkx as nx
from skimage.registration import phase_cross_correlation
from tqdm import tqdm
from PIL import Image
import os
from scipy.ndimage import binary_fill_holes, binary_erosion
import matplotlib.pyplot as plt
import math
import copy
import glob
import mrcfile
import argparse
import numpy as np


def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(
        description="Stitch square beam montage tomography data."
    )
    parser.add_argument(
        "-i", "--input", help="*.mrc wildcard for raw data", required=True, type=str
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory, if left blank output will be placed in a folder named ./[input_filename]_output",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-I",
        "--image_shifts",
        help="Path to text file containing the tilts and a list of image shifts at every tilt (requried).",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-g",
        "--gainref",
        help="Gain reference, if left blank a new gain reference will be calculated and saved in the output directory ",
        required=False,
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
        help="Size of Fresnel fringes at edge of beam, this will be removed from the gain reference (default 20).",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--tiltaxisrotation",
        help="Rotation of tilt axis relative to image, if not provided will take from .mdoc frile.",
        type=float,
        required=False,
    )

    return vars(parser.parse_args())


def calc_row_idx(k, n):
    """
    Calculate the row index in a condensed distance matrix.

    Parameters:
    -----------
    k : int
        Index in the condensed distance matrix.
    n : int
        Number of original data points.

    Returns:
    --------
    int : The row index corresponding to the given condensed index.
    """
    return int(
        math.ceil(
            (1 / 2.0) * (-((-8 * k + 4 * n**2 - 4 * n - 7) ** 0.5) + 2 * n - 1) - 1
        )
    )


def elem_in_i_rows(i, n):
    """
    Calculate the number of elements in the first i rows of a condensed distance matrix.

    Parameters:
    -----------
    i : int
        Row index.
    n : int
        Number of original data points.

    Returns:
    --------
    int : Number of elements in the first i rows.
    """
    return i * (n - 1 - i) + (i * (i + 1)) // 2


def calc_col_idx(k, i, n):
    """
    Calculate the column index in a condensed distance matrix.

    Parameters:
    -----------
    k : int
        Index in the condensed distance matrix.
    i : int
        Row index in the square matrix.
    n : int
        Number of original data points.

    Returns:
    --------
    int : The column index corresponding to the given condensed index and row index.
    """
    return int(n - elem_in_i_rows(i + 1, n) + k)


def condensed_to_square(k, n):
    """
    Convert a condensed distance matrix index to a square matrix index.

    Parameters:
    -----------
    k : int
        Index in the condensed distance matrix.
    n : int
        Number of original data points.

    Returns:
    --------
    tuple : The row and column indices in the square matrix.
    """
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j


def crop(arrayin, shapeout):
    """
    Crop or zero-pad the last `n` dimensions of `arrayin` to match the specified `shapeout`.

    If the dimensions in `shapeout` are smaller than those of `arrayin`, the function crops
    the array to match the specified size. If the dimensions in `shapeout` are larger, the
    function zero-pads the array to achieve the desired output shape.

    Parameters
    ----------
    arrayin : ndarray
        Input array of shape (..., N1, N2, ..., Nn). The array to be cropped or zero-padded
        along its last `n` dimensions.
    shapeout : tuple of int
        Desired shape for the final `n` dimensions of `arrayin`. The function preserves the
        leading dimensions (those not specified in `shapeout`) and modifies only the trailing
        dimensions.

    Returns
    -------
    arrayout : ndarray
        Output array with shape (..., N1', N2', ..., Nn'), where N1', N2', ..., Nn' match
        `shapeout`. Cropping or zero-padding is applied as necessary.

    Notes
    -----
    - For each dimension, if the desired size is smaller than the input size, the array is
      centered and cropped accordingly.
    - If the desired size is larger than the input size, the array is centered and zero-padded.

    Example
    -------
    >>> arrayin = np.array([[1, 2, 3], [4, 5, 6]])
    >>> shapeout = (4, 2)
    >>> crop(arrayin, shapeout)
    array([[0, 1],
           [4, 5],
           [0, 0],
           [0, 0]])
    """
    # Total number of dimensions in the input array
    ndim = arrayin.ndim

    # Number of trailing dimensions to crop or pad
    n = len(shapeout)

    # Number of leading dimensions not affected by cropping/padding
    nUntouched = ndim - n

    # Calculate the shape of the output array
    shapeout_ = arrayin.shape[:nUntouched] + tuple(shapeout)

    # Initialize the output array with zeros
    arrayout = np.zeros(shapeout_, dtype=arrayin.dtype)

    # Get the shapes of the trailing dimensions for input and output arrays
    oldshape = arrayin.shape[-n:]
    newshape = shapeout[-n:]

    def indices(y, y_):
        """
        Determine the slicing indices for cropping or zero-padding a single dimension.

        Parameters
        ----------
        y : int
            Size of the dimension in the input array.
        y_ : int
            Desired size of the dimension in the output array.

        Returns
        -------
        in_slice : slice
            Slicing indices for the input array.
        out_slice : slice
            Slicing indices for the output array.
        """
        if y > y_:
            # Crop: Center the cropping region within the input dimension
            y1, y2 = (y - y_) // 2, (y + y_) // 2
            in_slice = slice(y1, y2)
            out_slice = slice(0, y_)
        else:
            # Zero-pad: Center the input dimension within the padded output region
            y1_, y2_ = (y_ - y) // 2, (y_ + y) // 2
            in_slice = slice(0, y)
            out_slice = slice(y1_, y2_)
        return in_slice, out_slice

    # Compute the slicing indices for each trailing dimension
    ind = [indices(x, x_) for x, x_ in zip(oldshape, newshape)]
    inind, outind = map(tuple, zip(*ind))

    # Assign the cropped or padded data to the output array
    arrayout[nUntouched * (slice(None),) + outind] = arrayin[
        nUntouched * (slice(None),) + inind
    ]

    return arrayout


def rotation_matrix(theta):
    """
    Returns a 2D rotation matrix for a given angle in degrees

    Parameters:
    theta, float :: The rotation angle in degrees

    returns:
    numpy.ndarray: the 2x2 rotation matrix."""

    rad = np.deg2rad(theta)
    ct = np.cos(rad)
    st = np.sin(rad)

    return np.array([[ct, -st], [st, ct]])


def fourier_crop(ain, shapeout):
    """
    Crop or pad a Fourier-transformed array to match the desired output shape.

    Parameters:
    -----------
    ain : numpy.ndarray
        Input array to be cropped or padded in Fourier space.
    shapeout : tuple
        Desired shape of the output array.

    Returns:
    --------
    numpy.ndarray : Cropped or padded array in Fourier space.
    """

    def crop1d(array, s, d):
        # Number of dimensions of array
        N = len(array.shape)
        # Size of array that will be transferred to new grid
        s_ = min(array.shape[d], s)
        # Indices of grid region to transfer to new grid
        ind1 = (
            (np.s_[:],) * ((N + d) % N)
            + (np.s_[: s_ // 2 + s_ % 2],)
            + (np.s_[:],) * (N - (N + d) % N - 1)
        )
        ind2 = (
            (np.s_[:],) * ((N + d) % N)
            + (np.s_[-s_ // 2 + s_ % 2 :],)
            + (np.s_[:],) * (N - (N + d) % N - 1)
        )
        if s > array.shape[d]:
            xtra = list(array.shape)
            xtra[d] = s - array.shape[d]
            return np.concatenate(
                [array[ind1], np.zeros(xtra, dtype=array.dtype), array[ind2]], axis=d
            )
        else:
            return np.concatenate([array[ind1], array[ind2]], axis=d)

    array = copy.deepcopy(ain)
    for i, s in enumerate(shapeout):
        array = crop1d(array, s, i - len(shapeout))
    return array


def fourier_interpolate(
    ain, shapeout, norm="conserve_val", N=None, qspace_in=False, qspace_out=False
):
    """
    Perfom fourier interpolation on array ain so that its shape matches shapeout.

    Arguments
    ---------
    ain : (...,Ni,..,Ny,Nx) array_like
        Input numpy array, interpolation will be applied to the n trailing
        dimensions where n is the length of shapeout.
    shapeout : int (n,) , array_like
        Desired shape of output array
    norm : str, optional  {'conserve_val','conserve_norm','conserve_L1'}
        Normalization of output. If 'conserve_val' then array values are preserved
        if 'conserve_norm' L2 norm is conserved under interpolation and if
        'conserve_L1' L1 norm is conserved under interpolation
    N : int, optional
        Number of (trailing) dimensions to Fourier interpolate. By default take
        the length of shapeout
    qspace_in : bool, optional
        Set to True if the input array is in reciprocal space, False if not (default).
        Be careful with setting this to True for a non-complex array.
    qspace_out : bool, optional
        Set to True for reciprocal space output, False for real-space output (default).
    """
    # Import required FFT functions
    from numpy.fft import fftn

    if N is None:
        N = len(shapeout)

    # Make input complex
    aout = np.zeros(np.shape(ain)[:-N] + tuple(shapeout), dtype=complex)

    # Get input dimensions
    shapein = np.shape(ain)[-N:]

    # axes to Fourier transform
    axes = np.arange(-N, 0)

    if qspace_in:
        a = np.asarray(ain, dtype=complex)
    else:
        a = fftn(np.asarray(ain, dtype=complex), axes=axes)

    aout = fourier_crop(a, shapeout)
    # aout = np.fft.fftshift(crop(np.fft.fftshift(a,axes=axes),shapeout),axes=axes)

    # Fourier transform result with appropriate normalization
    if norm == "conserve_val":
        aout *= np.prod(shapeout) / np.prod(np.shape(ain)[-N:])
    elif norm == "conserve_norm":
        aout *= np.sqrt(np.prod(shapeout) / np.prod(np.shape(ain)[-N:]))

    if not qspace_out:
        aout = np.fft.ifftn(aout, axes=axes)

    # Return correct array data type
    if not np.iscomplexobj(ain):
        return np.real(aout)
    else:
        return aout


def get_imageshifts_for_tilt_angle(file_path, tilt):
    with open(file_path, "r") as file:
        lines = file.readlines()

    result = []
    index = 0
    while index < len(lines):
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

        # Add the parsed data to the result
        if float(tilt) == tilt_angle:
            return np.asarray(array)[:, :2]


def convolve(array1, array2, axes=None):
    """
    Fourier convolution of two arrays over specified axes.

    array2 is broadcast to match array1 so axes refers to the dimensions of
    array1
    """
    # input and output shape
    s = array1.shape
    # Broadcast array2 to match array1
    a2 = np.broadcast_to(array2, s)
    # Axes of transformation
    a = axes
    if a is not None:
        s = [s[i] for i in a]
    if np.iscomplexobj(array1) or np.iscomplexobj(a2):
        return np.fft.ifftn(np.fft.fftn(array1, s, a) * np.fft.fftn(a2, s, a), s, a)
    else:
        return np.fft.irfftn(np.fft.rfftn(array1, s, a) * np.fft.rfftn(a2, s, a), s, a)


def Gaussian(sigma, gridshape):
    r"""
    Calculate a 2D Gaussian function.

    Notes
    -----
    Functional form
    .. math:: 1 / \sqrt { 2 \pi \sigma }  e^{ - ( x^2 + y^2 ) / 2 / \sigma^2 }

    Parameters
    ----------
    sigma : float or (2,) array_like
        The standard deviation of the Gaussian function in pixels
    gridshape : (2,) array_like
        Number of pixels in the grid.
    """
    ysqr = np.fft.fftfreq(gridshape[0], d=1 / gridshape[0]) ** 2
    xsqr = np.fft.fftfreq(gridshape[1], d=1 / gridshape[1]) ** 2

    gaussian = np.exp(-(ysqr[:, None] + xsqr[None, :]) / sigma**2 / 2)
    return gaussian / np.sum(gaussian)


def circular_mask(size, radius=None, center=None):
    """
    Create a binary circular mask in a 2D array.

    Parameters:
    - size (int, int): Size of the array (rows, cols).
    - radius (int, optional): Radius of the circle. Defaults to min(size)/2.
    - center (int, int, optional): Center of the circle (row, col). Defaults to the center of the array.

    Returns:
    - mask (2D numpy array): Binary mask with a circle.
    """
    rows, cols = size

    if center is None:
        center = (rows // 2, cols // 2)
    if radius is None:
        radius = min(rows, cols) // 2

    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)

    return dist_from_center <= radius


def make_mask(im, shrinkn=20, smoothing_kernel=3):
    """
    Generate a binary mask from an image by applying a Gaussian filter, filling holes,
    and then shrinking the mask with morphological erosion.

    Parameters:
    -----------
    im : numpy.ndarray
        Input 2D image array.
    shrinkn : int, optional
        Factor determining the size of the structuring element for shrinking the mask.
        Default is 20.
    smoothing_kernel : float,optional
        Sigma of the gaussian smoothing kernel before applying the mask (3 by default)

    Returns:
    --------
    numpy.ndarray : Binary mask of the same shape as the input image.
    """
    # Step 1: Apply a Gaussian filter to smooth the image
    smoothed_im = convolve(im, Gaussian(smoothing_kernel, im.shape))

    # Step 2: Create an initial binary mask based on a threshold (0.4 * median of the original image)
    mask = smoothed_im > 0.4 * np.median(im)

    # Step 3: Fill any holes in the initial mask
    mask = binary_fill_holes(mask)

    # Step 4: Create a circular structuring element to use for binary erosion
    struct_elem = circular_mask([shrinkn * 2, shrinkn * 2], radius=shrinkn / 2)

    # Step 5: Apply binary erosion to shrink the mask
    mask = binary_erosion(mask, structure=struct_elem)

    return mask


def cross_correlate_alignment(im, template, returncoords=True):
    """
    Align an image to a template using cross-correlation.

    This function calculates the cross-correlation between a template and an image,
    identifies the point of maximum correlation (which indicates the best alignment),
    and either returns the coordinates of this point or returns the image aligned to the template.

    Parameters:
    -----------
    im : numpy.ndarray
        The input image to be aligned. This is the image that will be adjusted.
    template : numpy.ndarray
        The template image that is used as a reference for alignment.
    returncoords : bool, optional
        If True, the function returns the coordinates of the maximum correlation.
        If False, the function returns the aligned image. Default is True.

    Returns:
    --------
    tuple of int or numpy.ndarray :
        If `returncoords` is True, returns a tuple (y, x) representing the coordinates of the
        maximum correlation.
        If `returncoords` is False, returns the aligned image as a numpy array.
    """

    # Step 1: Calculate the cross-correlation between the template and the image
    corr = correlate(template, im)

    # Step 2: Find the coordinates (y, x) of the maximum value in the correlation matrix
    y, x = np.unravel_index(np.argmax(corr), im.shape)

    # Step 3: Depending on the value of returncoords, return the coordinates or the aligned image
    if returncoords:
        return y, x  # Return the coordinates of the maximum correlation
    else:
        # Align the image by rolling it to place the max correlation point at the origin
        return np.roll(im, (-y, -x), axis=(-2, -1))


def make_gain_ref(images, binning=8):
    """
    Generate a gain reference image by averaging a series of input images.

    This function processes a list of images to create a gain reference image.
    The images are first binned to reduce their size, then aligned using cross-correlation,
    and finally averaged together to produce the gain reference. The result is normalized
    by the median value of the template region in the gain reference.

    Parameters:
    -----------
    images : list of str
        A list of file paths to the input images (in MRC format) that will be used to create the gain reference.
    binning : int, optional
        The factor by which the images are binned to reduce their size. Default is 8.

    Returns:
    --------
    numpy.ndarray
        The resulting gain reference image, normalized by the median value of the template region.
    """

    # Loop through each image file
    for i, image in enumerate(tqdm(images, desc="Averaging images to gain ref")):
        # Open the MRC file and read the image data
        with mrcfile.open(image) as mrc:
            im = np.asarray(mrc.data)

        # Perform Fourier interpolation on the image to bin it
        im = fourier_interpolate(im, [x // binning for x in im.shape])

        # Create a mask for the image
        msk = make_mask(im)

        # For the first image, initialize the template and gain reference
        if i == 0:
            template = np.where(msk, 1, 0)  # Create the binary template mask
            gainref = copy.deepcopy(
                im
            )  # Initialize the gain reference with the first image
        else:
            # For subsequent images, align them to the template
            y, x = cross_correlate_alignment(
                np.where(msk, 1, 0), template, returncoords=True
            )
            # Add the aligned image to the gain reference
            gainref += np.roll(im, (y, x), axis=(-2, -1))

    # Normalize the gain reference by the median value of the template region
    median = np.median(gainref[template == 1])
    gainref /= median

    return gainref


def montage(
    images,
    positions,
    pixel_size,
    masks=None,
    binning=8,
    tiltaxisrotation=0,
    gainref=None,
    gainrefmask=None,
    skipcrosscorrelation=False,
):
    """
    Creates a montage image from a series of input images by aligning and stitching them together.

    This function takes a set of images, positions them according to provided coordinates,
    and aligns them based on mask images or generated masks. The images are then combined
    into a single montage image, with optional gain reference correction.

    Parameters:
    -----------
    images : list of str
        List of file paths to the input images (in MRC format) to be included in the montage.
    positions : numpy.ndarray
        Array of shape (N, 2) containing the (x, y) coordinates for positioning each image.
    pixel_size : float
        The pixel size in Angstroms.
    masks : list of str, optional
        List of file paths to mask images (in MRC format). If None, masks will be generated automatically.
    binning : int, optional
        Binning factor to reduce the image size. Default is 8.
    tiltaxisrotation : float, optional
        Rotation angle of the tilt axis, used to correct the positions. Default is 0.
    gainref : numpy.ndarray, optional
        Gain reference image used to normalize the input images. Default is None.

    Returns:
    --------
    numpy.ndarray
        The resulting stitched montage image.
    """

    # Get the shape and data type of the montage tiles from the first image
    with mrcfile.mrcmemmap.MrcMemmap(images[0]) as mrc:
        dtype = mrc.data.dtype
        pixels = np.asarray([x // binning for x in mrc.data.shape[-2:]], dtype=int)

    # Rotate the position coordinates back to camera coordinates based on tilt axis rotation
    positions = (rotation_matrix(-tiltaxisrotation) @ positions[:, :2].T).T
    # positions = -positions[:,::-1]
    positions[:, 0] *= -1  # Invert the x-axis for correct alignment
    # fig, ax = plt.subplots()
    # ax.plot(*positions.T, "ko")
    # plt.show(block=True)
    # positions = positions[:,::-1]

    # If masks are provided, process them; otherwise, initialize an empty mask list
    if masks is not None:
        msks = np.zeros((len(masks), *pixels), dtype=bool)
        for i in range(len(masks)):
            with mrcfile.open(masks[i]) as mrc:
                msk = np.asarray(mrc.data)
            msks[i] = fourier_interpolate(np.where(msk, 1, 0), pixels) > 0.9
    else:
        msks = []

    ims = []  # List to store the processed images
    for i, (image, position) in enumerate(zip(images, positions)):
        with mrcfile.open(image) as mrc:
            im = np.asarray(mrc.data)
        im = fourier_interpolate(im, [x // binning for x in im.shape])

        # Generate a mask if none is provided
        if masks is None:
            msk = make_mask(im)
            msks.append(msk)

        # Apply gain reference correction if provided
        if gainref is not None:
            y, x = cross_correlate_alignment(msks[-1], gainrefmask, returncoords=True)

            im[msk] /= np.roll(gainref, (-y, -x), axis=(-2, -1))[msk]
            im = np.where(msk, im, 0)

        # Normalize the image by replacing non-masked areas with random noise
        # std = np.std(im[msk])
        # mean = np.mean(im[msk])
        # im = np.where(msk, im, np.random.random_sample(size=im.shape) * std + mean)
        ims.append(im)

    # Calculate the overlaps between adjacent tiles
    overlaps = find_overlaps(positions, pixels, pixel_size * binning, msks, ims)

    # Refine the tile positions using cross-correlation between overlapping
    # tiles

    if not skipcrosscorrelation:
        positions = cross_correlate_tiles(
            positions, ims, msks, overlaps, pixel_size * binning, generate_plot=False
        )
    # ax.plot(*positions.T, "bo")
    # plt.show(block=True)

    # Determine the global range of tiles in Angstroms
    width = np.ptp(positions, axis=0) * 1e4
    origin = np.amin(positions, axis=0) * 1e4

    # Calculate the size of the global montage canvas in pixels
    size = (
        np.asarray(width / pixel_size / binning, dtype=int)
        + pixels
        + np.asarray([1, 1])
    )

    # Initialize the montage canvas and overlap map
    canvas = np.zeros(size)
    overlap = np.zeros(size, dtype=np.uint8)

    # Place each image onto the canvas, applying the masks
    for i, (im, position, msk) in enumerate(zip(ims, positions, msks)):
        s = im.shape
        x0, y0 = [int(x) for x in (position * 1e4 - origin) / pixel_size / binning]
        canvas[x0 : x0 + s[0], y0 : y0 + s[1]] += np.where(msk, im, 0)
        overlap[x0 : x0 + s[0], y0 : y0 + s[1]] += np.where(
            msk, np.uint8(1), np.uint8(0)
        )

    # Normalize the canvas by the overlap map to account for overlapping regions
    overlap = np.where(overlap > 1, overlap, 1)
    canvas /= overlap

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


def find_overlaps(
    positions, pixels, pixel_size, masks=None, images=None, minoverlapfrac=0.03
):
    def check_mask_overlaps(positions, masks, i, j):
        dx = positions[i] - positions[j]

    m = len(positions)

    from scipy.spatial.distance import pdist

    # Calculate (unique) distances between montage pieces
    # with scipy's pdist (pair-wise distance) function and
    # calculate which are within a "field of view" of each other
    # result is stored in a condensed distance matrix
    criterion = pixel_size * np.asarray(pixels) * 1e-4
    overlaps = np.logical_and(
        *[pdist(positions[..., i : i + 1]) / criterion[i] < 1.0 for i in range(2)]
    )
    # plot_overlaps(positions,overlaps)

    overlapping_inds = []
    criterion = minoverlapfrac * np.product(pixels)
    for n, val in enumerate(overlaps):
        if val:
            i, j = condensed_to_square(n, m)
            x1 = positions[i]
            x2 = positions[j]
            # Relative shift in pixels
            dx = [int(x) for x in (x2 - x1) / pixel_size * 1e4]
            i1, i2 = array_overlap(dx[0], pixels[0])
            j1, j2 = array_overlap(dx[1], pixels[1])
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
    return overlapping_inds


def correlate(array1, array2, axes=None):
    """
    Fourier correlation of two arrays over specified axes.

    array2 is broadcast to match array1 so axes refers to the dimensions of
    array1
    """
    # input and output shape
    s = array1.shape
    # Broadcast array2 to match array1
    a2 = np.broadcast_to(array2, s)
    # Axes of transformation
    a = axes
    if a is not None:
        s = [s[i] for i in a]
    if np.iscomplexobj(array1) or np.iscomplexobj(a2):
        return np.fft.ifftn(
            np.fft.fftn(array1, s, a) * np.conj(np.fft.fftn(a2, s, a)), s, a
        )
    else:
        return np.fft.irfftn(
            np.fft.rfftn(array1, s, a) * np.conj(np.fft.rfftn(a2, s, a)), s, a
        )


def cross_correlate_tiles(
    positions,
    tiles,
    masks,
    overlaps,
    pixel_size,
    max_correction=0.1,
    max_shift=50,
    generate_plot=False,
):
    pixels = tiles[0].shape

    # We will solve global alignment of tiles by least squares Ax = b
    # matrix problem (https://en.wikipedia.org/wiki/Linear_least_squares).
    # N is the number of tiles times two (for x and y coordinates)
    N = 2 * len(positions)
    # Initialize lists which we will append rows for the A and b matrices
    A = []
    b = []

    if generate_plot:
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
        ax = ax.ravel()
        ax[0].plot(*positions.T, "ko", label="Initial tile positions")
        for i, pos in enumerate(positions):
            ax[0].annotate(str(i), pos)

    G = nx.Graph()
    # Add nodes to graph
    G.add_nodes_from(list(range(len(positions))))
    for ind, (i, j) in enumerate(overlaps):
        # Align tiles by masked phase cross correlation, see:
        # https://scikit-image.org/docs/stable/auto_examples/registration/plot_masked_register_translation.html
        # and Padfield, Dirk. "Masked object registration in the Fourier domain." IEEE Transactions on image processing 21.5 (2011): 2706-2718.

        # Retrieve image shifts for the overlapping tiles
        x1 = positions[i]
        x2 = positions[j]

        # Calculate relative shift in pixels
        dx = [int(x) for x in (x2 - x1) / pixel_size * 1e4]

        # Get the indices for array overlap
        i1, i2 = array_overlap(dx[0], pixels[0])
        j1, j2 = array_overlap(dx[1], pixels[1])

        # Make masks for (estimated) overlapping areas of each array
        reference_mask = np.zeros_like(masks[i], dtype=bool)
        reference_mask[i1[0] : i1[1], j1[0] : j1[1]] = masks[i][
            i1[0] : i1[1], j1[0] : j1[1]
        ]
        moving_mask = np.zeros_like(masks[j], dtype=bool)
        moving_mask[i2[0] : i2[1], j2[0] : j2[1]] = masks[j][
            i2[0] : i2[1], j2[0] : j2[1]
        ]

        # Calculate shift by masked cross correlation
        detected_shift = phase_cross_correlation(
            tiles[i], tiles[j], reference_mask=reference_mask, moving_mask=moving_mask
        )

        shifttoolarge = np.linalg.norm(np.asarray(dx) - detected_shift) > max_shift

        if generate_plot:
            delta = np.asarray(detected_shift) * pixel_size * 1e-4
            if ind == 0:
                label = "Shift vector from cross-correlation"
            else:
                label = None

            if shifttoolarge and generate_plot:
                ax[0].plot(
                    [x1[0], x1[0] + delta[0]],
                    [x1[1], x1[1] + delta[1]],
                    "r-",
                    label=label,
                )
            elif generate_plot:
                ax[0].plot(
                    [x1[0], x1[0] + delta[0]],
                    [x1[1], x1[1] + delta[1]],
                    "b-",
                    label=label,
                )

        # Exclude large shifts from the determination of absolute positions
        if not shifttoolarge:
            # convert detected shift from pixels to microns
            detected_shift *= pixel_size * 1e-4
            # detected_shift is position j (x2) - position i (x1)
            Arow = np.zeros((2, N))
            Arow[0, 2 * j] = 1
            Arow[0, 2 * i] = -1
            Arow[1, 2 * j + 1] = 1
            Arow[1, 2 * i + 1] = -1
            A.append(Arow)
            b[len(b) :] = detected_shift.tolist()

            # Add valid connection to graph
            G.add_edge(i, j)

    A = np.concatenate(A, axis=0)

    # For some tiles there will be no reliable shift determinations
    # from cross-correlation linking them to the other tiles,
    # in this case default to using the original
    # shifts implied by the microscope image shifts if groups of tiles
    #
    if not nx.is_connected(G):
        extraA = []
        islands = sorted(nx.connected_components(G), key=len, reverse=True)
        largest_island = islands.pop()
        anchor = largest_island.pop()
        # for i in np.where(np.all(A == 0, axis=0))[0]:
        for x in islands:
            i = x.pop()

            # Loop over x and y components
            for j in range(2):
                # Make new row to add to A matrix
                Arow = np.zeros((N))
                Arow[2 * anchor + j] = 1
                Arow[2 * i + j] = -1
                extraA.append(Arow)
                b.append((positions[anchor][j] - positions[i // 2][j]))

            if generate_plot:
                x1 = positions[anchor]
                x2 = positions[i]
                ax[0].plot([x1[0], x2[0]], [x1[1], x2[1]], "k--")

        A = np.stack(A.tolist() + extraA, axis=0)
    # U,S,Vh = np.linalg.svd(A)
    # ax[1].imshow(U,cmap='bwr')
    # # ax[1].set_title('Difference matrix')
    # ax[2].imshow(Vh)
    # ax[3].plot(S)
    x, residuals, rank, s = np.linalg.lstsq(A, np.asarray(b), rcond=1e-1)
    newx = x[::2]
    newy = x[1::2]

    # The absolute position is unconstrained in least squares minimization
    # Pin the position closest to the origin to its original absolute position
    i = np.argmin(np.linalg.norm(positions, axis=1))
    newx -= newx[i] - positions[i][0]
    newy -= newy[i] - positions[i][1]

    positions[:, 0] = newx
    positions[:, 1] = newy

    if generate_plot:
        ax[0].plot(*positions.T, "bo", label="Initial tile positions")
        ax[0].legend()
        plt.show(block=True)

    return positions


def parse_image_shifts(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    result = []
    index = 0
    while index < len(lines):
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

        # Add the parsed data to the result
        result.append((tilt_angle, np.asarray(array)))

    # Sort the list by tilt angle and return
    return sorted(result, key=lambda x: x[0])


def savetomrc(array, fnam, overwrite=True):
    with mrcfile.new(fnam, overwrite=overwrite) as mrc:
        mrc.set_data(array)


def generate_image_file_names_from_template(name, datadir, tilt, positions):
    imagefiles = [
        os.path.join(datadir, "{0}_{1}_{2}.mrc".format(name, tilt, x))
        for x in range(positions.shape[0])
    ]

    return imagefiles


def parse_mdoc(file_path):
    parsed_data = {}
    current_section = None

    # Regular expressions to match key-value pairs and sections
    key_value_regex = re.compile(r"(\S+)\s*=\s*(.+)")
    section_regex = re.compile(r"\[(.+)\]")

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check if the line matches a section header in square brackets
            section_match = section_regex.match(line)
            if section_match:
                current_section = section_match.group(1)
                parsed_data[current_section] = {}
                continue

            # Match key-value pairs
            key_value_match = key_value_regex.match(line)
            if key_value_match:
                key, value = key_value_match.groups()
                if current_section:
                    parsed_data[current_section][key] = value
                else:
                    parsed_data[key] = value
    return parsed_data


def extract_tilt_axis_angle(file_path):
    tilt_axis_angle = None
    section_regex = re.compile(r"\[T\s*=\s*.*Tilt\s+axis\s+angle\s*=\s*([-\d.]+)")

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            # Check if the line contains the "Tilt axis angle"
            section_match = section_regex.search(line)
            if section_match:
                tilt_axis_angle = float(section_match.group(1))
                break

    return tilt_axis_angle


def setup_outputdir(output):
    if output is None:
        outputdir = os.path.split(args["input"])[1].replace("*.mrc", "_output")
    else:
        outputdir = args["output"]
    print("Results will be outputted to {0}".format(outputdir))
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    return outputdir


def setup_gainreference(gainreffile, filenames, outputdir):
    if os.path.exists(gainreffile):
        with mrcfile.open(gainreffile, "r+") as m:
            gainref = np.asarray(m.data)
    else:
        gainref = make_gain_ref(filenames, binning=4)
        gainreffile_ = os.path.join(outputdir, "gain.mrc")
        print("Saving generated Gain reference as {0}".format(gainreffile_))
        savetomrc(gainref.astype(np.float32), gainreffile_)

    # Create a mask of the gain reference, filling holes and using binary
    # erosion to remove the fringes
    gainrefmask = binary_fill_holes(gainref > 0.4 * np.median(gainref))
    gainrefmask = binary_erosion(
        gainrefmask, structure=circular_mask([20 * 2] * 2, radius=20 / 2)
    )

    return gainref, gainrefmask


if __name__ == "__main__":
    args = parse_commandline()
    outdir = setup_outputdir(args["output"])

    # Binning constant
    binning = int(args["binning"])

    files = glob.glob(args["input"])
    mdoc_files = [x.replace(".mrc", ".mrc.mdoc") for x in files]

    # Get pixel size, tilt axis rotation from first mdoc file
    mdoc = parse_mdoc(mdoc_files[0])
    pixelsize = float(mdoc["PixelSpacing"])
    if args["tiltaxisrotation"] is None:
        tiltaxisrotation = extract_tilt_axis_angle(mdoc_files[0])
    else:
        tiltaxisrotation = float(args["tiltaxisrotation"])

    # tiltaxisrotation = float(parse_mdoc(mdoc_files[0])["ZValue = 0"]["RotationAngle"])

    # get tilts and image shifts from all mdoc files
    tilts, imageshifts = [[], []]
    indices = []
    # sys.exit()
    for mdoc in mdoc_files:
        m = parse_mdoc(mdoc)
        tilts.append(float(m["ZValue = 0"]["TiltAngle"]))
        imageshifts.append([float(x) for x in m["ZValue = 0"]["ImageShift"].split(" ")])

        # Split the string at '.mrc' and then find the integer before 'mrc'
        split_filename = mdoc.split(".mrc")[0]
        index = split_filename.split("_")[-1]
        indices.append(index)

    imageshifts = np.asarray(imageshifts)

    # Get unique tilts and map of which images correspond to which tilts
    uniquetilts, ntilts = np.unique(tilts, return_counts=True)

    # Get image shifts from file
    # TODO make SerialEM put this in the mdoc file in microns (not
    # weird TFS units) to obviate this step
    imageshifts = parse_image_shifts(args["image_shifts"])

    # Load gain ref if available make a new one if not
    if args["gainref"] is not None:
        print("Loading gain reference {0}".format(args["gainref"]))
        with mrcfile.open(args["gainref"], "r+") as m:
            gainref = np.asarray(m.data)
    else:
        gainref = make_gain_ref(files, binning=binning)
        gainreffile = os.path.join(outdir, "gainref.mrc")
        print("Saving gain reference to {0}".format(gainreffile))
        savetomrc(gainref.astype(np.float32), gainreffile)

    fringe_size = args["fringe_size"]

    # make gain reference mask
    gainrefmask = binary_fill_holes(gainref > 0.4 * np.median(gainref))

    # Remove Fresnel fringes from gain reference mask
    gainrefmask = binary_erosion(
        gainrefmask,
        structure=circular_mask([fringe_size * 2] * 2, radius=fringe_size / 2),
    )
    for i, tilt in enumerate(tqdm(uniquetilts, desc="Stitching montages")):
        # For a given tilt get the images and image shifts for that tilt
        mask = np.isclose(tilts, tilt)
        # Sort image files by integer just before .mrc
        imagefiles = sorted(
            np.asarray(files)[mask],
            key=lambda x: int(re.search(r"([-\d]+)\.mrc", x).group(1)),
        )
        positions = imageshifts[i][1]
        mont = montage(
            imagefiles,
            positions,
            pixelsize,
            tiltaxisrotation=tiltaxisrotation,
            binning=binning,
            gainref=np.where(gainrefmask, gainref, 1),
            gainrefmask=gainrefmask,
            skipcrosscorrelation=args["skipcrosscorrelation"],
        )
        fileout = os.path.join(outdir, "Montage_{0}.tiff".format(int(tilt)))
        Image.fromarray(mont.astype(np.int16)).save(fileout)

    import sys

    sys.exit()

    tilt = 0.0
    # Imageshifts=
    tiltsandimageshifts = parse_image_shifts("Krios_imageshifts.txt")
    tilts = [x[0] for x in tiltsandimageshifts]
    # print(tilts)
    # import sys;sys.exit()
    imageshifts = [x[1] for x in tiltsandimageshifts]
    pixelsize = 1.664 * 2
    tiltaxisrotation = -3
    outdir = "/home/hbrown/Projects/Research_projects/20240715_square_beam_tomography/20240808_Montage_stiching/20240826_Kriosmontages/rough_montages"
    name = "Montagingtest1"
    datadir = "/home/hbrown/Mount/KriosFalcon4/Brown/20240826_montaging"

    allnames = sum(
        [
            generate_image_file_names_from_template(name, datadir, tilt, positions)
            for tilt, positions in zip(tilts, imageshifts)
        ],
        [],
    )

    gainreffile = os.path.join(outdir, "gain.mrc")
    if os.path.exists(gainreffile):
        with mrcfile.open(gainreffile, "r+") as m:
            gainref = np.asarray(m.data)
    else:
        gainref = make_gain_ref(allnames, binning=4)
        savetomrc(gainref.astype(np.float32), gainreffile)
    # fig,ax = plt.subplots(ncols=2)

    gainref = np.where(gainrefmask, gainref, np.min(gainref[gainrefmask]))

    # plt.show(block=True)
    # for i,tilt in enumerate(tilts):
    #     print(i,tilt)

    # import sys;sys.exit()
    # masks = ['/home/hbrown/Projects/Research_projects/20240715_square_beam_tomography/20240808_Montage_stiching/Montagetest6/gain_cor/Montagingtest6_0_{0}.mrc'.format(x) for x in range(14)]
    # imagefiles = sorted(list(glob.glob(args['image_paths'])))
    # positions = get_imageshifts_for_tilt_angle(Imageshifts,tilt)
    from tqdm.contrib import tzip

    n = 9
    print(tilts[n])
    for tilt, positions in tzip(tilts[:], imageshifts[:], desc="Stitching montages"):
        imagefiles = generate_image_file_names_from_template(
            name, datadir, tilt, positions
        )
        mont = montage(
            imagefiles,
            positions,
            float(pixelsize),
            tiltaxisrotation=tiltaxisrotation,
            binning=4,
            gainref=gainref,
        )
        fileout = os.path.join(outdir, "Montage_{0}.tiff".format(tilt))
        Image.fromarray(mont.astype(np.int16)).save(fileout)
