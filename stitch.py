import re
import networkx as nx
from skimage.registration import phase_cross_correlation
from tqdm import tqdm
from PIL import Image
import os
from scipy.ndimage import binary_fill_holes, binary_erosion
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import math
import copy
import glob
import mrcfile
import argparse
import numpy as np
from skimage.registration._masked_phase_cross_correlation import cross_correlate_masked
import h5py


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
        help="Skip cross-correlation alignment of montage tiles and default to using imageshifts to stitch montage.",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--tiltaxisrotation",
        help="Rotation of tilt axis relative to image, if not provided will take from .mdoc file.",
        type=float,
        required=False,
    )
    parser.add_argument(
        "-G",
        "--gain_corrected_files",
        help = "The program will write the files with the gain correction applied to an mrc. Use this option to re-use already gain corrected files and save time.",
        type=str,
        required=False
    )
    parser.add_argument(
        "-M",
        "--maximageshift",
        help = 'Limit the size of the total montage by excluding images in montage with image shift in microns greater than this value.',
        type=float,
        required=False
    )
    parser.add_argument(
        "-m",
        "--max_allowed_imshift_correction",
        help = 'Maximum allowed correction to tile alignments by cross-correlation.',
        type=float,
        default=0.05,
        required=False
    )

    parser.add_argument(
        "-S",
        "--correctimageshiftfilefactor",
        help = 'Sometimes imageshift file needs to be divided by a factor of 2 since serial-EM uses super-resolution pixels.',
        type=int,
        default=1,
        required=False
    )

    return vars(parser.parse_args())


def save_array_to_hdf5(arrays, filename, dataset_names,filemode ='w'):
    """
    Saves a numpy array to an HDF5 file.

    Parameters:
    array (np.ndarray): The numpy array to save.
    filename (str): The name of the file where the array will be saved.
    dataset_name (str): The name of the dataset in the HDF5 file.
    """
    with h5py.File(filename, filemode) as hdf:
        for name,data in zip(dataset_names,arrays):
            hdf.create_dataset(name, data=data)


def load_array_from_hdf5(filename, dataset_name):
    """
    Loads a numpy array from an HDF5 file.

    Parameters:
    filename (str): The name of the file from which to load the array.
    dataset_name (str): The name of the dataset in the HDF5 file.

    Returns:
    np.ndarray: The numpy array loaded from the file.
    """
    with h5py.File(filename, "r") as hdf:
        array = np.array(hdf[dataset_name])
    return array


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

    y, x = [(i + N // 2) % N - N // 2 for i, N in zip((y, x), im.shape)]

    # Step 3: Depending on the value of returncoords, return the coordinates or the aligned image
    if returncoords:
        return y, x  # Return the coordinates of the maximum correlation
    else:
        # Align the image by rolling it to place the max correlation point at the origin
        return np.roll(im, (-y, -x), axis=(-2, -1))


def roll_no_periodic(arr, shift, fill_value=0, axis=None):
    """
    Rolls an array along the given axis or axes but without periodic boundary conditions.
    Vacated positions will be filled with fill_value.

    Parameters:
    - arr: numpy array to be shifted.
    - shift: int or tuple of ints, amount to shift. Positive values shift right/down, negative values shift left/up.
    - fill_value: value to place in the vacated positions.
    - axis: int or tuple of ints, the axis or axes to roll along. If None, roll along all axes.

    Returns:
    - A new numpy array with the same shape as arr, but shifted.
    """
    # If axis is None, shift across all axes
    if axis is None:
        axis = tuple(range(arr.ndim))
        shift = (shift,) * arr.ndim
    elif isinstance(axis, int):
        axis = (axis,)
        shift = (shift,) if isinstance(shift, int) else shift

    result = np.full_like(arr, fill_value)  # Create an array filled with the fill_value

    if len(axis) != len(shift):
        raise ValueError("The number of shifts must match the number of axes.")
    src_slice = [slice(None)] * arr.ndim
    dst_slice = [slice(None)] * arr.ndim

    for ax, s in zip(axis, shift):
        if s == 0:
            continue  # No shift for this axis

        # Determine the slices that will remain after the shift
        if s > 0:
            src_slice[ax] = slice(0, -s)
            dst_slice[ax] = slice(s, None)
        elif s < 0:
            src_slice[ax] = slice(-s, None)
            dst_slice[ax] = slice(0, s)

    result[tuple(dst_slice)] = arr[tuple(src_slice)]

    return result


def make_gain_ref(images, shrinkn=20, binning=8, templateindex=0, maxnumber=None):
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
    templateindex: int,optional
        Which image to align all other images too.
    maxnumber: int or None
        Maximum number of frames that will be averaged to produce the gain reference.
        If None, all available frames will be used.
    Returns:
    --------
    numpy.ndarray
        The resulting gain reference image, normalized by the median value of the template region.
    """

    def get_image_and_mask(img):
        # Perform Fourier interpolation on the image to bin it
        im = np.asarray(img)
        im = fourier_interpolate(im, [x // binning for x in img.shape])

        # Create a mask for the image
        msk = make_mask(im, shrinkn=shrinkn / binning)
        return im, msk

    # Get target image
    memmap = mrcfile.mrcmemmap.MrcMemmap(images[0])
    img = memmap.data[templateindex]
    im, msk = get_image_and_mask(img)
    template = np.where(msk, 1, 0)  # Create the binary template mask
    # Initialize the gain reference with the first image
    gainref = copy.deepcopy(im)

    # Weight will track the denominator for the averaging step later on
    weight = np.ones_like(gainref)

    count = 0

    # Loop through each image file
    for i, image in enumerate(tqdm(images, desc="Averaging images to gain ref")):
        # Open the MRC file and read the image data
        memmap = mrcfile.mrcmemmap.MrcMemmap(image)

        for j, img in enumerate(tqdm(memmap.data, leave=False, desc="Frames in mrc")):
            im, msk = get_image_and_mask(img)

            # skip template image
            if i == 0 and j == templateindex:
                continue
            else:
                # Align images to the template
                y, x = cross_correlate_alignment(
                    template, np.where(msk, 1, 0), returncoords=True
                )
                gainref += roll_no_periodic(im, (-y, -x), axis=(-2, -1))
                weight += roll_no_periodic(np.ones_like(im), (-y, -x), axis=(-2, -1))
                count += 1
            if maxnumber is not None:
                if count >= maxnumber:
                    break
    gainref /= weight
    # Normalize the gain reference by the median value of the template region
    median = np.median(gainref[template == 1])
    gainref /= median

    return gainref

def stitch(ims,positions,msks,pixel_size,binning=1,montagewidth=None,montageorigin=None):
    # Determine the global range of tiles in Angstroms
    if montagewidth is None:
        width = np.ptp(positions, axis=0) * 1e4
    else:
        width = montagewidth
    if montageorigin is None:
        origin = np.amin(positions, axis=0) * 1e4
    else:
        origin = montageorigin

    pixels = ims[0].shape

    # Calculate the size of the global montage canvas in pixels
    size = (
        np.asarray(width / pixel_size / binning, dtype=int)[::-1]
        + pixels
        + np.asarray([1, 1])
    )

    # Initialize the montage canvas and overlap map
    canvas = np.zeros(size)
    overlap = np.zeros(size, dtype=np.uint8)

    # Place each image onto the canvas, applying the masks
    for i, (im, position, msk) in enumerate(
        tqdm(zip(ims, positions, msks), desc="Stitching")
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

    # Normalize the canvas by the overlap map to account for overlapping regions
    median = np.median(canvas[overlap == 1])
    canvas[overlap < 1] = median
    overlap = np.where(overlap > 1, overlap, 1)
    canvas /= overlap
    return canvas


def montage(
    image,
    outdir,
    positions,
    pixel_size,
    binning=8,
    gainref=None,
    gainrefmask=None,
    skipcrosscorrelation=False,
    montagewidth=None,
    montageorigin=None,
    tiles=None,
    gaincorrectedfile=None,
    maxshift = 0.05,
    fringe_size=20,
):
    """
    Creates a montage image from a series of input images by aligning and stitching them together.

    This function takes a set of images, positions them according to provided coordinates,
    and aligns them based on mask images or generated masks. The images are then combined
    into a single montage image, with optional gain reference correction.

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
    tiltaxisrotation : float, optional
        Rotation angle of the tilt axis, used to correct the positions. Default is 0.
    gainref : numpy.ndarray, optional
        Gain reference image used to normalize the input images. Default is None.
    montagewidth : None or (2,) array_like
        size in Angstrom of the full montage in both dimensions, useful for consistency
        with other tilts in the tilt series
    montageorigin : None or (2,) array_like
        Origin point (most negative image shift) in Angstrom of the full montage
        in both dimensions, useful for consistency with other tilts in the tilt
        series
    tiles : None or sliceobject, optional
        Slice object indicating tiles that will be stitched (mainly for testing purposes)
    gaincorrectedimagefile : None or string,optional
        Path to already gain corrected images, if not provided the program will generate 
        the gain corrected images and write them to a mrc file in the output directory.
    fringe_size : float, optional
        Size of the fringe region in pixels for removal from gain reference. Default is 20.
    Returns:
    --------
    numpy.ndarray
        The resulting stitched montage image.
    """

    # Load images as mrc memory map
    mrcmemmap = mrcfile.mrcmemmap.MrcMemmap(image)

    # Get the shape and data type of the montage tiles
    dtype = mrcmemmap.data.dtype
    pixels = np.asarray([x // binning for x in mrcmemmap.data.shape[-2:]], dtype=int)

    if tiles is None:
        M = slice(0, len(mrcmemmap.data))
    else:
        M = tiles
    # Convert positions from pixels to microns, remove 3rd dimension
    positions = positions[:,:2]*pixel_size * 1e-4

    # Initialize an empty mask list
    msks = []

    ims = []  # List to store the processed images
    if gaincorrectedfile is None:
        gaincorrectedfile = os.path.join(
                outdir, os.path.split(image)[1].replace(".mrc", "_gain_corrected.mrc")
            )
        for position, img in tqdm(
            zip(positions[M], mrcmemmap.data[M]),
            total=len(positions[M]),
            desc="Applying gain reference",
        ):
            
            im = np.asarray(img)
            im = fourier_interpolate(im, [x // binning for x in im.shape])

            # Generate a mask if none is provided
            msk = make_mask(im, shrinkn=fringe_size / binning)
            msks.append(msk)

            # Apply gain reference correction if provided
            if gainref is not None:
                y, x = cross_correlate_alignment(msks[-1], gainrefmask, returncoords=True)

                im[msk] /= np.roll(gainref, (-y, -x), axis=(-2, -1))[msk]
                im = np.where(msk, im, 0)
            ims.append(im)
        print('Saving gain corrected images to {0}'.format(gaincorrectedfile))
        savetomrc(np.asarray(ims,dtype='float32'),gaincorrectedfile)
    else:
        print('Loading gain corrected images from {0}'.format(gaincorrectedfile))
        ims = mrcfile.mrcmemmap.MrcMemmap(gaincorrectedfile).data#[M]

        #Check that binning of already gain corrected images is the same as requested
        if np.any(ims.shape[-2:] != pixels):
            raise ValueError("Binning of gain corrected images given in command line does not match requested binning")

        for im in ims:
            msks.append(im>0)



    positionfile = os.path.join(
        outdir, os.path.split(image)[1].replace(".mrc", "_refined_positions.h5")
    )
    # Calculate the overlaps between adjacent tiles
    overlaps = find_overlaps(
        positions[M], pixels, pixel_size * binning, msks, plot=False
    )
    if not skipcrosscorrelation:
        

        # Refine the tile positions using cross-correlation between overlapping
        # tiles
        
        original_positions=copy.deepcopy(positions)
        positions[M],xcorr,deltas = cross_correlate_tiles(
            positions[M],
            ims,
            msks,
            overlaps,
            pixel_size * binning,
            max_correction=maxshift,
            # generate_plot=True,
            # cross_corr_param_file=positionfile
        )

        save_array_to_hdf5([original_positions,positions,xcorr,overlaps,np.asarray(deltas)], positionfile, ["Original_positions","Refined_positions",'cross_correlations','overlaps','relative_shifts'])
    elif os.path.exists(positionfile):
        positions = load_array_from_hdf5(positionfile, "Refined_positions")
        xcorr = load_array_from_hdf5(positionfile, "cross_correlations")
        original_positions = positions
    else:
        original_positions = positions
        # xcorr = np.ones(len(overlaps))

    # plot_positions(positions[M])
    
    canvas = stitch(ims,positions[M],msks,pixel_size,binning=binning,montagewidth=montagewidth,montageorigin=montageorigin)
    fileout = os.path.join(
        outdir, os.path.splitext(os.path.split(image)[1])[0] + ".tif"
    )
    if not skipcrosscorrelation:
        ntiles = len(original_positions[M])
        figsize = 2*(int(np.ceil(np.sqrt(ntiles))),)
        xcorfig, xcorax = plt.subplots(figsize=figsize)
        cmean = np.mean(canvas)
        cstd = np.std(canvas)
        xcorax.imshow(canvas,cmap=plt.get_cmap('gist_gray'),origin='lower',vmin=cmean-cstd,vmax=cmean+cstd)
        s = np.asarray(ims[0].shape)[::-1]
        posi = (original_positions[M] *1e4-montageorigin)/pixel_size/binning
        posi += s/2
        xcorax.plot(*posi.T, "ko", label="Initial tile positions")
        for i, pos in enumerate(posi):
            xcorax.annotate(str(i), pos)
        posi = (positions[M] *1e4-montageorigin)/pixel_size/binning
        posi += s/2
        xcorax.plot(*posi.T, "bo", label="Refined tile positions")
        # xcorrmax = np.amax(xcorr)
        cmap = plt.get_cmap('viridis')
        for ind,(i,j) in enumerate(overlaps):
                # Retrieve image shifts for the overlapping tiles
            x1 = (original_positions[M][i] *1e4-montageorigin)/pixel_size/binning
            
            x2 = (original_positions[M][j] *1e4-montageorigin)/pixel_size/binning
            dx = [int(x) for x in (x2 - x1) ][::-1]
            x1 += s/2
            delta = (deltas[ind]*1e4)/pixel_size/binning
            
            shifttoolarge = np.linalg.norm(np.asarray(dx) - delta) > maxshift/pixel_size/binning*1e4
            if shifttoolarge:
                linestyle='--'
            else:
                linestyle='-'
            xcorax.plot(
                    [x1[0], x1[0] + delta[1]],
                    [x1[1], x1[1] + delta[0]],
                    linestyle=linestyle,
                    color='b'
                    # color=cmap(xcorr[ind]/xcorrmax ),
                    # label=label,
                )
        plotfile = os.path.join(
                outdir, os.path.split(image)[1].replace(".mrc", "_Plot.pdf")
            )
        xcorfig.savefig(plotfile)
    # Note that Python Image library does not support write 16-bit integer and writes
    # 32-bit real instead ¯\_(ツ)_/¯ 
    Image.fromarray(canvas.astype(np.int16)).save(fileout)
    # return canvas


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


def _masked_phase_cross_correlation(
    reference_image, moving_image, reference_mask, moving_mask=None, overlap_ratio=0.3
):
    """Masked image translation registration by masked normalized
    cross-correlation.

    Parameters
    ----------
    reference_image : ndarray
        Reference image.
    moving_image : ndarray
        Image to register. Must be same dimensionality as ``reference_image``,
        but not necessarily the same size.
    reference_mask : ndarray
        Boolean mask for ``reference_image``. The mask should evaluate
        to ``True`` (or 1) on valid pixels. ``reference_mask`` should
        have the same shape as ``reference_image``.
    moving_mask : ndarray or None, optional
        Boolean mask for ``moving_image``. The mask should evaluate to ``True``
        (or 1) on valid pixels. ``moving_mask`` should have the same shape
        as ``moving_image``. If ``None``, ``reference_mask`` will be used.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images. The correlation for
        translations corresponding with an overlap ratio lower than this
        threshold will be ignored. A lower `overlap_ratio` leads to smaller
        maximum translation, while a higher `overlap_ratio` leads to greater
        robustness against spurious matches due to small overlap between
        masked images.

    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``moving_image``
        with ``reference_image``. Axis ordering is consistent with
        numpy (e.g. Z, Y, X)

    References
    ----------
    .. [1] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
    .. [2] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           :DOI:`10.1109/CVPR.2010.5540032`

    """
    if moving_mask is None:
        if reference_image.shape != moving_image.shape:
            raise ValueError(
                "Input images have different shapes, moving_mask must "
                "be explicitely set."
            )
        moving_mask = reference_mask.astype(bool)

    # We need masks to be of the same size as their respective images
    for im, mask in [(reference_image, reference_mask), (moving_image, moving_mask)]:
        if im.shape != mask.shape:
            raise ValueError("Image sizes must match their respective mask sizes.")

    xcorr = cross_correlate_masked(
        moving_image,
        reference_image,
        moving_mask,
        reference_mask,
        axes=tuple(range(moving_image.ndim)),
        mode="full",
        overlap_ratio=overlap_ratio,
    )

    # Generalize to the average of multiple equal maxima
    maxima = np.stack(np.nonzero(xcorr == xcorr.max()), axis=1)
    center = np.mean(maxima, axis=0)
    shifts = center - np.array(reference_image.shape) + 1

    # The mismatch in size will impact the center location of the
    # cross-correlation
    size_mismatch = np.array(moving_image.shape) - np.array(reference_image.shape)

    return -shifts + (size_mismatch / 2), xcorr.max()


def cross_correlate_tiles(
    positions,
    tiles,
    masks,
    overlaps,
    pixel_size,
    max_correction=0.3,
    generate_plot=False,
    cross_corr_param_file=None,
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
    cmap = plt.get_cmap('viridis')

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
    
    weights = []
    xcorr = []
    deltas = []
    shifts = []

    # TODO make this threaded
    if cross_corr_param_file is not None:
        xcorr,deltas = [load_array_from_hdf5(cross_corr_param_file,x) for x in ("cross_correlations","relative_shifts")]
    for ind, (i, j) in enumerate(tqdm(overlaps, desc="Cross-correlation alignment")):
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

            # Make masks for (estimated) overlapping areas of each array
            reference_mask = np.zeros_like(masks[i], dtype=bool)
            reference_mask[i1[0] : i1[1], j1[0] : j1[1]] = masks[i][
                i1[0] : i1[1], j1[0] : j1[1]
            ]
            moving_mask = np.zeros_like(masks[j], dtype=bool)
            moving_mask[i2[0] : i2[1], j2[0] : j2[1]] = masks[j][
                i2[0] : i2[1], j2[0] : j2[1]
            ]
            
            # Calculate shift by masked cross correlation with ordering (Y,X)
            detected_shift, xcorrmax = _masked_phase_cross_correlation(
                tiles[i], tiles[j], reference_mask=reference_mask, moving_mask=moving_mask
            )
            # delta is measured shift in microns
            delta = np.asarray(detected_shift) * pixel_size * 1e-4
            xcorr.append(xcorrmax)
            deltas.append(delta)
        else:
            delta = deltas[ind]
            xcorrmax = xcorr[ind]
            detected_shift = delta /pixel_size*1e-4
            
        
        # shifttoolarge=True
        if generate_plot:
            
            if ind == 0:
                label = "Shift vector from cross-correlation"
            else:
                label = None

        shifttoolarge = np.linalg.norm(np.asarray(dx)*pixel_size * 1e-4 - delta) > max_correction
        # detected_shift is position j (x2) - position i (x1)
        if not shifttoolarge:
            Arow = np.zeros((2, N))
            Arow[0, 2 * j] = 1
            Arow[0, 2 * i] = -1
            Arow[1, 2 * j + 1] = 1
            Arow[1, 2 * i + 1] = -1
            A.append(xcorrmax*Arow)
            b[len(b) :] = (xcorrmax* delta).tolist()
            # Add valid connection to graph
            G.add_edge(i, j)
        # Weight by the maximum value of cross-correlation
        weights.append(xcorrmax)

    

    # Join rows of A into numpy matrix
    if len(A) > 0:
        A = np.concatenate(A, axis=0)
    else:
        A = np.asarray(A)
    # Normalize by mean weight
    meanweight = np.mean(weights)
    A /= meanweight
    b = (np.asarray(b)/meanweight).tolist()

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
    #TODO use RANSAC algorithm to find the best fit
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

            # fig,ax = plt.subplots()
            # ax.plot(*np.asarray(island_points).T,'ro',label='island')
            # for idx, (x, y) in enumerate(island_points):
            #     ax.annotate(str(idx), (x, y), textcoords="Island points", xytext=(5, 5), ha='center', fontsize=8)
            # ax.plot(*np.asarray(largest_island_points).T,'bo',label='Anchor')
            # for idx, (x, y) in enumerate(largest_island_points):
            #     ax.annotate(str(idx), (x, y), textcoords="Anchor island points", xytext=(5, 5), ha='center', fontsize=8)
            
            i,j = find_closest_points_with_kdtree(island_points, largest_island_points)
            
            i = list(island)[i]
            j = list(largest_island)[j]
            # i = x.pop()
            x1 = positions[j]
            x2 = positions[i]
            delta = x1 - x2
            # ax.plot([x1[0],x2[0]],[x1[1],x2[1]],'k-',label='connector')
            # fig.legend()
            # plt.show(block=True)
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

    from scipy.optimize import minimize
    xvec = positions[:,::-1].ravel()
    # res = minimize(objective_function,xvec,args = (A,b,copy.deepcopy(xvec),0.0))
    # x = res.x
    # newpositions[]

    x, residuals, rank, s = np.linalg.lstsq(A, np.asarray(b), rcond=-1)
    # matax[1].plot(s)
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
    # newpositions = res.x.reshape(positions.shape)[:,::-1]
    if show:
        plt.show(block=True)
    return newpositions,xcorr,deltas


def parse_image_shifts(file_path,superres=1):
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
        array[:,:2] /= superres
        # Add the parsed data to the result
        result.append((tilt_angle, array))

    # Sort the list by tilt angle and return
    return sorted(result, key=lambda x: x[0])


def savetomrc(array, fnam, overwrite=True):
    """
    Saves a NumPy array to an MRC file format.

    Parameters
    ----------
    array : numpy.ndarray
        The data array to be saved in MRC format. This is typically a 2D or 3D array.
    
    fnam : str
        The filename for the output MRC file, including the path if necessary.
    
    overwrite : bool, optional
        If True (default), overwrites the file if it already exists. If False, raises
        an error if the file exists.

    Returns
    -------
    None
        This function writes the data to disk and does not return a value.

    Notes
    -----
    The MRC (Medical Research Council) file format is commonly used for storing electron microscopy and 
    other volumetric data. This function uses the `mrcfile` library to write the NumPy array as MRC data.
    """
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


def setup_outputdir(args):
    if args["output"] is None:
        outputdir = os.path.split(args["input"])[1].replace("*.mrc", "_output")
    else:
        outputdir = args["output"]
    print("Results will be outputted to {0}".format(outputdir))
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    return outputdir


def setup_gainreference(gainreffile, filenames, outputdir,shrinkn=20):
    """
    Sets up the gain reference for image processing.

    Parameters:
    gainreffile (str): Path to the gain reference file.
    filenames (list of str): List of filenames to use for generating the gain reference if it does not exist.
    outputdir (str): Directory where the generated gain reference file will be saved.
    shrinkn (int, optional): Size of fringes for removal in unbinned pixels. Default is 20.

    Returns:
    tuple: A tuple containing:
        - gainref (numpy.ndarray): The gain reference array.
        - gainrefmask (numpy.ndarray): The mask of the gain reference after filling holes and applying binary erosion.
    """
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
        gainrefmask, structure=circular_mask([shrinkn] * 2, radius=shrinkn )
    )

    return gainref, gainrefmask


def find_closest_index(arr, target):
    """
    Finds the index of the closest value in the array to the target value.

    Parameters:
    arr (list or np.array): A list or numpy array of numbers
    target (float or int): The target value to find the closest match for

    Returns:
    int: The index of the closest value in the array
    """
    # Convert the input array to a numpy array if it's not already
    arr = np.array(arr)

    # Compute the absolute difference between each element and the target
    diff = np.abs(arr - target)

    # Find the index of the minimum difference
    closest_index = np.argmin(diff)

    return closest_index

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
    outdir = setup_outputdir(args)

    # Binning constant
    binning = int(args["binning"])

    files = glob.glob(args["input"])
    if len(files) < 1:
        raise FileNotFoundError("No files matching {0}".format(args["input"]))
    mdoc_files = [x.replace(".mrc", ".mrc.mdoc") for x in files]

    # Get pixel size, tilt axis rotation from first mdoc file
    mdoc = parse_mdoc(mdoc_files[0])
    pixelsize = float(mdoc["PixelSpacing"])
    if args["tiltaxisrotation"] is None:
        tiltaxisrotation = 0
        # tiltaxisrotation = extract_tilt_axis_angle(mdoc_files[0])
    else:
        tiltaxisrotation = float(args["tiltaxisrotation"])

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

    # Get image shifts from file
    # TODO make SerialEM put this in the mdoc file in microns (not
    # weird TFS units) to obviate this step
    imageshifts = parse_image_shifts(args["image_shifts"],superres=args['correctimageshiftfilefactor'])
    tiltsfromfile = [imageshifts[x][0] for x in range(len(imageshifts))]
    imageshifts = [imageshifts[x][1] for x in range(len(imageshifts))]
    reshapedimshifts = np.concatenate(imageshifts)[:, :2] * pixelsize
    if args['maximageshift'] is not None:
        # Convert image shift criterion from microns to Angstroms
        maxim = args['maximageshift']*1e4
        mask = np.logical_and(*(np.abs(reshapedimshifts)< maxim).T)
    else:
        mask = np.ones(reshapedimshifts.shape[0],dtype=bool)
    # Global range of tiles in Angstroms
    globalwidth = np.ptp(reshapedimshifts[mask], axis=0)
    globalorigin = np.amin(reshapedimshifts[mask], axis=0)

    fringe_size = args["fringe_size"]
    # Load gain ref if available make a new one if not
    checkforgainref = args["gainref"] is not None
    if checkforgainref:
        if os.path.exists(args["gainref"]):
            print("Loading gain reference {0}".format(args["gainref"]))
            with mrcfile.open(args["gainref"], "r+") as m:
                gainref = np.asarray(m.data)
        else:
            print("Gain reference file {0} not found so making new one".format(args["gainref"]))
            checkforgainref = False
    if not checkforgainref:
        gainref = make_gain_ref(files, binning=binning, shrinkn = fringe_size)
        gainreffile = os.path.join(outdir, "gainref.mrc")
        print("Saving gain reference to {0}".format(gainreffile))
        savetomrc(gainref.astype(np.float32), gainreffile)

    

    # make gain reference mask
    gainrefmask = binary_fill_holes(gainref > 0.4 * np.median(gainref))

    # Remove Fresnel fringes from gain reference mask
    gainrefmask = binary_erosion(
        gainrefmask,
        structure=circular_mask(
            [fringe_size / binning ] * 2, radius=fringe_size / binning 
        ),
    )
    for i, (file, tilt) in enumerate(
        tqdm(zip(files, tilts), total=len(files), desc="Stitching montages")
    ):
        # positions = imageshifts[i][1]
        indx = find_closest_index(tiltsfromfile, tilt)
        positions = imageshifts[indx]
        
        # plot_positions(positions[:,:2])
        if args['maximageshift'] is not None:
            tiles = np.where(np.logical_and(*(np.abs(positions) < maxim).T))[0]
        else:
            tiles = None
        # plot_positions(positions[tiles][:,:2],color='k')

        mont = montage(
            file,
            outdir,
            positions,
            pixelsize,
            binning=binning,
            gainref=np.where(gainrefmask, gainref, 1),
            gainrefmask=gainrefmask,
            skipcrosscorrelation=args["skipcrosscorrelation"],
            montagewidth=globalwidth,
            montageorigin=globalorigin,
            tiles = tiles,
            fringe_size=fringe_size,
            # tiles = [24,25,36,37,38,39,54,55,56,57],
            gaincorrectedfile=args["gain_corrected_files"],
            maxshift = args['max_allowed_imshift_correction']
        )


if __name__ == "__main__":
    main()
