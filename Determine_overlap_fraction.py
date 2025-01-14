#!Python
import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import argparse
from scipy.ndimage import binary_fill_holes, binary_erosion
import serialem as sem


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

def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(
        description="Determine actual overlap fraction for square beam data."
    )
    parser.add_argument(
        "-i", "--input", help="*.mrc file for raw data", required=True, type=str
    )
    parser.add_argument(
        "-o", "--overlapfactor", help="Overlap factor in x and y", required=True, type=int, nargs=2,metavar=("oy","ox")
    )
    return vars(parser.parse_args())


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


def main():
    # Take record image and read from serial EM buffer
    a = sem.R()
    image = np.asarray(sem.bufferImage('A'))
    
    sem.EnterString("inp","Enter tile overlap factor (x y)")
    inp = sem.GetVariable("inp").strip()
    oy,ox = [int(x) for x in inp.split(' ')]


    vmin = np.percentile(image, 10)
    vmax = np.percentile(image, 90)

    # Plot the original image and create space for the eroded image
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.25)

    ax_overlapy = plt.axes([0.33, 0.1, 0.25, 0.03])
    ax_overlapx = plt.axes([0.66, 0.1, 0.25, 0.03])
    oy,ox = [10,10]
    overlapy_slider = Slider(ax_overlapy, "Y overlap", 2, 20, valinit=oy, valstep=1,valfmt='%0.0f')
    overlapx_slider = Slider(ax_overlapx, "X overlap", 2, 20, valinit=ox, valstep=1,valfmt='%0.0f')
    # plt.subplots_adjust(bottom=0.25)

    # Display the original image
    ax[0].imshow(image,vmin=vmin,vmax=vmax)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Placeholder for blurred image
    mask = np.where(make_mask(image, 20,3),1,0)  # Initial blur with small sigma
    ax[2].axis("off")

    def makeoverlapmaps(oy,ox,mask):
        overlapmapy = copy.deepcopy(mask)
        s = mask.shape[0]
        overlapmapy[:s//oy] += mask[-s//oy:]
        overlapmapy[-s//oy:] += mask[:s//oy]
        overlapmapx = copy.deepcopy(mask)
        overlapmapx[:,:s//ox] += mask[:,-s//ox:]
        overlapmapx[:,-s//ox:] += mask[:,:s//ox]
        return overlapmapy,overlapmapx

    overlapmapy,overlapmapx = makeoverlapmaps(oy,ox,mask)
    ax[1].imshow(overlapmapy,vmin=0,vmax=2)
    ax[2].imshow(overlapmapx,vmin=0,vmax=2)

    def update(val):
        oy = overlapy_slider.val
        ox = overlapx_slider.val
        overlapmapy,overlapmapx = makeoverlapmaps(oy,ox,mask)
        overlapyfraction = np.sum(overlapmapy == 2) / np.prod(overlapmapy.shape)
        ax[1].set_data(overlapmapy)
        ax[1].set_title('Overlapy fraction: {0:.2f}'.format(overlapyfraction))
        
        overlapxfraction = np.sum(overlapmapx == 2) / np.prod(overlapmapx.shape)
        ax[2].set_data(overlapmapx)
        ax[2].set_title('Overlapx fraction: {0:.2f}'.format(overlapxfraction))
        fig.canvas.draw_idle()

    overlapx_slider.on_changed(update)
    overlapy_slider.on_changed(update)

    # Show the interactive plot
    plt.show()

if __name__=='__main__':
    main()