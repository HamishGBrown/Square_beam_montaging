import h5py
import numpy as np
import math
from typing import List, Tuple
from scipy.ndimage import correlate
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion
from tqdm import tqdm
import copy
import mrcfile
import matplotlib.pyplot as plt
import os
import png

def renormalize(array, oldmin=None, oldmax=None, newmax=1.0, newmin=0.0):
    """Rescales the array such that its maximum is newmax and its minimum is newmin."""
    if oldmin is not None:
        min_ = oldmin
    else:
        min_ = array.min()

    if oldmax is not None:
        max_ = oldmax
    else:
        max_ = array.max()

    return (
        np.clip((array - min_) / (max_ - min_), 0.0, 1.0) * (newmax - newmin) + newmin
    )



def array_to_RGB(arrayin, cmap=plt.get_cmap("viridis"), vmin=None, vmax=None):
    """Convert an array to RGB using a supplied colormap."""

    kwargs = {"oldmin": vmin, "oldmax": vmax}
    return (cmap(renormalize(arrayin, **kwargs))[..., :3] * 256).astype(np.uint8)


def RGB_to_PNG(RGB_array, fnam):
    """Output an RGB array [shape (n,m,3)] as a .png file."""
    # Get array shape
    n, m = RGB_array.shape[:2]

    # Replace filename ending with .png
    fnam_out = os.path.splitext(fnam)[0] + ".png"
    png.fromarray(RGB_array.reshape((n, m * 3)), mode="RGB").save(fnam_out)


def save_array_as_png(array, fnam, cmap=plt.get_cmap("viridis"), vmin=None, vmax=None):
    """Output a numpy array as a .png file."""
    # Convert numpy array to RGB and then output to .png file
    RGB_to_PNG(array_to_RGB(array, cmap, vmin=vmin, vmax=vmax), fnam)

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


def save_array_to_hdf5(
    arrays: List[np.ndarray],
    filename: str,
    dataset_names: List[str],
    filemode: str = "w",
) -> None:
    """
    Saves a numpy array to an HDF5 file.

    Parameters:
    arrays (List[np.ndarray]): The numpy arrays to save.
    filename (str): The name of the file where the arrays will be saved.
    dataset_names (List[str]): The names of the datasets in the HDF5 file.
    filemode (str): The file mode for opening the HDF5 file.
    """
    with h5py.File(filename, filemode) as hdf:
        for name, data in zip(dataset_names, arrays):
            hdf.create_dataset(name, data=data)


def load_array_from_hdf5(filename: str, dataset_name: str) -> np.ndarray:
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


def calc_row_idx(k: int, n: int) -> int:
    """
    Calculate the row index in a condensed distance matrix.

    Parameters:
    k (int): Index in the condensed distance matrix.
    n (int): Number of original data points.

    Returns:
    int: The row index corresponding to the given condensed index.
    """
    return int(
        math.ceil(
            (1 / 2.0) * (-((-8 * k + 4 * n**2 - 4 * n - 7) ** 0.5) + 2 * n - 1) - 1
        )
    )


def elem_in_i_rows(i: int, n: int) -> int:
    """
    Calculate the number of elements in the first i rows of a condensed distance matrix.

    Parameters:
    i (int): Row index.
    n (int): Number of original data points.

    Returns:
    int: Number of elements in the first i rows.
    """
    return i * (n - 1 - i) + (i * (i + 1)) // 2


def calc_col_idx(k: int, i: int, n: int) -> int:
    """
    Calculate the column index in a condensed distance matrix.

    Parameters:
    k (int): Index in the condensed distance matrix.
    i (int): Row index in the square matrix.
    n (int): Number of original data points.

    Returns:
    int: The column index corresponding to the given condensed index and row index.
    """
    return int(n - elem_in_i_rows(i + 1, n) + k)


def condensed_to_square(k: int, n: int) -> Tuple[int, int]:
    """
    Convert a condensed distance matrix index to a square matrix index.

    Parameters:
    k (int): Index in the condensed distance matrix.
    n (int): Number of original data points.

    Returns:
    tuple: The row and column indices in the square matrix.
    """
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j


def crop(arrayin: np.ndarray, shapeout: Tuple[int, ...]) -> np.ndarray:
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

    def indices(y: int, y_: int) -> Tuple[slice, slice]:
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


def rotation_matrix(theta: float) -> np.ndarray:
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


def fourier_crop(ain: np.ndarray, shapeout: Tuple[int, ...]) -> np.ndarray:
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

    def crop1d(array: np.ndarray, s: int, d: int) -> np.ndarray:
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
    ain: np.ndarray,
    shapeout: Tuple[int, ...],
    norm: str = "conserve_val",
    N: int = None,
    qspace_in: bool = False,
    qspace_out: bool = False,
) -> np.ndarray:
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


def get_imageshifts_for_tilt_angle(file_path: str, tilt: float) -> np.ndarray:
    """
    Retrieve image shifts for a specific tilt angle from an mdoc file.

    Parameters:
    -----------
    file_path : str
        Path to the file containing image shifts.
    tilt : float
        The tilt angle for which to retrieve the image shifts.

    Returns:
    --------
    np.ndarray
        A 2D numpy array containing the image shifts for the specified tilt angle.
    """
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

def center_of_mass_2d(array):
    """
    Calculate the center of mass of a 2D array.

    Parameters
    ----------
    array : 2D numpy array
        Input array for which to calculate the center of mass.

    Returns
    -------
    tuple
        Coordinates of the center of mass (row, col).
    """
    total = np.sum(array)
    if total == 0:
        raise ValueError("The sum of the array elements is zero, cannot determine center of mass.")
    
    rows, cols = np.indices(array.shape)
    center_row = np.sum(rows * array) / total
    center_col = np.sum(cols * array) / total
    
    return center_row, center_col


def broadcast_from_unmeshed(coords):
    """
    For an unmeshed set of coordinates broadcast to a meshed ND array.

    Examples
    --------
    >>> broadcast_from_unmeshed([np.arange(5),np.arange(6)])
    [array([[0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1],
       [2, 2, 2, 2, 2, 2],
       [3, 3, 3, 3, 3, 3],
       [4, 4, 4, 4, 4, 4]]), array([[0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5]])]
    """

    N = len(coords)
    pixels = [a.shape[0] for a in coords]

    # Broadcasting patterns
    R = np.ones((N, N), dtype=np.int16) + np.diag(pixels) - np.eye(N, dtype=np.int16)

    # Broadcast unmeshed grids
    return [np.broadcast_to(a.reshape(rr), pixels) for a, rr in zip(coords, R)]


def r_space_array(pixels, gridsize, meshed=True):
    """
    Return the appropriately scaled ND real space coordinates.

    Parameters
    -----------
    pixels : (N,) array_like
        Pixels in each dimension of a ND array
    gridsize : (N,) array_like
        Dimensions of the array in real space units
    meshed : bool, optional
        Option to output dense meshed grid (True) or output unbroadcasted
        arrays (False)
    """
    # N is the dimensionality of grid
    N = len(pixels)

    # Calculate unmeshed grids
    rspace = [np.fft.fftshift(np.fft.fftfreq(pixels[i], d=1 / gridsize[i])) for i in range(N)]

    # At this point we can return the arrays without broadcasting
    if meshed:
        return broadcast_from_unmeshed(rspace)
    else:
        return rspace


def flatten_beam(image,mask,rotation=0):
    """
    Flattens the beam profile in an image by fitting and subtracting a 4th order polynomial surface.

    Parameters:
    image (numpy.ndarray): The input 2D image array.
    mask (numpy.ndarray): A boolean mask array where True values indicate the region of interest.
    rotation (float, optional): The rotation angle in degrees to apply to the coordinates. Default is 0.
++
    Returns:
    numpy.ndarray: The flattened image with the beam profile subtracted.
    """
    im  = fourier_interpolate(image,[256,256])
    msk = make_mask(im)
    y,x = r_space_array(im.shape,gridsize=im.shape,meshed=True)
    y0,x0= center_of_mass_2d(np.where(msk,1,0))
    y = y - (y0 - image.shape[0]//2)
    x = x - (x0 - image.shape[1]//2)
    M = rotation_matrix(-rotation)
    y,x = np.einsum('ij,jkl->ikl',M,np.array([y,x]))
    
    x_masked = x[msk].flatten()
    y_masked = y[msk].flatten()
    m_masked = im[msk].flatten()
    # A = np.column_stack([x_masked.ravel()**4,y_masked.ravel()**4,np.ones_like(x_masked.ravel())])
    A = np.column_stack([x_masked.ravel()**4,y_masked.ravel()**4,x_masked.ravel()**2,y_masked.ravel()**2,np.ones_like(x_masked.ravel())])
    coeffs, _, _, _ = np.linalg.lstsq(A, m_masked, rcond=None)
    fitted_2d = np.zeros_like(image)
    fitted_2d = coeffs[0]*x**4+coeffs[1]*y**4+coeffs[2]*x**2+coeffs[3]*y**2
    # fig,ax = plt.subplots(ncols=3)
    # vmin=np.percentile(image[mask],1)
    # vmax=np.percentile(image[mask],99)
    # ax[0].imshow(image,vmin=vmin,vmax=vmax)
    # ax[1].imshow(fitted_2d)
    mean = np.mean(image[mask])
    image[mask]= image[mask]-fourier_interpolate(fitted_2d,image.shape)[mask]+mean
    # vmin=np.percentile(image[mask],1)
    # vmax=np.percentile(image[mask],99)
    # ax[1].imshow(image,vmin=vmin,vmax=vmax)
    # ax[2].imshow(fourier_interpolate(fitted_2d,image.shape))
    # plt.show(block=True)
    # image[mask] -= fitted_2d[mask]
    return image

def determine_square_beam_angle(m):
    """
    Determines the orientation angle of a square beam in an image.
    Parameters:
    m (numpy.ndarray): The input 2D array representing the image of the square beam.
    Returns:
    float: The orientation angle of the square beam in degrees, adjusted to be within the range [-45, 45] degrees.
    Notes:
    - The function first resizes the image such that the longest axis is 128 pixels to speed up the calculation.
    - It then pads the resized image with zeros.
    - The Radon transform is applied to the padded image to compute the sinogram.
    - The angle corresponding to the minimum value in the central row of the sinogram is determined.
    - This angle is adjusted to be within the range [-45, 45] degrees.
    """
    
    # Reshape so that longest axis is 128 pixels to speed up the calculation
    if m.shape[0]>m.shape[1]:
        news = [128,128*m.shape[1]//m.shape[0]]
    else:    
        news = [128*m.shape[0]//m.shape[1],128]
    m_ = fourier_interpolate(m,news)

    # Pad the image with zeros
    m_ = np.pad(m_,((64,64),(64,64)))
    
    N = 180
    theta = np.linspace(0, 180, N)

    from skimage.transform import radon
    sinogram = fourier_interpolate(radon(m_, theta=theta),[N,N])
    rot = theta[np.argmin(sinogram[sinogram.shape[0]//2])]
    rotmod45 = (rot+45)%90-45
    return rotmod45

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


def iterative_edge_smoothing(array, mask, niterations=5, pow=4, initial_radius=None):
    """
    Smooths the edges of a given array iteratively using a circular convolution kernel.
    Parameters:
    array (numpy.ndarray): The input array to be smoothed.
    mask (numpy.ndarray): A boolean mask array where True values indicate the regions to be preserved.
    niterations (int, optional): The number of iterations for the smoothing process. Default is 5.
    pow (int, optional): The power to which the iteration index is raised to control the radius of the smoothing kernel. Default is 4.
    Returns:
    numpy.ndarray: The smoothed array with edges processed according to the mask.
    """

    if initial_radius is None:
        R = np.mean(array.shape) / 10
    else:
        R = initial_radius
    for i in tqdm(range(niterations), desc="smoothing"):
        radius = (R - 1) / niterations**pow * np.abs(i - niterations) ** pow + 1
        kernel = np.where(circular_mask(array.shape, radius=radius, center=None), 1, 0)
        kernel = np.fft.fftshift(kernel)
        kernel[0, 0] = 0
        kernel = kernel / np.sum(kernel)
        array = np.where(mask, array, convolve(array, kernel))

    return array


def make_mask(im, shrinkn=20, smoothing_kernel=3,medianthreshold=0.4,absolutethreshold=None):
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
    if absolutethreshold is not None:
        mask = smoothed_im > absolutethreshold
    else:
        mask = smoothed_im > medianthreshold * np.median(im)

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

    weights = []
    xcorr = []
    deltas = []
    shifts = []

    # TODO make this threaded
    if cross_corr_param_file is not None:
        xcorr, deltas = [
            load_array_from_hdf5(cross_corr_param_file, x)
            for x in ("cross_correlations", "relative_shifts")
        ]
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
            # fig,ax = plt.subplots(nrows=2)
            # ax = ax.flatten()
            # for idx, (im, msk) in enumerate(zip([tiles[i],tiles[j]],[reference_mask,moving_mask])):
            #     ax[idx].imshow(im,cmap=plt.get_cmap('gist_gray'))
            #     ax[idx].imshow(msk,alpha=0.2)
            # plt.show(block=True)
            # Calculate shift by masked cross correlation with ordering (Y,X)
            detected_shift, xcorrmax = _masked_phase_cross_correlation(
                tiles[i],
                tiles[j],
                reference_mask=reference_mask,
                moving_mask=moving_mask,
            )
            # delta is measured shift in microns
            delta = np.asarray(detected_shift) * pixel_size * 1e-4
            xcorr.append(xcorrmax)
            deltas.append(delta)
        else:
            delta = deltas[ind]
            xcorrmax = xcorr[ind]
            detected_shift = delta / pixel_size * 1e-4

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
            A.append(xcorrmax * Arow)
            b[len(b) :] = (xcorrmax * delta).tolist()
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
    b = (np.asarray(b) / meanweight).tolist()

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

            # fig,ax = plt.subplots()
            # ax.plot(*np.asarray(island_points).T,'ro',label='island')
            # for idx, (x, y) in enumerate(island_points):
            #     ax.annotate(str(idx), (x, y), textcoords="Island points", xytext=(5, 5), ha='center', fontsize=8)
            # ax.plot(*np.asarray(largest_island_points).T,'bo',label='Anchor')
            # for idx, (x, y) in enumerate(largest_island_points):
            #     ax.annotate(str(idx), (x, y), textcoords="Anchor island points", xytext=(5, 5), ha='center', fontsize=8)

            i, j = find_closest_points_with_kdtree(island_points, largest_island_points)

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

    def objective_function(x, A, b, positions, lam):
        obj = np.linalg.norm(A @ x - b)
        obj += lam * np.linalg.norm(positions - x)
        return obj

    from scipy.optimize import minimize

    xvec = positions[:, ::-1].ravel()
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
        print(dx)
        i1, i2 = array_overlap(dx[0], pixels[0])
        j1, j2 = array_overlap(dx[1], pixels[1])
        print(i1, i2, j1, j2)
        reference_mask = np.zeros_like(masks[i], dtype=bool)
        reference_mask[i1[0] : i1[1], j1[0] : j1[1]] = masks[i][
            i1[0] : i1[1], j1[0] : j1[1]
        ]
        moving_mask = np.zeros_like(masks[j], dtype=bool)
        moving_mask[i2[0] : i2[1], j2[0] : j2[1]] = masks[j][
            i2[0] : i2[1], j2[0] : j2[1]
        ]

        detected_shift, _ = _masked_phase_cross_correlation(
            images[i], images[j], reference_mask=reference_mask, moving_mask=moving_mask
        )

        fig = plt.figure(figsize=(8, 12))
        axes = fig.subplot_mosaic([["Image1", "Image2"], ["Stitched", "Stitched"]])
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 2], hspace=0.3)

        axes["Image1"].imshow(images[i], cmap="gray")
        axes["Image1"].imshow(reference_mask, alpha=0.5, cmap="Reds")
        axes["Image1"].set_title(f"Image {i} with Mask")

        axes["Image2"].imshow(images[j], cmap="gray")
        axes["Image2"].imshow(moving_mask, alpha=0.5, cmap="Reds")
        axes["Image2"].set_title(f"Image {j} with Mask")
        print(
            [positions[i], positions[i] + detected_shift * pixel_size * 1e-4],
            detected_shift,
        )
        aligned_image = stitch(
            [images[i], images[j]],
            [positions[i], positions[i] + detected_shift[::-1] * pixel_size * 1e-4],
            [masks[i], masks[j]],
            pixel_size,
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
