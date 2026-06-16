import numpy as np
import matplotlib.pyplot as plt

def generate_montage_shifts(overlap_factor, tiles, shift=[0.0, 0.0]):
    """
    Generates a list of montage shift coordinates to cover a large area using a grid
    of overlapping detector tiles. The generated coordinates represent shifts in pixel
    space required to position the montage tiles, accounting for the overlap between tiles
    and shifting the grid each tilt to avoid excessive dose on any particular region.

    Parameters
    ----------
    overlap_factor : list or tuple of float
        A list or tuple of two values representing the fractional overlap in the x and y
        directions between adjacent tiles. Values should be between 0 and 1.
    tiles : tuple of int
        The number of tiles in the x (width) and y (height) directions (n, m).
    detector_pixels : list or tuple of int
        The dimensions of the detector in pixels (nx, ny).
    shift : list or tuple of float, optional
        An optional initial shift to apply to the montage coordinates in the x and y
        directions. Default is [0.0, 0.0].

    Returns
    -------
    coords : ndarray of shape (n_tiles, 2)
        An array of coordinates in pixel space, where each row corresponds to the shift
        needed for a particular tile in the montage. The x and y shifts are provided
        relative to the detector center.

    Notes
    -----
    - The function calculates the montage shifts such that the center of the grid is
      positioned at (0, 0), with each tile shifted by an amount determined by the overlap
      factor.
    - The x coordinates reverse direction on each row (serpentine or zigzag pattern) to
      minimize the magnitude of successive shifts.
    - The y coordinates are calculated to ensure coverage based on the overlap and tile
      dimensions.

    Example
    -------
    >>> generate_montage_shifts([0.1, 0.1], (3, 3), [1024, 1024])
    array([[ -512.,  -512.],
           [  512.,  -512.],
           [ 1536.,  -512.],
           [ 1536.,     0.],
           [  512.,     0.],
           [ -512.,     0.],
           [ -512.,   512.],
           [  512.,   512.],
           [ 1536.,   512.]])
    """

    # Get number of tiles in x and y direction
    n, m = tiles

    # reciprocal of overlap factor, the overlap fraction
    o = 1 / np.asarray(overlap_factor)

    # Fractional seperation of individual images
    sep = 1 - o

    # Generate a list of fractional y coordinates for the montage centers
    # centered on 0. (shift+0.5)%1.0)+0.5 ensures that we always start within
    # half a field of view of the desired area, m+0.5 ensures that we always
    # cover the desired area. -m/2 centers the coordinates on zero, the step size
    # is 1 - the overlap fraction
    M = (
        np.arange(
            sep[0] / 2 - ((shift[1] + sep[0] / 2) % sep[0]), m + sep[0] / 2, sep[0]
        )
        - m / 2
    )  # + 0.5*((m+1)%2)

    # N +=
    coords = []
    for iim, im in enumerate(M):
        # Generate a list of fractional x coordinates for the montage centers
        # centered on 0. (shift+0.5)%1.0)+0.5 ensures that we always start within
        # half a field of view of the desired area
        
        N = (
            np.arange(
                0.5 - ((shift[0] - 0.25 * (2 * (iim % 2) - 1) + 0.5) % sep[1]),
                n + 0.5 - o[1],
                sep[1],
            )
            - n / 2
        )
        # Reverse direction of x coordinate every row to minimize magnitude of
        # successive image shifts
        for iin in N[:: 1 - 2 * int(iim % 2)]:
            # Each row must be shifted either left or right by 1/4 of the x
            # dimension
            coords += [[iin, im]]

    # Return the result in units of detector pixels
    return np.asarray(coords)


if __name__=='__main__':
    overlap_factor = [4,3]
    tiles = [4,4]
    cmap = plt.get_cmap('viridis')
    fig,ax = plt.subplots()
    for i in range(12):
        coords = generate_montage_shifts(overlap_factor, tiles, shift=[i/12, i/12])
        
        ax.plot(*coords.T,'bo',color= cmap(i/12))
    plt.show(block=True)