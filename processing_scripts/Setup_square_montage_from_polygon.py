#!Python

import numpy as np
import os
import shapely
import re
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, box


def rectangles_with_overlap(
    polygon_points, rect_points, rect_width, rect_height, overlap_criterion
):
    """
    Function to find rectangles that overlap with a polygon by more than a given fractional area criterion.

    Parameters:
    - polygon_points: A list of points describing the polygon (list of tuples).
    - rect_points: A list of bottom-left corner points for each rectangle (list of tuples).
    - rect_width: The width of each rectangle.
    - rect_height: The height of each rectangle.
    - overlap_criterion: The fractional overlap criterion (between 0 and 1).

    Returns:
    - A list of bottom-left corner points of rectangles that have a fractional overlap greater than the criterion.
    """
    # Create the polygon using the input points
    polygon = Polygon(polygon_points)

    # List to hold rectangles that satisfy the overlap criterion
    overlapping_rectangles = []
    from matplotlib.patches import Rectangle

    #fig,ax = plt.subplots()
    #ax.plot(*np.asarray(polygon_points).T,'k--')
    # Loop through each rectangle
    for rect_point in rect_points:
        # Create the rectangle (as a box) using the bottom-left corner point and the given dimensions
        rect = box(
            rect_point[0],
            rect_point[1],
            rect_point[0] + rect_width,
            rect_point[1] + rect_height,
        )

        # Calculate the intersection area between the polygon and the rectangle
        intersection = polygon.intersection(rect)
        overlap_area = intersection.area

        # Calculate the area of the rectangle
        rect_area = rect_width * rect_height

        # Calculate the fractional overlap
        fractional_overlap = overlap_area / rect_area
        # ax.add_patch(Rectangle(rect_point,rect_width,rect_height,ec='b',fc='none'))
        #ax.plot((rect_point[0],rect_point[0],rect_point[0] + rect_width,rect_point[0] + rect_width),(rect_point[1], rect_point[1] + rect_height, rect_point[1] + rect_height,rect_point[1]),'r-')

        # If the fractional overlap is greater than the overlap criterion, add the rectangle to the result
        if fractional_overlap > overlap_criterion:
            overlapping_rectangles.append(rect_point)
            # ax.plot((rect_point[0],rect_point[0],rect_point[0] + rect_width,rect_point[0] + rect_width),(rect_point[1], rect_point[1] + rect_height, rect_point[1] + rect_height,rect_point[1]),'b-')
    #plt.show(block=True)
    if len(overlapping_rectangles) < 1:
        raise ValueError(f"No acquisitions fit inside given polygon.")
    return np.asarray(overlapping_rectangles)


def extract_points(file_path, item_index):
    """
    Extracts PtsX and PtsY for a given Item index from a SerialEM nav file using string find method.

    Parameters:
    - file_path: path to the file
    - item_index: the index of the Item to extract

    Returns:
    - A numpy array of shape (N, 2) where N is the number of points, containing the PtsX and PtsY values.

    Raises:
    - ValueError: If PtsX or PtsY are missing for the specified item.
    """
    with open(file_path, "r") as file:
        content = file.read()

    # Locate the start of the [Item = item_index] section
    item_header = f"[Item = {item_index}]"
    item_start = content.find(item_header)

    if item_start == -1:
        raise ValueError(f"Item {item_index} not found in the file.")

    # Find the end of this item section (next empty line or new item)
    item_end = content.find("\n\n", item_start)
    if item_end == -1:
        item_end = len(content)

    # Extract the content of the item section
    item_content = content[item_start:item_end]

    # Find and extract the PtsX and PtsY values
    ptsx_start = item_content.find("PtsX = ")
    ptsy_start = item_content.find("PtsY = ")

    if ptsx_start == -1:
        raise ValueError(f"PtsX not found for Item {item_index}.")
    if ptsy_start == -1:
        raise ValueError(f"PtsY not found for Item {item_index}.")

    # Extract the lines with PtsX and PtsY values
    ptsx_start += len("PtsX = ")
    ptsy_start += len("PtsY = ")

    ptsx_end = item_content.find("|\n", ptsx_start)
    ptsy_end = item_content.find("|\n", ptsy_start)

    ptsx_str = item_content[ptsx_start:ptsx_end].strip()
    ptsy_str = item_content[ptsy_start:ptsy_end].strip()

    # Convert the space-separated values to lists of floats
    ptsx = list(map(float, ptsx_str.split()))
    ptsy = list(map(float, ptsy_str.split()))

    if len(ptsx) != len(ptsy):
        raise ValueError("Mismatch between the number of PtsX and PtsY values.")

    # Combine PtsX and PtsY into a NumPy array
    points_array = np.column_stack((ptsx, ptsy))

    return points_array


def plot_points_in_serial_EM_navigator(imageshifts, M, nx, ny, x, y):
    import serialem as sem
    # Get a unique group number to add squares to
    gid = int(sem.GetUniqueNavID())

    # Plot zero-tilt image shifts on navigator
    ids = []
    for imshift in imageshifts:
        xx = imshift[0]
        yy = imshift[1]

        # Generate corners in pixel coordinates
        cornersx = [xx + nx / 2 * (1 - 2 * ((i % 4) // 2)) for i in range(5)]
        cornersy = [yy + ny / 2 * (1 - 2 * ((i + 1) % 4 // 2)) for i in range(5)]

        # Use matrix to convert to specimen coordinates
        cornersx, cornersy = M @ np.stack((cornersx, cornersy), axis=0)

        # Add "origin" (nav point coordinates) to
        cornersx += x
        cornersy += y

        # Write arrays back to serial-EM land
        sem.SetVariable("cornersx", "\n".join([str(a) for a in cornersx]))
        sem.SetVariable("cornersy", "\n".join([str(a) for a in cornersy]))

        # Add image shift acquisitions as polygons in display and group them
        ids.append(int(sem.AddStagePointsAsPolygon("cornersx", "cornersy", z)))
    for id_ in ids:
        sem.ChangeItemGroupID(id_, gid)
    return ids, gid


def dose_symmetric_tilts(maxtilt, step, nreflect):
    """
    Generates a sequence of symmetric positive and negative tilt angles
    based on the given maximum tilt, step size, and reflection group size.

    The function first creates arrays of positive and negative tilts, then
    iteratively adds a set number (`nreflect`) of positive and negative tilts
    in an alternating pattern until all possible tilts are used.

    Parameters
    ----------
    maxtilt : float
        The maximum absolute tilt angle to generate (positive and negative).
    step : float
        The step size between consecutive tilt angles.
    nreflect : int
        The number of tilt angles to alternate between positive and negative
        in each reflection group.

    Returns
    -------
    tilts : list of float
        A list of tilt angles arranged in alternating positive and negative
        groups, symmetrically distributed.

    Example
    -------
    >>> dose_symmetric_tilts(10, 2, 3)
    [0, 2, 4, -2, -4, 6, 8, -6, -8]
    """

    # Create an array of positive tilt values ranging from 0 to (maxtilt).
    # step/10 max np.arange inclusive
    postilts = np.arange(0, maxtilt + step / 10, step)

    # Create an array of negative tilt values ranging from -step to -(maxtilt - step).
    negtilts = np.arange(-step, -maxtilt - step / 10, -step)

    # Determine the number of positive and negative tilt values generated.
    npos = postilts.shape[0]
    nneg = negtilts.shape[0]

    # Initialize an empty list to store the final tilt sequence.
    tilts = []

    # Loop through the tilt arrays, alternating between positive and negative groups.
    # The range ensures we loop enough times to cover all the tilts, based on nreflect.
    for i in range(int(np.ceil(npos / nreflect))):

        # Add a block of 'nreflect' positive tilt values to the tilt list.
        # The minimum ensures we don't exceed the number of positive tilts available.
        tilts += list(postilts[i * nreflect : min((i + 1) * nreflect, npos)])

        # Add a block of 'nreflect' negative tilt values to the tilt list.
        # The minimum ensures we don't exceed the number of negative tilts available.
        tilts += list(negtilts[i * nreflect : min((i + 1) * nreflect, nneg)])

    # Return the final list of alternating positive and negative tilt values.
    return tilts


def generate_montage_shifts(overlap_factor, tiles, detector_pixels, shift=[0.0, 0.0]):
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
                0.5 - ((shift[1] - 0.25 * (2 * (iim % 2) - 1) + 0.5) % 1.0),
                n + 0.5 - o[1],
                1 - o[1],
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
    return np.asarray(coords) * np.asarray(detector_pixels)


def write_tilts_and_image_shifts_to_file(filename, tilts, imageshifts, zs):

    f = open(filename, "w", encoding="utf-8")
    f.write("{0:d}|\n".format(len(tilts)))

    for tilt, ishifts, z in zip(tilts, imageshifts, zs):
        f.write("{0}|\n{1}|\n".format(tilt, len(ishifts)))
        np.savetxt(
            f,
            np.concatenate((ishifts, z.reshape((z.shape[0], 1))), axis=-1),
            fmt="%5.3f",
            newline="|\n",
        )

    f.close()


def calculate_defocii(imageshifts, angles, M):

    # Generate tilt axis vector
    zs = []

    for alpha, imshift in zip(angles, imageshifts):
        # Use camera to specimen transformation matrix to calculate
        # component of image shift

        zs.append(np.tan(np.deg2rad(alpha)) * (M @ imshift.T).T[:, 1])
    return zs

def plot_tilts_and_image_shifts(polygonpoints,tilts,imageshifts,M,ny,nx,zs):
    fig = plt.figure()  
    camerapoints = np.asarray([[0,0],[nx,0],[nx,ny],[0,ny],[0,0]]) @ M.T
    ax = fig.add_subplot(projection='3d')
    
    threedpolypoints = np.concatenate((polygonpoints,polygonpoints[0:1,:]),axis=0)
    Z = np.zeros(threedpolypoints.shape[0])
    threedpolypoints = np.concatenate((threedpolypoints,Z.reshape((Z.shape[0],1))),axis=1)
    ax.plot3D(*(threedpolypoints.T), c='b',alpha=0.5)
    for i, (tilt, imagexy) in enumerate(zip(tilts, imageshifts)):
        if tilt<0:
            continue
        points = imagexy @ M.T
        print((points.reshape((points.shape[0], 1,points.shape[1])) + camerapoints).shape)

        for shift in points.reshape((points.shape[0], 1,points.shape[1])) + camerapoints:
            Z = shift[...,1]*np.tan(np.deg2rad(tilt))
            ax.plot3D(*(np.concatenate([shift,Z.reshape((Z.shape[0],1))],axis=1)).T, 'r-')
        # plt.show()
        # ax.plot(imshift[:, 0], imshift[:, 1], "o", label=f"Tilt: {tilt}")
        # for j, (x, y) in enumerate(imshift):
        #     ax.text(x, y, f"{j}", fontsize=8)
        # for j, (x, y) in enumerate(imshift):    
    plt.show()

def main_serialem(step,maxtilt,ntiltgroup,overlap,outdir,polyoverlapfrac):
    import serialem as sem

    overlapy,overlapx = [int(x) for x in overlap]

    # Get coordinates from polygon and label from last acquired item
    # this script is intended by run after acquisition using "Acquire at items"
    # in serial-EM so the nav item will be the polygon
    item_index, x, y , z, label = sem.ReportNavItem()

    label = sem.GetVariable("navLabel").strip()

    # No way to directly get polygon points in serialEM so the following hack will
    # have to surfice...
    tempnavfile = os.path.join(outdir, "Temp.nav")
    sem.SaveNavigator(tempnavfile)
    polygonpoints = extract_points(tempnavfile, label)
    os.remove(tempnavfile)

    # Get the stage position and label from the last acquired image which should be the view map.
    _________, vx, vy , vz,  label = sem.ReportOtherItem(-1)

    # Generate output filename using navigation item albel
    fnamout = "Montage_imageshifts_{0}.txt".format(label)
    fnamout = os.path.join(outdir, fnamout)

    # Get information about record preset (pixels and pixel size)
    sem.GoToLowDoseArea("R")
    nx, ny, rotflip, p, up, camnum = sem.CameraProperties()

    # Get camera to specimen matrix to convert between camera pixel
    # measurements and coordinates on the specimen
    # 2 is to adjust for the fact that "binning 1" for the K3 is really superres
    # see https://groups.google.com/a/colorado.edu/g/serialem/c/uoEtxqA4xRA
    # delete if not using a K2 or K3
    M = 2*np.asarray(sem.CameraToStageMatrix(0)).reshape((2,2))
    # M = np.asarray(sem.CameraToStageMatrix(0)).reshape((2, 2))
    Minv = np.linalg.inv(M)
    # Convert pixel size to micron
    p = p * 1e-4

    # Generate tilt series
    tilts = dose_symmetric_tilts(maxtilt, step, ntiltgroup)
    print(
        "{0} tilts between {1} and {2} in steps of {3}".format(
            len(tilts), min(tilts), max(tilts), step
        )
    )

    # Get size (range) and origin of polygon to fit in initial guess of montage
    # tiles
    transformed_poly_points = (Minv @ polygonpoints.T).T
    extent = np.ptp(transformed_poly_points, axis=0)
    origin = (
        np.max(transformed_poly_points, axis=0)
        + np.min(transformed_poly_points, axis=0)
    ) / 2

    satisfatory = False
    while not satisfatory:

        #sem.EnterString(
         #   "inp",
         #   "Enter montage tile x and y overlap factor (1/fractional overlap as a series of numbers.|\n eg. 5 4 8 8 for a 5 x 4 montage with 12.5% overlap",
        #)
        #inp = sem.GetVariable("inp").strip()
        #overlapy, overlapx = [int(x) for x in inp.split(" ")]
        ntilts = len(tilts)
        o = [overlapy, overlapx]

        imageshifts = []
        for itilts in range(ntilts):

            # Generate montage shifts for this tilt, shift by overlap
            # output is in pixels
            imagexy = generate_montage_shifts(
                o,
                [extent[0] / nx, extent[1] / ny],
                [nx, ny],
                shift=[(itilts / o[0]) % 1.0, (itilts / o[1]) % 1.0],
            )

            # Shift image shifts by 1/2 a Record field of view
            fov = np.asarray((nx, ny))
            imagexy -= fov / 2

            # Dilate the polygon by the cosine of the tilt angle to account for
            # the fact that the field of view is larger at higher tilts
            dilated_poly_points = transformed_poly_points.copy()
            dilated_poly_points[:,1] = (transformed_poly_points[:,1] - vy)*np.cos(np.deg2rad(tilts[itilts])) + vy

            # Remove images with insufficient overlap with the polygon
            imagexy = (
                rectangles_with_overlap(
                    transformed_poly_points, imagexy + origin, nx, ny, polyoverlapfrac
                )
                - origin + fov/2
            )

            imageshifts.append(imagexy)

        ids, gid = plot_points_in_serial_EM_navigator(imageshifts[0], M, nx, ny, vx, vy)
        
        # Script can be configured to ask for user input to check if the montage is satisfactory
        satisfactory = True
        # satisfatory = sem.YesNoBox("Satisfactory?") == 1

        # if not satisfatory:
        #     for id_ in ids[::-1]:
        #         sem.DeleteNavigatorItem(id_)
    # Calculate defocii to compensate for distance from tilt axis
    zs = calculate_defocii(imageshifts,tilts,M)

    # Save image shifts for Serial-EM in units of pixels in the camera
    # basis (x and y aligned with camera axes). Imageshifts will be 
    # converted to basis aligned with tilt axis inside SerialEM
    write_tilts_and_image_shifts_to_file(fnamout,tilts,imageshifts,zs)
    print('Save {0}'.format(fnamout))

def main_sandbox(step,maxtilt,ntiltgroup,overlap,outdir,polyoverlapfrac,polygonpoints=None,nx=5076,ny=4092,p=0.65):

    overlapy,overlapx = [int(x) for x in overlap]

    # Get coordinates from polygon and label from last acquired item
    # this script is intended by run after acquisition using "Acquire at items"
    # in serial-EM so the nav item will be the polygon
    if polygonpoints is None:
        polygonpoints = np.asarray([[-1.0,0.0],[-1.0,1.0],[-0.75,1.0],[-0.75,0.75],[0.75,0.75],[0.75,1.0],[1.0,1.0],[1.0,0.0]])
    
    vx,vy = np.mean(polygonpoints,axis=0)
    

    # Generate output filename using navigation item albel
    fnamout = "Montage_imageshifts.txt"
    fnamout = os.path.join(outdir, fnamout)

    # Convert pixel size to micron
    pmicron = p * 1e-4

    # Generate tilt series
    tilts = dose_symmetric_tilts(maxtilt, step, ntiltgroup)
    print(
        "{0} tilts between {1} and {2} in steps of {3}".format(
            len(tilts), min(tilts), max(tilts), step
        )
    )
    M = np.eye(2)*pmicron
    Minv = np.linalg.inv(M)

    # Get size (range) and origin of polygon to fit in initial guess of montage
    # tiles
    transformed_poly_points = (Minv @ polygonpoints.T).T
    extent = np.ptp(transformed_poly_points, axis=0)
    origin = (
        np.max(transformed_poly_points, axis=0)
        + np.min(transformed_poly_points, axis=0)
    ) / 2

    # Generate corners in pixel coordinates
    cornersx = np.asarray([nx / 2 * (1 - 2 * ((i % 4) // 2)) for i in range(5)])
    cornersy = np.asarray([ny / 2 * (1 - 2 * ((i + 1) % 4 // 2)) for i in range(5)])
    corners = np.stack((cornersx, cornersy), axis=1)

    satisfatory = False
    while not satisfatory:

        #sem.EnterString(
         #   "inp",
         #   "Enter montage tile x and y overlap factor (1/fractional overlap as a series of numbers.|\n eg. 5 4 8 8 for a 5 x 4 montage with 12.5% overlap",
        #)
        #inp = sem.GetVariable("inp").strip()
        #overlapy, overlapx = [int(x) for x in inp.split(" ")]
        ntilts = len(tilts)
        o = [overlapy, overlapx]

        imageshifts = []
        for itilts in range(ntilts):

            # Generate montage shifts for this tilt, shift by overlap
            # output is in pixels
            imagexy = generate_montage_shifts(
                o,
                [extent[0] / nx, extent[1] / ny],
                [nx, ny],
                shift=[(itilts / o[0]) % 1.0, (itilts / o[1]) % 1.0],
            )

            # Shift image shifts by 1/2 a Record field of view
            fov = np.asarray((nx, ny))
            imagexy -= fov / 2

            # Dilate the polygon by the cosine of the tilt angle to account for
            # the fact that the field of view is larger at higher tilts
            dilated_transformed_points = transformed_poly_points.copy()
            # dilated_transformed_points[:,1] = (transformed_poly_points[:,1] - vy)*np.cos(np.deg2rad(tilts[itilts])) + vy
            
            
            # Remove images with insufficient overlap with the polygon
            imagexy = (
                rectangles_with_overlap(
                    dilated_transformed_points, imagexy + origin, nx, ny, polyoverlapfrac
                )
                - origin + fov/2
            )

            fig,ax = plt.subplots()
            ax.plot(*np.asarray(dilated_transformed_points).T,'k--')
            points = np.asarray(imagexy+origin)
            for imxy in points:
                ax.plot(*(imxy+corners).T,'r-')
            # ax.plot(*.T,'ro')
            plt.show()
            import sys;sys.exit()
            imageshifts.append(imagexy)

        # ids, gid = plot_points_in_serial_EM_navigator(imageshifts[0], M, nx, ny, x, y)
        
        # Script can be configured to ask for user input to check if the montage is satisfactory
        satisfatory = True
        # satisfatory = sem.YesNoBox("Satisfactory?") == 1

        # if not satisfatory:
        #     for id_ in ids[::-1]:
        #         sem.DeleteNavigatorItem(id_)
    # Calculate defocii to compensate for distance from tilt axis
    zs = calculate_defocii(imageshifts,tilts,M)

    # Save image shifts for Serial-EM in units of pixels in the camera
    # basis (x and y aligned with camera axes). Imageshifts will be 
    # converted to basis aligned with tilt axis inside SerialEM
    write_tilts_and_image_shifts_to_file(fnamout,tilts,imageshifts,zs)
    plot_tilts_and_image_shifts(polygonpoints,tilts,imageshifts,M,ny,nx,zs)
    print('Save {0}'.format(fnamout))

if __name__ == "__main__":

    # Parse commandline arguments
    step = 3
    maxtilt = 60
    ntiltgroup = 3
    overlap= [5,7]
    outdir = "./"
    polyoverlapfrac = 0.2
    
    # main_serialem(step,maxtilt,ntiltgroup,overlap,outdir,polyoverlapfrac)
    main_sandbox(step,maxtilt,ntiltgroup,overlap,outdir,polyoverlapfrac)
    