#!Python
import serialem as sem
import numpy as np
import os


def dose_symmetric_tilts(maxtilt,step,nreflect):
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
    postilts = np.arange(0, maxtilt+step/10, step)
    
    # Create an array of negative tilt values ranging from -step to -(maxtilt - step).
    negtilts = np.arange(-step, -maxtilt-step/10, -step)
    
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
        tilts += list(postilts[i * nreflect:min((i + 1) * nreflect, npos)])
        
        # Add a block of 'nreflect' negative tilt values to the tilt list.
        # The minimum ensures we don't exceed the number of negative tilts available.
        tilts += list(negtilts[i * nreflect:min((i + 1) * nreflect, nneg)])

    # Return the final list of alternating positive and negative tilt values.
    return tilts

def generate_montage_shifts(overlap_factor,tiles,detector_pixels,shift=[0.0,0.0]):
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
    n,m = tiles

    # reciprocal of overlap factor, the overlap fraction
    o = 1/np.asarray(overlap_factor)

    # Fractional seperation of individual images
    sep = (1-o)
    
    # Generate a list of fractional y coordinates for the montage centers
    # centered on 0. (shift+0.5)%1.0)+0.5 ensures that we always start within
    # half a field of view of the desired area, m+0.5 ensures that we always
    # cover the desired area. -m/2 centers the coordinates on zero, the step size
    # is 1 - the overlap fraction 
    M = (np.arange(sep[0]/2-((shift[1]+sep[0]/2)%sep[0]),m+sep[0]/2,sep[0]) - m / 2) #+ 0.5*((m+1)%2)

    # N += 
    coords = []
    for iim,im in enumerate(M):
        # Generate a list of fractional x coordinates for the montage centers
        # centered on 0. (shift+0.5)%1.0)+0.5 ensures that we always start within
        # half a field of view of the desired area
        N = (np.arange(0.5-((shift[1]-0.25*(2*(iim%2)-1)+0.5)%1.0),n+0.5-o[1],1-o[1]) - n / 2)
        # Reverse direction of x coordinate every row to minimize magnitude of
        # successive image shifts
        for iin in N[::1-2*int(iim%2)]:
            # Each row must be shifted either left or right by 1/4 of the x
            # dimension
            coords += [[iin,im]]
    
    # Return the result in units of detector pixels
    return np.asarray(coords)*np.asarray(detector_pixels)

def write_tilts_and_image_shifts_to_file(filename,tilts,imageshifts,zs):

    f = open(filename,'w', encoding='utf-8')
    f.write('{0:d}|\r|\n'.format(len(tilts)))

    for tilt,ishifts,z in zip(tilts,imageshifts,zs):
        f.write('{0}|\r|\n{1}|\r|\n'.format(tilt,len(ishifts)))
        np.savetxt(f,np.concatenate((ishifts,z.reshape((z.shape[0],1))),axis=-1),fmt='%5.3f',newline='|\r|\n')

    f.close()
    
def calculate_defocii(imageshifts,angles,M):

    # Generate tilt axis vector
    zs = []
    
    for alpha,imshift in zip(angles,imageshifts):
        # Use camera to specimen transformation matrix to calculate
        # component of image shift 

        zs.append(np.tan(np.deg2rad(alpha))*( M @ imshift.T).T[:,1])
    return zs

def plot_points_in_serial_EM_navigator(imageshifts,M,nx,ny,x,y):
    # Get a unique group number to add squares to
    gid = int(sem.GetUniqueNavID())
    print('gid',gid)

    # Plot zero-tilt image shifts on navigator
    ids = []
    for imshift in imageshifts:
        xx = imshift[0]
        yy = imshift[1]

        # Generate corners in pixel coordinates
        cornersx = [xx+nx/2*(1-2*((i%4)//2)) for i in range(5)]
        cornersy = [yy+ny/2*(1-2*((i+1)%4//2)) for i in range(5)]
        
        # Use matrix to convert to specimen coordinates
        cornersx,cornersy = M @ np.stack((cornersx,cornersy),axis=0)

        # Add "origin" (nav point coordinates) to 
        cornersx += x
        cornersy += y

        # Write arrays back to serial-EM land
        sem.SetVariable('cornersx','\n'.join([str(a) for a in cornersx]))
        sem.SetVariable('cornersy','\n'.join([str(a) for a in cornersy]))

        # Add image shift acquisitions as polygons in display and group them
        ids.append(int(sem.AddStagePointsAsPolygon('cornersx','cornersy',z)))
    for id_ in ids:
        sem.ChangeItemGroupID(id_,gid)
    return ids,gid

if __name__=='__main__':
    
    # Parse commandline arguments
    step = 3
    maxtilt = 60
    ntiltgroup = 3
    m,n = [5,5]
    overlap = [8,8]
    outdir = 'Z:\Hamish'


    # Get information about navigation item
    indx, x, y, z, label = sem.ReportNextNavItem()
    label = sem.GetVariable("navLabel").strip()

    # Generate output filename using navigation item albel
    fnamout = 'Montage_imageshifts_{0}.txt'.format(label)
    fnamout = os.path.join(outdir,fnamout)

    # Get information about record preset (pixels and pixel size)
    sem.GoToLowDoseArea('R')
    nx,ny,rotflip,p,up,camnum = sem.CameraProperties()

    # Get camera to specimen matrix to convert between camera pixel 
    # measurements and coordinates on the specimen
    M = 2*np.asarray(sem.CameraToStageMatrix(0)).reshape((2,2))
    print(M)
    
    #Convert pixel size to micron
    p=p*1e-4

    # Generate tilt series
    tilts = dose_symmetric_tilts(maxtilt,step,ntiltgroup)
    print('{0} tilts between {1} and {2} in steps of {3}'.format(len(tilts),min(tilts),max(tilts),step))
       
    satisfatory = False
    while not satisfatory:

        sem.EnterString("inp","Enter number of x and y montage tiles and their x and y overlap factor (1/fractional overlap as a series of numbers.|\n eg. 5 4 8 8 for a 5 x 4 montage with 12.5% overlap")
        inp = sem.GetVariable("inp").strip()

        m,n,overlapy,overlapx = [int(x) for x in inp.split(' ')]
        ntilts = len(tilts)
        o = [overlapy,overlapx]

        imageshifts = []
        for itilts in range(ntilts):

            # Generate montage shifts for this tilt, shift by overlap
            # output is in pixels
            imagexy = generate_montage_shifts(o,[m,n],[nx,ny],shift=[(itilts/o[0])%1.0,(itilts/o[1])%1.0])

            imageshifts.append(imagexy)

        ids,gid = plot_points_in_serial_EM_navigator(imageshifts[0],M,nx,ny,x,y) 
        
        satisfatory = sem.YesNoBox('Satisfactory?') ==1
        
        if not satisfatory:
            for id_ in ids[::-1]:
                sem.DeleteNavigatorItem(id_)
            

    # Calculate defocii to compensate for distance from tilt axis
    zs = calculate_defocii(imageshifts,tilts,M)

    # Save image shifts for Serial-EM in units of pixels in the camera
    # basis (x and y aligned with camera axes). Imageshifts will be 
    # converted to basis aligned with tilt axis inside SerialEM
    write_tilts_and_image_shifts_to_file(fnamout,tilts,imageshifts,[z/1e4 for z in zs])
    print('Save {0}'.format(fnamout))
