import argparse
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt


def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(
        description="Stitch square beam montage tomography data."
    )
    parser.add_argument(
        "-tr", "--tiltaxisrotation", help="Rotation between tilt axis and camera axes (required)", required=True, type=float
    )

    parser.add_argument(
        "-ts", "--tiltstep", help="Step of tomography tilt series in degrees (3 by default)", required=False, type=float,default=3.0
    )

    parser.add_argument(
        "-tm", "--maxtilt", help="Maximum tilt in tilt series (required)", required=True, type=float
    )

    parser.add_argument(
        "-tg", "--dosesymmetrictiltgroup", help="Number of tilts in dose-symmetric tilt group (3 by default)", required=False, type=int, default = 3
    )

    parser.add_argument(
        "-p", "--pixelsize", help="Pixel size of camera in Angstrom  (required)", required=True, type=float
    )

    parser.add_argument(
        "-n", "--camerapixels", help="Detector size in pixels (required)", required=True, nargs=2,type=int
    )

    parser.add_argument(
        "-M", "--montagetiles", help="Number of tiles in montage (required)", required=True, nargs=2,type=int
    )

    parser.add_argument(
        "-ov", "--montageoverlap", help="Montage overlap factor (1/overlap fraction), default 10", required=False, nargs=2,type=float,default=[10.0,10.0]
    )

    parser.add_argument(
        "-o", "--output", help="Output basename (default Imageshifts)", required=False, type=str,default="Imageshits"
    )

    parser.add_argument(
        "-plt", "--plot", help="Plot imageshifts (default True)", required=False, action='store_true'
    )

    return vars(parser.parse_args())

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

    return np.array([[ct,-st],[st,ct]])

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

    # reciprocal of overlap factor
    o = 1/np.asarray(overlap_factor)

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

    for tilt,ishifts,z in zip(tilts,imageshifts,zs):
        f.write('{0}\r\n{1}\r\n'.format(tilt,len(ishifts)))
        np.savetxt(f,np.concatenate((ishifts,z.reshape((z.shape[0],1))),axis=-1),fmt='%5.3f',newline='\r\n')

    f.close()
    
def calculate_defocii(imageshifts,angles,tiltaxisrotation=0):

    # Generate tilt axis vector
    zs = []
    R = rotation_matrix(tiltaxisrotation)
    for alpha,imshift in zip(angles,imageshifts):
        # calculate cross product (component of image shift away from tilt axis)
        # times tangent of alpha

        zs.append(np.tan(np.deg2rad(alpha))*( R @ imshift.T).T[:,1])
    return zs



if __name__=='__main__':
    
    # Parse commandline arguments
    args = parse_commandline()
    step = float(args['tiltstep'])
    tiltaxisrotation = float(args['tiltaxisrotation'])
    maxtilt = float(args['maxtilt'])
    ntiltgroup = int(args['dosesymmetrictiltgroup'])
    p = float(args['pixelsize'])
    nx,ny = [int(x) for x in args['camerapixels']]
    m,n = [int(x) for x in args['montagetiles']]
    overlap = [float(x) for x in args['montageoverlap']]
    output = args['output']
    generate_plot = args['plot']

    # Generate tilt series
    tilts = dose_symmetric_tilts(maxtilt,step,ntiltgroup)
    print('{0} tilts between {1} and {2} in steps of {3}'.format(len(tilts),min(tilts),max(tilts),step))
    ntilts = len(tilts)
    o = overlap
    
    # Initialize plot if requested
    if generate_plot:
        ncols,nrows = 2*[int(np.ceil(np.sqrt(ntilts)))]
        fig,ax = plt.subplots(ncols=ncols,nrows=nrows,figsize=(4*ncols,4*nrows))

    imageshifts = []
    for itilts in range(ntilts):

        # Generate montage shifts for this tilt, shift by overlap
        # output is in pixels
        imagexy = generate_montage_shifts(o,[m,n],[nx,ny],shift=[(itilts/o[0])%1.0,(itilts/o[1])%1.0])

        if generate_plot:
            
            # Get axis and plot whole montage field of view
            i,j = [itilts//ncols,itilts%ncols]
            ax[i,j].add_artist(Rectangle([-nx/2*p*m,-ny/2*p*n],nx*p*m,ny*p*n,ec='k',fc='none',linestyle='--'))
            ax[i,j].set_title('{0:4.1f}°'.format(tilts[itilts]))
            a = ax[i,j]

            # Plot image-shift points
            a.plot(imagexy[:,0]*p,imagexy[:,1]*p,'k--',linewidth=0.25)

            # Add rectangles for camera field of view
            for i,xy in enumerate(imagexy):
                x,y = [x*p for x in xy]
                a.add_artist(Rectangle([x-nx/2*p,y-ny/2*p],nx*p,ny*p,ec='none',fc='b',alpha=1/5))
                ax[-1,-1].add_artist(Rectangle([x-nx/2*p,y-ny/2*p],nx*p,ny*p,ec='none',fc='b',alpha=1/ntilts/2))
                # Annotate first plot
                if itilts ==0:
                    a.annotate(str(i),xy)
            a.set_aspect('equal')
            
            if itilts ==0:
                minx,maxx = [np.amin(imagexy[:,0])*p-nx*p,np.amax(imagexy[:,0])*p+nx*p*1.5]
                miny,maxy = [np.amin(imagexy[:,1])*p-ny*p,np.amax(imagexy[:,1])*p+ny*p*1.5]
            # Plot tilt-axis
            a.plot([minx,maxx], [minx*np.tan(np.deg2rad(tiltaxisrotation)),maxx*np.tan(np.deg2rad(tiltaxisrotation))],'k--')
            # a.text((maxx-minx)/6*5+minx,(minx+(maxx-minx)/6*5),'Tilt axis',ha='left',va='center')
            a.set_xlim(minx,maxx)
            a.set_ylim(miny,maxy)

        # Apply tilt axis rotation since Serial-EM accepts shifts in basis
        # along and perpendicular to tilt axis.
        # Dep
        # imagexy = (rotation_matrix(tiltaxisrotation) @ imagexy.T).T
        imageshifts.append(imagexy)

    if generate_plot:
        # Hide unused sub-plots
        for itilts in np.arange(ntilts,ncols*nrows-1):
            i,j = [itilts//ncols,itilts%ncols]
            ax[i,j].set_axis_off()
        ax[-1,-1].set_xlim(minx,maxx)
        ax[-1,-1].set_ylim(miny,maxy)
        ax[-1,-1].set_title('Accumulated dose indication')
        fig.savefig(output + '.pdf')

    # Calculate defocii to compensate for distance from tilt axis
    zs = calculate_defocii(imageshifts,tilts)

    # Save image shifts for Serial-EM in units of pixels in the camera
    # basis (x and y aligned with camera axes). Imageshifts will be 
    # converted to basis aligned with tilt axis inside SerialEM
    write_tilts_and_image_shifts_to_file(output + '.txt',tilts,imageshifts,[z/1e4 for z in zs])