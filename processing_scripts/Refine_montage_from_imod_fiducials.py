#!/usr/bin/env python3
from .Utilities import *
from PIL import Image
import numpy as np
import argparse
import os

def parse_commandline() -> Dict[str, Any]:
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(
        description="Refine montage stitching using imod fiducial tracking information."
    )
    # parser.add_argument(
    #     "-i", "--input", help="*.mrc wildcard for raw data", required=True, type=str
    # )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory, if left blank output will be placed in a the same folder as the h5 file",
        required=False,
        type=str,
    )
    # parser.add_argument(
    #     "-m",
    #     "--mod",
    #     help="IMOD *.fid model file",
    #     required=True,
    #     type=str,
    # )
    parser.add_argument(
        "-r",
        "--resid",
        help="IMOD *.resid model residual file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-p",
        "--prexg",
        help="IMOD *.prexg prealignment output, which is the output of coarse alignment step in IMOD",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-h5",
        "--h5file",
        help="h5 file containing refined montage tile positions and indices to be corrected",
        required=True,
        type=str,
    )
    return vars(parser.parse_args())

def plot_residuals_on_montage(index_map, montage, x, y, rx, ry, correction,
                              positions_original=None, positions_adjusted=None,
                              pixel_size=None, title=None,scale=None):
    """
    Overlay residuals and per-tile average corrections on index_map and montage.

    Parameters
    ----------
    index_map : 2D array
        Tile index map for a single tilt (shape [ny, nx]).
    montage : array or PIL Image
        Montage image for the same tilt.
    x, y : 1D arrays
        Fiducial positions in montage pixel coordinates.
    rx, ry : 1D arrays
        Per-fiducial residuals.
    correction : dict or array-like of shape (ntiles, 2)
        correction[tile_idx] = [mean_rx, mean_ry] for that tile.
        If a dict, keys are tile indices; if an array, indexed by tile index.
    positions_original : array of shape (ntiles, 2), optional
        Tile positions in microns as loaded from the HDF5 file (before adjustment).
    positions_adjusted : array of shape (ntiles, 2), optional
        Tile positions in microns after applying corrections.
    pixel_size : float, optional
        Pixel size in Angstroms, used to convert positions from microns to pixels.
    title : str, optional
        Overall figure title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

    # --- left panel: index map ---
    axes[0].imshow(index_map, origin='upper', cmap='Set3')
    axes[0].set_title('Index map')

    # --- right panel: montage ---
    montage_arr = np.asarray(montage)
    vmin = np.percentile(montage_arr, 1)
    vmax = np.percentile(montage_arr, 99)
    axes[1].imshow(montage_arr, origin='upper', cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title('Montage')

    # Scale factor so arrows are visible relative to image size
    if scale is None:
        scale = max(index_map.shape) * 0.001

    for ax in axes:
        # Per-fiducial residuals as quiver
        ax.quiver(x, y, rx, ry,
                  color='yellow', alpha=0.7,
                  angles='xy', scale_units='xy', scale=1.0 / scale,
                  width=0.002, label='Residuals')

        # Per-tile average correction: place arrow at tile centroid
        ntiles = int(np.amax(index_map)) + 1
        for tile_idx in range(ntiles):
            mask = index_map == tile_idx
            if not np.any(mask):
                continue
            # centroid of tile pixels
            ys_tile, xs_tile = np.where(mask)
            cx, cy = xs_tile.mean(), ys_tile.mean()

            # get correction for this tile
            try:
                crx, cry = correction[tile_idx]
            except (KeyError, IndexError):
                continue

            ax.quiver(cx, cy, crx, cry,
                      color='red', alpha=0.9,
                      angles='xy', scale_units='xy', scale=1.0 / scale,
                      width=0.004)
            ax.text(cx, cy,
                    f'{tile_idx}\n({crx:.2f},{cry:.2f})',
                    color='red', fontsize=7, ha='center', va='bottom')

        # Plot tile positions from stitching (original and adjusted)
        if positions_original is not None and pixel_size is not None:
            microns_per_pixel = pixel_size * 1e-4
            px_orig = positions_original[:, 0] / microns_per_pixel
            py_orig = positions_original[:, 1] / microns_per_pixel
            ax.scatter(px_orig, py_orig, marker='o', s=40,
                       facecolors='none', edgecolors='cyan', linewidths=1.5,
                       label='Positions (original)', zorder=5)
            for tile_idx, (px, py) in enumerate(zip(px_orig, py_orig)):
                ax.text(px, py, str(tile_idx), color='cyan', fontsize=6,
                        ha='left', va='top')

        if positions_adjusted is not None and pixel_size is not None:
            microns_per_pixel = pixel_size * 1e-4
            px_adj = positions_adjusted[:, 0] / microns_per_pixel
            py_adj = positions_adjusted[:, 1] / microns_per_pixel
            ax.scatter(px_adj, py_adj, marker='+', s=60,
                       color='lime', linewidths=1.5,
                       label='Positions (adjusted)', zorder=5)

        ax.legend(loc='upper right', fontsize=8)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig, axes


def main():
    args = parse_commandline()

    if args['output'] is not None:
        output_dir = args['output']
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    else:
        output_dir = os.getcwd()
    
    # outputmodelfile = os.path.splitext(inputfidfile)[0] + '.txt'
    # os.system('model2point -inp {0} -ou {1} -c'.format(inputfidfile,outputmodelfile))
    # fiducials = np.loadtxt(outputmodelfile)
    residuals = np.loadtxt(args['resid'],skiprows=1)
    
    # Get shifts from prealignment
    prexgfile = args['prexg']
    prealignmentshifts = np.loadtxt(prexgfile)[:,-2:]

    h5filename = args["h5file"]
    positions_from_stitching = load_array_from_hdf5(h5filename,'Refined_positions')
    tile_indices = load_array_from_hdf5(h5filename,'index_map')
    tilts = load_array_from_hdf5(h5filename,'tilts')
    montagefnams = load_array_from_hdf5(h5filename,'Montage_filenames')
    pixel_size = load_array_from_hdf5(h5filename,'binned_pixel_size')[0]


    # Convert positions_from_stitching from microns to pixels
    # positions_from_stitching
    # Adjust poisitions using information from prealignment
    # Remember positions_from_stitching is in micron
    positions_from_stitching_original = positions_from_stitching.copy()
    positions_from_stitching += prealignmentshifts[:,None,:]*pixel_size*1e-4
    print(positions_from_stitching)
    # import sys;sys.exit()

    # c, x, y, z = fiducials.T
    # resid = np.atleast_2d(residuals)
    # print(residuals,residuals.shape)
    x,y,z, rx,ry = residuals.T
    # Adjust x and  y for prealignment shifts applied in IMOD cross-correlation step
    for iz in range(len(tilts)):
        m = z== iz
        x[m] -= prealignmentshifts[iz,0]
        y[m] -= prealignmentshifts[iz,1]
    y = tile_indices.shape[1] - y

    y_indx = np.clip(np.round(y).astype(int),0,tile_indices.shape[1],dtype=int)
    x_indx = np.clip(np.round(x).astype(int),0,tile_indices.shape[2],dtype=int)
    print(y_indx,x_indx,y_indx.shape,x_indx.shape,z.shape)
    # Get tile indices
    indx  = np.array([tile_indices[int(zz),yy,xx] for xx,yy,zz in zip(x_indx,y_indx,z)])

    # Average corrections to x and y, stored as dict per tilt;
    # also apply them directly to positions_from_stitching
    corrections_per_tilt = {}
    for iz in range(len(tilts)):
        m = z == iz
        corr = {}
        for ind in np.arange(int(np.amax(tile_indices[iz])) + 1):
            tile_mask = indx[m] == ind
            if tile_mask.any():
                crx = np.mean(rx[m][tile_mask])
                cry = -np.mean(ry[m][tile_mask])
                corr[ind] = [crx, cry]
                positions_from_stitching[iz, ind, 0] += crx*pixel_size*1e-4
                positions_from_stitching[iz, ind, 1] += cry*pixel_size*1e-4
        corrections_per_tilt[iz] = corr

        # Save corrected positions for this tilt to its own hdf5 file
        out_h5 = os.path.join(output_dir, f'tilt_{tilts[iz]:02.0f}_corrected_positions.h5')
        save_array_to_hdf5(
            [positions_from_stitching[iz]],
            out_h5,
            ['Refined_positions'],
        )
        print(f'Saved corrected positions for tilt {iz} to {out_h5}')

        # Visualise middle tilt
        if iz == 41//2:
            m = z == iz
            montage = Image.open(montagefnams[iz])
            fig,axes = plot_residuals_on_montage(
                index_map=tile_indices[iz],
                montage=montage,
                x=x[m], y=y[m],
                rx=rx[m], ry=-ry[m],
                correction=corrections_per_tilt[iz],
                # positions_original=positions_from_stitching_original[iz]+prealignmentshifts[iz,None,:]*pixel_size*1e-4,
                # positions_adjusted=positions_from_stitching[iz],
                pixel_size=pixel_size,
                title=f'Tilt {tilts[iz]:.1f}°',
                # scale=10
            )
            fig.savefig(os.path.join(output_dir,f'tilt_{tilts[iz]:02.0f}_corrected_positions.pdf'))
            fig,ax = plt.subplots()
            ax.imshow(montage,origin='upper')
            origin = np.amin(positions_from_stitching[iz], axis=0)/pixel_size*1e4
            print(origin)
            ax.plot(*(positions_from_stitching_original[iz]/pixel_size*1e4+prealignmentshifts[iz,None,:]-origin).T,'bo')
            ax.plot(*(positions_from_stitching[iz]/pixel_size*1e4-origin).T,'ro')
    plt.show(block=True)
    # print(set(z))
    # print(x[:2],y[:2],z[:2],rx[:2],ry[:2])
    # rx = resid[:, -2]
    # ry = resid[:, -1]
    # fig,ax = plt.subplots()
    # ax.quiver(x,y,rx,ry)
    # plt.show(block=True)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # # ax.scatter(x, y, z, c=c, cmap="viridis", s=6, marker=".")
    # ax.quiver(x, y, z, rx, ry, np.zeros_like(rx), length=1.0, normalize=False, linewidth=0.5)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # plt.show()
    # print(residuals.shape,fiducials.shape)
    # c,x,y,z = fiducials.T
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(x, y, z, c=c, cmap="viridis", s=6, marker=".")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # plt.show()

if __name__=='__main__':
    main()
