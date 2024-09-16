import mrcfile
from tqdm import tqdm
import glob
import numpy as np
import re
import os
from PIL import Image
import argparse

fnams = ['Montage_{0:.1f}.tiff'.format(x) for x in np.arange(-48,49,3)]


def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(
        description="Join Montage tiff files into single mrc for IMOD, filling blank areas in montage."
    )
    parser.add_argument(
        "-i", "--input", help="Directory containing stitched montage tilt series tiff files", required=True, type=str
    )

    parser.add_argument(
        "-o", "--output", help="Output directory, if not supplied, output will be placed in same directory as input", required=False, type=str
    )

    return vars(parser.parse_args())


def extract_number(filename):
    match = re.search(r'(-?\d+)\.tif', filename)
    return float(match.group(1)) if match else 0

if __name__=='__main__':

    args = parse_commandline()
    inputdir = args['input']
    if args['output'] is None:
        outputdir = args['input']
    else:
        outputdir = args['output']

    fnams = glob.glob(os.path.join(inputdir,'*.tiff'))

    # Sort filenames in order of increasing tilt
    fnams = sorted(fnams,key=extract_number)
    print(fnams)

    maxsize = Image.open(fnams[0]).size

    for fnam in fnams[1:]:
        maxsize= [max(x,xx) for x,xx in zip(Image.open(fnam).size,maxsize)]

    cropped_names = [fnam.replace('.tiff','_cropped.mrc') for fnam in fnams]

    # for fnam in fnams:
    for i,fnam in enumerate(fnams):
        mrcname = fnam.replace('.tiff','.mrc')
        command = 'tif2mrc {0} {1}'.format(fnam,mrcname)
        print(command)
        os.system(command)
        command = 'clip resize -ox {0} -oy {1} -p 0 {2} {3}'.format(*maxsize,mrcname,cropped_names[i])
        print(command)
        os.system(command)
        print('Removing {0}'.format(mrcname))
        os.system('rm {0}'.format(mrcname))
    
    # Join all in stack
    command = 'newstack {0} {1}'.format(' '.join(cropped_names),'Montage_stack.mrc')
    # sys.exit()

    for i,fnam in enumerate(tqdm(cropped_names,desc='filling black areas')):
        with mrcfile.open(fnam,'r+') as m:
            data = np.asarray(m.data)
            fill = np.median(data[data>2])
            data[data<2] = fill
            m.data[:] = data[:]
        m.close()

    

    # Save tilts in rawtlt.tlt file for IMOD
    tilts = [extract_number(fnam) for fnam in fnams]
    tilt_file = os.path.join(outputdir,'tilt_series.rawtlt')
    with open(tilt_file, 'w') as file:
        for tilt in tilts:
            file.write(f"{tilt}\n")

    outputfile = os.path.join(outputdir,'tilt_series.mrc')
    command = 'newstack -tilt {0} {1} {2} '.format(tilt_file,' '.join(cropped_names),outputfile )
    print(command)
    os.system(command)

    # Delete intermediates
    for mrcname in cropped_names:
        print('Removing {0} and {0}~'.format(mrcname))
        os.system('rm {0}'.format(mrcname))
        os.system('rm {0}~'.format(mrcname))