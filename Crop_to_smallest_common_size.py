from tqdm import tqdm
import numpy as np
import mrcfile
import re
import sys
import os
from PIL import Image

fnams = ['Montage_{0:.1f}.tiff'.format(x) for x in np.arange(-48,49,3)]

def extract_number(filename):
    match = re.search(r'(-?\d+\.?\d*)', filename)
    return float(match.group(0)) if match else 0

if __name__=='__main__':
    fnams = [os.path.join('./rough_montages',x) for x in fnams]

    minsize = Image.open(fnams[0]).size

    for fnam in fnams[1:]:
        minsize= [max(x,xx) for x,xx in zip(Image.open(fnam).size,minsize)]
    misize = [x + x//2 for x in minsize]

    cropped_names = [fnam.replace('.tiff','_cropped.mrc') for fnam in fnams]

    # for fnam in fnams:
    for i,fnam in enumerate(fnams):
        command = 'tif2mrc {0} {1}'.format(fnam,fnam.replace('.tiff','.mrc'))
        print(command)
        os.system(command)
        command = 'clip resize -ox {0} -oy {1} -p 0 {2} {3}'.format(*minsize,fnam.replace('.tiff','.mrc'),cropped_names[i])
        print(command)
        os.system(command)
    
    command = 'newstack {0} {1}'.format(' '.join(cropped_names),'Montage_stack.mrc')
    # sys.exit()

    win = [870,410,1100,1100]
    for i,fnam in enumerate(tqdm(cropped_names,desc='filling black areas')):
        with mrcfile.open(fnam,'r+') as m:
            data = np.asarray(m.data)
            fill = np.mean(data[win[0]:win[0]+win[2],win[1]:win[1]+win[3]])
            data[data<2] = fill
            m.data[:] = data[:]
        m.close()

    

    tilts = [extract_number(fnam) for fnam in fnams]
    tilt_file = './rough_montages/tilt_series.tlt'
    with open(tilt_file, 'w') as file:
        for tilt in tilts:
            file.write(f"{tilt}\n")

    command = 'newstack -tilt {0} {1} ./rough_montages/tilt_series.mrc'.format(tilt_file,' '.join(cropped_names) )
    # command = 'tif2mrc {0} ./stitched/tilt_series.mrc'.format(' '.join(cropped_names[:1]) )
    print(command)
    os.system(command)