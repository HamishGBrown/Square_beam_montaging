Code for setting up, executing (in SerialEM) and pre-processing cryo-TEM montages with square and rectangular beams.

# Install

Download Github repo

```git clone https://github.com/HamishGBrown/Square_beam_montaging.git```

Install 

```
cd Square_beam_montaging
pip install -e .
```

# Setting up the coordinates for a montage

```
$ python Generate_Image_shifts_for_Montage.py -h

usage: Generate_Image_shifts_for_Montage.py [-h] -tr TILTAXISROTATION [-ts TILTSTEP] -tm
                                            MAXTILT [-tg DOSESYMMETRICTILTGROUP] -p
                                            PIXELSIZE -n CAMERAPIXELS CAMERAPIXELS -M
                                            MONTAGETILES MONTAGETILES
                                            [-ov MONTAGEOVERLAP MONTAGEOVERLAP]
                                            [-o OUTPUT] [-plt]

Stitch square beam montage tomography data.

optional arguments:
  -h, --help            show this help message and exit
  -tr TILTAXISROTATION, --tiltaxisrotation TILTAXISROTATION
                        Rotation between tilt axis and camera axes (required)
  -ts TILTSTEP, --tiltstep TILTSTEP
                        Step of tomography tilt series in degrees (3 by default)
  -tm MAXTILT, --maxtilt MAXTILT
                        Maximum tilt in tilt series (required)
  -tg DOSESYMMETRICTILTGROUP, --dosesymmetrictiltgroup DOSESYMMETRICTILTGROUP
                        Number of tilts in dose-symmetric tilt group (3 by default)
  -p PIXELSIZE, --pixelsize PIXELSIZE
                        Pixel size of camera in Angstrom (required)
  -n CAMERAPIXELS CAMERAPIXELS, --camerapixels CAMERAPIXELS CAMERAPIXELS
                        Detector size in pixels (required)
  -M MONTAGETILES MONTAGETILES, --montagetiles MONTAGETILES MONTAGETILES
                        Number of tiles in montage (required)
  -ov MONTAGEOVERLAP MONTAGEOVERLAP, --montageoverlap MONTAGEOVERLAP MONTAGEOVERLAP
                        Montage overlap factor (1/overlap fraction), default 10
  -o OUTPUT, --output OUTPUT
                        Output basename (default Imageshifts)
  -plt, --plot          Plot imageshifts (default True)

```

eg. to create a 4 x 3 montage with a K3 (4092 x 5760 pixels) at 50 kx magnfication (1.56 Å pixel size on our instrument) and a dose-symmetric tilt series between -60 and 60 with a step size of 3° and alternatig betwen positive and negative tilts every 3 tilt series. Use the following command:

```
$ python Generate_Image_shifts_for_Montage.py  -tr -96.7 -ts 3.0 -tm 60 -tg 3 -p 1.56 -n 4092 5760 -M 4 3 -o Imageshifts --plot
41 tilts between -60.0 and 60.0 in steps of 3.0
```

Which (optionally) generates the following plot

![image](https://github.com/user-attachments/assets/b4e5e8cc-04e5-46b0-974e-74dc991f787d)

And a text file ```Imageshifts.txt``` which Serial-EM will read, transfer this to the Serial-EM computer

# Running Serial-EM

In low-dose mode create some nav points maps using the view preset and run the ```acquire_montage.txt``` script at these points using the "Acquire at items" feature of Serial-EM, the output for a single tilt will look like this:

![image](https://github.com/HamishGBrown/Square_beam_montaging/blob/main/SingleMontage.gif)

# Stitching montages

Use the python script ```stitch.py``` to stitch images in montage:

```
$ python stitch.py -h

usage: stitch.py [-h] -i INPUT [-o OUTPUT] -I IMAGE_SHIFTS [-g GAINREF] [-b BINNING]
                 [-f FRINGE_SIZE] [-s] [-t TILTAXISROTATION]

Stitch square beam montage tomography data.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        *.mrc wildcard for raw data
  -o OUTPUT, --output OUTPUT
                        Output directory, if left blank output will be placed in a
                        folder named ./[input_filename]_output
  -I IMAGE_SHIFTS, --image_shifts IMAGE_SHIFTS
                        Path to text file containing the tilts and a list of image
                        shifts at every tilt (requried).
  -g GAINREF, --gainref GAINREF
                        Gain reference, if left blank a new gain reference will be
                        calculated and saved in the output directory
  -b BINNING, --binning BINNING
                        Binning of input data, defaults to 1
  -f FRINGE_SIZE, --fringe_size FRINGE_SIZE
                        Size of Fresnel fringes at edge of beam, this will be removed
                        from the gain reference (default 20).
  -s, --skipcrosscorrelation
                        Skip cross-correlation alignment of montage tiles and default to
                        using imageshifts to stitch montage.
  -t TILTAXISROTATION, --tiltaxisrotation TILTAXISROTATION
                        Rotation of tilt axis relative to image, if not provided will
                        take from .mdoc file.
```

For example:

```
$ python stitch.py -i  "/home/hbrown/Mount/KriosFalcon4/Brown/20240826_montaging/Montagingtest1_*.mrc" -b 4 -I Krios_imageshifts.txt  -t -3
```

After this you will need to the ```Crop_to_smallest_common_size.py``` script to join all the montages into a single mrc for IMOD

```
$ python Crop_to_smallest_common_size.py -h
usage: Crop_to_smallest_common_size.py [-h] -i INPUT [-o OUTPUT]

Join Montage tiff files into single mrc for IMOD, filling blank areas in montage.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Directory containing stitched montage tilt series tiff files
  -o OUTPUT, --output OUTPUT
                        Output directory, if not supplied, output will be placed in same
                        directory as input

```

Produces the following output (after rough alignment in IMOD):

![image](https://github.com/HamishGBrown/Square_beam_montaging/blob/main/tilt_series_preali.gif)
