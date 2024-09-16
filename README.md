Code for setting up, executing (in SerialEM) and pre-processing cryo-TEM montages with square and rectangular beams.

![image](https://github.com/HamishGBrown/Square_beam_montaging/blob/main/SingleMontage.gif)

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

```
$ python stitch.py -i  "/home/hbrown/Mount/KriosFalcon4/Brown/20240826_montaging/Montagingtest1_*.mrc" -b 4 -I Krios_imageshifts.txt  -t -3
```

Produces the following output:

![image](https://github.com/HamishGBrown/Square_beam_montaging/blob/main/tilt_series_preali-1.gif)
