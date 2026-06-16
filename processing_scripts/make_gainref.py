from stitch import *


def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(
        description="Stitch square beam montage tomography data."
    )
    parser.add_argument(
        "-i", "--input", help="*.mrc wildcard for raw data", required=True, type=str
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory, if left blank output will be placed in a folder named ./[input_filename]_output",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-t",
        "--templateindex",
        help="Index of input mrc to use as a template, all other images will be aligned to this to create the gain reference. Defaults to 0 (ie. the first image)",
        required=False,
        type=int,
        default=0,
    )
    parser.add_argument(
        "-b",
        "--binning",
        help="Binning of input data, defaults to 1",
        required=False,
        type=int,
        default=1,
    )

    parser.add_argument(
        "-N",
        "--maxnumber",
        help="Max number of frames that will contribute to the gain reference, by default all frames are used.",
        required=False,
        type=int,
    )

    return vars(parser.parse_args())


def main():
    args = parse_commandline()
    outdir = setup_outputdir(args)

    # Binning constant
    binning = int(args["binning"])

    files = glob.glob(args["input"])
    if len(files) < 1:
        raise FileNotFoundError("No files matching {0}".format(args["input"]))

    gainref = make_gain_ref(
        files,
        binning=binning,
        templateindex=args["templateindex"],
        maxnumber=args["maxnumber"],
    )
    gainreffile = os.path.join(outdir, "gainref.mrc")
    print("Saving gain reference to {0}".format(gainreffile))
    savetomrc(gainref.astype(np.float32), gainreffile)


if __name__ == "__main__":
    main()
