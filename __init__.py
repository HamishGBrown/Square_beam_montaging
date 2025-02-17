"""
Square Beam Montaging Package

This package provides tools for stitching square beam montage tomography data.
"""

from .stitch import (
    parse_commandline,
    make_gain_ref,
    stitch,
    montage,
    plot_overlaps,
    array_overlap,
    find_overlaps,
    cross_correlate_tiles,
    plot_individual_cross_correlation,
    parse_image_shifts,
    savetomrc,
    generate_image_file_names_from_template,
    parse_mdoc,
    extract_tilt_axis_angle,
    setup_outputdir,
    setup_gainreference,
    plot_positions,
    main,
)
from .Setup_square_montage import dose_symmetric_tilts, generate_montage_shifts, write_tilts_and_image_shifts_to_file, calculate_defocii
from .make_gainref import main as make_gainref_main
from .Generate_Image_shifts_for_Montage import main as generate_image_shifts_main
from .Determine_overlap_fraction import main as determine_overlap_fraction_main
from .Crop_to_smallest_common_size import main as crop_to_smallest_common_size_main
from .Choose_fringe_size import main as choose_fringe_size_main
from .Utilities import *
