#!/bin/bash
#SBATCH --job-name=montage2relion
#SBATCH --output=/home/hgbrown/logs/montage2relion_%A.out
#SBATCH --error=/home/hgbrown/logs/montage2relion_%A.out
#SBATCH -p sapphire
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# Turn 3D picks made in a montage tomogram into a RELION-5 tomogram +
# particle set, one "tilt image" per (tilt, tile).
#
# STAGE 1 validates the coordinate chain and STAGE 2 exports it. Run stage 1
# first and LOOK AT THE PNGs. The exporter will happily write a well-formed
# star file full of noise if a sign convention is wrong.
#
# ── edit these ────────────────────────────────────────────────────────────────
BASE=/home/hgbrown/20240917_Montage_stitching/20260713Yeastattempt2

PICKS="${BASE}/AreTomo_motioncorr/Ribosomes.txt"
TOMOGRAM="${BASE}/AreTomo_motioncorr/recon_patch53mask_bin2.mrc"
ALN="${BASE}/AreTomo_motioncorr/Montage_9-A_inpainted.aln"
POSITIONS_DIR="${BASE}/stitched_motioncorr"
TILE_DIR="${BASE}/motion_corrected"
CTF_RESULTS="${TILE_DIR}/ctf_results.txt"
CTF_BASE="Montage_9-A_9-A"

ALIGNED_STACK="${BASE}/AreTomo_motioncorr/recon_patch53mask_bin2tiltseries.mrc"
CANVAS_STACK="${POSITIONS_DIR}/Montage_9-A.mrc"

OVERLAY_DIR="${BASE}/relion/overlays"
OUT_DIR="${BASE}/relion/tomo"

BOX=256          # extraction box, unbinned tile pixels (256 x 3.426 A = 877 A)
DOSE_PER_TILT=0  # e-/A^2 per exposure. 0 disables dose weighting -- set it!
STAGE="${STAGE:-1}"
# ── end of edit section ───────────────────────────────────────────────────────

# NOTE: the `Stitch` conda env is currently broken -- ~/.conda/envs/Stitch has
# only conda-meta/ and etc/, no bin/, so `conda activate Stitch` yields no
# python. That breaks 02_stitch.sh and 03_inpaint_apply.sh too and wants
# rebuilding. Until then run out of the repo with the base Anaconda3, which has
# numpy/scipy/h5py/mrcfile/matplotlib and is all these two tools need.
module load Anaconda3
REPO=/home/hgbrown/gitprojects/Square_beam_montaging
# Take the interpreter from the module root, not from PATH. `python` on PATH is
# whatever env happens to be first -- the IsoNet2 build env here, which has no
# h5py -- and stage 3 sources sbgrid for point2model, which puts python 2.7.2
# in front of everything and turns any annotated signature into a SyntaxError.
PY="${EBROOTANACONDA3}/bin/python"
cd "${REPO}"

set -eo pipefail
export MPLBACKEND=Agg
unset DISPLAY

mkdir -p "${OVERLAY_DIR}" "${OUT_DIR}"

if [ "${STAGE}" = "1" ]; then
    # Stage 1: does the chain land the picks on density?
    #
    # Three views, each isolating one link: `aligned` tests only the
    # tomogram->aligned shear, `canvas` adds the inverse alignment (and settles
    # the 180 deg that footprint-based rotation detection cannot), `patches`
    # tests the whole thing by cutting boxes out of the raw tiles.
    #
    # The number to read off is the mean patch's |peak|/rms. Below ~2 the
    # geometry is wrong (or the picks are not real) and stage 2 is pointless.
    # If it reports a residual offset, feed that back as --extra-shift.
    ${PY} -m processing_scripts.overlay_picks_on_montage \
        --picks          "${PICKS}" \
        --tomogram       "${TOMOGRAM}" \
        --aln            "${ALN}" \
        --positions-dir  "${POSITIONS_DIR}" \
        --tile-dir       "${TILE_DIR}" \
        --aligned-stack  "${ALIGNED_STACK}" \
        --canvas-stack   "${CANVAS_STACK}" \
        --mode all \
        --box  ${BOX} \
        -o     "${OVERLAY_DIR}"
elif [ "${STAGE}" = "3" ]; then
    # Stage 3: IMOD models, to check the chain against the images in 3dmod.
    #
    #   picks_tiltseries_canvas.mod  one contour per particle, one point per
    #       tilt, on the stitched montage. Select a contour and step through Z:
    #       the point should stay on the same feature. Drift that grows with
    #       |tilt| is the Z/shear term, a constant offset is TX/TY.
    #   picks_tiles_<tilt>.mod       the raw per-tilt stack, where each Z
    #       section is a TILE. Green = the tile each particle was assigned to,
    #       red = boxes rejected for crossing the square-beam edge (they should
    #       form a band around each tile's illuminated square).
    #
    # Model Z is 0-based, matching the image index; 3dmod's slider counts from 1.
    # sbgrid.shrc ends with `let SB_RUNTIME=endtime-starttime`, and `let`
    # returns status 1 when the expression evaluates to 0. Sourcing it in under
    # a second therefore makes `source` return 1, and under `set -e` the job
    # dies right there with an empty log. It is a race against the clock, so it
    # fails intermittently -- job 28842562 passed, 28843292 did not, same script.
    set +e
    source /programs/sbgrid.shrc   # for point2model
    set -e
    ${PY} -m processing_scripts.picks_to_imod_model \
        --picks             "${PICKS}" \
        --tomogram          "${TOMOGRAM}" \
        --aln               "${ALN}" \
        --positions-dir     "${POSITIONS_DIR}" \
        --tile-dir          "${TILE_DIR}" \
        --frame             canvas \
        --tiltseries-image  "${CANVAS_STACK}" \
        --box               ${BOX} \
        --mode both \
        -o "${OVERLAY_DIR}"
else
    # Stage 2: write the star files.
    #
    # --extra-shift comes from stage 1. --handedness flips the sign of the
    # along-beam (defocus) axis; the log reports which sign the CTF fits prefer,
    # but if they cannot call it, refine both ways and keep the better.
    ${PY} -m processing_scripts.montage_to_relion \
        --picks          "${PICKS}" \
        --tomogram       "${TOMOGRAM}" \
        --aln            "${ALN}" \
        --positions-dir  "${POSITIONS_DIR}" \
        --tile-dir       "${TILE_DIR}" \
        --ctf-results    "${CTF_RESULTS}" \
        --ctf-base       "${CTF_BASE}" \
        --box            ${BOX} \
        --dose-per-tilt  ${DOSE_PER_TILT} \
        --extra-shift    0 0 \
        --handedness     1 \
        -o "${OUT_DIR}"

    echo
    echo "Then, with RELION 5 on PATH (source /programs/sbgrid.shrc):"
    echo "  relion_tomo_subtomo --i ${OUT_DIR}/optimisation_set.star \\"
    echo "      --b ${BOX} --crop $((BOX/2)) --bin 2 --j 8 --o Subtomograms/"
fi

my-job-stats -a -n -s
