#!/bin/bash
#SBATCH --job-name=stitch_montage
#SBATCH --output=logs/stitch_%j.out
#SBATCH --error=logs/stitch_%j.err
#SBATCH -p cpu
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#
# Stitch per-tile motion-corrected MRCs into a single tilt series MRC stack.
# Runs stitch_square_beam followed by crop_to_smallest_common_size.
#
# ── edit these ────────────────────────────────────────────────────────────────
MOTIONCORR_DIR="EDIT_ME"      # directory of per-tile MRCs from 01_motion_correction
IMAGE_SHIFTS="EDIT_ME"        # Imageshifts.txt from generate_image_shifts
STITCHED_DIR="EDIT_ME"        # output directory for per-tilt stitched images
STACK_DIR="EDIT_ME"           # output directory for the final joined MRC stack
N_THREADS=16                  # match --cpus-per-task above
# ── end of edit section ───────────────────────────────────────────────────────

source /programs/sbgrid.shrc

mkdir -p logs

stitch_square_beam \
    -i "${MOTIONCORR_DIR}" \
    -I "${IMAGE_SHIFTS}" \
    -o "${STITCHED_DIR}" \
    --mark-uncovered \
    -nt ${N_THREADS}

crop_to_smallest_common_size \
    -i "${STITCHED_DIR}" \
    -o "${STACK_DIR}"

my-job-stats -a -n -s
