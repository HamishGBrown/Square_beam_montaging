#!/bin/bash
#SBATCH --job-name=mc2_montage
#SBATCH --output=logs/mc2_%A_%a.out
#SBATCH --error=logs/mc2_%A_%a.err
#SBATCH -p gpu-l40s
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-EDIT_ME    # set upper bound to (number of TIFFs - 1)
#
# Motion-correct one multi-frame TIFF per array task.
# Run beam_mask_motioncorr --print-slurm to auto-generate this script
# with the array size and paths filled in for your dataset.
#
# ── edit these ────────────────────────────────────────────────────────────────
FRAMES_DIR="EDIT_ME"          # directory containing raw multi-frame TIFFs
OUTPUT_DIR="EDIT_ME"          # where to write motion-corrected MRCs
PIXEL_SIZE=EDIT_ME            # pixel size in Å

MOTIONCOR2=/programs/x86_64-linux/motioncor2/1.6.4/MotionCor2_1.6.4_Cuda121_Mar312023
# ── end of edit section ───────────────────────────────────────────────────────

source /programs/sbgrid.shrc
module load CUDA/12.2.0

mkdir -p "${OUTPUT_DIR}/logs"

mapfile -t TIFFS < <(ls "${FRAMES_DIR}"/*.tif | sort)
INPUT="${TIFFS[$SLURM_ARRAY_TASK_ID]}"

beam_mask_motioncorr \
    --input         "${INPUT}" \
    --output-dir    "${OUTPUT_DIR}" \
    --pixel-size    ${PIXEL_SIZE} \
    --motioncor2    ${MOTIONCOR2} \
    --save-diagnostic \
    --gpu 0

my-job-stats -a -n -s
