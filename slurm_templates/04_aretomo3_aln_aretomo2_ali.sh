#!/bin/bash
#SBATCH --job-name=AT3aln_AT2ali
#SBATCH --output=logs/aretomo_aln_%j.out
#SBATCH --error=logs/aretomo_aln_%j.err
#SBATCH -p gpu-l40s
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#
# Two-step alignment for a montage tilt series:
#   Step 1 — AreTomo3 computes alignment and writes a .aln file (-VolZ 0, no recon).
#   Step 2 — AreTomo2 applies the .aln to produce an aligned tilt stack (-VolZ 0).
#
# The aligned stack is used as input for 05_aretomo2_strip_recon, not this script.
# Reconstruction is done per-strip because the full montage is too large for a
# single GPU.
#
# ── edit these ────────────────────────────────────────────────────────────────
INDIR="EDIT_ME"               # directory containing the input MRC tilt series
STEM="EDIT_ME"                # filename stem, e.g. "Montage_182-A_"  (no extension)
OUTDIR="${INDIR}/AreTomo3aln_AreTomo2ali"
RECON_SCRIPT="EDIT_ME"        # path to 05_aretomo2_strip_recon.sh (for auto-submission)

PIXEL_SIZE=EDIT_ME            # unbinned pixel size in Å
CS=2.7                        # spherical aberration in mm
KV=300                        # accelerating voltage in kV
# ── end of edit section ───────────────────────────────────────────────────────

module load foss/2022a CUDA/11.8.0 UCX-CUDA/1.13.1-CUDA-11.8.0 cuDNN/8.7.0.84-CUDA-11.8.0
module load GCC/11.3.0 OpenMPI/4.1.4
source /programs/sbgrid.shrc

mkdir -p "${OUTDIR}" logs

cp "${INDIR}/tiltAngles.txt" "${INDIR}/${STEM}.rawtlt"

# ── Step 1: AreTomo3 alignment only (no reconstruction) ──────────────────────
AreTomo3 \
    -InPrefix "${INDIR}/${STEM}.mrc" \
    -OutDir   "${OUTDIR}" \
    -TiltCor  1 \
    -AtBin    1 \
    -Cs       ${CS} \
    -Kv       ${KV} \
    -PixSize  ${PIXEL_SIZE} \
    -Gpu      0 \
    -VolZ     0 \
    -Cmd      0

ALN_FILE="${OUTDIR}/${STEM}.aln"
if [[ ! -f "${ALN_FILE}" ]]; then
    echo "ERROR: AreTomo3 did not produce ${ALN_FILE} — aborting." >&2
    exit 1
fi

# ── Step 2: AreTomo2 applies .aln, writes aligned stack (no reconstruction) ──
AreTomo2 \
    -InMrc   "${INDIR}/${STEM}.mrc" \
    -OutMrc  "${OUTDIR}/${STEM}_ali.mrc" \
    -AlnFile "${ALN_FILE}" \
    -AngFile "${INDIR}/${STEM}.rawtlt" \
    -VolZ    0 \
    -TiltCor 0 \
    -AtBin   1 \
    -Cs      ${CS} \
    -Kv      ${KV} \
    -PixSize ${PIXEL_SIZE} \
    -Gpu     0

# ── Submit strip reconstruction as a dependent job ────────────────────────────
sbatch --dependency=afterok:${SLURM_JOB_ID} "${RECON_SCRIPT}"

my-job-stats -a -n -s
