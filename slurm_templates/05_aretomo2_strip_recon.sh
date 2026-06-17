#!/bin/bash
#SBATCH --job-name=AT2_strip_recon
#SBATCH --output=logs/strip_%A_%a.out
#SBATCH --error=logs/strip_%A_%a.err
#SBATCH -p gpu-l40s
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-3           # set to 0-(N_STRIPS-1)
#
# Reconstruct a montage tilt series by splitting the aligned stack into
# horizontal strips and reconstructing each independently.
#
# Why strips?  The full montage is too large for a single GPU.  Because the
# tilt axis is along Y in the aligned stack, each horizontal strip is
# self-contained — no information from adjacent strips is needed.
#
# Input: the aligned tilt stack produced by 04_aretomo3_aln_aretomo2_ali.sh.
# Alignment is already baked into the stack, so each strip gets an identity
# .aln (zero TX/TY/ROT, GMAG=1) with tilt angles copied from the original .aln.
#
# After all strips finish, stitch the sub-tomograms along Y.
#
# ── edit these ────────────────────────────────────────────────────────────────
ALN_DIR="EDIT_ME"             # AreTomo3aln_AreTomo2ali output directory
STEM="EDIT_ME"                # filename stem, e.g. "Montage_182-A_"

N_STRIPS=4                    # must match --array upper bound + 1
VOLZ=1000                     # reconstruction thickness in pixels
ATBIN=1                       # additional binning (1 = no extra binning)
PIXEL_SIZE=EDIT_ME            # pixel size of the aligned stack in Å

CS=2.7
KV=300
# ── end of edit section ───────────────────────────────────────────────────────

INDIR=$(dirname "${ALN_DIR}")
INPREFIX="${ALN_DIR}/${STEM}_ali.mrc"
ALN_FILE="${ALN_DIR}/${STEM}.aln"
RAWTLT="${INDIR}/${STEM}.rawtlt"
STRIPS_ROOT="${ALN_DIR}/strips"

TASK=${SLURM_ARRAY_TASK_ID}
STRIP_DIR="${STRIPS_ROOT}/strip_$(printf '%03d' ${TASK})"
STRIP_MRC="${STRIP_DIR}/strip_$(printf '%03d' ${TASK}).mrc"
STRIP_ALI="${STRIP_DIR}/strip_$(printf '%03d' ${TASK})_ali.mrc"
STRIP_ALN="${STRIP_DIR}/strip_$(printf '%03d' ${TASK}).aln"
STRIP_RAWTLT="${STRIP_DIR}/strip_$(printf '%03d' ${TASK}).rawtlt"

mkdir -p "${STRIP_DIR}" logs

module load foss/2022a CUDA/11.8.0 UCX-CUDA/1.13.1-CUDA-11.8.0 cuDNN/8.7.0.84-CUDA-11.8.0
module load GCC/11.3.0 OpenMPI/4.1.4
source /programs/sbgrid.shrc

# ── Cut strip from aligned stack and write identity .aln ─────────────────────
python3 << PYEOF
import mrcfile, numpy as np, re

task      = int("${TASK}")
n_strips  = int("${N_STRIPS}")
in_mrc    = "${INPREFIX}"
aln_src   = "${ALN_FILE}"
strip_mrc = "${STRIP_MRC}"
strip_aln = "${STRIP_ALN}"
pix_size  = float("${PIXEL_SIZE}")

with mrcfile.mmap(in_mrc, mode='r', permissive=True) as m:
    n_tilts, ny, nx = m.data.shape

strip_h  = ny // n_strips
y_start  = task * strip_h
y_end    = (task + 1) * strip_h if task < n_strips - 1 else ny
strip_ny = y_end - y_start

print(f"Strip {task}: Y=[{y_start}:{y_end}]", flush=True)

with mrcfile.mmap(in_mrc, mode='r', permissive=True) as m:
    strip_data = np.asarray(m.data[:, y_start:y_end, :])

with mrcfile.new(strip_mrc, overwrite=True) as m:
    m.set_data(strip_data)
    m.voxel_size = pix_size

print(f"Saved strip MRC: {strip_mrc}  shape={strip_data.shape}", flush=True)

# Identity .aln: zero all transforms, keep tilt angles only.
# The aligned stack already has all transforms applied.
with open(aln_src) as fh:
    lines = fh.readlines()

out = []
data_re = re.compile(r'^\s*\d+\s+[+-]?\d')
for line in lines:
    if '# RawSize' in line:
        out.append(f'# RawSize = {nx} {strip_ny} {n_tilts}\n')
    elif data_re.match(line):
        p = line.split()
        if len(p) >= 10:
            sec  = int(p[0])
            tilt = float(p[9])
            out.append(
                f'{sec:5d}{0.0:12.4f}{1.0:11.5f}{0.0:11.3f}{0.0:11.3f}'
                f'{0.0:9.2f}{0.0:9.2f}{1.0:9.2f}{0.0:9.2f}{tilt:10.2f}\n'
            )
        else:
            out.append(line)
    else:
        out.append(line)

with open(strip_aln, 'w') as fh:
    fh.writelines(out)

print(f"Saved identity .aln: {strip_aln}", flush=True)
PYEOF

cp "${RAWTLT}" "${STRIP_RAWTLT}"

# ── Reconstruct strip (SIRT back-projection, no re-alignment) ─────────────────
AreTomo2 \
    -InMrc   "${STRIP_MRC}" \
    -OutMrc  "${STRIP_ALI}" \
    -AlnFile "${STRIP_ALN}" \
    -AngFile "${STRIP_RAWTLT}" \
    -VolZ    ${VOLZ} \
    -TiltCor 0 \
    -AtBin   ${ATBIN} \
    -Cs      ${CS} \
    -Kv      ${KV} \
    -PixSize ${PIXEL_SIZE} \
    -Gpu     0

my-job-stats -a -n -s
