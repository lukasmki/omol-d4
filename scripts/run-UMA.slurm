#!/bin/sh
#SBATCH --account=m2834_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --job-name=uma-wb
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --time=24:00:00

cd $SCRATCH/uma/
source venv/bin/activate

# The stages are thin drivers over the omol-d4 package, so it has to be
# importable in this venv: `uv pip install -e /path/to/omol-dispersion`.

export OMP_NUM_THREADS=128
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Shared stage 1 (no --three-body) + plain-omol NPT baseline.
# 64 H2O (--n-side 4) rather than the historical 216: a deliberate cost/size
# tradeoff so run-UMA-D4.slurm's on-the-fly D4 three-body NPT fits in its
# time limit (see omol_d4/calculators.py and the omol_d4/boxes.py box-size
# warning for the caveats).
python 1-nvt.py --model uma-s-1p2p1 --task omol --n-side 4
python 2-npt.py --model uma-s-1p2p1 --task omol --n-side 4 --barostat mc
