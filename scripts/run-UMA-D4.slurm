#!/bin/sh
#SBATCH --account=m2834_g
#SBATCH --constraint=gpu 
#SBATCH --qos=regular
#SBATCH --job-name=umad4-wb
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --time=48:00:00

cd $SCRATCH/uma/
source venv/bin/activate

# The stages are thin drivers over the omol-d4 package, so it has to be
# importable in this venv: `uv pip install -e /path/to/omol-dispersion`.

export OMP_NUM_THREADS=128
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# D4 three-body arm: reuses run-UMA.slurm's stage-1 output (water_equilibrated
# is shared, un-tagged for three-body, per omol_d4/calculators.py's docstring - ATM
# forces are tiny, so equilibrating without them and sampling with them at
# stage 2 is defensible and much cheaper). Submit this AFTER run-UMA.slurm has
# completed (e.g. `sbatch --dependency=afterok:<jobid> run-UMA-D4.slurm`), or
# just run it once run-UMA.slurm's water_equilibrated_n4_omol.traj exists.
#
# --n-side 4 (64 H2O) keeps the on-the-fly D4 ATM cost affordable: profiled at
# ~1.96 s/step (54 ms UMA + 1905 ms D4) for this size, so 60,000 steps is
# ~32.6 h, fitting the 48h limit below. At 216 H2O it would be ~122 h.
#
# NOTE: that 1905 ms was measured when --disp3-cutoff was silently ignored,
# i.e. at dftd4's own 40 Bohr default. The cutoff below is now actually applied
# (omol_d4.calculators.realspace_cutoff), which is roughly 20x cheaper at 12 A
# and reproduces the default's ATM pressure to within a bar - so this arm
# should now finish well inside the limit. Re-run 0-profile.py --cutoff-scan
# to get the current numbers for your box before trusting a wall time.
python 2-npt.py --model uma-s-1p2p1 --task omol --n-side 4 \
    --barostat mc --three-body --disp3-cutoff 12.0
