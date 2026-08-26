#!/bin/sh
#SBATCH --account=m2834_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --job-name=uma-profile
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --time=00:30:00

cd $SCRATCH/omol-d4/
source venv/bin/activate
cd scripts/

# The stages are thin drivers over the omol-d4 package, so it has to be
# importable in this venv: `uv pip install -e /path/to/omol-dispersion`.

# Must match the production jobs: dftd4 is OpenMP-parallel and the ATM triple
# sum is the whole cost being measured here, so a profile taken at a different
# thread count does not transfer to run-UMA-D4.slurm.
export OMP_NUM_THREADS=64
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Stage 0: answer "can I afford --three-body on the fly?" before submitting the
# 48h run-UMA-D4.slurm. Cheap enough for the debug QOS; if the size sweep grows
# (larger --sizes, or --disp3-cutoff default) switch to --qos=regular above and
# raise --time.
#
# Profile the same model and task the production runs use, or the timings are
# for a different calculator than the one that will actually run.

# 1. Size sweep: UMA and the D4 ATM term timed separately across box sizes, at
#    the 12 A cutoff run-UMA-D4.slurm uses. The MLIP is roughly linear in the
#    atom count and the triple sum is not, so the crossover is what decides the
#    box size. --check-sum verifies that the combined SumCalculator really does
#    cost the sum of the two.
python 0-profile.py --model uma-s-1p2p1 --task omol \
    --sizes 3 4 5 6 --n-calls 5 --disp3-cutoff 12.0 --check-sum \
    --csv profile-size.csv

# 2. Cutoff scan at the production box size (64 H2O, --n-side 4). This one
#    needs no MLIP: it trades D4 cost against how converged the three-body
#    pressure is, and always measures dftd4's own default alongside the
#    explicit cutoffs as the converged reference they approach.
python 0-profile.py --model uma-s-1p2p1 --task omol \
    --cutoff-scan --sizes 4 --cutoffs 6 8 10 12 16 --n-calls 5 \
    --csv profile-cutoff.csv

# Reading the output: the size sweep's 'sum' column x 60,000 steps is the wall
# time of an on-the-fly --three-body NPT run, and the cutoff scan's P_ATM
# column says how much three-body pressure a cheaper cutoff throws away.
