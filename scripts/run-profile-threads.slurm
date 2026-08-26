#!/bin/sh
#SBATCH --account=m2834_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --job-name=uma-threads
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --time=00:30:00

cd $SCRATCH/omol-d4/
source venv/bin/activate

cd scripts/

# The stages are thin drivers over the omol-d4 package, so it has to be
# importable in this venv: `uv pip install -e /path/to/omol-dispersion`.

# NOTE: deliberately no OMP_NUM_THREADS here - it is the variable being swept,
# and 0-profile-threads.py sets it per subprocess. The placement variables are
# exported because they are inherited by those subprocesses, and a scan taken
# without the binding the production job uses does not describe that job.
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# --constraint=gpu even though no MLIP is loaded: the answer wanted is how the
# ATM sum scales on the CPUs of the node run-UMA-D4.slurm will actually get.

# Powers of two up to a full node, on the 64 H2O production box at the 12 A
# cutoff run-UMA-D4.slurm uses. The knee of the resulting curve - not the core
# count - is what OMP_NUM_THREADS should be set to there.
python 0-profile-threads.py --task omol \
    --threads 1 2 4 8 16 32 64 128 \
    --n-side 4 --disp3-cutoff 12.0 --n-calls 5 \
    --csv profile-threads.csv

# The same scan at dftd4's default cutoff, which is ~20x more work per call and
# so keeps scaling further before it saturates. Worth having if you are
# considering running the three-body arm uncut. It starts at 8 threads rather
# than 1 - a single-threaded call at this cutoff is a couple of minutes on its
# own - so its speedups are relative to 8 threads, not to serial.
python 0-profile-threads.py --task omol \
    --threads 8 16 32 64 128 \
    --n-side 4 --disp3-cutoff default --n-calls 2 \
    --csv profile-threads-default-cutoff.csv
