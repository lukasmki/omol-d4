#!/bin/sh
#SBATCH --account=m2834_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --job-name=uma-wb
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --time=08:00:00

cd $SCRATCH/omol-d4/
source venv/bin/activate
cd scripts/

# The stages are thin drivers over the omol-d4 package, so it has to be
# importable in this venv: `uv pip install -e /path/to/omol-dispersion`.

export OMP_NUM_THREADS=64
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Shared stage 1 (no --three-body) + plain-omol NPT baseline.
#
# SIZE: 125 H2O (--n-side 5, L = 15.54 A). The previous 64-molecule box
# (L = 12.43 A) sits below the 14 A SAFE_MIN_EDGE in omol_d4/boxes.py, i.e. it
# does not comfortably exceed twice the model's ~6 A cutoff. n-side 5 is the
# smallest box that clears that line, and it is nearly free to make the jump:
# the density error is set by sigma_rho/rho = sqrt(kB T kappa_T / V), so error^2
# scales as 1/V while cost scales as V * t. The V cancels, and the cost to reach
# a *fixed* precision is flat in box size for UMA (3.2 h at 64 H2O vs 4.1 h at
# 125, from scripts/profile-size.csv). Size therefore buys physics, not error
# bars, so it is chosen on the finite-size criterion alone.
#
# DURATION: 20 ps NVT (40,000 steps) + 65 ps NPT (130,000 steps) at 0.5 fs.
#   Stage 1 has to erase the initial cubic grid, whose spacing is 3.11 A at any
#   box size (the density fixes it). At D ~ 0.23 A^2/ps a molecule needs ~7 ps
#   to diffuse one spacing, so the old 10 ps only just cleared one; 20 ps costs
#   1 GPU-h and removes the doubt.
#   Stage 2 is sized by the measurement it feeds. The volume autocorrelation
#   time in ../data is tau_int = 0.71 ps (UMA) / 1.08 ps (UMA+D4), so the old
#   30 ps runs held only 14 and 9 independent volume samples and the three-body
#   effect came out at 1.06 sigma - unresolved. Resolving the predicted
#   -0.0296 g/cm^3 at 5 sigma needs 0.0042 g/cm^3 per arm, which is 50 ps of
#   production; 15 ps of equilibration in front of that gives 65 ps total.
#
# Projected: 0.97 h stage 1 + 3.16 h stage 2 = ~4.1 h, inside the 8 h above.
python 1-nvt.py --model uma-s-1p2p1 --task omol --n-side 5 --n-steps 40000
python 2-npt.py --model uma-s-1p2p1 --task omol --n-side 5 --barostat mc \
    --n-steps 130000
