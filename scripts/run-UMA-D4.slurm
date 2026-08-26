#!/bin/sh
#SBATCH --account=m2834_g
#SBATCH --constraint=gpu 
#SBATCH --qos=regular
#SBATCH --job-name=umad4-wb
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --time=30:00:00

cd $SCRATCH/omol-d4/
source venv/bin/activate
cd scripts/

# The stages are thin drivers over the omol-d4 package, so it has to be
# importable in this venv: `uv pip install -e /path/to/omol-dispersion`.

export OMP_NUM_THREADS=64
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# D4 three-body arm: reuses run-UMA.slurm's stage-1 output (water_equilibrated
# is shared, un-tagged for three-body, per omol_d4/calculators.py's docstring - ATM
# forces are tiny, so equilibrating without them and sampling with them at
# stage 2 is defensible and much cheaper). Submit this AFTER run-UMA.slurm has
# completed (e.g. `sbatch --dependency=afterok:<jobid> run-UMA-D4.slurm`), or
# just run it once run-UMA.slurm's water_equilibrated_uma-s-1p2p1_omol_n5.traj
# exists (omol_d4.paths.equilibrated_input falls back to it automatically).
#
# SIZE: must match run-UMA.slurm exactly - the measurement is the *difference*
# between the two arms, so any box-size mismatch would contaminate it with a
# finite-size effect larger than the three-body signal. See that script for why
# 125 H2O (L = 15.54 A) rather than the previous 64 (L = 12.43 A, below the
# 14 A SAFE_MIN_EDGE in omol_d4/boxes.py).
#
# This arm is what the size choice actually costs. Unlike UMA, whose cost to a
# fixed precision is flat in box size, the ATM triple sum scales ~N^1.3, so the
# 64 -> 125 move is a genuine 12.3 h -> 22.5 h premium (scripts/profile-size.csv).
# 216 H2O would be 45.6 h; that is the size this arm rules out, not 125.
#
# DURATION: 65 ps (130,000 steps at 0.5 fs) = 15 ps equilibration + 50 ps
# production, matching run-UMA.slurm. The volume autocorrelation time is longer
# on this arm (tau_int = 1.08 ps vs 0.71 ps), and the old 30 ps run left the
# three-body effect at 1.06 sigma - i.e. it could not resolve the thing it was
# run to measure. 50 ps of production reaches 0.0042 g/cm^3 per arm, enough for
# 5 sigma on the -0.0296 g/cm^3 predicted by the perturbative kappa_T * P_ATM
# estimate. This arm also needs the longer 15 ps discard: its volume was still
# drifting at 10 ps (-3.0 A^3/ps) and only settled by ~12 ps.
#
# The 12 A cutoff stays: it reproduces the library default's ATM pressure to
# 1.3% (619 vs 627 bar) at 1/12 the cost (157 vs 1949 ms), per profile-cutoff.csv.
#
# Projected: 130,000 x (87 ms UMA + 390 ms D4) = ~17.2 h, inside the 30 h above.
# That assumes the two calculators simply add; re-run 0-profile.py with
# --check-sum at this box size if you want that verified before committing.
python 2-npt.py --model uma-s-1p2p1 --task omol --n-side 5 \
    --barostat mc --three-body --disp3-cutoff 12.0 --n-steps 130000
