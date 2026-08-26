# Scripts

Command-line drivers for the water-density workflow. All of the actual work
lives in the `omol_d4` package; each script here is a thin argparse front end
over one of its entry points, so anything a script does can also be done from
Python.

| Script | Package entry point | What it does |
| --- | --- | --- |
| `0-profile.py` | `omol_d4.profiling.size_sweep` / `cutoff_scan` | Optional. Times UMA against the D4 three-body term, by box size or by ATM cutoff, so you can decide whether an on-the-fly `--three-body` run is affordable. |
| `1-nvt.py` | `omol_d4.nvt.run_nvt` | Stage 1. Builds a periodic water box at the experimental density and equilibrates it at fixed volume. |
| `2-npt.py` | `omol_d4.npt.run_npt`, `omol_d4.analysis` | Stage 2. Lets the volume float at 298 K / 1 atm, streams the volume trace, and reports the density, its error bar and the compressibility. |

## SLURM

`run-UMA.slurm` and `run-UMA-D4.slurm` submit the two arms of the comparison
(UMA alone, and UMA + D4 three-body) at NERSC. `run-profile.slurm` runs stage 0
first, so you can size the three-body arm before committing 48 hours to it —
it is short enough for the debug QOS. All three require `omol-d4` to be
installed in the venv they activate, and export the same `OMP_NUM_THREADS`,
which the profile has to match for its timings to transfer.

```sh
sbatch run-profile.slurm                              # decide the box size
JOB=$(sbatch --parsable run-UMA.slurm)                # stage 1 + baseline NPT
sbatch --dependency=afterok:$JOB run-UMA-D4.slurm     # three-body arm
```

## Usage

```sh
# Stage 1, then stage 2, for 216 H2O with UMA/omc
python 1-nvt.py
python 2-npt.py

# The 64-molecule omol runs that produced ../data
python 1-nvt.py --model uma-s-1p2p1 --task omol --n-side 4
python 2-npt.py --model uma-s-1p2p1 --task omol --n-side 4 --barostat mc
python 2-npt.py --model uma-s-1p2p1 --task omol --n-side 4 --barostat mc --three-body

# Re-analyse a finished run without touching a GPU
python 2-npt.py --analyse --model uma-s-1p2p1 --task omol --n-side 4 --outdir ../data

# Estimate the three-body effect from an existing UMA-only trajectory
python 2-npt.py --atm-correction --model uma-s-1p2p1 --task omol --n-side 4
```

Stage 1 and stage 2 output files are tagged by model, task, box size and
whether the three-body term was on (`omol_d4.calculators.suffix`), so runs that
are not the same measurement cannot overwrite each other. The default
216-molecule UMA/omc run is untagged.

## From Python

```python
from omol_d4 import run_nvt, run_npt, analyse_run

run_nvt(n_side=4, task="omol", model="uma-s-1p2p1")
run_npt(n_side=4, task="omol", model="uma-s-1p2p1", barostat="mc")
print(analyse_run(n_side=4, task="omol", model="uma-s-1p2p1").report())
```

Both stage functions take a `calc=` override, which is how the stages are
tested without downloading an MLIP checkpoint.
