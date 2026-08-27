# Scripts

Command-line drivers for the water-density workflow. All of the actual work
lives in the `omol_d4` package; each script here is a thin argparse front end
over one of its entry points, so anything a script does can also be done from
Python.

| Script | Package entry point | What it does |
| --- | --- | --- |
| `0-profile.py` | `omol_d4.profiling.size_sweep` / `cutoff_scan` | Optional. Times UMA against the D4 three-body term, by box size or by ATM cutoff, so you can decide whether an on-the-fly `--three-body` run is affordable. |
| `0-profile-threads.py` | `omol_d4.profiling.thread_scan` | Optional. Times the D4 three-body term against `OMP_NUM_THREADS`, to decide how many cores that run is worth giving. |
| `1-nvt.py` | `omol_d4.nvt.run_nvt` | Stage 1. Builds a periodic water box at the experimental density and equilibrates it at fixed volume. |
| `2-npt.py` | `omol_d4.npt.run_npt`, `omol_d4.analysis` | Stage 2. Lets the volume float at 298 K / 1 atm, streams the volume trace, and reports the density, its error bar and the compressibility. |
| `3-plot.py` | `omol_d4.plotting.plot_runs` | Stage 3. Plots density, volume and temperature against time for every `npt_volume*.csv` in a directory, one line per run. |

## SLURM

`run-UMA.slurm` and `run-UMA-D4.slurm` submit the two arms of the comparison
(UMA alone, and UMA + D4 three-body) at NERSC. `run-profile.slurm` and
`run-profile-threads.slurm` run stage 0 first, so you can size the three-body
arm before committing 48 hours to it — both are short enough for the debug QOS.
All of them require `omol-d4` to be installed in the venv they activate.

```sh
sbatch run-profile.slurm                              # box size and ATM cutoff
sbatch run-profile-threads.slurm                      # how many cores to ask for
JOB=$(sbatch --parsable run-UMA.slurm)                # stage 1 + baseline NPT
sbatch --dependency=afterok:$JOB run-UMA-D4.slurm     # three-body arm
```

`run-profile.slurm` exports the same `OMP_NUM_THREADS` as the production jobs,
which its timings have to match to transfer. `run-profile-threads.slurm`
deliberately does not: that variable is the one it sweeps, and it sets it per
subprocess instead.

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

# Plot every run in a directory: both arms of the n4 comparison in ../data,
# then whatever finished runs are sitting here
python 3-plot.py --outdir ../data
python 3-plot.py

# One figure, from traces in two different directories
python 3-plot.py --quantity density \
    --csv "../data/npt_volume_uma-s-1p2p1_omol_n4.csv:64 H2O" \
    --csv "npt_volume_uma-s-1p2p1_omol_n5.csv:125 H2O"

# Estimate the three-body effect from an existing UMA-only trajectory
python 2-npt.py --atm-correction --model uma-s-1p2p1 --task omol --n-side 4
```

Stage 1 and stage 2 output files are tagged by model, task, box size and
whether the three-body term was on (`omol_d4.calculators.suffix`), so runs that
are not the same measurement cannot overwrite each other. The default
216-molecule UMA/omc run is untagged. Stage 3 reads that tag back
(`omol_d4.calculators.parse_tag`), which is where its legend labels come from,
so a new run appears in the figures without anything being named in code.

## From Python

```python
from omol_d4 import run_nvt, run_npt, analyse_run, plot_runs

run_nvt(n_side=4, task="omol", model="uma-s-1p2p1")
run_npt(n_side=4, task="omol", model="uma-s-1p2p1", barostat="mc")
print(analyse_run(n_side=4, task="omol", model="uma-s-1p2p1").report())
plot_runs(outdir=".")
```

The plots are also on the installed CLI: `omold4 plot --outdir ../data`.

Both stage functions take a `calc=` override, which is how the stages are
tested without downloading an MLIP checkpoint.
