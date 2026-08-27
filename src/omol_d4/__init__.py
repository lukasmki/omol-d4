"""Liquid water density from the UMA MLIP, with an optional D4 three-body term.

The workflow is three stages, each a module here and each also available as a
script under `scripts/`:

    omol_d4.profiling   stage 0, optional: is an on-the-fly three-body run
                        affordable at this box size and ATM cutoff?
    omol_d4.nvt         stage 1: build a water box and equilibrate it at
                        fixed volume.
    omol_d4.npt         stage 2: let the volume float, and stream the trace.
    omol_d4.plotting    stage 3: the trace as density, volume and temperature
                        against time, one line per run found on disk.
    omol_d4.analysis    the density, its error bar, the compressibility, and
                        the perturbative three-body correction.

Supporting modules: `calculators` (UMA, and the ATM-only D4 term to sum with
it), `integrator` (a Monte Carlo barostat for tasks with no stress tensor),
`boxes` (the starting configuration), `paths` (the filename convention that
keeps runs from overwriting each other) and `constants` (the state point).
"""

from .analysis import (
    ATMCorrection,
    DensityResult,
    analyse_run,
    analyse_volume,
    atm_correction,
    compressibility,
    density,
)
from .boxes import (
    build_water_box,
    check_box_size,
    contact_summary,
    n_molecules,
    report_contacts,
)
from .calculators import (
    PERIODIC_TASKS,
    RunTag,
    SumWithFreeEnergy,
    build_calculator,
    functional_for,
    make_d4_atm,
    parse_tag,
    require_stress,
    suffix,
)
from .integrator import NPTLangevinMonteCarloBarostat
from .npt import make_barostat, resolve_barostat, run_npt
from .nvt import equilibrate, run_nvt
from .paths import equilibrated_input, npt_csvs, stage_paths
from .plotting import label_for, plot_quantity, plot_runs
from .profiling import cutoff_scan, size_sweep, time_calculator, write_rows

__all__ = [
    "ATMCorrection",
    "DensityResult",
    "NPTLangevinMonteCarloBarostat",
    "PERIODIC_TASKS",
    "RunTag",
    "SumWithFreeEnergy",
    "analyse_run",
    "analyse_volume",
    "atm_correction",
    "build_calculator",
    "build_water_box",
    "check_box_size",
    "compressibility",
    "contact_summary",
    "cutoff_scan",
    "density",
    "equilibrate",
    "equilibrated_input",
    "functional_for",
    "label_for",
    "make_barostat",
    "make_d4_atm",
    "n_molecules",
    "npt_csvs",
    "parse_tag",
    "plot_quantity",
    "plot_runs",
    "report_contacts",
    "require_stress",
    "resolve_barostat",
    "run_npt",
    "run_nvt",
    "size_sweep",
    "stage_paths",
    "suffix",
    "time_calculator",
    "write_rows",
]
