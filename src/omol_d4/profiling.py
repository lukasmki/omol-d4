"""Stage 0 (optional): profile the cost of UMA against UMA + D4 three-body.

The point is to decide whether an on-the-fly three-body run is affordable before
starting one. Two things dominate the answer and both are measured here:

  * How the two scale with system size. The MLIP is roughly linear in the atom
    count; the ATM triple sum is not, so the crossover matters more than any
    single-size timing.
  * The ATM real-space cutoff, which trades cost against how converged the
    three-body pressure is.
"""

import os
import time

import numpy as np
from ase.units import Bohr

from .boxes import build_water_box, n_molecules
from .calculators import (
    DISP3_CUTOFF,
    MODEL,
    TASK,
    build_calculator,
    functional_for,
    make_d4_atm,
)
from .constants import EV_PER_A3_TO_BAR, SEED

TIMESTEP_FS = 0.5      # only used to project MD throughput
JITTER = 0.005         # A, per call, to defeat the calculators' result caches

# dftd4's own default three-body cutoff, for the "images" column of a scan that
# includes the library default alongside explicit cutoffs.
DEFAULT_DISP3_BOHR = 40.0


def cuda_sync(device):
    """Block until the GPU is idle.

    Without this every UMA timing is nonsense: the CUDA kernels are queued
    asynchronously, so `get_forces` returns long before the work is done and
    the first synchronising call further down inherits the whole backlog.
    """
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def time_calculator(atoms, calc, n_calls, device, warmup=2, rng=None,
                    want_stress=True):
    """Per-call wall time for one force (+ stress) evaluation, in seconds.

    The first calls are discarded: they carry model loading, CUDA context setup,
    kernel autotuning and, for D4, construction of the dispersion model object,
    none of which recur during an MD run.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    atoms.calc = calc
    # A non-periodic task (omol) has no stress; profile forces only there.
    want_stress = want_stress and "stress" in getattr(
        calc, "implemented_properties", ["stress"]
    )
    times = []

    for i in range(warmup + n_calls):
        # Nudge the positions so the calculator cannot serve a cached result.
        atoms.positions += rng.normal(scale=JITTER, size=atoms.positions.shape)

        cuda_sync(device)
        start = time.perf_counter()
        atoms.get_forces()
        if want_stress:
            atoms.get_stress()   # same calculate() call, so this is a dict hit
        cuda_sync(device)
        elapsed = time.perf_counter() - start

        if i >= warmup:
            times.append(elapsed)

    return np.array(times)


def describe(times):
    """Mean +/- std in ms, for one column of the profile table."""
    return (f"{times.mean() * 1e3:9.1f} +/- {times.std() * 1e3:6.1f}"
            if len(times) > 1 else f"{times.mean() * 1e3:9.1f}")


def throughput(seconds_per_call, timestep_fs=TIMESTEP_FS):
    """ps of MD per hour of wall clock, at `timestep_fs`."""
    return 3600.0 / seconds_per_call * timestep_fs / 1000.0


def environment_note(device="cuda", model=MODEL, task=TASK):
    """Print the things that silently invalidate a timing comparison."""
    threads = os.environ.get("OMP_NUM_THREADS", "unset")
    print(f"  device={device}  model={model}  task={task}")
    print(f"  OMP_NUM_THREADS={threads} (cores visible: {os.cpu_count()})")
    if threads == "unset":
        print("  note: dftd4 is OpenMP-parallel and the MLIP is not competing "
              "for those cores; set OMP_NUM_THREADS to profile realistically.")
    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            print(f"  gpu={torch.cuda.get_device_name(0)}")
    except ImportError:
        pass


def _probe_box(n_side):
    """A reproducible box of the given size, tagged for the omol task."""
    atoms = build_water_box(n_side=n_side, rng=np.random.default_rng(0))
    atoms.info.update({"charge": 0, "spin": 1})
    return atoms


def size_sweep(sizes=(3, 4, 5, 6), n_calls=5, device="cuda", model=MODEL,
               task=TASK, functional=None, disp3_cutoff=DISP3_CUTOFF,
               check_sum=False, seed=SEED):
    """Time UMA and the ATM term separately across box sizes. Returns rows."""
    # Pass a representative box so build_calculator can warn about running a
    # non-periodic task (omol) on a periodic cell.
    probe = _probe_box(sizes[0])
    uma = build_calculator(device=device, seed=seed, three_body=False,
                           model=model, task=task, atoms=probe)
    d4 = make_d4_atm(functional or functional_for(task),
                     disp3_cutoff=disp3_cutoff)

    print(f"\n{'n_H2O':>6} {'atoms':>6} {'L/A':>6} "
          f"{'UMA/ms':>18} {'D4/ms':>18} {'sum/ms':>10} {'D4 share':>9} "
          f"{'ps/hour':>9}")
    print("-" * 88)

    rows = []
    for n_side in sizes:
        atoms = _probe_box(n_side)
        length = atoms.cell.lengths()[0]

        t_uma = time_calculator(atoms.copy(), uma, n_calls, device,
                                rng=np.random.default_rng(1))
        t_d4 = time_calculator(atoms.copy(), d4, n_calls, device,
                               rng=np.random.default_rng(1))
        total = t_uma.mean() + t_d4.mean()

        print(f"{n_molecules(atoms):6d} {len(atoms):6d} {length:6.2f} "
              f"{describe(t_uma):>18} {describe(t_d4):>18} "
              f"{total * 1e3:10.1f} {t_d4.mean() / total * 100:8.1f}% "
              f"{throughput(total):9.2f}")

        rows.append({
            "mode": "size", "n_mol": n_molecules(atoms), "n_atoms": len(atoms),
            "cell_A": length, "disp3_cutoff_A": disp3_cutoff,
            "uma_ms": t_uma.mean() * 1e3, "uma_std_ms": t_uma.std() * 1e3,
            "d4_ms": t_d4.mean() * 1e3, "d4_std_ms": t_d4.std() * 1e3,
        })

    # Sanity check that summing the two really does cost the sum of the two.
    if check_sum:
        atoms = _probe_box(sizes[-1])
        both = build_calculator(device=device, seed=seed, three_body=True,
                                model=model, task=task, functional=functional,
                                disp3_cutoff=disp3_cutoff, atoms=atoms)
        t_both = time_calculator(atoms, both, n_calls, device,
                                 rng=np.random.default_rng(1))
        print(f"\n  SumCalculator at n_side={sizes[-1]}: "
              f"{describe(t_both)} ms measured, "
              f"{rows[-1]['uma_ms'] + rows[-1]['d4_ms']:.1f} ms expected")

    return rows


def cutoff_scan(cutoffs=(6.0, 8.0, 10.0, 12.0, 16.0), n_side=6, n_calls=5,
                device="cuda", task=TASK, functional=None,
                include_default=True):
    """Scan the ATM real-space cutoff at one box size. Returns rows.

    `None` is appended to `cutoffs` when `include_default`, which measures
    dftd4's own defaults as the converged reference the others approach.
    """
    functional = functional or functional_for(task)
    atoms = _probe_box(n_side)
    length = atoms.cell.lengths()[0]
    print(f"\nD4 three-body cutoff scan ({functional} damping), "
          f"{n_molecules(atoms)} H2O, L = {length:.2f} A")
    print(f"\n{'disp3/A':>8} {'D4/ms':>18} {'P_ATM/bar':>11} {'images':>8}")
    print("-" * 50)

    rows = []
    for cutoff in list(cutoffs) + ([None] if include_default else []):
        d4 = make_d4_atm(functional, disp3_cutoff=cutoff, quiet=True)
        probe = atoms.copy()
        probe.calc = d4
        pressure = -probe.get_stress()[:3].mean() * EV_PER_A3_TO_BAR
        times = time_calculator(atoms.copy(), d4, n_calls, device,
                                rng=np.random.default_rng(1))
        span = DEFAULT_DISP3_BOHR * Bohr if cutoff is None else cutoff
        print(f"{'default' if cutoff is None else f'{cutoff:.1f}':>8} "
              f"{describe(times):>18} {pressure:11.1f} "
              f"{(2 * span / length + 1) ** 3:8.1f}")
        rows.append({
            "mode": "cutoff", "n_mol": n_molecules(atoms),
            "n_atoms": len(atoms), "cell_A": length,
            "disp3_cutoff_A": "default" if cutoff is None else cutoff,
            "d4_ms": times.mean() * 1e3, "d4_std_ms": times.std() * 1e3,
            "p_atm_bar": pressure,
        })

    print("\n  'images' is the rough number of periodic cells the triple sum "
          "spans; the ATM cost grows with it, not with the atom count alone.")
    return rows


def write_rows(rows, path):
    """Write the profile rows to CSV, one column per key seen in any row."""
    if not rows:
        return None
    fields = sorted({k for row in rows for k in row})
    with open(path, "w") as fh:
        fh.write(",".join(fields) + "\n")
        for row in rows:
            fh.write(",".join(f"{row.get(k, '')}" for k in fields) + "\n")
    print(f"\nWrote {path}")
    return path
