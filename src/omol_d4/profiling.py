"""Stage 0 (optional): profile the cost of UMA against UMA + D4 three-body.

The point is to decide whether an on-the-fly three-body run is affordable before
starting one. Two things dominate the answer and both are measured here:

  * How the two scale with system size. The MLIP is roughly linear in the atom
    count; the ATM triple sum is not, so the crossover matters more than any
    single-size timing.
  * The ATM real-space cutoff, which trades cost against how converged the
    three-body pressure is.
"""

import ctypes
import json
import os
import subprocess
import sys
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

TIMESTEP_FS = 0.5  # only used to project MD throughput
JITTER = 0.005  # A, per call, to defeat the calculators' result caches

# Thread counts for `thread_scan`. Powers of two up to a full Perlmutter CPU
# node; counts above the visible core count are still measured, and flagged.
DEFAULT_THREAD_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128)

# How a worker subprocess hands its measurement back to `thread_scan`. The
# worker is spawned with -c rather than -m: `omol_d4/__init__.py` already
# imports this module, and re-running it under -m warns about the duplicate.
WORKER_MARKER = "@@omol_d4.profiling@@ "
WORKER_COMMAND = (
    "import sys; from omol_d4.profiling import _worker_main; _worker_main(sys.argv)"
)

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


def time_calculator(atoms, calc, n_calls, device, warmup=2, rng=None, want_stress=True):
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
            atoms.get_stress()  # same calculate() call, so this is a dict hit
        cuda_sync(device)
        elapsed = time.perf_counter() - start

        if i >= warmup:
            times.append(elapsed)

    return np.array(times)


def describe(times):
    """Mean +/- std in ms, for one column of the profile table."""
    return (
        f"{times.mean() * 1e3:9.1f} +/- {times.std() * 1e3:6.1f}"
        if len(times) > 1
        else f"{times.mean() * 1e3:9.1f}"
    )


def throughput(seconds_per_call, timestep_fs=TIMESTEP_FS):
    """ps of MD per hour of wall clock, at `timestep_fs`."""
    return 3600.0 / seconds_per_call * timestep_fs / 1000.0


def environment_note(device="cuda", model=MODEL, task=TASK):
    """Print the things that silently invalidate a timing comparison."""
    threads = os.environ.get("OMP_NUM_THREADS", "unset")
    print(f"  device={device}  model={model}  task={task}")
    print(f"  OMP_NUM_THREADS={threads} (cores visible: {os.cpu_count()})")
    if threads == "unset":
        print(
            "  note: dftd4 is OpenMP-parallel and the MLIP is not competing "
            "for those cores; set OMP_NUM_THREADS to profile realistically."
        )
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


def size_sweep(
    sizes=(3, 4, 5, 6),
    n_calls=5,
    device="cuda",
    model=MODEL,
    task=TASK,
    functional=None,
    disp3_cutoff=DISP3_CUTOFF,
    check_sum=False,
    seed=SEED,
):
    """Time UMA and the ATM term separately across box sizes. Returns rows."""
    # Pass a representative box so build_calculator can warn about running a
    # non-periodic task (omol) on a periodic cell.
    probe = _probe_box(sizes[0])
    uma = build_calculator(
        device=device, seed=seed, three_body=False, model=model, task=task, atoms=probe
    )
    d4 = make_d4_atm(functional or functional_for(task), disp3_cutoff=disp3_cutoff)

    print(
        f"\n{'n_H2O':>6} {'atoms':>6} {'L/A':>6} "
        f"{'UMA/ms':>18} {'D4/ms':>18} {'sum/ms':>10} {'D4 share':>9} "
        f"{'ps/hour':>9}"
    )
    print("-" * 88)

    rows = []
    for n_side in sizes:
        atoms = _probe_box(n_side)
        length = atoms.cell.lengths()[0]

        t_uma = time_calculator(
            atoms.copy(), uma, n_calls, device, rng=np.random.default_rng(1)
        )
        t_d4 = time_calculator(
            atoms.copy(), d4, n_calls, device, rng=np.random.default_rng(1)
        )
        total = t_uma.mean() + t_d4.mean()

        print(
            f"{n_molecules(atoms):6d} {len(atoms):6d} {length:6.2f} "
            f"{describe(t_uma):>18} {describe(t_d4):>18} "
            f"{total * 1e3:10.1f} {t_d4.mean() / total * 100:8.1f}% "
            f"{throughput(total):9.2f}"
        )

        rows.append(
            {
                "mode": "size",
                "n_mol": n_molecules(atoms),
                "n_atoms": len(atoms),
                "cell_A": length,
                "disp3_cutoff_A": disp3_cutoff,
                "uma_ms": t_uma.mean() * 1e3,
                "uma_std_ms": t_uma.std() * 1e3,
                "d4_ms": t_d4.mean() * 1e3,
                "d4_std_ms": t_d4.std() * 1e3,
            }
        )

    # Sanity check that summing the two really does cost the sum of the two.
    if check_sum:
        atoms = _probe_box(sizes[-1])
        both = build_calculator(
            device=device,
            seed=seed,
            three_body=True,
            model=model,
            task=task,
            functional=functional,
            disp3_cutoff=disp3_cutoff,
            atoms=atoms,
        )
        t_both = time_calculator(
            atoms, both, n_calls, device, rng=np.random.default_rng(1)
        )
        print(
            f"\n  SumCalculator at n_side={sizes[-1]}: "
            f"{describe(t_both)} ms measured, "
            f"{rows[-1]['uma_ms'] + rows[-1]['d4_ms']:.1f} ms expected"
        )

    return rows


def cutoff_scan(
    cutoffs=(6.0, 8.0, 10.0, 12.0, 16.0),
    n_side=6,
    n_calls=5,
    device="cuda",
    task=TASK,
    functional=None,
    include_default=True,
):
    """Scan the ATM real-space cutoff at one box size. Returns rows.

    `None` is appended to `cutoffs` when `include_default`, which measures
    dftd4's own defaults as the converged reference the others approach.
    """
    functional = functional or functional_for(task)
    atoms = _probe_box(n_side)
    length = atoms.cell.lengths()[0]
    print(
        f"\nD4 three-body cutoff scan ({functional} damping), "
        f"{n_molecules(atoms)} H2O, L = {length:.2f} A"
    )
    print(f"\n{'disp3/A':>8} {'D4/ms':>18} {'P_ATM/bar':>11} {'images':>8}")
    print("-" * 50)

    rows = []
    for cutoff in list(cutoffs) + ([None] if include_default else []):
        d4 = make_d4_atm(functional, disp3_cutoff=cutoff, quiet=True)
        probe = atoms.copy()
        probe.calc = d4
        pressure = -probe.get_stress()[:3].mean() * EV_PER_A3_TO_BAR
        times = time_calculator(
            atoms.copy(), d4, n_calls, device, rng=np.random.default_rng(1)
        )
        span = DEFAULT_DISP3_BOHR * Bohr if cutoff is None else cutoff
        print(
            f"{'default' if cutoff is None else f'{cutoff:.1f}':>8} "
            f"{describe(times):>18} {pressure:11.1f} "
            f"{(2 * span / length + 1) ** 3:8.1f}"
        )
        rows.append(
            {
                "mode": "cutoff",
                "n_mol": n_molecules(atoms),
                "n_atoms": len(atoms),
                "cell_A": length,
                "disp3_cutoff_A": "default" if cutoff is None else cutoff,
                "d4_ms": times.mean() * 1e3,
                "d4_std_ms": times.std() * 1e3,
                "p_atm_bar": pressure,
            }
        )

    print(
        "\n  'images' is the rough number of periodic cells the triple sum "
        "spans; the ATM cost grows with it, not with the atom count alone."
    )
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


# ---------------------------------------------------------------------------
# OpenMP thread scaling
# ---------------------------------------------------------------------------
# dftd4 is the OpenMP-parallel half of the calculator and the ATM triple sum is
# what an on-the-fly three-body run spends its time in, so how well that sum
# scales decides how many cores are worth asking for.
#
# OMP_NUM_THREADS is read by the OpenMP runtime when it loads, which is the
# first time dftd4 is imported. Re-assigning os.environ afterwards does nothing
# at all, so a scan inside one process would silently report the same thread
# count for every point. `omp_set_num_threads` would take effect, but it leaves
# the thread affinity established at load time in place, which is not what a
# fresh job with OMP_PROC_BIND set would do. So each point is measured in its
# own subprocess, exactly as the batch script would launch it.


def omp_max_threads():
    """What the OpenMP runtime itself reports, or None if it cannot be probed.

    Worth recording next to every timing: it is the only direct evidence that
    OMP_NUM_THREADS was actually honoured, and a dftd4 built without OpenMP
    shows up here rather than as a mysteriously flat scaling curve.
    """
    try:
        return int(ctypes.CDLL(None).omp_get_max_threads())
    except Exception:  # noqa: BLE001 - a probe, never worth failing a run over
        return None


def d4_timing(
    n_side=4,
    functional="pbe",
    disp3_cutoff=DISP3_CUTOFF,
    n_calls=5,
    warmup=2,
    seed=SEED,
):
    """Time one ATM-only D4 force+stress evaluation. Returns a plain dict.

    Plain data rather than arrays because this is what a worker subprocess
    serialises back to `thread_scan`.
    """
    atoms = _probe_box(n_side)
    d4 = make_d4_atm(functional, disp3_cutoff=disp3_cutoff, quiet=True)

    probe = atoms.copy()
    probe.calc = d4
    pressure = -probe.get_stress()[:3].mean() * EV_PER_A3_TO_BAR

    # device="cpu": dftd4 is CPU-only, so there is no GPU queue to drain.
    times = time_calculator(
        atoms.copy(), d4, n_calls, "cpu", warmup=warmup, rng=np.random.default_rng(seed)
    )
    return {
        "n_mol": n_molecules(atoms),
        "n_atoms": len(atoms),
        "cell_A": float(atoms.cell.lengths()[0]),
        "disp3_cutoff_A": disp3_cutoff,
        "d4_ms": float(times.mean() * 1e3),
        "d4_std_ms": float(times.std() * 1e3),
        "p_atm_bar": float(pressure),
        "omp_max_threads": omp_max_threads(),
    }


def thread_scan(
    threads=DEFAULT_THREAD_COUNTS,
    n_side=4,
    task=TASK,
    functional=None,
    disp3_cutoff=DISP3_CUTOFF,
    n_calls=5,
    warmup=2,
    seed=SEED,
):
    """Time the D4 ATM term once per OMP_NUM_THREADS value. Returns rows.

    Each point runs in a fresh subprocess with OMP_NUM_THREADS set before the
    OpenMP runtime loads; everything else in the environment (OMP_PLACES,
    OMP_PROC_BIND, the CPU binding SLURM applied) is inherited, so the numbers
    describe the job this is running inside.

    Speedup and efficiency are relative to the *first* thread count measured,
    which is not always 1: a serial call at an expensive cutoff can take
    minutes, so a scan may reasonably start at 8. Efficiency is scaled by the
    same baseline, so the first row is 100% either way.
    """
    functional = functional or functional_for(task)
    visible = os.cpu_count()
    print(
        f"\nD4 three-body OpenMP scaling ({functional} damping), "
        f"{n_side**3} H2O, disp3 cutoff "
        f"{'library default' if disp3_cutoff is None else f'{disp3_cutoff:.1f} A'}"
    )
    print(
        f"  {len(threads)} subprocesses, {visible} cores visible; speedup "
        f"and efficiency are relative to {threads[0]} thread"
        f"{'' if threads[0] == 1 else 's'}"
    )
    print(
        f"\n{'threads':>8} {'omp_seen':>9} {'D4/ms':>18} {'speedup':>8} "
        f"{'efficiency':>11} {'P_ATM/bar':>11}"
    )
    print("-" * 70)

    payload = {
        "n_side": n_side,
        "functional": functional,
        "disp3_cutoff": disp3_cutoff,
        "n_calls": n_calls,
        "warmup": warmup,
        "seed": seed,
    }

    rows, baseline_ms, baseline_threads = [], None, threads[0]
    for n_threads in threads:
        result = _run_worker(payload, n_threads)

        if baseline_ms is None:
            baseline_ms = result["d4_ms"]
        speedup = baseline_ms / result["d4_ms"]
        efficiency = speedup / (n_threads / baseline_threads) * 100
        seen = result["omp_max_threads"]

        row = {
            "mode": "threads",
            "threads": n_threads,
            "baseline_threads": baseline_threads,
            "speedup": speedup,
            "efficiency_pct": efficiency,
            **result,
        }
        rows.append(row)

        flag = "  (oversubscribed)" if visible and n_threads > visible else ""
        print(
            f"{n_threads:8d} {'?' if seen is None else seen:>9} "
            f"{result['d4_ms']:9.1f} +/- {result['d4_std_ms']:6.1f} "
            f"{speedup:8.2f} {efficiency:10.1f}% "
            f"{result['p_atm_bar']:11.1f}{flag}"
        )

    _warn_about_the_scan(rows, visible)
    return rows


def _run_worker(payload, n_threads):
    """Measure one point in a subprocess with OMP_NUM_THREADS pre-set."""
    env = dict(os.environ, OMP_NUM_THREADS=str(n_threads))
    proc = subprocess.run(
        [sys.executable, "-c", WORKER_COMMAND, json.dumps(payload)],
        capture_output=True,
        text=True,
        env=env,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(WORKER_MARKER):
            return json.loads(line[len(WORKER_MARKER) :])
    raise RuntimeError(
        f"profiling worker failed at OMP_NUM_THREADS={n_threads} "
        f"(exit {proc.returncode}):\n{proc.stderr or proc.stdout}"
    )


def _warn_about_the_scan(rows, visible):
    """Call out the two ways this measurement is commonly misread."""
    seen = {r["omp_max_threads"] for r in rows}
    if seen == {None}:
        print(
            "\n  note: the OpenMP runtime could not be probed, so there is "
            "no confirmation that OMP_NUM_THREADS was honoured."
        )
    elif len(seen) == 1:
        print(
            f"\n  WARNING: every run reported {seen.pop()} threads, so "
            "OMP_NUM_THREADS is not reaching the dispersion library - it may "
            "have been built without OpenMP. The timings below are all the "
            "same configuration."
        )

    pressures = np.array([r["p_atm_bar"] for r in rows])
    if np.ptp(pressures) > 1e-6 * abs(pressures).max():
        print(
            "  note: P_ATM should not depend on the thread count; a spread "
            "here means the parallel sum is not reproducible."
        )

    if visible:
        useful = [r for r in rows if r["threads"] <= visible]
        if len(useful) > 1:
            best = max(useful, key=lambda r: r["speedup"])
            print(
                f"\n  Best speedup within the {visible} visible cores: "
                f"{best['speedup']:.1f}x at {best['threads']} threads "
                f"({best['efficiency_pct']:.0f}% efficiency). Past the knee "
                "the extra cores buy little, and on a shared node they are "
                "better spent elsewhere."
            )


def _worker_main(argv):
    """Measure one point and print it for the parent; see `thread_scan`.

    Spawned via WORKER_COMMAND. To reproduce a single point by hand:

        OMP_NUM_THREADS=8 python -c "import sys; \
            from omol_d4.profiling import _worker_main; _worker_main(sys.argv)" \
            '{"n_side": 4, "functional": "pbe", "disp3_cutoff": 12.0}'
    """
    result = d4_timing(**json.loads(argv[1]))
    print(WORKER_MARKER + json.dumps(result))
