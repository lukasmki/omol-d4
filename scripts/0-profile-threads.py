#!/usr/bin/env python
"""
Stage 0 (optional): profile the D4 three-body evaluation time as a function of
the OpenMP thread count, to decide how many cores an on-the-fly --three-body
run is worth giving.

dftd4 is the OpenMP-parallel half of the calculator and the ATM triple sum is
where such a run spends its time, so this is the measurement that says whether
a full node earns its keep or whether the sum stops scaling well before it.

Each thread count is measured in a *fresh subprocess* with OMP_NUM_THREADS set
before the OpenMP runtime loads. That is not a detail: the runtime reads the
variable once at load time, so a scan inside a single process would report the
same thread count for every point and produce a flat, meaningless curve. The
subprocess also inherits OMP_PLACES, OMP_PROC_BIND and whatever CPU binding
SLURM applied, so the numbers describe the job this is running inside - run it
on a compute node, under the same environment as the production job, or they
do not transfer.

The `omp_seen` column is the runtime's own omp_get_max_threads(), i.e. direct
evidence that OMP_NUM_THREADS was honoured. P_ATM is printed for every row as
a control: it must not depend on the thread count.

Run:
    python 0-profile-threads.py                       # 1..128 threads, 64 H2O
    python 0-profile-threads.py --threads 1 8 32 128
    python 0-profile-threads.py --n-side 6            # 216 H2O instead
    python 0-profile-threads.py --disp3-cutoff default
    python 0-profile-threads.py --threads 1 2 4 --n-calls 2   # quick check

Output:
    profile-threads.csv   one row per thread count measured
"""

import argparse

from omol_d4.calculators import DISP3_CUTOFF, TASK
from omol_d4.profiling import DEFAULT_THREAD_COUNTS, thread_scan, write_rows


def cutoff(value):
    """A float, or 'default' to leave dftd4's own real-space cutoffs alone."""
    if value.lower() in ("default", "none"):
        return None
    return float(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=list(DEFAULT_THREAD_COUNTS),
        help="OMP_NUM_THREADS values to measure; the first is "
        "the baseline the speedups are relative to "
        f"(default {' '.join(map(str, DEFAULT_THREAD_COUNTS))})",
    )
    parser.add_argument(
        "--n-side",
        type=int,
        default=4,
        help="n^3 water molecules in the profiled box "
        "(default 4, i.e. the 64 H2O production box)",
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=5,
        help="timed calls per thread count (after warmup)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="untimed calls first, absorbing the dispersion model construction",
    )
    parser.add_argument(
        "--disp3-cutoff",
        type=cutoff,
        default=DISP3_CUTOFF,
        help="ATM real-space cutoff in Angstrom, or 'default' "
        "for dftd4's own (40 Bohr = 21.2 A)",
    )
    parser.add_argument(
        "--task",
        default=TASK,
        help="only selects the D4 damping parameter set; no "
        "MLIP is loaded by this script",
    )
    parser.add_argument(
        "--functional", default=None, help="override the D4 damping parameter set"
    )
    parser.add_argument("--csv", default="profile-threads.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = thread_scan(
        threads=args.threads,
        n_side=args.n_side,
        task=args.task,
        functional=args.functional,
        disp3_cutoff=args.disp3_cutoff,
        n_calls=args.n_calls,
        warmup=args.warmup,
    )
    write_rows(rows, args.csv)
    print(
        "\nSet OMP_NUM_THREADS in run-UMA-D4.slurm to the knee of this curve, "
        "not to the core count: past it the extra threads mostly add "
        "synchronisation."
    )


if __name__ == "__main__":
    main()
