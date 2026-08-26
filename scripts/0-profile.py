#!/usr/bin/env python
"""
Stage 0 (optional): profile the cost of UMA against UMA + D4 three-body, so you
can decide whether an on-the-fly --three-body run is affordable before starting
one.

Two things dominate the answer and both are measured here:

  * How the two scale with system size. The MLIP is roughly linear in the atom
    count; the ATM triple sum is not, so the crossover matters more than any
    single-size timing.
  * The ATM real-space cutoff, which trades cost against how converged the
    three-body pressure is.

The measurements live in `omol_d4.profiling`; this script is only their command
line.

Run:
    python 0-profile.py                          # size sweep, both calculators
    python 0-profile.py --sizes 4 5 6 --n-calls 10
    python 0-profile.py --cutoff-scan --sizes 6  # disp3 cutoff sweep
    python 0-profile.py --device cpu --sizes 2 3 # quick smoke test

Output:
    profile.csv   one row per (calculator, size or cutoff) measured
"""

import argparse

from omol_d4.calculators import DISP3_CUTOFF, KNOWN_MODELS, MODEL, TASK
from omol_d4.profiling import (
    TIMESTEP_FS,
    cutoff_scan,
    environment_note,
    size_sweep,
    write_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6],
        help="n_side values; n_side^3 water molecules each",
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=5,
        help="timed calls per configuration (after warmup)",
    )
    parser.add_argument(
        "--cutoff-scan",
        action="store_true",
        help="scan the ATM cutoff at the largest size instead",
    )
    parser.add_argument(
        "--cutoffs",
        type=float,
        nargs="+",
        default=[6.0, 8.0, 10.0, 12.0, 16.0],
        help="explicit ATM cutoffs to scan; the library "
        "default is always measured alongside them",
    )
    parser.add_argument(
        "--disp3-cutoff",
        type=float,
        default=DISP3_CUTOFF,
        help="ATM real-space cutoff in Angstrom used by the size sweep",
    )
    parser.add_argument(
        "--check-sum", action="store_true", help="also time the combined SumCalculator"
    )
    parser.add_argument(
        "--model", default=MODEL, help=f"UMA checkpoint, e.g. {', '.join(KNOWN_MODELS)}"
    )
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--functional", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--csv", default="profile.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    environment_note(device=args.device, model=args.model, task=args.task)

    if args.cutoff_scan:
        # The cutoff scan needs no MLIP, so it is measured at one size only.
        rows = cutoff_scan(
            cutoffs=args.cutoffs,
            n_side=args.sizes[-1],
            n_calls=args.n_calls,
            device=args.device,
            task=args.task,
            functional=args.functional,
        )
    else:
        rows = size_sweep(
            sizes=args.sizes,
            n_calls=args.n_calls,
            device=args.device,
            model=args.model,
            task=args.task,
            functional=args.functional,
            disp3_cutoff=args.disp3_cutoff,
            check_sum=args.check_sum,
        )

    write_rows(rows, args.csv)
    print(
        "\nProjected wall time is per force+stress call, so an NPT run of "
        f"60,000 steps at {TIMESTEP_FS} fs costs 60,000 x the 'sum' column."
    )


if __name__ == "__main__":
    main()
