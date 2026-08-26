#!/usr/bin/env python
"""
Stage 1: build a periodic water box and equilibrate it in the NVT ensemble
with the UMA MLIP.

The box is built at the experimental density of liquid water (0.997 g/cm^3 at
300 K) and held at fixed volume while the structure relaxes from the artificial
lattice-like starting configuration into a proper liquid. Stage 2 (2-npt.py)
then lets the volume float so the *model's* density can be measured.

All of the work happens in `omol_d4.nvt.run_nvt`; this script is only its
command line. Note that "omol" is the task for isolated (non-periodic)
molecules and cannot supply a stress, so a stage 2 run following it needs the
Monte Carlo barostat; "omc" (organic molecular crystals) is the periodic task
covering molecular condensed phases.

Run:
    python 1-nvt.py                        # UMA/omc only
    python 1-nvt.py --three-body           # UMA + D4 ATM three-body term
    python 1-nvt.py --task omol            # OMol25 (wB97M-V) instead
    python 1-nvt.py --model uma-m-1p1
    python 1-nvt.py --n-side 4             # 64 H2O instead of 216

Output (suffixed with _atm when --three-body is used, _nN for a --n-side
other than the default 6):
    water_nvt.traj           full NVT trajectory (for sanity checks)
    water_nvt.log            time / energy / temperature log
    water_equilibrated.traj  final frame, incl. velocities -> input to stage 2
"""

import argparse

from omol_d4.boxes import DEFAULT_N_SIDE
from omol_d4.calculators import KNOWN_MODELS, MODEL, TASK
from omol_d4.constants import NVT_STEPS
from omol_d4.nvt import run_nvt


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--three-body",
        action="store_true",
        help="add the D4 Axilrod-Teller-Muto three-body dispersion term",
    )
    parser.add_argument(
        "--model", default=MODEL, help=f"UMA checkpoint, e.g. {', '.join(KNOWN_MODELS)}"
    )
    parser.add_argument(
        "--task", default=TASK, help="UMA task: omc (periodic) or omol (molecular)"
    )
    parser.add_argument(
        "--functional", default=None, help="override the D4 damping parameter set"
    )
    parser.add_argument(
        "--n-side",
        type=int,
        default=DEFAULT_N_SIDE,
        help=f"n^3 water molecules per box (default {DEFAULT_N_SIDE})",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=NVT_STEPS,
        help=f"NVT steps after the soft start (default {NVT_STEPS})",
    )
    parser.add_argument("--outdir", default=".", help="where to write the outputs")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    run_nvt(
        n_side=args.n_side,
        steps=args.n_steps,
        three_body=args.three_body,
        model=args.model,
        task=args.task,
        functional=args.functional,
        device=args.device,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
