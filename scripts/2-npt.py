#!/usr/bin/env python
"""
Stage 2: NPT simulation of the equilibrated water box at 298 K / 1 atm, and
measurement of the equilibrium density predicted by the model.

Two barostats are available (--barostat, default auto):
  mtk  Martyna-Tobias-Klein isotropic barostat (ase.md.nose_hoover_chain),
       which samples a true isothermal-isobaric ensemble from the
       calculator's stress tensor. Requires a periodic task (e.g. omc).
  mc   Monte Carlo barostat + Langevin thermostat (omol_d4.integrator), which
       accepts/rejects isotropic volume moves from energy differences alone
       and needs no stress. The only option for a task with no stress
       labels, e.g. omol (OMol25 is a non-periodic dataset).
"auto" picks mc for a non-periodic task and mtk otherwise. Both sample a true
NPT ensemble, so either way the isothermal compressibility can be read off
the run's volume fluctuations, which --atm-correction relies on.

The work happens in `omol_d4.npt` (the run) and `omol_d4.analysis` (the
density, its error bar, and the perturbative three-body estimate); this script
is only their command line.

Run:
    python 2-npt.py                          # UMA only, barostat auto-picked
    python 2-npt.py --three-body             # UMA + D4 ATM, on the fly
    python 2-npt.py --task omol --barostat mc  # explicit, though this is the
                                                # default for a non-periodic task
    python 2-npt.py --model uma-m-1p1
    python 2-npt.py --analyse                # re-analyse an existing csv
    python 2-npt.py --atm-correction         # estimate the ATM effect from an
                                              # existing UMA-only trajectory

Output (suffixed with _atm when --three-body is used, _nN for a --n-side
other than stage 1's default):
    water_npt.traj    trajectory
    water_npt.log     energy / temperature log
    npt_volume.csv    time, volume, temperature, instantaneous density
"""

import argparse

from omol_d4.analysis import analyse_run, atm_correction
from omol_d4.calculators import DISP3_CUTOFF, KNOWN_MODELS, MODEL, TASK
from omol_d4.constants import NPT_STEPS
from omol_d4.npt import BAROSTATS, run_npt


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
        "--three-body",
        action="store_true",
        help="add the D4 Axilrod-Teller-Muto three-body term during the MD",
    )
    parser.add_argument(
        "--analyse",
        "--analyze",
        action="store_true",
        dest="analyse",
        help="skip the MD and analyse an existing csv",
    )
    parser.add_argument(
        "--atm-correction",
        action="store_true",
        help="estimate the three-body effect perturbatively from an existing "
        "UMA-only trajectory instead of rerunning the MD",
    )
    parser.add_argument(
        "--n-frames", type=int, default=25, help="frames used by --atm-correction"
    )
    parser.add_argument(
        "--disp3-cutoff",
        type=cutoff,
        default=DISP3_CUTOFF,
        help="ATM real-space cutoff in Angstrom, or 'default' "
        "for dftd4's own (40 Bohr = 21.2 A)",
    )
    parser.add_argument(
        "--barostat",
        choices=BAROSTATS,
        default="auto",
        help="auto picks mc for tasks without a stress tensor "
        "(e.g. omol), mtk otherwise",
    )
    parser.add_argument(
        "--mc-interval",
        type=int,
        default=25,
        help="steps between Monte Carlo volume moves (--barostat mc)",
    )
    parser.add_argument(
        "--mc-volume-scale",
        type=float,
        default=None,
        help="initial MC volume move size in A^3 (--barostat mc); "
        "default is 1%% of the initial cell volume",
    )
    parser.add_argument(
        "--n-side",
        type=int,
        default=None,
        help="box size tag matching stage 1's --n-side, if not default",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=NPT_STEPS,
        help=f"NPT steps (default {NPT_STEPS})",
    )
    parser.add_argument(
        "--model", default=MODEL, help=f"UMA checkpoint, e.g. {', '.join(KNOWN_MODELS)}"
    )
    parser.add_argument(
        "--task",
        default=TASK,
        help="UMA task; must be periodic to run the mtk barostat",
    )
    parser.add_argument(
        "--functional", default=None, help="override the D4 damping parameter set"
    )
    parser.add_argument(
        "--outdir", default=".", help="where the stage 1 inputs live and the outputs go"
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.atm_correction:
        print(
            atm_correction(
                task=args.task,
                model=args.model,
                n_side=args.n_side,
                outdir=args.outdir,
                functional=args.functional,
                disp3_cutoff=args.disp3_cutoff,
                n_frames=args.n_frames,
            ).report()
        )
        return

    if not args.analyse:
        run_npt(
            steps=args.n_steps,
            three_body=args.three_body,
            model=args.model,
            task=args.task,
            functional=args.functional,
            disp3_cutoff=args.disp3_cutoff,
            device=args.device,
            n_side=args.n_side,
            barostat=args.barostat,
            mc_interval=args.mc_interval,
            mc_volume_scale=args.mc_volume_scale,
            outdir=args.outdir,
        )

    result = analyse_run(
        three_body=args.three_body,
        task=args.task,
        model=args.model,
        n_side=args.n_side,
        outdir=args.outdir,
    )
    print(result.report(note="  [with D4 three-body]" if args.three_body else ""))


if __name__ == "__main__":
    main()
