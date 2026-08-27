#!/usr/bin/env python
"""
Stage 3: plot the NPT traces - density, volume and temperature against time.

Every `npt_volume*.csv` in --outdir becomes one line per figure, labelled from
the model, task, box size and three-body flag recorded in its own filename. So
the two arms of a comparison (UMA alone, UMA + D4 three-body) plot together
without being named here, and a new run plots by existing.

These are the diagnostic plots, not the measurement: the density and its error
bar come from stage 2's --analyse, which averages over the production window.
What a plot shows that a number cannot is whether the trace is still drifting,
how large the volume fluctuations are, and whether the thermostat held.

All of the work happens in `omol_d4.plotting`; this script is only its command
line.

Run:
    python 3-plot.py                       # every run in the current directory
    python 3-plot.py --outdir ../data      # the 64-molecule omol comparison
    python 3-plot.py --quantity density    # just the one figure
    python 3-plot.py --csv ../data/npt_volume_uma-s-1p2p1_omol_n4.csv \
                     --csv "npt_volume_uma-s-1p2p1_omol_n5.csv:125 H2O"

Output (in --plotdir, default --outdir):
    density_vs_time.png
    volume_vs_time.png
    temperature_vs_time.png
"""

import argparse

from omol_d4.plotting import DPI, QUANTITIES, parse_series_spec, plot_runs


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--outdir", default=".", help="where the npt_volume*.csv files live"
    )
    parser.add_argument(
        "--plotdir", default=None, help="where the figures go (default: --outdir)"
    )
    parser.add_argument(
        "--quantity",
        action="append",
        choices=list(QUANTITIES),
        help="plot only this quantity; repeatable (default: all three)",
    )
    parser.add_argument(
        "--csv",
        action="append",
        metavar="PATH[:LABEL]",
        help="plot these traces instead of everything in --outdir; repeatable",
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    return parser.parse_args()


def main():
    args = parse_args()
    plot_runs(
        outdir=args.outdir,
        plotdir=args.plotdir,
        quantities=args.quantity,
        series=[parse_series_spec(spec) for spec in args.csv] if args.csv else None,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
