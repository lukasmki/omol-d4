"""Stage 3: the NPT trace as three figures.

Density, volume and temperature against time, one line per run. The point of
the figures is the comparison - UMA against UMA + D4 three-body, or one box size
against another - so a run is not named in code here: every `npt_volume*.csv` in
the output directory is a series, labelled from the tag in its own filename
(`omol_d4.paths.npt_csvs`). A new run plots by existing.

These are diagnostic plots, not the measurement: the density and its error bar
come from `omol_d4.analysis`, which averages over the production window. What a
plot shows that a number cannot is whether the trace is drifting, how big the
volume fluctuations are, and whether the thermostat is holding - the things you
look at before believing the number.
"""

from pathlib import Path

import matplotlib

# Chosen before pyplot is imported: these run on compute nodes with no display,
# and the default backend would fail there rather than write a file.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend choice

from .analysis import load_volume_csv  # noqa: E402
from .calculators import TASK, parse_tag  # noqa: E402
from .paths import NPT_CSV_STEM, npt_csvs  # noqa: E402

# What each figure plots, and what it is called. The three plots differ only by
# these four strings, so they are data rather than three near-identical
# functions.
QUANTITIES = {
    "density": {
        "column": "density_g_cm3",
        "ylabel": "Density (g/cm$^3$)",
        "title": "NPT density vs. time",
        "filename": "density_vs_time.png",
    },
    "volume": {
        "column": "volume_A3",
        "ylabel": "Volume (Å$^3$)",
        "title": "NPT volume vs. time",
        "filename": "volume_vs_time.png",
    },
    "temperature": {
        "column": "temperature_K",
        "ylabel": "Temperature (K)",
        "title": "NPT temperature vs. time",
        "filename": "temperature_vs_time.png",
    },
}

FIGSIZE = (7, 4.5)
DPI = 200


def model_label(model):
    """`uma-s-1p2p1` -> `UMA-s-1.2.1`, the spelling used in the papers."""
    family, _, version = model.partition("-1p")
    if not version:
        return model.replace("uma", "UMA", 1)
    return f"{family.replace('uma', 'UMA', 1)}-1.{version.replace('p', '.')}"


def label_for(run):
    """Legend entry for a `RunTag`: which model, which box, with ATM or without.

    Only what distinguishes this run from another one in the same figure is
    spelled out, so a default task stays silent and the common case reads as
    plain model names.
    """
    label = model_label(run.model)
    if run.task != TASK:
        label += f"/{run.task}"
    label += f", {run.n_molecules} H2O"
    if run.three_body:
        label += " (3B-D4)"
    return label


def load_series(outdir="."):
    """Every run in `outdir` as `(label, columns)`, ready to plot."""
    runs = npt_csvs(outdir)
    if not runs:
        raise SystemExit(
            f"No npt_volume*.csv in {Path(outdir).resolve()} - run stage 2 first, "
            "or point --outdir at the directory holding a finished run."
        )
    return [(label_for(run), load_volume_csv(path)) for run, path in runs]


def quantity_spec(quantity):
    """The `QUANTITIES` entry for `quantity`, or a clear failure."""
    try:
        return QUANTITIES[quantity]
    except KeyError:
        raise SystemExit(
            f"Unknown quantity '{quantity}'; expected one of {', '.join(QUANTITIES)}."
        ) from None


def parse_series_spec(spec):
    """`path` or `path:Label` -> one `(label, columns)` series.

    Lets a figure be assembled from runs that no single directory holds - two
    box sizes from two output directories, say. Without a label the run's own
    filename tag supplies one, exactly as discovery would.
    """
    path = Path(spec)
    label = None
    if not path.exists() and ":" in spec:
        head, _, label = spec.rpartition(":")
        path = Path(head)
    if not path.exists():
        raise SystemExit(f"No such volume trace: {path}")
    if label is None:
        label = label_for(parse_tag(path.stem.removeprefix(NPT_CSV_STEM)))
    return label, load_volume_csv(path)


def plot_quantity(series, quantity, out, dpi=DPI):
    """One figure: `quantity` against time, one line per entry in `series`.

    `series` is a list of `(label, columns)`, where `columns` is what
    `analysis.load_volume_csv` returns.
    """
    spec = quantity_spec(quantity)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, columns in series:
        ax.plot(
            columns["time_ps"], columns[spec["column"]], lw=1.0, alpha=0.8, label=label
        )

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(spec["title"])
    # A drifting trace is read off the grid; the lines belong behind the data.
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()

    out = Path(out)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def plot_runs(outdir=".", plotdir=None, quantities=None, series=None, dpi=DPI):
    """Write one figure per quantity for every run in `outdir`.

    `series` overrides the discovery, for a comparison the filenames do not
    describe (two directories, say, or a subset of the runs in one). Each CSV is
    read once and reused across the figures.
    """
    if series is None:
        series = load_series(outdir)
    plotdir = Path(outdir if plotdir is None else plotdir)
    plotdir.mkdir(parents=True, exist_ok=True)

    written = []
    for quantity in quantities or QUANTITIES:
        out = plot_quantity(
            series, quantity, plotdir / quantity_spec(quantity)["filename"], dpi=dpi
        )
        print(f"wrote {out}")
        written.append(out)
    return written
