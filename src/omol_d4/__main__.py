from typer import Option, Typer

from .plotting import DPI, QUANTITIES, plot_runs

APP = Typer()


@APP.command()
def download():
    # download the OMol-D4-4M dataset (location TBD)
    pass


@APP.command()
def plot(
    outdir: str = Option(".", help="where the npt_volume*.csv files live"),
    plotdir: str = Option(None, help="where the figures go (default: --outdir)"),
    quantity: list[str] = Option(
        None, help=f"plot only this quantity ({', '.join(QUANTITIES)}); repeatable"
    ),
    dpi: int = Option(DPI, help="figure resolution"),
):
    """Plot the NPT traces of every run in a directory (stage 3)."""
    plot_runs(outdir=outdir, plotdir=plotdir, quantities=quantity or None, dpi=dpi)


def main():
    APP()


if __name__ == "__main__":
    main()
