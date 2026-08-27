"""Output file names for the two MD stages, in one place.

Stage 1 writes files that stage 2 reads and stage 2 writes files that the
analysis reads, so the naming convention is a genuine interface between them
rather than a detail of either. `suffix()` (in `omol_d4.calculators`) builds the
tag that keeps different model/task/size/three-body runs from overwriting each
other; this module turns that tag into paths, and (`npt_csvs`) turns a
directory of finished runs back into tags.
"""

from dataclasses import dataclass
from pathlib import Path

from .calculators import MODEL, TASK, parse_tag, suffix

# The stage 2 CSV's fixed prefix, shared by the path builder below and by
# `npt_csvs`, which globs for it and reads the tag back out of what follows.
NPT_CSV_STEM = "npt_volume"


@dataclass(frozen=True)
class StagePaths:
    """Every file the two stages read or write for one run tag."""

    tag: str
    outdir: Path

    @property
    def nvt_traj(self) -> Path:
        return self.outdir / f"water_nvt{self.tag}.traj"

    @property
    def nvt_log(self) -> Path:
        return self.outdir / f"water_nvt{self.tag}.log"

    @property
    def equilibrated(self) -> Path:
        return self.outdir / f"water_equilibrated{self.tag}.traj"

    @property
    def npt_traj(self) -> Path:
        return self.outdir / f"water_npt{self.tag}.traj"

    @property
    def npt_log(self) -> Path:
        return self.outdir / f"water_npt{self.tag}.log"

    @property
    def npt_csv(self) -> Path:
        return self.outdir / f"{NPT_CSV_STEM}{self.tag}.csv"


def stage_paths(
    three_body=False, task=TASK, model=MODEL, n_side=None, outdir="."
) -> StagePaths:
    """Paths for one run, identified the same way `suffix()` identifies it."""
    return StagePaths(suffix(three_body, task, model, n_side), Path(outdir))


def equilibrated_input(
    three_body=False, task=TASK, model=MODEL, n_side=None, outdir="."
) -> Path:
    """Stage 1 output to start stage 2 from, falling back to the non-ATM one.

    Running stage 1 without ATM and stage 2 with it is a legitimate (and much
    cheaper) combination, so a missing _atm input is not an error. Same goes for
    a missing _n<N>_atm input: stage 1 is shared (no three-body) across both
    arms at a given box size, so the plain _n<N> file is the expected starting
    point for a three-body stage 2 run at that size.
    """
    path = stage_paths(three_body, task, model, n_side, outdir).equilibrated
    if three_body and not path.exists():
        fallback = stage_paths(False, task, model, n_side, outdir).equilibrated
        if fallback.exists():
            print(f"  {path} not found, starting from {fallback}")
            return fallback
    return path


def npt_csvs(outdir="."):
    """Every stage 2 volume trace in `outdir`, as `(RunTag, Path)` pairs.

    The tag is the only record a finished run leaves of what it was, so a
    directory of CSVs is a set of runs and this reads it as one. Sorted by
    `RunTag.sort_key`, which groups a run with the ones it is comparable to and
    puts the plain arm before its three-body partner - the order the two arms
    should be drawn in.
    """
    outdir = Path(outdir)
    runs = [
        (parse_tag(path.stem.removeprefix(NPT_CSV_STEM)), path)
        for path in outdir.glob(f"{NPT_CSV_STEM}*.csv")
    ]
    return sorted(runs, key=lambda run: run[0].sort_key())
