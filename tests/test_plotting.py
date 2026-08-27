"""Stage 3: finding the runs in a directory, naming them, and drawing them.

The figures themselves are not asserted on pixel by pixel - what is checked is
the machinery around them: that the filename tag survives a round trip through
`suffix()` and `parse_tag()`, that discovery orders the two arms of a comparison
the way they should be drawn, and that a directory of CSVs turns into files on
disk without a display attached.
"""

import numpy as np
import pytest

from omol_d4.analysis import density
from omol_d4.boxes import DEFAULT_N_SIDE
from omol_d4.calculators import MODEL, TASK, parse_tag, suffix
from omol_d4.npt import VolumeRecorder
from omol_d4.paths import npt_csvs, stage_paths
from omol_d4.plotting import QUANTITIES, label_for, parse_series_spec, plot_runs

MASS = 64 * 18.01528  # 64 H2O, the box the shipped data was measured on


def write_trace(tmp_path, three_body=False, task="omol", model="uma-s-1p2p1", n_side=4):
    """A short, valid volume trace on disk, written the way stage 2 writes one."""
    paths = stage_paths(three_body, task, model, n_side, tmp_path)
    time = np.linspace(0.0, 30.0, 50)
    volume = 1800.0 + np.sin(time)
    with open(paths.npt_csv, "w") as fh:
        fh.write(VolumeRecorder.CSV_HEADER + "\n")
        for t, v in zip(time, volume):
            fh.write(f"{t:.4f},{v:.4f},298.00,{density(MASS, v):.5f}\n")
    return paths.npt_csv


# --- the tag, read back ----------------------------------------------------


def test_parse_tag_inverts_suffix():
    """The two ends of the naming interface have to agree."""
    run = parse_tag(suffix(True, "omol", "uma-s-1p2p1", 4))
    assert (run.model, run.task, run.n_side, run.three_body) == (
        "uma-s-1p2p1",
        "omol",
        4,
        True,
    )
    assert run.n_molecules == 64


def test_an_untagged_run_is_the_default_run():
    """The empty tag says 'every choice was the default', not 'unknown'."""
    run = parse_tag(suffix(False))
    assert (run.model, run.task, run.n_side, run.three_body) == (
        MODEL,
        TASK,
        DEFAULT_N_SIDE,
        False,
    )


def test_tag_components_are_recognised_by_shape_not_position():
    run = parse_tag("_n5_atm")
    assert (run.n_side, run.three_body, run.model) == (5, True, MODEL)


# --- discovery -------------------------------------------------------------


def test_both_arms_are_found_with_the_plain_one_first(tmp_path):
    """A three-body run is drawn over the baseline it is compared against."""
    write_trace(tmp_path, three_body=True)
    write_trace(tmp_path, three_body=False)
    (tmp_path / "water_npt_uma-s-1p2p1_omol_n4.log").write_text("not a csv")

    found = npt_csvs(tmp_path)
    assert [run.three_body for run, _ in found] == [False, True]
    assert [path.name for _, path in found] == [
        "npt_volume_uma-s-1p2p1_omol_n4.csv",
        "npt_volume_uma-s-1p2p1_omol_n4_atm.csv",
    ]


def test_label_says_what_distinguishes_the_run():
    plain, atm = (parse_tag(suffix(b, "omol", "uma-s-1p2p1", 4)) for b in (False, True))
    assert label_for(plain) == "UMA-s-1.2.1/omol, 64 H2O"
    assert label_for(atm) == "UMA-s-1.2.1/omol, 64 H2O (3B-D4)"


def test_a_series_spec_can_carry_its_own_label(tmp_path):
    csv = write_trace(tmp_path)
    assert parse_series_spec(str(csv))[0] == "UMA-s-1.2.1/omol, 64 H2O"
    assert parse_series_spec(f"{csv}:whatever I like")[0] == "whatever I like"


def test_a_missing_trace_is_a_clear_failure(tmp_path):
    with pytest.raises(SystemExit, match="No such volume trace"):
        parse_series_spec(str(tmp_path / "npt_volume.csv"))


# --- the figures -----------------------------------------------------------


def test_every_quantity_is_written(tmp_path):
    write_trace(tmp_path, three_body=False)
    write_trace(tmp_path, three_body=True)

    written = plot_runs(tmp_path)
    assert [path.name for path in written] == [
        spec["filename"] for spec in QUANTITIES.values()
    ]
    assert all(path.stat().st_size > 0 for path in written)


def test_plotdir_is_created_and_a_subset_can_be_asked_for(tmp_path):
    write_trace(tmp_path)
    written = plot_runs(tmp_path, tmp_path / "figures", quantities=["density"])
    assert [path.name for path in written] == ["density_vs_time.png"]
    assert written[0].parent.name == "figures"


def test_an_empty_directory_says_so(tmp_path):
    with pytest.raises(SystemExit, match="No npt_volume"):
        plot_runs(tmp_path)
