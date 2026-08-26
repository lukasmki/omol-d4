"""The OpenMP thread scan, whose correctness rests on process isolation.

OMP_NUM_THREADS is read once when the OpenMP runtime loads, so a scan that
reused one process would report the same thread count at every point and draw a
flat curve that looks like "the sum does not parallelise". These tests pin the
subprocess protocol that avoids that, and the diagnostics that would catch it
if it broke.
"""

import numpy as np
import pytest

from omol_d4.profiling import (
    d4_timing,
    describe,
    omp_max_threads,
    throughput,
    thread_scan,
    write_rows,
)

TINY = dict(n_side=2, n_calls=1, warmup=0, functional="pbe", disp3_cutoff=6.0)


def test_omp_max_threads_is_a_count_or_unavailable():
    threads = omp_max_threads()
    assert threads is None or (isinstance(threads, int) and threads >= 1)


def test_d4_timing_reports_a_timed_call_and_its_conditions():
    result = d4_timing(**TINY)
    assert result["n_mol"] == 8
    assert result["n_atoms"] == 24
    assert result["d4_ms"] > 0.0
    assert result["disp3_cutoff_A"] == 6.0
    # A three-body-only D4 term is repulsive in a condensed phase.
    assert result["p_atm_bar"] > 0.0


def test_d4_timing_result_survives_the_worker_round_trip():
    """`thread_scan` moves this dict through JSON, so it must be plain data."""
    import json

    result = d4_timing(**TINY)
    assert json.loads(json.dumps(result)) == result


@pytest.mark.slow
def test_thread_scan_measures_each_point_in_its_own_process():
    """Each subprocess must actually see the OMP_NUM_THREADS it was given."""
    rows = thread_scan(threads=(1, 2), task="omol", **TINY)

    assert [row["threads"] for row in rows] == [1, 2]
    seen = [row["omp_max_threads"] for row in rows]
    if seen[0] is not None:      # only assert when the runtime is probeable
        assert seen == [1, 2], "OMP_NUM_THREADS did not reach the subprocess"

    # The first thread count is the baseline the speedups are relative to.
    assert rows[0]["speedup"] == pytest.approx(1.0)
    assert rows[0]["efficiency_pct"] == pytest.approx(100.0)
    assert rows[1]["speedup"] == pytest.approx(
        rows[0]["d4_ms"] / rows[1]["d4_ms"]
    )


@pytest.mark.slow
def test_thread_count_does_not_change_the_answer():
    """P_ATM is the control: parallelism must not move the physics."""
    rows = thread_scan(threads=(1, 2), task="omol", **TINY)
    pressures = np.array([row["p_atm_bar"] for row in rows])
    assert np.ptp(pressures) < 1e-6 * abs(pressures).max()


def test_write_rows_covers_every_key_any_row_has():
    """Size, cutoff and thread rows have different keys and share one writer."""
    rows = [{"mode": "threads", "threads": 1, "d4_ms": 1.0},
            {"mode": "cutoff", "d4_ms": 2.0, "p_atm_bar": 3.0}]
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "profile.csv"
        write_rows(rows, path)
        lines = path.read_text().splitlines()

    assert lines[0] == "d4_ms,mode,p_atm_bar,threads"
    assert lines[1] == "1.0,threads,,1"     # missing keys stay empty
    assert lines[2] == "2.0,cutoff,3.0,"


def test_write_rows_does_nothing_without_rows(tmp_path):
    assert write_rows([], tmp_path / "profile.csv") is None
    assert not (tmp_path / "profile.csv").exists()


def test_throughput_projects_md_time_from_a_call_time():
    """1 s per call at 0.5 fs => 3600 calls per hour => 1.8 ps/hour."""
    assert throughput(1.0, timestep_fs=0.5) == pytest.approx(1.8)


def test_describe_handles_a_single_measurement():
    """n_calls=1 leaves no spread to report, and must not crash the table."""
    assert "+/-" not in describe(np.array([0.5]))
    assert "+/-" in describe(np.array([0.5, 0.7]))
