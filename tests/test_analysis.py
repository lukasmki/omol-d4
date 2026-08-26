"""Density and compressibility from a volume trace, and the run naming scheme.

The statistics are checked against traces whose answer can be written down
analytically, so a change in the estimator shows up as a failure rather than as
a slightly different density.
"""

import numpy as np
import pytest
from ase import units

from omol_d4.analysis import (
    analyse_run,
    analyse_volume,
    compressibility,
    density,
    load_volume_csv,
    system_mass,
)
from omol_d4.calculators import MODEL, TASK, suffix
from omol_d4.constants import AMU_PER_A3_TO_G_PER_CM3, EV_PER_A3_TO_BAR
from omol_d4.npt import VolumeRecorder, resolve_barostat
from omol_d4.paths import equilibrated_input, stage_paths

TEMPERATURE_K = 298.0
MASS = 64 * 18.01528  # 64 H2O, the box the shipped data was measured on


def constant_trace(n=500, volume=1800.0, t_end=30.0):
    """A trace with no volume fluctuation at all."""
    return (np.linspace(0.0, t_end, n), np.full(n, volume), np.full(n, TEMPERATURE_K))


# --- the estimators --------------------------------------------------------


def test_density_matches_the_hand_calculation():
    assert density(MASS, 1800.0) == pytest.approx(
        MASS / 1800.0 * AMU_PER_A3_TO_G_PER_CM3
    )


def test_compressibility_of_a_rigid_box_is_zero():
    """kappa_T is a fluctuation formula, so a fixed volume must give exactly 0."""
    assert compressibility(np.full(100, 1800.0), TEMPERATURE_K) == 0.0


def test_compressibility_matches_its_closed_form():
    rng = np.random.default_rng(0)
    v = 1800.0 + rng.normal(scale=50.0, size=5000)
    expected = v.var() / (units.kB * TEMPERATURE_K * v.mean()) / EV_PER_A3_TO_BAR
    assert compressibility(v, TEMPERATURE_K) == pytest.approx(expected)


def test_density_is_reported_as_mass_over_mean_volume():
    """rho = M / <V>, not <M/V>: the volume is the fluctuating quantity."""
    time, volume, temperature = constant_trace()
    volume = volume + np.tile([-100.0, 100.0], len(volume) // 2)
    result = analyse_volume(time, volume, temperature, MASS, equil_ps=0.0)
    assert result.density == pytest.approx(density(MASS, volume.mean()))
    assert result.density != pytest.approx(np.mean(density(MASS, volume)))


# --- the production window and the block error bar -------------------------


def test_equilibration_window_is_discarded():
    time, volume, temperature = constant_trace(n=600, t_end=30.0)
    volume[time < 10.0] = 9999.0  # junk that must not reach the average
    result = analyse_volume(time, volume, temperature, MASS, equil_ps=10.0)
    assert result.volume_mean == pytest.approx(1800.0)
    assert result.time_range_ps[0] >= 10.0


def test_too_few_production_samples_is_a_clear_failure():
    time, volume, temperature = constant_trace(n=20, t_end=30.0)
    with pytest.raises(SystemExit, match="production samples"):
        analyse_volume(time, volume, temperature, MASS, equil_ps=29.0)


def test_block_error_bar_vanishes_for_a_constant_trace():
    time, volume, temperature = constant_trace()
    result = analyse_volume(time, volume, temperature, MASS, equil_ps=0.0)
    assert result.density_stderr == pytest.approx(0.0)
    assert len(result.block_densities) == 5
    assert not result.drifting


def test_a_steadily_drifting_trace_is_flagged():
    """The first/second half check is the cheap 'not equilibrated yet' alarm."""
    time, _, temperature = constant_trace()
    volume = np.linspace(1600.0, 2000.0, len(time))
    result = analyse_volume(time, volume, temperature, MASS, equil_ps=0.0)
    assert result.drifting
    assert "still drifting" in result.report()


# --- reading a run back off disk ------------------------------------------


def test_system_mass_is_recovered_from_a_volume_trace():
    """A CSV carries mass implicitly, so an archived run can stand alone."""
    volume = np.array([1800.0, 1750.0, 1900.0])
    columns = {"volume_A3": volume, "density_g_cm3": density(MASS, volume)}
    assert system_mass(columns) == pytest.approx(MASS)


def test_analyse_run_reads_a_csv_written_by_the_recorder(tmp_path):
    """End to end: the columns stage 2 writes are the columns analysis reads."""
    time, volume, temperature = constant_trace()
    paths = stage_paths(False, "omol", "uma-s-1p2p1", 4, tmp_path)
    with open(paths.npt_csv, "w") as fh:
        fh.write(VolumeRecorder.CSV_HEADER + "\n")
        for t, v, temp in zip(time, volume, temperature):
            fh.write(f"{t:.4f},{v:.4f},{temp:.2f},{density(MASS, v):.5f}\n")

    columns = load_volume_csv(paths.npt_csv)
    assert set(columns) == {"time_ps", "volume_A3", "temperature_K", "density_g_cm3"}

    result = analyse_run(False, "omol", "uma-s-1p2p1", 4, tmp_path, equil_ps=0.0)
    assert result.density == pytest.approx(density(MASS, 1800.0), rel=1e-4)


# --- run naming ------------------------------------------------------------


def test_default_run_is_untagged():
    """The historical filenames must keep working."""
    assert suffix(False, TASK, MODEL, 6) == ""
    assert stage_paths().npt_csv.name == "npt_volume.csv"


def test_every_distinguishing_choice_reaches_the_filename():
    tag = suffix(True, "omol", "uma-s-1p2p1", 4)
    assert tag == "_uma-s-1p2p1_omol_n4_atm"
    paths = stage_paths(True, "omol", "uma-s-1p2p1", 4)
    assert paths.npt_csv.name == "npt_volume_uma-s-1p2p1_omol_n4_atm.csv"
    assert paths.equilibrated.name == "water_equilibrated_uma-s-1p2p1_omol_n4_atm.traj"


def test_three_body_run_falls_back_to_the_shared_stage_one_input(tmp_path):
    """Stage 1 is shared across both arms, so the plain file is the input."""
    plain = stage_paths(False, "omol", "uma-s-1p2p1", 4, tmp_path).equilibrated
    plain.write_text("")
    assert equilibrated_input(True, "omol", "uma-s-1p2p1", 4, tmp_path) == plain


def test_barostat_defaults_to_monte_carlo_without_a_stress_tensor():
    """omol has no stress labels, so it cannot drive the MTK barostat."""
    assert resolve_barostat("auto", "omol") == "mc"
    assert resolve_barostat("auto", "omc") == "mtk"
    assert resolve_barostat("mc", "omc") == "mc"
    with pytest.raises(ValueError):
        resolve_barostat("berendsen", "omc")
