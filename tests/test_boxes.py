"""The starting configuration stage 1 builds, and the checks it runs on it."""

import numpy as np
import pytest

from omol_d4.boxes import (
    HARD_MIN_EDGE,
    SAFE_MIN_EDGE,
    box_length_for_density,
    build_water_box,
    check_box_size,
    contact_summary,
    n_molecules,
    random_rotation,
)
from omol_d4.constants import AMU_PER_A3_TO_G_PER_CM3, INIT_DENSITY


def test_box_is_built_at_the_requested_density():
    """The whole point of the builder: the cell realises the target density."""
    atoms = build_water_box(n_side=3, rng=np.random.default_rng(0))
    rho = atoms.get_masses().sum() / atoms.get_volume() * AMU_PER_A3_TO_G_PER_CM3
    # Not exact: ASE's tabulated masses differ slightly from the M_H2O the cell
    # size is derived from.
    assert rho == pytest.approx(INIT_DENSITY, rel=1e-3)


def test_box_has_the_requested_number_of_molecules():
    atoms = build_water_box(n_side=3, rng=np.random.default_rng(0))
    assert n_molecules(atoms) == 27
    assert len(atoms) == 81
    assert atoms.get_chemical_symbols()[:3] == ["O", "H", "H"]
    assert atoms.pbc.all()


def test_orientation_search_avoids_hard_overlaps():
    """Random orientations at liquid density give ~1 A contacts; these must not."""
    atoms = build_water_box(n_side=3, rng=np.random.default_rng(0), min_dist=1.75)
    assert min(contact_summary(atoms).values()) >= 1.7


def test_contact_summary_ignores_intramolecular_pairs():
    """The 0.96 A O-H bond is intramolecular and must not count as a contact."""
    atoms = build_water_box(n_side=3, rng=np.random.default_rng(0))
    assert contact_summary(atoms)["O-H"] > 1.5


def test_box_length_matches_the_density_it_was_asked_for():
    length = box_length_for_density(216, 0.997)
    assert length == pytest.approx(18.64, abs=0.01)


def test_random_rotation_is_a_proper_rotation():
    for seed in range(5):
        r = random_rotation(np.random.default_rng(seed))
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0)


def test_check_box_size_stops_on_a_degenerate_cell():
    atoms = build_water_box(n_side=3, rng=np.random.default_rng(0))
    atoms.set_cell([HARD_MIN_EDGE - 1] * 3, scale_atoms=True)
    with pytest.raises(SystemExit, match="too small"):
        check_box_size(atoms)


def test_check_box_size_only_warns_below_the_safety_margin(capsys):
    """A small box is a documented tradeoff, not an error."""
    atoms = build_water_box(n_side=3, rng=np.random.default_rng(0))
    edge = check_box_size(atoms)
    assert HARD_MIN_EDGE < edge <= SAFE_MIN_EDGE
    assert "WARNING" in capsys.readouterr().out


def test_check_box_size_is_quiet_for_a_large_box(capsys):
    atoms = build_water_box(n_side=6, rng=np.random.default_rng(0))
    check_box_size(atoms)
    assert capsys.readouterr().out == ""
