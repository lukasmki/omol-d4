"""Validation of the Monte Carlo barostat against analytic reference results.

Each test pins one property of OpenMM's ``MonteCarloBarostatImpl``, the
implementation that ``NPTLangevinMonteCarloBarostat`` is derived from.
"""

import numpy as np
import pytest
from ase import units

from omol_d4.integrator import NPTLangevinMonteCarloBarostat

from conftest import N_ATOMS_PER_WATER, ScriptedRNG, IntramolecularSpring, water_box

TEMPERATURE_K = 300.0
KT = units.kB * TEMPERATURE_K
TIMESTEP = 1.0 * units.fs


def make_dyn(atoms, rng, volume_scale, pressure_au=0.0, bsinterval=1):
    return NPTLangevinMonteCarloBarostat(
        atoms,
        TIMESTEP,
        pressure_au=pressure_au,
        bsinterval=bsinterval,
        volume_scale=volume_scale,
        temperature_K=TEMPERATURE_K,
        friction=0.01,
        rng=rng,
    )


def oh_distances(atoms):
    pos = atoms.get_positions()
    return np.array([
        np.linalg.norm(pos[h] - pos[o])
        for o in range(0, len(atoms), N_ATOMS_PER_WATER)
        for h in (o + 1, o + 2)
    ])


# --- the Metropolis criterion itself (expected to pass) --------------------

def test_downhill_move_is_accepted_without_a_metropolis_draw(waters):
    """dE + P dV - N kT ln(V'/V) < 0 must be accepted unconditionally."""
    v0 = waters.get_volume()
    # Expansion at zero pressure with zero energy => w = -N kT ln(V'/V) < 0.
    rng = ScriptedRNG(proposals=[1.0], accept_draws=[])
    dyn = make_dyn(waters, rng, volume_scale=100.0)
    dyn.run(2)

    assert waters.get_volume() == pytest.approx(v0 + 100.0)
    assert rng.n_accept_draws_used == 0, "downhill move should not consult the RNG"
    assert dyn.total_accepted == 1


def test_strongly_uphill_move_is_rejected(waters):
    """A move with an overwhelming P dV penalty must be rejected."""
    v0 = waters.get_volume()
    rng = ScriptedRNG(proposals=[1.0], accept_draws=[0.999999])
    dyn = make_dyn(waters, rng, volume_scale=100.0, pressure_au=1.0)  # 1 eV/A^3
    dyn.run(2)

    assert waters.get_volume() == pytest.approx(v0)
    assert dyn.total_accepted == 0


# --- defect 1: the ideal-gas term must count molecules, not atoms ----------

def test_acceptance_weight_counts_molecules_not_atoms(waters):
    """w must use N_molecules in -N kT ln(V'/V) (OpenMM: numMolecules).

    Eight waters (24 atoms), a -100 A^3 move from 1728 A^3 at zero pressure:

        w/kT = -N ln(1628/1728) = 0.4767 (N = 8 molecules)
                                = 1.4302 (N = 24 atoms)

    so the acceptance probability is 0.621 per molecule counting and 0.239 per
    atom counting. A draw of 0.4 separates the two.
    """
    n_molecules = len(waters) // N_ATOMS_PER_WATER
    v0 = waters.get_volume()
    dv = -100.0
    w_molecules = -n_molecules * KT * np.log((v0 + dv) / v0)
    w_atoms = -len(waters) * KT * np.log((v0 + dv) / v0)
    draw = 0.4
    assert np.exp(-w_atoms / KT) < draw < np.exp(-w_molecules / KT)

    rng = ScriptedRNG(proposals=[-1.0], accept_draws=[draw])
    dyn = make_dyn(waters, rng, volume_scale=abs(dv))
    dyn.run(2)

    assert waters.get_volume() == pytest.approx(v0 + dv), (
        "move rejected: the ideal-gas term is using the atom count"
    )
    assert dyn.total_accepted == 1


@pytest.mark.slow
def test_ideal_gas_mean_volume_matches_analytic_result():
    """For non-interacting molecules P(V) ~ V^N exp(-PV/kT), so <V> = (N+1)kT/P.

    Counting atoms instead of molecules inflates <V> by roughly a factor of
    (3N+1)/(N+1) = 2.8 for water.
    """
    n_molecules = 8
    target_volume = 1728.0
    pressure = (n_molecules + 1) * KT / target_volume

    atoms = water_box(n_molecules=n_molecules)
    dyn = NPTLangevinMonteCarloBarostat(
        atoms,
        TIMESTEP,
        pressure_au=pressure,
        bsinterval=1,
        volume_scale=0.05 * atoms.get_volume(),
        temperature_K=TEMPERATURE_K,
        friction=0.01,
        rng=np.random.default_rng(20260825),
    )

    volumes = []
    for _ in range(6000):
        dyn.run(1)
        volumes.append(atoms.get_volume())
    sampled = np.array(volumes[2000:])

    assert sampled.mean() == pytest.approx(target_volume, rel=0.15), (
        f"<V> = {sampled.mean():.0f} A^3, expected {target_volume:.0f} A^3 "
        f"(atom counting would give {(3 * n_molecules + 1) * KT / pressure:.0f})"
    )
    # Var(V)/<V>^2 = 1/(N+1) for the same distribution.
    assert (sampled.var() / sampled.mean() ** 2) == pytest.approx(
        1.0 / (n_molecules + 1), rel=0.35
    )


# --- defect 2: volume moves must not deform molecules ----------------------

def test_volume_move_preserves_intramolecular_geometry():
    """OpenMM scales molecular centres of mass; internal geometry is rigid.

    With an energy that depends only on O-H bond lengths, a correct volume move
    changes nothing about the energy.
    """
    atoms = water_box(n_molecules=8, calculator=IntramolecularSpring())
    v0 = atoms.get_volume()
    d0 = oh_distances(atoms)
    e0 = atoms.get_potential_energy()

    rng = ScriptedRNG(proposals=[1.0], accept_draws=[])  # expansion, always accepted
    dyn = make_dyn(atoms, rng, volume_scale=100.0)
    dyn.run(2)

    assert atoms.get_volume() == pytest.approx(v0 + 100.0), "move was not accepted"
    assert oh_distances(atoms) == pytest.approx(d0, rel=1e-9), (
        "O-H bonds were stretched by the volume move (set_cell(scale_atoms=True) "
        "scales every atom instead of molecular centres of mass)"
    )
    assert atoms.get_potential_energy() == pytest.approx(e0, abs=1e-9), (
        "a purely intramolecular energy changed under a volume move"
    )


# --- defect 3: adaptive volume_scale bookkeeping ---------------------------

def test_default_volume_scale_is_one_percent_of_the_volume(waters):
    """OpenMM initialises volumeScale to 0.01 * volume."""
    dyn = NPTLangevinMonteCarloBarostat(
        waters, TIMESTEP, temperature_K=TEMPERATURE_K, friction=0.01
    )
    assert dyn.volume_scale == pytest.approx(0.01 * waters.get_volume())


def test_counters_reset_every_ten_attempts_even_when_in_band(waters):
    """OpenMM resets the windowed counters every 10 attempts, unconditionally.

    Alternating accept/reject puts the window at 50%, inside the 25-75% band,
    so no rescaling happens -- but the window must still restart, otherwise the
    counters become a sticky long-run average that adapts ever more slowly.
    """
    # Alternate expansion (always accepted) and contraction (rejected by a
    # draw of 1.0), ten attempts total.
    proposals = [1.0, -1.0] * 5
    rng = ScriptedRNG(proposals=proposals, accept_draws=[1.0] * 5)
    dyn = make_dyn(waters, rng, volume_scale=10.0)
    dyn.run(11)

    assert dyn.total_attempted == 10
    assert dyn.total_accepted == 5, "test setup: expected a 50% acceptance window"
    assert dyn.volume_scale == pytest.approx(10.0), "50% acceptance should not rescale"
    assert (dyn.num_attempted, dyn.num_accepted) == (0, 0), (
        "windowed counters were not reset after 10 attempts"
    )


def test_volume_scale_shrinks_when_acceptance_is_low(waters):
    """Below 25% acceptance the move size must shrink (expected to pass)."""
    # Ten contraction attempts, all rejected.
    rng = ScriptedRNG(proposals=[-1.0] * 10, accept_draws=[1.0] * 10)
    dyn = make_dyn(waters, rng, volume_scale=10.0)
    dyn.run(11)

    assert dyn.total_accepted == 0
    assert dyn.volume_scale == pytest.approx(10.0 / 1.1)
    assert (dyn.num_attempted, dyn.num_accepted) == (0, 0)
