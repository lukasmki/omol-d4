"""Validation of the Monte Carlo barostat against analytic reference results.

Each test pins one property of OpenMM's ``MonteCarloBarostatImpl`` /
``ReferenceMonteCarloBarostat``, the implementation this is derived from.
"""

import numpy as np
import pytest
from ase import units

from omol_d4.integrator import NPTLangevinMonteCarloBarostat

from conftest import N_ATOMS_PER_WATER, ScriptedRNG, water_box

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
    return np.array(
        [
            np.linalg.norm(pos[h] - pos[o])
            for o in range(0, len(atoms), N_ATOMS_PER_WATER)
            for h in (o + 1, o + 2)
        ]
    )


# --- the Metropolis criterion itself ---------------------------------------


def test_downhill_move_is_accepted_without_a_metropolis_draw(waters):
    """dE + P dV - N kT ln(V'/V) < 0 must be accepted unconditionally."""
    v0 = waters.get_volume()
    # Expansion at zero pressure with zero energy => w = -N kT ln(V'/V) < 0.
    rng = ScriptedRNG(proposals=[1.0], accept_draws=[])
    dyn = make_dyn(waters, rng, volume_scale=100.0)
    dyn.run(1)

    assert waters.get_volume() == pytest.approx(v0 + 100.0)
    assert rng.n_accept_draws_used == 0, "downhill move should not consult the RNG"
    assert dyn.total_accepted == 1


def test_strongly_uphill_move_is_rejected(waters):
    """A move with an overwhelming P dV penalty must be rejected."""
    v0 = waters.get_volume()
    rng = ScriptedRNG(proposals=[1.0], accept_draws=[0.999999])
    dyn = make_dyn(waters, rng, volume_scale=100.0, pressure_au=1.0)  # 1 eV/A^3
    dyn.run(1)

    assert waters.get_volume() == pytest.approx(v0)
    assert dyn.total_accepted == 0


def test_rejected_move_restores_coordinates_exactly(waters):
    """OpenMM saves and restores coordinates; the round trip must be bit-exact.

    With zero forces, zero velocities and scripted (noiseless) draws, the
    Langevin sub-step is the identity, so any change in the positions after a
    rejected move is the barostat failing to restore them.
    """
    cell0 = waters.get_cell().array.copy()
    positions0 = waters.get_positions()

    rng = ScriptedRNG(proposals=[1.0], accept_draws=[0.999999])
    dyn = make_dyn(waters, rng, volume_scale=100.0, pressure_au=1.0)  # 1 eV/A^3
    dyn.run(1)

    assert dyn.total_attempted == 1 and dyn.total_accepted == 0
    assert np.array_equal(waters.get_cell().array, cell0)
    assert np.array_equal(waters.get_positions(), positions0)


# --- the acceptance weight counts atoms ------------------------------------


def test_acceptance_weight_counts_atoms(waters):
    """Atoms are scaled individually, so w uses N_atoms (getNumParticles()).

    Eight waters (24 atoms), a -100 A^3 move from 1728 A^3 at zero pressure:

        w/kT = -N ln(1628/1728) = 1.4302 (N = 24 atoms)
                                = 0.4767 (N = 8 molecules)

    so the acceptance probability is 0.239 counting atoms and 0.621 counting
    molecules. A draw of 0.4 separates the two: it must reject.
    """
    v0 = waters.get_volume()
    dv = -100.0
    w_atoms = -len(waters) * KT * np.log((v0 + dv) / v0)
    w_molecules = -(len(waters) // N_ATOMS_PER_WATER) * KT * np.log((v0 + dv) / v0)
    draw = 0.4
    assert np.exp(-w_atoms / KT) < draw < np.exp(-w_molecules / KT)

    rng = ScriptedRNG(proposals=[-1.0], accept_draws=[draw])
    dyn = make_dyn(waters, rng, volume_scale=abs(dv))
    assert dyn.nscaled == len(waters)
    dyn.run(1)

    assert waters.get_volume() == pytest.approx(v0), (
        "move accepted: the ideal-gas term is not using the atom count"
    )


@pytest.mark.slow
def test_ideal_gas_mean_volume_matches_analytic_result():
    """For non-interacting particles P(V) ~ V^N exp(-PV/kT), so <V> = (N+1)kT/P.

    Scaling every atom and counting every atom is internally consistent, so the
    sampled volume distribution is exact for N = N_atoms. Counting molecules
    while scaling atoms (or the reverse) would sample neither distribution.
    """
    n_molecules = 8
    atoms = water_box(n_molecules=n_molecules)
    n_scaled = len(atoms)

    target_volume = 1728.0
    pressure = (n_scaled + 1) * KT / target_volume

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
        f"(counting molecules would give {(n_molecules + 1) * KT / pressure:.0f})"
    )
    # Var(V)/<V>^2 = 1/(N+1) for the same distribution.
    assert (sampled.var() / sampled.mean() ** 2) == pytest.approx(
        1.0 / (n_scaled + 1), rel=0.35
    )


# --- coordinate scaling ----------------------------------------------------


def test_volume_move_scales_every_atom(waters):
    """A move must map x -> length_scale * x, as set_cell(scale_atoms=True) does."""
    positions0 = waters.get_positions()
    v0 = waters.get_volume()
    rng = ScriptedRNG(proposals=[1.0], accept_draws=[])
    dyn = make_dyn(waters, rng, volume_scale=100.0)
    dyn.run(1)

    length_scale = ((v0 + 100.0) / v0) ** (1 / 3)
    assert waters.get_volume() == pytest.approx(v0 + 100.0), "move was not accepted"
    assert waters.get_positions() == pytest.approx(length_scale * positions0, rel=1e-9)


def test_volume_move_strains_intramolecular_geometry(waters):
    """Scaling atoms individually stretches bonds -- inherent to this mode.

    Pinned so the behaviour is a recorded property of the barostat rather than
    a surprise: every trial move is charged for deforming the molecules, which
    depresses acceptance and mixes intramolecular stiffness into dE.
    """
    d0 = oh_distances(waters)
    v0 = waters.get_volume()
    rng = ScriptedRNG(proposals=[1.0], accept_draws=[])
    dyn = make_dyn(waters, rng, volume_scale=100.0)
    dyn.run(1)

    length_scale = ((v0 + 100.0) / v0) ** (1 / 3)
    assert oh_distances(waters) == pytest.approx(length_scale * d0, rel=1e-9)


# --- adaptive volume_scale bookkeeping -------------------------------------


def test_default_volume_scale_is_one_percent_of_the_volume(waters):
    """OpenMM initialises volumeScale to 0.01 * volume."""
    dyn = NPTLangevinMonteCarloBarostat(
        waters, TIMESTEP, temperature_K=TEMPERATURE_K, friction=0.01
    )
    assert dyn.volume_scale == pytest.approx(0.01 * waters.get_volume())


def test_volume_scale_beyond_the_thirty_percent_cap_is_rejected(waters):
    """The adaptive cap is volume*0.3; a larger seed could propose V' <= 0."""
    with pytest.raises(ValueError, match="volume_scale"):
        make_dyn(waters, ScriptedRNG(), volume_scale=0.31 * waters.get_volume())


def test_volume_scale_shrinks_when_acceptance_is_low(waters):
    """Below 25% acceptance the move size shrinks and the window restarts."""
    # Ten contraction attempts, all rejected.
    rng = ScriptedRNG(proposals=[-1.0] * 10, accept_draws=[1.0] * 10)
    dyn = make_dyn(waters, rng, volume_scale=10.0)
    dyn.run(10)

    assert dyn.total_accepted == 0
    assert dyn.volume_scale == pytest.approx(10.0 / 1.1)
    assert (dyn.num_attempted, dyn.num_accepted) == (0, 0)


def test_volume_scale_grows_when_acceptance_is_high(waters):
    """Above 75% acceptance the move size grows, capped at 30% of the volume."""
    rng = ScriptedRNG(proposals=[1.0] * 10, accept_draws=[])
    dyn = make_dyn(waters, rng, volume_scale=10.0)
    dyn.run(10)

    assert dyn.total_accepted == 10
    assert dyn.volume_scale == pytest.approx(10.0 * 1.1)
    assert (dyn.num_attempted, dyn.num_accepted) == (0, 0)


def test_in_band_window_keeps_accumulating(waters):
    """OpenMM restarts the window only when it rescales; 25-75% keeps counting.

    Alternating accept/reject puts the window at 50%, so no rescaling happens
    and the counters carry over -- matching MonteCarloBarostatImpl, where
    numAttempted/numAccepted are zeroed only inside the two branches.
    """
    proposals = [1.0, -1.0] * 5
    rng = ScriptedRNG(proposals=proposals, accept_draws=[1.0] * 5)
    dyn = make_dyn(waters, rng, volume_scale=10.0)
    dyn.run(10)

    assert (dyn.total_attempted, dyn.total_accepted) == (10, 5)
    assert dyn.volume_scale == pytest.approx(10.0), "50% acceptance must not rescale"
    assert (dyn.num_attempted, dyn.num_accepted) == (10, 5)


# --- move frequency --------------------------------------------------------


def test_barostat_fires_every_bsinterval_steps(waters):
    """OpenMM attempts a move on every `frequency`-th call, starting at the first."""
    rng = ScriptedRNG(proposals=[1.0] * 4, accept_draws=[])
    dyn = make_dyn(waters, rng, volume_scale=10.0, bsinterval=25)

    dyn.run(24)  # ASE runs 24 further steps, i.e. up to step 24
    assert dyn.total_attempted == 0
    dyn.run(1)  # step 25
    assert dyn.total_attempted == 1
    dyn.run(24)  # up to step 49
    assert dyn.total_attempted == 1
    dyn.run(1)  # step 50
    assert dyn.total_attempted == 2


def test_zero_bsinterval_disables_the_barostat(waters):
    """OpenMM treats frequency == 0 as 'never'."""
    v0 = waters.get_volume()
    rng = ScriptedRNG(proposals=[], accept_draws=[])
    dyn = make_dyn(waters, rng, volume_scale=10.0, bsinterval=0)
    dyn.run(50)

    assert dyn.total_attempted == 0
    assert waters.get_volume() == pytest.approx(v0)
