"""Shared fixtures for the Monte Carlo barostat tests.

The barostat is exercised against analytically solvable toy systems so that the
expected behaviour can be written down in closed form:

* ``ZeroCalculator``      -> ideal gas of molecules; P(V) is known exactly.
* ``IntramolecularSpring`` -> energy depends *only* on internal geometry, so a
  correct (centre-of-mass scaling) barostat must produce dE == 0 for every
  trial move.
* ``ScriptedRNG``         -> makes a single Monte Carlo move deterministic.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

N_ATOMS_PER_WATER = 3
OH_BOND = 0.9572


class ZeroCalculator(Calculator):
    """Non-interacting particles: zero energy, zero forces, for any geometry."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties or self.implemented_properties,
                          system_changes)
        self.results = {
            "energy": 0.0,
            "free_energy": 0.0,
            "forces": np.zeros((len(atoms), 3)),
        }


class IntramolecularSpring(Calculator):
    """Harmonic O-H bonds only; no intermolecular interaction at all.

    Atoms are assumed to be ordered O, H, H per molecule. Because the energy is
    invariant to rigid translation of whole molecules, a barostat that scales
    molecular centres of mass (as OpenMM's does) leaves the energy exactly
    unchanged by a volume move.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, k=10.0, r0=OH_BOND, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.r0 = r0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties or self.implemented_properties,
                          system_changes)
        pos = atoms.get_positions()
        forces = np.zeros_like(pos)
        energy = 0.0
        for o in range(0, len(atoms), N_ATOMS_PER_WATER):
            for h in (o + 1, o + 2):
                d = pos[h] - pos[o]
                r = np.linalg.norm(d)
                energy += 0.5 * self.k * (r - self.r0) ** 2
                f = self.k * (r - self.r0) * d / r
                forces[h] -= f
                forces[o] += f
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}


class ScriptedRNG:
    """RNG with deterministic Monte Carlo draws.

    The barostat calls ``uniform(-1, 1)`` to size the volume move and
    ``uniform()`` (no arguments) for the Metropolis draw, so the two are
    scripted independently. Langevin's Gaussian draws are zeroed out, which
    removes the thermostat noise and makes each test a pure barostat test.
    """

    def __init__(self, proposals=(), accept_draws=()):
        self.proposals = list(proposals)
        self.accept_draws = list(accept_draws)
        self.n_proposals_used = 0
        self.n_accept_draws_used = 0

    def standard_normal(self, size=None):
        return np.zeros(size) if size is not None else 0.0

    def uniform(self, low=None, high=None, size=None):
        if low is None and high is None:
            self.n_accept_draws_used += 1
            return self.accept_draws.pop(0)
        self.n_proposals_used += 1
        value = self.proposals.pop(0)
        assert low <= value <= high, f"scripted proposal {value} outside [{low}, {high}]"
        return value

    def random(self, size=None):
        return self.uniform()


def water_box(n_molecules=8, cell=12.0, calculator=None):
    """Cubic periodic box of ``n_molecules`` rigid-geometry waters on a grid."""
    # Built exactly at IntramolecularSpring's equilibrium bond length so the
    # spring exerts no force and the Langevin sub-step is a no-op: whatever the
    # geometry does during a test is then the barostat's doing alone.
    angle = np.radians(104.52)
    geometry = np.array([[0.0, 0.0, 0.0],
                         [OH_BOND, 0.0, 0.0],
                         [OH_BOND * np.cos(angle), OH_BOND * np.sin(angle), 0.0]])
    per_side = int(np.ceil(n_molecules ** (1 / 3)))
    spacing = cell / per_side
    positions = []
    placed = 0
    for i in range(per_side):
        for j in range(per_side):
            for k in range(per_side):
                if placed == n_molecules:
                    break
                origin = np.array([i, j, k]) * spacing + 0.5 * spacing
                positions.extend(geometry + origin)
                placed += 1
    atoms = Atoms("OHH" * n_molecules, positions=np.array(positions),
                  cell=[cell] * 3, pbc=True)
    atoms.calc = calculator if calculator is not None else ZeroCalculator()
    return atoms


@pytest.fixture
def waters():
    """Eight non-interacting waters (24 atoms) in a 12 A cubic box."""
    return water_box(n_molecules=8)
