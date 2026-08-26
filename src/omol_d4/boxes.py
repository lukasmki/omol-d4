"""Construction and sanity-checking of the periodic water box.

Stage 1 starts from a cubic grid of randomly oriented water molecules at the
experimental density and melts it; this module builds that starting point and
provides the two checks worth running before committing a GPU to it, namely
that there are no atomic overlaps and that the cell is large enough for a
periodic liquid.
"""

import itertools

import numpy as np
from ase import Atoms
from ase.build import molecule

from .constants import INIT_DENSITY, M_H2O, N_ATOMS_PER_WATER, N_AVOGADRO

# The cell edge should comfortably exceed twice the model cutoff (~6 A) so an
# atom never interacts with two periodic images of the same neighbour, and a D4
# three-body cutoff close to L/2 sums over more of its own periodic images than
# a well-separated box would. Below SAFE_MIN_EDGE this is a deliberate,
# documented tradeoff (e.g. a small box to afford three-body MD), not an error,
# so it warns rather than aborts. Below HARD_MIN_EDGE something is very likely
# wrong (an empty or degenerate cell) and it stops.
SAFE_MIN_EDGE = 14.0
HARD_MIN_EDGE = 8.0

DEFAULT_N_SIDE = 6      # n^3 molecules: 6 -> 216 H2O (648 atoms), L ~ 18.6 A


def box_length_for_density(n_molecules, density_g_cm3=INIT_DENSITY):
    """Edge in A of a cubic cell holding `n_molecules` H2O at that density."""
    volume = n_molecules * M_H2O / (density_g_cm3 * N_AVOGADRO) * 1e24
    return volume ** (1 / 3)


def random_rotation(rng):
    """Uniform random rotation matrix from a random unit quaternion."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def build_water_box(n_side=DEFAULT_N_SIDE, density_g_cm3=INIT_DENSITY, rng=None,
                    min_dist=1.75, n_tries=60):
    """Water molecules on a cubic grid, each given a random orientation.

    Purely random orientations at liquid density produce ~1 A H...H contacts,
    which give enormous forces on the first step. So for each site we sample
    several orientations and keep the one that maximises the closest contact
    with the molecules placed so far, stopping early once `min_dist` is met.

    The result is not a physical liquid configuration, but it is an
    overlap-free starting point that melts into one within a few ps of NVT.
    """
    rng = np.random.default_rng() if rng is None else rng

    n_mol = n_side**3
    length = box_length_for_density(n_mol, density_g_cm3)
    spacing = length / n_side

    template = molecule("H2O")
    template.positions -= template.get_center_of_mass()
    ref = template.positions.copy()

    placed = np.empty((0, 3))
    all_pos = []
    for i, j, k in itertools.product(range(n_side), repeat=3):
        centre = (np.array([i, j, k]) + 0.5) * spacing
        centre = centre + rng.normal(scale=0.1, size=3)

        best, best_gap = None, -np.inf
        for _ in range(n_tries):
            pos = ref @ random_rotation(rng).T + centre
            if len(placed) == 0:
                best, best_gap = pos, np.inf
                break
            delta = pos[:, None, :] - placed[None, :, :]
            delta -= np.round(delta / length) * length      # minimum image
            gap = np.sqrt((delta**2).sum(-1)).min()
            if gap > best_gap:
                best, best_gap = pos, gap
            if best_gap >= min_dist:
                break
        all_pos.append(best)
        placed = np.vstack([placed, best])

    box = Atoms(
        "OHH" * n_mol,
        positions=np.vstack(all_pos),
        cell=[length, length, length],
        pbc=True,
    )
    box.wrap()
    return box


def n_molecules(atoms):
    """Number of water molecules in a box of O,H,H triples."""
    return len(atoms) // N_ATOMS_PER_WATER


def contact_summary(atoms):
    """Closest intermolecular O-O, O-H and H-H distance, in Angstrom."""
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    sym = np.array(atoms.get_chemical_symbols())
    o, h = sym == "O", sym == "H"
    # mask out intramolecular pairs (atoms are stored as O,H,H per molecule)
    mol_id = np.arange(len(atoms)) // N_ATOMS_PER_WATER
    inter = np.where(mol_id[:, None] == mol_id[None, :], np.inf, d)
    return {
        "O-O": inter[np.ix_(o, o)].min(),
        "O-H": inter[np.ix_(o, h)].min(),
        "H-H": inter[np.ix_(h, h)].min(),
    }


def report_contacts(atoms):
    """Cheap sanity check that the initial packing has no bad overlaps."""
    for pair, distance in contact_summary(atoms).items():
        print(f"  min intermolecular {pair} : {distance:.2f} A")


def check_box_size(atoms, safe_min_edge=SAFE_MIN_EDGE,
                   hard_min_edge=HARD_MIN_EDGE):
    """Stop on a degenerate cell; warn on a small but deliberate one."""
    min_edge = atoms.cell.lengths().min()
    if min_edge <= hard_min_edge:
        raise SystemExit(
            f"Box edge {min_edge:.2f} A is too small to be a sane periodic "
            f"liquid (need > {hard_min_edge:.1f} A)."
        )
    if min_edge <= safe_min_edge:
        print(f"  WARNING: box edge {min_edge:.2f} A is below the "
              f"~{safe_min_edge:.0f} A (2x cutoff) safety margin; the MLIP and "
              "any three-body D4 term will see extra periodic images of "
              "themselves. Treat results at this size as a deliberate "
              "finite-size tradeoff, not directly comparable to a larger box.")
    return min_edge
