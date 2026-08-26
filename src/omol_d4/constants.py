"""Physical constants, unit conversions, and the state point both MD stages share.

Everything here is a *default*, not a hard-wired setting: the stage functions in
`omol_d4.nvt` and `omol_d4.npt` take each of these as a keyword argument. They
live in one module so that stage 1 and stage 2 cannot drift apart on the
temperature, the timestep or the sampling interval, which would silently make
the two halves of a density measurement incomparable.
"""

from ase import units

# ---------------------------------------------------------------------------
# The state point
# ---------------------------------------------------------------------------
TEMPERATURE_K = 298.0
PRESSURE = 1.01325 * units.bar  # 1 atm, in eV/A^3
INIT_DENSITY = 0.997  # g/cm^3, experimental value at 298 K / 1 atm

# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------
TIMESTEP = 0.5 * units.fs  # 0.5 fs: the flexible O-H stretch is ~9 fs
FRICTION = 0.01 / units.fs  # strong-ish coupling; fine for equilibration
SEED = 0

# Soft start: the grid configuration stage 1 begins from is orientationally
# unrelaxed, so the first 0.25 ps runs with a short timestep and heavy friction
# to bleed off the initial strain without integrating anything unstable.
SOFT_START_STEPS = 1000
SOFT_START_TIMESTEP = 0.25 * units.fs
SOFT_START_FRICTION = 0.05 / units.fs

NVT_STEPS = 20_000  # 10 ps
NPT_STEPS = 60_000  # 30 ps
LOG_INTERVAL = 20  # write every 10 fs
SAMPLE_INTERVAL = 20  # sample every 10 fs

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
EQUIL_PS = 15.0  # discarded from the front of the average
N_BLOCKS = 5  # for the error estimate

# ---------------------------------------------------------------------------
# Geometry and unit conversions
# ---------------------------------------------------------------------------
N_ATOMS_PER_WATER = 3  # atoms are stored O, H, H per molecule
M_H2O = 18.01528  # g/mol
N_AVOGADRO = 6.02214076e23
AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660
EV_PER_A3_TO_BAR = 1.602176634e6
FS_PER_PS = 1000 * units.fs  # divide an ASE time by this to get ps

# ---------------------------------------------------------------------------
# Reference values, for the comparison lines in the reports
# ---------------------------------------------------------------------------
EXPERIMENTAL_DENSITY = 0.9970  # g/cm^3 at 298 K / 1 atm
EXPERIMENTAL_KAPPA_T = 45.2e-6  # 1/bar
