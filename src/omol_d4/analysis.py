"""Density, compressibility, and the perturbative three-body correction.

Reads the volume trace stage 2 streamed to CSV and turns it into the numbers the
runs exist to produce. Kept separate from `omol_d4.npt` so that a finished (or a
half-finished) run can be re-analysed without importing an MLIP or touching a
GPU.
"""

from dataclasses import dataclass, field

import numpy as np
from ase import units
from ase.io import Trajectory, read

from .boxes import n_molecules
from .calculators import DISP3_CUTOFF, MODEL, TASK, functional_for, make_d4_atm
from .constants import (
    AMU_PER_A3_TO_G_PER_CM3,
    EQUIL_PS,
    EV_PER_A3_TO_BAR,
    EXPERIMENTAL_DENSITY,
    EXPERIMENTAL_KAPPA_T,
    FS_PER_PS,
    N_BLOCKS,
    SAMPLE_INTERVAL,
    TEMPERATURE_K,
    TIMESTEP,
)
from .paths import equilibrated_input, stage_paths


def load_volume_csv(path):
    """Columns of a `npt_volume*.csv` as a dict of arrays."""
    data = np.genfromtxt(str(path), delimiter=",", names=True)
    return {name: data[name] for name in data.dtype.names}


def density(mass_amu, volume_A3):
    """g/cm^3 from a mass in amu and a volume in A^3."""
    return mass_amu / volume_A3 * AMU_PER_A3_TO_G_PER_CM3


def compressibility(volume, temperature_K=TEMPERATURE_K):
    """Isothermal compressibility in 1/bar from NPT volume fluctuations.

    kappa_T = <dV^2> / (kB T <V>). This is only meaningful because the barostat
    (MTK or the Monte Carlo one) samples the true NPT distribution; a Berendsen
    barostat would give a reasonable mean volume but meaningless fluctuations,
    and this number would be junk.
    """
    kappa = volume.var() / (units.kB * temperature_K * volume.mean())
    return kappa / EV_PER_A3_TO_BAR


@dataclass
class DensityResult:
    """What one NPT run says about the model's liquid water density."""

    n_samples: int
    time_range_ps: tuple
    temperature_mean: float
    temperature_std: float
    volume_mean: float
    volume_std: float
    density: float
    density_stderr: float
    kappa_T: float
    block_densities: np.ndarray = field(repr=False)
    first_half_density: float = 0.0
    second_half_density: float = 0.0

    @property
    def drifting(self):
        """True if the two halves disagree by much more than the error bar.

        Then the run is not equilibrated and the mean is not meaningful yet.
        """
        return (
            abs(self.first_half_density - self.second_half_density)
            > 3 * self.density_stderr
        )

    def report(self, note=""):
        """The same summary the original stage-2 script printed."""
        t0, t1 = self.time_range_ps
        lines = [
            f"\nProduction: {t0:.1f}-{t1:.1f} ps, {self.n_samples} samples{note}",
            f"  <T>        = {self.temperature_mean:.1f} "
            f"+/- {self.temperature_std:.1f} K",
            f"  <V>        = {self.volume_mean:.1f} +/- {self.volume_std:.1f} "
            f"A^3 (L = {self.volume_mean ** (1 / 3):.2f} A)",
            f"  density    = {self.density:.4f} +/- {self.density_stderr:.4f} g/cm^3",
            f"  experiment = {EXPERIMENTAL_DENSITY:.4f} g/cm^3 at 298 K / 1 atm",
            f"  kappa_T    = {self.kappa_T * 1e6:.1f} x 10^-6 /bar "
            f"(experiment {EXPERIMENTAL_KAPPA_T * 1e6:.1f})",
            f"  block means: {np.array2string(self.block_densities, precision=4)}",
            f"  first/second half: {self.first_half_density:.4f} / "
            f"{self.second_half_density:.4f} g/cm^3"
            f"{'  <-- still drifting, run longer' if self.drifting else ''}",
        ]
        return "\n".join(lines)


def analyse_volume(
    time_ps,
    volume,
    temperature,
    mass_amu,
    equil_ps=EQUIL_PS,
    n_blocks=N_BLOCKS,
    temperature_K=TEMPERATURE_K,
):
    """Density and compressibility from a volume trace, with a block error bar."""
    prod = time_ps >= equil_ps
    if prod.sum() < n_blocks * 10:
        raise SystemExit(
            f"Only {prod.sum()} production samples after {equil_ps} ps - "
            "run longer or lower equil_ps."
        )
    v, t = volume[prod], temperature[prod]

    # Report rho = M / <V>: the volume is the fluctuating quantity, so its mean
    # is the well-defined ensemble average. <M/V> differs only at second order.
    rho = density(mass_amu, v.mean())
    block_rho = np.array(
        [density(mass_amu, b.mean()) for b in np.array_split(v, n_blocks)]
    )
    stderr = block_rho.std(ddof=1) / np.sqrt(n_blocks)

    half = len(v) // 2
    return DensityResult(
        n_samples=len(v),
        time_range_ps=(time_ps[prod][0], time_ps[-1]),
        temperature_mean=t.mean(),
        temperature_std=t.std(),
        volume_mean=v.mean(),
        volume_std=v.std(),
        density=rho,
        density_stderr=stderr,
        kappa_T=compressibility(v, temperature_K),
        block_densities=block_rho,
        first_half_density=density(mass_amu, v[:half].mean()),
        second_half_density=density(mass_amu, v[half:].mean()),
    )


def system_mass(columns):
    """Recover the system mass in amu from a volume trace's own columns.

    Every row satisfies density = mass / volume * AMU_PER_A3_TO_G_PER_CM3, so
    the mass can be read back out of the CSV when the stage 1 trajectory it was
    produced from is not at hand (an archived run, a directory of CSVs). The
    columns are written rounded, which costs ~1e-5 relative accuracy - four
    orders of magnitude below the density error bar, but it is an inference
    rather than a measurement, so the caller is told.
    """
    mass = np.median(
        columns["density_g_cm3"] * columns["volume_A3"] / AMU_PER_A3_TO_G_PER_CM3
    )
    return float(mass)


def analyse_run(
    three_body=False,
    task=TASK,
    model=MODEL,
    n_side=None,
    outdir=".",
    equil_ps=EQUIL_PS,
    n_blocks=N_BLOCKS,
    temperature_K=TEMPERATURE_K,
    mass_amu=None,
):
    """Analyse one run's CSV, taking the system mass from its stage 1 input.

    Falls back to `system_mass` when that input is missing, so an archived CSV
    can still be re-analysed on its own.
    """
    paths = stage_paths(three_body, task, model, n_side, outdir)
    columns = load_volume_csv(paths.npt_csv)
    if mass_amu is None:
        source = equilibrated_input(three_body, task, model, n_side, outdir)
        if source.exists():
            mass_amu = read(str(source)).get_masses().sum()
        else:
            mass_amu = system_mass(columns)
            print(
                f"  {source} not found; taking the system mass "
                f"({mass_amu:.2f} amu) from the volume trace itself"
            )
    return analyse_volume(
        columns["time_ps"],
        columns["volume_A3"],
        columns["temperature_K"],
        mass_amu,
        equil_ps=equil_ps,
        n_blocks=n_blocks,
        temperature_K=temperature_K,
    )


@dataclass
class ATMCorrection:
    """Perturbative estimate of the three-body effect on the density."""

    energy_meV_per_molecule: float
    pressure_bar: float
    pressure_stderr_bar: float
    kappa_T: float
    density: float
    corrected_density: float
    n_frames: int

    @property
    def percent_change(self):
        return (self.corrected_density - self.density) / self.density * 100

    def report(self):
        return "\n".join(
            [
                f"\n  E_ATM      = {self.energy_meV_per_molecule:+.3f} meV/molecule",
                f"  P_ATM      = {self.pressure_bar:+.1f} "
                f"+/- {self.pressure_stderr_bar:.1f} bar",
                f"  kappa_T    = {self.kappa_T * 1e6:.1f} x 10^-6 /bar",
                f"  density    = {self.density:.4f} g/cm^3 (UMA only)",
                f"  corrected  = {self.corrected_density:.4f} g/cm^3 "
                f"({self.percent_change:+.2f} %, three-body estimate)",
                f"  experiment = {EXPERIMENTAL_DENSITY:.4f} g/cm^3",
            ]
        )


def atm_correction(
    task=TASK,
    model=MODEL,
    n_side=None,
    outdir=".",
    functional=None,
    disp3_cutoff=DISP3_CUTOFF,
    n_frames=25,
    equil_ps=EQUIL_PS,
    sample_interval=SAMPLE_INTERVAL,
    timestep=TIMESTEP,
    temperature_K=TEMPERATURE_K,
    n_blocks=N_BLOCKS,
):
    """Estimate the ATM effect on the density without rerunning the MD.

    Three-body forces are ~3 orders of magnitude smaller than the MLIP forces,
    so ATM barely perturbs the liquid structure; what it does contribute is a
    nearly uniform outward pressure. To linear order the box then expands by
    dV/V = kappa_T <P_ATM>, so

        rho_ATM ~ rho / (1 + kappa_T <P_ATM>)

    with both kappa_T and <P_ATM> taken from the existing UMA-only run. This
    costs a few dozen dispersion evaluations instead of 60,000, at the price of
    assuming the structural response is negligible. Validate it against a short
    explicit three-body run before relying on it.
    """
    # The plain (non-three-body) run at the same model/task/size is the input.
    result = analyse_run(
        False,
        task,
        model,
        n_side,
        outdir,
        equil_ps=equil_ps,
        n_blocks=n_blocks,
        temperature_K=temperature_K,
    )
    print(result.report())

    paths = stage_paths(False, task, model, n_side, outdir)
    frames = Trajectory(str(paths.npt_traj))
    n_equil = int(equil_ps * FS_PER_PS / (sample_interval * timestep))
    if n_equil >= len(frames) - 1:
        raise SystemExit("Trajectory is shorter than the equilibration window.")
    indices = np.linspace(n_equil, len(frames) - 1, n_frames, dtype=int)
    print(f"\nEvaluating D4 three-body on {len(indices)} frames (of {len(frames)})...")

    d4 = make_d4_atm(functional or functional_for(task), disp3_cutoff=disp3_cutoff)
    pressures, energies = [], []
    for count, i in enumerate(indices, 1):
        atoms = frames[int(i)]
        atoms.calc = d4
        pressures.append(-atoms.get_stress()[:3].mean() * EV_PER_A3_TO_BAR)
        energies.append(atoms.get_potential_energy() / n_molecules(atoms) * 1000)
        print(f"\r  frame {count}/{len(indices)}", end="", flush=True)
    print()

    p_atm = np.array(pressures)
    p_mean = p_atm.mean()
    return ATMCorrection(
        energy_meV_per_molecule=float(np.mean(energies)),
        pressure_bar=p_mean,
        pressure_stderr_bar=p_atm.std(ddof=1) / np.sqrt(len(p_atm)),
        kappa_T=result.kappa_T,
        density=result.density,
        corrected_density=result.density / (1.0 + result.kappa_T * p_mean),
        n_frames=len(indices),
    )
