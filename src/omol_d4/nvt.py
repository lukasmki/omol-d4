"""Stage 1: build a periodic water box and equilibrate it in the NVT ensemble.

The box is built at the experimental density of liquid water (0.997 g/cm^3 at
300 K) and held at fixed volume while the structure relaxes from the artificial
lattice-like starting configuration into a proper liquid. Stage 2 (`omol_d4.npt`)
then lets the volume float so the *model's* density can be measured.

`run_nvt` is the whole stage end to end; `equilibrate` is the MD part alone, for
callers that have already built and attached a calculator to their own system.
"""

from ase.io import Trajectory, write
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, ZeroRotation

try:  # renamed in ASE 3.28
    from ase.md.velocitydistribution import thermalize_momenta
except ImportError:  # pragma: no cover
    from ase.md.velocitydistribution import (
        MaxwellBoltzmannDistribution as thermalize_momenta,
    )

import numpy as np

from .boxes import (
    DEFAULT_N_SIDE,
    build_water_box,
    check_box_size,
    n_molecules,
    report_contacts,
)
from .calculators import DISP3_CUTOFF, MODEL, TASK, build_calculator
from .constants import (
    FRICTION,
    FS_PER_PS,
    INIT_DENSITY,
    LOG_INTERVAL,
    NVT_STEPS,
    SEED,
    SOFT_START_FRICTION,
    SOFT_START_STEPS,
    SOFT_START_TIMESTEP,
    TEMPERATURE_K,
    TIMESTEP,
)
from .paths import stage_paths


def thermalize(atoms, temperature_K=TEMPERATURE_K, rng=None):
    """Draw Maxwell-Boltzmann momenta, then remove net translation/rotation."""
    thermalize_momenta(atoms, temperature_K=temperature_K, rng=rng)
    Stationary(atoms)  # zero net linear momentum
    ZeroRotation(atoms)  # harmless under PBC, keeps the start clean
    return atoms


def soft_start(
    atoms,
    steps=SOFT_START_STEPS,
    temperature_K=TEMPERATURE_K,
    timestep=SOFT_START_TIMESTEP,
    friction=SOFT_START_FRICTION,
    rng=None,
):
    """Short, heavily damped Langevin run that bleeds off the initial strain.

    The grid configuration is orientationally unrelaxed, so the first 0.25 ps
    takes a short timestep and heavy friction rather than integrating something
    unstable at the production settings.
    """
    if steps <= 0:
        return atoms
    Langevin(
        atoms,
        timestep=timestep,
        temperature_K=temperature_K,
        friction=friction,
        rng=rng,
    ).run(steps=steps)
    print(
        "Soft start done; E_pot = "
        f"{atoms.get_potential_energy() / n_molecules(atoms):.4f} eV/molecule"
    )
    return atoms


def equilibrate(
    atoms,
    steps=NVT_STEPS,
    temperature_K=TEMPERATURE_K,
    timestep=TIMESTEP,
    friction=FRICTION,
    log_interval=LOG_INTERVAL,
    rng=None,
    traj_path=None,
    log_path=None,
):
    """Run `steps` of NVT Langevin dynamics, writing a trajectory and a log."""
    dyn = Langevin(
        atoms,
        timestep=timestep,
        temperature_K=temperature_K,
        friction=friction,
        rng=rng,
    )

    traj = Trajectory(str(traj_path), "w", atoms) if traj_path else None
    if traj is not None:
        dyn.attach(traj.write, interval=log_interval)
    if log_path is not None:
        dyn.attach(
            MDLogger(dyn, atoms, str(log_path), header=True, stress=False, mode="w"),
            interval=log_interval,
        )

    print(f"Running {steps} steps ({steps * timestep / FS_PER_PS:.1f} ps) NVT...")
    try:
        dyn.run(steps=steps)
    finally:
        if traj is not None:
            traj.close()
    return dyn


def run_nvt(
    n_side=DEFAULT_N_SIDE,
    steps=NVT_STEPS,
    three_body=False,
    model=MODEL,
    task=TASK,
    functional=None,
    disp3_cutoff=DISP3_CUTOFF,
    device="cuda",
    seed=SEED,
    temperature_K=TEMPERATURE_K,
    timestep=TIMESTEP,
    friction=FRICTION,
    log_interval=LOG_INTERVAL,
    density_g_cm3=INIT_DENSITY,
    soft_start_steps=SOFT_START_STEPS,
    outdir=".",
    calc=None,
):
    """Build a water box, equilibrate it at fixed volume, and write it out.

    Returns the equilibrated `Atoms`. The final frame is written as a .traj so
    that momenta survive and stage 2 can continue from a hot, equilibrated box
    rather than re-thermalising a cold one.
    """
    paths = stage_paths(three_body, task, model, n_side, outdir)
    rng = np.random.default_rng(seed)

    atoms = build_water_box(n_side=n_side, density_g_cm3=density_g_cm3, rng=rng)
    atoms.info.update({"charge": 0, "spin": 1})  # ignored by non-omol tasks
    print(
        f"Built {n_molecules(atoms)} H2O ({len(atoms)} atoms) in a "
        f"{atoms.cell.lengths()[0]:.2f} A cube at {density_g_cm3} g/cm^3"
    )
    report_contacts(atoms)
    check_box_size(atoms)

    atoms.calc = (
        calc
        if calc is not None
        else build_calculator(
            device=device,
            seed=seed,
            three_body=three_body,
            model=model,
            task=task,
            functional=functional,
            disp3_cutoff=disp3_cutoff,
            atoms=atoms,
        )
    )

    thermalize(atoms, temperature_K=temperature_K, rng=rng)
    soft_start(atoms, steps=soft_start_steps, temperature_K=temperature_K, rng=rng)
    equilibrate(
        atoms,
        steps=steps,
        temperature_K=temperature_K,
        timestep=timestep,
        friction=friction,
        log_interval=log_interval,
        rng=rng,
        traj_path=paths.nvt_traj,
        log_path=paths.nvt_log,
    )

    write(str(paths.equilibrated), atoms)
    print(f"Wrote {paths.equilibrated}")
    return atoms
