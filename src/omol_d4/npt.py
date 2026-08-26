"""Stage 2: NPT simulation of the equilibrated water box, at 298 K / 1 atm.

Two barostats are available (`barostat=`, default "auto"):

  mtk  Martyna-Tobias-Klein isotropic barostat (`ase.md.nose_hoover_chain`),
       which samples a true isothermal-isobaric ensemble from the calculator's
       stress tensor. Requires a periodic task (e.g. omc).
  mc   Monte Carlo barostat + Langevin thermostat (`omol_d4.integrator`), which
       accepts/rejects isotropic volume moves from energy differences alone and
       needs no stress. The only option for a task with no stress labels, e.g.
       omol (OMol25 is a non-periodic dataset).

"auto" picks mc for a non-periodic task and mtk otherwise. Both sample a true
NPT ensemble, so either way the isothermal compressibility can be read off the
run's volume fluctuations, which `omol_d4.analysis` relies on.

The volume trace is streamed to a CSV as it is produced, so a run that dies
partway is still analysable.
"""

from ase.io import Trajectory, read
from ase.md.nose_hoover_chain import IsotropicMTKNPT

from .boxes import n_molecules
from .calculators import (
    DISP3_CUTOFF,
    MODEL,
    PERIODIC_TASKS,
    TASK,
    build_calculator,
    require_stress,
)
from .constants import (
    AMU_PER_A3_TO_G_PER_CM3,
    FRICTION,
    FS_PER_PS,
    NPT_STEPS,
    PRESSURE,
    SAMPLE_INTERVAL,
    SEED,
    TEMPERATURE_K,
    TIMESTEP,
)
from .integrator import NPTLangevinMonteCarloBarostat
from .paths import equilibrated_input, stage_paths

BAROSTATS = ("auto", "mtk", "mc")


def resolve_barostat(barostat="auto", task=TASK):
    """Pick mtk (needs a stress tensor) or mc (energy-only)."""
    if barostat not in BAROSTATS:
        raise ValueError(f"barostat must be one of {BAROSTATS}, got {barostat!r}")
    if barostat == "auto":
        return "mc" if task not in PERIODIC_TASKS else "mtk"
    return barostat


def make_barostat(
    atoms,
    barostat="mtk",
    task=TASK,
    temperature_K=TEMPERATURE_K,
    pressure_au=PRESSURE,
    timestep=TIMESTEP,
    friction=FRICTION,
    mc_interval=25,
    mc_volume_scale=1.0,
):
    """Construct the requested barostat, already attached to `atoms`."""
    if barostat == "mtk":
        require_stress(atoms, task)
        dyn = IsotropicMTKNPT(
            atoms,
            timestep=timestep,
            temperature_K=temperature_K,
            pressure_au=pressure_au,
            tdamp=100 * timestep,  # thermostat time constant, 50 fs
            pdamp=1000 * timestep,  # barostat time constant, 500 fs
            tchain=3,
            pchain=3,
        )
        print("  Barostat: MTK (stress-driven)")
        return dyn

    # No stress needed: volume moves are accepted/rejected from energy
    # differences alone (Frenkel-Smit / OpenMM MonteCarloBarostat), which is
    # the only option for a task like omol that has none.
    dyn = NPTLangevinMonteCarloBarostat(
        atoms,
        timestep=timestep,
        temperature_K=temperature_K,
        pressure_au=pressure_au,
        friction=friction,
        bsinterval=mc_interval,
        volume_scale=mc_volume_scale,
    )
    print(
        f"  Barostat: Monte Carlo (energy-only, interval={mc_interval}, "
        f"volume_scale={dyn.volume_scale:.2f} A^3)"
    )
    return dyn


class VolumeRecorder:
    """Writes the volume trace to a human-readable log and a machine-readable CSV.

    A barostat run lives and dies by its volume, so the log needs it alongside
    energy and temperature - not just the CSV. Plain `MDLogger(stress=False)`
    never carries volume or density for either barostat, so the columns are
    written here instead of using it.

    For the MC barostat the cumulative acceptance ratio and current move size
    are logged too: a stuck (~0%) or saturated (~100%) acceptance ratio means
    the density trace cannot be trusted regardless of how long it ran.
    """

    HEADER = (
        f"{'Time[ps]':<10} {'Etot[eV]':>12} {'Epot[eV]':>12} "
        f"{'Ekin[eV]':>12} {'T[K]':>7} {'Volume[A^3]':>12} "
        f"{'Density[g/cm3]':>15}"
    )
    MC_HEADER = f" {'MCaccept[%]':>12} {'MCscale[A^3]':>13}"
    CSV_HEADER = "time_ps,volume_A3,temperature_K,density_g_cm3"
    MC_CSV_HEADER = ",mc_accept_pct,mc_volume_scale_A3"

    def __init__(self, dyn, atoms, log_path, csv_path, mc_diagnostics=False):
        self.dyn = dyn
        self.atoms = atoms
        self.mass = atoms.get_masses().sum()
        self.mc = mc_diagnostics
        self.log = open(log_path, "w")
        self.csv = open(csv_path, "w")
        self.log.write(self.HEADER + (self.MC_HEADER if self.mc else "") + "\n")
        self.csv.write(self.CSV_HEADER + (self.MC_CSV_HEADER if self.mc else "") + "\n")

    def __call__(self):
        atoms = self.atoms
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = atoms.get_temperature()
        volume = atoms.get_volume()
        density = self.mass / volume * AMU_PER_A3_TO_G_PER_CM3
        time_ps = self.dyn.get_time() / FS_PER_PS

        line = (
            f"{time_ps:<10.4f} {epot + ekin:12.4f} {epot:12.4f} "
            f"{ekin:12.4f} {temp:7.1f} {volume:12.4f} {density:15.5f}"
        )
        row = f"{time_ps:.4f},{volume:.4f},{temp:.2f},{density:.5f}"
        if self.mc:
            _, _, ratio = self.dyn.get_acceptance_ratio()
            line += f" {ratio * 100:12.1f} {self.dyn.volume_scale:13.3f}"
            row += f",{ratio * 100:.1f},{self.dyn.volume_scale:.3f}"
        self.log.write(line + "\n")
        self.log.flush()
        self.csv.write(row + "\n")
        self.csv.flush()

    def close(self):
        self.log.close()
        self.csv.close()


def load_equilibrated(
    three_body=False, task=TASK, model=MODEL, n_side=None, outdir="."
):
    """Read stage 1's final frame and make it safe to hand to a barostat."""
    atoms = read(str(equilibrated_input(three_body, task, model, n_side, outdir)))
    # IsotropicMTKNPT refuses to run with constraints attached, and a stage 1
    # constraint would have travelled here inside the .traj file.
    atoms.set_constraint()
    atoms.info.update({"charge": 0, "spin": 1})
    print(
        f"Loaded {n_molecules(atoms)} H2O ({len(atoms)} atoms), "
        f"L = {atoms.cell.lengths()[0]:.2f} A"
    )
    return atoms


def run_npt(
    steps=NPT_STEPS,
    three_body=False,
    model=MODEL,
    task=TASK,
    functional=None,
    disp3_cutoff=DISP3_CUTOFF,
    device="cuda",
    seed=SEED,
    n_side=None,
    barostat="auto",
    temperature_K=TEMPERATURE_K,
    pressure_au=PRESSURE,
    timestep=TIMESTEP,
    friction=FRICTION,
    sample_interval=SAMPLE_INTERVAL,
    mc_interval=25,
    mc_volume_scale=1.0,
    outdir=".",
    atoms=None,
    calc=None,
):
    """Run the NPT stage and return the `StagePaths` it wrote to."""
    paths = stage_paths(three_body, task, model, n_side, outdir)

    if atoms is None:
        atoms = load_equilibrated(three_body, task, model, n_side, outdir)
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

    kind = resolve_barostat(barostat, task)
    dyn = make_barostat(
        atoms,
        barostat=kind,
        task=task,
        temperature_K=temperature_K,
        pressure_au=pressure_au,
        timestep=timestep,
        friction=friction,
        mc_interval=mc_interval,
        mc_volume_scale=mc_volume_scale,
    )

    traj = Trajectory(str(paths.npt_traj), "w", atoms)
    dyn.attach(traj.write, interval=sample_interval)
    recorder = VolumeRecorder(
        dyn, atoms, paths.npt_log, paths.npt_csv, mc_diagnostics=kind == "mc"
    )
    dyn.attach(recorder, interval=sample_interval)

    print(
        f"Running {steps} steps ({steps * timestep / FS_PER_PS:.1f} ps) NPT "
        f"at {temperature_K} K / {pressure_au / PRESSURE:.3g} atm..."
    )
    try:
        dyn.run(steps=steps)
    finally:
        traj.close()
        recorder.close()
    return paths
