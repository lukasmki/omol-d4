"""
Calculator construction shared by the NVT and NPT stages.

The baseline is the UMA MLIP; the task is selectable. Optionally a DFT-D4
calculator is summed in that contributes *only* the Axilrod-Teller-Muto (ATM)
three-body dispersion energy, with both two-body scalings zeroed.

Choosing a task also chooses which D4 damping parameters are consistent with
it, because a1/a2 are fitted per functional and the ATM zero damping is built
on the same critical radii:

    omc   OMC25, PBE-D3               -> pbe    (periodic, has stress)
    omat  OMat24, PBE(+U)             -> pbe    (periodic, has stress)
    omol  OMol25, wB97M-V/def2-TZVPD  -> wb97m  (NOT periodic, no stress)

Two caveats specific to omol / wB97M:

  * OMol25 is a non-periodic molecular dataset with no stress labels. The omol
    task therefore cannot drive a barostat, and stage 2 will stop with a clear
    message rather than silently integrating a bogus cell. Use omol for
    clusters and single molecules; use omc for the periodic liquid.
  * wB97M-V already contains dispersion through the VV10 non-local correlation
    functional, which is not three-body but is not nothing either: because it
    is built from the density, the effective C6 already respond to their
    environment. The `wb97m` D4 parameters in the dftd4 table come from the
    wB97M-D4 fit, where D4 *replaces* VV10 rather than supplementing it, so
    a1/a2 were optimised in a different context than the one they are used in
    here. They are the best available proxy, not a consistent choice, and they
    are not gentle: on a 216-molecule water box the ATM pressure is 620 bar
    with wB97M damping against 404 bar with PBE damping.

For omc, the argument is cleaner: OMC25 is PBE-D3 and D3 as commonly applied
is two-body only, so the model inherits pairwise dispersion but no ATM term
and adding one is a correction rather than double counting. In both cases the
MLIP's ~6 A cutoff already absorbs some short-range many-body character from
the reference data, so the genuinely new physics is the long-range tail.

The density is measured from the stage 2 ensemble, so that is the run whose
Hamiltonian defines the answer. Stage 1 only prepares a starting configuration,
and the ATM forces are tiny (~3e-4 eV/A, against ~1 eV/A from the MLIP), so
equilibrating without ATM and then sampling with it is defensible and saves a
lot of time. The flag is available on both stages for consistency; if you use
it on only one, use it on stage 2.
"""

import os
import re

from ase.calculators.mixing import SumCalculator
from ase.units import Bohr

from .boxes import DEFAULT_N_SIDE

MODEL = "uma-s-1p2"
TASK = "omc"

# Known UMA checkpoints, for the --model help text and a clearer error. Not
# authoritative: fairchem's pretrained_mlip.available_models is, and new
# checkpoints appear without this list changing.
KNOWN_MODELS = ("uma-s-1p1", "uma-s-1p2", "uma-m-1p1")

# Reference level of theory behind each UMA task -> consistent D4 damping.
TASK_FUNCTIONAL = {
    "omc": "pbe",
    "omat": "pbe",
    "omol": "wb97m",
    "odac": "pbe",
    "oc20": "pbe",
}

# Tasks trained on periodic data, i.e. the ones that can supply a stress.
PERIODIC_TASKS = {"omc", "omat", "odac", "oc20"}

# Fallbacks if the parameter table cannot be read (bj-eeq-atm entries).
FALLBACK_DAMPING = {
    "pbe": {"a1": 0.38574991, "a2": 4.80688534},
    "wb97m": {"a1": 0.7514, "a2": 2.7099},
}

# ATM converges slowly with the real-space cutoff. On a 216-molecule box the
# three-body pressure (PBE damping) is 235 bar at 6 A, 374 at 8 A, 404 at 12 A
# and 410 at 16 A, while the cost scales steeply, so 12 A is a compromise.
DISP3_CUTOFF = 12.0
DISP2_CUTOFF = 10.0   # irrelevant here: s6 = s8 = 0 means no two-body sum
CN_CUTOFF = 15.0      # still matters: C9 derives from CN-dependent C6


def functional_for(task):
    """D4 damping parameter set matching a task's reference level of theory."""
    try:
        return TASK_FUNCTIONAL[task]
    except KeyError:
        raise SystemExit(
            f"No reference functional known for task '{task}'. Pass "
            "--functional explicitly, and check it against the dataset paper."
        ) from None


def d4_damping(functional):
    """Read a1/a2 for `functional` from the table shipped with dftd4."""
    try:
        try:
            import tomllib
        except ModuleNotFoundError:      # Python < 3.11
            import tomli as tomllib

        import dftd4

        path = os.path.join(os.path.dirname(dftd4.__file__), "parameters.toml")
        with open(path, "rb") as fh:
            entry = tomllib.load(fh)["parameter"][functional.lower()]["d4"]
        return {"a1": entry["bj-eeq-atm"]["a1"], "a2": entry["bj-eeq-atm"]["a2"]}
    except Exception as exc:  # noqa: BLE001 - fall back rather than die
        fallback = FALLBACK_DAMPING.get(functional.lower())
        if fallback is None:
            raise RuntimeError(
                f"Could not read D4 damping parameters for {functional}"
            ) from exc
        print(f"  (using hard-coded {functional.upper()} damping: {exc})")
        return dict(fallback)


def make_d4_atm(functional="pbe", disp3_cutoff=DISP3_CUTOFF, quiet=False):
    """DFT-D4 calculator returning only the three-body (ATM) dispersion.

    `method` is deliberately not set: when a method name is given, the tabulated
    damping parameters are used and the s6/s8 tweaks below are ignored, which
    would silently give the full two-body energy. a1/a2 are still required
    because the ATM zero damping is built on the same critical radii.

    `disp3_cutoff=None` leaves dftd4's own real-space cutoffs in place (disp3 =
    40 Bohr = 21.2 A), which is what the ATM values quoted in this module's
    docstring converge towards; passing a number truncates the triple sum there
    instead. That is worth doing: on a 216-molecule box the sum at 12 A is ~20x
    cheaper than the default and reproduces its pressure to within a bar, and
    an on-the-fly three-body MD run is dominated by exactly this cost.
    """
    from dftd4.ase import DFTD4

    damping = d4_damping(functional)
    params = {"s6": 0.0, "s8": 0.0, "s9": 1.0, "alp": 16.0, **damping}
    if not quiet:
        print(f"  D4 ATM-only: s6=0, s8=0, s9=1, "
              f"a1={params['a1']:.4f}, a2={params['a2']:.4f} "
              f"({functional.upper()}), disp3 cutoff "
              f"{'library default' if disp3_cutoff is None else f'{disp3_cutoff:.1f} A'}")

    return DFTD4(
        params_tweaks=params,
        realspace_cutoff=realspace_cutoff(disp3_cutoff),
    )


def realspace_cutoff(disp3_cutoff=DISP3_CUTOFF):
    """`realspace_cutoff` kwargs for DFTD4; empty dict keeps dftd4's defaults."""
    if disp3_cutoff is None:
        return {}
    return {
        "disp2": DISP2_CUTOFF,
        "disp3": disp3_cutoff,
        "cn": CN_CUTOFF,
        # Smooth the cutoffs: with the two-body term switched off there is
        # no large smooth background to hide a discontinuity in the forces.
        "width2": 0.05 * Bohr,
        "width3": 0.05 * Bohr,
    }


class SumWithFreeEnergy(SumCalculator):
    """SumCalculator that also exposes `free_energy`, aliased to `energy`.

    The mixer only offers properties common to every sub-calculator, and
    neither DFTD4 nor FAIRChemCalculator lists `free_energy` in
    `implemented_properties` (DFTD4 sets it in `results` but does not declare
    it). Anything asking for a force-consistent energy would then raise -
    notably `IsotropicMTKNPT.get_conserved_energy`, which is the main
    diagnostic for whether the NPT integration is behaving.
    """

    def __init__(self, calcs):
        super().__init__(calcs)
        if "free_energy" not in self.implemented_properties:
            self.implemented_properties = list(self.implemented_properties)
            self.implemented_properties.append("free_energy")

    def calculate(self, atoms, properties, system_changes):
        wanted = [p for p in properties if p != "free_energy"]
        if "energy" not in wanted:
            wanted.append("energy")
        super().calculate(atoms, wanted, system_changes)
        self.results["free_energy"] = self.results["energy"]


def build_calculator(device="cuda", seed=0, three_body=False, model=MODEL,
                     task=TASK, functional=None, disp3_cutoff=DISP3_CUTOFF,
                     atoms=None):
    """UMA `model` on `task`, optionally summed with the ATM-only D4 term."""
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    functional = functional or functional_for(task)

    if atoms is not None and atoms.pbc.any() and task not in PERIODIC_TASKS:
        print(f"  WARNING: task '{task}' was trained on non-periodic data "
              f"({functional.upper()}); running it on a periodic cell is an "
              "extrapolation, and it will not provide a stress.")

    try:
        predictor = pretrained_mlip.get_predict_unit(model, device=device,
                                                     seed=seed)
    except Exception as exc:  # noqa: BLE001 - remap to something actionable
        available = getattr(pretrained_mlip, "available_models", KNOWN_MODELS)
        raise SystemExit(
            f"\nCould not load UMA model '{model}': {exc}\n"
            f"Available: {', '.join(map(str, available))}\n"
            "Gated checkpoints also need a Hugging Face token with access to "
            "the UMA repository."
        ) from exc

    calc = FAIRChemCalculator(predictor, task_name=task)
    if not three_body:
        print(f"  Calculator: {model} / {task}")
        return calc

    print(f"  Calculator: {model} / {task} + D4 three-body")
    return SumWithFreeEnergy([calc, make_d4_atm(functional, disp3_cutoff)])


def require_stress(atoms, task):
    """Fail fast, and legibly, if the calculator cannot drive a barostat."""
    try:
        atoms.get_stress()
    except Exception as exc:  # noqa: BLE001 - any failure here is fatal anyway
        raise SystemExit(
            f"\nTask '{task}' did not return a stress tensor ({exc}).\n"
            "A barostat cannot run without one. OMol25 is a non-periodic "
            "dataset with no stress labels, so the omol task cannot be used "
            "for an NPT density; use --task omc for the periodic liquid.\n"
            "You can still run stage 1 (NVT) with omol if you want to compare "
            "structure at fixed volume."
        ) from exc




def suffix(three_body, task=TASK, model=MODEL, n_side=None):
    """Filename tag so different runs do not overwrite each other.

    Sizes are tagged against `boxes.DEFAULT_N_SIDE` (n_side=6, 216 H2O), the
    historical box the un-tagged filenames already belong to.

    Empty for the default model/task/size, so existing output names are
    unchanged. The model belongs in the tag: a density from uma-m is a
    different number from a density from uma-s, and silently overwriting one
    with the other is the easiest way to lose a comparison. Likewise for box
    size: a 64-molecule density and a 216-molecule density are not the same
    measurement and must not share a filename.
    """
    model_tag = "" if model == MODEL else "_" + re.sub(r"[^A-Za-z0-9]+", "-", model)
    task_tag = "" if task == TASK else f"_{task}"
    size_tag = "" if n_side in (None, DEFAULT_N_SIDE) else f"_n{n_side}"
    return model_tag + task_tag + size_tag + ("_atm" if three_body else "")
