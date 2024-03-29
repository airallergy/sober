from collections.abc import Sequence
from pathlib import Path, PurePath

from . import config as cf
from ._tools import _run
from ._typing import AnyStrPath


#############################################################################
#######                     SIMULATION FUNCTIONS                      #######
#############################################################################
def _run_epmacro(cwd: Path) -> None:
    """runs EPMacro"""

    cmd_args = (cf._config["exec.epmacro"],)
    _run(cmd_args, cwd)

    # rename output files
    (cwd / "out.idf").rename(cwd / "in.idf")
    (cwd / "audit.out").rename(cwd / "epmacro.audit")


def _run_expandobjects(cwd: Path) -> None:
    """runs ExpandObjects"""

    # make a symlink to the idd file in cwd
    idd_file = cwd / Path(cf._config["schema.energyplus"]).name
    idd_file.symlink_to(cf._config["schema.energyplus"])

    cmd_args = (cf._config["exec.expandobjects"],)
    _run(cmd_args, cwd)

    # rename output files
    (cwd / "expanded.idf").rename(cwd / "in.idf")
    if (cwd / "expandedidf.err").exists():
        (cwd / "expandedidf.err").rename(cwd / "expandobjects.audit")
    idd_file.unlink()


def _run_energyplus(cwd: Path) -> None:
    """runs EnergyPlus"""

    cmd_args = (cf._config["exec.energyplus"],)
    _run(cmd_args, cwd)


def _run_readvars(cwd: Path, rvi_file: Path, frequency: str) -> None:
    """runs ReadVarsESO"""

    cmd_args = (
        cf._config["exec.readvars"],
        rvi_file,
        "Unlimited",
        "FixHeader",
        frequency,  # ReadVarsESO will ignore empty string
    )
    _run(cmd_args, cwd)


#############################################################################
#######                       PARSING FUNCTIONS                       #######
#############################################################################
def _resolved_path(path: AnyStrPath, default_parent: Path) -> Path:
    """a helper function to resolve paths of mixed absolute and relative"""

    pure_path = PurePath(path)
    if pure_path.is_absolute():
        return Path(pure_path).resolve()
    else:
        return (default_parent / pure_path).resolve()


def _resolved_macros(macro_lines: Sequence[str], model_dir: Path) -> list[str]:
    """resolves paths in macro commands used to incorporate external files
    macro lines should have been trimmed before passed in"""

    # set the model directory as fileprefix
    # in case the first macro command is a relative one
    fileprefix = model_dir
    resolved_macro_lines = []
    for line in macro_lines:
        if line.startswith("##fileprefix"):
            # update fileprefix
            # len("##fileprefix") == 12
            fileprefix = _resolved_path(line[13:], model_dir)
        elif line.startswith("##include"):
            # resolve include paths using current fileprefix
            # len("##include") == 9
            resolved_macro_lines.append(
                "##include " + str(_resolved_path(line[10:], fileprefix))
            )
        else:
            # leave macro commands of other types as is
            # NOTE: there might be some corner cases that need catering
            resolved_macro_lines.append(line)
    return resolved_macro_lines


def _split_model(model: str, model_dir: Path) -> tuple[str, str]:
    """splits an EnergyPlus model into macros and regulars"""

    macro_lines = []
    regular_lines = []
    for line in model.splitlines():
        trimmed_line = line.strip()

        # ignore comment or empty lines
        if trimmed_line.startswith("!") or not trimmed_line:
            continue

        if trimmed_line.startswith("##"):
            macro_lines.append(trimmed_line)
        else:
            regular_lines.append(trimmed_line)
    return (
        "\n".join(_resolved_macros(macro_lines, model_dir)) + "\n",
        "\n".join(regular_lines) + "\n",
    )
