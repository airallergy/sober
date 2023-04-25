from pathlib import Path, PurePath
from collections.abc import Iterable

from ._tools import _run
from . import config as cf
from ._typing import AnyCmdArgs, AnyStrPath


def _run_epmacro(cwd: Path) -> None:
    cmd_args: AnyCmdArgs = (cf._config["exec.epmacro"],)
    _run(cmd_args, cwd)

    (cwd / "out.idf").rename(cwd / "in.idf")
    (cwd / "audit.out").rename(cwd / "epmacro.audit")


def _run_expandobjects(cwd: Path) -> None:
    # TODO: Basement & Slab
    idd_file = cwd / Path(cf._config["schema.energyplus"]).name
    idd_file.symlink_to(cf._config["schema.energyplus"])

    cmd_args: AnyCmdArgs = (cf._config["exec.expandobjects"],)
    _run(cmd_args, cwd)

    (cwd / "expanded.idf").rename(cwd / "in.idf")
    if (cwd / "expandedidf.err").exists():
        (cwd / "expandedidf.err").rename(cwd / "expandobjects.audit")
    idd_file.unlink()


def _run_energyplus(cwd: Path) -> None:
    cmd_args: AnyCmdArgs = (cf._config["exec.energyplus"],)
    _run(cmd_args, cwd)


def _run_readvars(cwd: Path, rvi_file: Path, frequency: str) -> None:
    cmd_args: AnyCmdArgs = (
        cf._config["exec.readvars"],
        rvi_file,
        "Unlimited",
        "FixHeader",
    )
    if frequency:
        cmd_args += (frequency,)
    _run(cmd_args, cwd)


def _resolved_path(path: AnyStrPath, default_parent: Path) -> Path:
    pure_path = PurePath(path)
    if pure_path.is_absolute():
        return Path(pure_path).resolve(strict=True)
    else:
        return (default_parent / pure_path).resolve(strict=True)


def _resolved_macros(macro_lines: Iterable[str], model_directory: Path) -> list[str]:
    # lines should have been trimmed
    # model_directory should have been resolved
    fileprefix = model_directory
    resolved_macro_lines = []
    for line in macro_lines:
        if line.startswith("##fileprefix"):
            fileprefix = _resolved_path(line.split(" ", 1)[1], model_directory)
        elif line.startswith("##include"):
            resolved_macro_lines.append(
                "##include " + str(_resolved_path(line.split(" ", 1)[1], fileprefix))
            )
        else:
            resolved_macro_lines.append(line)
    return resolved_macro_lines


def _split_model(model: str, model_directory: Path) -> tuple[str, str]:
    macro_lines = []
    regular_lines = []
    for line in model.splitlines():
        trimmed_line = line.strip()
        if trimmed_line.startswith("##"):
            macro_lines.append(trimmed_line)
        elif trimmed_line:
            regular_lines.append(trimmed_line)
    return (
        "\n".join(_resolved_macros(macro_lines, model_directory)) + "\n",
        "\n".join(regular_lines) + "\n",
    )
