from os.path import relpath
from typing import TypeAlias
from pathlib import Path, PurePath
from collections.abc import Iterable
from subprocess import PIPE, STDOUT, run

from . import config as cf
from ._tools import AnyStrPath

CMD: TypeAlias = tuple[AnyStrPath, ...]


def _run_epmacro(imf_file: Path) -> Path:
    if imf_file.stem != "in":
        (imf_file.parent / "in.imf").symlink_to(imf_file)

    commands: CMD = (cf._config["exec.epmacro"],)
    run(commands, stdout=PIPE, stderr=STDOUT, cwd=imf_file.parent, text=True)

    if imf_file.stem != "in":
        (imf_file.parent / "in.imf").unlink()

    return imf_file.with_name("out.idf").rename(imf_file.with_name("in.idf"))


def _run_energyplus(
    idf_file: Path, epw_file: Path, cwd: Path, has_templates: bool = False
) -> None:
    commands: CMD = (cf._config["exec.energyplus"],)
    if has_templates:
        commands += ("-x",)
    commands += (
        "-w",
        relpath(epw_file, cwd),
        relpath(idf_file, cwd),
    )
    run(commands, stdout=PIPE, stderr=STDOUT, cwd=cwd, text=True)


def _run_readvars(rvi_file: Path, cwd: Path, frequency: str = "") -> None:
    commands: CMD = (
        cf._config["exec.readvars"],
        relpath(rvi_file, cwd),
        "Unlimited",
        "FixHeader",
    )
    if frequency:
        commands += (frequency,)
    run(commands, stdout=PIPE, stderr=STDOUT, cwd=cwd, text=True)


def _resolved_path(path: AnyStrPath, default_parent: Path) -> Path:
    pure_path = PurePath(path)
    if pure_path.is_absolute():
        return Path(pure_path).resolve(strict=True)
    else:
        return default_parent / pure_path


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


def _split_model(model_file: Path) -> tuple[str, str]:
    macro_lines = []
    regular_lines = []
    with model_file.open("rt") as fp:
        for line in fp:
            trimmed_line = line.strip()
            if trimmed_line.startswith("##"):
                macro_lines.append(trimmed_line)
            elif trimmed_line != "":
                regular_lines.append(trimmed_line)
    return (
        "\n".join(_resolved_macros(macro_lines, model_file.parent)) + "\n",
        "\n".join(regular_lines) + "\n",
    )
