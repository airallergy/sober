from os.path import relpath
from pathlib import Path, PurePath
from subprocess import PIPE, STDOUT, run

from .config import _config
from ._tools import AnyStrPath


def _run_epmacro(imf_file: Path) -> Path:
    if imf_file.stem != "in":
        (imf_file.parent / "in.imf").symlink_to(imf_file)

    commands = (_config["exec.epmacro"],)
    run(commands, stdout=PIPE, stderr=STDOUT, cwd=imf_file.parent, text=True)

    if imf_file.stem != "in":
        (imf_file.parent / "in.imf").unlink()

    return imf_file.with_name("out.idf").rename(imf_file.with_name("in.idf"))


def _run_energyplus(
    idf_file: Path,
    epw_file: Path,
    job_directory: Path,
    has_templates: bool,
) -> None:
    commands = (
        (_config["exec.energyplus"],)
        + (("-x",) if has_templates else ())
        + ("-w", relpath(epw_file, job_directory), relpath(idf_file, job_directory))
    )
    run(commands, stdout=PIPE, stderr=STDOUT, cwd=job_directory, text=True)


def _run_readvars(
    rvi_file: Path,
    job_directory: Path,
    frequency: str,
) -> None:
    commands = (
        _config["exec.readvars"],
        relpath(rvi_file, job_directory),
        "Unlimited",
        "FixHeader",
    ) + ((frequency,) if frequency else ())
    run(commands, stdout=PIPE, stderr=STDOUT, cwd=job_directory, text=True)


def _resolved_path(path: AnyStrPath, default_parent: AnyStrPath) -> Path:
    pure_path = PurePath(path)
    if pure_path.is_absolute():
        return Path(pure_path).resolve()
    else:
        return Path(default_parent).resolve() / pure_path


def _resolved_macros(macro_lines: list[str], model_directory: Path) -> list[str]:
    # lines should have been trimmed
    fileprefix = model_directory.resolve()
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
