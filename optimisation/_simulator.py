from platform import system
from pathlib import Path, PurePath
from subprocess import PIPE, STDOUT, run

from .config import _CONFIG
from ._tools import AnyStrPath


def _default_root(major: int, minor: int, patch: int = 0) -> Path:
    version = "-".join((str(major), str(minor), str(patch)))
    match system():
        case "Linux":
            return Path(f"/usr/local/EnergyPlus-{version}")
        case "Darwin":
            return Path(f"/Applications/EnergyPlus-{version}")
        case "Windows":
            return Path(rf"C:\EnergyPlusV{version}")
        case _ as system_name:
            raise NotImplementedError(f"unsupported system: '{system_name}'.")


def _run_epmacro(imf_file: Path) -> None:
    if imf_file.stem != "in":
        (imf_file.parent / "in.imf").symlink_to(imf_file)

    commands = (_CONFIG["exec.epmacro"],)
    run(commands, stdout=PIPE, stderr=STDOUT, cwd=imf_file.parent, text=True)

    if imf_file.stem != "in":
        (imf_file.parent / "in.imf").unlink()


def _run_energyplus(
    idf_file: Path,
    epw_file: Path,
    output_directory: Path,
    has_templates: bool,
) -> None:
    commands = (
        (_CONFIG["exec.energyplus"],)
        + (("-x",) if has_templates else ())
        + ("-w", epw_file, idf_file)
    )
    run(commands, stdout=PIPE, stderr=STDOUT, cwd=output_directory, text=True)


def _run_readvars(
    rvi_file: Path,
    output_directory: Path,
    frequency: str,
) -> None:
    commands = (
        _CONFIG["exec.readvars"],
        rvi_file,
        "Unlimited",
        "FixHeader",
    ) + ((frequency,) if frequency else ())
    run(commands, stdout=PIPE, stderr=STDOUT, cwd=output_directory, text=True)


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
