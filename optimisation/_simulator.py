from pathlib import Path, PurePath
from platform import system
from subprocess import run, PIPE, STDOUT

from .config import _CONFIG

from ._tools import AnyStrPath


def _default_root(major: int, minor: int, patch: int = 0) -> Path:
    version = "-".join((major, minor, patch))
    match system():
        case "Linux":
            return Path(f"/usr/local/EnergyPlus-{version}")
        case "Darwin":
            return Path(f"/Applications/EnergyPlus-{version}")
        case "Windows":
            return Path(rf"C:\EnergyPlusV{version}")
        case _ as system_name:
            raise NotImplementedError(f"unsupported system: '{system_name}'.")


def _run_energyplus(
    model_file: Path,
    weather_file: Path,
    output_directory: Path,
    has_macros: bool,
    has_templates: bool,
) -> None:
    commands = (
        (_CONFIG["exec.energyplus"],)
        + (("-m",) if has_macros else ())
        + (("-x",) if has_templates else ())
        + ("-w", weather_file, model_file)
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


def _resolved_path(path: AnyStrPath, default_parent: AnyStrPath) -> str:
    pure_path = PurePath(path)
    if pure_path.is_absolute():
        return str(Path(pure_path).resolve())
    else:
        return str(Path(default_parent).resolve() / pure_path)


def _resolved_macros(macro_lines: list[str], model_directory: Path) -> list[str]:
    # lines should have been trimmed
    fileprefix = model_directory.resolve()
    resolved_macro_lines = []
    for line in macro_lines:
        if line.startswith("##fileprefix"):
            fileprefix = _resolved_path(line.split(" ", 1)[1], model_directory)
        elif line.startswith("##include"):
            resolved_macro_lines.append(
                "##include " + _resolved_path(line.split(" ", 1)[1], fileprefix)
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
