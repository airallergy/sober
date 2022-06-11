from pathlib import Path
from platform import system
from subprocess import run, PIPE, STDOUT


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
    energyplus_root: Path,
) -> None:
    commands = (
        (energyplus_root / "energyplus",)
        + (("-m",) if has_macros else ())
        + (("-x",) if has_templates else ())
        + ("-w", weather_file, model_file)
    )
    run(commands, stdout=PIPE, stderr=STDOUT, cwd=output_directory, text=True)


def _run_readvars(
    rvi_file: Path,
    output_directory: Path,
    frequency: str,
    energyplus_root: Path,
) -> None:
    commands = (
        energyplus_root / "PostProcess" / "ReadVarsESO",
        rvi_file,
        "Unlimited",
        "FixHeader",
    ) + ((frequency,) if frequency else ())
    run(commands, stdout=PIPE, stderr=STDOUT, cwd=output_directory, text=True)


def _model_splited(model_file: Path) -> tuple[list, list]:
    macro_lines = []
    regular_lines = []
    with model_file.open("rt") as fp:
        for line in fp:
            if line.lstrip().startswith("##"):
                macro_lines.append(line)
            else:
                regular_lines.append(line)
    return macro_lines, regular_lines
