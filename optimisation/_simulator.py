from pathlib import Path, PurePath
from collections.abc import Iterable

from . import config as cf
from ._tools import AnyCli, AnyStrPath, _run


def _run_epmacro(imf_file: Path) -> Path:
    if imf_file.stem != "in":
        (imf_file.parent / "in.imf").symlink_to(imf_file)

    commands: AnyCli = (cf._config["exec.epmacro"],)
    _run(commands, imf_file.parent)

    if imf_file.stem != "in":
        (imf_file.parent / "in.imf").unlink()

    return imf_file.with_name("out.idf").rename(imf_file.with_name("in.idf"))


def _run_energyplus(
    idf_file: Path, epw_file: Path, cwd: Path, has_templates: bool = False
) -> None:
    commands: AnyCli = (cf._config["exec.energyplus"],)
    if has_templates:
        commands += ("-x",)
    commands += ("-w", epw_file, idf_file)
    _run(commands, cwd)


def _run_readvars(rvi_file: Path, cwd: Path, frequency: str = "") -> None:
    commands: AnyCli = (cf._config["exec.readvars"], rvi_file, "Unlimited", "FixHeader")
    if frequency:
        commands += (frequency,)
    _run(commands, cwd)


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
