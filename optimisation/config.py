from pathlib import Path
from platform import system
from typing import Literal, TypeAlias

from typing_extensions import Required, TypedDict  # TODO: from typing after 3.11

from ._tools import AnyStrPath

AnyLanguage: TypeAlias = Literal["python"]
Config = TypedDict(
    "Config",
    {
        "schema.energyplus": Required[Path],
        "exec.energyplus": Required[Path],
        "exec.epmacro": Path,
        "exec.readvars": Path,
        "exec.python": Path,
    },
    total=False,
)

_config: Config


def _update_config(config: Config) -> None:
    global _config

    _config = config


def _check_config(
    model_type: str, uses_rvi: bool, used_languages: set[AnyLanguage]
) -> None:
    if model_type == ".imf":
        assert (
            "exec.epmacro" in _config
        ), f"a macro model is input, but the epmacro executable is not configured: {_config}."

    if uses_rvi:
        assert (
            "exec.readvars" in _config
        ), f"an RVICollector is used, but the readvars executable is not configured: {_config}."

    # TODO: revision after PEP 675/3.11
    if "python" in used_languages:
        assert (
            "exec.python" in _config
        ), f"an ScriptCollector of {'python'} is used, but the {'python'} executable is not configured: {_config}."


def _default_energyplus_root(major: str, minor: str, patch: str = "0") -> Path:
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


def config_energyplus(
    *,
    version: str | None = None,
    root: AnyStrPath | None = None,
    schema: AnyStrPath | None = None,
    energyplus_exec: AnyStrPath | None = None,
    epmacro_exec: AnyStrPath | None = None,
    readvars_exec: AnyStrPath | None = None,
) -> None:
    global _config

    if version is not None:
        root = _default_energyplus_root(*version.split("."))

    if root is not None:
        root = Path(root)
        schema = root / "Energy+.idd"
        energyplus_exec = root / "energyplus"
        epmacro_exec = root / "EPMacro"
        readvars_exec = root / "PostProcess" / "ReadVarsESO"

    if (energyplus_exec is None) or (schema is None):
        raise ValueError(
            "One of version_parts, root, (schema, energyplus_exec) needs to be provided."
        )

    _config = {
        "schema.energyplus": Path(schema).resolve(strict=True),
        "exec.energyplus": Path(energyplus_exec).resolve(strict=True),
    }
    if epmacro_exec is not None:
        _config["exec.epmacro"] = Path(epmacro_exec).resolve(strict=True)
    if readvars_exec is not None:
        _config["exec.readvars"] = Path(readvars_exec).resolve(strict=True)


def config_script(python: AnyStrPath | None = None) -> None:
    # TODO: **kwargs from PEP 692/3.12
    global _config

    if "_config" not in globals():
        raise NameError("configure energyplus first.")

    if python is not None:
        _config["exec.python"] = Path(python).resolve(strict=True)
