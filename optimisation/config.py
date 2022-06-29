from pathlib import Path
from platform import system
from typing import Iterable, TypedDict

from ._tools import AnyStrPath
from .collector import RVICollector, _Collector

Config = TypedDict(
    "Config",
    {
        "exec.energyplus": Path,
        "exec.epmacro": Path | None,
        "exec.readvars": Path | None,
        "schema.energyplus": Path,
    },
)

_config: Config
_config_directory: Path


def _update_config(config: Config) -> None:
    global _config

    if config.keys() != Config.__annotations__.keys():
        raise TypeError(f"configuration must follow '{Config.__annotations__}'.")

    _config = config


def _check_config(model_type: str, outputs: Iterable[_Collector]) -> None:
    if model_type == ".imf":
        assert (
            _config["exec.epmacro"] is not None
        ), f"a macro model is input, but epmacro executable is not configured: {_config}."

    if any(isinstance(output, RVICollector) for output in outputs):
        assert (
            _config["exec.readvars"] is not None
        ), f"an RVICollector is used, but readvars executable is not configured: {_config}."


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
    version: str | None = None,
    root: AnyStrPath | None = None,
    energyplus_exec: AnyStrPath | None = None,
    epmacro_exec: AnyStrPath | None = None,
    readvars_exec: AnyStrPath | None = None,
    schema: AnyStrPath | None = None,
) -> None:
    global _config

    if version is not None:
        root = _default_energyplus_root(*version.split("."))

    if root is not None:
        root = Path(root)
        energyplus_exec = root / "energyplus"
        epmacro_exec = root / "EPMacro"
        readvars_exec = root / "PostProcess" / "ReadVarsESO"
        schema = root / "Energy+.idd"

    if (energyplus_exec is not None) and (schema is not None):
        _config = {
            "exec.energyplus": Path(energyplus_exec).resolve(strict=True),
            "exec.epmacro": (
                epmacro_exec
                if epmacro_exec is None
                else Path(epmacro_exec).resolve(strict=True)
            ),
            "exec.readvars": (
                readvars_exec
                if readvars_exec is None
                else Path(readvars_exec).resolve(strict=True)
            ),
            "schema.energyplus": Path(schema).resolve(strict=True),
        }
    else:
        raise ValueError(
            "One of version_parts, root, (energyplus_exec, epmacro_exec, readvars_exec, schema) needs to be provided."
        )
