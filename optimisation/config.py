from pathlib import Path
from platform import system
from typing import Iterable, TypedDict

from ._tools import AnyStrPath
from .collector import RVICollector, _Collector

Config = TypedDict(
    "Config",
    {
        "schema.energyplus": Path | None,
        "exec.energyplus": Path | None,
        "exec.epmacro": Path | None,
        "exec.readvars": Path | None,
        "exec.python": Path | None,
    },
)

_config: Config = {
    "schema.energyplus": None,
    "exec.energyplus": None,
    "exec.epmacro": None,
    "exec.readvars": None,
    "exec.python": None,
}
_config_directory: Path


def _update_config(config: Config) -> None:
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
    *,
    version: str | None = None,
    root: AnyStrPath | None = None,
    schema: AnyStrPath | None = None,
    energyplus_exec: AnyStrPath | None = None,
    epmacro_exec: AnyStrPath | None = None,
    readvars_exec: AnyStrPath | None = None,
) -> None:
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

    _config["schema.energyplus"] = Path(schema).resolve(strict=True)
    _config["exec.energyplus"] = Path(energyplus_exec).resolve(strict=True)
    if epmacro_exec is not None:
        _config["exec.epmacro"] = Path(epmacro_exec).resolve(strict=True)
    if readvars_exec is not None:
        _config["exec.readvars"] = Path(readvars_exec).resolve(strict=True)


def config_script(python: AnyStrPath | None = None) -> None:
    if python is not None:
        _config[f"exec.python"] = Path(python).resolve(strict=True)
