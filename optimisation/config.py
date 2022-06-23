from typing import Any
from pathlib import Path
from platform import system

from ._tools import AnyStrPath

_CONFIG: dict[str, Any] = {
    "exec.energyplus": None,
    "exec.epmacro": None,
    "exec.readvars": None,
    "schema.energyplus": None,
}


def _update_config(config: dict[str, Any]) -> None:
    for key, val in config.items():
        _CONFIG[key] = val


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
    if version is not None:
        root = _default_energyplus_root(*version.split("."))

    if root is not None:
        root = Path(root)
        energyplus_exec = root / "energyplus"
        epmacro_exec = root / "EPMacro"
        readvars_exec = root / "PostProcess" / "ReadVarsESO"
        schema = root / "Energy+.idd"

    if (
        (energyplus_exec is not None)
        and (epmacro_exec is not None)
        and (readvars_exec is not None)
        and (schema is not None)
    ):
        _CONFIG["exec.energyplus"] = Path(energyplus_exec).resolve(strict=True)
        _CONFIG["exec.epmacro"] = Path(epmacro_exec).resolve(strict=True)
        _CONFIG["exec.readvars"] = Path(readvars_exec).resolve(strict=True)
        _CONFIG["schema.energyplus"] = Path(schema).resolve(strict=True)
    else:
        raise ValueError(
            "One of version_parts, root, (energyplus_exec, epmacro_exec, readvars_exec, schema) needs to be provided."
        )
