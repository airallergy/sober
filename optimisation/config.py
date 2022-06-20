from typing import Any
from pathlib import Path

from ._tools import AnyStrPath
from ._simulator import _default_root

_CONFIG: dict[str, Any] = {
    "exec.energyplus": None,
    "exec.epmacro": None,
    "exec.readvars": None,
    "schema.energyplus": None,
}


def config_energyplus(
    version_parts: tuple[int, int, int] = None,
    root: AnyStrPath = None,
    energyplus_exec: AnyStrPath = None,
    epmacro_exec: AnyStrPath = None,
    readvars_exec: AnyStrPath = None,
    schema: AnyStrPath = None,
) -> None:
    if version_parts is not None:
        root = _default_root(*version_parts)

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
