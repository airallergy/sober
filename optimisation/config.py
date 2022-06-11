from pathlib import Path
from ._simulator import _default_root

from ._tools import AnyPath

CONFIG = {"exec.energyplus": None, "exec.readvars": None, "schema.energyplus": None}


def config_energyplus(
    version_parts: tuple[int, int, int] = None,
    root: AnyPath = None,
    energyplus_exec: AnyPath = None,
    readvars_exec: AnyPath = None,
    schema: AnyPath = None,
) -> None:
    if version_parts is not None:
        root = _default_root(*version_parts)

    if root is not None:
        root = Path(root)
        energyplus_exec = root / "energyplus"
        readvars_exec = root / "PostProcess" / "ReadVarsESO"
        schema = root / "Energy+.idd"

    CONFIG["exec.energyplus"] = Path(energyplus_exec).resolve(strict=True)
    CONFIG["exec.readvars"] = Path(readvars_exec).resolve(strict=True)
    CONFIG["schema.energyplus"] = Path(schema).resolve(strict=True)
