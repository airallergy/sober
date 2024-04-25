from __future__ import annotations

import os
import platform
import warnings
from typing import TYPE_CHECKING

import psutil

from sober._tools import _parsed_path

if TYPE_CHECKING:
    from typing import Final, Required, TypedDict

    from sober._typing import AnyLanguage, AnyModelType, AnyStrPath

    _Config = TypedDict(
        "_Config",
        {
            "schema.energyplus": Required[str],
            "exec.energyplus": Required[str],
            "exec.epmacro": str,
            "exec.expandobjects": str,
            "exec.readvars": str,
            "exec.python": str,
            "n.processes": int,
        },
        total=False,
    )

    class _RecordsFilenames(TypedDict):
        task: str
        job: str
        batch: str


#############################################################################
#######                       GLOBAL CONSTANTS                        #######
#############################################################################
_RECORDS_FILENAMES: Final[_RecordsFilenames] = {  # python/typing#1388
    "task": "task_records.csv",
    "job": "job_records.csv",
    "batch": "batch_records.csv",
}


#############################################################################
#######                       GLOBAL VARIABLES                        #######
#############################################################################
_has_batches: bool = True  # only used in the parent process

_config: _Config


#############################################################################
#######                    CONFIGURATION FUNCTIONS                    #######
#############################################################################
def _update_config(config: _Config) -> None:
    """updates configuration globally in the current python interpreter process
    this is to copy configuration into child processes when using multiprocessing"""

    global _config

    _config = config


def _check_config(
    model_type: AnyModelType,
    has_templates: bool,
    uses_rvi: bool,
    used_languages: set[AnyLanguage],
) -> None:
    """checks the configuration sufficiency"""

    if model_type == ".imf" and ("exec.epmacro" not in _config):
        raise ValueError(
            f"a macro model is input, but the epmacro executable is not configured: {_config}."
        )

    if has_templates and ("exec.expandobjects" not in _config):
        raise ValueError(
            f"hvac templates are used, but the expandobjects executable is not configured: {_config}."
        )

    if uses_rvi and ("exec.readvars" not in _config):
        raise ValueError(
            f"an RVICollector is used, but the readvars executable is not configured: {_config}."
        )

    for language in used_languages:
        if "exec." + language not in _config:
            raise ValueError(
                f"an ScriptCollector of {language} is used, but the {language} executable is not configured: {_config}."
            )


def _default_energyplus_root(major: str, minor: str, patch: str = "0") -> str:
    """returns the default EnergyPlus installation directory"""

    version = f"{major}-{minor}-{patch}"
    match platform.system():
        case "Linux":
            return f"/usr/local/EnergyPlus-{version}"
        case "Darwin":
            return f"/Applications/EnergyPlus-{version}"
        case "Windows":
            return rf"C:\EnergyPlusV{version}"
        case _ as system_name:
            raise NotImplementedError(f"unsupported system: '{system_name}'.")


def config_energyplus(
    *,
    version: str | None = None,
    root: AnyStrPath | None = None,
    schema: AnyStrPath | None = None,
    exec: AnyStrPath | None = None,
    epmacro_exec: AnyStrPath | None = None,
    expandobjects_exec: AnyStrPath | None = None,
    readvars_exec: AnyStrPath | None = None,
) -> None:
    """sets EnergyPlus-related configuration
    this initialise _config, so needs to happen before all others"""
    # TODO: change this to non-mandatory when metamodelling is supported

    global _config

    if version is not None:
        root = _default_energyplus_root(*version.split("."))

    if root is not None:
        root = _parsed_path(root, "energyplus root")
        schema = root / "Energy+.idd"
        exec = root / "energyplus"
        epmacro_exec = root / "EPMacro"
        expandobjects_exec = root / "ExpandObjects"
        readvars_exec = root / "PostProcess" / "ReadVarsESO"

    if (schema is None) or (exec is None):
        raise ValueError("One of version, root or (schema, exec) needs to be provided.")

    _config = {
        "schema.energyplus": os.fsdecode(_parsed_path(schema, "energyplus schema")),
        "exec.energyplus": os.fsdecode(_parsed_path(exec, "energyplus executable")),
    }
    if epmacro_exec is not None:
        _config["exec.epmacro"] = os.fsdecode(
            _parsed_path(epmacro_exec, "epmacro executable")
        )
    if expandobjects_exec is not None:
        _config["exec.expandobjects"] = os.fsdecode(
            _parsed_path(expandobjects_exec, "expandobjects executable")
        )
    if readvars_exec is not None:
        _config["exec.readvars"] = os.fsdecode(
            _parsed_path(readvars_exec, "readvars executable")
        )


def _check_config_init() -> None:
    """checks if EnergyPlus-related configuration has completed"""

    if "_config" not in globals():
        raise NameError("configure energyplus first.")


def config_parallel(*, n_processes: int | None = None) -> None:
    """sets parallel-related configuration"""

    _check_config_init()

    if ("n.processes" in _config) and (_config["n.processes"] != n_processes):
        warnings.warn(
            f"n_processes has been configured to '{_config['n.processes']}', and will be overriden by '{n_processes}'.",
            stacklevel=2,
        )

    # the default number of processes is the number of physical cores - 1
    # this leaves one physical core idle
    _config["n.processes"] = (
        psutil.cpu_count(logical=False) - 1 if n_processes is None else n_processes
    )


def config_script(*, python_exec: AnyStrPath | None = None) -> None:
    """sets script-related configuration
    only supports python currently"""
    # TODO: **kwargs from PEP 692/3.12

    _check_config_init()

    if python_exec is not None:
        python_exec = _parsed_path(python_exec, "python executable")

        if ("exec.python" in _config) and (
            not python_exec.samefile(_config["exec.python"])
        ):
            warnings.warn(
                f"python_exec has been configured to '{_config['exec.python']}', and will be overriden by '{python_exec}'.",
                stacklevel=2,
            )

        _config["exec.python"] = os.fsdecode(python_exec)
