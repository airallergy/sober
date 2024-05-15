from __future__ import annotations

import os
import platform
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, overload

import psutil

from sober._tools import _parsed_path

if TYPE_CHECKING:
    from typing import Final, Literal, TypeAlias, TypedDict

    from sober._typing import AnyLanguage, AnyModelType, AnyStrPath, NoiseSampleKwargs

    class _RecordsFilenames(TypedDict):
        task: str
        job: str
        batch: str

    _Config = TypedDict(
        "_Config",
        {
            "schema.energyplus": str,
            "exec.energyplus": str,
            "exec.epmacro": str,
            "exec.expandobjects": str,
            "exec.readvars": str,
            "exec.python": str,
            "n.processes": int,
        },
        total=False,
    )

    _AnyPathConfigName: TypeAlias = Literal[
        "schema.energyplus",
        "exec.energyplus",
        "exec.epmacro",
        "exec.expandobjects",
        "exec.readvars",
        "exec.python",
    ]
    _AnyIntConfigName: TypeAlias = Literal["n.processes"]
    _AnyConfigName: TypeAlias = Literal[_AnyPathConfigName, _AnyIntConfigName]

    class _PackageAttrs(TypedDict):
        config: _Config
        noise_sample_kwargs: NoiseSampleKwargs
        removes_subdirs: bool


#############################################################################
#######                      PACKAGE ATTRIBUTES                       #######
#############################################################################
# constant
_RECORDS_FILENAMES: Final[_RecordsFilenames] = {  # python/typing#1388
    "task": "task_records.csv",
    "job": "job_records.csv",
    "batch": "batch_records.csv",
}

# variable, pass across processes
_config: _Config

# variable, only used in the parent process
_noise_sample_kwargs: NoiseSampleKwargs
_removes_subdirs: bool

## dependent on analysis type (parametrics or optimisation)
_has_batches: bool


@overload
def __getattr__(name: Literal["_noise_sample_kwargs"], /) -> NoiseSampleKwargs: ...  # type: ignore[misc]  # python/mypy#8203
@overload
def __getattr__(name: Literal["_removes_subdirs"], /) -> bool: ...  # type: ignore[misc]  # python/mypy#8203
@overload
def __getattr__(  # type: ignore[misc]  # python/mypy#8203
    name: Literal["_config"], /
) -> _Config: ...
def __getattr__(name: str, /) -> object:
    """lazily set these attributes when they are called for the first time"""
    match name:
        case "_config":
            global _config

            _config = {}
            return _config
        case "_noise_sample_kwargs":
            global _noise_sample_kwargs

            _noise_sample_kwargs = {"mode": "auto"}
            return _noise_sample_kwargs
        case "_removes_subdirs":
            global _removes_subdirs

            _removes_subdirs = False
            return _removes_subdirs
        case _:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'.")


def _package_attrs() -> _PackageAttrs:
    """returns package attributes"""
    return {
        "noise_sample_kwargs": _noise_sample_kwargs,
        "removes_subdirs": _removes_subdirs,
        "config": _config,
    }


def _update_package_attrs(
    config: _Config, noise_sample_kwargs: NoiseSampleKwargs, removes_subdirs: bool
) -> None:
    """updates package attributes"""

    _update_config(config)

    global _noise_sample_kwargs
    global _removes_subdirs

    _noise_sample_kwargs = noise_sample_kwargs
    _removes_subdirs = removes_subdirs


#############################################################################
#######                    CONFIGURATION FUNCTIONS                    #######
#############################################################################
def _update_config(config: _Config) -> None:
    """updates configuration in the current python interpreter process
    this is to copy configuration into child processes when using multiprocessing"""

    global _config

    _config = config


def _check_config(
    model_type: AnyModelType,
    has_templates: bool,
    uses_rvi: bool,
    languages: set[AnyLanguage],
) -> None:
    """checks the configuration sufficiency"""

    if model_type == ".imf" and ("exec.epmacro" not in _config):
        raise ValueError(
            f"a macro model is input, but the epmacro executable is not configured: {_config}."
        )

    if has_templates and ("exec.expandobjects" not in _config):
        raise ValueError(
            f"HVAC templates are used, but the expandobjects executable is not configured: {_config}."
        )

    if uses_rvi and ("exec.readvars" not in _config):
        raise ValueError(
            f"an RVICollector is used, but the readvars executable is not configured: {_config}."
        )

    for item in languages:
        if "exec." + item not in _config:
            raise ValueError(
                f"an ScriptCollector of {item} is used, but the {item} executable is not configured: {_config}."
            )


@overload
def _set_config_item(name: _AnyPathConfigName, value: Path) -> None: ...
@overload
def _set_config_item(name: _AnyIntConfigName, value: int) -> None: ...
def _set_config_item(name: _AnyConfigName, value: Path | int) -> None:
    if name in sys.modules[__name__]._config:  # pep562 usage
        is_equal = (
            os.path.samefile(_config[name], value)
            if isinstance(value, Path)
            else _config[name] == value
        )

        if not is_equal:
            warnings.warn(
                f"'{name.replace('.','_')}' has been configured to '{_config[name]}', and will be overriden by '{value}'.",
                stacklevel=2,
            )

    _config[name] = os.fsdecode(value) if isinstance(value, Path) else value


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
    schema_energyplus: AnyStrPath | None = None,
    exec_energyplus: AnyStrPath | None = None,
    exec_epmacro: AnyStrPath | None = None,
    exec_expandobjects: AnyStrPath | None = None,
    exec_readvars: AnyStrPath | None = None,
) -> None:
    """sets EnergyPlus-related configuration
    this initialise _config, so needs to happen before all others"""
    # TODO: change this to non-mandatory when metamodelling is supported

    if version is not None:
        root = _default_energyplus_root(*version.split("."))

    if root is not None:
        root = _parsed_path(root, "energyplus root")
        schema_energyplus = root / "Energy+.idd"
        exec_energyplus = root / "energyplus"
        exec_epmacro = root / "EPMacro"
        exec_expandobjects = root / "ExpandObjects"
        exec_readvars = root / "PostProcess" / "ReadVarsESO"

    if (schema_energyplus is None) or (exec_energyplus is None):
        raise ValueError(
            "one of 'version', 'root' or 'schema_energyplus & exec_energyplus' needs to be provided."
        )

    _set_config_item(
        "schema.energyplus", _parsed_path(schema_energyplus, "energyplus schema")
    )
    _set_config_item(
        "exec.energyplus", _parsed_path(exec_energyplus, "energyplus executable")
    )

    if exec_epmacro is not None:
        _set_config_item(
            "exec.epmacro", _parsed_path(exec_epmacro, "epmacro executable")
        )
    if exec_expandobjects is not None:
        _set_config_item(
            "exec.expandobjects",
            _parsed_path(exec_expandobjects, "expandobjects executable"),
        )
    if exec_readvars is not None:
        _set_config_item(
            "exec.readvars", _parsed_path(exec_readvars, "readvars executable")
        )


def config_parallel(*, n_processes: int | None = None) -> None:
    """sets parallel-related configuration"""

    # the default number of processes is the number of physical cores - 1
    # this leaves one physical core idle
    n_processes = (
        psutil.cpu_count(logical=False) - 1 if n_processes is None else n_processes
    )

    _set_config_item("n.processes", n_processes)


def config_script(*, python_exec: AnyStrPath | None = None) -> None:
    """sets script-related configuration
    only supports python currently"""
    # TODO: **kwargs from PEP 692/3.12

    if python_exec is not None:
        _set_config_item("exec.python", _parsed_path(python_exec, "python executable"))
