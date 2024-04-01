from pathlib import Path
from platform import system
from warnings import warn

from psutil import cpu_count

from sober._typing import AnyLanguage, AnyModelType, AnyStrPath, Config

#############################################################################
#######                       GLOBAL CONSTANTS                        #######
#############################################################################
_RECORDS_FILENAMES: dict[str, str] = {
    "task": "task_records.csv",
    "job": "job_records.csv",
    "batch": "batch_records.csv",
}


#############################################################################
#######                       GLOBAL VARIABLES                        #######
#############################################################################
_has_batches: bool = True  # only used in the parent process

_config: Config


#############################################################################
#######                    CONFIGURATION FUNCTIONS                    #######
#############################################################################
def _update_config(config: Config) -> None:
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


def _default_energyplus_root(major: str, minor: str, patch: str = "0") -> Path:
    """returns the default EnergyPlus installation directory"""

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
    expandobjects_exec: AnyStrPath | None = None,
    readvars_exec: AnyStrPath | None = None,
) -> None:
    """sets EnergyPlus-related configuration
    this initialise _config, so needs to happen before all others"""

    global _config

    if version is not None:
        root = _default_energyplus_root(*version.split("."))

    if root is not None:
        root = Path(root)
        schema = root / "Energy+.idd"
        energyplus_exec = root / "energyplus"
        epmacro_exec = root / "EPMacro"
        expandobjects_exec = root / "ExpandObjects"
        readvars_exec = root / "PostProcess" / "ReadVarsESO"

    if (energyplus_exec is None) or (schema is None):
        raise ValueError(
            "One of version_parts, root, (schema, energyplus_exec) needs to be provided."
        )

    _config = {
        "schema.energyplus": str(Path(schema).resolve(strict=True)),
        "exec.energyplus": str(Path(energyplus_exec).resolve(strict=True)),
    }
    if epmacro_exec is not None:
        _config["exec.epmacro"] = str(Path(epmacro_exec).resolve(strict=True))
    if expandobjects_exec is not None:
        _config["exec.expandobjects"] = str(
            Path(expandobjects_exec).resolve(strict=True)
        )
    if readvars_exec is not None:
        _config["exec.readvars"] = str(Path(readvars_exec).resolve(strict=True))


def _check_config_init() -> None:
    """checks if EnergyPlus-related configuration has completed"""

    if "_config" not in globals():
        raise NameError("configure energyplus first.")


def config_parallel(*, n_processes: int | None = None) -> None:
    """sets parallel-related configuration"""

    _check_config_init()

    global _config

    if ("n.processes" in _config) and (_config["n.processes"] != n_processes):
        warn(
            f"n_processes has been configured to '{_config['n.processes']}', and will be overriden by '{n_processes}'."
        )

    # the default number of processes is the number of physical cores - 1
    # this leaves one physical core idle
    _config["n.processes"] = (
        cpu_count(logical=False) - 1 if n_processes is None else n_processes
    )


def config_script(*, python_exec: AnyStrPath | None = None) -> None:
    """sets script-related configuration
    only supports python currently"""
    # TODO: **kwargs from PEP 692/3.12

    _check_config_init()

    global _config

    if python_exec is not None:
        if ("exec.python" in _config) and (
            Path(_config["exec.python"]).resolve(True)
            != Path(python_exec).resolve(True)
        ):
            warn(
                f"python_exec has been configured to '{_config['exec.python']}', and will be overriden by '{python_exec}'."
            )

        _config["exec.python"] = str(Path(python_exec).resolve(strict=True))
