from pathlib import Path
from shutil import copyfile
from typing import TypedDict
from collections.abc import Callable, Iterable

from . import config as cf
from .results import _ResultsManager
from ._tools import _multiprocessing_context
from ._simulator import _run_epmacro, _run_energyplus
from .parameters import AnyParameter, AnyIntParameter, _ParametersManager

MetaParams = TypedDict(
    "MetaParams",
    {
        "tagged_model": str,
        "parameters_manager": _ParametersManager[AnyParameter],
        "evaluation_directory": Path,
        "model_type": cf.AnyModelType,
    },
)
## TODO: the following may be generalised after 3.11 | python/mypy#3863 ##
MetaIntParams = TypedDict(
    "MetaIntParams",
    {
        "tagged_model": str,
        "parameters_manager": _ParametersManager[AnyIntParameter],
        "evaluation_directory": Path,
        "model_type": cf.AnyModelType,
    },
)
##########################################################################

_meta_params: MetaParams


def _product_evaluate(job_uid: str, tasks: Iterable[cf.AnyIntTask]) -> None:
    tagged_model = _meta_params["tagged_model"]
    parameters_manager = _meta_params["parameters_manager"]
    evaluation_directory = _meta_params["evaluation_directory"]
    model_type = _meta_params["model_type"]

    # create job folder
    job_directory = evaluation_directory / job_uid
    job_directory.mkdir(exist_ok=True)

    # uncertainty
    for task_uid, vu_mat in tasks:
        # TODO: some may better go to parameters manager
        weather_vu_row = vu_mat[0]
        parameter_vu_rows = vu_mat[1:]

        # create task folder
        task_directory = job_directory / task_uid
        task_directory.mkdir(exist_ok=True)

        # copy task weather files
        task_epw_file = task_directory / "in.epw"
        copyfile(parameters_manager._weather[weather_vu_row], task_epw_file)

        # detag model with parameter values
        model = parameters_manager._detagged_model(tagged_model, parameter_vu_rows)

        # write task model file
        task_model_file = task_directory / ("in" + model_type)
        with open(task_model_file, "wt") as f:
            f.write(model)

        # run epmacro if needed
        if model_type == ".imf":
            task_idf_file = _run_epmacro(task_model_file)
        elif model_type == ".idf":
            task_idf_file = task_model_file
        else:
            raise

        # run energyplus
        _run_energyplus(task_idf_file, task_epw_file, task_directory, False)


def _pymoo_evaluate():
    ...


def _initialise(config: cf.Config, meta_params: MetaParams) -> None:
    cf._update_config(config)
    global _meta_params
    _meta_params = meta_params


def _parallel_evaluate(
    func: Callable,
    jobs: tuple[cf.AnyJob, ...],
    results_manager: _ResultsManager,
    **meta_params,  # TODO: **MetaParams after PEP 692/3.12
) -> None:
    ctx = _multiprocessing_context()
    with ctx.Manager() as manager:
        _meta_params = manager.dict(meta_params)
        with ctx.Pool(
            cf._config["n.processes"],
            initializer=_initialise,
            initargs=(cf._config, _meta_params),
        ) as pool:
            pool.starmap(func, jobs)

    # TODO: remove typing after PEP 692/3.12
    evaluation_directory: Path = meta_params["evaluation_directory"]

    results_manager._collect_batch(
        evaluation_directory,
        (
            (job_uid, tuple(task_uid for task_uid, _ in tasks))
            for job_uid, tasks in jobs
        ),
    )
