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
    evaluation_directory = _meta_params["evaluation_directory"]

    # create job folder
    job_directory = evaluation_directory / job_uid

    # uncertainty
    for task_uid, _ in tasks:
        # create task folder
        task_directory = job_directory / task_uid

        # copy task weather files
        task_epw_file = task_directory / "in.epw"

        task_idf_file = task_directory / "in.idf"

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
    parameters_manager: _ParametersManager,
    results_manager: _ResultsManager,
    evaluation_directory,
    **meta_params,  # TODO: **MetaParams after PEP 692/3.12
) -> None:
    # TODO: remove typing after PEP 692/3.12
    parameters_manager._make_batch(evaluation_directory, jobs, meta_params)

    ctx = _multiprocessing_context()
    with ctx.Manager() as manager:
        _meta_params = manager.dict(meta_params)
        with ctx.Pool(
            cf._config["n.processes"],
            initializer=_initialise,
            initargs=(cf._config, _meta_params),
        ) as pool:
            pool.starmap(func, jobs)

    results_manager._collect_batch(
        evaluation_directory,
        (
            (job_uid, tuple(task_uid for task_uid, _ in tasks))
            for job_uid, tasks in jobs
        ),
    )
