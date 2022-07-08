from pathlib import Path
from collections.abc import Callable, Iterable

from . import config as cf
from .results import _ResultsManager
from ._simulator import _run_energyplus
from .parameters import _ParametersManager
from ._tools import _multiprocessing_context

_evaluation_directory: Path


def _product_evaluate(job_uid: str, tasks: Iterable[cf.AnyIntTask]) -> None:
    # create job folder
    job_directory = _evaluation_directory / job_uid

    # uncertainty
    for task_uid, _ in tasks:
        # create task folder
        task_directory = job_directory / task_uid

        # copy task weather files
        task_epw_file = task_directory / "in.epw"

        task_idf_file = task_directory / "in.idf"

        # run energyplus
        _run_energyplus(task_idf_file, task_epw_file, task_directory, False)


def _pymoo_evaluate() -> None:
    ...


def _initialise(config: cf.Config, evaluation_directory: Path) -> None:
    cf._update_config(config)
    global _evaluation_directory
    _evaluation_directory = evaluation_directory


def _parallel_evaluate(
    func: Callable,
    jobs: tuple[cf.AnyJob, ...],
    parameters_manager: _ParametersManager,
    results_manager: _ResultsManager,
    evaluation_directory: Path,
) -> None:
    # TODO: remove typing after PEP 692/3.12
    parameters_manager._make_batch(evaluation_directory, jobs)

    ctx = _multiprocessing_context()
    with ctx.Pool(
        cf._config["n.processes"],
        initializer=_initialise,
        initargs=(cf._config, evaluation_directory),
    ) as pool:
        pool.starmap(func, jobs)

    results_manager._collect_batch(
        evaluation_directory,
        (
            (job_uid, tuple(task_uid for task_uid, _ in tasks))
            for job_uid, tasks in jobs
        ),
    )
