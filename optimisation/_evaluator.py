from pathlib import Path

from . import config as cf
from ._tools import _Parallel
from .results import _ResultsManager
from ._simulator import _run_energyplus
from .parameters import _ParametersManager
from ._typing import AnyBatchResults, AnyVariationVec


def _evaluate(
    *variation_vecs: AnyVariationVec,
    parameters_manager: _ParametersManager,
    results_manager: _ResultsManager,
    batch_directory: Path,
) -> None:
    jobs = tuple(parameters_manager._jobs(*variation_vecs))

    parameters_manager._make_batch(batch_directory, jobs)

    with _Parallel(
        cf._config["n.processes"],
        initializer=cf._update_config,
        initargs=(cf._config,),
    ) as parallel:
        parallel.map(
            _run_energyplus,
            (
                batch_directory / job_uid / task_uid
                for job_uid, tasks in jobs
                for task_uid, _ in tasks
            ),
        )

    results_manager._collect_batch(batch_directory, jobs)


def _pymoo_evaluate(
    *variation_vecs: AnyVariationVec,
    parameters_manager: _ParametersManager,
    results_manager: _ResultsManager,
    batch_directory: Path,
) -> tuple[AnyBatchResults, AnyBatchResults]:
    _evaluate(
        *variation_vecs,
        parameters_manager=parameters_manager,
        results_manager=results_manager,
        batch_directory=batch_directory,
    )

    return (
        results_manager._recorded_objectives(batch_directory),
        results_manager._recorded_constraints(batch_directory),
    )
