from pathlib import Path

from . import config as cf
from ._tools import _Parallel
from .results import _ResultsManager
from ._simulator import _run_energyplus
from .parameters import _ParametersManager


def _pymoo_evaluate() -> None:
    ...


def _product_evaluate(
    *variation_vecs: cf.AnyIntVariationVec,
    parameters_manager: _ParametersManager,
    results_manager: _ResultsManager,
    evaluation_directory: Path,
) -> None:
    jobs = tuple(parameters_manager._jobs(*variation_vecs))  # type: ignore[arg-type] # might be resolved after python/mypy#12280

    parameters_manager._make_batch(evaluation_directory, jobs)

    with _Parallel(
        cf._config["n.processes"],
        initializer=cf._update_config,
        initargs=(cf._config,),
    ) as parallel:
        parallel.map(
            _run_energyplus,
            (
                evaluation_directory / job_uid / task_uid
                for job_uid, tasks in jobs
                for task_uid, _ in tasks
            ),
        )

    results_manager._collect_batch(evaluation_directory, jobs)
