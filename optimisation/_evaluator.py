from pathlib import Path

from . import config as cf
from .results import _ResultsManager
from ._simulator import _run_energyplus
from .parameters import _ParametersManager
from ._tools import _multiprocessing_context


def _pymoo_evaluate() -> None:
    ...


def _product_evaluate(
    jobs: tuple[cf.AnyJob, ...],
    parameters_manager: _ParametersManager,
    results_manager: _ResultsManager,
    evaluation_directory: Path,
) -> None:
    task_directories = parameters_manager._make_batch(evaluation_directory, jobs)

    with _multiprocessing_context().Pool(
        cf._config["n.processes"],
        initializer=cf._update_config,
        initargs=(cf._config,),
    ) as pool:
        pool.starmap(
            _run_energyplus,
            (
                (task_directory / "in.idf", task_directory / "in.epw", task_directory)
                for task_directory in task_directories
            ),
        )

    results_manager._collect_batch(evaluation_directory, jobs)
