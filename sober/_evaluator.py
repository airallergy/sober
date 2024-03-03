from pathlib import Path

from . import config as cf
from ._tools import _Parallel
from .results import _ResultsManager
from .parameters import _ParametersManager
from ._typing import AnyBatchResults, AnyCandidateVec


def _evaluate(
    *candidate_vecs: AnyCandidateVec,
    parameters_manager: _ParametersManager,
    results_manager: _ResultsManager,
    batch_directory: Path,
) -> None:
    jobs = tuple(parameters_manager._jobs(*candidate_vecs))

    with _Parallel(
        cf._config["n.processes"],
        initializer=cf._update_config,
        initargs=(cf._config,),
    ) as parallel:

        parameters_manager._make_batch(batch_directory, jobs, parallel)

        parameters_manager._simulate_batch(batch_directory, jobs, parallel)

        results_manager._collect_batch(batch_directory, jobs, parallel)

        results_manager._clean_batch(batch_directory, jobs, parallel)


def _pymoo_evaluate(
    *candidate_vecs: AnyCandidateVec,
    parameters_manager: _ParametersManager,
    results_manager: _ResultsManager,
    batch_directory: Path,
) -> tuple[AnyBatchResults, AnyBatchResults]:
    _evaluate(
        *candidate_vecs,
        parameters_manager=parameters_manager,
        results_manager=results_manager,
        batch_directory=batch_directory,
    )

    return (
        results_manager._recorded_objectives(batch_directory),
        results_manager._recorded_constraints(batch_directory),
    )
