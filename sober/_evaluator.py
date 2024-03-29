from pathlib import Path

from . import config as cf
from ._tools import _Parallel
from ._typing import AnyBatchOutputs, AnyCandidateVec
from .input import _InputManager
from .output import _OutputManager


def _evaluate(
    *candidate_vecs: AnyCandidateVec,
    input_manager: _InputManager,
    output_manager: _OutputManager,
    batch_directory: Path,
) -> None:
    jobs = tuple(input_manager._jobs(*candidate_vecs))

    with _Parallel(
        cf._config["n.processes"],
        initializer=cf._update_config,
        initargs=(cf._config,),
    ) as parallel:
        input_manager._make_batch(batch_directory, jobs, parallel)

        input_manager._simulate_batch(batch_directory, jobs, parallel)

        output_manager._collect_batch(batch_directory, jobs, parallel)

        output_manager._clean_batch(batch_directory, jobs, parallel)


def _pymoo_evaluate(
    *candidate_vecs: AnyCandidateVec,
    input_manager: _InputManager,
    output_manager: _OutputManager,
    batch_directory: Path,
) -> tuple[AnyBatchOutputs, AnyBatchOutputs]:
    _evaluate(
        *candidate_vecs,
        input_manager=input_manager,
        output_manager=output_manager,
        batch_directory=batch_directory,
    )

    return (
        output_manager._recorded_objectives(batch_directory),
        output_manager._recorded_constraints(batch_directory),
    )
