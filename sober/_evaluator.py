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
    batch_dir: Path,
) -> None:
    jobs = tuple(input_manager._jobs(*candidate_vecs))

    with _Parallel(
        cf._config["n.processes"], cf._update_config, (cf._config,)
    ) as parallel:
        input_manager._make_batch(batch_dir, jobs, parallel)

        input_manager._simulate_batch(batch_dir, jobs, parallel)

        output_manager._collect_batch(batch_dir, jobs, parallel)

        output_manager._clean_batch(batch_dir, jobs, parallel)


def _pymoo_evaluate(
    *candidate_vecs: AnyCandidateVec,
    input_manager: _InputManager,
    output_manager: _OutputManager,
    batch_dir: Path,
) -> tuple[AnyBatchOutputs, AnyBatchOutputs]:
    _evaluate(
        *candidate_vecs,
        input_manager=input_manager,
        output_manager=output_manager,
        batch_dir=batch_dir,
    )

    return (
        output_manager._recorded_objectives(batch_dir),
        output_manager._recorded_constraints(batch_dir),
    )
