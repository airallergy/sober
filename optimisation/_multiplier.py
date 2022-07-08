from pathlib import Path
from itertools import product

from .results import _ResultsManager
from ._evaluator import _product_evaluate
from .parameters import AnyIntParameter, _ParametersManager


def _multiply(
    parameters_manager: _ParametersManager[AnyIntParameter],
    results_manager: _ResultsManager,
    evaluation_directory: Path,
) -> None:
    variation_idxs_iter = product(
        range(parameters_manager._weather._n_variations),
        *(
            range(parameter._n_variations)
            for parameter in parameters_manager._parameters
        ),
    )
    jobs = tuple(parameters_manager._jobs(*variation_idxs_iter))  # type: ignore[arg-type] # might be resolved after python/mypy#12280

    _product_evaluate(jobs, parameters_manager, results_manager, evaluation_directory)
