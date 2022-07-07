from pathlib import Path
from itertools import product

from .results import _ResultsManager
from .parameters import AnyIntParameter, _ParametersManager
from ._evaluator import _product_evaluate, _parallel_evaluate


def _multiply(
    tagged_model: str,
    parameters_manager: _ParametersManager[AnyIntParameter],
    results_manager: _ResultsManager,
    evaluation_directory: Path,
    model_type: str,
) -> None:
    variation_idxs_iter = product(
        range(parameters_manager._weather._n_variations),
        *(
            range(parameter._n_variations)
            for parameter in parameters_manager._parameters
        ),
    )
    jobs = tuple(parameters_manager._jobs(*variation_idxs_iter))  # type: ignore[arg-type] # might be resolved after python/mypy#12280

    _parallel_evaluate(
        _product_evaluate,
        jobs,
        parameters_manager,
        results_manager,
        evaluation_directory,
        tagged_model=tagged_model,
        model_type=model_type,
    )
