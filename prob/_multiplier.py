from pathlib import Path
from itertools import product

from ._evaluator import _evaluate
from .results import _ResultsManager
from .parameters import AnyIntParameter, _ParametersManager


def _multiply(
    parameters_manager: _ParametersManager[AnyIntParameter],
    results_manager: _ResultsManager,
    evaluation_directory: Path,
) -> None:
    _evaluate(
        *product(  # type: ignore[arg-type] # might be resolved after python/mypy#12280
            range(parameters_manager._weather._n_variations),
            *(
                range(parameter._n_variations)
                for parameter in parameters_manager._parameters
            ),
        ),
        parameters_manager=parameters_manager,
        results_manager=results_manager,
        batch_directory=evaluation_directory,
    )
