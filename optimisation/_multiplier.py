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
    _product_evaluate(
        *product(
            range(parameters_manager._weather._n_variations),
            *(
                range(parameter._n_variations)
                for parameter in parameters_manager._parameters
            ),
        ),
        parameters_manager=parameters_manager,
        results_manager=results_manager,
        evaluation_directory=evaluation_directory,
    )
