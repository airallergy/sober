from pathlib import Path
from itertools import product

import numpy as np

from ._evaluator import _evaluate
from .results import _ResultsManager
from .parameters import AnyIntParameter, _ParametersManager


def _multiply(
    parameters_manager: _ParametersManager[AnyIntParameter],
    results_manager: _ResultsManager,
    evaluation_directory: Path,
    sample_size: int,
    seed: int | None,
) -> None:
    search_space = np.fromiter(
        product(
            range(parameters_manager._weather._n_variations),
            *(
                range(parameter._n_variations)
                for parameter in parameters_manager._parameters
            ),
        ),
        dtype=tuple,
    )

    if sample_size <= 0:
        sample_slice = np.arange(len(search_space))
    else:
        rng = np.random.default_rng(seed)
        sample_slice = rng.choice(len(search_space), sample_size, replace=False)
        sample_slice.sort()

    _evaluate(
        *search_space[sample_slice],
        parameters_manager=parameters_manager,
        results_manager=results_manager,
        batch_directory=evaluation_directory,
    )
