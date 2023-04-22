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
    ns_variations = tuple(parameter._n_variations for parameter in parameters_manager)
    full_search_space = np.asarray(
        tuple(product(*map(range, ns_variations))), dtype=np.int_
    )

    rng = np.random.default_rng(seed)

    if sample_size < 0:
        search_space = full_search_space
    elif sample_size == 0:
        max_n_variations = max(ns_variations)
        search_space = np.asarray(
            tuple(
                np.resize(rng.permutation(n), max_n_variations) for n in ns_variations
            ),
            dtype=np.int_,
        ).T
    else:
        sample_slice = rng.choice(len(full_search_space), sample_size, replace=False)
        search_space = full_search_space[sorted(sample_slice)]

    _evaluate(
        *search_space,
        parameters_manager=parameters_manager,
        results_manager=results_manager,
        batch_directory=evaluation_directory,
    )
