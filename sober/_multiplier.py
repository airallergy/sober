import math
import operator
from pathlib import Path
from itertools import accumulate
from typing import Generic, TypeVar, overload
from collections.abc import Iterable, Collection

import numpy as np

from ._evaluator import _evaluate
from .results import _ResultsManager
from .parameters import AnyIntParameter, _ParametersManager

_T = TypeVar("_T")


class _LazyCartesianProduct(Generic[_T]):
    # adapted from: https://github.com/tylerburdsall/lazy-cartesian-product-python
    # this allows indexing a Cartesian product without evaluating all
    # which enables super fast sampling
    def __init__(self, *iterables: Iterable[_T]):
        # self._tuples = tuple(map(tuple, iterables))  # python/mypy#11682
        self._tuples = tuple(tuple(item) for item in iterables)
        self._n_tuples = len(self._tuples)

        tuple_lens = tuple(map(len, self._tuples))
        self._n_products = math.prod(tuple_lens)
        self._divs = tuple(accumulate(tuple_lens[::-1], operator.mul, initial=1))[
            -2::-1
        ]
        self._mods = tuple_lens

    @overload
    def __getitem__(self, key: int) -> tuple[_T, ...]:
        ...

    @overload
    def __getitem__(self, key: Collection[int]) -> tuple[tuple[_T, ...], ...]:
        ...

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < -self._n_products or key > self._n_products - 1:
                raise IndexError("index out of range.")
            elif key < 0:
                key += self._n_products

            return tuple(
                self._tuples[idx][key // self._divs[idx] % self._mods[idx]]
                for idx in range(self._n_tuples)
            )
        elif isinstance(key, Collection) and all(isinstance(item, int) for item in key):
            return tuple(self[item] for item in key)
        else:
            raise TypeError("index must be integers or a collection of integers.")


def _multiply(
    parameters_manager: _ParametersManager[AnyIntParameter],
    results_manager: _ResultsManager,
    evaluation_directory: Path,
    sample_size: int,
    seed: int | None,
) -> None:
    ns_variations = tuple(parameter._n_variations for parameter in parameters_manager)
    full_search_space = _LazyCartesianProduct(*map(range, ns_variations))
    n_products = full_search_space._n_products

    rng = np.random.default_rng(seed)

    if sample_size < 0:
        if n_products > 1e7:
            raise NotImplementedError(
                f"a search space of more than 1e7 candidates is forbidden due to high computing cost: {n_products}."
            )

        sample_idxs = tuple(range(n_products))
        search_space = full_search_space[sample_idxs]
    elif sample_size == 0:
        max_n_variations = max(ns_variations)

        # permute variations of each paramter
        permuted = tuple(map(rng.permutation, ns_variations))
        # fill each row to the longest one by cycling
        filled = np.asarray(
            tuple(np.resize(row, max_n_variations) for row in permuted), dtype=np.int_
        )

        search_space = tuple(tuple(map(int, row)) for row in filled.T)
    else:
        sample_idxs_ = rng.choice(n_products, sample_size, replace=False)
        sample_idxs_.sort()
        sample_idxs = tuple(map(int, sample_idxs_))
        search_space = full_search_space[sample_idxs]

    _evaluate(
        *search_space,
        parameters_manager=parameters_manager,
        results_manager=results_manager,
        batch_directory=evaluation_directory,
    )
