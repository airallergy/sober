import inspect
import itertools as it
import math
import operator
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Generic, TypeGuard, TypeVar, cast, overload

import numpy as np

from sober._evaluator import _evaluate
from sober._io_managers import _InputManager, _OutputManager
from sober.input import AnyModifierVal, _IntegralModifier

##############################  module typing  ##############################
_T = TypeVar("_T")


def each_item_is_non_empty(
    args: tuple[tuple[_T, ...], ...],
) -> TypeGuard[tuple[tuple[_T, *tuple[_T, ...]], ...]]:
    return all(len(item) >= 1 for item in args)


#############################################################################


class _LazyCartesianProduct(Generic[_T]):
    """allows indexing a Cartesian product without evaluating all
    which enables super fast sampling
    adapted from: https://github.com/tylerburdsall/lazy-cartesian-product-python"""

    _tuples: tuple[tuple[_T, ...], ...]
    _n_tuples: int
    _n_products: int
    _divs: tuple[int, ...]
    _mods: tuple[int, ...]

    __slots__ = ("_tuples", "_n_tuples", "_n_products", "_divs", "_mods")

    def __init__(self, *iterables: Iterable[_T]):
        self._tuples = tuple(tuple(item) for item in iterables)  # python/mypy#11682
        self._n_tuples = len(self._tuples)

        tuple_lens = tuple(map(len, self._tuples))
        self._n_products = math.prod(tuple_lens)
        self._divs = tuple(it.accumulate(tuple_lens[::-1], operator.mul, initial=1))[
            -2::-1
        ]
        self._mods = tuple_lens

    @overload
    def __getitem__(self, key: int) -> tuple[_T, ...]: ...
    @overload
    def __getitem__(self, key: Sequence[int]) -> tuple[tuple[_T, ...], ...]: ...
    def __getitem__(self, key):
        if isinstance(key, int):
            if key < -self._n_products or key > self._n_products - 1:
                raise IndexError("index out of range.")
            elif key < 0:
                key += self._n_products

            return tuple(
                self._tuples[i][key // self._divs[i] % self._mods[i]]
                for i in range(self._n_tuples)
            )
        elif isinstance(key, Sequence) and all(isinstance(item, int) for item in key):
            return tuple(self[item] for item in key)
        else:
            raise TypeError("index must be integers or a collection of integers.")


def _multiply(
    input_manager: _InputManager,
    output_manager: _OutputManager,
    evaluation_dir: Path,
    sample_size: int,
    seed: int | None,
) -> None:
    """populates parametrics by subsetting the full search space"""
    if input_manager._has_real_ctrls:
        frames = inspect.stack()
        caller_name = ""
        for item in frames[1:]:
            if item.function.startswith("run_") and ("self" in item.frame.f_locals):
                caller_name = item.function
            else:
                assert caller_name
                break

        raise NotImplementedError(
            f"'{caller_name}' for real control variables has yet to be implemented."
        )

    ctrl_lens = tuple(
        len(cast(_IntegralModifier[AnyModifierVal], item))
        if item._is_ctrl
        else item._hype_ctrl_len()
        for item in input_manager
    )
    search_space = _LazyCartesianProduct(*map(range, ctrl_lens))
    n_products = search_space._n_products

    rng = np.random.default_rng(seed)

    if sample_size < 0:
        # brute force

        if n_products > 1e7:
            raise NotImplementedError(
                f"a search space of more than 1e7 candidates is forbidden due to high computing cost: {n_products}."
            )

        sample_idxes = tuple(range(n_products))
        ctrl_key_vecs = search_space[sample_idxes]
    elif sample_size == 0:
        # test each control options with fewest simulations

        max_ctrl_len = max(ctrl_lens)

        # permute control keys
        permuted = tuple(map(rng.permutation, ctrl_lens))

        # fill each row to the longest one by cycling
        filled = np.asarray(
            tuple(np.resize(row, max_ctrl_len) for row in permuted), dtype=np.int_
        )

        ctrl_key_vecs = tuple(tuple(map(int, row)) for row in filled.T)
    else:
        # proper subset

        sample_idxes_ = rng.choice(n_products, sample_size, replace=False)
        sample_idxes_.sort()
        sample_idxes = tuple(map(int, sample_idxes_))
        ctrl_key_vecs = search_space[sample_idxes]

    if each_item_is_non_empty(ctrl_key_vecs):
        _evaluate(
            *ctrl_key_vecs,
            input_manager=input_manager,
            output_manager=output_manager,
            batch_dir=evaluation_dir,
        )
    else:
        # impossible, there is at least the weather modifer
        raise IndexError("no modifiers are defined.")
