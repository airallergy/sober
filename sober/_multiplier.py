from __future__ import annotations

import inspect
import itertools as it
import math
import operator
from abc import ABC, abstractmethod
from collections.abc import Sequence  # isinstance
from typing import TYPE_CHECKING, Generic, TypeVar, cast, overload

import numpy as np
import scipy.stats.qmc

import sober.config as cf
from sober._evaluator import _evaluate
from sober._typing import AnyModifierVal
from sober.input import _IntegralModifier

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from typing import TypeGuard  # this interestingly needs no runtime import

    from sober._io_managers import _InputManager, _OutputManager
    from sober._typing import AnySampleMethod


##############################  module typing  ##############################
# https://github.com/python/typing/issues/60#issuecomment-869757075
# this can be removed with the new type syntax from py3.12
_T = TypeVar("_T")
#############################################################################


def each_tuple_is_non_empty_and_starts_with_int(
    args: tuple[tuple[_T, ...], ...],
) -> TypeGuard[tuple[tuple[int, *tuple[_T, ...]], ...]]:
    # python/mypy#3497
    # the non empty check may be removed after python/mypy#4573, python/mypy#7853
    return all(len(item) >= 1 for item in args) and all(
        isinstance(item[0], int) for item in args
    )


#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Multiplier(ABC):
    __slots__ = ("_input_manager", "_output_manager", "_evaluation_dir")

    _input_manager: _InputManager
    _output_manager: _OutputManager
    _evaluation_dir: Path

    def __init__(
        self,
        input_manager: _InputManager,
        output_manager: _OutputManager,
        evaluation_dir: Path,
    ) -> None:
        self._input_manager = input_manager
        self._output_manager = output_manager
        self._evaluation_dir = evaluation_dir

        self._prepare()

    @abstractmethod
    def _check_args(self) -> None: ...

    def _prepare(self) -> None:
        self._check_args()

        cf._has_batches = False


#############################################################################
#######                      ELEMENTWISE PRODUCT                      #######
#############################################################################
class _ElementwiseMultiplier(_Multiplier):
    __slots__ = ()

    def _check_args(self) -> None:
        pass

    def _elementwise_multiply(
        self, sample_size: int, sample_method: AnySampleMethod, seed: int | None
    ) -> None:
        rng = np.random.default_rng(seed)
        n_inputs = len(self._input_manager)

        if sample_method == "random":
            quantile_samples = rng.uniform(size=(n_inputs, sample_size)).tolist()
        else:
            sampler = scipy.stats.qmc.LatinHypercube(n_inputs, seed=rng)
            quantile_samples = sampler.random(sample_size).T.tolist()

        quantile_samples = cast(list[list[float]], quantile_samples)  # numpy shape

        ctrl_key_vecs = tuple(
            zip(
                *(
                    item._key_sample(sample)
                    if item._is_ctrl
                    else it.repeat(item._hype_ctrl_key())
                    for item, sample in zip(
                        self._input_manager, quantile_samples, strict=True
                    )
                ),
                strict=False,
            )
        )
        # cast: python/mypy#5247
        ctrl_key_vecs = cast(tuple[tuple[float | int, ...], ...], ctrl_key_vecs)

        if each_tuple_is_non_empty_and_starts_with_int(ctrl_key_vecs):
            _evaluate(
                *ctrl_key_vecs,
                input_manager=self._input_manager,
                output_manager=self._output_manager,
                batch_dir=self._evaluation_dir,
            )
        else:
            # impossible, there is at least the weather modifer
            raise IndexError("no modifiers are defined.")

    def _sample(self, size: int, method: AnySampleMethod, seed: int | None) -> None:
        self._elementwise_multiply(size, method, seed)


#############################################################################
#######                       CARTESIAN PRODUCT                       #######
#############################################################################
class _LazyCartesianProduct(Generic[_T]):
    """allows indexing a Cartesian product without evaluating all
    which enables super fast sampling
    adapted from: https://github.com/tylerburdsall/lazy-cartesian-product-python
    see also: https://medium.com/hackernoon/generating-the-nth-cartesian-product-e48db41bed3f"""

    __slots__ = ("_tuples", "_n_tuples", "_n_products", "_divs", "_mods")

    _tuples: tuple[tuple[_T, ...], ...]
    _n_tuples: int
    _n_products: int
    _divs: tuple[int, ...]
    _mods: tuple[int, ...]

    def __init__(self, *iterables: Iterable[_T]):
        self._tuples = tuple(map(tuple, iterables))
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
    def __getitem__(
        self, key: int | Sequence[int]
    ) -> tuple[_T, ...] | tuple[tuple[_T, ...], ...]:
        if isinstance(key, int):
            if key < -self._n_products or key > self._n_products - 1:
                raise IndexError("index out of range.")

            if key < 0:
                key += self._n_products

            return tuple(
                self._tuples[i][key // self._divs[i] % self._mods[i]]
                for i in range(self._n_tuples)
            )
        elif isinstance(key, Sequence) and all(isinstance(item, int) for item in key):
            return tuple(self[item] for item in key)
        else:
            raise TypeError("index must be integers or a sequence of integers.")


class _CartesianMultiplier(_Multiplier):
    __slots__ = ()

    def _check_args(self) -> None:
        if self._input_manager._has_real_ctrls:
            frames = inspect.stack()
            caller_name = ""
            # _check_args <- _prepare <- __init__ <- __getattr__ <- run_...
            for item in frames[4:]:
                if item.function.startswith("run_") and ("self" in item.frame.f_locals):
                    caller_name = item.function
                else:
                    assert caller_name
                    break

            raise ValueError(
                f"'{caller_name}' is incompatible with real control variables."
            )

    def _cartesian_multiply(self, sample_size: int, seed: int | None) -> None:
        ctrl_lens = tuple(
            len(cast(_IntegralModifier[AnyModifierVal], item))  # mypy
            if item._is_ctrl
            else item._hype_ctrl_len()
            for item in self._input_manager
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

        if each_tuple_is_non_empty_and_starts_with_int(ctrl_key_vecs):
            _evaluate(
                *ctrl_key_vecs,
                input_manager=self._input_manager,
                output_manager=self._output_manager,
                batch_dir=self._evaluation_dir,
            )
        else:
            # impossible, there is at least the weather modifer
            raise IndexError("no modifiers are defined.")

    def _sample(self, size: int, seed: int | None) -> None:
        self._cartesian_multiply(size, seed)
