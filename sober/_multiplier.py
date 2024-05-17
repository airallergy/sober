from __future__ import annotations

import inspect
import itertools as it
import math
import operator
import shutil
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar, cast

import numpy as np
import scipy.stats.qmc

from sober._evaluator import _evaluate
from sober._tools import _pre_evaluation_hook
from sober._typing import AnyModifierValue
from sober.input import _IntegralModifier

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from typing import Any, Final, TypeGuard

    from sober._io_managers import _InputManager, _OutputManager


##############################  module typing  ##############################
# https://github.com/python/typing/issues/60#issuecomment-869757075
# this can be removed with the new type syntax from py3.12
_T = TypeVar("_T")
#############################################################################


def _each_tuple_is_non_empty_and_starts_with_int(
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
    """an abstract base class for multipliers"""

    _HAS_BATCHES: Final = False

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
    def __call__(self, *proxies: Any) -> None: ...

    @abstractmethod
    def _check_args(self) -> None: ...

    def _prepare(self) -> None:
        self._check_args()

    @_pre_evaluation_hook
    def _evaluate(self, *ctrl_key_vecs: tuple[float, ...]) -> None:
        if _each_tuple_is_non_empty_and_starts_with_int(ctrl_key_vecs):
            batch = _evaluate(
                *ctrl_key_vecs,
                input_manager=self._input_manager,
                output_manager=self._output_manager,
                batch_dir=self._evaluation_dir,
            )
        else:
            # impossible, there is at least the weather modifer
            raise IndexError("no modifiers are defined.")

        if self._output_manager._removes_subdirs:
            for job_uid, _ in batch:
                shutil.rmtree(self._evaluation_dir / job_uid)


#############################################################################
#######                      ELEMENTWISE PRODUCT                      #######
#############################################################################
class _InverseTransformQuantile:
    """generates quantiles for inverse transform sampling"""

    __slots__ = ("_n_dims",)

    _n_dims: int

    def __init__(self, n_dims: int) -> None:
        self._n_dims = n_dims

    def _random(self, size: int, seed: int | None) -> list[list[float]]:
        rng = np.random.default_rng(seed)

        sample_quantile_vecs = rng.uniform(size=(self._n_dims, size)).tolist()

        # cast: numpy/numpy#16544
        return cast(list[list[float]], sample_quantile_vecs)

    def _latin_hypercube(self, size: int, seed: int | None) -> list[list[float]]:
        rng = np.random.default_rng(seed)

        sampler = scipy.stats.qmc.LatinHypercube(self._n_dims, seed=rng)
        sample_quantile_vecs = sampler.random(size).T.tolist()

        # cast: numpy/numpy#16544
        return cast(list[list[float]], sample_quantile_vecs)


class _ElementwiseMultiplier(_Multiplier):
    """samples an elementwise product"""

    __slots__ = ("_quantile",)

    _quantile: _InverseTransformQuantile

    def __call__(self, *proxies: list[float]) -> None:
        n_repeats = len(proxies[0]) if self._input_manager._has_ctrls else 1

        ctrl_key_vecs = tuple(
            zip(
                *(
                    item._key_icdf(*quantiles)
                    if item._is_ctrl
                    else it.repeat(item._hype_ctrl_key(), n_repeats)
                    for item, quantiles in zip(
                        self._input_manager, proxies, strict=True
                    )
                ),
                strict=True,
            )
        )

        # cast: python/mypy#5247
        ctrl_key_vecs = cast(tuple[tuple[float | int, ...], ...], ctrl_key_vecs)

        self._evaluate(*ctrl_key_vecs)

    def _check_args(self) -> None:
        pass

    def _prepare(self) -> None:
        super()._prepare()

        # set the quantile
        self._quantile = _InverseTransformQuantile(len(self._input_manager))

    def _random(self, size: int, seed: int | None) -> None:
        sample_quantile_vecs = self._quantile._random(size, seed)

        self(*sample_quantile_vecs)

    def _latin_hypercube(self, size: int, seed: int | None) -> None:
        sample_quantile_vecs = self._quantile._latin_hypercube(size, seed)

        self(*sample_quantile_vecs)


#############################################################################
#######                       CARTESIAN PRODUCT                       #######
#############################################################################
class _LazyCartesianProduct(Generic[_T]):
    """allows indexing a Cartesian product without evaluating all
    which enables super fast sampling
    inspired by http://phrogz.net/lazy-cartesian-product"""

    __slots__ = ("_tuples", "_divs", "_mods", "_len")

    _tuples: tuple[tuple[_T, ...], ...]
    _divs: tuple[int, ...]
    _mods: tuple[int, ...]
    _len: int

    def __init__(self, *iterables: Iterable[_T]) -> None:
        self._tuples = tuple(map(tuple, iterables))
        tuple_lens = tuple(map(len, self._tuples))

        self._divs = tuple(it.accumulate(tuple_lens[::-1], operator.mul, initial=1))[
            -2::-1
        ]
        self._mods = tuple_lens

        self._len = math.prod(tuple_lens)

    def __getitem__(self, key: int, /) -> tuple[_T, ...]:
        if key < -self._len or key > self._len - 1:
            raise IndexError("index out of range.")

        if key < 0:
            key += self._len

        return tuple(
            self._tuples[i][key // self._divs[i] % self._mods[i]]
            for i in range(len(self._tuples))
        )


class _CartesianMultiplier(_Multiplier):
    """samples a cartesian product"""

    __slots__ = ("_product",)

    _product: _LazyCartesianProduct[int]

    def __call__(self, *proxies: int) -> None:
        if len(proxies) > 1e7:
            # this is almost certainly unintended
            raise NotImplementedError(
                f"a sample size larger than 1e7 is forbidden due to high computing cost: {len(proxies)}."
            )

        ctrl_key_vecs = tuple(self._product[i] for i in proxies)

        self._evaluate(*ctrl_key_vecs)

    def _check_args(self) -> None:
        if self._input_manager._has_real_ctrls:
            frames = inspect.stack()
            caller_name = ""
            #    _CartesianMultiplier._check_args <- _CartesianMultiplier._prepare
            # <- _Multiplier._prepare <- _Multiplier.__init__
            # <- Problem.__getattr__ <- Problem.run_...
            for item in frames[5:]:
                if item.function.startswith("run_") and ("self" in item.frame.f_locals):
                    caller_name = item.function
                else:
                    assert caller_name
                    break

            raise ValueError(
                f"'{caller_name}' is incompatible with real control variables."
            )

    def _prepare(self) -> None:
        super()._prepare()

        # set the lazy cartesian product
        ctrl_lens = tuple(
            len(cast(_IntegralModifier[AnyModifierValue], item))  # mypy
            if item._is_ctrl
            else item._hype_ctrl_len()
            for item in self._input_manager
        )
        self._product = _LazyCartesianProduct(*map(range, ctrl_lens))

    def _random(self, size: int, seed: int | None) -> None:
        len_product = self._product._len

        if size > len_product:
            raise ValueError(
                f"the sample size '{size}' is larger than the outcome count of the sample space '{len_product}'."
            )

        rng = np.random.default_rng(seed)

        sample_indices = rng.choice(len_product, size, replace=False).tolist()

        # cast: numpy/numpy#16544
        sample_indices = cast(list[int], sample_indices)

        self(*sample_indices)

    def _exhaustive(self) -> None:
        len_product = self._product._len

        sample_indices = range(len_product)

        self(*sample_indices)
