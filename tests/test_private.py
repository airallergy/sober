from __future__ import annotations

import itertools as it
import random
from typing import TYPE_CHECKING

import pytest

from sober._multiplier import _LazyCartesianProduct

if TYPE_CHECKING:
    from collections.abc import Iterable


# borrowed from CPython tests for itertools.product
@pytest.mark.parametrize(
    "args",
    [
        (),  # zero iterables
        "ab",  # one iterable
        (range(2), range(3)),  # two iterables
        (range(0), range(2), range(3)),  # first iterable with zero length
        (range(2), range(0), range(3)),  # middle iterable with zero length
        (range(2), range(3), range(0)),  # last iterable with zero length
    ]
    + [
        tuple(
            random.choices(
                (
                    "",
                    "abc",
                    "",
                    range(0),
                    range(4),
                    {"a": 1, "b": 2, "c": 3},
                    set("abcdefg"),
                    range(11),
                    tuple(range(13)),
                ),
                k=random.randrange(7),
            )
        )
        for _ in range(100)
    ],
)
def test_lazy_cartesian_product(args: tuple[Iterable[object]]) -> None:
    product = _LazyCartesianProduct(*args)
    assert tuple(product[i] for i in range(product._len)) == tuple(it.product(*args))
