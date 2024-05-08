from __future__ import annotations

import abc
import itertools as it
import typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal, Protocol, TypedDict, TypeGuard

    class _SupportsSlots(Protocol):
        __slots__: tuple[str, ...] = ()

    class _SupportsSober(_SupportsSlots, Protocol):
        pass  # python/mypy#5504

    class _InitAttrNames(TypedDict):
        _ARG_NAMES: tuple[str, ...]
        _STAR_ARG_NAMES: tuple[str, ...]
        _KWARG_NAMES: tuple[str, ...]


def _all_sober_classes(classes: list[type]) -> TypeGuard[list[_SupportsSober]]:
    return all(cls.__module__.split(".")[0] == "sober" for cls in classes)


def _sober_mro(cls: type) -> list[_SupportsSober]:
    """gets and trims mro for sober classes"""

    # get mro
    classes = cls.mro()

    # pop the last one, which is definitely 'object'
    assert classes.pop() is object

    # pop the last two ('Generic', 'ABC') or one('ABC')
    match classes.pop():
        case typing.Generic:
            assert classes.pop() is abc.ABC
        case abc.ABC:
            pass
        case _:
            raise TypeError

    # guard the remaining to _SupportsSober
    if _all_sober_classes(classes):
        return classes
    else:
        raise TypeError


def _retraced_init_attr_names(
    cls: type, key: Literal["_ARG_NAMES", "_STAR_ARG_NAMES", "_KWARG_NAMES"]
) -> tuple[str, ...]:
    """retraces init attribute names"""
    return tuple(
        it.chain.from_iterable(
            names
            for item in _sober_mro(cls)
            if hasattr(item, key)
            and (set(names := getattr(item, key)) <= set(item.__slots__))
        )
    )


def _retraced_init_attr_names_map(cls: type) -> _InitAttrNames:
    """retraces a map for init attribute names"""
    return {
        "_ARG_NAMES": _retraced_init_attr_names(cls, "_ARG_NAMES"),
        "_STAR_ARG_NAMES": _retraced_init_attr_names(cls, "_STAR_ARG_NAMES"),
        "_KWARG_NAMES": _retraced_init_attr_names(cls, "_KWARG_NAMES"),
    }
