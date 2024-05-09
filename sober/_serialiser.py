from __future__ import annotations

import abc
import itertools as it
import os
import typing
from pathlib import Path
from typing import TYPE_CHECKING, cast, overload

import inflection
import tomlkit as toml

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Literal, Protocol, TypeAlias, TypedDict, TypeGuard

    class _SupportsSlots(Protocol):
        __slots__: tuple[str, ...] = ()

    class _SupportsSober(_SupportsSlots, Protocol):
        pass  # python/mypy#5504

    class _InitAttrMap(TypedDict):
        _ARG_NAMES: tuple[str, ...]
        _STAR_ARG_NAMES: tuple[str, ...]
        _KWARG_NAMES: tuple[str, ...]
        _GETATTR_NAMES: tuple[str, ...]

    _AnyCoreInitAttrKey: TypeAlias = Literal["_ARG_NAMES", "_KWARG_NAMES"]
    _AnyInitAttrKey: TypeAlias = Literal[
        _AnyCoreInitAttrKey, "_STAR_ARG_NAMES", "_GETATTR_NAMES"
    ]

    # NOTE: this is incomplete for simplicity, e.g. sober has not datetime objects
    #       may need to extend after python/mypy#7981
    _AnyTOMLItemisable: TypeAlias = bool | int | float | str


def _all_sober_classes(classes: list[type]) -> TypeGuard[list[_SupportsSober]]:
    return all(cls.__module__.split(".")[0] == "sober" for cls in classes)


def _sober_mro(cls: type) -> list[_SupportsSober]:
    """gets and trims mro for sober classes"""

    # get mro
    classes = cls.mro()

    # pop the last one, which is definitely 'object'
    assert classes.pop() is object

    # pop the last two ('Generic', 'ABC') or one ('ABC') or zero
    match classes[-1]:
        case typing.Generic:
            classes.pop()
            assert classes.pop() is abc.ABC
        case abc.ABC:
            classes.pop()
        case _:
            pass

    # guard the remaining to _SupportsSober
    if _all_sober_classes(classes):
        return classes
    else:
        raise TypeError


def _mro_init_attr_names(cls: type, key: _AnyInitAttrKey) -> Iterator[tuple[str, ...]]:
    """iterates init attribute names over mro"""

    for item in _sober_mro(cls):
        # cls has to have the key
        if not hasattr(item, key):
            continue

        names = getattr(item, key)
        names = cast(tuple[str, ...], names)  # mypy: python/mypy#5504

        # names in key have to be a subset of cls's __slots__
        if not (set(names) <= set(item.__slots__)):
            continue

        yield names


def _init_attr_names(cls: type, key: _AnyInitAttrKey) -> tuple[str, ...]:
    """retraces init attribute names"""

    # concatenate mro attr names
    iterables = _mro_init_attr_names(cls, key)
    if key == "_STAR_ARG_NAMES":
        # only _STAR_ARG_NAMES in the leftmost mro is accounted
        # NOTE: '_STAR_ARG_NAMES' is overriding
        #       whilst other keys are concatenatig like __slots__
        attr_names = list(next(iterables, ()))

        # expected to only apply to _bounds vs _options
        match next(iterables, 0):
            case ("_bounds",):
                assert next(iterables, 0) == 0
            case 0:
                pass
            case _:
                raise ValueError
    else:
        attr_names = list(it.chain(*iterables))

    # only leaf classes should have _EXCLUDE_NAMES
    for item in _sober_mro(cls)[1:]:
        assert not hasattr(item, "_EXCLUDE_NAMES")

    # remove names in _EXCLUDE_NAMES if any
    for name in getattr(cls, "_EXCLUDE_NAMES", ()):
        name = cast(str, name)  # mypy: python/mypy#5504

        if name not in attr_names:
            continue

        assert attr_names.count(name) == 1
        attr_names.remove(name)

    return tuple(attr_names)


def _init_attr_map(cls: type) -> _InitAttrMap:
    """retraces a map for init attribute names"""

    return {
        "_ARG_NAMES": _init_attr_names(cls, "_ARG_NAMES"),
        "_STAR_ARG_NAMES": _init_attr_names(cls, "_STAR_ARG_NAMES"),
        "_KWARG_NAMES": _init_attr_names(cls, "_KWARG_NAMES"),
        "_GETATTR_NAMES": _init_attr_names(cls, "_GETATTR_NAMES"),
    }


@overload
def _toml_itemisable(value: Path) -> str: ...
@overload
def _toml_itemisable(value: _AnyTOMLItemisable) -> _AnyTOMLItemisable: ...
def _toml_itemisable(value: Path | _AnyTOMLItemisable) -> _AnyTOMLItemisable:
    if isinstance(value, Path):
        return os.fsdecode(value)
    else:
        return value


def _to_toml(obj: object) -> toml.TOMLDocument:
    cls = obj.__class__
    init_attr_map = _init_attr_map(cls)

    doc = toml.document()

    table = toml.table()

    for name in it.chain(
        init_attr_map["_ARG_NAMES"],
        init_attr_map["_STAR_ARG_NAMES"],
        init_attr_map["_KWARG_NAMES"],
    ):
        value = getattr(obj, name)
        item = toml.item(_toml_itemisable(value))
        table.add(name.removeprefix("_"), item)

    for name in init_attr_map["_GETATTR_NAMES"]:
        pass

    doc.add(inflection.underscore(cls.__name__), table)

    return doc
