from __future__ import annotations

import abc
import itertools as it
import os
import typing
from pathlib import Path
from typing import TYPE_CHECKING, cast, overload

import tomlkit as toml
from tomlkit.items import AoT, Array, Table

from sober.input import _Modifier, _Tagger
from sober.output import _Collector

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Literal, Protocol, TypeAlias, TypedDict, TypeGuard

    from tomlkit.items import String

    from sober._typing import AnyModifier, AnyTagger

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

        # names in key have to be either a property or a subset of cls's __slots__
        non_property_names = tuple(
            item for item in names if not isinstance(getattr(cls, item), property)
        )
        if not (set(non_property_names) <= set(item.__slots__)):
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
def _toml_encoder(value: Path) -> String: ...
@overload
def _toml_encoder(value: frozenset[object]) -> Array: ...
@overload
def _toml_encoder(value: AnyTagger | AnyModifier | _Collector) -> Table: ...
@toml.register_encoder
def _toml_encoder(
    value: Path | frozenset[object] | AnyModifier | _Collector | AnyTagger,
) -> String | Array | Table:
    if isinstance(value, Path):
        return toml.item(os.fsdecode(value))
    elif isinstance(value, frozenset):
        # NOTE: can add other frozenset item types if need be
        assert all(isinstance(item, str) for item in value)
        value = cast(frozenset[str], value)  # mypy: python/mypy#13069
        return toml.item(sorted(value))
    elif isinstance(value, _Tagger | _Modifier | _Collector):
        table = toml.table()

        cls = type(value)
        table.add("class", (cls.__module__, cls.__qualname__))
        table.update(_to_toml_table(value))

        return table
    else:
        return toml.item(str(value))  # TODO: delete
        raise TypeError  # tomlkit handles this exception


def _to_toml_table(obj: object) -> Table:
    init_attr_map = _init_attr_map(type(obj))

    table = toml.table()

    for name in it.chain(
        init_attr_map["_ARG_NAMES"],
        init_attr_map["_STAR_ARG_NAMES"],
        init_attr_map["_KWARG_NAMES"],
    ):
        value = getattr(obj, name)
        item = toml.item(value)
        table.add(name.removeprefix("_"), item)

    for name in init_attr_map["_GETATTR_NAMES"]:
        getattr_obj = getattr(obj, name)
        getattr_table = _to_toml_table(getattr_obj)

        # below handles modifiers and collectors as AoT
        # which are not handled properly as their sequence is parsed before them
        # such that tomlkit is not aware they are tables beforehand
        for getattr_key, getattr_value in getattr_table.items():
            is_array = isinstance(getattr_value, Array)

            if is_array and all(isinstance(item, Table) for item in getattr_value):
                # sequence of modifiers or collectors
                table.add(getattr_key, AoT(getattr_value))
            elif is_array and any(isinstance(item, Table) for item in getattr_value):
                # should never go in here
                raise TypeError
            else:
                # others
                table.add(getattr_key, getattr_value)

    return table
