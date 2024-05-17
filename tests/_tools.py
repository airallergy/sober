from __future__ import annotations

import inspect
from types import FunctionType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any, TypeAlias, TypeGuard, TypeVar

    # borrowed from typeshed
    _T = TypeVar("_T")
    _GetObjsPredicateTypeGuard: TypeAlias = Callable[[Any], TypeGuard[_T]]
    _GetObjsReturnTypeGuard: TypeAlias = list[_T]


def _is_cls_or_func(obj: object) -> TypeGuard[type[Any] | FunctionType]:
    # NOTE: the return typing is not useful due to list invariance
    #       since _get_local_objs will return a mixture of both types
    return isinstance(obj, type | FunctionType)


def _get_local_objs(
    module: ModuleType, predicate: _GetObjsPredicateTypeGuard[_T]
) -> _GetObjsReturnTypeGuard[_T]:
    return [
        obj
        for _, obj in inspect.getmembers(module, predicate)
        if inspect.getmodule(obj) is module
    ]
