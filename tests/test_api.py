from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from pymoo.termination.max_gen import MaximumGenerationTermination

import sober
from tests._tools import _get_local_objs, _is_cls_or_func

if TYPE_CHECKING:
    from types import FunctionType
    from typing import Any


def test_all_public_objs_exposed_as_apis() -> None:
    pymoo_apis = {MaximumGenerationTermination}

    apis = [
        obj
        for _, obj in inspect.getmembers(sober, _is_cls_or_func)
        if obj not in pymoo_apis
    ]

    public_objs: list[type[Any] | FunctionType] = []
    for _, module in inspect.getmembers(sober, inspect.ismodule):
        public_objs += (
            obj
            for obj in _get_local_objs(module, _is_cls_or_func)
            if not obj.__name__.startswith("_")
        )

    apis = sorted(apis, key=lambda x: x.__name__)
    public_objs = sorted(public_objs, key=lambda x: x.__name__)

    assert apis == public_objs
