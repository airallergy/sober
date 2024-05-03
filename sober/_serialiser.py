from __future__ import annotations

import itertools as it


def _slots(cls: type) -> tuple[str, ...]:
    return tuple(
        it.chain.from_iterable(getattr(cls, "__slots__", ()) for cls in cls.__mro__)
    )
