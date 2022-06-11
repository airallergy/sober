from typing import Iterator, TypeVar


_A = TypeVar("_A", str, tuple, list, dict)


def _checked_empty(*args: _A, arg_type: type) -> Iterator[_A]:
    return (arg_type() if arg is None else arg for arg in args)
