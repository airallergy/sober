import sys
from pathlib import Path
from itertools import starmap
from multiprocessing.pool import Pool
from multiprocessing import get_context
from subprocess import PIPE, STDOUT, run
from contextlib import AbstractContextManager
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, TypeVar

from typing_extensions import Unpack  # TODO: remove Unpack after 3.11
from typing_extensions import TypeVarTuple  # NOTE: from typing after 3.11

from ._logger import _log
from ._typing import AnyCmdArgs


def _run(cmd_args: AnyCmdArgs, cwd: Path) -> None:
    cmd_args = tuple(
        item.resolve(strict=True) if isinstance(item, Path) else item
        for item in cmd_args
    )
    with _log(cwd, caller_depth=1, cmd_args=cmd_args) as l:
        l.res = run(cmd_args, stdout=PIPE, stderr=STDOUT, cwd=cwd, text=True)


# this bit is purely to make mypy happy :(
if sys.platform != "win32":
    _MULTIPROCESSING_CONTEXT = get_context("forkserver")
else:
    _MULTIPROCESSING_CONTEXT = get_context("spawn")

InitArgs = TypeVarTuple("InitArgs")  # type: ignore[misc] # TODO: after 3.11
_P = TypeVar("_P")  # TODO: python/mypy#11855, python/typeshed#4827
_R = TypeVar("_R", covariant=True)


class _Pool(Pool):
    if TYPE_CHECKING:
        _processes: int  # make mypy happy

    def __init__(
        self,
        processes: int,
        initializer: Callable[[Unpack[InitArgs]], None] | None,  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
        initargs: tuple[Unpack[InitArgs]],  # type: ignore[misc, assignment] # python/mypy#12280 # TODO: Unpack -> * after 3.11
    ) -> None:
        super().__init__(
            processes, initializer, initargs, context=_MULTIPROCESSING_CONTEXT
        )

    def _chunk_size(self, n_tasks: int) -> int:
        return max(n_tasks // self._processes, 1)

    def map_(self, func: Callable[[_P], _R], iterable: Iterable[_P]) -> Iterable[_R]:
        return super().imap(
            func, x := tuple(iterable), chunksize=self._chunk_size(len(x))
        )

    def starmap_(
        self, func: Callable[..., _R], iterable: Iterable[Iterable[Any]]
    ) -> list[_R]:
        return super().starmap(
            func, x := tuple(iterable), chunksize=self._chunk_size(len(x))
        )


class _Loop(AbstractContextManager):
    def __enter__(self) -> "_Loop":  # TODO: use typing.Self after 3.11
        return self

    def __exit__(self, *args) -> None:
        pass

    def map_(self, func: Callable[[_P], _R], iterable: Iterable[_P]) -> Iterable[_R]:
        return map(func, iterable)

    def starmap_(
        self, func: Callable[..., _R], iterable: Iterable[Iterable[Any]]
    ) -> list[_R]:
        return list(starmap(func, iterable))


def _Parallel(
    n_processes: int,
    initializer: Callable[[Unpack[InitArgs]], None] | None = None,  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
    initargs: tuple[Unpack[InitArgs]] = (),  # type: ignore[misc,assignment] # python/mypy#12280 # TODO: Unpack -> * after 3.11
) -> _Pool | _Loop:
    if n_processes > 1:
        return _Pool(n_processes, initializer, initargs)
    else:
        return _Loop()
