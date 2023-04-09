import sys
from pathlib import Path
from itertools import starmap
from multiprocessing.pool import Pool
from uuid import NAMESPACE_X500, uuid5
from multiprocessing import get_context
from subprocess import PIPE, STDOUT, run
from contextlib import AbstractContextManager
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, TypeVar

from typing_extensions import Unpack  # TODO: remove Unpack after 3.11
from typing_extensions import TypeVarTuple  # NOTE: from typing after 3.11

from ._logger import _log
from ._typing import AnyCmdArgs


def _uuid(*description: str) -> str:
    return str(uuid5(NAMESPACE_X500, "-".join(description)))


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


# [1] quite a few mypy complaints due to typeshed,
#     stemmed from the implementation of starmap/starimap
#     (related but stale: python/cpython#72567),
#     a lot of private involved, so not likely to be PRed to typeshed
if TYPE_CHECKING:  # [1]
    starmapstar: Callable
    from multiprocessing.pool import IMapIterator as IMapIterator_

    class IMapIterator(IMapIterator_):
        _job: int
        _set_length: Callable

else:
    from multiprocessing.pool import IMapIterator, starmapstar


class _Pool(Pool):
    if TYPE_CHECKING:  # [1]
        from queue import SimpleQueue

        _processes: int
        _check_running: Callable
        _get_tasks: Callable
        _taskqueue: SimpleQueue
        _guarded_task_generation: Callable

    def __init__(
        self,
        processes: int,
        initializer: Callable[[Unpack[InitArgs]], None] | None,  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
        initargs: tuple[Unpack[InitArgs]],  # type: ignore[misc, assignment] # python/mypy#12280 # TODO: Unpack -> * after 3.11
    ) -> None:
        super().__init__(
            processes, initializer, initargs, context=_MULTIPROCESSING_CONTEXT
        )

    def map_(self, func: Callable[[_P], _R], iterable: Iterable[_P]) -> Iterable[_R]:
        return super().imap(func, iterable, 1)

    def starmap_(
        self, func: Callable[..., _R], iterable: Iterable[Iterable[Any]]
    ) -> Iterable[_R]:
        # borrowed from https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
        self._check_running()

        task_batches = self._get_tasks(func, iterable, 1)
        result = IMapIterator(self)
        self._taskqueue.put(
            (
                self._guarded_task_generation(result._job, starmapstar, task_batches),
                result._set_length,
            )
        )
        return (item for chunk in result for item in chunk)


class _Loop(AbstractContextManager):
    def __enter__(self) -> "_Loop":  # TODO: use typing.Self after 3.11
        return self

    def __exit__(self, *args) -> None:
        pass

    def map_(self, func: Callable[[_P], _R], iterable: Iterable[_P]) -> Iterable[_R]:
        return map(func, iterable)

    def starmap_(
        self, func: Callable[..., _R], iterable: Iterable[Iterable[Any]]
    ) -> Iterable[_R]:
        return starmap(func, iterable)


def _Parallel(
    n_processes: int,
    initializer: Callable[[Unpack[InitArgs]], None] | None = None,  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
    initargs: tuple[Unpack[InitArgs]] = (),  # type: ignore[misc,assignment] # python/mypy#12280 # TODO: Unpack -> * after 3.11
) -> _Pool | _Loop:
    if n_processes > 1:
        return _Pool(n_processes, initializer, initargs)
    else:
        return _Loop()
