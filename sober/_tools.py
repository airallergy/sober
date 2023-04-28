import sys
from math import log10
from pathlib import Path
from itertools import starmap
from multiprocessing.pool import Pool
from uuid import NAMESPACE_X500, uuid5
from multiprocessing import get_context
from subprocess import PIPE, STDOUT, run
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, TypeVar
from collections.abc import Callable, Iterable, Iterator

from typing_extensions import Unpack  # TODO: remove Unpack after 3.11
from typing_extensions import TypeVarTuple  # NOTE: from typing after 3.11

from ._logger import _log
from ._typing import AnyCmdArgs


#############################################################################
#######                    MISCELLANEOUS FUNCTIONS                    #######
#############################################################################
def _natural_width(x: int) -> int:
    """returns the digit count of a natural number"""

    assert isinstance(x, int) and x > 0

    return int(log10(x)) + 1


def _uuid(*feature_group: str) -> str:
    """an uuid generator"""

    return str(uuid5(NAMESPACE_X500, "-".join(feature_group)))


def _run(cmd_args: AnyCmdArgs, cwd: Path) -> None:
    """a helper function for subprocess.run to enable logging"""

    # run subprocess and pass the result object to logging
    with _log(cwd, caller_depth=1, cmd_args=cmd_args) as l:
        l._result = run(cmd_args, stdout=PIPE, stderr=STDOUT, cwd=cwd, text=True)


#############################################################################
#######                      PARALLEL FUNCTIONS                       #######
#############################################################################
# Common:
#     1. typing map/imap and starmap/starimap follows multiprocessing
#     2. chunksize is set to 1 for performance
#        a larger chunksize drops performance, possibly because the progress tracking

# get multiprocessing context
# follow the use of sys.platform by multiprocessing, see also python/mypy#8166
# don't use fork on posix, better safe than sorry
if sys.platform != "win32":
    _MULTIPROCESSING_CONTEXT = get_context("forkserver")
else:
    _MULTIPROCESSING_CONTEXT = get_context("spawn")

InitArgs = TypeVarTuple("InitArgs")  # type: ignore[misc] # TODO: after 3.11
_P = TypeVar("_P", contravariant=True)
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
    """a helper class for multiprocessing.Pool
    this includes setting defaults, unifying method names and implementing starimap"""

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

    def _map(self, func: Callable[[_P], _R], iterable: Iterable[_P]) -> Iterator[_R]:
        return super().imap(func, iterable, 1)

    def _starmap(
        self, func: Callable[..., _R], iterable: Iterable[Iterable[Any]]
    ) -> Iterator[_R]:
        """an implementation of starimap
        borrowed from https://stackoverflow.com/a/57364423"""

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
    """a helper class for loop
    this includes making a context manager and unifying method names"""

    def __enter__(self) -> "_Loop":  # TODO: use typing.Self after 3.11
        return self

    def __exit__(self, *args) -> None:
        pass

    def _map(self, func: Callable[[_P], _R], iterable: Iterable[_P]) -> Iterator[_R]:
        return map(func, iterable)

    def _starmap(
        self, func: Callable[..., _R], iterable: Iterable[Iterable[Any]]
    ) -> Iterator[_R]:
        return starmap(func, iterable)


def _Parallel(
    n_processes: int,
    initializer: Callable[[Unpack[InitArgs]], None] | None = None,  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
    initargs: tuple[Unpack[InitArgs]] = (),  # type: ignore[misc,assignment] # python/mypy#12280 # TODO: Unpack -> * after 3.11
) -> _Pool | _Loop:
    """a helper function to distribute parallel computation
    based on the requested number of processes"""

    # allows n_processes <= 0 for now
    if n_processes > 1:
        return _Pool(n_processes, initializer, initargs)
    else:
        return _Loop()
