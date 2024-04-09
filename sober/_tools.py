import csv
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager
from itertools import starmap
from math import log10
from multiprocessing import get_context
from multiprocessing.pool import Pool
from pathlib import Path
from subprocess import PIPE, STDOUT, run
from typing import TYPE_CHECKING, Any, Self, TypeAlias, TypeVar, TypeVarTuple
from uuid import NAMESPACE_X500, uuid5

from sober._logger import _log
from sober._typing import AnyCmdArgs


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


def _write_records(
    record_file: Path, header_row: Iterable[Any], *record_rows: Iterable[Any]
) -> None:
    with record_file.open("wt", newline="") as fp:
        writer = csv.writer(fp, dialect="excel")

        # write header
        writer.writerow(header_row)
        # write values
        writer.writerows(record_rows)


def _rectified_str_iterable(s: str | Iterable[str]) -> tuple[str, ...]:
    """converts str or an iterable of str to a tuple of str"""
    if isinstance(s, str):
        return (s,)
    else:
        return tuple(s)


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

# [1] quite a few mypy complaints due to typeshed,
#     stemmed from the implementation of starmap/starimap
#     (related but stale: python/cpython#72567),
#     a lot of private involved, so not likely to be PRed into typeshed
if TYPE_CHECKING:  # [1]
    starmapstar: Callable
    from multiprocessing.pool import IMapIterator as IMapIterator_

    class IMapIterator(IMapIterator_):
        _job: int
        _set_length: Callable

else:
    from multiprocessing.pool import IMapIterator, starmapstar

##############################  module typing  ##############################
_InitArgs = TypeVarTuple("_InitArgs")
_P = TypeVar("_P", contravariant=True)
_R = TypeVar("_R", covariant=True)
#############################################################################


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
        initialiser: Callable[[*_InitArgs], None] | None,
        initargs: tuple[*_InitArgs],
    ) -> None:
        super().__init__(
            processes, initialiser, initargs, context=_MULTIPROCESSING_CONTEXT
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

    __slots__ = ()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        pass

    def _map(self, func: Callable[[_P], _R], iterable: Iterable[_P]) -> Iterator[_R]:
        return map(func, iterable)

    def _starmap(
        self, func: Callable[..., _R], iterable: Iterable[Iterable[Any]]
    ) -> Iterator[_R]:
        return starmap(func, iterable)


##############################  module typing  ##############################
AnyParallel: TypeAlias = _Pool | _Loop
#############################################################################


def _Parallel(  # noqa: N802
    n_processes: int,
    initialiser: Callable[[*_InitArgs], None] | None = None,
    initargs: tuple[*_InitArgs] = (),  # type:ignore[assignment] # TODO: wait to see the multiprocessing typeshed
) -> AnyParallel:
    """a helper function to distribute parallel computation
    based on the requested number of processes"""

    # allows n_processes <= 0 for now
    if n_processes > 1:
        return _Pool(n_processes, initialiser, initargs)
    else:
        return _Loop()
