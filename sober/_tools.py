import csv
import functools as ft
import itertools as it
import math
import subprocess as sp
import sys
import uuid
from collections.abc import Callable, Iterable, Iterator
from multiprocessing import get_context
from multiprocessing.pool import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Self, TypeAlias, TypeVar, TypeVarTuple

import sober.config as cf
from sober._logger import _log
from sober._typing import AnyCmdArgs

##############################  module typing  ##############################
_InitArgs = TypeVarTuple("_InitArgs")
_T = TypeVar("_T", contravariant=True)
_R = TypeVar("_R", covariant=True)
#############################################################################


#############################################################################
#######                    MISCELLANEOUS FUNCTIONS                    #######
#############################################################################
def _natural_width(x: int) -> int:
    """returns the digit count of a natural number"""

    assert isinstance(x, int) and x > 0

    return int(math.log10(x)) + 1


def _uuid(*feature_group: str) -> str:
    """an uuid generator"""

    return str(uuid.uuid5(uuid.NAMESPACE_X500, "-".join(feature_group)))


def _run(cmd_args: AnyCmdArgs, cwd: Path) -> None:
    """a helper function for subprocess.run to enable logging"""

    # run subprocess and pass the result object to logging
    with _log(cwd, caller_depth=1, cmd_args=cmd_args) as l:
        l._result = sp.run(
            cmd_args, stdout=sp.PIPE, stderr=sp.STDOUT, cwd=cwd, text=True
        )


def _write_records(
    record_file: Path, header_row: Iterable[Any], *record_rows: Iterable[Any]
) -> None:
    with record_file.open("wt", newline="") as fp:
        writer = csv.writer(fp, dialect="excel")

        # write header
        writer.writerow(header_row)
        # write values
        writer.writerows(record_rows)


@ft.cache  # NOTE: ruff: B019  # TODO: consider generalised to _read_records
def _recorded_batch(batch_dir: Path) -> tuple[tuple[str, ...], ...]:
    # read job records
    with (batch_dir / cf._RECORDS_FILENAMES["job"]).open("rt", newline="") as fp:
        reader = csv.reader(fp, dialect="excel")

        # skip the header row
        next(reader)

        return tuple(map(tuple, reader))


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
    _MULTIPROCESSING_CONTEXT: Final = get_context("forkserver")
else:
    _MULTIPROCESSING_CONTEXT: Final = get_context("spawn")

# [1] quite a few mypy complaints due to typeshed,
#     stemmed from the implementation of starmap/starimap
#     (related but stale: python/cpython#72567),
#     a lot of private involved, so not likely to be PRed into typeshed
if TYPE_CHECKING:  # [1]
    from multiprocessing.pool import IMapIterator as IMapIterator_

    class IMapIterator(IMapIterator_[Any]):
        _job: int
        _set_length: Callable[..., Any]

    starmapstar: Callable[..., Any]
else:
    from multiprocessing.pool import IMapIterator, starmapstar


class _Pool(Pool):
    """a helper class for multiprocessing.pool.Pool
    this includes setting defaults, unifying method names and implementing starimap"""

    if TYPE_CHECKING:  # [1]
        import queue

        _processes: int
        _check_running: Callable[..., Any]
        _get_tasks: Callable[..., Any]
        _taskqueue: queue.SimpleQueue[Any]
        _guarded_task_generation: Callable[..., Any]

    def __init__(
        self,
        processes: int,
        initialiser: Callable[[*_InitArgs], None] | None,
        initargs: tuple[*_InitArgs],
    ) -> None:
        super().__init__(
            processes, initialiser, initargs, context=_MULTIPROCESSING_CONTEXT
        )

    def _map(self, func: Callable[[_T], _R], iterable: Iterable[_T]) -> Iterator[_R]:
        return super().imap(func, iterable, 1)

    def _starmap(
        self, func: Callable[..., _R], iterable: Iterable[Iterable[Any]]
    ) -> Iterator[_R]:
        """an implementation of starimap
        borrowed from https://stackoverflow.com/a/57364423"""

        self._check_running()

        task_batches = _Pool._get_tasks(func, iterable, 1)
        result = IMapIterator(self)
        self._taskqueue.put(
            (
                self._guarded_task_generation(result._job, starmapstar, task_batches),
                result._set_length,
            )
        )
        return (item for chunk in result for item in chunk)


class _Loop:
    """a helper class for loop
    this includes making a context manager and unifying method names"""

    __slots__ = ()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        pass

    def _map(self, func: Callable[[_T], _R], iterable: Iterable[_T]) -> Iterator[_R]:
        return map(func, iterable)

    def _starmap(
        self, func: Callable[..., _R], iterable: Iterable[Iterable[Any]]
    ) -> Iterator[_R]:
        return it.starmap(func, iterable)


#############################  package typing  ##############################
# this technically belongs to _typing.py, but put here to avoid circular import
AnyParallel: TypeAlias = _Pool | _Loop
#############################################################################


def _Parallel(  # noqa: N802
    n_processes: int,
    initialiser: Callable[[*_InitArgs], None] | None = None,
    initargs: tuple[*_InitArgs] = (),  # type: ignore[assignment]  # python/mypy#17113
) -> AnyParallel:
    """a helper function to distribute parallel computation
    based on the requested number of processes"""

    # allows n_processes <= 0 for now
    if n_processes > 1:
        return _Pool(n_processes, initialiser, initargs)
    else:
        return _Loop()
