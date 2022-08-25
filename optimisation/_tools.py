import sys
from pathlib import Path
from itertools import starmap
from types import TracebackType
from multiprocessing import get_context
from subprocess import PIPE, STDOUT, run
from contextlib import AbstractContextManager
from collections.abc import Callable, Iterable
from typing import Any, Type, Generic, TypeVar

from typing_extensions import Unpack  # TODO: remove Unpack after 3.11
from typing_extensions import TypeVarTuple  # NOTE: from typing after 3.11

from ._logger import _log
from ._typing import AnyCli


def _run(commands: AnyCli, cwd: Path) -> None:
    commands = (
        commands[0],
        *(
            command.resolve(strict=True) if isinstance(command, Path) else command
            for command in commands[1:]
        ),
    )
    res = run(commands, stdout=PIPE, stderr=STDOUT, cwd=cwd, text=True)

    _log(cwd, res.args, res.returncode, res.stdout)


# this bit is purely to make mypy happy :(
if sys.platform != "win32":
    _MULTIPROCESSING_CONTEXT = get_context("forkserver")
else:
    _MULTIPROCESSING_CONTEXT = get_context("spawn")

InitArgs = TypeVarTuple("InitArgs")  # type: ignore[misc] # TODO: after 3.11
_P = TypeVar("_P")
_R = TypeVar("_R", covariant=True)


class _Parallel(AbstractContextManager, Generic[Unpack[InitArgs]]):  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
    _n_processes: int
    _initializer: Callable[[Unpack[InitArgs]], None] | None  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
    _initargs: tuple[Unpack[InitArgs]]  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11

    def __init__(
        self,
        n_processes: int,
        initializer: Callable[[Unpack[InitArgs]], None] | None = None,  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
        initargs: tuple[Unpack[InitArgs]] = (),  # type: ignore[misc, assignment] # python/mypy#12280 # TODO: Unpack -> * after 3.11
    ) -> None:
        self._n_processes = n_processes
        self._initializer = initializer
        self._initargs = initargs  # type: ignore[assignment] # python/mypy#12280

    def _chunk_size(self, n_tasks: int) -> int:
        return max(n_tasks // self._n_processes, 1)

    def map(self, func: Callable[[_P], _R], iterable: Iterable[_P]) -> tuple[_R, ...]:
        if self._n_processes > 1:
            with _MULTIPROCESSING_CONTEXT.Pool(
                self._n_processes,
                initializer=self._initializer,
                initargs=self._initargs,
            ) as pool:
                return tuple(
                    pool.map(
                        func, x := tuple(iterable), chunksize=self._chunk_size(len(x))
                    )
                )
        else:
            return tuple(map(func, iterable))

    def starmap(
        self, func: Callable[..., _R], iterable: Iterable[Iterable[Any]]
    ) -> tuple[_R, ...]:
        if self._n_processes > 1:
            with _MULTIPROCESSING_CONTEXT.Pool(
                self._n_processes,
                initializer=self._initializer,
                initargs=self._initargs,
            ) as pool:
                return tuple(
                    pool.starmap(
                        func, x := tuple(iterable), chunksize=self._chunk_size(len(x))
                    )
                )
        else:
            return tuple(starmap(func, iterable))

    def __enter__(self) -> "_Parallel":  # TODO: use typing.Self after 3.11
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return None
