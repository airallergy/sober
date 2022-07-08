import sys
from os import PathLike
from pathlib import Path
from typing import TypeAlias
from multiprocessing import get_context
from subprocess import PIPE, STDOUT, run
from multiprocessing.context import SpawnContext, ForkServerContext

AnyStrPath: TypeAlias = str | PathLike[str]
AnyCli: TypeAlias = tuple[AnyStrPath, ...]


def _run(commands: AnyCli, cwd: Path) -> None:
    commands = (
        commands[0],
        *(
            command.resolve(strict=True) if isinstance(command, Path) else command
            for command in commands[1:]
        ),
    )
    run(commands, stdout=PIPE, stderr=STDOUT, cwd=cwd, text=True)


# this bit is purely to make mypy happy :(
if sys.platform == "win32":

    def _multiprocessing_context() -> SpawnContext:
        return get_context("spawn")

else:

    def _multiprocessing_context() -> ForkServerContext:
        return get_context("forkserver")


def _chunk_size(n_tasks: int, n_processes: int) -> int:
    return max(n_tasks // n_processes, 1)
