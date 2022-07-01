from os import PathLike
from pathlib import Path
from platform import system
from typing import TypeAlias
from multiprocessing import get_context
from subprocess import PIPE, STDOUT, run
from multiprocessing.context import BaseContext

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


def _multiprocessing_context() -> BaseContext:
    match system():
        case "Linux" | "Darwin":
            return get_context("forkserver")
        case "Windows":
            return get_context("spawn")
        case _ as system_name:
            raise NotImplementedError(f"unsupported system: '{system_name}'.")
