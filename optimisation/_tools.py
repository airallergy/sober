from os import PathLike
from pathlib import Path
from os.path import relpath
from typing import TypeAlias
from subprocess import PIPE, STDOUT, run

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
