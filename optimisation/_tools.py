from os import PathLike
from typing import TypeAlias

AnyStrPath: TypeAlias = str | PathLike[str]
AnyCli: TypeAlias = tuple[AnyStrPath, ...]
