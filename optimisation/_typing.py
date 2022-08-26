from os import PathLike
from pathlib import Path
from collections.abc import Callable
from typing import Literal, Protocol, TypeAlias

from typing_extensions import Required, TypedDict
from typing_extensions import Unpack  # TODO: remove Unpack after 3.11

from . import _pymoo_namespace as pm

AnyStrPath: TypeAlias = str | PathLike[str]
AnyCmdArgs: TypeAlias = tuple[AnyStrPath, ...]

AnyModelType: TypeAlias = Literal[".idf", ".imf"]
AnyLanguage: TypeAlias = Literal["python"]

AnyIntVURow: TypeAlias = tuple[int, int]
AnyFloatVURow: TypeAlias = tuple[float]
AnyVURow: TypeAlias = AnyIntVURow | AnyFloatVURow  # type: ignore[operator] #python/typeshed#4819
AnyVariationVec: TypeAlias = tuple[int, Unpack[tuple[float, ...]]]  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
AnyUncertaintyVec: TypeAlias = tuple[int, Unpack[tuple[float, ...]]]  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
AnyVUMat: TypeAlias = tuple[
    AnyIntVURow, Unpack[tuple[AnyVURow, ...]]  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
]
AnyTask: TypeAlias = tuple[str, AnyVUMat]
AnyJob: TypeAlias = tuple[str, tuple[AnyTask, ...]]
## TODO: the following may be generalised after python/mypy#12280 ##
AnyIntVariationVec: TypeAlias = tuple[int, ...]
AnyIntUncertaintyVec: TypeAlias = tuple[int, ...]
AnyIntVUMat: TypeAlias = tuple[AnyIntVURow, ...]
AnyIntTask: TypeAlias = tuple[str, AnyIntVUMat]
AnyIntJob: TypeAlias = tuple[str, tuple[AnyIntTask, ...]]
####################################################################

AnyUIDs: TypeAlias = tuple[str, ...]
AnyUIDsPair: TypeAlias = tuple[str, AnyUIDs]
AnyBatchResults: TypeAlias = tuple[tuple[float, ...], ...]

Config = TypedDict(
    "Config",
    {
        "schema.energyplus": Required[Path],
        "exec.energyplus": Required[Path],
        "exec.epmacro": Path,
        "exec.readvars": Path,
        "exec.python": Path,
        "n.processes": int,
    },
    total=False,
)

AnyCallback: TypeAlias = pm.Callback | Callable[[pm.Algorithm], None] | None


class SubprocessRes(Protocol):
    args: AnyCmdArgs
    returncode: int
    stdout: str
