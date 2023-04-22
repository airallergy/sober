from os import PathLike
from pathlib import Path
from collections.abc import Callable
from typing import Literal, TypeVar, Protocol, TypeAlias

from typing_extensions import Required, TypedDict
from typing_extensions import Unpack  # TODO: remove Unpack after 3.11

from . import _pymoo_namespace as pm

AnyStrPath: TypeAlias = str | PathLike[str]
AnyCmdArgs: TypeAlias = tuple[AnyStrPath, ...]

AnyModelType: TypeAlias = Literal[".idf", ".imf"]
AnyLanguage: TypeAlias = Literal["python"]

_S = TypeVar("_S", int, float)
AnyVURow: TypeAlias = tuple[_S, _S]  # TODO: double check this for float params
AnyIntVURow: TypeAlias = AnyVURow[int]
AnyRealVURow: TypeAlias = AnyVURow[float]
AnyVariationMap: TypeAlias = dict[str, _S]
AnyVariationVec: TypeAlias = tuple[int, Unpack[tuple[_S, ...]]]  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
AnyUncertaintyVec: TypeAlias = tuple[int, Unpack[tuple[_S, ...]]]  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
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
        "exec.expandobjects": Path,
        "exec.readvars": Path,
        "exec.python": Path,
        "n.processes": int,
    },
    total=False,
)

AnyCallback: TypeAlias = pm.Callback | Callable[[pm.Algorithm], None] | None


class SubprocessRes(Protocol):
    returncode: int
    stdout: str
