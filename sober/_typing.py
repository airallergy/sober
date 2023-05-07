from os import PathLike
from typing import Literal, TypeVar, Protocol, TypeAlias

import numpy as np
from typing_extensions import Required, TypedDict
from typing_extensions import Unpack  # TODO: remove Unpack after 3.11

# TODO: Refactor this module, along with typing in other modules after 3.11

AnyStrPath: TypeAlias = str | PathLike[str]
AnyCmdArgs: TypeAlias = tuple[AnyStrPath, ...]

AnyModelType: TypeAlias = Literal[".idf", ".imf"]
AnyLanguage: TypeAlias = Literal["python"]

_S = TypeVar("_S", int, float)
AnyVURow: TypeAlias = tuple[_S, _S]  # TODO: double check this for float params
AnyIntegralVURow: TypeAlias = AnyVURow[int]
AnyRealVURow: TypeAlias = AnyVURow[float]
AnyVariationMap: TypeAlias = dict[str, np.integer | np.floating]
AnyVariationVec: TypeAlias = tuple[int, Unpack[tuple[_S, ...]]]  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
AnyUncertaintyVec: TypeAlias = tuple[int, Unpack[tuple[_S, ...]]]  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
AnyVUMat: TypeAlias = tuple[
    AnyIntegralVURow, Unpack[tuple[AnyVURow, ...]]  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
]
AnyTask: TypeAlias = tuple[str, AnyVUMat]
AnyJob: TypeAlias = tuple[str, tuple[AnyTask, ...]]
## TODO: the following may be generalised after python/mypy#12280 ##
AnyIntegralVariationVec: TypeAlias = tuple[int, ...]
AnyIntegralUncertaintyVec: TypeAlias = tuple[int, ...]
AnyIntegralVUMat: TypeAlias = tuple[AnyIntegralVURow, ...]
AnyIntegralTask: TypeAlias = tuple[str, AnyIntegralVUMat]
AnyIntegralJob: TypeAlias = tuple[str, tuple[AnyIntegralTask, ...]]
####################################################################

AnyUIDs: TypeAlias = tuple[str, ...]
AnyUIDsPair: TypeAlias = tuple[str, AnyUIDs]
AnyBatchResults: TypeAlias = tuple[tuple[float, ...], ...]

Config = TypedDict(
    "Config",
    {
        "schema.energyplus": Required[str],
        "exec.energyplus": Required[str],
        "exec.epmacro": str,
        "exec.expandobjects": str,
        "exec.readvars": str,
        "exec.python": str,
        "n.processes": int,
    },
    total=False,
)


class SubprocessResult(Protocol):
    returncode: int
    stdout: str
