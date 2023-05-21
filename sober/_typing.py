from os import PathLike
from collections.abc import Callable
from typing import Literal, TypeVar, Protocol, Required, TypeAlias, TypedDict

import numpy as np

from . import _pymoo_namespace as pm

# generic
AnyStrPath: TypeAlias = str | PathLike[str]
AnyCmdArgs: TypeAlias = tuple[AnyStrPath, ...]


# config
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
AnyModelType: TypeAlias = Literal[".idf", ".imf"]
AnyLanguage: TypeAlias = Literal["python"]


# parameters
_S = TypeVar("_S", int, float)

AnyDuo: TypeAlias = tuple[_S, _S]  # TODO: double check this for float params
AnyIntegralDuo: TypeAlias = AnyDuo[int]
AnyRealDuo: TypeAlias = AnyDuo[float]
AnyDuoVec: TypeAlias = tuple[  # type: ignore[valid-type,misc] # python/mypy#12280
    AnyIntegralDuo, *tuple[AnyDuo, ...]  # type: ignore[misc] # python/mypy#12280
]

AnyCandidateVec: TypeAlias = tuple[int, *tuple[_S, ...]]  # type: ignore[valid-type,misc] # python/mypy#12280
AnyScenarioVec: TypeAlias = tuple[int, *tuple[_S, ...]]  # type: ignore[valid-type,misc] # python/mypy#12280

AnyTask: TypeAlias = tuple[str, AnyDuoVec]
AnyJob: TypeAlias = tuple[str, tuple[AnyTask, ...]]


# results
AnyUIDs: TypeAlias = tuple[str, ...]
AnyBatchResults: TypeAlias = tuple[tuple[float, ...], ...]


# pymoo
AnyPymooCallback: TypeAlias = pm.Callback | Callable[[pm.Algorithm], None] | None
AnyCandidateMap: TypeAlias = dict[str, np.integer | np.floating]
PymooOut = TypedDict("PymooOut", {"F": None | np.ndarray, "G": None | np.ndarray})
PymooOperators = TypedDict(
    "PymooOperators",
    {
        "sampling": pm.Population,
        "mating": pm.MixedVariableMating,
        "eliminate_duplicates": pm.MixedVariableDuplicateElimination,
    },
)


# logger
class SubprocessResult(Protocol):
    returncode: int
    stdout: str
