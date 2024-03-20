from collections.abc import Callable
from os import PathLike
from typing import Literal, Protocol, Required, TypeAlias, TypedDict, TypeVar

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
AnyDuoVec: TypeAlias = tuple[AnyIntegralDuo, *tuple[AnyDuo, ...]]

AnyCandidateVec: TypeAlias = tuple[int, *tuple[_S, ...]]
AnyScenarioVec: TypeAlias = tuple[int, *tuple[_S, ...]]

AnyTask: TypeAlias = tuple[str, AnyDuoVec]
AnyJob: TypeAlias = tuple[str, tuple[AnyTask, ...]]


# results
AnyUIDs: TypeAlias = tuple[str, ...]
AnyBatchResults: TypeAlias = tuple[tuple[float, ...], ...]


# pymoo
AnyPymooCallback: TypeAlias = pm.Callback | Callable[[pm.Algorithm], None] | None
AnyCandidateMap: TypeAlias = dict[
    str, np.integer | np.floating
]  # TODO: find a way to type the first element as int


class PymooOut(TypedDict):
    F: np.ndarray | None
    G: np.ndarray | None


class PymooOperators(TypedDict):
    sampling: pm.Population
    mating: pm.MixedVariableMating
    eliminate_duplicates: pm.MixedVariableDuplicateElimination


# logger
class SubprocessResult(Protocol):
    returncode: int
    stdout: str
