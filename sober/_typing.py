from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Concatenate,
    Literal,
    Protocol,
    Required,
    TypeAlias,
    TypedDict,
    TypeGuard,
    TypeVar,
)

import numpy as np
from numpy.typing import NDArray

import sober._pymoo_namespace as pm

# shared - python
AnyStrPath: TypeAlias = str | PathLike[str]
AnyCmdArgs: TypeAlias = tuple[AnyStrPath, ...]

# shared - sober
AnyCoreLevel: TypeAlias = Literal["task", "job"]
AnyLevel: TypeAlias = Literal[AnyCoreLevel, "batch", "epoch"]

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


# input
AnyModifierKey: TypeAlias = float | int
AnyModelModifierVal: TypeAlias = float | str
AnyModifierVal: TypeAlias = AnyStrPath | AnyModelModifierVal
AnyFunc: TypeAlias = Callable[
    Concatenate[tuple[AnyModifierVal, ...], ...], AnyModelModifierVal
]

## Val is omitted in naming below
AnyModelTask: TypeAlias = tuple[AnyModelModifierVal, ...]
AnyTask: TypeAlias = tuple[Path, *AnyModelTask]
AnyTaskItem: TypeAlias = tuple[str, AnyTask]
AnyJob: TypeAlias = tuple[AnyTaskItem, ...]
AnyJobItem: TypeAlias = tuple[str, AnyJob]
AnyBatch: TypeAlias = tuple[AnyJobItem, ...]

## this contains hype ctrl keys only used for populating jobs
AnyCtrlKeyVec: TypeAlias = tuple[int, *tuple[AnyModifierKey, ...]]

### this TypeGuard helps narrow down AnyCtrlKeyVec
_T = TypeVar("_T")


def each_item_is_non_empty(
    args: tuple[tuple[_T, ...], ...],
) -> TypeGuard[tuple[tuple[_T, *tuple[_T, ...]], ...]]:
    return all(len(item) >= 1 for item in args)


# output
AnyUIDs: TypeAlias = tuple[str, ...]
AnyBatchOutputs: TypeAlias = tuple[tuple[float, ...], ...]


# pymoo
AnyPymooCallback: TypeAlias = pm.Callback | Callable[[pm.Algorithm], None] | None
AnyPymooX: TypeAlias = dict[str, np.integer[Any] | np.floating[Any]]


class PymooOut(TypedDict):
    F: NDArray[np.float_] | None
    G: NDArray[np.float_] | None


class PymooOperators(TypedDict):
    sampling: pm.Population
    mating: pm.MixedVariableMating
    eliminate_duplicates: pm.MixedVariableDuplicateElimination


# logger
class SubprocessResult(Protocol):
    returncode: int
    stdout: str
