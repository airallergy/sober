from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import (
    Concatenate,
    Literal,
    Protocol,
    Required,
    TypeAlias,
    TypedDict,
    TypeVar,
)

import numpy as np

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
AnyModelModifierVal: TypeAlias = float | str
AnyModifierVal: TypeAlias = AnyStrPath | AnyModelModifierVal
AnyFunc: TypeAlias = Callable[
    Concatenate[tuple[AnyModifierVal, ...], ...], AnyModelModifierVal
]

MK = TypeVar("MK", float, int)  # AnyModifierKey
MV = TypeVar("MV", bound=AnyModifierVal)  # AnyModifierValue

## this contains hype ctrl keys only used for populating jobs
AnyCtrlKeyVec: TypeAlias = tuple[int, *tuple[MK, ...]]

## Val is omitted in naming below
AnyModelTask: TypeAlias = tuple[AnyModelModifierVal, ...]
AnyTask: TypeAlias = tuple[Path, *AnyModelTask]
AnyTaskItem: TypeAlias = tuple[str, AnyTask]
AnyJob: TypeAlias = tuple[AnyTaskItem, ...]
AnyJobItem: TypeAlias = tuple[str, AnyJob]
AnyBatch: TypeAlias = tuple[AnyJobItem, ...]


# output
AnyUIDs: TypeAlias = tuple[str, ...]
AnyBatchOutputs: TypeAlias = tuple[tuple[float, ...], ...]


# pymoo
AnyPymooCallback: TypeAlias = pm.Callback | Callable[[pm.Algorithm], None] | None
AnyPymooX: TypeAlias = dict[str, np.integer | np.floating]


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
