from collections.abc import Callable
from os import PathLike
from typing import Literal, TypeAlias

import sober._pymoo_namespace as pm

# python
AnyStrPath: TypeAlias = str | PathLike[str]
AnyCmdArgs: TypeAlias = tuple[AnyStrPath, ...]


# config
AnyModelType: TypeAlias = Literal[".idf", ".imf"]
AnyLanguage: TypeAlias = Literal["python"]


# input
AnyModifierKey: TypeAlias = float | int
AnyModelModifierVal: TypeAlias = float | str
AnyModifierVal: TypeAlias = AnyStrPath | AnyModelModifierVal

## this contains hype ctrl keys only used for populating jobs
AnyCtrlKeyVec: TypeAlias = tuple[int, *tuple[AnyModifierKey, ...]]


# output
AnyCoreLevel: TypeAlias = Literal["task", "job"]
AnyLevel: TypeAlias = Literal[AnyCoreLevel, "batch", "epoch"]


# pymoo
AnyPymooCallback: TypeAlias = pm.Callback | Callable[[pm.Algorithm], None] | None
