from os import PathLike
from typing import TYPE_CHECKING, Literal, TypeAlias

##############################  package typing  ##############################
# need to remove casts etc. to resolve this block

# python
AnyStrPath: TypeAlias = str | PathLike[str]


# input
AnyModifierKey: TypeAlias = float | int
AnyModelModifierVal: TypeAlias = float | str
AnyModifierVal: TypeAlias = AnyStrPath | AnyModelModifierVal

## this contains hype ctrl keys only used for populating jobs
AnyCtrlKeyVec: TypeAlias = tuple[int, *tuple[AnyModifierKey, ...]]


# output
AnyCoreLevel: TypeAlias = Literal["task", "job"]
AnyLevel: TypeAlias = Literal[AnyCoreLevel, "batch", "epoch"]
#############################################################################

if TYPE_CHECKING:
    from collections.abc import Callable

    import sober._pymoo_namespace as pm

    # python
    AnyCmdArgs: TypeAlias = tuple[AnyStrPath, ...]

    # config
    AnyModelType: TypeAlias = Literal[".idf", ".imf"]
    AnyLanguage: TypeAlias = Literal["python"]

    # pymoo
    AnyPymooCallback: TypeAlias = pm.Callback | Callable[[pm.Algorithm], None] | None
