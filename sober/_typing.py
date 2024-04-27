from os import PathLike
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

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


# pymoo
AnyX: TypeAlias = dict[str, np.int_ | np.float_]
AnyReferenceDirections: TypeAlias = NDArray[np.float_]
#############################################################################

if TYPE_CHECKING:
    from typing import NotRequired, TypedDict

    from sober.input import (
        CategoricalModifier,
        ContinuousModifier,
        DiscreteModifier,
        FunctionalModifier,
    )

    # python
    AnyCmdArgs: TypeAlias = tuple[AnyStrPath, ...]

    # config
    AnyModelType: TypeAlias = Literal[".idf", ".imf"]
    AnyLanguage: TypeAlias = Literal["python"]

    class NoiseSampleKwargs(TypedDict):
        size: int
        method: Literal["random", "latin hypercube"]
        seed: NotRequired[int]

    # input
    # TODO: use Intersection after python/typing#213
    AnyIntegralModelModifier: TypeAlias = (
        DiscreteModifier | CategoricalModifier | FunctionalModifier
    )
    AnyRealModelModifier: TypeAlias = ContinuousModifier
    AnyModelModifier: TypeAlias = AnyRealModelModifier | AnyIntegralModelModifier
