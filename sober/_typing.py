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
    from typing import NotRequired, TypedDict

    import numpy as np
    from numpy.typing import NDArray

    from sober.input import (
        CategoricalModifier,
        ContinuousModifier,
        DiscreteModifier,
        FunctionalModifier,
    )

    # python
    AnyCmdArgs: TypeAlias = tuple[AnyStrPath, ...]

    # input
    # TODO: use Intersection after python/typing#213
    AnyIntegralModelModifier: TypeAlias = (
        DiscreteModifier | CategoricalModifier | FunctionalModifier
    )
    AnyRealModelModifier: TypeAlias = ContinuousModifier
    AnyModelModifier: TypeAlias = AnyRealModelModifier | AnyIntegralModelModifier

    # problem
    AnySampleMode: TypeAlias = Literal["elementwise", "cartesian", "auto"]

    # config
    AnyModelType: TypeAlias = Literal[".idf", ".imf"]
    AnyLanguage: TypeAlias = Literal["python"]

    class ElementwiseNoiseSampleKwargs(TypedDict):
        mode: Literal["elementwise"]
        size: int
        method: Literal["random", "latin hypercube"]
        seed: NotRequired[int]

    class CartesianNoiseSampleKwargs(TypedDict):
        mode: Literal["cartesian"]

    class AutoNoiseSampleKwargs(TypedDict):
        mode: Literal["auto"]
        size: NotRequired[int]
        method: NotRequired[Literal["random", "latin hypercube"]]
        seed: NotRequired[int]

    NoiseSampleKwargs: TypeAlias = (
        ElementwiseNoiseSampleKwargs
        | CartesianNoiseSampleKwargs
        | AutoNoiseSampleKwargs
    )

    # pymoo
    AnyX: TypeAlias = dict[str, np.int_ | np.float_]
    AnyF: TypeAlias = NDArray[np.float_]
    AnyG: TypeAlias = NDArray[np.float_]
    AnyReferenceDirections: TypeAlias = NDArray[np.float_]
