from abc import ABC, abstractmethod
from collections.abc import Sequence
from uuid import NAMESPACE_X500, uuid5
from dataclasses import field, dataclass
from typing import Any, TypeVar, ClassVar, Optional

from eppy.modeleditor import IDF
from eppy.bunchhelpers import makefieldname

from ._tools import DATACLASS_PARAMS

AnyModel = TypeVar("AnyModel", IDF, str)


#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
@dataclass(**DATACLASS_PARAMS)  # type: ignore[misc] # python/mypy#5374
class _Tagger(ABC):
    tag: str = field(init=False)
    _loc: ClassVar[str]

    @classmethod
    def _uuid(cls, *descriptions: str) -> str:
        return str(uuid5(NAMESPACE_X500, cls.__name__ + "-".join(descriptions)))

    @abstractmethod
    def _tagged(self, model: Any) -> Any:
        ...


@dataclass(**DATACLASS_PARAMS)
class _Parameter(ABC):
    tagger: _Tagger  # TODO: may remove out of abc, as weather does not need one
    low: float
    high: float


@dataclass(**DATACLASS_PARAMS)
class _IntegerParameter(_Parameter):
    low: int = field(init=False)
    high: int = field(init=False)
    variations: Sequence[Any]
    uncertainties: Optional[Sequence[Sequence[Any]]] = None


#############################################################################
#######                        TAGGER CLASSES                         #######
#############################################################################
@dataclass(**DATACLASS_PARAMS)
class IndexTagger(_Tagger):
    """Tagger for regular commands by indexing.

    No support for nested regular commands.
    """

    _loc: ClassVar[str] = "regular"
    class_name: str
    object_name: str
    field_name: str

    def __post_init__(self) -> None:
        self.tag = self._uuid(self.class_name, self.object_name, self.field_name)

    def _tagged(self, model: IDF) -> IDF:
        # NOTE: maybe applicable to multiple fields
        model.getobject(self.class_name, self.object_name)[
            makefieldname(self.field_name)
        ] = self.tag
        return model


@dataclass(**DATACLASS_PARAMS)
class StringTagger(_Tagger):
    """Tagger for macro commands by string replacement.

    No support for nested macro commands.
    """

    _loc: ClassVar[str] = "macro"
    string: str
    prefix: str = ""
    suffix: str = ""

    def __post_init__(self) -> None:
        if (not self.string.startswith(self.prefix)) or (
            not self.string.endswith(self.suffix)
        ):
            raise ValueError("string needs to share the prefix and the suffix.")

        self.tag = self._uuid(self.string)

    def _tagged(self, model: str) -> str:
        return model.replace(self.string, self.prefix + self.tag + self.suffix)


#############################################################################
#######                       PARAMETER CLASSES                       #######
#############################################################################
@dataclass(**DATACLASS_PARAMS)
class ContinuousParameter(_Parameter):
    ...


@dataclass(**DATACLASS_PARAMS)
class DiscreteParameter(_IntegerParameter):
    variations: Sequence[float]
    uncertainties: Optional[Sequence[Sequence[float]]] = None


@dataclass(**DATACLASS_PARAMS)
class CategoricalParameter(_IntegerParameter):
    variations: Sequence[str]
    uncertainties: Optional[Sequence[Sequence[str]]] = None
