from abc import ABC, abstractmethod
from uuid import NAMESPACE_X500, uuid5
from typing import Any, TypeVar, ClassVar, Iterable

from eppy.modeleditor import IDF
from eppy.bunchhelpers import makefieldname

AnyModel = TypeVar("AnyModel", IDF, str)


#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Tagger(ABC):
    _loc: ClassVar[str]
    _tag: str

    def __init__(self, uuid_descriptions: tuple[str, ...]) -> None:
        self._tag = self._uuid(*uuid_descriptions)

    @classmethod
    def _uuid(cls, *descriptions: str) -> str:
        return str(uuid5(NAMESPACE_X500, cls.__name__ + "-".join(descriptions)))

    @abstractmethod
    def _tagged(self, model: Any) -> Any:
        ...


class _Parameter(ABC):
    tagger: _Tagger  # TODO: may remove out of here, as weather does not need one
    low: float
    high: float

    def __init__(self, tagger: _Tagger) -> None:
        self.tagger = tagger


class _FloatParameter(_Parameter):
    def __init__(self, tagger: _Tagger, low: float, high: float) -> None:
        super().__init__(tagger)

        self.low = low
        self.high = high


class _IntParameter(_Parameter):
    low: int
    high: int
    variations: tuple[Any, ...]
    uncertainties: tuple[tuple[Any, ...], ...]

    def __init__(
        self,
        tagger: _Tagger,
        variations: Iterable[Any],
        uncertainties: Iterable[Iterable[Any]] = ((),),
    ) -> None:
        super().__init__(tagger)

        self.variations = tuple(variations)
        self.uncertainties = tuple(tuple(uncertainty) for uncertainty in uncertainties)


#############################################################################
#######                        TAGGER CLASSES                         #######
#############################################################################
class IndexTagger(_Tagger):
    """Tagger for regular commands by indexing.

    No support for nested regular commands.
    """

    _loc: ClassVar[str] = "regular"
    class_name: str
    object_name: str
    field_name: str

    def __init__(self, class_name: str, object_name: str, field_name: str) -> None:
        self.class_name = class_name
        self.object_name = object_name
        self.field_name = field_name

        super().__init__((self.class_name, self.object_name, self.field_name))

    def _tagged(self, model: IDF) -> IDF:
        # NOTE: maybe applicable to multiple fields
        model.getobject(self.class_name, self.object_name)[
            makefieldname(self.field_name)
        ] = self._tag
        return model


class StringTagger(_Tagger):
    """Tagger for macro commands by string replacement.

    No support for nested macro commands.
    """

    _loc: ClassVar[str] = "macro"
    string: str
    prefix: str
    suffix: str

    def __init__(self, string: str, prefix: str = "", suffix: str = "") -> None:
        if not (string.startswith(prefix) and string.endswith(suffix)):
            raise ValueError("string needs to share the prefix and the suffix.")

        self.string = string
        self.prefix = prefix
        self.suffix = suffix

        super().__init__((self.string,))

    def _tagged(self, model: str) -> str:
        return model.replace(self.string, self.prefix + self._tag + self.suffix)


#############################################################################
#######                       PARAMETER CLASSES                       #######
#############################################################################
class ContinuousParameter(_FloatParameter):
    ...


class DiscreteParameter(_IntParameter):
    variations: tuple[float, ...]
    uncertainties: tuple[tuple[float, ...], ...]


class CategoricalParameter(_IntParameter):
    variations: tuple[str, ...]
    uncertainties: tuple[tuple[str, ...], ...]
