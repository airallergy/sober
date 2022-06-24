from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import Iterable
from uuid import NAMESPACE_X500, uuid5
from typing import Any, ClassVar, TypeAlias, overload

from eppy.modeleditor import IDF
from eppy.bunchhelpers import makefieldname

from ._tools import AnyStrPath


#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Tagger(ABC):
    _LOCATION: ClassVar[str]
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
    low: float
    high: float


class _ModelParameterMixin(ABC):
    tagger: _Tagger

    def __init__(self, tagger: _Tagger, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # NOTE: to _FloatParameter/_IntParameter
        self.tagger = tagger


class _FloatParameter(_Parameter):
    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high


class _IntParameter(_Parameter):
    low: int
    high: int
    variations: tuple[Any, ...]
    uncertainties: tuple[tuple[Any, ...], ...] | tuple[Any, ...]
    _is_uncertain: bool

    def __init__(
        self,
        variations: Iterable[Any],
        uncertainties: Iterable[Iterable[Any]] | Iterable[Any],
    ) -> None:
        self.variations = tuple(variations)
        if uncertainties == ():
            self._is_uncertain = False
        else:
            self._is_uncertain = True
            self.uncertainties = (
                tuple(tuple(uncertainty) for uncertainty in uncertainties)
                if any(isinstance(item, Iterable) for item in uncertainties)
                else (tuple(uncertainties),) * len(self.variations)
            )

        self.low = 0
        self.high = len(self.variations)


#############################################################################
#######                        TAGGER CLASSES                         #######
#############################################################################
class IndexTagger(_Tagger):
    """Tagger for regular commands by indexing.

    No support for nested regular commands.
    """

    _LOCATION: ClassVar[str] = "regular"
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

    _LOCATION: ClassVar[str] = "macro"
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
class ContinuousParameter(_ModelParameterMixin, _FloatParameter):
    ...


class DiscreteParameter(_ModelParameterMixin, _IntParameter):
    variations: tuple[float, ...]
    uncertainties: tuple[tuple[float, ...], ...] | tuple[float, ...]

    def __init__(
        self,
        tagger: _Tagger,
        variations: Iterable[float],
        uncertainties: Iterable[Iterable[float]] | Iterable[float] = (),
    ) -> None:
        super().__init__(tagger, variations, uncertainties)


class CategoricalParameter(_ModelParameterMixin, _IntParameter):
    variations: tuple[str, ...]
    uncertainties: tuple[tuple[str, ...], ...] | tuple[str, ...]

    def __init__(
        self,
        tagger: _Tagger,
        variations: Iterable[str],
        uncertainties: Iterable[Iterable[str]] | Iterable[str] = (),
    ) -> None:
        super().__init__(tagger, variations, uncertainties)


class WeatherParameter(_IntParameter):
    variations: tuple[Path | str, ...]
    uncertainties: tuple[tuple[Path, ...], ...] | tuple[Path, ...]

    @overload
    def __init__(
        self,
        variations: Iterable[AnyStrPath],
        uncertainties: tuple[()],
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        variations: Iterable[str],
        uncertainties: Iterable[Iterable[AnyStrPath]] | Iterable[AnyStrPath],
    ) -> None:
        ...

    def __init__(self, variations, uncertainties=()):
        super().__init__(
            (Path(variation) for variation in variations)
            if uncertainties == ()
            else (variation for variation in variations),
            uncertainties,
        )


AnyIntModelParameter: TypeAlias = DiscreteParameter | CategoricalParameter
AnyModelParameter: TypeAlias = ContinuousParameter | AnyIntModelParameter
