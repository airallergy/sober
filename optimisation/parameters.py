from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import Iterable
from uuid import NAMESPACE_X500, uuid5
from typing import Any, ClassVar, TypeAlias, SupportsIndex, overload

from eppy.modeleditor import IDF
from eppy.bunchhelpers import makefieldname

from ._tools import AnyStrPath


#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Tagger(ABC):
    _LOCATION: ClassVar[str]
    _tag: str

    @abstractmethod
    def __init__(self, uuid_descriptions: tuple[str, ...]) -> None:
        self._tag = self._uuid(*uuid_descriptions)

    @classmethod
    def _uuid(cls, *descriptions: str) -> str:
        return str(uuid5(NAMESPACE_X500, cls.__name__ + "-".join(descriptions)))

    @abstractmethod
    def _tagged(self, model: Any) -> Any:
        ...


class _Parameter(ABC):
    _low: float
    _high: float


class _ModelParameterMixin(ABC):
    _tagger: _Tagger

    @abstractmethod
    def __init__(self, tagger: _Tagger, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # NOTE: to _FloatParameter/_IntParameter
        self._tagger = tagger


class _FloatParameter(_Parameter):
    @abstractmethod
    def __init__(self, low: float, high: float) -> None:
        self._low = low
        self._high = high


class _IntParameter(_Parameter):
    _low: int
    _high: int
    _variations: tuple[Any, ...]
    _uncertainties: tuple[tuple[Any, ...], ...] | tuple[Any, ...]
    _is_uncertain: bool

    @abstractmethod
    def __init__(
        self,
        variations: Iterable[Any],
        uncertainties: Iterable[Iterable[Any]] | Iterable[Any],
    ) -> None:
        self._variations = tuple(variations)
        if uncertainties == ():
            self._is_uncertain = False
        else:
            self._is_uncertain = True
            self._uncertainties = (
                tuple(tuple(uncertainty) for uncertainty in uncertainties)
                if any(isinstance(item, Iterable) for item in uncertainties)
                else (tuple(uncertainties),) * len(self._variations)
            )

        self._low = 0
        self._high = len(self._variations)

    def __getitem__(self, index: SupportsIndex | tuple[SupportsIndex, SupportsIndex]):
        match self._is_uncertain, index:
            case True, (int() as x, int() as y):
                return self._uncertainties[x][y]
            case False, int() as x:
                return self._variations[x]
            case _:
                raise


#############################################################################
#######                        TAGGER CLASSES                         #######
#############################################################################
class IndexTagger(_Tagger):
    """Tagger for regular commands by indexing.

    No support for nested regular commands.
    """

    _LOCATION: ClassVar[str] = "regular"
    _class_name: str
    _object_name: str
    _field_name: str

    def __init__(self, class_name: str, object_name: str, field_name: str) -> None:
        self._class_name = class_name
        self._object_name = object_name
        self._field_name = field_name

        super().__init__((self._class_name, self._object_name, self._field_name))

    def _tagged(self, model: IDF) -> IDF:
        # NOTE: maybe applicable to multiple fields
        model.getobject(self._class_name, self._object_name)[
            makefieldname(self._field_name)
        ] = self._tag
        return model


class StringTagger(_Tagger):
    """Tagger for macro commands by string replacement.

    No support for nested macro commands.
    """

    _LOCATION: ClassVar[str] = "macro"
    _string: str
    _prefix: str
    _suffix: str

    def __init__(self, string: str, prefix: str = "", suffix: str = "") -> None:
        if not (string.startswith(prefix) and string.endswith(suffix)):
            raise ValueError("string needs to share the prefix and the suffix.")

        self._string = string
        self._prefix = prefix
        self._suffix = suffix

        super().__init__((self._string,))

    def _tagged(self, model: str) -> str:
        return model.replace(self._string, self._prefix + self._tag + self._suffix)


#############################################################################
#######                       PARAMETER CLASSES                       #######
#############################################################################
class ContinuousParameter(_ModelParameterMixin, _FloatParameter):
    ...


class DiscreteParameter(_ModelParameterMixin, _IntParameter):
    _variations: tuple[float, ...]
    _uncertainties: tuple[tuple[float, ...], ...] | tuple[float, ...]

    def __init__(
        self,
        tagger: _Tagger,
        variations: Iterable[float],
        uncertainties: Iterable[Iterable[float]] | Iterable[float] = (),
    ) -> None:
        super().__init__(tagger, variations, uncertainties)


class CategoricalParameter(_ModelParameterMixin, _IntParameter):
    _variations: tuple[str, ...]
    _uncertainties: tuple[tuple[str, ...], ...] | tuple[str, ...]

    def __init__(
        self,
        tagger: _Tagger,
        variations: Iterable[str],
        uncertainties: Iterable[Iterable[str]] | Iterable[str] = (),
    ) -> None:
        super().__init__(tagger, variations, uncertainties)


class WeatherParameter(_IntParameter):
    _variations: tuple[Path | str, ...]
    _uncertainties: tuple[tuple[Path, ...], ...] | tuple[Path, ...]

    @overload
    def __init__(self, variations: Iterable[AnyStrPath]) -> None:
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
