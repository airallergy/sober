from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import Iterable
from uuid import NAMESPACE_X500, uuid5
from typing import Generic, TypeVar, ClassVar, TypeAlias, SupportsIndex, overload

from eppy.modeleditor import IDF
from eppy.bunchhelpers import makefieldname

from ._tools import AnyStrPath

_M = TypeVar("_M")  # AnyModel
_V = TypeVar("_V")  # AnyVariation
_U = TypeVar("_U")  # AnyUncertaintyVar
#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Tagger(ABC, Generic[_M]):
    _LOCATION: ClassVar[str]
    _tag: str

    @abstractmethod
    def __init__(self, uuid_descriptions: tuple[str, ...]) -> None:
        self._tag = self._uuid(*uuid_descriptions)

    @classmethod
    def _uuid(cls, *descriptions: str) -> str:
        return str(uuid5(NAMESPACE_X500, cls.__name__ + "-".join(descriptions)))

    @abstractmethod
    def _tagged(self, model: _M) -> _M:
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


class _IntParameter(_Parameter, Generic[_V, _U]):
    _low: int
    _high: int
    _variations: tuple[_V, ...]
    _uncertainties: tuple[tuple[_U, ...], ...]
    _n_variations: int
    _ns_uncertainty: tuple[int, ...]
    _is_uncertain: bool

    @abstractmethod
    def __init__(
        self,
        variations: Iterable[_V],
        *uncertainties: Iterable[_U],
    ) -> None:
        self._variations = tuple(variations)
        self._n_variations = len(self._variations)
        match len(uncertainties):
            case 0:
                self._ns_uncertainty = ()
            case 1:
                self._uncertainties = (tuple(uncertainties[0]),) * self._n_variations
                self._ns_uncertainty = (
                    len(self._uncertainties[0]),
                ) * self._n_variations
            case self._n_variations:
                # allow different uncertainty counts for each variation
                self._uncertainties = tuple(
                    tuple(uncertainty) for uncertainty in uncertainties
                )
                self._ns_uncertainty = tuple(map(len, self._uncertainties))
            case _:
                raise ValueError(
                    f"The number of uncertainties is different from that of variations: '{len(uncertainties)}', '{self._n_variations}'."
                )

        self._is_uncertain = len(self._ns_uncertainty) != 0
        self._low = 0
        self._high = self._n_variations

    @overload
    def __getitem__(self, index: SupportsIndex) -> _V:
        ...

    @overload
    def __getitem__(self, index: tuple[SupportsIndex, SupportsIndex]) -> _U:
        ...

    def __getitem__(self, index):
        match self._is_uncertain, index:
            case False, int() as x:
                return self._variations[x]
            case True, (int() as x, int() as y):
                return self._uncertainties[x][y]
            case _, int() | (int(), int()):
                raise IndexError(
                    "no " * (not self._is_uncertain) + "uncertainties defined."
                )
            case _:
                raise IndexError(f"invalid index: {index}.")


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
    _uncertainties: tuple[tuple[float, ...], ...]

    def __init__(
        self,
        tagger: _Tagger,
        variations: Iterable[float],
        *uncertainties: Iterable[float],
    ) -> None:
        super().__init__(tagger, variations, *uncertainties)


class CategoricalParameter(_ModelParameterMixin, _IntParameter):
    _variations: tuple[str, ...]
    _uncertainties: tuple[tuple[str, ...], ...]

    def __init__(
        self,
        tagger: _Tagger,
        variations: Iterable[str],
        *uncertainties: Iterable[str],
    ) -> None:
        super().__init__(tagger, variations, *uncertainties)


class WeatherParameter(_IntParameter):
    _variations: tuple[Path | str, ...]
    _uncertainties: tuple[tuple[Path, ...], ...]

    @overload
    def __init__(self, variations: Iterable[AnyStrPath]) -> None:
        ...

    @overload
    def __init__(
        self, variations: Iterable[str], *uncertainties: Iterable[AnyStrPath]
    ) -> None:
        ...

    def __init__(self, variations, *uncertainties):
        super().__init__(
            map(Path, variations) if len(uncertainties) else variations, *uncertainties
        )
        if self._is_uncertain:
            self._uncertainties = tuple(
                tuple(map(Path, item)) for item in self._uncertainties
            )


AnyIntModelParameter: TypeAlias = DiscreteParameter | CategoricalParameter
AnyModelParameter: TypeAlias = ContinuousParameter | AnyIntModelParameter
