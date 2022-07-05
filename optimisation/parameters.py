from math import log10
from io import StringIO
from pathlib import Path
from itertools import product
from abc import ABC, abstractmethod
from uuid import NAMESPACE_X500, uuid5
from collections.abc import Iterable, Iterator
from typing import (
    Any,
    Generic,
    TypeVar,
    TypeAlias,
    TypeGuard,
    SupportsIndex,
    cast,
    overload,
)

from eppy import openidf
from eppy.modeleditor import IDF
from eppy.bunchhelpers import makefieldname

from . import config as cf
from ._tools import AnyStrPath
from ._simulator import _split_model

_V = TypeVar("_V")  # AnyVariation
_U = TypeVar("_U")  # AnyUncertaintyVar


#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Tagger(ABC):
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


class _RegularTagger(_Tagger):
    @abstractmethod
    def _tagged(self, model: IDF) -> IDF:
        ...


class _MacroTagger(_Tagger):
    @abstractmethod
    def _tagged(self, model: str) -> str:
        ...


class _Parameter(ABC):
    _low: float
    _high: float


class _ModelParameterMixin(ABC):
    _tagger: _Tagger

    @abstractmethod
    def __init__(self, tagger: _Tagger, *args, **kwargs) -> None:
        self._tagger = tagger

        super().__init__(*args, **kwargs)  # NOTE: to _FloatParameter/_IntParameter


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
                self._ns_uncertainty = (1,) * self._n_variations
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

        self._is_uncertain = set(self._ns_uncertainty) != {1}
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
            case False, (int() as x, 0):
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
class IndexTagger(_RegularTagger):
    """Tagger for regular commands by indexing.

    No support for nested regular commands.
    """

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


class StringTagger(_MacroTagger):
    """Tagger for macro commands by string replacement.

    No support for nested macro commands.
    """

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
    def __init__(self, tagger: _Tagger, low: float, high: float) -> None:
        super().__init__(tagger, low, high)


class DiscreteParameter(_ModelParameterMixin, _IntParameter[float, float]):
    _variations: tuple[float, ...]
    _uncertainties: tuple[tuple[float, ...], ...]

    def __init__(
        self,
        tagger: _Tagger,
        variations: Iterable[float],
        *uncertainties: Iterable[float],
    ) -> None:
        super().__init__(tagger, variations, *uncertainties)


class CategoricalParameter(_ModelParameterMixin, _IntParameter[str, str]):
    _variations: tuple[str, ...]
    _uncertainties: tuple[tuple[str, ...], ...]

    def __init__(
        self,
        tagger: _Tagger,
        variations: Iterable[str],
        *uncertainties: Iterable[str],
    ) -> None:
        super().__init__(tagger, variations, *uncertainties)


class WeatherParameter(_IntParameter[Path | str, Path]):
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


AnyIntParameter: TypeAlias = DiscreteParameter | CategoricalParameter
AnyParameter: TypeAlias = ContinuousParameter | AnyIntParameter
Parameter = TypeVar("Parameter", AnyParameter, AnyIntParameter)

#############################################################################
#######                  PARAMETERS MANAGER CLASSES                   #######
#############################################################################
class _ParametersManager(Generic[Parameter]):
    _weather: WeatherParameter
    _parameters: tuple[Parameter, ...]

    def __init__(
        self, weather: WeatherParameter, parameters: Iterable[Parameter]
    ) -> None:
        self._weather = weather
        self._parameters = tuple(parameters)

    def _tagged_model(self, model_file: Path) -> str:
        macros, regulars = _split_model(model_file)
        if hasattr(cf, "_config"):
            idf = openidf(StringIO(regulars), cf._config["schema.energyplus"])
        else:
            idf = openidf(StringIO(regulars))
            cf.config_energyplus(
                version=idf.idfobjects["Version"][0]["Version_Identifier"]
            )

        for parameter in self._parameters:
            tagger = parameter._tagger
            # NOTE: match-case crashes mypy here
            if isinstance(tagger, _RegularTagger):
                idf = tagger._tagged(idf)
            elif isinstance(tagger, _MacroTagger):
                macros = tagger._tagged(macros)

        return macros + idf.idfstr()

    def _jobs(self, *variation_vecs: cf.AnyVariationVec) -> Iterator[cf.AnyJob]:
        len_job_count = int(log10(len(variation_vecs))) + 1
        for job_idx, variation_vec in enumerate(variation_vecs):
            # TODO: remove typing after python/mypy#12280 / 3.11
            weather_variation_idx: int = variation_vec[0]
            parameter_variation_vals = variation_vec[1:]
            job_uid = f"J{job_idx:0{len_job_count}}"

            # TODO: mypy infers uncertainty_vecs incorrectly, might be resolved after python/mypy#12280
            # NOTE: there may be a better way than cast()
            uncertainty_vecs = tuple(
                product(
                    range(self._weather._ns_uncertainty[weather_variation_idx]),
                    *(
                        (cast(float, val),)
                        if isinstance(parameter, ContinuousParameter)
                        else range(parameter._ns_uncertainty[cast(int, val)])
                        for val, parameter in zip(
                            parameter_variation_vals, self._parameters
                        )
                    ),
                )
            )
            uncertainty_vecs = cast(tuple[cf.AnyUncertaintyVec, ...], uncertainty_vecs)

            len_task_count = int(log10(len(uncertainty_vecs))) + 1
            tasks = tuple(
                (
                    f"T{task_idx:0{len_task_count}}",
                    tuple(zip(variation_vec, uncertainty_vec)),
                )
                for task_idx, uncertainty_vec in enumerate(uncertainty_vecs)
            )
            tasks = cast(tuple[tuple[str, cf.AnyVUMat], ...], tasks)
            yield job_uid, tasks

    def _detagged_model(
        self, tagged_model: str, parameter_vu_rows: tuple[cf.AnyVURow, ...]
    ) -> str:
        for parameter_vu_row, parameter in zip(parameter_vu_rows, self._parameters):
            tagged_model = tagged_model.replace(
                parameter._tagger._tag,
                str(
                    cast(cf.AnyFloatVURow, parameter_vu_row)[0]  # NOTE: cast
                    if isinstance(parameter, ContinuousParameter)
                    else parameter[cast(cf.AnyIntVURow, parameter_vu_row)]  # NOTE: cast
                ),
            )
        return tagged_model


def _all_int_parameters(
    parameters_manager: _ParametersManager[AnyParameter],
) -> TypeGuard[_ParametersManager[AnyIntParameter]]:
    return not any(
        isinstance(parameter, ContinuousParameter)
        for parameter in parameters_manager._parameters
    )
