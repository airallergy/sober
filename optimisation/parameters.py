import csv
from math import log10
from io import StringIO
from pathlib import Path
from shutil import copyfile
from abc import ABC, abstractmethod
from itertools import chain, product
from uuid import NAMESPACE_X500, uuid5
from collections.abc import Callable, Iterable, Iterator
from typing import (
    Any,
    Generic,
    Literal,
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
from ._tools import _Parallel
from ._logger import _log, _LoggerManager
from ._simulator import _run_epmacro, _split_model, _run_energyplus, _run_expandobjects
from ._typing import (
    AnyJob,
    AnyTask,
    AnyVUMat,
    AnyVURow,
    AnyStrPath,
    AnyIntVURow,
    AnyModelType,
    AnyFloatVURow,
    AnyVariationVec,
    AnyUncertaintyVec,
)

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
    _idx: int


class _ModelParameterMixin(ABC):
    _tagger: _Tagger

    @abstractmethod
    def __init__(self, tagger: _Tagger, *args, **kwargs) -> None:
        self._tagger = tagger

        super().__init__(*args, **kwargs)  # NOTE: to _FloatParameter/_IntParameter

    @abstractmethod
    def _detagged(
        self, tagged_model: str, parameter_vu_row: Any, task_parameter_vals: list[Any]
    ) -> str:
        ...


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
        self._high = self._n_variations - 1

    @overload
    def __getitem__(self, index: SupportsIndex) -> _V:
        ...

    @overload
    def __getitem__(self, index: tuple[SupportsIndex, SupportsIndex]) -> _U:
        ...

    def __getitem__(self, index):
        match self._is_uncertain, index:
            case _, int() as x:
                return self._variations[x]
            case False, (int() as x, 0):
                return self._variations[x]
            case True, (int() as x, int() as y):
                return self._uncertainties[x][y]
            case False, (int(), int()):
                raise IndexError("no uncertainties defined.")
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


class ContinuousParameter(_ModelParameterMixin, _FloatParameter):
    def __init__(self, tagger: _Tagger, low: float, high: float) -> None:
        super().__init__(tagger, low, high)

    def _detagged(
        self,
        tagged_model: str,
        parameter_vu_row: AnyFloatVURow,
        task_parameter_vals: list[Any],
    ) -> str:
        val = parameter_vu_row[0]
        task_parameter_vals[self._idx] = val
        return tagged_model.replace(self._tagger._tag, str(val))


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

    def _detagged(
        self,
        tagged_model: str,
        parameter_vu_row: AnyIntVURow,
        task_parameter_vals: list[Any],
    ) -> str:
        val = self[parameter_vu_row]
        task_parameter_vals[self._idx] = val
        return tagged_model.replace(self._tagger._tag, str(val))


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

    def _detagged(
        self,
        tagged_model: str,
        parameter_vu_row: AnyIntVURow,
        task_parameter_vals: list[Any],
    ) -> str:
        val = self[parameter_vu_row]
        task_parameter_vals[self._idx] = val
        return tagged_model.replace(self._tagger._tag, str(val))


class FunctionalParameter(_ModelParameterMixin, _IntParameter[_V, _U]):
    _func: Callable[..., _V]
    _parameter_indices: tuple[int, ...]
    _args: tuple[str, ...]

    def __init__(
        self,
        tagger: _Tagger,
        func: Callable[..., _V],
        parameter_indices: tuple[int, ...],
        *args,
    ) -> None:
        super().__init__(tagger, (1,), *())
        self._func = func  # TODO: this has been solved in the master branch
        self._parameter_indices = parameter_indices
        self._args = tuple(map(str, args))

    def _detagged(
        self,
        tagged_model: str,
        parameter_vu_row: AnyIntVURow,
        task_parameter_vals: list[Any],
    ) -> str:
        val = self._func(
            *(task_parameter_vals[idx] for idx in self._parameter_indices),
            *self._args,
        )
        task_parameter_vals[self._idx] = val
        return tagged_model.replace(self._tagger._tag, str(val))


AnyIntParameter: TypeAlias = (
    DiscreteParameter | CategoricalParameter | FunctionalParameter
)
AnyParameter: TypeAlias = ContinuousParameter | AnyIntParameter
Parameter = TypeVar("Parameter", AnyParameter, AnyIntParameter)
# this TypeVar definition is so to differ a parameter manager with mixed parameter types from one that only has integers

#############################################################################
#######                  PARAMETERS MANAGER CLASSES                   #######
#############################################################################
MODEL_TYPES: frozenset[AnyModelType] = frozenset({".idf", ".imf"})


class _ParametersManager(Generic[Parameter]):
    _weather: WeatherParameter
    _parameters: tuple[Parameter, ...]
    _model_type: AnyModelType
    _tagged_model: str
    _has_templates: bool

    def __init__(
        self,
        weather: WeatherParameter,
        parameters: Iterable[Parameter],
        model_file: Path,
        has_templates: bool,
    ) -> None:
        self._weather = weather
        self._parameters = tuple(parameters)

        self._weather._idx = 0
        for idx, parameter in enumerate(self._parameters):
            parameter._idx = idx + 1

        suffix = model_file.suffix
        if suffix not in MODEL_TYPES:
            raise NotImplementedError(f"a '{suffix}' model is not supported.")

        self._model_type = suffix  # type: ignore[assignment] # python/mypy#12535
        self._tagged_model = self._tagged(model_file)
        self._has_templates = has_templates

    def __iter__(self) -> Iterator[WeatherParameter | Parameter]:
        for parameter in chain((self._weather,), self._parameters):
            yield parameter

    def __len__(self) -> int:
        return 1 + len(self._parameters)

    def _tagged(self, model_file: Path) -> str:
        macros, regulars = _split_model(model_file)
        if (not macros.rstrip()) ^ (self._model_type == ".idf"):
            raise ValueError(
                f"a '{self._model_type}' model is input, but "
                + ("no " if self._model_type == ".imf" else "")
                + "macro commands are found."
            )

        if hasattr(cf, "_config"):
            idf = openidf(StringIO(regulars), str(cf._config["schema.energyplus"]))
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

    def _jobs(self, *variation_vecs: AnyVariationVec) -> Iterator[AnyJob]:
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
            uncertainty_vecs = cast(tuple[AnyUncertaintyVec, ...], uncertainty_vecs)

            len_task_count = int(log10(len(uncertainty_vecs))) + 1
            tasks = tuple(
                (
                    f"T{task_idx:0{len_task_count}}",
                    tuple(zip(variation_vec, uncertainty_vec)),
                )
                for task_idx, uncertainty_vec in enumerate(uncertainty_vecs)
            )
            tasks = cast(tuple[tuple[str, AnyVUMat], ...], tasks)
            yield job_uid, tasks

    def _detagged(
        self,
        tagged_model: str,
        parameter_vu_mat: tuple[AnyVURow, ...],
        task_parameter_vals: list[Any],
    ) -> str:
        for parameter_vu_row, parameter in zip(parameter_vu_mat, self._parameters):
            if isinstance(parameter, ContinuousParameter):
                tagged_model = parameter._detagged(
                    tagged_model,
                    cast(AnyFloatVURow, parameter_vu_row),  # NOTE: cast
                    task_parameter_vals,
                )
            else:
                tagged_model = parameter._detagged(
                    tagged_model,
                    cast(AnyIntVURow, parameter_vu_row),  # NOTE: cast
                    task_parameter_vals,
                )
        return tagged_model

    def _record(
        self,
        level: Literal["task", "job"],
        record_directory: Path,
        rows: list[list[Any]],
    ) -> None:
        with (
            record_directory / getattr(cf, f"_{level.upper()}_RECORDS_FILENAME")
        ).open("wt") as fp:
            writer = csv.writer(fp, dialect="excel")

            writer.writerow(
                chain(
                    (f"{level.capitalize()}UID", "W"),
                    (f"P{parameter._idx}" for parameter in self._parameters),
                )
            )
            writer.writerows(map(str, row) for row in rows)

    @_LoggerManager(cwd_index=1, is_first=True)
    def _make_task(self, task_directory: Path, vu_mat: AnyVUMat) -> list[Any]:
        task_parameter_vals: list[Any] = [None] * len(self)

        # copy task weather files
        task_epw_file = task_directory / "in.epw"
        src_epw_file = self._weather[vu_mat[0]]
        task_parameter_vals[self._weather._idx] = src_epw_file
        copyfile(src_epw_file, task_epw_file)

        _log(task_directory, "created in.epw")

        # detag model with parameter values
        model = self._detagged(self._tagged_model, vu_mat[1:], task_parameter_vals)

        # write task model file
        with (task_directory / ("in" + self._model_type)).open("wt") as f:
            f.write(model)

        # run epmacro if needed
        if self._model_type == ".imf":
            _run_epmacro(task_directory)

        # run expandobjects if needed
        if self._has_templates:
            _run_expandobjects(task_directory)

        _log(task_directory, "created in.idf")

        return task_parameter_vals

    @_LoggerManager(cwd_index=1, is_first=True)
    def _make_job(self, job_directory: Path, tasks: tuple[AnyTask, ...]) -> list[Any]:
        job_parameter_vals: list[Any] = [None] * len(self)

        task_rows = []
        for task_uid, vu_mat in tasks:
            vals = self._make_task(job_directory / task_uid, vu_mat)
            task_rows.append([task_uid] + vals)

            _log(job_directory, f"made {task_uid}")

            if None in job_parameter_vals:
                job_parameter_vals = list(
                    (
                        self._weather[vu_mat[0][0]],
                        *(
                            val
                            if isinstance(parameter, ContinuousParameter)
                            else parameter[val]
                            for (val, _), parameter in zip(vu_mat[1:], self._parameters)  # type: ignore[has-type] # might be resolved after python/mypy#12280
                        ),
                    )
                )

        self._record("task", job_directory, task_rows)

        _log(job_directory, "recorded parameter values")

        return job_parameter_vals

    @_LoggerManager(cwd_index=1, is_first=True)
    def _make_batch(self, batch_directory: Path, jobs: tuple[AnyJob, ...]) -> None:
        with _Parallel(
            cf._config["n.processes"],
            initializer=cf._update_config,
            initargs=(cf._config,),
        ) as p:
            it = p.starmap_(
                self._make_job,
                ((batch_directory / job_uid, tasks) for job_uid, tasks in jobs),
            )

            job_rows = []
            for (job_uid, _), vals in zip(jobs, it):
                job_rows.append([job_uid] + vals)

                _log(batch_directory, f"made {job_uid}")

        self._record("job", batch_directory, job_rows)

        _log(batch_directory, "recorded parameter values")

    @_LoggerManager(cwd_index=1)
    def _simulate_task(self, task_directory: Path) -> None:
        _run_energyplus(task_directory)

    @_LoggerManager(cwd_index=1)
    def _simulate_batch(self, batch_directory: Path, jobs: tuple[AnyJob, ...]) -> None:
        with _Parallel(
            cf._config["n.processes"],
            initializer=cf._update_config,
            initargs=(cf._config,),
        ) as p:
            pairs = tuple(
                (job_uid, task_uid) for job_uid, tasks in jobs for task_uid, _ in tasks
            )
            it = p.map_(
                self._simulate_task,
                (batch_directory / job_uid / task_uid for job_uid, task_uid in pairs),
            )

            for pair, _ in zip(pairs, it):
                _log(batch_directory, f"simulated {'-'.join(pair)}")


def _all_int_parameters(
    parameters_manager: _ParametersManager[AnyParameter],
) -> TypeGuard[_ParametersManager[AnyIntParameter]]:
    return not any(
        isinstance(parameter, ContinuousParameter)
        for parameter in parameters_manager._parameters
    )
