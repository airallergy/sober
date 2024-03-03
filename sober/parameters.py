from io import StringIO
from pathlib import Path
from shutil import copyfile
from abc import ABC, abstractmethod
from itertools import chain, product
from collections.abc import Callable, Iterable, Iterator
from typing import (
    Any,
    Final,
    Generic,
    Literal,
    TypeVar,
    TypeAlias,
    TypeGuard,
    cast,
    overload,
)

from eppy import openidf
from eppy.modeleditor import IDF
from eppy.bunchhelpers import makefieldname

from . import config as cf
from ._logger import _log, _LoggerManager
from ._tools import _uuid, _Parallel, _natural_width, _write_records
from ._simulator import _run_epmacro, _split_model, _run_energyplus, _run_expandobjects
from ._typing import (
    AnyJob,
    AnyTask,
    AnyDuoVec,
    AnyRealDuo,
    AnyStrPath,
    AnyModelType,
    AnyIntegralDuo,
    AnyScenarioVec,
    AnyCandidateVec,
)

##############################  module typing  ##############################
_T = TypeVar("_T")  # AnyValueToTag # python/typing#548
_V = TypeVar("_V")  # AnyVariationValue
_U = TypeVar("_U")  # AnyUncertaintyValue
#############################################################################


#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Tagger(ABC, Generic[_T]):
    """an abstract base class for taggers"""

    _tags: tuple[str, ...]

    __slots__ = ("_tags",)

    @abstractmethod
    def __init__(self, *feature_groups: tuple[str, ...]) -> None:
        self._tags = tuple(
            _uuid(self.__class__.__name__, *item) for item in feature_groups
        )

    @abstractmethod
    def _tagged(self, model: Any) -> Any: ...

    def _detagged(self, tagged_model: str, *values: _T) -> str:
        match len(values):
            case 0:
                raise ValueError("no values for detagging.")
            case 1:
                value = values[0]
                for tag in self._tags:
                    tagged_model = tagged_model.replace(tag, str(value))
            case _:
                for tag, value in zip(self._tags, values, strict=True):
                    tagged_model = tagged_model.replace(tag, str(value))

        return tagged_model


class _IDFTagger(_Tagger):
    """an abstract base class for taggers in the IDF format"""

    @abstractmethod
    def _tagged(self, model: IDF) -> IDF: ...


class _TextTagger(_Tagger):
    """an abstract base class for taggers in the text format"""

    @abstractmethod
    def _tagged(self, model: str) -> str: ...


class _Modifier(ABC):
    """an abstract base class for parameter modifiers"""

    _low: float
    _high: float
    _index: int
    _label: str
    _is_uncertain: bool

    __slots__ = ("_low", "_high", "_index", "_label", "_is_uncertain")

    def _check_args(self) -> None:
        pass

    @abstractmethod
    def _value(self, duo: Any, *args) -> Any:
        # *args is only used by FunctionalModifier for now

        ...


class _ModelModifierMixin(ABC, Generic[_T]):
    """an abstract base class for common functions in model modification
    (as opposed to the weather modifier)"""

    _tagger: _Tagger[_T]

    # __slots__ = ("_tagger",)  # a slots/mixin issue

    @abstractmethod
    def __init__(self, tagger: _Tagger[_T], *args, **kwargs) -> None:
        self._tagger = tagger

        super().__init__(*args, **kwargs)  # NOTE: to _RealModifier/_IntegralModifier

    def _detagged(self, tagged_model: str, *values: _T) -> str:
        return self._tagger._detagged(tagged_model, *values)


class _RealModifier(_Modifier):
    """an abstract base class for parameter modifiers of real variables in pymoo"""

    @abstractmethod
    def __init__(self, low: float, high: float) -> None:
        self._low = low
        self._high = high
        self._is_uncertain = False  # hard-coded for now


class _IntegralModifier(_Modifier, Generic[_V, _U]):
    """an abstract base class for parameter modifiers of integral variables in pymoo"""

    _low: int
    _high: int
    _variations: tuple[_V, ...]
    _uncertainties: tuple[tuple[_U, ...], ...]
    _n_variations: int
    _ns_uncertainties: tuple[int, ...]

    __slots__ = (
        "_low",
        "_high",
        "_variations",
        "_uncertainties",
        "_n_variations",
        "_ns_uncertainties",
    )

    @abstractmethod
    def __init__(
        self,
        variations: Iterable[_V],
        *uncertainties: Iterable[_U],
    ) -> None:
        # parse and count variations
        self._variations = tuple(variations)
        self._n_variations = len(self._variations)

        # parse and count uncertainties
        # NOTE: when no uncertainty, the count of uncertainties is 1
        match len(uncertainties):
            case 0:
                # no uncertainty
                # self._uncertainties = ()  # leave undefined
                self._ns_uncertainties = (1,) * self._n_variations
            case 1:
                # each variation has the same uncertainty
                self._uncertainties = (tuple(uncertainties[0]),) * self._n_variations
                self._ns_uncertainties = (
                    len(self._uncertainties[0]),
                ) * self._n_variations
            case self._n_variations:
                # each variation has different uncertainties
                self._uncertainties = tuple(map(tuple, uncertainties))
                self._ns_uncertainties = tuple(map(len, self._uncertainties))
            case _:
                raise ValueError(
                    f"the number of uncertainties is different from that of variations: '{len(uncertainties)}', '{self._n_variations}'."
                )

        self._is_uncertain = set(self._ns_uncertainties) != {1}
        self._low = 0
        self._high = self._n_variations - 1

    @overload
    def __getitem__(self, index: int) -> _V: ...

    @overload
    def __getitem__(self, index: tuple[int, int]) -> _U: ...

    def __getitem__(self, index):
        match self._is_uncertain, index:
            case _, int() as idx:
                return self._variations[idx]
            case False, (int() as idx, 0):
                return self._variations[idx]
            case True, (int() as idx, int() as jdx):
                return self._uncertainties[idx][jdx]
            case False, (int(), int()):
                raise IndexError("no uncertainties defined.")
            case _:
                raise IndexError(f"invalid index: {index}.")


#############################################################################
#######                        TAGGER CLASSES                         #######
#############################################################################
class IndexTagger(_IDFTagger):
    """tags regular commands by indexing

    index_trio == (class_name, object_name, field_name)

    no support for nested regular commands inside macro files
    """

    _index_trios: tuple[tuple[str, str, str], ...]

    __slots__ = ("_index_trios",)

    def __init__(self, /, *index_trios: Iterable[str]) -> None:
        _index_trios = tuple(tuple(item) for item in index_trios)  # python/mypy#11682

        if any(len(item) != 3 for item in _index_trios):
            raise ValueError(
                "each index trio should contain exactly three elements: class_name, object_name, field_name."
            )

        # TODO: cast may be removed after python/mypy#1178
        _index_trios = cast(tuple[tuple[str, str, str], ...], _index_trios)

        # remove duplicate trios
        self._index_trios = tuple(set(_index_trios))

        super().__init__(*self._index_trios)

    def _tagged(self, model: IDF) -> IDF:
        for (class_name, object_name, field_name), tag in zip(
            self._index_trios, self._tags, strict=True
        ):
            obj = model.getobject(class_name, object_name)
            if obj is None:
                raise ValueError(f"object is not found in the model: '{object_name}'.")
                # eppy throws a proper error for unknown field names

            obj[makefieldname(field_name)] = tag
        return model


class StringTagger(_TextTagger):
    """tags regular and macro commands by string replacement

    string_trio == (string, prefix, suffix)

    no support for nested macro commands inside macro files
    """

    _string_trios: tuple[tuple[str, str, str], ...]

    __slots__ = ("_string_trios",)

    def __init__(self, /, *string_trios: Iterable[str]) -> None:
        _string_trios = tuple(tuple(item) for item in string_trios)  # python/mypy#11682

        if any(len(item) == 0 for item in _string_trios):
            raise ValueError(
                "each string trio should contain at least one element: string."
            )
        if any(len(item) > 3 for item in _string_trios):
            raise ValueError(
                "each string trio should contain at most three elements: string, prefix, suffix."
            )

        # assign empty string to prefix/suffix if absent
        _string_trios = tuple(item + ("",) * (3 - len(item)) for item in _string_trios)

        # TODO: cast may be removed after python/mypy#1178
        _string_trios = cast(tuple[tuple[str, str, str], ...], _string_trios)

        for string, prefix, suffix in _string_trios:
            if not (string.startswith(prefix) and string.endswith(suffix)):
                raise ValueError(
                    f"string needs to share the prefix and the suffix: '{string}, {prefix}, {suffix}'."
                )

        # remove duplicate trios
        self._string_trios = tuple(set(_string_trios))

        super().__init__(*self._string_trios)

    def _tagged(self, model: str) -> str:
        for (string, prefix, suffix), tag in zip(
            self._string_trios, self._tags, strict=True
        ):
            if string not in model:
                raise ValueError(f"string is not found in the model: '{string}'.")

            model = model.replace(string, prefix + tag + suffix)
        return model


#############################################################################
#######                       PARAMETER CLASSES                       #######
#############################################################################
class WeatherModifier(_IntegralModifier[Path | str, Path]):
    """modifies the weather parameter"""

    _variations: tuple[Path | str, ...]
    _uncertainties: tuple[tuple[Path, ...], ...]

    __slots__ = ("_variations", "_uncertainties")

    @overload
    def __init__(self, variations: Iterable[AnyStrPath], /) -> None: ...

    @overload
    def __init__(
        self, variations: Iterable[str], /, *uncertainties: Iterable[AnyStrPath]
    ) -> None: ...

    def __init__(self, variations, /, *uncertainties):
        super().__init__(variations, *uncertainties)

        # resolve weather file paths
        if self._is_uncertain:
            self._variations = tuple(str(item) for item in self._variations)
            self._uncertainties = tuple(
                tuple(Path(item).resolve(True) for item in uncertainty)
                for uncertainty in self._uncertainties
            )
        else:
            self._variations = tuple(
                Path(item).resolve(True) for item in self._variations
            )
            assert not hasattr(self, "_uncertainties")

    def _value(self, duo: AnyIntegralDuo, *args) -> Path:
        return self[duo]


class ContinuousModifier(_ModelModifierMixin[float], _RealModifier):
    """modifies continuous parameters"""

    def __init__(self, tagger: _Tagger, low: float, high: float, /) -> None:
        # TODO: uncertainty of continuous parameters
        super().__init__(tagger, low, high)

    def _value(self, duo: AnyRealDuo, *args) -> float:
        return duo[0]

    def _detagged(self, tagged_model: str, *values: float) -> str:
        return super()._detagged(tagged_model, *values)


class DiscreteModifier(_ModelModifierMixin[float], _IntegralModifier[float, float]):
    """modifies discrete parameters"""

    _variations: tuple[float, ...]
    _uncertainties: tuple[tuple[float, ...], ...]

    __slots__ = ("_variations", "_uncertainties")

    def __init__(
        self,
        tagger: _Tagger,
        variations: Iterable[float],
        /,
        *uncertainties: Iterable[float],
    ) -> None:
        super().__init__(tagger, variations, *uncertainties)

    def _value(self, duo: AnyIntegralDuo, *args) -> float:
        return self[duo]

    def _detagged(self, tagged_model: str, *values: float) -> str:
        return super()._detagged(tagged_model, *values)


class CategoricalModifier(_ModelModifierMixin[str], _IntegralModifier[str, str]):
    """modifies categorical parameters"""

    _variations: tuple[str, ...]
    _uncertainties: tuple[tuple[str, ...], ...]

    __slots__ = ("_variations", "_uncertainties")

    def __init__(
        self,
        tagger: _Tagger,
        variations: Iterable[str],
        /,
        *uncertainties: Iterable[str],
    ) -> None:
        super().__init__(tagger, variations, *uncertainties)

    def _value(self, duo: AnyIntegralDuo, *args) -> str:
        return self[duo]

    def _detagged(self, tagged_model: str, *values: str) -> str:
        return super()._detagged(tagged_model, *values)


class FunctionalModifier(_ModelModifierMixin[_T], _IntegralModifier[_V, _U]):
    """modifies functional parameters"""

    _func: Callable[..., _V | Iterable[_V]]
    _parameter_indices: tuple[int, ...]
    _extra_args: tuple[Any, ...]
    _is_scalar: bool

    __slots__ = ("_func", "_parameter_indices", "_extra_args", "_is_scalar")

    @overload
    def __init__(
        self,
        tagger: _Tagger,
        func: Callable[..., _V],
        parameter_indices: Iterable[int],
        /,
        *extra_args: Any,  # TODO: restrict this for serialisation
        is_scalar: Literal[True],
    ) -> None: ...

    @overload
    def __init__(
        self,
        tagger: _Tagger,
        func: Callable[..., Iterable[_V]],
        parameter_indices: Iterable[int],
        /,
        *extra_args: Any,
        is_scalar: Literal[False],
    ) -> None: ...

    def __init__(self, tagger, func, parameter_indices, /, *extra_args, is_scalar=True):
        func_name = f"<function {func.__module__ + '.' + func.__code__.co_qualname}>"
        super().__init__(tagger, (func_name,))

        self._func = func
        self._parameter_indices = tuple(parameter_indices)
        self._extra_args = extra_args
        self._is_scalar = is_scalar

    def _check_args(self) -> None:
        super()._check_args()

        if any(item >= self._index for item in self._parameter_indices):
            raise ValueError(
                f"only previous parameters can be referred to: {self._index}, {self._parameter_indices}."
            )

    def _value(self, duo: AnyIntegralDuo, *args) -> _V | tuple[_V, ...]:
        # NOTE: args are values for referenced parameters
        #       args are passed as tuple whilst _extra_args unpacked
        #       this is to align their corresponding arguments in __init__

        if self._is_scalar:
            self._func = cast(Callable[..., _V], self._func)

            return self._func(args, *self._extra_args)

        else:
            self._func = cast(Callable[..., Iterable[_V]], self._func)

            return tuple(self._func(args, *self._extra_args))

    def _detagged(self, tagged_model: str, *values: _T) -> str:
        return super()._detagged(tagged_model, *values)


#############################################################################
#######                  PARAMETERS MANAGER CLASSES                   #######
#############################################################################

##############################  module typing  ##############################
# this technically belongs to _typing.py, but put here to avoid circular import
AnyIntegralModelModifier: TypeAlias = (
    DiscreteModifier | CategoricalModifier | FunctionalModifier
)
AnyRealModelModifier: TypeAlias = ContinuousModifier
AnyModelModifier: TypeAlias = AnyRealModelModifier | AnyIntegralModelModifier
# this TypeVar is defined this way
# to differ a parameter manager with mixed parameter types from one that only has integers
ModelModifier = TypeVar("ModelModifier", AnyModelModifier, AnyIntegralModelModifier)
#############################################################################


class _ParametersManager(Generic[ModelModifier]):
    """manages parameters modification"""

    MODEL_TYPES: Final = frozenset({".idf", ".imf"})

    _weather_parameter: WeatherModifier
    _model_parameters: tuple[ModelModifier, ...]
    _model_type: AnyModelType
    _tagged_model: str
    _has_templates: bool
    _has_uncertainties: bool

    __slots__ = (
        "_weather_parameter",
        "_model_parameters",
        "_model_type",
        "_tagged_model",
        "_has_templates",
        "_has_uncertainties",
    )

    def __init__(
        self,
        weather_parameter: WeatherModifier,
        model_parameters: Iterable[ModelModifier],
        model_file: Path,
        has_templates: bool,
    ) -> None:
        self._weather_parameter = weather_parameter
        self._model_parameters = tuple(model_parameters)

        # add index and label to each parameter
        for idx, parameter in enumerate(self):
            parameter._index = idx

            if isinstance(parameter, WeatherModifier):
                parameter._label = "W"  # P0
            else:
                parameter._label = f"P{parameter._index}"

        # check model type
        suffix = model_file.suffix
        if suffix not in self.MODEL_TYPES:
            raise NotImplementedError(f"a '{suffix}' model is not supported.")
        self._model_type = suffix  # type: ignore[assignment] # python/mypy#12535

        self._tagged_model = self._tagged(model_file)
        self._has_templates = has_templates
        self._has_uncertainties = any(parameter._is_uncertain for parameter in self)

        self._check_args()

    def __iter__(self) -> Iterator[WeatherModifier | ModelModifier]:
        yield self._weather_parameter
        yield from self._model_parameters

    def __len__(self) -> int:
        return 1 + len(self._model_parameters)

    def _check_args(self) -> None:
        # check each parameter
        for parameter in self:
            parameter._check_args()

    def _tagged(self, model_file: Path) -> str:
        # read the model file as str
        with model_file.open("rt") as fp:
            model = fp.read()

        # tag all parameters with a _TextTagger
        # this has to happen first
        # as eppy changes the format
        # and the split below resolve the macro command paths
        # both of which affects string matching
        for parameter in self._model_parameters:
            tagger = parameter._tagger
            if isinstance(tagger, _TextTagger):
                model = tagger._tagged(model)

        # split the model into macro and regular commands
        macros, regulars = _split_model(model, model_file.parent)

        # check if macro existence matches the model file suffix
        if (not macros.strip()) ^ (self._model_type == ".idf"):
            raise ValueError(
                f"a '{self._model_type}' model is input, but "
                + ("no " if self._model_type == ".imf" else "")
                + "macro commands are found."
            )

        # read regular commands into eppy
        # and configure energyplus if not yet
        if hasattr(cf, "_config"):
            idf = openidf(StringIO(regulars), cf._config["schema.energyplus"])
        else:
            idf = openidf(StringIO(regulars))
            cf.config_energyplus(
                version=idf.idfobjects["Version"][0]["Version_Identifier"]
            )

        # tag all parameters with a _IDFTagger
        for parameter in self._model_parameters:
            tagger = parameter._tagger
            if isinstance(tagger, _IDFTagger):
                idf = tagger._tagged(idf)

        return macros + idf.idfstr()

    def _jobs(self, *candidate_vecs: AnyCandidateVec) -> Iterator[AnyJob]:
        job_idx_width = _natural_width(len(candidate_vecs))

        for job_idx, candidate_vec in enumerate(candidate_vecs):
            job_uid = f"J{job_idx:0{job_idx_width}}"

            # TODO: mypy infers scenario_vecs incorrectly, might be resolved after python/mypy#12280
            # NOTE: there may be a better way than cast()
            scenario_vecs = tuple(
                product(
                    *(
                        (
                            (cast(float, component),)
                            if isinstance(parameter, ContinuousModifier)
                            else range(
                                parameter._ns_uncertainties[cast(int, component)]
                            )
                        )
                        for parameter, component in zip(
                            self, candidate_vec, strict=True
                        )
                    ),
                )
            )
            scenario_vecs = cast(tuple[AnyScenarioVec, ...], scenario_vecs)

            task_idx_width = _natural_width(len(scenario_vecs))

            tasks = tuple(
                (
                    f"T{task_idx:0{task_idx_width}}",
                    tuple(zip(candidate_vec, scenario_vec, strict=True)),
                )
                for task_idx, scenario_vec in enumerate(scenario_vecs)
            )
            tasks = cast(tuple[tuple[str, AnyDuoVec], ...], tasks)

            yield job_uid, tasks

    def _detagged(self, tagged_model: str, task_parameter_values: list[Any]) -> str:
        for parameter, value in zip(
            self._model_parameters, task_parameter_values[1:], strict=True
        ):
            if isinstance(parameter, FunctionalModifier) and not parameter._is_scalar:
                # each tag has its own value
                tagged_model = parameter._detagged(tagged_model, *value)
            else:
                # all tags have the same value
                tagged_model = parameter._detagged(tagged_model, value)
        return tagged_model

    def _record_final(
        self,
        level: Literal["task", "job"],
        record_directory: Path,
        record_rows: list[list[Any]],
    ) -> None:
        header_row = (
            f"{level.capitalize()}UID",
            *(parameter._label for parameter in self),
        )

        # write records
        _write_records(
            record_directory / cf._RECORDS_FILENAMES[level], header_row, *record_rows
        )

    @_LoggerManager(cwd_index=1, is_first=True)
    def _make_task(self, task_directory: Path, duo_vec: AnyDuoVec) -> list[Any]:
        # create an empty list to store task parameter values
        parameter_values: list[Any] = [None] * len(self)

        # convert duo to value and store for each parameter
        for idx, (parameter, duo) in enumerate(zip(self, duo_vec, strict=True)):
            if isinstance(parameter, FunctionalModifier):
                parameter_values[idx] = parameter._value(
                    duo,
                    *(parameter_values[jdx] for jdx in parameter._parameter_indices),
                )
            else:
                parameter_values[idx] = parameter._value(duo)

        # copy the task weather file
        task_epw_file = task_directory / "in.epw"
        src_epw_file = parameter_values[0]
        copyfile(src_epw_file, task_epw_file)

        _log(task_directory, "created in.epw")

        # detag the tagged model with task parameter values
        model = self._detagged(self._tagged_model, parameter_values)

        # write the task model file
        with (task_directory / ("in" + self._model_type)).open("wt") as fp:
            fp.write(model)

        # run epmacro if needed
        if self._model_type == ".imf":
            _run_epmacro(task_directory)

        # run expandobjects if needed
        if self._has_templates:
            _run_expandobjects(task_directory)

        _log(task_directory, "created in.idf")

        return parameter_values

    @_LoggerManager(cwd_index=1, is_first=True)
    def _make_job(self, job_directory: Path, tasks: tuple[AnyTask, ...]) -> list[Any]:
        # record tasks parameter values
        task_record_rows = []
        for task_uid, duo_vec in tasks:
            task_parameter_values = self._make_task(job_directory / task_uid, duo_vec)
            task_record_rows.append([task_uid] + task_parameter_values)

            _log(job_directory, f"made {task_uid}")

        self._record_final("task", job_directory, task_record_rows)

        _log(job_directory, "recorded parameters")

        # curate job parameter value
        # NOTE: use duo_vec from the last loop
        parameter_values = list(
            (
                component
                if isinstance(parameter, ContinuousModifier)
                else parameter[component]
            )
            for parameter, (component, _) in zip(self, duo_vec, strict=True)
        )

        return parameter_values

    @_LoggerManager(cwd_index=1, is_first=True)
    def _make_batch(self, batch_directory: Path, jobs: tuple[AnyJob, ...]) -> None:
        # make batch in parallel
        with _Parallel(
            cf._config["n.processes"],
            initializer=cf._update_config,
            initargs=(cf._config,),
        ) as p:
            scheduled = p._starmap(
                self._make_job,
                ((batch_directory / job_uid, tasks) for job_uid, tasks in jobs),
            )

            job_record_rows = []
            for (job_uid, _), job_parameter_values in zip(jobs, scheduled, strict=True):
                job_record_rows.append([job_uid] + job_parameter_values)

                _log(batch_directory, f"made {job_uid}")

        # record job parameter values
        self._record_final("job", batch_directory, job_record_rows)

        _log(batch_directory, "recorded parameters")

    @_LoggerManager(cwd_index=1)
    def _simulate_task(self, task_directory: Path) -> None:
        # simulate the task mdoel
        _run_energyplus(task_directory)

    @_LoggerManager(cwd_index=1)
    def _simulate_batch(self, batch_directory: Path, jobs: tuple[AnyJob, ...]) -> None:
        # simulate batch in parallel
        with _Parallel(
            cf._config["n.processes"],
            initializer=cf._update_config,
            initargs=(cf._config,),
        ) as p:
            pairs = tuple(
                (job_uid, task_uid) for job_uid, tasks in jobs for task_uid, _ in tasks
            )
            scheduled = p._map(
                self._simulate_task,
                (batch_directory / job_uid / task_uid for job_uid, task_uid in pairs),
            )

            for pair, _ in zip(pairs, scheduled, strict=True):
                _log(batch_directory, f"simulated {'-'.join(pair)}")


def _all_integral_modifiers(
    parameters_manager: _ParametersManager[AnyModelModifier],
) -> TypeGuard[_ParametersManager[AnyIntegralModelModifier]]:
    """checks if all integral modifiers"""

    return not any(
        isinstance(parameter, _RealModifier) for parameter in parameters_manager
    )
