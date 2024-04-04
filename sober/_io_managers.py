import csv
from collections.abc import Callable, Iterable, Iterator
from functools import cache
from io import StringIO
from itertools import chain, product
from os.path import isabs, normpath
from pathlib import Path
from shutil import copyfile
from typing import (
    Any,
    Final,
    Generic,
    Literal,
    TypeAlias,
    TypeGuard,
    cast,
)
from warnings import warn

from eppy import openidf

import sober.config as cf
from sober._logger import _log, _LoggerManager
from sober._simulator import (
    _run_energyplus,
    _run_epmacro,
    _run_expandobjects,
    _split_model,
)
from sober._tools import (
    AnyParallel,
    _natural_width,
    _rectified_str_iterable,
    _write_records,
)
from sober._typing import (
    AnyBatchOutputs,
    AnyCandidateVec,
    AnyDuoVec,
    AnyJob,
    AnyModelType,
    AnyScenarioVec,
    AnyTask,
    AnyUIDs,
)
from sober.input import (
    AnyIntegralModelModifier,
    AnyModelModifier,
    ContinuousModifier,
    FunctionalModifier,
    ModelModifier,
    WeatherModifier,
    _IDFTagger,
    _RealModifier,
    _TextTagger,
)
from sober.output import RVICollector, _AnyLevel, _Collector, _CopyCollector

##############################  module typing  ##############################
_AnyConverter: TypeAlias = Callable[[float], float]
#############################################################################


#############################################################################
#######                    INPUTS MANAGER CLASSES                     #######
#############################################################################
class _InputManager(Generic[ModelModifier]):
    """manages input modification"""

    MODEL_TYPES: Final = frozenset({".idf", ".imf"})

    _weather_input: WeatherModifier
    _model_inputs: tuple[ModelModifier, ...]
    _has_templates: bool
    _tagged_model: str
    _model_type: AnyModelType
    _has_uncertainties: bool

    __slots__ = (
        "_weather_input",
        "_model_inputs",
        "_has_templates",
        "_tagged_model",
        "_model_type",
        "_has_uncertainties",
    )

    def __init__(
        self,
        weather_input: WeatherModifier,
        model_inputs: Iterable[ModelModifier],
        has_templates: bool,
    ) -> None:
        self._weather_input = weather_input
        self._model_inputs = tuple(model_inputs)
        self._has_templates = has_templates

    def __iter__(self) -> Iterator[WeatherModifier | ModelModifier]:
        yield self._weather_input
        yield from self._model_inputs

    def __len__(self) -> int:
        return 1 + len(self._model_inputs)

    def _prepare(self, model_file: Path) -> None:
        # check model type
        suffix = model_file.suffix
        if suffix not in self.MODEL_TYPES:
            raise NotImplementedError(f"a '{suffix}' model is not supported.")
        self._model_type = suffix  # type: ignore[assignment] # python/mypy#12535

        self._tagged_model = self._tagged(model_file)
        self._has_uncertainties = any(input._is_uncertain for input in self)

        # assign index and label to each input
        has_names = any(input._name for input in self)
        for i, input in enumerate(self):
            input._index = i

            if isinstance(input, WeatherModifier):
                input._label = "W"  # P0
            else:
                input._label = f"P{input._index}"

            if has_names:
                if not input._name:
                    warn(f"no name is specified for '{input._label}'.")

                input._label += f":{input._name}"

        self._check_args()

    def _check_args(self) -> None:
        # check each input
        for input in self:
            input._check_args()

    def _tagged(self, model_file: Path) -> str:
        # read the model file as str
        with model_file.open("rt") as fp:
            model = fp.read()

        # tag all inputs with a _TextTagger
        # this has to happen first
        # as eppy changes the format
        # and the split below resolve the macro command paths
        # both of which affects string matching
        for item in self._model_inputs:
            tagger = item._tagger
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

        # tag all inputs with a _IDFTagger
        for item in self._model_inputs:
            tagger = item._tagger
            if isinstance(tagger, _IDFTagger):
                idf = tagger._tagged(idf)

        return macros + idf.idfstr()

    def _jobs(self, *candidate_vecs: AnyCandidateVec) -> Iterator[AnyJob]:
        i_job_width = _natural_width(len(candidate_vecs))

        for i_job, candidate_vec in enumerate(candidate_vecs):
            job_uid = f"J{i_job:0{i_job_width}}"

            # TODO: mypy infers scenario_vecs incorrectly, might be resolved after python/mypy#12280
            # NOTE: there may be a better way than cast()
            scenario_vecs = tuple(
                product(
                    *(
                        (
                            (cast(float, component),)
                            if isinstance(input, ContinuousModifier)
                            else range(input._ns_uncertainties[cast(int, component)])
                        )
                        for input, component in zip(self, candidate_vec, strict=True)
                    )
                )
            )
            scenario_vecs = cast(tuple[AnyScenarioVec, ...], scenario_vecs)

            i_task_width = _natural_width(len(scenario_vecs))

            tasks = tuple(
                (
                    f"T{i_task:0{i_task_width}}",
                    tuple(zip(candidate_vec, scenario_vec, strict=True)),
                )
                for i_task, scenario_vec in enumerate(scenario_vecs)
            )
            tasks = cast(tuple[tuple[str, AnyDuoVec], ...], tasks)

            yield job_uid, tasks

    def _detagged(self, tagged_model: str, task_input_values: list[Any]) -> str:
        for input, value in zip(self._model_inputs, task_input_values[1:], strict=True):
            if isinstance(input, FunctionalModifier) and not input._is_scalar:
                # each tag has its own value
                tagged_model = input._detagged(tagged_model, *value)
            else:
                # all tags have the same value
                tagged_model = input._detagged(tagged_model, value)
        return tagged_model

    def _record_final(
        self,
        level: Literal["task", "job"],
        record_dir: Path,
        record_rows: list[list[Any]],
    ) -> None:
        header_row = (f"{level.capitalize()}UID", *(input._label for input in self))

        # write records
        _write_records(
            record_dir / cf._RECORDS_FILENAMES[level], header_row, *record_rows
        )

    @_LoggerManager(cwd_index=1, is_first=True)
    def _make_task(self, task_dir: Path, duo_vec: AnyDuoVec) -> list[Any]:
        # create an empty list to store task input values
        input_values: list[Any] = [None] * len(self)

        # convert duo to value and store for each input
        for i, (input, duo) in enumerate(zip(self, duo_vec, strict=True)):
            if isinstance(input, FunctionalModifier):
                input_values[i] = input._value(
                    duo, *(input_values[j] for j in input._input_indices)
                )
            else:
                input_values[i] = input._value(duo)

        # copy the task weather file
        task_epw_file = task_dir / "in.epw"
        src_epw_file = input_values[0]
        copyfile(src_epw_file, task_epw_file)

        _log(task_dir, "created in.epw")

        # detag the tagged model with task input values
        model = self._detagged(self._tagged_model, input_values)

        # write the task model file
        with (task_dir / ("in" + self._model_type)).open("wt") as fp:
            fp.write(model)

        # run epmacro if needed
        if self._model_type == ".imf":
            _run_epmacro(task_dir)

        # run expandobjects if needed
        if self._has_templates:
            _run_expandobjects(task_dir)

        _log(task_dir, "created in.idf")

        return input_values

    @_LoggerManager(cwd_index=1, is_first=True)
    def _make_job(self, job_dir: Path, tasks: tuple[AnyTask, ...]) -> list[Any]:
        # record tasks input values
        task_record_rows = []
        for task_uid, duo_vec in tasks:
            task_input_values = self._make_task(job_dir / task_uid, duo_vec)
            task_record_rows.append([task_uid] + task_input_values)

            _log(job_dir, f"made {task_uid}")

        self._record_final("task", job_dir, task_record_rows)

        _log(job_dir, "recorded inputs")

        # curate job input value
        # NOTE: use duo_vec from the last loop
        input_values = list(
            (component if isinstance(input, ContinuousModifier) else input[component])
            for input, (component, _) in zip(self, duo_vec, strict=True)
        )

        return input_values

    @_LoggerManager(cwd_index=1, is_first=True)
    def _make_batch(
        self, batch_dir: Path, jobs: tuple[AnyJob, ...], parallel: AnyParallel
    ) -> None:
        # make batch in parallel
        scheduled = parallel._starmap(
            self._make_job, ((batch_dir / job_uid, tasks) for job_uid, tasks in jobs)
        )

        job_record_rows = []
        for (job_uid, _), job_input_values in zip(jobs, scheduled, strict=True):
            job_record_rows.append([job_uid] + job_input_values)

            _log(batch_dir, f"made {job_uid}")

        # record job input values
        self._record_final("job", batch_dir, job_record_rows)

        _log(batch_dir, "recorded inputs")

    @_LoggerManager(cwd_index=1)
    def _simulate_task(self, task_dir: Path) -> None:
        # simulate the task mdoel
        _run_energyplus(task_dir)

    @_LoggerManager(cwd_index=1)
    def _simulate_batch(
        self, batch_dir: Path, jobs: tuple[AnyJob, ...], parallel: AnyParallel
    ) -> None:
        # simulate batch in parallel
        pairs = tuple(
            (job_uid, task_uid) for job_uid, tasks in jobs for task_uid, _ in tasks
        )
        scheduled = parallel._map(
            self._simulate_task,
            (batch_dir / job_uid / task_uid for job_uid, task_uid in pairs),
        )

        for pair, _ in zip(pairs, scheduled, strict=True):
            _log(batch_dir, f"simulated {'-'.join(pair)}")


def _all_integral_modifiers(
    input_manager: _InputManager[AnyModelModifier],
) -> TypeGuard[_InputManager[AnyIntegralModelModifier]]:
    """checks if all integral modifiers"""

    return not any(isinstance(item, _RealModifier) for item in input_manager)


#############################################################################
#######                    OUTPUTS MANAGER CLASSES                    #######
#############################################################################
class _OutputManager:
    """manages output collection"""

    _DEFAULT_CLEAN_PATTERNS: Final = ("*.audit", "*.end", "sqlite.err")

    _task_outputs: tuple[_Collector, ...]
    _job_outputs: tuple[_Collector, ...]
    _clean_patterns: frozenset[str]
    _objectives: tuple[str, ...]
    _constraints: tuple[str, ...]
    _objective_indices: tuple[int, ...]
    _constraint_indices: tuple[int, ...]
    _to_objectives: tuple[_AnyConverter, ...]
    _to_constraints: tuple[_AnyConverter, ...]

    __slots__ = (
        "_task_outputs",
        "_job_outputs",
        "_clean_patterns",
        "_objectives",
        "_constraints",
        "_objective_indices",
        "_constraint_indices",
        "_to_objectives",
        "_to_constraints",
    )

    def __init__(
        self, outputs: Iterable[_Collector], clean_patterns: str | Iterable[str]
    ) -> None:
        # split collectors as per their level
        outputs = tuple(outputs)
        self._task_outputs = tuple(item for item in outputs if item._level == "task")
        self._job_outputs = tuple(item for item in outputs if item._level == "job")

        # parse clean patterns without duplicates
        self._clean_patterns = frozenset(
            normpath(item) for item in _rectified_str_iterable(clean_patterns)
        )

    def __iter__(self) -> Iterator[_Collector]:
        yield from self._task_outputs
        yield from self._job_outputs

    def __len__(self) -> int:
        return len(self._task_outputs) + len(self._job_outputs)

    def _prepare(self, config_dir: Path, has_uncertainties: bool) -> None:
        # add copy collectors if no uncertainty
        if not has_uncertainties:
            if len(self._job_outputs):
                raise ValueError(
                    "all output collectors need to be on the task level with no uncertain input."
                )

            copy_outputs = []
            for item in self._task_outputs:
                if item._is_final:
                    copy_outputs.append(
                        _CopyCollector(
                            item._filename,
                            item._objectives,
                            item._constraints,
                            item._direction,
                            item._bounds,
                        )
                    )
                    item._is_copied = True
            self._job_outputs = tuple(copy_outputs)

        # gather objective and constraint labels
        self._objectives = tuple(
            chain.from_iterable(item._objectives for item in self._job_outputs)
        )
        self._constraints = tuple(
            chain.from_iterable(item._constraints for item in self._job_outputs)
        )

        # touch rvi files
        self._touch_rvi(config_dir)

        self._check_args()

    def _check_args(self) -> None:
        # check each output
        for output in self:
            output._check_args()

        # make sure the clean patterns provided do not interfere parent folders
        # NOTE: this check may not be comprehensive
        if any(item.startswith("..") or isabs(item) for item in self._clean_patterns):
            raise ValueError(
                f"only files inside the task directory can be cleaned: {tuple(self._clean_patterns)}."
            )

        # check duplicates in objectives and constraints
        for name in ("_objectives", "_constraints"):
            labels = getattr(self, name)
            if len(labels) != len(set(labels)):
                raise ValueError(f"duplicates found in {name[1:]}: {labels}.")

    def _touch_rvi(self, config_dir: Path) -> None:
        for item in self._task_outputs:
            if isinstance(item, RVICollector):
                item._touch(config_dir)

    def _record_final(self, level: _AnyLevel, record_dir: Path, uids: AnyUIDs) -> None:
        # only final outputs
        with (record_dir / cf._RECORDS_FILENAMES[level]).open("rt", newline="") as fp:
            reader = csv.reader(fp, dialect="excel")

            header_row = next(reader)
            record_rows = list(reader)

        # store objective and contraint indices and conversion funcs
        # [1] this only happens once on the batch level
        #     TODO: ideally this should only happen once on the epoch level if optimisation
        if level == "job":  # [1]
            self._objective_indices = ()
            self._constraint_indices = ()
            self._to_objectives = ()
            self._to_constraints = ()

        for i, uid in enumerate(uids):
            # TODO: consider changing this assert to using a dict with uid as keys
            #       this may somewhat unify the _record_final func for inputs and outputs
            assert uid == record_rows[i][0]

            for output in getattr(self, f"_{level}_outputs"):
                if not output._is_final:
                    continue

                with (record_dir / uid / output._filename).open("rt", newline="") as fp:
                    reader = csv.reader(fp, dialect="excel")

                    if i:
                        next(reader)
                    else:
                        # append final output headers
                        # and prepare objectives and constraints
                        # do this only once when i is 0

                        # append output headers
                        output_headers = next(reader)[1:]
                        header_row += output_headers

                        # append objective and constraint indices and conversion funcs
                        if level == "job":  # [1]
                            begin_count = len(header_row) - len(output_headers)

                            # NOTE: use of .index() relies heavily on uniqueness of labels for final outputs
                            #       uniqueness of objectives/constraints within an individual output is checked via _check_args
                            #       otherwise is hard to check, hence at users' discretion
                            if output._objectives:
                                self._objective_indices += tuple(
                                    begin_count + output_headers.index(item)
                                    for item in output._objectives
                                )
                                self._to_objectives += (output._to_objective,) * len(
                                    output._objectives
                                )
                            if output._constraints:
                                self._constraint_indices += tuple(
                                    begin_count + output_headers.index(item)
                                    for item in output._constraints
                                )
                                self._to_constraints += (output._to_constraint,) * len(
                                    output._constraints
                                )

                    # append final output values
                    record_rows[i] += next(reader)[1:]

                    # check if any final output is no scalar
                    if __debug__:
                        try:
                            next(reader)
                        except StopIteration:
                            pass
                        else:
                            warn(
                                f"multiple output lines found in '{output._filename}', only the first collected."
                            )

        # write records
        _write_records(
            record_dir / cf._RECORDS_FILENAMES[level], header_row, *record_rows
        )

    @_LoggerManager(cwd_index=1)
    def _collect_task(self, task_dir: Path) -> None:
        # collect each task output
        for item in self._task_outputs:
            item._collect(task_dir)

    @_LoggerManager(cwd_index=1)
    def _collect_job(self, job_dir: Path, task_uids: AnyUIDs) -> None:
        # collect each task
        for task_uid in task_uids:
            self._collect_task(job_dir / task_uid)

            _log(job_dir, f"collected {task_uid}")

        # record task output values
        self._record_final("task", job_dir, task_uids)

        _log(job_dir, "recorded final outputs")

        for item in self._job_outputs:
            item._collect(job_dir)

    @_LoggerManager(cwd_index=1)
    def _collect_batch(
        self, batch_dir: Path, jobs: tuple[AnyJob, ...], parallel: AnyParallel
    ) -> None:
        # collect batch in parallel
        scheduled = parallel._starmap(
            self._collect_job,
            (
                (batch_dir / job_uid, tuple(task_uid for task_uid, _ in tasks))
                for job_uid, tasks in jobs
            ),
        )

        for (job_uid, _), _ in zip(jobs, scheduled, strict=True):
            _log(batch_dir, f"collected {job_uid}")

        # record job output values
        self._record_final("job", batch_dir, tuple(job_uid for job_uid, _ in jobs))

        _log(batch_dir, "recorded final outputs")

    @_LoggerManager(cwd_index=1)
    def _clean_task(self, task_dir: Path) -> None:
        # clean task files
        for path in task_dir.glob("*"):
            if any(
                path.match(pattern) and path.is_file()
                for pattern in self._clean_patterns
            ):
                path.unlink()  # NOTE: missing is handled by the is_file check

                _log(task_dir, f"deleted {path.relative_to(task_dir)}")

    @_LoggerManager(cwd_index=1)
    def _clean_batch(
        self, batch_dir: Path, jobs: tuple[AnyJob, ...], parallel: AnyParallel
    ) -> None:
        # clean batch in parallel
        pairs = tuple(
            (job_uid, task_uid) for job_uid, tasks in jobs for task_uid, _ in tasks
        )
        scheduled = parallel._map(
            self._clean_task,
            (batch_dir / job_uid / task_uid for job_uid, task_uid in pairs),
        )

        for pair, _ in zip(pairs, scheduled, strict=True):
            _log(batch_dir, f"cleaned {'-'.join(pair)}")

    @cache
    def _recorded_batch(self, batch_dir: Path) -> tuple[tuple[str, ...], ...]:
        # read job records
        with (batch_dir / cf._RECORDS_FILENAMES["job"]).open("rt", newline="") as fp:
            reader = csv.reader(fp, dialect="excel")

            # skip the header row
            next(reader)

            return tuple(map(tuple, reader))

    def _recorded_objectives(self, batch_dir: Path) -> AnyBatchOutputs:
        # slice objective values
        return tuple(
            tuple(
                func(float(job_values[i]))
                for i, func in zip(
                    self._objective_indices, self._to_objectives, strict=True
                )
            )
            for job_values in self._recorded_batch(batch_dir)
        )

    def _recorded_constraints(self, batch_dir: Path) -> AnyBatchOutputs:
        # slice constraints values
        return tuple(
            tuple(
                func(float(job_values[i]))
                for i, func in zip(
                    self._constraint_indices, self._to_constraints, strict=True
                )
            )
            for job_values in self._recorded_batch(batch_dir)
        )
