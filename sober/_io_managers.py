from __future__ import annotations

import csv
import itertools as it
import os.path
import shutil
import warnings
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eppy import openidf

import sober.config as cf
from sober._logger import _log, _LoggerManager
from sober._multiplier import _InverseTransformQuantile
from sober._simulator import (
    _run_energyplus,
    _run_epmacro,
    _run_expandobjects,
    _split_model,
)
from sober._tools import (
    _natural_width,
    _parsed_str_iterable,
    _recorded_batch,
    _write_records,
)
from sober._typing import AnyModifierValue  # cast
from sober.input import (
    FunctionalModifier,
    WeatherModifier,
    _IDFTagger,
    _IntegralModifier,
    _RealModifier,
    _TextTagger,
)
from sober.output import RVICollector, ScriptCollector, _Collector, _CopyCollector

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from typing import Any, Final, TypeGuard

    from sober._tools import AnyParallel
    from sober._typing import (
        AnyBatch,
        AnyCoreLevel,
        AnyCtrlKeyVec,
        AnyJob,
        AnyLanguage,
        AnyModelModifier,
        AnyModelModifierValue,
        AnyModelTask,
        AnyModelType,
        AnyTask,
        NoiseSampleKwargs,
    )
    from sober.input import _Modifier

    type _AnyConverter = Callable[[float], float]
    type _AnyUIDs = tuple[str, ...]
    type _AnyBatchOutputs = tuple[tuple[float, ...], ...]


def _each_job_is_non_empty_and_starts_with_path(
    args: tuple[tuple[str, tuple[AnyModelModifierValue | AnyModifierValue, ...]], ...],
) -> TypeGuard[tuple[tuple[str, tuple[Path, *tuple[AnyModelModifierValue, ...]]], ...]]:
    # a first-item-being-Path issue
    # consider changing all relevant batch/job/task items to dict after py3.13 pep728
    # they are now in a dict.items() structure
    # they could all benefit from pep728
    return all(len(item) >= 1 for _, item in args) and all(
        isinstance(item[0], Path) for _, item in args
    )


#############################################################################
#######                    INPUTS MANAGER CLASSES                     #######
#############################################################################
class _InputManager:
    """manages input modification"""

    _MODEL_TYPES: Final = frozenset({".idf", ".imf"})

    __slots__ = (
        "_weather_input",
        "_model_inputs",
        "_has_templates",
        "_noise_sample_kwargs",
        "_tagged_model",
        "_model_type",
    )

    _weather_input: WeatherModifier
    _model_inputs: tuple[AnyModelModifier, ...]
    _has_templates: bool
    _noise_sample_kwargs: NoiseSampleKwargs
    _tagged_model: str
    _model_type: AnyModelType

    def __init__(
        self,
        weather_input: WeatherModifier,
        model_inputs: Iterable[AnyModelModifier],
        has_templates: bool,
        noise_sample_kwargs: NoiseSampleKwargs | None,
    ) -> None:
        self._weather_input = weather_input
        self._model_inputs = tuple(model_inputs)
        self._has_templates = has_templates
        self._noise_sample_kwargs = (
            {"mode": "auto"} if noise_sample_kwargs is None else noise_sample_kwargs
        )

    def __iter__(self) -> Iterator[_Modifier[Any, AnyModifierValue]]:
        yield self._weather_input
        yield from self._model_inputs

    def __len__(self) -> int:
        return 1 + len(self._model_inputs)

    @property
    def _has_ctrls(self) -> bool:
        return any(input._is_ctrl for input in self)

    @property
    def _has_noises(self) -> bool:
        return any(input._is_noise for input in self)

    @property
    def _has_real_ctrls(self) -> bool:
        return any(isinstance(input, _RealModifier) for input in self if input._is_ctrl)

    @property
    def _has_real_noises(self) -> bool:
        return any(
            isinstance(input, _RealModifier) for input in self if input._is_noise
        )

    def _prepare(self, model_file: Path) -> None:
        # check model type
        suffix = model_file.suffix
        if suffix not in self._MODEL_TYPES:
            raise NotImplementedError(f"a '{suffix}' model is not supported.")
        self._model_type = suffix  # type: ignore[assignment] # python/mypy#12535

        self._tagged_model = self._tagged(model_file)

        # assign index and label to each input
        has_names = any(input._name for input in self)
        for i, input in enumerate(self):
            input._index = i
            input._label = f"I{input._index}"

            if has_names:
                if not input._name:
                    warnings.warn(
                        f"no name is specified for '{input._label}'.", stacklevel=2
                    )

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
        if "schema_energyplus" in cf._config:
            idf = openidf(StringIO(regulars), cf._config["schema_energyplus"])
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

        return macros + cast(str, idf.idfstr())  # eppy

    def _task_items(self, ctrl_key_vec: AnyCtrlKeyVec) -> AnyJob:
        # align ctrl and noise keys and convert non-functional keys
        # TODO: consider reusing the multiplier facility here after py3.13 pep728, but maybe not worth it
        if (self._noise_sample_kwargs["mode"] == "elementwise") or (
            (self._noise_sample_kwargs["mode"] == "auto") and self._has_real_noises
        ):
            size = self._noise_sample_kwargs.get("size", None)
            method = self._noise_sample_kwargs.get("method", None)
            seed = self._noise_sample_kwargs.get("seed", None)

            if (size is None) or (method is None):
                raise ValueError(
                    "the"
                    + (
                        " auto determined"
                        if self._noise_sample_kwargs["mode"] == "auto"
                        else ""
                    )
                    + " noise sample mode is 'elementwise', but the size and the method is not specified."
                )

            quantile = _InverseTransformQuantile(len(self))

            if method == "random":
                sample_quantile_vecs = quantile._random(size, seed)
            else:
                sample_quantile_vecs = quantile._latin_hypercube(size, seed)

            # set n_repeats to 1 when there is no noise but user-defined size
            # TODO: consider adding a warning here
            # TODO: consider writing a function to parse the sample kwargs for both ctrl and noise centrally
            n_repeats = size if self._has_noises else 1

            aligned = tuple(
                zip(
                    *(
                        map(input, input._key_icdf(*quantiles))
                        if input._is_noise
                        else it.repeat(input(key) if input._is_ctrl else key, n_repeats)
                        for input, key, quantiles in zip(
                            self, ctrl_key_vec, sample_quantile_vecs, strict=True
                        )
                    ),
                    strict=True,
                )
            )

            # cast: python/mypy#5247
            aligned = cast(tuple[tuple[AnyModifierValue, ...], ...], aligned)
        else:
            if self._has_real_noises:
                raise ValueError(
                    "noise sample mode 'cartesian' is incompatible with real noise variables."
                )

            aligned = tuple(
                it.product(
                    *(
                        tuple(cast(_IntegralModifier[AnyModifierValue], input))  # mypy
                        if input._is_noise
                        else (input(key) if input._is_ctrl else key,)
                        for input, key in zip(self, ctrl_key_vec, strict=True)
                    )
                )
            )

        # generate task uids
        n_tasks = len(aligned)
        i_task_width = _natural_width(n_tasks)

        # get functional values
        job = tuple(
            (
                f"T{i:0{i_task_width}}",
                tuple(
                    input(item, *(task[j] for j in input._input_indices))
                    if isinstance(input, FunctionalModifier)
                    else item
                    for input, item in zip(self, task, strict=True)
                ),
            )
            for i, task in enumerate(aligned)
        )

        if _each_job_is_non_empty_and_starts_with_path(job):
            return job
        else:
            # impossible, there is at least the weather modifer
            raise IndexError("no modifiers are defined.")

    def _job_items(self, *ctrl_key_vecs: AnyCtrlKeyVec) -> AnyBatch:
        # generate job uids
        n_jobs = len(ctrl_key_vecs)
        i_job_width = _natural_width(n_jobs)

        return tuple(
            (f"J{i:0{i_job_width}}", self._task_items(ctrl_key_vec))
            for i, ctrl_key_vec in enumerate(ctrl_key_vecs)
        )

    def _detagged(self, tagged_model: str, model_task: AnyModelTask) -> str:
        for input, value in zip(self._model_inputs, model_task, strict=True):
            tagged_model = input._detagged(tagged_model, value)
        return tagged_model

    def _record_final(
        self,
        level: AnyCoreLevel,
        record_dir: Path,
        record_rows: Iterable[Iterable[object]],
    ) -> None:
        header_row = (f"{level.capitalize()}UID", *(input._label for input in self))

        # write records
        _write_records(
            record_dir / cf._RECORDS_FILENAMES[level], header_row, *record_rows
        )

    @_LoggerManager(is_first=True)
    def _make_task(self, task_dir: Path, task: AnyTask) -> None:
        # copy the task weather file
        task_epw_file = task_dir / "in.epw"
        src_epw_file = task[0]
        shutil.copyfile(src_epw_file, task_epw_file)

        _log(task_dir, "created in.epw")

        # detag the tagged model with task input values
        model = self._detagged(self._tagged_model, task[1:])

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

    @_LoggerManager(is_first=True)
    def _make_job(self, job_dir: Path, job: AnyJob) -> None:
        # make tasks
        for task_uid, task in job:
            self._make_task(job_dir / task_uid, task)

            _log(job_dir, f"made {task_uid}")

        # record tasks
        task_record_rows = tuple((task_uid, *task) for task_uid, task in job)
        self._record_final("task", job_dir, task_record_rows)

        _log(job_dir, "recorded inputs")

    @_LoggerManager(is_first=True)
    def _make_batch(
        self, batch_dir: Path, batch: AnyBatch, parallel: AnyParallel
    ) -> None:
        # schedule and make jobs
        scheduled = parallel._starmap(
            self._make_job, ((batch_dir / job_uid, job) for job_uid, job in batch)
        )

        for i, _ in enumerate(scheduled):
            _log(batch_dir, f"made {batch[i][0]}")

        # record jobs
        job_record_rows = tuple(
            (
                job_uid,
                *(
                    job[0][1][i] if input._is_ctrl else input._hype_ctrl_value()
                    for i, input in enumerate(self)
                ),
            )
            for job_uid, job in batch
        )
        self._record_final("job", batch_dir, job_record_rows)

        _log(batch_dir, "recorded inputs")

    @_LoggerManager()
    def _simulate_task(self, task_dir: Path) -> None:
        # simulate the task model
        _run_energyplus(task_dir)

    @_LoggerManager()
    def _simulate_batch(
        self, batch_dir: Path, batch: AnyBatch, parallel: AnyParallel
    ) -> None:
        # schedule and simulate tasks
        uid_pairs = tuple(
            (job_uid, task_uid) for job_uid, job in batch for task_uid, _ in job
        )
        scheduled = parallel._map(
            self._simulate_task,
            (batch_dir / job_uid / task_uid for job_uid, task_uid in uid_pairs),
        )

        for item, _ in zip(uid_pairs, scheduled, strict=True):
            _log(batch_dir, f"simulated {'-'.join(item)}")


#############################################################################
#######                    OUTPUTS MANAGER CLASSES                    #######
#############################################################################
class _OutputManager:
    """manages output collection"""

    _DEFAULT_CLEAN_PATTERNS: Final = frozenset({"*.audit", "*.end", "sqlite.err"})

    __slots__ = (
        "_task_outputs",
        "_job_outputs",
        "_clean_patterns",
        "_removes_subdirs",
        "_objectives",
        "_constraints",
        "_objective_indices",
        "_constraint_indices",
        "_to_objectives",
        "_to_constraints",
    )

    _task_outputs: tuple[_Collector, ...]
    _job_outputs: tuple[_Collector, ...]
    _clean_patterns: frozenset[str]
    _removes_subdirs: bool
    _objectives: tuple[str, ...]
    _constraints: tuple[str, ...]
    _objective_indices: tuple[int, ...]
    _constraint_indices: tuple[int, ...]
    _to_objectives: tuple[_AnyConverter, ...]
    _to_constraints: tuple[_AnyConverter, ...]

    def __init__(
        self,
        outputs: Iterable[_Collector],
        clean_patterns: str | Iterable[str],
        removes_subdirs: bool,
    ) -> None:
        # split collectors as per their level
        outputs = tuple(outputs)
        self._task_outputs = tuple(item for item in outputs if item._level == "task")
        self._job_outputs = tuple(item for item in outputs if item._level == "job")

        # parse clean patterns without duplicates
        self._clean_patterns = frozenset(
            map(
                os.path.normpath, _parsed_str_iterable(clean_patterns, "clean patterns")
            )
        )
        self._removes_subdirs = removes_subdirs

    def __iter__(self) -> Iterator[_Collector]:
        yield from self._task_outputs
        yield from self._job_outputs

    def __len__(self) -> int:
        return len(self._task_outputs) + len(self._job_outputs)

    @property
    def _has_rvis(self) -> bool:
        return any(isinstance(output, RVICollector) for output in self)

    @property
    def _languages(self) -> frozenset[AnyLanguage]:
        return frozenset(
            output._script_language
            for output in self
            if isinstance(output, ScriptCollector)
        )

    def _prepare(self, config_dir: Path, has_noises: bool) -> None:
        # add copy collectors if no noise
        if not has_noises:
            if len(self._job_outputs):
                raise ValueError(
                    "all output collectors need to be on the task level with no noise input."
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
                            item._is_final,
                        )
                    )
                    item._is_copied = True
            self._job_outputs = tuple(copy_outputs)

        # gather objective and constraint labels
        self._objectives = tuple(
            it.chain.from_iterable(item._objectives for item in self._job_outputs)
        )
        self._constraints = tuple(
            it.chain.from_iterable(item._constraints for item in self._job_outputs)
        )

        # touch rvi files
        for item in self._task_outputs:
            if isinstance(item, RVICollector):
                item._touch(config_dir)

        self._check_args()  # has to be here after copying outputs

    def _check_args(self) -> None:
        # check each output
        for output in self:
            output._check_args()

        # make sure the clean patterns provided do not interfere parent folders
        # NOTE: this check may not be comprehensive
        if any(
            item.startswith("..") or os.path.isabs(item)
            for item in self._clean_patterns
        ):
            raise ValueError(
                f"only files inside the task directory can be cleaned: {tuple(self._clean_patterns)}."
            )

        # check duplicates in objectives and constraints
        for name in ("_objectives", "_constraints"):
            labels = getattr(self, name)
            if len(labels) != len(set(labels)):
                raise ValueError(f"duplicates found in {name[1:]}: {labels}.")

    def _record_final(
        self, level: AnyCoreLevel, record_dir: Path, uids: _AnyUIDs
    ) -> None:
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
                output = cast(_Collector, output)  # cast: python/mypy#11142
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
                            warnings.warn(
                                f"multiple output lines found in '{output._filename}', only the first recorded.",
                                stacklevel=2,
                            )

        # write records
        _write_records(
            record_dir / cf._RECORDS_FILENAMES[level], header_row, *record_rows
        )

    @_LoggerManager()
    def _scan_task(self, task_dir: Path) -> None:
        # collect task outputs
        for item in self._task_outputs:
            item(task_dir)

    @_LoggerManager()
    def _scan_job(self, job_dir: Path, task_uids: _AnyUIDs) -> None:
        # scan tasks
        for task_uid in task_uids:
            self._scan_task(job_dir / task_uid)

            _log(job_dir, f"scanned {task_uid}")

        # record task output values
        self._record_final("task", job_dir, task_uids)

        _log(job_dir, "recorded final outputs")

        # collect job outputs
        for item in self._job_outputs:
            item(job_dir)

    @_LoggerManager()
    def _scan_batch(
        self, batch_dir: Path, batch: AnyBatch, parallel: AnyParallel
    ) -> None:
        # schedule and scan jobs
        scheduled = parallel._starmap(
            self._scan_job,
            (
                (batch_dir / job_uid, tuple(task_uid for task_uid, _ in job))
                for job_uid, job in batch
            ),
        )

        for (job_uid, _), _ in zip(batch, scheduled, strict=True):
            _log(batch_dir, f"scanned {job_uid}")

        # record job output values
        self._record_final("job", batch_dir, tuple(job_uid for job_uid, _ in batch))

        _log(batch_dir, "recorded final outputs")

    @_LoggerManager()
    def _clean_task(self, task_dir: Path) -> None:
        # clean task files
        for path in task_dir.glob("*"):
            if any(
                path.match(pattern) and path.is_file()
                for pattern in self._clean_patterns
            ):
                path.unlink()  # NOTE: missing is handled by the is_file check

                _log(task_dir, f"deleted {path.relative_to(task_dir)}")

    @_LoggerManager()
    def _clean_batch(
        self, batch_dir: Path, batch: AnyBatch, parallel: AnyParallel
    ) -> None:
        # schedule and clean tasks
        uid_pairs = tuple(
            (job_uid, task_uid) for job_uid, job in batch for task_uid, _ in job
        )
        scheduled = parallel._map(
            self._clean_task,
            (batch_dir / job_uid / task_uid for job_uid, task_uid in uid_pairs),
        )

        for item, _ in zip(uid_pairs, scheduled, strict=True):
            _log(batch_dir, f"cleaned {'-'.join(item)}")

    def _recorded_objectives(self, batch_dir: Path) -> _AnyBatchOutputs:
        # slice objective values
        return tuple(
            tuple(
                func(float(job_values[i]))
                for i, func in zip(
                    self._objective_indices, self._to_objectives, strict=True
                )
            )
            for job_values in _recorded_batch(batch_dir)
        )

    def _recorded_constraints(self, batch_dir: Path) -> _AnyBatchOutputs:
        # slice constraints values
        return tuple(
            tuple(
                func(float(job_values[i]))
                for i, func in zip(
                    self._constraint_indices, self._to_constraints, strict=True
                )
            )
            for job_values in _recorded_batch(batch_dir)
        )
