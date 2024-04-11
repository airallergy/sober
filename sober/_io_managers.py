from __future__ import annotations

import csv
import itertools as it
import os.path
import shutil
import warnings
from io import StringIO
from typing import TYPE_CHECKING, cast

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
    _natural_width,
    _recorded_batch,
    _rectified_str_iterable,
    _write_records,
)
from sober._typing import AnyModifierVal  # cast
from sober.input import (
    FunctionalModifier,
    WeatherModifier,
    _IDFTagger,
    _IntegralModifier,
    _RealModifier,
    _TextTagger,
)
from sober.output import RVICollector, _Collector, _CopyCollector

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path
    from typing import Any, Final, TypeAlias

    from sober._tools import AnyParallel
    from sober._typing import (
        AnyCoreLevel,
        AnyCtrlKeyVec,
        AnyModelModifier,
        AnyModelModifierVal,
        AnyModelType,
    )
    from sober.input import _Modifier

    _AnyModelTask: TypeAlias = tuple[AnyModelModifierVal, ...]
    _AnyTask: TypeAlias = tuple[Path, *_AnyModelTask]
    _AnyTaskItem: TypeAlias = tuple[str, _AnyTask]
    _AnyJob: TypeAlias = tuple[_AnyTaskItem, ...]
    _AnyJobItem: TypeAlias = tuple[str, _AnyJob]
    _AnyBatch: TypeAlias = tuple[_AnyJobItem, ...]

    _AnyConverter: TypeAlias = Callable[[float], float]
    _AnyUIDs: TypeAlias = tuple[str, ...]
    _AnyBatchOutputs: TypeAlias = tuple[tuple[float, ...], ...]


#############################################################################
#######                    INPUTS MANAGER CLASSES                     #######
#############################################################################
class _InputManager:
    """manages input modification"""

    MODEL_TYPES: Final = frozenset({".idf", ".imf"})

    _weather_input: WeatherModifier
    _model_inputs: tuple[AnyModelModifier, ...]
    _has_templates: bool
    _tagged_model: str
    _model_type: AnyModelType
    _has_noises: bool
    _has_real_ctrls: bool
    _has_real_noises: bool

    __slots__ = (
        "_weather_input",
        "_model_inputs",
        "_has_templates",
        "_tagged_model",
        "_model_type",
        "_has_noises",
        "_has_real_ctrls",
        "_has_real_noises",
    )

    def __init__(
        self,
        weather_input: WeatherModifier,
        model_inputs: Iterable[AnyModelModifier],
        has_templates: bool,
    ) -> None:
        self._weather_input = weather_input
        self._model_inputs = tuple(model_inputs)
        self._has_templates = has_templates

    def __iter__(self) -> Iterator[_Modifier[Any, Any]]:
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
        self._has_noises = any(input._is_noise for input in self)

        self._has_real_ctrls = any(
            isinstance(input, _RealModifier) for input in self if input._is_ctrl
        )
        self._has_real_noises = any(
            isinstance(input, _RealModifier) for input in self if input._is_noise
        )

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

        return macros + cast(str, idf.idfstr())  # eppy

    def _task_items(self, ctrl_key_vec: AnyCtrlKeyVec) -> _AnyJob:
        if self._has_real_noises:
            raise NotImplementedError("real noise vars have not been implemented.")
        # align ctrl and noise keys and convert non-functional keys
        aligned = tuple(
            it.product(
                *(
                    tuple(cast(_IntegralModifier[AnyModifierVal], input))  # mypy
                    if input._is_noise
                    else (input(key) if input._is_ctrl else key,)
                    for input, key in zip(self, ctrl_key_vec, strict=True)
                )
            )
        )

        # generate task uids
        n_tasks = len(aligned)
        i_task_width = _natural_width(n_tasks)

        # get functional vals
        return tuple(
            (
                f"T{i:0{i_task_width}}",
                tuple(
                    input(item, *(aligned[i][j] for j in input._input_indices))
                    if isinstance(input, FunctionalModifier)
                    else item
                    for input, item in zip(self, task, strict=True)
                ),
            )
            for i, task in enumerate(aligned)
        )

    def _job_items(self, *ctrl_key_vecs: AnyCtrlKeyVec) -> _AnyBatch:
        # generate job uids
        n_jobs = len(ctrl_key_vecs)
        i_job_width = _natural_width(n_jobs)

        return tuple(
            (f"J{i:0{i_job_width}}", self._task_items(ctrl_key_vec))
            for i, ctrl_key_vec in enumerate(ctrl_key_vecs)
        )

    def _detagged(self, tagged_model: str, model_task: _AnyModelTask) -> str:
        for input, value in zip(self._model_inputs, model_task, strict=True):
            tagged_model = input._detagged(tagged_model, value)
        return tagged_model

    def _record_final(
        self,
        level: AnyCoreLevel,
        record_dir: Path,
        record_rows: Iterable[Iterable[Any]],
    ) -> None:
        header_row = (f"{level.capitalize()}UID", *(input._label for input in self))

        # write records
        _write_records(
            record_dir / cf._RECORDS_FILENAMES[level], header_row, *record_rows
        )

    @_LoggerManager(cwd_index=1, is_first=True)
    def _make_task(self, task_dir: Path, task: _AnyTask) -> None:
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

    @_LoggerManager(cwd_index=1, is_first=True)
    def _make_job(self, job_dir: Path, job: _AnyJob) -> None:
        # make tasks
        for task_uid, task in job:
            self._make_task(job_dir / task_uid, task)

            _log(job_dir, f"made {task_uid}")

        # record tasks
        task_record_rows = tuple((task_uid, *task) for task_uid, task in job)
        self._record_final("task", job_dir, task_record_rows)

        _log(job_dir, "recorded inputs")

    @_LoggerManager(cwd_index=1, is_first=True)
    def _make_batch(
        self, batch_dir: Path, batch: _AnyBatch, parallel: AnyParallel
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
                    job[0][1][i] if input._is_ctrl else input._hype_ctrl_val()
                    for i, input in enumerate(self)
                ),
            )
            for job_uid, job in batch
        )
        self._record_final("job", batch_dir, job_record_rows)

        _log(batch_dir, "recorded inputs")

    @_LoggerManager(cwd_index=1)
    def _simulate_task(self, task_dir: Path) -> None:
        # simulate the task mdoel
        _run_energyplus(task_dir)

    @_LoggerManager(cwd_index=1)
    def _simulate_batch(
        self, batch_dir: Path, batch: _AnyBatch, parallel: AnyParallel
    ) -> None:
        # simulate batch in parallel
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
            os.path.normpath(item) for item in _rectified_str_iterable(clean_patterns)
        )

    def __iter__(self) -> Iterator[_Collector]:
        yield from self._task_outputs
        yield from self._job_outputs

    def __len__(self) -> int:
        return len(self._task_outputs) + len(self._job_outputs)

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
        self._touch_rvi(config_dir)

        self._check_args()

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

    def _touch_rvi(self, config_dir: Path) -> None:
        for item in self._task_outputs:
            if isinstance(item, RVICollector):
                item._touch(config_dir)

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
                                f"multiple output lines found in '{output._filename}', only the first collected.",
                                stacklevel=2,
                            )

        # write records
        _write_records(
            record_dir / cf._RECORDS_FILENAMES[level], header_row, *record_rows
        )

    @_LoggerManager(cwd_index=1)
    def _collect_task(self, task_dir: Path) -> None:
        # collect task outputs
        for item in self._task_outputs:
            item._collect(task_dir)

    @_LoggerManager(cwd_index=1)
    def _collect_job(self, job_dir: Path, task_uids: _AnyUIDs) -> None:
        # collect tasks
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
        self, batch_dir: Path, batch: _AnyBatch, parallel: AnyParallel
    ) -> None:
        # schedule and collect jobs
        scheduled = parallel._starmap(
            self._collect_job,
            (
                (batch_dir / job_uid, tuple(task_uid for task_uid, _ in job))
                for job_uid, job in batch
            ),
        )

        for (job_uid, _), _ in zip(batch, scheduled, strict=True):
            _log(batch_dir, f"collected {job_uid}")

        # record job output values
        self._record_final("job", batch_dir, tuple(job_uid for job_uid, _ in batch))

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
        self, batch_dir: Path, batch: _AnyBatch, parallel: AnyParallel
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
