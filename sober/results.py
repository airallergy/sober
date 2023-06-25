import csv
from pathlib import Path
from warnings import warn
from functools import cache
from shutil import copyfile
from abc import ABC, abstractmethod
from os.path import isabs, normpath
from itertools import chain, product
from typing import Final, Literal, TypeAlias
from collections.abc import Callable, Iterable, Iterator

from . import config as cf
from ._simulator import _run_readvars
from ._logger import _log, _LoggerManager
from ._typing import AnyJob, AnyUIDs, AnyStrPath, AnyBatchResults
from ._tools import _run, _uuid, _Parallel, _write_records, _rectified_str_iterable

##############################  module typing  ##############################
_AnyLevel: TypeAlias = Literal["task", "job"]
_AnyDirection: TypeAlias = Literal["minimise", "maximise"]
_AnyBounds: TypeAlias = tuple[None, float] | tuple[float, None] | tuple[float, float]
_AnyConverter: TypeAlias = Callable[[float], float]
_AnyOutputType: TypeAlias = Literal["variable", "meter"]
#############################################################################


#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Collector(ABC):
    """an abstract base class for result collector"""

    _filename: str
    _level: _AnyLevel
    _objectives: tuple[str, ...]
    _constraints: tuple[str, ...]
    _direction: _AnyDirection
    _bounds: _AnyBounds
    _is_final: bool
    _is_copied: bool

    __slots__ = (
        "_filename",
        "_level",
        "_objectives",
        "_constraints",
        "_direction",
        "_bounds",
        "_is_final",
        "_is_copied",
    )

    @abstractmethod
    def __init__(
        self,
        filename: str,
        level: _AnyLevel,
        objectives: str | Iterable[str],
        constraints: str | Iterable[str],
        direction: _AnyDirection,
        bounds: _AnyBounds,
        is_final: bool,
    ) -> None:
        self._filename = filename
        self._level = level
        self._objectives = _rectified_str_iterable(objectives)
        self._constraints = _rectified_str_iterable(constraints)
        self._direction = direction
        self._bounds = bounds
        self._is_final = is_final
        self._is_copied = False

    def _check_args(self) -> None:
        if self._objectives:
            if (self._level != "job") and (not self._is_copied):
                raise ValueError(
                    f"a collector containing objectives needs to be at the 'job' level: {self._filename}."
                )

            if not self._is_final:
                raise ValueError(
                    f"a collector containing objectives needs to be final: {self._filename}."
                )

        if self._constraints:
            if (self._level != "job") and (not self._is_copied):
                raise ValueError(
                    f"a collector containing constraints needs to be at the 'job' level: {self._filename}."
                )

            if not self._is_final:
                raise ValueError(
                    f"a collector containing constraints needs to be final: {self._filename}."
                )

            if (self._bounds[0] and self._bounds[1]) and (
                self._bounds[0] >= self._bounds[1]
            ):
                raise ValueError(
                    f"the lower bound should be less than the upper bound for constraints: {self._filename}."
                )

        if self._is_final and (self._filename.split(".")[-1] != "csv"):
            raise ValueError(
                f"a final result needs to be a csv file: {self._filename}."
            )

    @abstractmethod
    def _collect(self, cwd: Path) -> None:
        ...

    def _to_objective(self, value: float) -> float:
        # convert each objective to minimise
        return value * {"minimise": 1, "maximise": -1}[self._direction]

    def _to_constraint(self, value: float) -> float:
        # convert each constraint to <= 0
        match self._bounds:
            case (None, float() as upper):
                return value - upper
            case (float() as lower, None):
                return lower - value
            case (float() as lower, float() as upper):
                # lower <= value <= upper
                # ==> - (upper - lower) / 2 <= value - (upper + lower) / 2 <= (upper - lower) / 2
                # ==> abs(2 < value - (upper + lower) / 2) <= (upper - lower) / 2
                return abs(value - (upper + lower) / 2) - (upper - lower) / 2
            case _:
                raise ValueError(f"bounds not recognised: {self._bounds}.")


#############################################################################
#######                       COLLECTOR CLASSES                       #######
#############################################################################
class RVICollector(_Collector):
    """collects rvi results"""

    SUFFIXES: Final = {"variable": "eso", "meter": "mtr"}

    _output_names: tuple[str, ...]
    _output_type: _AnyOutputType
    _keys: tuple[str, ...]
    _frequency: str
    _rvi_file: Path

    __slots__ = ("_output_names", "_output_type", "_keys", "_frequency", "_rvi_file")

    def __init__(
        self,
        output_names: str | Iterable[str],
        output_type: _AnyOutputType,
        filename: str,
        /,
        keys: str | Iterable[str] = (),
        frequency: str = "",
        *,
        level: _AnyLevel = "task",
        objectives: str | Iterable[str] = (),
        constraints: str | Iterable[str] = (),
        direction: _AnyDirection = "minimise",
        bounds: _AnyBounds = (None, 0),
        is_final: bool = True,
    ) -> None:
        self._output_names = _rectified_str_iterable(output_names)
        self._output_type = output_type
        self._keys = _rectified_str_iterable(keys)
        self._frequency = frequency

        super().__init__(
            filename, level, objectives, constraints, direction, bounds, is_final
        )

    def _check_args(self) -> None:
        super()._check_args()

        if self._filename.split(".")[-1] != "csv":
            raise ValueError(
                f"a RVICollector result needs to be a csv file: {self._filename}."
            )
        if self._level != "task":
            raise ValueError(f"a RVICollector result needs to be on the task level.")

    def _touch(self, config_directory: Path) -> None:
        rvi_str = f"eplusout.{self.SUFFIXES[self._output_type]}\n{self._filename}\n"
        match self._keys:
            case ():
                rvi_str += "\n".join(self._output_names)
            case _:
                if self._output_type == "meter":
                    raise ValueError("meter variables do not accept keys.")

                rvi_str += "\n".join(
                    f"{key},{name}"
                    for key, name in product(self._keys, self._output_names)
                )
        rvi_str += "\n0\n"

        rvi_filestem = _uuid(self.__class__.__name__, *rvi_str.splitlines())
        self._rvi_file = config_directory / (rvi_filestem + ".rvi")
        with self._rvi_file.open("wt") as fp:
            fp.write(rvi_str)

    def _collect(self, cwd: Path) -> None:
        _run_readvars(cwd, self._rvi_file, self._frequency)

        # remove trailing space
        # with (cwd / self._filename).open("rt") as fp:
        #     lines = fp.read().splitlines()
        # with (cwd / self._filename).open("wt") as fp:
        #     fp.write("\n".join(line.strip() for line in lines) + "\n")


class ScriptCollector(_Collector):
    """collects script results"""

    _script_file: Path
    _language: cf.AnyLanguage
    _extra_args: tuple[str, ...]

    __slots__ = ("_script_file", "_language", "_extra_args")

    def __init__(
        self,
        script_file: AnyStrPath,
        language: cf.AnyLanguage,
        filename: str,
        /,
        *extra_args: str,
        level: _AnyLevel = "task",
        objectives: str | Iterable[str] = (),
        constraints: str | Iterable[str] = (),
        direction: _AnyDirection = "minimise",
        bounds: _AnyBounds = (None, 0),
        is_final: bool = True,
    ) -> None:
        self._script_file = Path(script_file)
        self._language = language
        self._extra_args = extra_args

        super().__init__(
            filename, level, objectives, constraints, direction, bounds, is_final
        )

    def _collect(self, cwd: Path) -> None:
        # TODO: python/mypy#12554
        language_exec = cf._config[
            "exec." + self._language  # type:ignore[literal-required]
        ]

        cmd_args = (
            language_exec,
            self._script_file,
            cwd,
            self._filename,
            ",".join(self._objectives) + ";" + ",".join(self._constraints),
            ",".join(self._extra_args),
        )

        _run(cmd_args, cwd)


class _CopyCollector(_Collector):
    """copies task final results as job final results
    NOTE: this is for handling non-uncertain cases only"""

    def __init__(
        self,
        filename: str,
        objectives: str | Iterable[str],
        constraints: str | Iterable[str],
        direction: _AnyDirection,
        bounds: _AnyBounds,
    ) -> None:
        super().__init__(
            filename, "job", objectives, constraints, direction, bounds, True
        )

    def _collect(self, cwd: Path) -> None:
        copyfile(cwd / "T0" / self._filename, cwd / self._filename)


#############################################################################
#######                    RESULTS MANAGER CLASSES                    #######
#############################################################################
class _ResultsManager:
    """manages results collection"""

    _DEFAULT_CLEAN_PATTERNS: Final = (
        "*.audit",
        "*.end",
        "sqlite.err",
    )

    _task_results: tuple[_Collector, ...]
    _job_results: tuple[_Collector, ...]
    _clean_patterns: frozenset[str]
    _objectives: tuple[str, ...]
    _constraints: tuple[str, ...]
    _objective_indices: tuple[int, ...]
    _constraint_indices: tuple[int, ...]
    _to_objectives: tuple[_AnyConverter, ...]
    _to_constraints: tuple[_AnyConverter, ...]

    __slots__ = (
        "_task_results",
        "_job_results",
        "_clean_patterns",
        "_objectives",
        "_constraints",
        "_objective_indices",
        "_constraint_indices",
        "_to_objectives",
        "_to_constraints",
    )

    def __init__(
        self,
        results: Iterable[_Collector],
        clean_patterns: str | Iterable[str],
        has_uncertainties: bool,
    ) -> None:
        # parse collectors
        results = tuple(results)

        # add copy collectors if no uncertainty
        auto_results = []
        if not has_uncertainties:
            for item in results:
                if item._level == "job":
                    raise ValueError(
                        "all results collectors need to be on the task level when no uncertainty in parameters."
                    )

                if item._is_final:
                    auto_results.append(
                        _CopyCollector(
                            item._filename,
                            item._objectives,
                            item._constraints,
                            item._direction,
                            item._bounds,
                        )
                    )
                    item._is_copied = True
            results += tuple(auto_results)

        # split collectors as per their level
        self._task_results = tuple(
            result for result in results if result._level == "task"
        )
        self._job_results = tuple(
            result for result in results if result._level == "job"
        )

        # parse clean patterns without duplicates
        self._clean_patterns = frozenset(
            normpath(item) for item in _rectified_str_iterable(clean_patterns)
        )

        # gather objective and constraint labels
        self._objectives = tuple(
            chain.from_iterable(item._objectives for item in self._job_results)
        )
        self._constraints = tuple(
            chain.from_iterable(item._constraints for item in self._job_results)
        )

        self._check_args()

    def __iter__(self) -> Iterator[_Collector]:
        yield from self._task_results
        yield from self._job_results

    def __len__(self) -> int:
        return len(self._task_results) + len(self._job_results)

    def _check_args(self) -> None:
        # check each result
        for result in self:
            result._check_args()

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

    def _touch_rvi(self, config_directory: Path) -> None:
        for result in self._task_results:
            if isinstance(result, RVICollector):
                result._touch(config_directory)

    def _record_final(
        self, level: _AnyLevel, record_directory: Path, uids: AnyUIDs
    ) -> None:
        # only final results
        with (record_directory / cf._RECORDS_FILENAMES[level]).open(
            "rt", newline=""
        ) as fp:
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

        for idx, uid in enumerate(uids):
            # TODO: consider changing this assert to using a dict with uid as keys
            #       this may somewhat unify the _record_final func for parameters and results
            assert uid == record_rows[idx][0]

            for result in getattr(self, f"_{level}_results"):
                if not result._is_final:
                    continue

                with (record_directory / uid / result._filename).open(
                    "rt", newline=""
                ) as fp:
                    reader = csv.reader(fp, dialect="excel")

                    if idx:
                        next(reader)
                    else:
                        # append final results headers
                        # and prepare objectives and constraints
                        # do this only once when idx is 0

                        # append result headers
                        result_headers = next(reader)[1:]
                        header_row += result_headers

                        # append objective and constraint indices and conversion funcs
                        if level == "job":  # [1]
                            begin_count = len(header_row) - len(result_headers)

                            # NOTE: use of .index() relies heavily on uniqueness of labels for final results
                            #       uniqueness of objectives/constraints within an individual result is checked via _check_args
                            #       otherwise is hard to check, hence at users' discretion
                            if result._objectives:
                                self._objective_indices += tuple(
                                    begin_count + result_headers.index(item)
                                    for item in result._objectives
                                )
                                self._to_objectives += (result._to_objective,) * len(
                                    result._objectives
                                )
                            if result._constraints:
                                self._constraint_indices += tuple(
                                    begin_count + result_headers.index(item)
                                    for item in result._constraints
                                )
                                self._to_constraints += (result._to_constraint,) * len(
                                    result._constraints
                                )

                    # append final results values
                    record_rows[idx] += next(reader)[1:]

                    # check if any final result is no scalar
                    if __debug__:
                        try:
                            next(reader)
                        except StopIteration:
                            pass
                        else:
                            warn(
                                f"multiple result lines found in '{result._filename}', only the first collected."
                            )

        # write records
        _write_records(
            record_directory / cf._RECORDS_FILENAMES[level], header_row, *record_rows
        )

    @_LoggerManager(cwd_index=1)
    def _collect_task(self, task_directory: Path) -> None:
        # collect each task result
        for result in self._task_results:
            result._collect(task_directory)

    @_LoggerManager(cwd_index=1)
    def _collect_job(self, job_directory: Path, task_uids: AnyUIDs) -> None:
        # collect each task
        for task_uid in task_uids:
            self._collect_task(job_directory / task_uid)

            _log(job_directory, f"collected {task_uid}")

        # record task result values
        self._record_final("task", job_directory, task_uids)

        _log(job_directory, "recorded final results")

        for result in self._job_results:
            result._collect(job_directory)

    @_LoggerManager(cwd_index=1)
    def _collect_batch(self, batch_directory: Path, jobs: tuple[AnyJob, ...]) -> None:
        # collect batch in parallel
        with _Parallel(
            cf._config["n.processes"],
            initializer=cf._update_config,
            initargs=(cf._config,),
        ) as p:
            scheduled = p._starmap(
                self._collect_job,
                (
                    (
                        batch_directory / job_uid,
                        tuple(task_uid for task_uid, _ in tasks),
                    )
                    for job_uid, tasks in jobs
                ),
            )

            for (job_uid, _), _ in zip(jobs, scheduled, strict=True):
                _log(batch_directory, f"collected {job_uid}")

        # record job result values
        self._record_final(
            "job", batch_directory, tuple(job_uid for job_uid, _ in jobs)
        )

        _log(batch_directory, "recorded final results")

    @_LoggerManager(cwd_index=1)
    def _clean_task(self, task_directory: Path) -> None:
        # clean task files
        for path in task_directory.glob("*"):
            if any(
                path.match(pattern) and path.is_file()
                for pattern in self._clean_patterns
            ):
                path.unlink()  # NOTE: missing is handled by the is_file check

                _log(task_directory, f"deleted {path.relative_to(task_directory)}")

    @_LoggerManager(cwd_index=1)
    def _clean_batch(self, batch_directory: Path, jobs: tuple[AnyJob, ...]) -> None:
        # clean batch in parallel
        with _Parallel(
            cf._config["n.processes"],
            initializer=cf._update_config,
            initargs=(cf._config,),
        ) as p:
            pairs = tuple(
                (job_uid, task_uid) for job_uid, tasks in jobs for task_uid, _ in tasks
            )
            scheduled = p._map(
                self._clean_task,
                (batch_directory / job_uid / task_uid for job_uid, task_uid in pairs),
            )

            for pair, _ in zip(pairs, scheduled, strict=True):
                _log(batch_directory, f"cleaned {'-'.join(pair)}")

    @cache
    def _recorded_batch(self, batch_directory: Path) -> tuple[tuple[str, ...], ...]:
        # read job records
        with (batch_directory / cf._RECORDS_FILENAMES["job"]).open(
            "rt", newline=""
        ) as fp:
            reader = csv.reader(fp, dialect="excel")

            # skip the header row
            next(reader)

            return tuple(map(tuple, reader))

    def _recorded_objectives(self, batch_directory: Path) -> AnyBatchResults:
        # slice objective values
        return tuple(
            tuple(
                func(float(job_values[idx]))
                for idx, func in zip(
                    self._objective_indices, self._to_objectives, strict=True
                )
            )
            for job_values in self._recorded_batch(batch_directory)
        )

    def _recorded_constraints(self, batch_directory: Path) -> AnyBatchResults:
        # slice constraints values
        return tuple(
            tuple(
                func(float(job_values[idx]))
                for idx, func in zip(
                    self._constraint_indices, self._to_constraints, strict=True
                )
            )
            for job_values in self._recorded_batch(batch_directory)
        )
