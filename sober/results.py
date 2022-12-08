import csv
from pathlib import Path
from warnings import warn
from functools import cache
from itertools import chain
from shutil import copyfile
from os.path import normpath
from abc import ABC, abstractmethod
from typing import Literal, ClassVar, TypeAlias
from collections.abc import Callable, Iterable, Iterator

from typing_extensions import Unpack  # TODO: remove Unpack after 3.11

from . import config as cf
from ._tools import _run, _Parallel
from ._simulator import _run_readvars
from ._logger import _log, _LoggerManager
from ._typing import AnyJob, AnyUIDs, AnyCmdArgs, AnyStrPath, AnyBatchResults

AnyLevel: TypeAlias = Literal["task", "job"]
AnyKind: TypeAlias = Literal["objective", "constraint", "extra"]
AnyDirection: TypeAlias = Literal["minimise", "maximise"]
# AnyBounds: TypeAlias = tuple[None, float] | tuple[float, None] | tuple[float, float]
# this crashes mypy currently
AnyBounds: TypeAlias = tuple[float | None, float | None]  # TODO: python/mypy#11098
AnyConverter: TypeAlias = Callable[[float], float]
AnyOutputType: TypeAlias = Literal["variable", "meter"]

#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Collector(ABC):
    _filename: str
    _level: AnyLevel
    _kind: AnyKind
    _direction: AnyDirection
    _bounds: AnyBounds
    _is_final: bool

    @abstractmethod
    def __init__(
        self,
        filename: str,
        level: AnyLevel,
        kind: AnyKind,
        direction: AnyDirection,
        bounds: AnyBounds,
        is_final: bool,
    ) -> None:
        self._filename = filename
        self._level = level
        self._kind = kind
        self._direction = direction
        self._bounds = bounds
        self._is_final = is_final

    def _check_args(self) -> None:
        if self._kind in ("objective", "constraint"):
            assert (
                self._level == "job"
            ), f"an '{self._kind}' result needs to be at the 'job' level: {self._filename}."
            assert (
                self._is_final == True
            ), f"an '{self._kind}' result needs to be final: {self._filename}."

        if self._kind == "constraint":
            if self._bounds[0] and self._bounds[1]:
                # TODO: add support for equality after pymoo 0.60
                assert (
                    self._bounds[0] < self._bounds[1]
                ), f"the lower bound should be less than the upper bound in an '{self._kind}' result: {self._filename}."

        if self._is_final:
            assert (
                self._filename.split(".")[-1] == "csv"
            ), f"a final result needs to be a csv file: {self._filename}."
        if self.__class__.__name__ == "RVICollector":
            assert (
                self._filename.split(".")[-1] == "csv"
            ), f"a RVICollector result needs to be a csv file: {self._filename}."
            assert (
                self._level == "task"
            ), f"a RVICollector result needs to be on the task level."

    @abstractmethod
    def _collect(self, cwd: Path) -> None:
        ...

    def _to_objective(self, x: float) -> float:
        return x * {"minimise": 1, "maximise": -1}[self._direction]

    def _to_constraint(  # type:ignore[return] # python/mypy#12534
        self, x: float
    ) -> float:
        match self._bounds:
            case (None, None):
                raise ValueError(f"bounds not defined: {self._filename}")
            case (None, _ as upper):
                return x - upper
            case (_ as lower, None):
                return lower - x
            case (lower, upper):
                return abs(x - (upper + lower) / 2) - (upper - lower) / 2


#############################################################################
#######                       COLLECTOR CLASSES                       #######
#############################################################################
class RVICollector(_Collector):
    _output_name: str
    _output_type: AnyOutputType
    _rvi_file: Path
    _keys: tuple[str, ...]
    _frequency: str

    def __init__(
        self,
        output_name: str,
        output_type: AnyOutputType,
        filename: str,
        level: AnyLevel,
        kind: AnyKind,
        direction: AnyDirection = "minimise",
        bounds: AnyBounds = (None, 0),
        is_final: bool = True,
        keys: Iterable[str] = (),
        frequency: str = "",
    ) -> None:
        self._output_name = output_name
        self._output_type = output_type
        self._keys = tuple(keys)
        self._frequency = frequency

        super().__init__(filename, level, kind, direction, bounds, is_final)

    def _touch(self, config_directory: Path) -> None:
        self._rvi_file = (
            config_directory
            / f"{self._output_name.replace(' ', '_').replace(':', '_').lower()}.rvi"
        )

        suffixes = {"variable": "eso", "meter": "mtr"}
        joined_rvi_lines = f"eplusout.{suffixes[self._output_type]}\n{self._filename}\n"
        match self._keys:
            case ():
                joined_rvi_lines += self._output_name
            case _:
                if self._output_type == "meter":
                    raise ValueError("meter variables do not accept keys.")

                joined_rvi_lines += "\n".join(
                    f"{key},{self._output_name}" for key in self._keys
                )
        joined_rvi_lines += "\n0\n"

        with self._rvi_file.open("wt") as fp:
            fp.write(joined_rvi_lines)

    def _collect(self, cwd: Path) -> None:
        _run_readvars(cwd, self._rvi_file, self._frequency)


class ScriptCollector(_Collector):
    _script_file: Path
    _language: cf.AnyLanguage
    _script_args: AnyCmdArgs

    def __init__(
        self,
        script_file: AnyStrPath,
        language: cf.AnyLanguage,
        filename: str,
        level: AnyLevel,
        kind: AnyKind,
        direction: AnyDirection = "minimise",
        bounds: AnyBounds = (None, 0),
        is_final: bool = True,
        *script_args: Unpack[AnyCmdArgs],  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
    ) -> None:
        self._script_file = Path(script_file)
        self._language = language
        self._script_args = script_args
        super().__init__(filename, level, kind, direction, bounds, is_final)

    def _collect(self, cwd: Path) -> None:
        cmd_args: AnyCmdArgs = (
            cf._config["exec.python"],
            self._script_file,
            cwd,
            self._filename,
            *self._script_args,
        )

        _run(cmd_args, cwd)


class _CopyCollector(_Collector):
    # NOTE: this is for handling non-uncertain cases only
    #       copy final results from task folders to job ones

    def __init__(
        self,
        filename: str,
        kind: AnyKind,
        direction: AnyDirection = "minimise",
        bounds: AnyBounds = (None, 0),
    ) -> None:
        super().__init__(filename, "job", kind, direction, bounds, True)

    def _collect(self, cwd: Path) -> None:
        copyfile(cwd / "T0" / self._filename, cwd / self._filename)


#############################################################################
#######                    RESULTS MANAGER CLASSES                    #######
#############################################################################
class _ResultsManager:
    _DEFAULT_CLEAN_PATTERNS: ClassVar[tuple[str, ...]] = (
        "*.audit",
        "*.end",
        "sqlite.err",
    )

    _task_results: tuple[_Collector, ...]
    _job_results: tuple[_Collector, ...]
    _clean_patterns: frozenset[str]
    _objectives: tuple[_Collector, ...]
    _constraints: tuple[_Collector, ...]
    _extras: tuple[_Collector, ...]
    _objective_idxs: tuple[int, ...]
    _constraint_idxs: tuple[int, ...]
    _to_objectives: tuple[AnyConverter, ...]
    _to_constraints: tuple[AnyConverter, ...]

    def __init__(
        self,
        results: Iterable[_Collector],
        clean_patterns: Iterable[str],
        has_uncertainties: bool,
    ) -> None:
        results = tuple(results)
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
                            item._kind,
                            item._direction,
                            item._bounds,
                        )
                    )
                    item._kind = "extra"
            results += tuple(auto_results)

        for item in results:
            item._check_args()

        self._task_results = tuple(
            result for result in results if result._level == "task"
        )
        self._job_results = tuple(
            result for result in results if result._level == "job"
        )
        self._clean_patterns = frozenset(normpath(item) for item in clean_patterns)

        if any(item.startswith(("..", "/")) for item in self._clean_patterns):
            raise ValueError(
                f"only files inside the task directory can be cleaned: {tuple(self._clean_patterns)}"
            )

    def __iter__(self) -> Iterator[_Collector]:
        for collector in chain(self._task_results, self._job_results):
            yield collector

    def __getattr__(self, name: str) -> tuple[_Collector, ...]:
        # TODO: python/mypy#8203
        if name not in frozenset({"_objectives", "_constraints", "_extras"}):
            raise AttributeError

        return tuple(collector for collector in self if collector._kind == name[1:-1])

    def _touch_rvi(self, config_directory: Path) -> None:
        for result in self._task_results:
            if isinstance(result, RVICollector):
                result._touch(config_directory)

    def _record(
        self, level: AnyLevel, record_directory: Path, uids: tuple[str, ...]
    ) -> None:
        # only final results
        with (
            record_directory / getattr(cf, f"_{level.upper()}_RECORDS_FILENAME")
        ).open("rt") as fp:
            reader = csv.reader(fp, dialect="excel")

            headers = ["#"] + next(reader)
            records = list(reader)

        if level == "job":
            self._objective_idxs = ()
            self._constraint_idxs = ()
            self._to_objectives = ()
            self._to_constraints = ()

        for idx, uid in enumerate(uids):
            assert uid == records[idx][0]

            records[idx] = [str(idx)] + records[idx]
            for result in getattr(self, f"_{level}_results"):
                if not result._is_final:
                    continue

                with (record_directory / uid / result._filename).open("rt") as fp:
                    reader = csv.reader(fp, dialect="excel")

                    if idx:
                        next(reader)
                    else:  # do this only once (when idx is 0)
                        begin_count = len(headers)
                        headers += next(reader)[1:]
                        end_count = len(headers)

                        if level == "job":
                            if result._kind == "objective":
                                self._objective_idxs += tuple(
                                    range(begin_count, end_count)
                                )
                                self._to_objectives += (result._to_objective,) * (
                                    end_count - begin_count
                                )
                            elif result._kind == "constraint":
                                self._constraint_idxs += tuple(
                                    range(begin_count, end_count)
                                )
                                self._to_constraints += (result._to_constraint,) * (
                                    end_count - begin_count
                                )

                    records[idx] += next(reader)[1:]

                    if __debug__:
                        try:
                            next(reader)
                        except StopIteration:
                            pass
                        else:
                            warn(
                                f"multiple result lines found in '{result._filename}', only the first collected."
                            )

        with (
            record_directory / getattr(cf, f"_{level.upper()}_RECORDS_FILENAME")
        ).open("wt") as fp:
            writer = csv.writer(fp, dialect="excel")

            writer.writerow(headers)
            writer.writerows(records)

    @_LoggerManager(cwd_index=1)
    def _collect_task(self, task_directory: Path) -> None:
        for result in self._task_results:
            result._collect(task_directory)

    @_LoggerManager(cwd_index=1)
    def _collect_job(self, job_directory: Path, task_uids: AnyUIDs) -> None:
        for task_uid in task_uids:
            self._collect_task(job_directory / task_uid)

            _log(job_directory, f"collected {task_uid}")

        self._record("task", job_directory, task_uids)

        _log(job_directory, "recorded final results")

        for result in self._job_results:
            result._collect(job_directory)

    @_LoggerManager(cwd_index=1)
    def _collect_batch(self, batch_directory: Path, jobs: tuple[AnyJob, ...]) -> None:
        with _Parallel(
            cf._config["n.processes"],
            initializer=cf._update_config,
            initargs=(cf._config,),
        ) as p:
            it = p.starmap_(
                self._collect_job,
                (
                    (
                        batch_directory / job_uid,
                        tuple(task_uid for task_uid, _ in tasks),
                    )
                    for job_uid, tasks in jobs
                ),
            )

            for (job_uid, _), _ in zip(jobs, it):
                _log(batch_directory, f"collected {job_uid}")

        self._record("job", batch_directory, tuple(job_uid for job_uid, _ in jobs))

        _log(batch_directory, "recorded final results")

    @_LoggerManager(cwd_index=1)
    def _clean_task(self, task_directory: Path) -> None:
        for path in task_directory.glob("*"):
            for pattern in self._clean_patterns:
                if path.match(pattern) and path.is_file():
                    path.unlink()  # NOTE: missing is handled by the is_file check

                    _log(task_directory, f"deleted {path.relative_to(task_directory)}")
                    break

    @_LoggerManager(cwd_index=1)
    def _clean_batch(self, batch_directory: Path, jobs: tuple[AnyJob, ...]) -> None:
        with _Parallel(
            cf._config["n.processes"],
            initializer=cf._update_config,
            initargs=(cf._config,),
        ) as p:
            pairs = tuple(
                (job_uid, task_uid) for job_uid, tasks in jobs for task_uid, _ in tasks
            )
            it = p.map_(
                self._clean_task,
                (batch_directory / job_uid / task_uid for job_uid, task_uid in pairs),
            )

            for pair, _ in zip(pairs, it):
                _log(batch_directory, f"cleaned {'-'.join(pair)}")

    @cache
    def _recorded_batch(self, batch_directory: Path) -> tuple[list[str], ...]:
        with (batch_directory / cf._JOB_RECORDS_FILENAME).open("rt") as fp:
            reader = csv.reader(fp, dialect="excel")

            next(reader)
            return tuple(reader)

    def _recorded_objectives(self, batch_directory: Path) -> AnyBatchResults:
        return tuple(
            tuple(
                func(float(job_vals[idx]))
                for idx, func in zip(self._objective_idxs, self._to_objectives)
            )
            for job_vals in self._recorded_batch(batch_directory)
        )

    def _recorded_constraints(self, batch_directory: Path) -> AnyBatchResults:
        return tuple(
            tuple(
                func(float(job_vals[idx]))
                for idx, func in zip(self._constraint_idxs, self._to_constraints)
            )
            for job_vals in self._recorded_batch(batch_directory)
        )