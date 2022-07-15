from pathlib import Path
from warnings import warn
from functools import cache
from itertools import chain
from abc import ABC, abstractmethod
from typing import Literal, TypeAlias
from collections.abc import Iterable, Iterator

from typing_extensions import Unpack  # TODO: remove Unpack after 3.11

from . import config as cf
from ._simulator import _run_readvars
from ._tools import AnyCli, AnyStrPath, _run, _Parallel

AnyLevel: TypeAlias = Literal["task", "job"]
AnyKind: TypeAlias = Literal["objective", "constraint", "extra"]
AnyOutputType: TypeAlias = Literal["variable", "meter"]

#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Collector(ABC):
    _csv_filename: str
    _level: AnyLevel
    _kind: AnyKind
    _is_final: bool

    @abstractmethod
    def __init__(
        self, csv_name: str, level: AnyLevel, kind: AnyKind, is_final: bool
    ) -> None:
        self._csv_filename = csv_name + ".csv"
        self._level = level
        self._kind = kind
        self._is_final = is_final

        self._check_args()

    def _check_args(self) -> None:
        if self._kind in ("objective", "constraint"):
            assert (
                self._level == "job"
            ), f"an '{self._kind}' result needs to be at the 'job' level."
            assert (
                self._is_final == True
            ), f"an '{self._kind}' result needs to be final."

    @abstractmethod
    def _collect(self, cwd: Path) -> None:
        ...


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
        csv_name: str,
        level: AnyLevel,
        kind: AnyKind,
        is_final: bool = True,
        keys: Iterable[str] = (),
        frequency: str = "",
    ) -> None:
        self._output_name = output_name
        self._output_type = output_type
        self._keys = tuple(keys)
        self._frequency = frequency

        super().__init__(csv_name, level, kind, is_final)

    def _touch(self, config_directory: Path) -> None:
        self._rvi_file = (
            config_directory / f"{self._output_name.replace(' ', '_').lower()}.rvi"
        )

        suffixes = {"variable": "eso", "meter": "mtr"}
        joined_rvi_lines = (
            f"eplusout.{suffixes[self._output_type]}\n{self._csv_filename}\n"
        )
        match self._keys:
            case ():
                joined_rvi_lines += self._output_name
            case _:
                joined_rvi_lines += "\n".join(
                    f"{key},{self._output_name}" for key in self._keys
                )
        joined_rvi_lines += "\n0\n"

        with self._rvi_file.open("wt") as fp:
            fp.write(joined_rvi_lines)

    def _collect(self, cwd: Path) -> None:
        _run_readvars(self._rvi_file, cwd, self._frequency)


class ScriptCollector(_Collector):
    _script_file: Path
    _language: cf.AnyLanguage
    _script_args: AnyCli

    def __init__(
        self,
        script_file: AnyStrPath,
        language: cf.AnyLanguage,
        csv_name: str,
        level: AnyLevel,
        kind: AnyKind,
        is_final: bool = True,
        *script_args: Unpack[AnyCli],  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
    ) -> None:
        self._script_file = Path(script_file)
        self._language = language
        self._script_args = script_args
        super().__init__(csv_name, level, kind, is_final)

    def _collect(self, cwd: Path) -> None:
        commands: AnyCli = (
            cf._config["exec.python"],
            self._script_file,
            cwd,
            self._csv_filename,
            *self._script_args,
        )

        _run(commands, cwd)


#############################################################################
#######                    RESULTS MANAGER CLASSES                    #######
#############################################################################
class _ResultsManager:
    _task_results: tuple[_Collector, ...]
    _job_results: tuple[_Collector, ...]
    _objectives: tuple[_Collector, ...]
    _constraints: tuple[_Collector, ...]
    _extras: tuple[_Collector, ...]
    _objective_idxs: tuple[int, ...]
    _constraint_idxs: tuple[int, ...]

    def __init__(self, results: Iterable[_Collector]) -> None:
        results = tuple(results)
        self._task_results = tuple(
            result for result in results if result._level == "task"
        )
        self._job_results = tuple(
            result for result in results if result._level == "job"
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

    def _record_final(self, level: AnyLevel, cwd: Path, uids: tuple[str, ...]) -> None:
        header_line = f"#,{level.capitalize()}UID"
        joined_val_lines = ""
        if level == "job":
            count = 0
            self._objective_idxs = ()
            self._constraint_idxs = ()

        for idx, uid in enumerate(uids):
            joined_val_lines += f"{idx},{uid}"
            for result in getattr(self, f"_{level}_results"):
                if not result._is_final:
                    continue

                with (cwd / uid / result._csv_filename).open("rt") as fp:
                    line = next(fp)
                    if idx == 0:
                        header_line += "," + line.rstrip().split(",", 1)[-1]

                        if level == "job":
                            if result._kind == "objective":
                                self._objective_idxs += tuple(
                                    range(count, count := count + line.count(","))
                                )
                            elif result._kind == "constraint":
                                self._constraint_idxs += tuple(
                                    range(count, count := count + line.count(","))
                                )

                    line = next(fp)
                    joined_val_lines += "," + line.rstrip().split(",", 1)[-1]

                    if __debug__:
                        try:
                            next(fp)
                        except StopIteration:
                            pass
                        else:
                            warn(
                                f"multiple result lines found in '{result._csv_filename}', only the first collected."
                            )
            joined_val_lines += "\n"

        with (cwd / f"{level}_records.csv").open("wt") as fp:
            fp.write(header_line + "\n" + joined_val_lines)

    def _collect_task(self, task_directory: Path) -> None:
        for result in self._task_results:
            result._collect(task_directory)

    def _collect_job(self, job_directory: Path, task_uids: cf.AnyUIDs) -> None:
        for task_uid in task_uids:
            self._collect_task(job_directory / task_uid)

        self._record_final("task", job_directory, task_uids)

        for result in self._job_results:
            result._collect(job_directory)

    def _collect_batch(
        self, batch_directory: Path, jobs: tuple[cf.AnyJob, ...]
    ) -> None:
        with _Parallel(
            cf._config["n.processes"],
            initializer=cf._update_config,
            initargs=(cf._config,),
        ) as parallel:
            parallel.starmap(
                self._collect_job,
                (
                    (
                        batch_directory / job_uid,
                        tuple(task_uid for task_uid, _ in tasks),
                    )
                    for job_uid, tasks in jobs
                ),
            )

        self._record_final(
            "job", batch_directory, tuple(job_uid for job_uid, _ in jobs)
        )

    @cache
    def _recorded_batch(self, batch_directory: Path) -> tuple[tuple[float, ...], ...]:
        with (batch_directory / "job_records.csv").open("rt") as fp:
            next(fp)
            val_lines = fp.read().splitlines()
        return tuple(tuple(map(float, line.split(",")[2:])) for line in val_lines)

    def _recorded_objectives(
        self, batch_directory: Path
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(job_vals[idx] for idx in self._objective_idxs)
            for job_vals in self._recorded_batch(batch_directory)
        )

    def _recorded_constraints(
        self, batch_directory: Path
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(job_vals[idx] for idx in self._constraint_idxs)
            for job_vals in self._recorded_batch(batch_directory)
        )
