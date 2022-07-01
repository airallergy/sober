from itertools import chain
from pathlib import Path, PurePath
from abc import ABC, abstractmethod
from typing import Literal, TypeAlias
from collections.abc import Iterable, Iterator

from typing_extensions import Unpack  # TODO: remove Unpack after 3.11

from . import config as cf
from ._simulator import _run_readvars
from ._tools import AnyCli, AnyStrPath, _run

AnyResultLevel: TypeAlias = Literal["task", "job", "batch"]
AnyResultKind: TypeAlias = Literal["objective", "constraint", "extra"]
AnyOutputType: TypeAlias = Literal["variable", "meter"]

#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Collector(ABC):
    _csv_filename: PurePath
    _level: AnyResultLevel
    _kind: AnyResultKind

    @abstractmethod
    def __init__(
        self, csv_name: str, level: AnyResultLevel, kind: AnyResultKind
    ) -> None:
        self._csv_filename = PurePath(csv_name + ".csv")
        self._level = level
        self._kind = kind

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
        kind: AnyResultKind,
        keys: Iterable[str] = (),
        frequency: str = "",
    ) -> None:
        self._output_name = output_name
        self._output_type = output_type
        self._keys = tuple(keys)
        self._frequency = frequency

        super().__init__(csv_name, "task", kind)

    def _touch(self, config_directory: Path) -> None:
        self._rvi_file = (
            config_directory / f"{self._output_name.replace(' ', '_').lower()}.rvi"
        )

        suffixes = {"variable": "eso", "meter": "mtr"}
        rvi_lines = f"eplusout.{suffixes[self._output_type]}\n{self._csv_filename}\n"
        match self._keys:
            case ():
                rvi_lines += self._output_name
            case _:
                rvi_lines += "\n".join(
                    f"{key},{self._output_name}" for key in self._keys
                )
        rvi_lines += "\n0\n"

        with self._rvi_file.open("wt") as fp:
            fp.write(rvi_lines)

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
        level: AnyResultLevel,
        kind: AnyResultKind,
        *script_args: Unpack[AnyCli],  # type: ignore[misc] # python/mypy#12280 # TODO: Unpack -> * after 3.11
    ) -> None:
        self._script_file = Path(script_file)
        self._language = language
        self._script_args = script_args
        super().__init__(csv_name, level, kind)

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
    _task_collectors: tuple[_Collector, ...]
    _job_collectors: tuple[_Collector, ...]
    _batch_collectors: tuple[_Collector, ...]
    _objectives: tuple[_Collector, ...]
    _constraints: tuple[_Collector, ...]
    _extras: tuple[_Collector, ...]

    def __init__(self, results: Iterable[_Collector]) -> None:
        self._task_collectors = tuple(
            result for result in results if result._level == "task"
        )
        self._job_collectors = tuple(
            result for result in results if result._level == "job"
        )
        self._batch_collectors = tuple(
            result for result in results if result._level == "batch"
        )

    def __iter__(self) -> Iterator[_Collector]:
        for collector in chain(
            self._task_collectors, self._job_collectors, self._batch_collectors
        ):
            yield collector

    def __getattr__(self, name: str) -> tuple[_Collector, ...]:
        # TODO: python/mypy#8203
        if name not in frozenset({"_objectives", "_constraints", "_extras"}):
            raise AttributeError

        return tuple(collector for collector in self if collector._kind == name)
