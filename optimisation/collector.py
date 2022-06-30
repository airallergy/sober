from pathlib import Path, PurePath
from abc import ABC, abstractmethod
from typing import Literal, TypeAlias

from ._simulator import _run_readvars

AnyLevel: TypeAlias = Literal["task", "job", "model"]

#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Collector(ABC):
    _csv_filename: PurePath
    _level: AnyLevel

    @abstractmethod
    def __init__(self, csv_filename: str, level: AnyLevel) -> None:
        self._csv_filename = PurePath(csv_filename + ".csv")
        self._level = level

    @abstractmethod
    def _collect(self, cwd: Path) -> None:
        ...


#############################################################################
#######                       COLLECTOR CLASSES                       #######
#############################################################################
class RVICollector(_Collector):
    _output_name: str
    _output_type: str
    _rvi_file: Path
    _frequency: str

    def __init__(
        self,
        output_name: str,
        output_type: str,
        csv_filename: str,
        frequency: str = "",
    ) -> None:
        self._output_name = output_name
        self._output_type = output_type.lower()
        self._frequency = frequency

        super().__init__(csv_filename, "task")

    def _touch(self, config_directory: Path) -> None:
        self._rvi_file = (
            config_directory / f"{self._output_name.replace(' ', '_').lower()}.rvi"
        )
        suffixes = {"variable": "eso", "meter": "mtr"}
        with self._rvi_file.open("wt") as fp:
            fp.write(
                f"eplusout.{suffixes[self._output_type]}\n{self._csv_filename}\n{self._output_name}\n0\n"
            )

    def _collect(self, cwd: Path) -> None:
        _run_readvars(self._rvi_file, cwd, self._frequency)
