from pathlib import Path, PurePath
from abc import ABC, abstractmethod
from typing import Literal, Iterable, TypeAlias

from ._simulator import _run_readvars

AnyLevel: TypeAlias = Literal["task", "job", "model"]
AnyOutputType: TypeAlias = Literal["variable", "meter"]

#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Collector(ABC):
    _csv_filename: PurePath
    _level: AnyLevel

    @abstractmethod
    def __init__(self, csv_name: str, level: AnyLevel) -> None:
        self._csv_filename = PurePath(csv_name + ".csv")
        self._level = level

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
        keys: Iterable[str] = (),
        frequency: str = "",
    ) -> None:
        self._output_name = output_name
        self._output_type = output_type
        self._keys = tuple(keys)
        self._frequency = frequency

        super().__init__(csv_name, "task")

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
