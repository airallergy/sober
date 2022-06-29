#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
from pathlib import Path
from abc import ABC, abstractmethod

from . import config as cf
from ._simulator import _run_readvars


class _Collector(ABC):
    _csv_name: str

    @abstractmethod
    def __init__(self, csv_name: str) -> None:
        self._csv_name = csv_name

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
        csv_name: str,
        frequency: str = "",
    ) -> None:
        self._output_name = output_name
        self._output_type = output_type.lower()
        self._frequency = frequency

        super().__init__(csv_name)

    def _touch(self) -> None:
        self._rvi_file = (
            cf._config_directory / f"{self._output_name.replace(' ', '_').lower()}.rvi"
        )
        suffixes = {"variable": "eso", "meter": "mtr"}
        with self._rvi_file.open("wt") as fp:
            fp.write(
                f"eplusout.{suffixes[self._output_type]}\n{self._csv_name}.csv\n{self._output_name}\n0\n"
            )

    def _collect(self, cwd: Path) -> None:
        _run_readvars(self._rvi_file, cwd, self._frequency)
