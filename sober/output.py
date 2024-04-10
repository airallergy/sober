import enum
import itertools as it
import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypeAlias, get_args

import sober.config as cf
from sober._simulator import _run_readvars
from sober._tools import _rectified_str_iterable, _run, _uuid
from sober._typing import AnyCoreLevel, AnyStrPath


##############################  module typing  ##############################
@enum.unique
class _Direction(enum.IntEnum):
    MINIMISE = 1
    MAXIMISE = -1


@enum.unique
class _EPOutputType(enum.StrEnum):
    VARIABLE = "eplusout.eso"
    METER = "eplusout.mtr"


_AnyDirection: TypeAlias = Literal["minimise", "maximise"]
_AnyBounds: TypeAlias = tuple[None, float] | tuple[float, None] | tuple[float, float]
_AnyEPOutputType: TypeAlias = Literal["variable", "meter"]

# TODO: below goes to tests when made
assert [item.upper() for item in get_args(_AnyDirection)] == _Direction._member_names_
assert [
    item.upper() for item in get_args(_AnyEPOutputType)
] == _EPOutputType._member_names_
#############################################################################


#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Collector(ABC):
    """an abstract base class for output collector"""

    _filename: str
    _level: AnyCoreLevel
    _objectives: tuple[str, ...]
    _constraints: tuple[str, ...]
    _direction: _Direction
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
        level: AnyCoreLevel,
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
        self._direction = _Direction[direction.upper()]
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
                f"a final output needs to be a csv file: {self._filename}."
            )

    @abstractmethod
    def _collect(self, cwd: Path) -> None: ...

    def _to_objective(self, value: float) -> float:
        # convert each objective to minimise
        return value * self._direction.value

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
    """collects rvi outputs"""

    # TODO: consider switching to/adding EP native csv once NREL/EnergyPlus#9395

    _ep_output_names: tuple[str, ...]
    _ep_output_type: _EPOutputType
    _keys: tuple[str, ...]
    _frequency: str
    _rvi_file: Path

    __slots__ = (
        "_ep_output_names",
        "_ep_output_type",
        "_keys",
        "_frequency",
        "_rvi_file",
    )

    def __init__(
        self,
        ep_output_names: str | Iterable[str],
        ep_output_type: _AnyEPOutputType,
        filename: str,
        /,
        keys: str | Iterable[str] = (),
        frequency: str = "",
        *,
        objectives: str | Iterable[str] = (),
        constraints: str | Iterable[str] = (),
        direction: _AnyDirection = "minimise",
        bounds: _AnyBounds = (None, 0),
        is_final: bool = True,
    ) -> None:
        self._ep_output_names = _rectified_str_iterable(ep_output_names)
        self._ep_output_type = _EPOutputType[ep_output_type.upper()]
        self._keys = _rectified_str_iterable(keys)
        self._frequency = frequency

        super().__init__(
            filename, "task", objectives, constraints, direction, bounds, is_final
        )

    def _check_args(self) -> None:
        super()._check_args()

        if self._filename.split(".")[-1] != "csv":
            raise ValueError(
                f"an RVICollector output needs to be a csv file: {self._filename}."
            )

        if self._ep_output_type is _EPOutputType.METER and self._keys:
            raise ValueError("meter variables do not accept keys.")

    def _touch(self, config_dir: Path) -> None:
        rvi_str = f"{self._ep_output_type.value}\n{self._filename}\n"
        if self._keys:
            rvi_str += "\n".join(
                f"{key},{name}"
                for key, name in it.product(self._keys, self._ep_output_names)
            )
        else:
            rvi_str += "\n".join(self._ep_output_names)

        rvi_str += "\n0\n"

        rvi_filestem = _uuid(self.__class__.__name__, *rvi_str.splitlines())
        self._rvi_file = config_dir / (rvi_filestem + ".rvi")
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
    """collects script outputs"""

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
        level: AnyCoreLevel = "task",
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
        language_exec = cf._config["exec." + self._language]  # type: ignore[literal-required]  # python/mypy#12554

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
    """copies task final outputs as job final outputs
    NOTE: this is for handling non-noisy cases only"""

    __slots__ = ()

    def __init__(
        self,
        filename: str,
        objectives: tuple[str, ...],
        constraints: tuple[str, ...],
        direction: _Direction,
        bounds: _AnyBounds,
    ) -> None:
        # overwrite _Collector's __init__, as all args have been rectified
        self._filename = filename
        self._level = "job"
        self._objectives = objectives
        self._constraints = constraints
        self._direction = direction
        self._bounds = bounds
        self._is_final = True
        self._is_copied = False

    def _collect(self, cwd: Path) -> None:
        shutil.copyfile(cwd / "T0" / self._filename, cwd / self._filename)
