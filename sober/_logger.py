import logging
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager, ContextDecorator
from functools import reduce, wraps
from inspect import currentframe
from pathlib import Path
from platform import node
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    ParamSpec,
    Self,
    TypeVar,
    get_args,
)

import sober.config as cf
from sober._typing import AnyCmdArgs, AnyLevel, SubprocessResult

HOST_STEM = node().split(".")[0]


def _logger_identifier(cwd: Path) -> str:
    """returns a unique logger identifier
    currently just using the full path to cwd"""

    return str(cwd)


class _Filter(logging.Filter):
    """indents stdout/stderr before formatting"""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno == logging.DEBUG:
            # indent each line and remove the last \n if present
            record.msg = "\t" + "\n\t".join(record.msg.splitlines())

            # if empty message, skip logging
            # check this only for DEBUG messages
            # as DEBUG messages are controlled by EnergyPlus or users
            # INFO messages, controlled by sober, should go to fallback
            # even if they are empty
            if not record.msg.strip():
                return False

        # fallback
        return super().filter(record)


class _Formatter(logging.Formatter):
    """enables logging detailed stdout/stderr from called programmes
    e.g. simulation progress from EnergyPlus
         prints in user Python scripts
    this is realised by reiniting format as per logging level
    currently DEBUG means stdout/stderr
              INFO means high-level progress"""

    _FMT_DEFAULT: Final = (
        f"%(asctime)s {HOST_STEM} %(caller_name)s[%(process)d]: %(message)s"
    )
    _FMT_DEBUG: Final = "%(message)s"

    def __init__(self, fmt: str = _FMT_DEFAULT) -> None:
        super().__init__(fmt, datefmt="%c", style="%")

    def format(self, record: logging.LogRecord) -> str:
        assert record.levelno in (logging.DEBUG, logging.INFO)

        if record.levelno == logging.DEBUG:
            # needs to use explicit _Formatter, see python/mypy#13173
            _Formatter.__init__(self, self._FMT_DEBUG)
            fmted = super().format(record)
            _Formatter.__init__(self, self._FMT_DEFAULT)
        else:
            fmted = super().format(record)

        return fmted


##############################  module typing  ##############################
_P = ParamSpec("_P")
_R = TypeVar("_R", covariant=True)
#############################################################################


class _LoggerManager(AbstractContextManager, ContextDecorator):
    """manages the logger at each level for each action
    and dies upon each completion
    each directory/log file has their own logger
    differentiated by the logger name"""

    _cwd_index: int
    _is_first: bool
    _name: str
    _level: AnyLevel
    _log_file: Path
    _logger: logging.Logger

    __slots__ = ("_cwd_index", "_is_first", "_name", "_level", "_log_file", "_logger")

    if TYPE_CHECKING:
        _recreate_cm: Callable

    def __init__(self, cwd_index: int, is_first: bool = False) -> None:
        self._cwd_index = cwd_index
        self._is_first = is_first

    def __call__(self, func: Callable[_P, _R]) -> Callable[_P, _R]:  # type: ignore[override]
        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            # get the cwd of the decorated func
            cwd = args[self._cwd_index]
            assert isinstance(
                cwd, Path
            ), f"the {self._cwd_index}th argument is no Path."
            # mkdir for all level folders happens here currently
            # this may not make logical sense (mkdir in logging)
            # will look to move this to main modules later
            cwd.mkdir(parents=True, exist_ok=True)

            # get the logger identifier
            self._name = _logger_identifier(cwd)

            # get the level from the func name
            # the func name should follow the pattern of _{action}_{level}
            level = func.__code__.co_name.split("_")[-1]
            assert level in get_args(
                AnyLevel
            ), f"the func name pattern is not recognised: {func.__code__.co_name}."
            self._level = level  # type:ignore[assignment] # python/mypy#12535, python/mypy#15106

            # set the log filename
            self._log_file = cwd / f"{self._level}.log"
            # delete the previous log file at the first call
            if self._is_first:
                self._log_file.unlink(missing_ok=True)

            with self._recreate_cm():
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self) -> Self:
        # create a logger
        self.logger = logging.getLogger(self._name)
        self.logger.setLevel(logging.DEBUG)

        # create a file handler
        fh = logging.FileHandler(self._log_file, "at")
        fh.setLevel(logging.DEBUG)
        fh.addFilter(_Filter())
        fh.setFormatter(_Formatter())
        self.logger.addHandler(fh)

        # create a stream handler at the highest level
        # the highest level for parametrics is batch
        #                   for optimisation is epoch
        if (self._level == "batch" and not cf._has_batches) or (
            self._level == "epoch" and cf._has_batches
        ):
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.DEBUG)
            sh.addFilter(_Filter())
            sh.setFormatter(_Formatter())
            self.logger.addHandler(sh)
        return self

    def __exit__(self, *args) -> None:
        # manually delete all handlers
        self.logger.handlers.clear()

        # shutdown logging
        logging.shutdown()


class _SubprocessLogger(AbstractContextManager):
    """facilitates retrieving stdout/stderr from a subprocess"""

    _logger: logging.LoggerAdapter
    _cmd: str
    _result: SubprocessResult

    __slots__ = ("_logger", "_cmd", "_result")

    def __init__(self, logger: logging.LoggerAdapter, cmd_args: AnyCmdArgs) -> None:
        self._logger = logger
        self._cmd = " ".join(str(cmd_arg) for cmd_arg in cmd_args)

    def __enter__(self) -> Self:
        self._logger.info(f"started '{self._cmd}'")
        return self

    def __exit__(self, *args) -> None:
        result = self._result
        self._logger.debug(result.stdout.strip("\n"))  # stderr was merged into stdout
        self._logger.info(f"completed with exit code {result.returncode}")


def _rgetattr(obj: object, names: tuple[str, ...]) -> Any:
    """a recursive getattr"""
    return reduce(getattr, names, obj)


def _log(
    cwd: Path, msg: str = "", caller_depth: Literal[0, 1] = 0, cmd_args: AnyCmdArgs = ()
) -> _SubprocessLogger:
    """transfers the log message
    inside each function with a managed logger (decorated by _LoggerManager)
    caller_depth is either 0 or 1, 0 for calling that passes in message directly
                                   1 for calling along with a subprocess"""

    # get the logger identifier
    name = _logger_identifier(cwd)
    assert name in logging.Logger.manager.loggerDict, f"unmanaged logger: {name}."

    # get the name of the function that calls this _log function
    # caller_depth + 1, as this _log function always adds one more depth
    caller_name = _rgetattr(
        currentframe(), ("f_back",) * (caller_depth + 1) + ("f_code", "co_name")
    )

    # add the caller name to the contextual info of the logger
    logger = logging.LoggerAdapter(
        logging.getLogger(name), extra={"caller_name": caller_name}
    )

    # log the message if not empty
    # this should happen when this _log function is called directly
    # rather than as a context manager
    if msg:
        logger.info(msg)

    # this is useful only when this _log function is called as a context manager
    # as the class name suggests, the only use case is for subprocess
    # where the function that calls a subprocess is more meaningful
    return _SubprocessLogger(logger, cmd_args)
