import sys
import logging
from pathlib import Path
from platform import node
from inspect import currentframe
from functools import wraps, reduce
from collections.abc import Callable
from typing import Any, Literal, TypeVar, ClassVar
from contextlib import ContextDecorator, AbstractContextManager

from . import config as cf
from ._typing import AnyCmdArgs, SubprocessRes

# this follows the ContextDecorator signature
# just to make my life easier
_F = TypeVar("_F", bound=Callable[..., Any])


HOST_STEM = node().split(".")[0]


def _logger_identifier(cwd: Path) -> str:
    """returns a unique logger identifier
    currently just using the full path to cwd"""

    return str(cwd)


class _Filter(logging.Filter):
    """indents stdout/stderr before formatting"""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno == logging.DEBUG:
            # indent each line and remove the last \n if presnet
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

    _FMT_DEFAULT: ClassVar[
        str
    ] = f"%(asctime)s {HOST_STEM} %(caller_name)s[%(process)d]: %(message)s"
    _FMT_DEBUG: ClassVar[str] = "%(message)s"

    def __init__(self, fmt: str = _FMT_DEFAULT) -> None:
        super().__init__(fmt, datefmt="%c", style="%")

    def format(self, record: logging.LogRecord) -> str:
        assert record.levelno in (logging.DEBUG, logging.INFO)

        if record.levelno == logging.DEBUG:
            _Formatter.__init__(self, self._FMT_DEBUG)  # python/mypy#13173
            fmted = super().format(record)
            _Formatter.__init__(self, self._FMT_DEFAULT)  # python/mypy#13173
        else:
            fmted = super().format(record)

        return fmted


class _LoggerManager(AbstractContextManager, ContextDecorator):
    """manages the logger at each level for each action
    and dies upon each completion
    each directory/log file has their own logger
    differentiated by the logger name"""

    _cwd_index: int
    _is_first: bool
    _name: str
    _level: Literal["task", "job", "batch", "epoch"]
    _log_file: Path
    _logger: logging.Logger

    def __init__(self, cwd_index: int, is_first: bool = False) -> None:
        self._cwd_index = cwd_index
        self._is_first = is_first

    def __call__(self, f: _F) -> _F:
        @wraps(f)
        def wrapper(*args, **kwargs) -> Any:
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
            level = f.__code__.co_name.split("_")[-1]
            assert level in (
                "task",
                "job",
                "batch",
                "epoch",
            ), f"the func name pattern is not recognised: {f.__code__.co_name}."
            self._level = level  # type:ignore[assignment] # python/mypy#12535

            # set the log filename
            self._log_file = cwd / f"{self._level}.log"
            # delete the previous log file at the first call
            if self._is_first:
                self._log_file.unlink(missing_ok=True)

            with self._recreate_cm():  # type: ignore[attr-defined] # no idea why
                # not entirely sure why
                # but nothing will be logged without this context
                return f(*args, **kwargs)

        return wrapper  # type:ignore[return-value] # python/mypy#1927 says solved, but

    def __enter__(self) -> "_LoggerManager":  # TODO: use typing.Self after 3.11
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
        # the highest level for parametric is batch
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
    res: SubprocessRes

    def __init__(self, logger: logging.LoggerAdapter, cmd_args: AnyCmdArgs) -> None:
        self._logger = logger
        self._cmd = " ".join(str(cmd_arg) for cmd_arg in cmd_args)

    def __enter__(self) -> "_SubprocessLogger":  # TODO: use typing.Self after 3.11
        self._logger.info(f"started '{self._cmd}'")
        return self

    def __exit__(self, *args) -> None:
        self._logger.debug(self.res.stdout.strip())  # stderr was merged into stdout
        self._logger.info(f"completed with exit code {self.res.returncode}")


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
    # TODO: change co_name to co_qualname after 3.11, see python/cpython#88696
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
