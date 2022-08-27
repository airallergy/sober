import logging
from pathlib import Path
from platform import node
from functools import wraps
from collections.abc import Callable
from contextlib import ContextDecorator, AbstractContextManager
from typing import TypeVar, ClassVar, ParamSpec, TypeAlias, SupportsIndex, final

from ._typing import AnyCmdArgs, SubprocessRes

_P = ParamSpec("_P")
_R = TypeVar("_R")

_F: TypeAlias = Callable[_P, _R]  # type: ignore[misc] # python/mypy#11855

HOST_STEM = node().split(".")[0]


def _cwd_to_logger_name(cwd: Path) -> str:
    return str(cwd)


class _Filter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno == logging.DEBUG:
            record.msg = "\t" + "\n\t".join(record.msg.splitlines())
        if not record.msg.strip():
            return False
        return super().filter(record)


@final  # NOTE: to consciously call __init__, see python/mypy#13173
class _Formatter(logging.Formatter):
    _FMT_DEFAULT: ClassVar[str] = f"%(asctime)s {HOST_STEM}: %(message)s"
    _FMT_DEBUG: ClassVar[str] = "%(message)s"

    def __init__(self, fmt: str = _FMT_DEFAULT) -> None:
        super().__init__(fmt, datefmt="%c", style="%")

    def format(self, record: logging.LogRecord) -> str:
        assert record.levelno != logging.NOTSET

        if record.levelno == logging.DEBUG:
            self.__init__(self._FMT_DEBUG)  # type: ignore[misc]
            fmted = super().format(record)
            self.__init__(self._FMT_DEFAULT)  # type: ignore[misc]
        else:
            fmted = super().format(record)

        return fmted


class _LoggerManager(AbstractContextManager, ContextDecorator):
    _cwd_index: SupportsIndex
    _is_first: bool
    _name: str
    _log_file: Path
    _logger: logging.Logger

    def __init__(self, cwd_index: SupportsIndex, is_first: bool = False) -> None:
        self._cwd_index = cwd_index
        self._is_first = is_first

    def __call__(self, f: _F) -> _F:
        @wraps(f)
        def wrapper(*args, **kwargs) -> _R:
            cwd: Path = args[self._cwd_index]
            cwd.mkdir(parents=True, exist_ok=True)

            self._name = _cwd_to_logger_name(cwd)
            self._log_file = cwd / "console.log"
            if self._is_first:
                self._log_file.unlink(missing_ok=True)

            with self._recreate_cm():  # type: ignore[attr-defined] # NOTE: why?
                return f(*args, **kwargs)

        return wrapper

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
        return self

    def __exit__(self, *args) -> None:
        self.logger.handlers.clear()
        logging.shutdown()


class _SubprocessLogger(AbstractContextManager):
    _logger: logging.Logger
    _cmd: str
    res: SubprocessRes

    def __init__(self, logger: logging.Logger, cmd_args: AnyCmdArgs) -> None:
        self._logger = logger
        self._cmd = " ".join(str(cmd_arg) for cmd_arg in cmd_args)

    def __enter__(self) -> "_SubprocessLogger":  # TODO: use typing.Self after 3.11
        self._logger.info(" ".join(("running", f"'{self._cmd}'")))
        return self

    def __exit__(self, *args) -> None:
        self._logger.debug(self.res.stdout.strip("\n"))
        self._logger.info(f"completed with exit code {self.res.returncode}")


def _log(cwd: Path, msg: str = "", cmd_args: AnyCmdArgs = ()) -> _SubprocessLogger:
    name = _cwd_to_logger_name(cwd)
    assert name in logging.Logger.manager.loggerDict, f"unmanaged logger: {name}."

    logger = logging.getLogger(name)
    if msg:
        logger.info(msg)

    return _SubprocessLogger(logger, cmd_args)
