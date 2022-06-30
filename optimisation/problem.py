from io import StringIO
from pathlib import Path
from itertools import chain
from collections.abc import Iterable

import numpy as np
from eppy import openidf
from numpy.typing import NDArray
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem as _PymooProblem

from . import config as cf
from ._tools import AnyStrPath
from ._multiplier import _multiply
from ._simulator import _split_model
from ._evaluator import _pymoo_evaluate
from .results import RVICollector, ScriptCollector, _Collector
from .parameters import WeatherParameter, AnyModelParameter, AnyIntModelParameter

ModelTypes: frozenset[cf.AnyModelType] = frozenset({".idf", ".imf"})


class PymooProblem(_PymooProblem):
    def __init__(
        self,
        n_var: int,
        n_obj: int,
        n_constr: int,
        xl: NDArray[np.float_],
        xu: NDArray[np.float_],
        callback: Callback | None,
    ) -> None:
        super().__init__(
            n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu, callback=callback
        )

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        _pymoo_evaluate()


#############################################################################
#######                        PROBLEM CLASSES                        #######
#############################################################################
class Problem:
    _model_file: Path
    _weather: WeatherParameter
    _parameters: tuple[AnyModelParameter, ...]
    _objectives: tuple[_Collector, ...]
    _constraints: tuple[_Collector, ...]
    _extra_outputs: tuple[_Collector, ...]
    _callback: Callback | None
    _model_directory: Path
    _outputs_directory: Path
    _config_directory: Path
    _model_type: cf.AnyModelType
    _tagged_model: str

    def __init__(
        self,
        model_file: AnyStrPath,
        weather: WeatherParameter,
        parameters: Iterable[AnyModelParameter],
        objectives: Iterable[_Collector] = (),
        constraints: Iterable[_Collector] = (),
        extra_outputs: Iterable[_Collector] = (),
        callback: Callback | None = None,
        outputs_directory: AnyStrPath | None = None,
        python_exec: AnyStrPath | None = None,
    ) -> None:
        self._model_file = Path(model_file).resolve(strict=True)
        self._weather = weather
        self._parameters = tuple(parameters)
        self._objectives = tuple(objectives)
        self._constraints = tuple(constraints)
        self._extra_outputs = tuple(extra_outputs)
        self._callback = callback
        self._model_directory = self._model_file.parent
        self._outputs_directory = (
            self._model_directory / "outputs"
            if outputs_directory is None
            else Path(outputs_directory)
        )
        self._config_directory = self._model_directory / f"{__package__}.config"

        suffix = self._model_file.suffix
        if suffix not in ModelTypes:
            raise NotImplementedError(f"a '{self._model_type}' model is not supported.")

        self._model_type = suffix  # type: ignore[assignment] # python/mypy#12535

        self._prepare(python_exec)

    def _mkdir(self) -> None:
        self._outputs_directory.mkdir(exist_ok=True)
        self._config_directory.mkdir(exist_ok=True)

    def _tag_model(self) -> None:
        macros, regulars = _split_model(self._model_file)
        if hasattr(cf, "_config"):
            idf = openidf(StringIO(regulars), cf._config["schema.energyplus"])
        else:
            idf = openidf(StringIO(regulars))
            cf.config_energyplus(
                version=idf.idfobjects["Version"][0]["Version_Identifier"]
            )

        for parameter in self._parameters:
            tagger = parameter._tagger
            match tagger._LOCATION:
                case "regular":
                    idf = tagger._tagged(idf)
                case "macro":
                    macros = tagger._tagged(macros)

        self._tagged_model = macros + idf.idfstr()

    def _touch_rvi(self) -> None:
        for output in chain(self._objectives, self._constraints, self._extra_outputs):
            if isinstance(output, RVICollector):
                output._touch(self._config_directory)

    def _check_config(self) -> None:
        outputs_iter = chain(self._objectives, self._constraints, self._extra_outputs)

        cf._check_config(
            self._model_type,
            any(isinstance(output, RVICollector) for output in outputs_iter),
            set(
                output._language
                for output in outputs_iter
                if isinstance(output, ScriptCollector)
            ),
        )

    def _prepare(self, python_exec: AnyStrPath | None = None) -> None:
        self._mkdir()
        self._tag_model()
        cf.config_script(python_exec)
        self._touch_rvi()
        self._check_config()

    def _to_pymoo(self) -> PymooProblem:
        if self._objectives == ():
            raise ValueError("Optimisation needs at least one objective")

        return PymooProblem(
            len(self._parameters),
            len(self._objectives),
            len(self._constraints),
            np.fromiter(
                (parameter._low for parameter in self._parameters), dtype=np.float_
            ),
            np.fromiter(
                (parameter._high for parameter in self._parameters), dtype=np.float_
            ),
            self._callback,
        )

    def run_parametric(self) -> None:
        if not all(
            isinstance(parameter, AnyIntModelParameter) for parameter in self._parameters  # type: ignore[misc, arg-type] # python/mypy#11673
        ):
            raise ValueError("With continous parameters cannot run parametric.")

        _multiply(
            self._tagged_model,
            self._weather,
            self._parameters,  # type: ignore[arg-type] # python/mypy/#7853
            self._objectives + self._constraints + self._extra_outputs,
            self._outputs_directory,
            self._model_type,
        )
