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
from .collector import RVICollector, _Collector
from .parameters import WeatherParameter, AnyModelParameter, AnyIntModelParameter


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
    _outputs_directory: Path
    _model_type: str
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
    ) -> None:
        self._model_file = Path(model_file)
        self._weather = weather
        self._parameters = tuple(parameters)
        self._objectives = tuple(objectives)
        self._constraints = tuple(constraints)
        self._extra_outputs = tuple(extra_outputs)
        self._callback = callback
        self._outputs_directory = (
            self._model_file.parent / "outputs"
            if outputs_directory is None
            else Path(outputs_directory)
        )

        cf._config_directory = self._model_file.parent / f"{__package__}.config"
        cf._config_directory.mkdir(exist_ok=True)

        self._model_type = self._model_file.suffix
        if self._model_type not in (".idf", ".imf"):
            raise NotImplementedError(f"a '{self._model_type}' model is not supported.")

    def _tag_model(self) -> None:
        macros, regulars = _split_model(self._model_file)
        if hasattr(cf, "_config"):
            idf = openidf(StringIO(regulars), cf._config["schema.energyplus"])
            cf._check_config(
                self._model_type,
                chain(self._objectives, self._constraints, self._extra_outputs),
            )
        else:
            idf = openidf(StringIO(regulars))
            cf.config_energyplus(idf.idfobjects["Version"][0]["Version_Identifier"])

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
                output._touch()

    def _prepare(self) -> None:
        self._tag_model()
        self._outputs_directory.mkdir(exist_ok=True)
        self._touch_rvi()

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

        self._prepare()

        _multiply(
            self._tagged_model,
            self._weather,
            self._parameters,  # type: ignore[arg-type] # python/mypy/#7853
            self._objectives + self._constraints + self._extra_outputs,
            self._outputs_directory,
            self._model_type,
        )
