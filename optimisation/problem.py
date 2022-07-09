from pathlib import Path
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem as _PymooProblem

from . import config as cf
from ._tools import AnyStrPath
from ._multiplier import _multiply
from ._evaluator import _pymoo_evaluate
from .results import RVICollector, ScriptCollector, _Collector, _ResultsManager
from .parameters import (
    AnyParameter,
    WeatherParameter,
    _ParametersManager,
    _all_int_parameters,
)


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
    _parameters_manager: _ParametersManager[AnyParameter]
    _results_manager: _ResultsManager
    _callback: Callback | None
    _model_directory: Path
    _evaluation_directory: Path
    _config_directory: Path

    def __init__(
        self,
        model_file: AnyStrPath,
        weather: WeatherParameter,
        parameters: Iterable[AnyParameter],
        results: Iterable[_Collector] = (),
        callback: Callback | None = None,
        evaluation_directory: AnyStrPath | None = None,
        n_processes: int | None = None,
        python_exec: AnyStrPath | None = None,
    ) -> None:
        self._model_file = Path(model_file).resolve(strict=True)
        self._parameters_manager = _ParametersManager(
            weather, parameters, self._model_file
        )
        self._results_manager = _ResultsManager(results)
        self._callback = callback
        self._model_directory = self._model_file.parent
        self._evaluation_directory = (
            self._model_directory / "evaluation"
            if evaluation_directory is None
            else Path(evaluation_directory)
        )
        self._config_directory = self._model_directory / f".{__package__}"

        self._prepare(n_processes, python_exec)

    def _mkdir(self) -> None:
        self._evaluation_directory.mkdir(exist_ok=True)
        self._config_directory.mkdir(exist_ok=True)

    def _check_config(self) -> None:
        cf._check_config(
            self._parameters_manager._model_type,
            any(isinstance(result, RVICollector) for result in self._results_manager),
            set(
                result._language
                for result in self._results_manager
                if isinstance(result, ScriptCollector)
            ),
        )

    def _prepare(self, n_processes: int | None, python_exec: AnyStrPath | None) -> None:
        self._mkdir()
        cf.config_multiprocessing(n_processes)
        cf.config_script(python_exec)
        self._results_manager._touch_rvi(self._config_directory)
        self._check_config()

    def _to_pymoo(self) -> PymooProblem:
        if len(self._results_manager._objectives) == 0:
            raise ValueError("Optimisation needs at least one objective")

        return PymooProblem(
            len(self._parameters_manager),
            len(self._results_manager._objectives),
            len(self._results_manager._constraints),
            np.fromiter(
                (parameter._low for parameter in self._parameters_manager),
                dtype=np.float_,
            ),
            np.fromiter(
                (parameter._high for parameter in self._parameters_manager),
                dtype=np.float_,
            ),
            self._callback,
        )

    def run_brute_force(self) -> None:
        if _all_int_parameters(self._parameters_manager):
            _multiply(
                self._parameters_manager,
                self._results_manager,
                self._evaluation_directory,
            )
        else:
            raise ValueError("With continous parameters cannot run brute force.")
