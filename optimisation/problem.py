from math import log10
from pathlib import Path
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from ._logger import _log
from . import config as cf
from ._multiplier import _multiply
from . import _pymoo_namespace as pm
from ._evaluator import _pymoo_evaluate
from ._optimiser import _optimise_epoch
from ._typing import AnyStrPath, AnyCallback, AnyVariationVec
from .results import RVICollector, ScriptCollector, _Collector, _ResultsManager
from .parameters import (
    AnyParameter,
    WeatherParameter,
    _ParametersManager,
    _all_int_parameters,
)


class PymooProblem(pm.Problem):
    _parameters_manager: _ParametersManager
    _results_manager: _ResultsManager
    _evaluation_directory: Path
    _len_batch_count: int

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        n_constr: int,
        xl: NDArray[np.float_],
        xu: NDArray[np.float_],
        callback: AnyCallback,
        parameters_manager: _ParametersManager,
        results_manager: _ResultsManager,
        evaluation_directory: Path,
        expected_max_n_generation: int,
    ) -> None:
        super().__init__(
            n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu, callback=callback
        )
        self._parameters_manager = parameters_manager
        self._results_manager = results_manager
        self._evaluation_directory = evaluation_directory

        self._len_batch_count = int(log10(expected_max_n_generation)) + 1

    def _evaluate(
        self,
        x: tuple[AnyVariationVec, ...],
        out,
        *args,
        algorithm: pm.Algorithm,
        **kwargs,
    ) -> None:
        batch_uid = f"B{algorithm.n_gen - (1 - algorithm.is_initialized):0{self._len_batch_count}}"
        out["F"], out["G"] = _pymoo_evaluate(
            *x,
            parameters_manager=self._parameters_manager,
            results_manager=self._results_manager,
            batch_directory=self._evaluation_directory / batch_uid,
        )

        _log(self._evaluation_directory, f"evaluated {batch_uid}")


#############################################################################
#######                        PROBLEM CLASSES                        #######
#############################################################################
class Problem:
    _parameters_manager: _ParametersManager[AnyParameter]
    _results_manager: _ResultsManager
    _model_directory: Path
    _evaluation_directory: Path
    _config_directory: Path

    def __init__(
        self,
        model_file: AnyStrPath,
        weather: WeatherParameter,
        parameters: Iterable[AnyParameter] = (),
        results: Iterable[_Collector] = (),
        has_templates: bool = False,
        clean_patterns: Iterable[str] = _ResultsManager._DEFAULT_CLEAN_PATTERNS,
        evaluation_directory: AnyStrPath | None = None,
        n_processes: int | None = None,
        python_exec: AnyStrPath | None = None,
    ) -> None:
        model_file = Path(model_file).resolve(strict=True)
        self._parameters_manager = _ParametersManager(
            weather, parameters, model_file, has_templates
        )
        self._results_manager = _ResultsManager(
            results, clean_patterns, self._parameters_manager._has_uncertainties
        )
        self._model_directory = model_file.parent
        self._evaluation_directory = (
            self._model_directory / "evaluation"
            if evaluation_directory is None
            else Path(evaluation_directory)
        )
        self._config_directory = self._model_directory / (
            "." + __package__.split(".")[-1]
        )

        self._prepare(n_processes, python_exec)

    def _mkdir(self) -> None:
        self._evaluation_directory.mkdir(exist_ok=True)
        self._config_directory.mkdir(exist_ok=True)

    def _check_config(self) -> None:
        cf._check_config(
            self._parameters_manager._model_type,
            self._parameters_manager._has_templates,
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

    def _to_pymoo(
        self, callback: AnyCallback, expected_max_n_generation: int
    ) -> PymooProblem:
        if not len(self._results_manager._objectives):
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
            callback,
            self._parameters_manager,
            self._results_manager,
            self._evaluation_directory,
            expected_max_n_generation,
        )

    def run_brute_force(self) -> None:
        cf._has_batches = False

        if _all_int_parameters(self._parameters_manager):
            _multiply(
                self._parameters_manager,
                self._results_manager,
                self._evaluation_directory,
            )
        else:
            raise ValueError("With continous parameters cannot run brute force.")

    def run_nsga2(
        self,
        population_size: int,
        termination: pm.Termination,
        p_crossover: float = 1.0,
        p_mutation: float = 0.2,
        callback: AnyCallback = None,
        saves_history: bool = True,
        expected_max_n_generation: int = 9999,
        seed: int | None = None,
    ) -> pm.Result:
        if isinstance(termination, pm.MaximumGenerationTermination):
            expected_max_n_generation = termination.n_max_gen

        return _optimise_epoch(
            self._evaluation_directory,
            self._to_pymoo(callback, expected_max_n_generation),
            population_size,
            termination,
            p_crossover,
            p_mutation,
            saves_history,
            seed,
        )
