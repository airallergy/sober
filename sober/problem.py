import pickle
from math import log10
from pathlib import Path
from shutil import rmtree
from collections.abc import Iterable

import numpy as np

from . import config as cf
from ._multiplier import _multiply
from ._optimiser import _algorithm
from . import _pymoo_namespace as pm
from ._evaluator import _pymoo_evaluate
from ._logger import _log, _LoggerManager
from ._typing import AnyStrPath, AnyCallback, AnyVariationMap
from .results import RVICollector, ScriptCollector, _Collector, _ResultsManager
from .parameters import (
    AnyParameter,
    WeatherParameter,
    ContinuousParameter,
    _ParametersManager,
    _all_int_parameters,
)


class _PymooProblem(pm.Problem):
    _parameters_manager: _ParametersManager
    _results_manager: _ResultsManager
    _evaluation_directory: Path
    _len_batch_count: int
    _saves_batches: bool

    def __init__(
        self,
        parameters_manager: _ParametersManager,
        results_manager: _ResultsManager,
        evaluation_directory: Path,
        callback: AnyCallback,
        expected_max_n_generation: int,
        saves_batches: bool,
    ) -> None:
        n_parameters = len(parameters_manager)
        variables = {
            f"x{idx:0{int(log10(n_parameters)) + 1}}": (
                pm.Real(bounds=(parameter._low, parameter._high))
                if isinstance(parameter, ContinuousParameter)
                else pm.Integer(bounds=(parameter._low, parameter._high))
            )
            for idx, parameter in enumerate(parameters_manager)
        }

        super().__init__(
            n_obj=len(results_manager._objectives),
            n_ieq_constr=len(results_manager._constraints),
            vars=variables,
            callback=callback,
        )
        self._parameters_manager = parameters_manager
        self._results_manager = results_manager
        self._evaluation_directory = evaluation_directory
        self._len_batch_count = int(log10(expected_max_n_generation)) + 1
        self._saves_batches = saves_batches

    def _evaluate(
        self,
        x: tuple[AnyVariationMap, ...],
        out,
        *args,
        algorithm: pm.Algorithm,
        **kwargs,
    ) -> None:
        batch_uid = f"B{algorithm.n_gen - 1:0{self._len_batch_count}}"
        objectives, constraints = _pymoo_evaluate(
            *(
                tuple(item.values()) for item in x
            ),  # type:ignore[arg-type] # python/mypy#12280
            parameters_manager=self._parameters_manager,
            results_manager=self._results_manager,
            batch_directory=self._evaluation_directory / batch_uid,
        )
        out["F"] = np.asarray(objectives, dtype=np.float_)
        out["G"] = np.asarray(constraints, dtype=np.float_)

        if not self._saves_batches:
            rmtree(self._evaluation_directory / batch_uid)

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
        self,
        callback: AnyCallback,
        expected_max_n_generation: int,
        saves_batches: bool,
    ) -> _PymooProblem:
        if not len(self._results_manager._objectives):
            raise ValueError("Optimisation needs at least one objective")

        return _PymooProblem(
            self._parameters_manager,
            self._results_manager,
            self._evaluation_directory,
            callback,
            expected_max_n_generation,
            saves_batches,
        )

    def run_sample(self, sample_size: int, seed: int | None = None) -> None:
        cf._has_batches = False

        if _all_int_parameters(self._parameters_manager):
            _multiply(
                self._parameters_manager,
                self._results_manager,
                self._evaluation_directory,
                sample_size,
                seed,
            )
        else:
            raise ValueError("With continous parameters cannot run sample.")

    def run_brute_force(self) -> None:
        if _all_int_parameters(self._parameters_manager):
            self.run_sample(-1)
        else:
            raise ValueError("With continous parameters cannot run brute force.")

    def run_each_variation(self, seed: int | None = None) -> None:
        if _all_int_parameters(self._parameters_manager):
            self.run_sample(0, seed)
        else:
            raise ValueError("With continous parameters cannot run each variation.")

    @_LoggerManager(cwd_index=1, is_first=True)
    def _optimise_epoch(
        self,
        cwd: Path,
        problem: _PymooProblem,
        algorithm: pm.Algorithm,
        termination: pm.Termination,
        save_history: bool,
        checkpoint_interval: int,
        seed: int | None,
    ) -> pm.Result:
        if checkpoint_interval <= 0:
            return pm.minimize(
                problem, algorithm, termination, save_history=save_history, seed=seed
            )
        else:
            if not isinstance(termination, pm.MaximumGenerationTermination):
                raise ValueError(
                    "checkpoints only work with max generation termination."
                )

            n_loops = termination.n_max_gen // checkpoint_interval + 1
            for idx in range(n_loops):
                algorithm.has_terminated = False
                current_termination = (
                    termination
                    if idx + 1 == n_loops
                    else pm.MaximumGenerationTermination(
                        (idx + 1) * checkpoint_interval
                    )
                )

                if algorithm.n_gen and (
                    algorithm.n_gen >= current_termination.n_max_gen
                ):
                    # for resume
                    # >= is in alignment with pm.MaximumGenerationTermination
                    continue

                result = pm.minimize(
                    problem,
                    algorithm,
                    current_termination,
                    save_history=save_history,
                    seed=seed,
                )

                with (cwd / "checkpoint.pickle").open("wb") as fp:
                    pickle.dump((self, result), fp)
                _log(
                    cwd,
                    f"created checkpoint at generation {current_termination.n_max_gen-1}",
                )
                algorithm = result.algorithm
            return result

    @staticmethod
    def resume(
        checkpoint_file: AnyStrPath,
        termination: pm.Termination,
        checkpoint_interval: int = 0,
    ) -> pm.Result:
        checkpoint_file = Path(checkpoint_file)
        with checkpoint_file.open("rb") as fp:
            problem, result = pickle.load(fp)
        if not (isinstance(problem, Problem) and isinstance(result, pm.Result)):
            raise TypeError(f"invalid checkpoint file: {checkpoint_file.resolve()}.")

        return problem._optimise_epoch(
            problem._evaluation_directory,
            result.problem,
            result.algorithm,
            termination,
            result.algorithm.save_history,  # void
            checkpoint_interval,
            result.algorithm.seed,  # void
        )  # void: only termination will be updated in algorithm

    def run_nsga2(
        self,
        population_size: int,
        termination: pm.Termination,
        p_crossover: float = 1.0,
        p_mutation: float = 0.2,
        callback: AnyCallback = None,
        saves_history: bool = True,
        expected_max_n_generation: int = 9999,
        saves_batches: bool = True,
        checkpoint_interval: int = 0,
        seed: int | None = None,
    ) -> pm.Result:
        if isinstance(termination, pm.MaximumGenerationTermination):
            expected_max_n_generation = termination.n_max_gen

        problem = self._to_pymoo(callback, expected_max_n_generation, saves_batches)
        algorithm = _algorithm("nsga2", population_size, p_crossover, p_mutation)

        return self._optimise_epoch(
            self._evaluation_directory,
            problem,
            algorithm,
            termination,
            saves_history,
            checkpoint_interval,
            seed,
        )

    def run_nsga3(
        self,
        population_size: int,
        termination: pm.Termination,
        p_crossover: float = 1.0,
        p_mutation: float = 0.2,
        reference_directions: pm.ReferenceDirectionFactory | None = None,
        callback: AnyCallback = None,
        saves_history: bool = True,
        expected_max_n_generation: int = 9999,
        saves_batches: bool = True,
        checkpoint_interval: int = 0,
        seed: int | None = None,
    ) -> pm.Result:
        if isinstance(termination, pm.MaximumGenerationTermination):
            expected_max_n_generation = termination.n_max_gen

        problem = self._to_pymoo(callback, expected_max_n_generation, saves_batches)
        if not reference_directions:
            reference_directions = pm.RieszEnergyReferenceDirectionFactory(
                problem.n_obj, population_size, seed=seed
            ).do()

        algorithm = _algorithm(
            "nsga3", population_size, p_crossover, p_mutation, reference_directions
        )

        return self._optimise_epoch(
            self._evaluation_directory,
            problem,
            algorithm,
            termination,
            saves_history,
            checkpoint_interval,
            seed,
        )
