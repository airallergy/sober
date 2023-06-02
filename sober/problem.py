import pickle
from pathlib import Path
from typing import Literal
from itertools import chain
from collections.abc import Iterable

from . import config as cf
from ._multiplier import _multiply
from ._tools import _write_records
from . import _pymoo_namespace as pm
from ._logger import _log, _LoggerManager
from ._typing import AnyStrPath, AnyPymooCallback
from ._optimiser import _sampling, _survival, _algorithm, _PymooProblem
from .results import RVICollector, ScriptCollector, _Collector, _ResultsManager
from .parameters import (
    WeatherModifier,
    AnyModelModifier,
    ContinuousModifier,
    _ParametersManager,
    _all_integral_modifiers,
)


#############################################################################
#######                         PROBLEM CLASS                         #######
#############################################################################
class Problem:
    """defines the parametrics/optimisation problem"""

    _model_file: Path
    _parameters_manager: _ParametersManager[AnyModelModifier]
    _results_manager: _ResultsManager
    _evaluation_directory: Path
    _config_directory: Path

    __slots__ = (
        "_model_file",
        "_parameters_manager",
        "_results_manager",
        "_evaluation_directory",
        "_config_directory",
    )

    def __init__(
        self,
        model_file: AnyStrPath,
        weather_parameter: WeatherModifier,
        /,
        model_parameters: Iterable[AnyModelModifier] = (),
        results: Iterable[_Collector] = (),
        *,
        evaluation_directory: AnyStrPath | None = None,
        has_templates: bool = False,
        clean_patterns: str | Iterable[str] = _ResultsManager._DEFAULT_CLEAN_PATTERNS,
        n_processes: int | None = None,
        python_exec: AnyStrPath | None = None,
    ) -> None:
        self._model_file = Path(model_file).resolve(strict=True)
        self._parameters_manager = _ParametersManager(
            weather_parameter, model_parameters, self._model_file, has_templates
        )
        self._results_manager = _ResultsManager(
            results, clean_patterns, self._parameters_manager._has_uncertainties
        )
        self._evaluation_directory = (
            self._model_file.parent / "evaluation"
            if evaluation_directory is None
            else Path(evaluation_directory)
        )
        self._config_directory = self._evaluation_directory / (
            "." + __package__.split(".")[-1]
        )

        self._prepare(n_processes, python_exec)

    def _prepare(self, n_processes: int | None, python_exec: AnyStrPath | None) -> None:
        # mkdir
        # intentionally assumes parents exist
        self._evaluation_directory.mkdir(exist_ok=True)
        self._config_directory.mkdir(exist_ok=True)

        # config
        cf.config_parallel(n_processes=n_processes)
        cf.config_script(python_exec=python_exec)
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

        # touch rvi
        # leave this here, otherwise need to pass _config_directory to _results_manager
        self._results_manager._touch_rvi(self._config_directory)

    def run_sample(self, size: int, /, *, seed: int | None = None) -> None:
        """runs a sample of the full search space"""

        cf._has_batches = False

        if _all_integral_modifiers(self._parameters_manager):
            _multiply(
                self._parameters_manager,
                self._results_manager,
                self._evaluation_directory,
                size,
                seed,
            )
        else:
            raise ValueError("with continous parameters cannot run sample.")

    def run_brute_force(self) -> None:
        """runs the full search space"""

        if _all_integral_modifiers(self._parameters_manager):
            self.run_sample(-1)
        else:
            raise ValueError("with continous parameters cannot run brute force.")

    def run_each_variation(self, *, seed: int | None = None) -> None:
        """runs a minimum sample of the full search space that contains all parameter variations
        this helps check the validity of each variation"""

        if _all_integral_modifiers(self._parameters_manager):
            self.run_sample(0, seed=seed)
        else:
            raise ValueError("with continous parameters cannot run each variation.")

    def _to_pymoo(
        self,
        callback: AnyPymooCallback,
        saves_batches: bool,
        expected_max_n_generations: int,
    ) -> _PymooProblem:
        if not len(self._results_manager._objectives):
            raise ValueError("optimisation needs at least one objective.")

        return _PymooProblem(
            self._parameters_manager,
            self._results_manager,
            self._evaluation_directory,
            callback,
            saves_batches,
            expected_max_n_generations,
        )

    def _record_survival(
        self, level: Literal["batch"], record_directory: Path, result: pm.Result
    ) -> None:
        # get evaluated individuals
        if result.algorithm.save_history:
            # from all generations
            _individuals = tuple(
                chain.from_iterable(item.pop for item in result.history)
            )
        else:
            # from the last generation
            _individuals = tuple(result.pop)

        # re-evaluate survival of individuals
        individuals = pm.Population.create(
            *sorted(_individuals, key=lambda x: tuple(x.X.values()))
        )
        individuals = _survival(individuals, result.algorithm)

        # create header row
        header_row = (
            tuple(individuals[0].X.keys())
            + tuple(self._results_manager._objectives)
            + tuple(self._results_manager._constraints)
            + ("is_pareto", "is_feasible")
        )

        # convert pymoo candidates to actual values
        candidate_vecs = tuple(
            tuple(component.item() for component in individual.X.values())
            for individual in individuals
        )
        value_vecs = tuple(
            tuple(
                component
                if isinstance(parameter, ContinuousModifier)
                else parameter[component]
                for parameter, component in zip(
                    self._parameters_manager, candidate_vec, strict=True
                )
            )
            for candidate_vec in candidate_vecs
        )

        # create record rows
        # NOTE: pymoo sums cvs of all constraints
        #       hence all(individual.FEAS) == individual.FEAS[0]
        record_rows = [
            value_vec
            + tuple(individual.F)
            + tuple(individual.G)
            + (individual.get("rank") == 0, all(individual.FEAS))
            for individual, value_vec in zip(individuals, value_vecs, strict=True)
        ]

        # write records
        _write_records(
            record_directory / cf._RECORDS_FILENAMES[level], header_row, *record_rows
        )

    @_LoggerManager(cwd_index=1, is_first=True)
    def _optimise_epoch(
        self,
        epoch_directory: Path,
        problem: _PymooProblem,
        algorithm: pm.Algorithm,
        termination: pm.Termination,
        save_history: bool,
        checkpoint_interval: int,
        seed: int | None,
    ) -> pm.Result:
        if checkpoint_interval <= 0:
            result = pm.minimize(
                problem, algorithm, termination, save_history=save_history, seed=seed
            )
        else:
            if not isinstance(termination, pm.MaximumGenerationTermination):
                # TODO: add support for other termination criteria
                #       possibly with a while loop
                raise ValueError(
                    "checkpoints only work with max generation termination."
                )

            n_loops = termination.n_max_gen // checkpoint_interval + 1
            for idx in range(n_loops):
                current_termination = (
                    termination
                    if idx + 1 == n_loops
                    else pm.MaximumGenerationTermination(
                        (idx + 1) * checkpoint_interval
                    )
                )

                # NOTE: in pymoo0.6
                #           algorithm.n_gen will increase by one at the end of optimisation
                #           at least when using a MaximumGenerationTermination
                if algorithm.n_gen and (
                    algorithm.n_gen - 1 >= current_termination.n_max_gen
                ):
                    # this should only be invoked by resume
                    continue

                # NOTE: in pymoo0.6
                #           algorithm.setup() is only triggered when algorithm.problem is None
                #           but seed will be reset too
                #           manually change algorithm.termination to avoid resetting seed
                algorithm.termination = current_termination

                result = pm.minimize(
                    problem,
                    algorithm,
                    current_termination,
                    save_history=save_history,
                    seed=seed,
                )

                # update algorithm
                algorithm = result.algorithm

                checkpoint_idx = (idx + 1) * checkpoint_interval - 1
                if algorithm.n_gen - 1 == checkpoint_idx + 1:
                    # TODO: explore implementing custom serialisation for self(Problem) via TOML/YAML
                    with (epoch_directory / "checkpoint.pickle").open("wb") as fp:
                        pickle.dump((self, result), fp)

                    _log(
                        epoch_directory,
                        f"created checkpoint at generation {checkpoint_idx}",
                    )

        self._record_survival("batch", epoch_directory, result)

        return result

    def run_nsga2(
        self,
        population_size: int,
        termination: pm.Termination,
        /,
        *,
        p_crossover: float = 1.0,
        p_mutation: float = 0.2,
        init_population_size: int = -1,
        callback: AnyPymooCallback = None,
        saves_history: bool = True,
        saves_batches: bool = True,
        checkpoint_interval: int = 0,
        expected_max_n_generations: int = 9999,
        seed: int | None = None,
    ) -> pm.Result:
        """runs optimisation using the NSGA2 algorithm"""

        if isinstance(termination, pm.MaximumGenerationTermination):
            expected_max_n_generations = termination.n_max_gen

        problem = self._to_pymoo(callback, saves_batches, expected_max_n_generations)

        if init_population_size <= 0:
            init_population_size = population_size
        sampling = _sampling(problem, init_population_size)

        algorithm = _algorithm(
            "nsga2", population_size, p_crossover, p_mutation, sampling
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

    def run_nsga3(
        self,
        population_size: int,
        termination: pm.Termination,
        /,
        reference_directions: pm.ReferenceDirectionFactory | None = None,
        *,
        p_crossover: float = 1.0,
        p_mutation: float = 0.2,
        init_population_size: int = -1,
        callback: AnyPymooCallback = None,
        saves_history: bool = True,
        saves_batches: bool = True,
        checkpoint_interval: int = 0,
        expected_max_n_generations: int = 9999,
        seed: int | None = None,
    ) -> pm.Result:
        """runs optimisation using the NSGA3 algorithm"""

        if isinstance(termination, pm.MaximumGenerationTermination):
            expected_max_n_generations = termination.n_max_gen

        problem = self._to_pymoo(callback, saves_batches, expected_max_n_generations)

        if init_population_size <= 0:
            init_population_size = population_size
        sampling = _sampling(problem, init_population_size)

        if not reference_directions:
            reference_directions = pm.RieszEnergyReferenceDirectionFactory(
                problem.n_obj, population_size, seed=seed
            ).do()

        algorithm = _algorithm(
            "nsga3",
            population_size,
            p_crossover,
            p_mutation,
            sampling,
            reference_directions,
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

    @staticmethod
    def resume(
        checkpoint_file: AnyStrPath,
        termination: pm.Termination,
        /,
        *,
        checkpoint_interval: int = 0,
    ) -> pm.Result:
        """resumes optimisation using checkpoint files"""
        # NOTE: although seed will not be reset
        #       randomness is not reproducible when resuming for some unknown reason
        # TODO: explore implementing custom serialisation for Problem via TOML/YAML
        # TODO: also consider moving this func outside Problem to be called directly
        # TODO: termination may not be necessary, as the old one may be reused

        checkpoint_file = Path(checkpoint_file).resolve(True)
        with checkpoint_file.open("rb") as fp:
            problem, result = pickle.load(fp)

        # checks validity of the check point file
        # currently only checks the object type, but there might be better checks
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
