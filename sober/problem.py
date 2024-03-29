import pickle
from collections.abc import Iterable
from itertools import chain
from pathlib import Path
from typing import Literal

from . import _pymoo_namespace as pm
from . import config as cf
from ._logger import _log, _LoggerManager
from ._multiplier import _multiply
from ._optimiser import _algorithm, _PymooProblem, _sampling, _survival
from ._tools import _write_records
from ._typing import AnyPymooCallback, AnyStrPath
from .input import (
    AnyModelModifier,
    ContinuousModifier,
    WeatherModifier,
    _all_integral_modifiers,
    _InputManager,
)
from .output import RVICollector, ScriptCollector, _Collector, _OutputManager


#############################################################################
#######                         PROBLEM CLASS                         #######
#############################################################################
class Problem:
    """defines the parametrics/optimisation problem"""

    _model_file: Path
    _input_manager: _InputManager[AnyModelModifier]
    _output_manager: _OutputManager
    _evaluation_dir: Path
    _config_dir: Path

    __slots__ = (
        "_model_file",
        "_input_manager",
        "_output_manager",
        "_evaluation_dir",
        "_config_dir",
    )

    def __init__(
        self,
        model_file: AnyStrPath,
        weather_input: WeatherModifier,
        /,
        model_inputs: Iterable[AnyModelModifier] = (),
        outputs: Iterable[_Collector] = (),
        *,
        evaluation_dir: AnyStrPath | None = None,
        has_templates: bool = False,
        clean_patterns: str | Iterable[str] = _OutputManager._DEFAULT_CLEAN_PATTERNS,
        n_processes: int | None = None,
        python_exec: AnyStrPath | None = None,
    ) -> None:
        self._model_file = Path(model_file).resolve(strict=True)
        self._input_manager = _InputManager(
            weather_input, model_inputs, self._model_file, has_templates
        )
        self._output_manager = _OutputManager(
            outputs, clean_patterns, self._input_manager._has_uncertainties
        )
        self._evaluation_dir = (
            self._model_file.parent / "evaluation"
            if evaluation_dir is None
            else Path(evaluation_dir)
        )
        self._config_dir = self._evaluation_dir / ("." + __package__.split(".")[-1])

        self._prepare(n_processes, python_exec)

    def _prepare(self, n_processes: int | None, python_exec: AnyStrPath | None) -> None:
        # mkdir
        # intentionally assumes parents exist
        self._evaluation_dir.mkdir(exist_ok=True)
        self._config_dir.mkdir(exist_ok=True)

        # config
        cf.config_parallel(n_processes=n_processes)
        cf.config_script(python_exec=python_exec)
        cf._check_config(
            self._input_manager._model_type,
            self._input_manager._has_templates,
            any(isinstance(item, RVICollector) for item in self._output_manager),
            set(
                item._language
                for item in self._output_manager
                if isinstance(item, ScriptCollector)
            ),
        )

        # touch rvi
        # leave this here, otherwise need to pass _config_dir to _output_manager
        self._output_manager._touch_rvi(self._config_dir)

    def run_sample(self, size: int, /, *, seed: int | None = None) -> None:
        """runs a sample of the full search space"""

        cf._has_batches = False

        if _all_integral_modifiers(self._input_manager):
            _multiply(
                self._input_manager,
                self._output_manager,
                self._evaluation_dir,
                size,
                seed,
            )
        else:
            raise ValueError("with continuous inputs cannot run sample.")

    def run_brute_force(self) -> None:
        """runs the full search space"""

        if _all_integral_modifiers(self._input_manager):
            self.run_sample(-1)
        else:
            raise ValueError("with continuous inputs cannot run brute force.")

    def run_each_variation(self, *, seed: int | None = None) -> None:
        """runs a minimum sample of the full search space that contains all input variations
        this helps check the validity of each variation"""

        if _all_integral_modifiers(self._input_manager):
            self.run_sample(0, seed=seed)
        else:
            raise ValueError("with continuous inputs cannot run each variation.")

    def _to_pymoo(
        self,
        callback: AnyPymooCallback,
        saves_batches: bool,
        expected_n_generations: int,
    ) -> _PymooProblem:
        if not len(self._output_manager._objectives):
            raise ValueError("optimisation needs at least one objective.")

        return _PymooProblem(
            self._input_manager,
            self._output_manager,
            self._evaluation_dir,
            callback,
            saves_batches,
            expected_n_generations,
        )

    def _record_survival(
        self, level: Literal["batch"], record_dir: Path, result: pm.Result
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
            + tuple(self._output_manager._objectives)
            + tuple(self._output_manager._constraints)
            + ("is_pareto", "is_feasible")
        )

        # convert pymoo candidates to actual values
        candidate_vecs = tuple(
            tuple(component.item() for component in individual.X.values())
            for individual in individuals
        )
        value_vecs = tuple(
            tuple(
                (
                    component
                    if isinstance(input, ContinuousModifier)
                    else input[component]
                )
                for input, component in zip(
                    self._input_manager, candidate_vec, strict=True
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
            record_dir / cf._RECORDS_FILENAMES[level], header_row, *record_rows
        )

    @_LoggerManager(cwd_index=1, is_first=True)
    def _optimise_epoch(
        self,
        epoch_dir: Path,
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
                    with (epoch_dir / "checkpoint.pickle").open("wb") as fp:
                        pickle.dump((self, result), fp)

                    _log(
                        epoch_dir, f"created checkpoint at generation {checkpoint_idx}"
                    )

        self._record_survival("batch", epoch_dir, result)

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
        expected_n_generations: int = 9999,
        seed: int | None = None,
    ) -> pm.Result:
        """runs optimisation using the NSGA2 algorithm"""

        if isinstance(termination, pm.MaximumGenerationTermination):
            expected_n_generations = termination.n_max_gen

        problem = self._to_pymoo(callback, saves_batches, expected_n_generations)

        if init_population_size <= 0:
            init_population_size = population_size
        sampling = _sampling(problem, init_population_size)

        algorithm = _algorithm(
            "nsga2", population_size, p_crossover, p_mutation, sampling
        )

        return self._optimise_epoch(
            self._evaluation_dir,
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
        expected_n_generations: int = 9999,
        seed: int | None = None,
    ) -> pm.Result:
        """runs optimisation using the NSGA3 algorithm"""

        if isinstance(termination, pm.MaximumGenerationTermination):
            expected_n_generations = termination.n_max_gen

        problem = self._to_pymoo(callback, saves_batches, expected_n_generations)

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
            self._evaluation_dir,
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
            problem._evaluation_dir,
            result.problem,
            result.algorithm,
            termination,
            result.algorithm.save_history,  # void
            checkpoint_interval,
            result.algorithm.seed,  # void
        )  # void: only termination will be updated in algorithm
