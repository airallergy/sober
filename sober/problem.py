from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import sober._pymoo_namespace as pm
import sober.config as cf
from sober._evolver import _algorithm, _PymooProblem, _sampling
from sober._io_managers import _InputManager, _OutputManager
from sober._multiplier import _multiply
from sober.output import RVICollector, ScriptCollector, _Collector

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sober._typing import AnyPymooCallback, AnyStrPath
    from sober.input import AnyModelModifier, WeatherModifier


#############################################################################
#######                         PROBLEM CLASS                         #######
#############################################################################
class Problem:
    """defines the parametrics/optimisation problem"""

    _model_file: Path
    _input_manager: _InputManager
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
        self._input_manager = _InputManager(weather_input, model_inputs, has_templates)
        self._output_manager = _OutputManager(outputs, clean_patterns)
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

        # prepare io managers
        self._input_manager._prepare(self._model_file)
        self._output_manager._prepare(self._config_dir, self._input_manager._has_noises)

        # config
        cf.config_parallel(n_processes=n_processes)
        cf.config_script(python_exec=python_exec)
        cf._check_config(
            self._input_manager._model_type,
            self._input_manager._has_templates,
            any(isinstance(item, RVICollector) for item in self._output_manager),
            {
                item._language
                for item in self._output_manager
                if isinstance(item, ScriptCollector)
            },
        )

    def run_sample(self, size: int, /, *, seed: int | None = None) -> None:
        """runs a sample of the full search space"""

        cf._has_batches = False

        _multiply(
            self._input_manager, self._output_manager, self._evaluation_dir, size, seed
        )

    def run_brute_force(self) -> None:
        """runs the full search space"""

        self.run_sample(-1)

    def run_each_option(self, *, seed: int | None = None) -> None:
        """runs a minimum sample of the full search space that contains all control options
        this helps check the validity of each option"""

        self.run_sample(0, seed=seed)

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

        return problem._evolve_epoch(
            self._evaluation_dir,
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

        return problem._evolve_epoch(
            self._evaluation_dir,
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
            epoch_dir, result = pickle.load(fp)

        # checks validity of the check point file
        # currently only checks the object type, but there might be better checks
        if not (isinstance(epoch_dir, Path) and isinstance(result, pm.Result)):
            raise TypeError(f"invalid checkpoint file: {checkpoint_file.resolve()}.")

        return result.algorithm.problem._evolve_epoch(
            epoch_dir,
            result.problem,
            result.algorithm,
            termination,
            result.algorithm.save_history,  # void
            checkpoint_interval,
            result.algorithm.seed,  # void
        )  # void: only termination will be updated in algorithm
