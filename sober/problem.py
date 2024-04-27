from __future__ import annotations

from typing import TYPE_CHECKING, overload

import sober._evolver_pymoo as pm
import sober.config as cf
from sober._evolver import _PymooEvolver
from sober._io_managers import _InputManager, _OutputManager
from sober._multiplier import _CartesianMultiplier, _ElementwiseMultiplier
from sober._tools import _parsed_path
from sober.output import RVICollector, ScriptCollector, _Collector

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from typing import Literal, TypeAlias

    from sober._typing import (
        AnyModelModifier,
        AnyReferenceDirections,
        AnyStrPath,
        NoiseSampleKwargs,
    )
    from sober.input import WeatherModifier

    _AnyRandomMode: TypeAlias = Literal["elementwise", "cartesian", "auto"]


#############################################################################
#######                         PROBLEM CLASS                         #######
#############################################################################
class Problem:
    """defines the parametrics/optimisation problem"""

    __slots__ = (
        "_model_file",
        "_input_manager",
        "_output_manager",
        "_evaluation_dir",
        "_config_dir",
        "_elementwise",
        "_cartesian",
        "_pymoo",
    )

    _model_file: Path
    _input_manager: _InputManager
    _output_manager: _OutputManager
    _evaluation_dir: Path
    _config_dir: Path
    _elementwise: _ElementwiseMultiplier
    _cartesian: _CartesianMultiplier
    _pymoo: _PymooEvolver

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
        noise_sample_kwargs: NoiseSampleKwargs | None = None,
        removes_subdirs: bool = False,
    ) -> None:
        self._model_file = _parsed_path(model_file, "model file")
        self._input_manager = _InputManager(weather_input, model_inputs, has_templates)
        self._output_manager = _OutputManager(outputs, clean_patterns)
        self._evaluation_dir = (
            self._model_file.parent / "evaluation"
            if evaluation_dir is None
            else _parsed_path(evaluation_dir)
        )
        self._config_dir = self._evaluation_dir / ("." + __package__.split(".")[-1])

        self._prepare(n_processes, python_exec, noise_sample_kwargs, removes_subdirs)

    @overload
    def __getattr__(self, name: Literal["_elementwise"]) -> _ElementwiseMultiplier: ...  # type: ignore[misc]  # python/mypy#8203
    @overload
    def __getattr__(self, name: Literal["_cartesian"]) -> _CartesianMultiplier: ...  # type: ignore[misc]  # python/mypy#8203
    @overload
    def __getattr__(self, name: Literal["_pymoo"]) -> _PymooEvolver: ...  # type: ignore[misc]  # python/mypy#8203
    def __getattr__(  # type: ignore[misc]  # python/mypy#8203
        self, name: Literal["_elementwise", "_cartesian", "_pymoo"]
    ) -> _ElementwiseMultiplier | _CartesianMultiplier | _PymooEvolver:
        """lazily set these attributes when they are called for the first time"""
        match name:
            case "_elementwise":
                self._elementwise = _ElementwiseMultiplier(
                    self._input_manager, self._output_manager, self._evaluation_dir
                )
                return self._elementwise
            case "_cartesian":
                self._cartesian = _CartesianMultiplier(
                    self._input_manager, self._output_manager, self._evaluation_dir
                )
                return self._cartesian
            case "_pymoo":
                self._pymoo = _PymooEvolver(
                    self._input_manager, self._output_manager, self._evaluation_dir
                )
                return self._pymoo
            case _:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'."
                )

    def _check_args(self) -> None:
        pass

    def _prepare(
        self,
        n_processes: int | None,
        python_exec: AnyStrPath | None,
        noise_sample_kwargs: NoiseSampleKwargs | None,
        removes_subdirs: bool,
    ) -> None:
        self._check_args()

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

        # global variables
        cf._noise_sample_kwargs = noise_sample_kwargs
        cf._removes_subdirs = removes_subdirs

    def run_random(
        self, size: int, /, *, mode: _AnyRandomMode = "auto", seed: int | None = None
    ) -> None:
        """runs parametrics via a random sample"""

        if mode == "auto":
            mode = "elementwise" if self._input_manager._has_real_ctrls else "cartesian"

        if mode == "elementwise":
            self._elementwise._random(size, seed)
        else:
            self._cartesian._random(size, seed)

    def run_latin_hypercube(self, size: int, /, *, seed: int | None = None) -> None:
        """runs parametrics via a latin hypercube sample"""

        self._elementwise._latin_hypercube(size, seed)

    def run_exhaustive(self) -> None:
        """runs parametrics via the exhaustive sample"""

        self._cartesian._exhaustive()

    def run_nsga2(
        self,
        population_size: int,
        termination: pm.Termination,
        /,
        *,
        p_crossover: float = 1.0,
        p_mutation: float = 0.2,
        init_population_size: int = 0,
        saves_history: bool = True,
        checkpoint_interval: int = 0,
        seed: int | None = None,
    ) -> pm.Result:
        """runs optimisation via the NSGA2 algorithm"""

        return self._pymoo._nsga2(
            population_size,
            termination,
            p_crossover,
            p_mutation,
            init_population_size,
            saves_history,
            checkpoint_interval,
            seed,
        )

    def run_nsga3(
        self,
        population_size: int,
        termination: pm.Termination,
        /,
        reference_directions: AnyReferenceDirections | None = None,
        *,
        p_crossover: float = 1.0,
        p_mutation: float = 0.2,
        init_population_size: int = 0,
        saves_history: bool = True,
        checkpoint_interval: int = 0,
        seed: int | None = None,
    ) -> pm.Result:
        """runs optimisation via the NSGA3 algorithm"""

        return self._pymoo._nsga3(
            population_size,
            termination,
            reference_directions,
            p_crossover,
            p_mutation,
            init_population_size,
            saves_history,
            checkpoint_interval,
            seed,
        )

    def resume(
        self,
        checkpoint_file: AnyStrPath,
        termination: pm.Termination,
        /,
        *,
        checkpoint_interval: int = 0,
    ) -> pm.Result:
        """resumes optimisation using a checkpoint file"""
        return self._pymoo.resume(checkpoint_file, termination, checkpoint_interval)
