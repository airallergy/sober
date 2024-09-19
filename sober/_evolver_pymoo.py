from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, cast, overload

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.algorithms.moo.nsga3 import NSGA3, comp_by_cv_then_random
from pymoo.core.mixed import MixedVariableDuplicateElimination, MixedVariableMating
from pymoo.core.problem import Problem
from pymoo.core.variable import Integer as Integral  # follow the numbers stdlib
from pymoo.core.variable import Real
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.termination.max_gen import MaximumGenerationTermination

from sober._evaluator import _evaluate
from sober._logger import _log
from sober._tools import _natural_width
from sober._typing import AnyCtrlKeyVec
from sober.input import _RealModifier

if TYPE_CHECKING:
    # ruff: noqa: PLC0414  # astral-sh/ruff#3711
    from collections.abc import Iterable, Iterator
    from pathlib import Path
    from typing import Literal, NotRequired, Protocol, Self, TypedDict

    from numpy.typing import NDArray
    from pymoo.algorithms.base.genetic import GeneticAlgorithm
    from pymoo.core.termination import Termination as Termination

    from sober._io_managers import _InputManager, _OutputManager
    from sober._typing import AnyF, AnyG, AnyReferenceDirections, AnyX

    class _PymooOut(TypedDict):
        F: AnyF | None
        G: NotRequired[AnyG | None]

    class _PymooOperators(TypedDict):
        sampling: Population
        mating: MixedVariableMating
        eliminate_duplicates: MixedVariableDuplicateElimination

    # minimum pymoo stubs
    from pymoo.core.algorithm import Algorithm as Algorithm_
    from pymoo.core.mixed import MixedVariableSampling as MixedVariableSampling_
    from pymoo.util.ref_dirs.energy import (
        RieszEnergyReferenceDirectionFactory as RieszEnergyReferenceDirectionFactory_,
    )

    class Individual(Protocol):
        __slots__ = ("X", "F", "G", "FEAS")

        X: AnyX
        F: AnyF
        G: AnyG
        FEAS: NDArray[np.bool_]

        def get(self, arg: Literal["rank"]) -> int: ...

    class Population(Protocol):
        __slots__ = ()

        def __iter__(self) -> Iterator[Individual]: ...
        @classmethod
        def create(cls, *args: Individual) -> Self: ...

    class MixedVariableSampling(MixedVariableSampling_):  # type: ignore[misc]
        # this cannot be defined via Protocol as Protocol cannot be instantiated
        __slots__ = ()

        def __call__(self, problem: Problem, n_samples: int) -> Population: ...

    class Result(Protocol):
        __slots__ = ("pop", "history", "algorithm")

        pop: Population
        history: list[Algorithm]
        algorithm: Algorithm

    class Algorithm(Algorithm_):  # type: ignore[misc]
        # this cannot be defined via Protocol as Protocol cannot be runtime checked
        __slots__ = (
            "problem",
            "termination",
            "save_history",
            "seed",
            "is_initialized",
            "n_gen",
            "pop",
        )

        problem: Problem
        termination: Termination
        save_history: bool
        seed: int | None
        is_initialized: bool
        n_gen: int
        # NOTE: pymoo sets n_gen to None before initialisation
        #       which is problematic given the very short time before initialisation
        #       restrict it to int here
        #       and handle pre-initialisation using is_initialized
        pop: Population

        def setup(
            self,
            problem: Problem,
            termination: Termination,
            save_history: bool,
            seed: int | None,
            **kwargs: object,
        ) -> Self: ...
        def has_next(self) -> bool: ...
        def next(self) -> None: ...
        def result(self) -> Result: ...

    class RieszEnergyReferenceDirectionFactory(RieszEnergyReferenceDirectionFactory_):  # type: ignore[misc]
        # this cannot be defined via Protocol as Protocol cannot be instantiated
        __slots__ = ()

        def __init__(self, n_dim: int, n_points: int, *, seed: int | None) -> None: ...
        def do(self) -> AnyReferenceDirections: ...

    def minimize(
        problem: Problem,
        algorithm: Algorithm,
        termination: Termination,
        *,
        save_history: bool,
        seed: int | None,
    ) -> Result: ...

else:
    from pymoo.core.algorithm import Algorithm  # used in resume
    from pymoo.core.mixed import MixedVariableSampling
    from pymoo.core.population import Population
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory

__all__ = ("Algorithm", "MaximumGenerationTermination", "minimize")


#############################################################################
#######                   PYMOO PROBLEM CHILD CLASS                   #######
#############################################################################
class _Problem(Problem):  # type: ignore[misc]  # pymoo
    """interfaces the pymoo problem"""

    _input_manager: _InputManager
    _output_manager: _OutputManager
    _evaluation_dir: Path
    _i_batch_width: int

    def __init__(
        self,
        input_manager: _InputManager,
        output_manager: _OutputManager,
        evaluation_dir: Path,
    ) -> None:
        self._input_manager = input_manager
        self._output_manager = output_manager
        self._evaluation_dir = evaluation_dir

        # NOTE: pymoo0.6 asks for a dict from input uids to pymoo variable types
        # only control variables are passed
        ctrl_vars = {
            item._label: (
                (Real if isinstance(item, _RealModifier) else Integral)(
                    bounds=item._bounds
                )
            )
            for item in (item for item in input_manager if item._is_ctrl)
        }

        super().__init__(
            n_obj=len(output_manager._objectives),
            n_ieq_constr=len(output_manager._constraints),
            vars=ctrl_vars,
            requires_kwargs=True,
        )

    def _evaluate(
        self,
        x: Iterable[AnyX],
        out: _PymooOut,
        *args: object,
        algorithm: Algorithm,
        **kwargs: object,
    ) -> None:
        # NOTE: in pymoo0.6
        #           n_gen follows 1, 2, 3, ...
        #           x is a numpy array of dicts, each dict is a control key map
        #               whose keys are input labels
        #                     values are control key vectors
        #           out has to be a dict of numpy arrays

        i_batch = algorithm.n_gen - 1

        # set self._i_batch_width in the initial generation
        if i_batch == 0:
            if isinstance(algorithm.termination, MaximumGenerationTermination):
                expected_n_generations = algorithm.termination.n_max_gen
            else:
                expected_n_generations = 9999

            self._i_batch_width = _natural_width(expected_n_generations)

        batch_uid = f"B{i_batch:0{self._i_batch_width}}"

        # convert pymoo x to ctrl key vecs
        ctrl_key_vecs = tuple(
            tuple(
                ctrl_key_map[item._label].item()
                if item._is_ctrl
                else item._hype_ctrl_key()
                for item in self._input_manager
            )
            for ctrl_key_map in x
        )

        ctrl_key_vecs = cast(tuple[AnyCtrlKeyVec, ...], ctrl_key_vecs)  # mypy

        # evaluate and get objectives and constraints
        batch_dir = self._evaluation_dir / batch_uid
        _evaluate(
            *ctrl_key_vecs,
            input_manager=self._input_manager,
            output_manager=self._output_manager,
            batch_dir=batch_dir,
        )
        objectives = self._output_manager._recorded_objectives(batch_dir)
        constraints = self._output_manager._recorded_constraints(batch_dir)

        out["F"] = np.asarray(objectives, dtype=float)
        if self._output_manager._constraints:
            out["G"] = np.asarray(constraints, dtype=float)

        _log(self._evaluation_dir, f"evaluated {batch_uid}")

        if self._output_manager._removes_subdirs:
            shutil.rmtree(batch_dir)


#############################################################################
#######                      OPERATOR FUNCTIONS                       #######
#############################################################################
def _sampling(problem: Problem, init_population_size: int) -> Population:
    """samples the initial generation"""

    return MixedVariableSampling()(problem, init_population_size)


def _operators(
    algorithm_name: Literal["nsga2", "nsga3"],
    p_crossover: float,
    p_mutation: float,
    sampling: Population,
) -> _PymooOperators:
    """a pymoo operators constructor"""

    # defaults from respective algorithm classes in pymoo
    selections = {
        "nsga2": TournamentSelection(func_comp=binary_tournament),
        "nsga3": TournamentSelection(func_comp=comp_by_cv_then_random),
    }
    etas = {
        "nsga2": {"crossover": 15, "mutation": 20},
        "nsga3": {"crossover": 30, "mutation": 20},
    }

    crossover_kwargs = {"prob": p_crossover, "eta": etas[algorithm_name]["crossover"]}

    # NOTE: in pymoo0.6 mutation
    #           prob (0.5) -> prob_var, controlling mutation for each gene/variable
    #           prob controls the whole mutation operation
    #           see https://github.com/anyoptimization/pymoo/discussions/360
    #           though the answer is not entirely correct
    mutation_kwargs = {
        "prob": 1.0,
        "prob_var": p_mutation,
        "eta": etas[algorithm_name]["mutation"],
    }

    return {
        "sampling": sampling,
        "mating": MixedVariableMating(
            selection=selections[algorithm_name],
            crossover={
                Real: SimulatedBinaryCrossover(**crossover_kwargs),
                Integral: SimulatedBinaryCrossover(
                    **crossover_kwargs, vtype=float, repair=RoundingRepair()
                ),
            },
            mutation={
                Real: PolynomialMutation(**mutation_kwargs),
                Integral: PolynomialMutation(
                    **mutation_kwargs, vtype=float, repair=RoundingRepair()
                ),
            },
            eliminate_duplicates=MixedVariableDuplicateElimination(),
        ),
        "eliminate_duplicates": MixedVariableDuplicateElimination(),
    }


#############################################################################
#######                      ALGORITHM FUNCTIONS                      #######
#############################################################################
@overload
def _algorithm(
    algorithm_name: Literal["nsga2"],
    population_size: int,
    p_crossover: float,
    p_mutation: float,
    sampling: Population,
) -> NSGA2: ...
@overload
def _algorithm(
    algorithm_name: Literal["nsga3"],
    population_size: int,
    p_crossover: float,
    p_mutation: float,
    sampling: Population,
    reference_directions: AnyReferenceDirections,
) -> NSGA3: ...
def _algorithm(
    algorithm_name: Literal["nsga2", "nsga3"],
    population_size: int,
    p_crossover: float,
    p_mutation: float,
    sampling: Population,
    reference_directions: None | AnyReferenceDirections = None,
) -> NSGA2 | NSGA3:
    """a pymoo algorithm constructor"""

    if algorithm_name == "nsga2":
        return NSGA2(
            population_size,
            **_operators(algorithm_name, p_crossover, p_mutation, sampling),
        )
    else:
        if TYPE_CHECKING:
            assert reference_directions is not None

        return NSGA3(
            reference_directions,
            population_size,
            **_operators(algorithm_name, p_crossover, p_mutation, sampling),
        )


#############################################################################
#######                       UTILITY FUNCTIONS                       #######
#############################################################################
def _survival(
    individuals: Iterable[Individual], algorithm: GeneticAlgorithm
) -> Population:
    """evaluates survival of individuals"""
    population = Population.create(*individuals)

    # remove duplicates
    population = MixedVariableDuplicateElimination().do(population)

    # runs survival
    # this resets the value of Individual().data["rank"] for each individual
    algorithm.survival.do(algorithm.problem, population, algorithm=algorithm)

    return population


def _default_reference_directions(
    n_dims: int, population_size: int, seed: int | None
) -> AnyReferenceDirections:
    return RieszEnergyReferenceDirectionFactory(n_dims, population_size, seed=seed).do()
