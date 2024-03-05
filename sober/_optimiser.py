from collections.abc import Iterable
from pathlib import Path
from shutil import rmtree
from typing import Literal, overload

import numpy as np

from . import _pymoo_namespace as pm
from ._evaluator import _pymoo_evaluate
from ._logger import _log
from ._tools import _natural_width
from ._typing import AnyCandidateMap, AnyPymooCallback, PymooOperators, PymooOut
from .parameters import ContinuousModifier, _ParametersManager
from .results import _ResultsManager


#############################################################################
#######                   PYMOO PROBLEM CHILD CLASS                   #######
#############################################################################
class _PymooProblem(pm.Problem):
    """interfaces the pymoo problem"""

    _parameters_manager: _ParametersManager
    _results_manager: _ResultsManager
    _evaluation_directory: Path
    _saves_batches: bool
    _batch_idx_width: int

    def __init__(
        self,
        parameters_manager: _ParametersManager,
        results_manager: _ResultsManager,
        evaluation_directory: Path,
        callback: AnyPymooCallback,
        saves_batches: bool,
        expected_n_generations: int,
    ) -> None:
        # NOTE: pymoo0.6 asks for a map from parameter uids to pymoo variable types
        variables = {
            parameter._label: (
                pm.Real(bounds=(parameter._low, parameter._high))
                if isinstance(parameter, ContinuousModifier)
                else pm.Integral(bounds=(parameter._low, parameter._high))
            )
            for parameter in parameters_manager
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
        self._saves_batches = saves_batches
        self._batch_idx_width = _natural_width(expected_n_generations)

    def _evaluate(
        self,
        x: Iterable[AnyCandidateMap],
        out: PymooOut,
        *args,
        algorithm: pm.Algorithm,
        **kwargs,
    ) -> None:
        # NOTE: in pymoo0.6
        #           n_gen follows 1, 2, 3, ...
        #           x is a numpy array of dicts, each dict is a candidate map
        #               whose keys are parameter labels
        #                     values are candidate vectors
        #           out has to be a dict of numpy arrays

        batch_idx = algorithm.n_gen - 1
        batch_uid = f"B{batch_idx:0{self._batch_idx_width}}"

        candidate_vecs = tuple(
            tuple(component.item() for component in candidate_map.values())
            for candidate_map in x
        )

        objectives, constraints = _pymoo_evaluate(
            *candidate_vecs,
            parameters_manager=self._parameters_manager,
            results_manager=self._results_manager,
            batch_directory=self._evaluation_directory / batch_uid,
        )

        out["F"] = np.asarray(objectives, dtype=np.float_)
        if self._results_manager._constraints:
            out["G"] = np.asarray(constraints, dtype=np.float_)

        if not self._saves_batches:
            rmtree(self._evaluation_directory / batch_uid)

        _log(self._evaluation_directory, f"evaluated {batch_uid}")


#############################################################################
#######                      OPERATOR FUNCTIONS                       #######
#############################################################################
def _sampling(problem: pm.Problem, init_population_size: int) -> pm.Population:
    """samples the initial generation"""

    return pm.MixedVariableSampling()(problem, init_population_size)


def _operators(
    algorithm_name: Literal["nsga2", "nsga3"],
    p_crossover: float,
    p_mutation: float,
    sampling: pm.Population,
) -> PymooOperators:
    """a pymoo operators constructor"""

    # defaults from respective algoritm classes in pymoo
    selections = {
        "nsga2": pm.TournamentSelection(func_comp=pm.binary_tournament),
        "nsga3": pm.TournamentSelection(func_comp=pm.comp_by_cv_then_random),
    }
    etas = {
        "nsga2": {"crossover": 15, "mutation": 20},
        "nsga3": {"crossover": 30, "mutation": 20},
    }

    crossover_kwargs = dict(prob=p_crossover, eta=etas[algorithm_name]["crossover"])

    # NOTE: in pymoo0.6 mutation
    #           prob (0.5) -> prob_var, controlling mutation for each gene/variable
    #           prob controls the whole mutation operation
    #           see https://github.com/anyoptimization/pymoo/discussions/360
    #           though the answer is not entirely correct
    mutation_kwargs = dict(
        prob=1.0, prob_var=p_mutation, eta=etas[algorithm_name]["mutation"]
    )

    return {
        "sampling": sampling,
        "mating": pm.MixedVariableMating(
            selection=selections[algorithm_name],
            crossover={
                pm.Real: pm.SimulatedBinaryCrossover(**crossover_kwargs),
                pm.Integral: pm.SimulatedBinaryCrossover(
                    **crossover_kwargs, vtype=float, repair=pm.RoundingRepair()
                ),
            },
            mutation={
                pm.Real: pm.PolynomialMutation(**mutation_kwargs),
                pm.Integral: pm.PolynomialMutation(
                    **mutation_kwargs, vtype=float, repair=pm.RoundingRepair()
                ),
            },
            eliminate_duplicates=pm.MixedVariableDuplicateElimination(),
        ),
        "eliminate_duplicates": pm.MixedVariableDuplicateElimination(),
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
    sampling: pm.Population,
) -> pm.NSGA2: ...


@overload
def _algorithm(
    algorithm_name: Literal["nsga3"],
    population_size: int,
    p_crossover: float,
    p_mutation: float,
    sampling: pm.Population,
    reference_directions: pm.ReferenceDirectionFactory,
) -> pm.NSGA3: ...


def _algorithm(
    algorithm_name,
    population_size,
    p_crossover,
    p_mutation,
    sampling,
    reference_directions=None,
):
    """a pymoo algorithm constructor"""

    if algorithm_name == "nsga2":
        return pm.NSGA2(
            population_size,
            **_operators(algorithm_name, p_crossover, p_mutation, sampling),
        )
    else:
        return pm.NSGA3(
            reference_directions,
            population_size,
            **_operators(algorithm_name, p_crossover, p_mutation, sampling),
        )


#############################################################################
#######                      SURVIVAL FUNCTIONS                       #######
#############################################################################
def _survival(
    individuals: pm.Population, algorithm: pm.GeneticAlgorithm
) -> pm.Population:
    """evaluates survival of individuals"""
    # remove duplicates
    individuals = pm.MixedVariableDuplicateElimination().do(individuals)

    # runs survival
    # this resets the value of Individual().data["rank"] for each individual
    algorithm.survival.do(algorithm.problem, individuals, algorithm=algorithm)

    return individuals
