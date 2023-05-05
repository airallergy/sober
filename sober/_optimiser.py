from typing import Literal, TypedDict, overload

from . import _pymoo_namespace as pm


#############################################################################
#######                      OPERATOR FUNCTIONS                       #######
#############################################################################
class Operators(TypedDict):
    sampling: pm.Population
    mating: pm.MixedVariableMating
    eliminate_duplicates: pm.MixedVariableDuplicateElimination


def _sampling(problem: pm.Problem, init_population_size: int) -> pm.Population:
    """samples the initial generation"""

    return pm.MixedVariableSampling()(problem, init_population_size)


def _operators(
    algorithm_name: Literal["nsga2", "nsga3"],
    p_crossover: float,
    p_mutation: float,
    sampling: pm.Population,
) -> Operators:
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
) -> pm.Algorithm:
    ...


@overload
def _algorithm(
    algorithm_name: Literal["nsga3"],
    population_size: int,
    p_crossover: float,
    p_mutation: float,
    sampling: pm.Population,
    reference_directions: pm.ReferenceDirectionFactory,
) -> pm.Algorithm:
    ...


def _algorithm(
    algorithm_name,
    population_size,
    p_crossover,
    p_mutation,
    sampling,
    reference_directions=None,
) -> pm.Algorithm:
    """a pymoo algorithm constructor"""

    if algorithm_name == "nsga2":
        return pm.NSGA2(
            population_size,
            **_operators(algorithm_name, p_crossover, p_mutation, sampling)
        )
    else:
        return pm.NSGA3(
            reference_directions,
            population_size,
            **_operators(algorithm_name, p_crossover, p_mutation, sampling)
        )
