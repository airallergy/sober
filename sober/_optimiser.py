from typing import Literal, TypedDict, overload

from . import _pymoo_namespace as pm


#############################################################################
#######                      OPERATOR FUNCTIONS                       #######
#############################################################################
class Operators(TypedDict):
    sampling: pm.MixedVariableSampling
    mating: pm.MixedVariableMating
    eliminate_duplicates: pm.MixedVariableDuplicateElimination


def _operators(
    algorithm_name: Literal["nsga2", "nsga3"], p_crossover: float, p_mutation: float
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

    return {
        "sampling": pm.MixedVariableSampling(),
        "mating": pm.MixedVariableMating(
            selection=selections[algorithm_name],
            crossover={
                pm.Real: pm.SimulatedBinaryCrossover(
                    prob=p_crossover, eta=etas[algorithm_name]["crossover"]
                ),
                pm.Integer: pm.SimulatedBinaryCrossover(
                    prob=p_crossover,
                    eta=etas[algorithm_name]["crossover"],
                    vtype=float,
                    repair=pm.RoundingRepair(),
                ),
            },
            mutation={
                pm.Real: pm.PolynomialMutation(
                    prob=p_mutation, eta=etas[algorithm_name]["mutation"]
                ),
                pm.Integer: pm.PolynomialMutation(
                    prob=p_mutation,
                    eta=etas[algorithm_name]["mutation"],
                    vtype=float,
                    repair=pm.RoundingRepair(),
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
) -> pm.Algorithm:
    ...


@overload
def _algorithm(
    algorithm_name: Literal["nsga3"],
    population_size: int,
    p_crossover: float,
    p_mutation: float,
    reference_directions: pm.ReferenceDirectionFactory,
) -> pm.Algorithm:
    ...


def _algorithm(
    algorithm_name,
    population_size,
    p_crossover,
    p_mutation,
    reference_directions=None,
) -> pm.Algorithm:
    """a pymoo algorithm constructor"""

    if algorithm_name == "nsga2":
        return pm.NSGA2(
            population_size, **_operators(algorithm_name, p_crossover, p_mutation)
        )
    else:
        return pm.NSGA3(
            reference_directions,
            population_size,
            **_operators(algorithm_name, p_crossover, p_mutation)
        )
