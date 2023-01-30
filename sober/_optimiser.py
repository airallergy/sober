from typing import Literal, TypedDict, overload

from . import _pymoo_namespace as pm

#############################################################################
#######                      OPERATOR FUNCTIONS                       #######
#############################################################################
Operators = TypedDict(
    "Operators",
    {
        "sampling": pm.MixedVariableSampling,
        "mating": pm.MixedVariableMating,
        "eliminate_duplicates": pm.MixedVariableDuplicateElimination,
    },
)


def _operators(p_crossover: float, p_mutation: float) -> Operators:
    return {
        "sampling": pm.MixedVariableSampling(),
        "mating": pm.MixedVariableMating(
            crossover={
                pm.Real: pm.SimulatedBinaryCrossover(prob=p_crossover),
                pm.Integer: pm.SimulatedBinaryCrossover(
                    prob=p_crossover, vtype=float, repair=pm.RoundingRepair()
                ),
            },
            mutation={
                pm.Real: pm.PolynomialMutation(prob=p_mutation),
                pm.Integer: pm.PolynomialMutation(
                    prob=p_mutation, vtype=float, repair=pm.RoundingRepair()
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
    algorithm: Literal["nsga2"],
    population_size: int,
    p_crossover: float,
    p_mutation: float,
) -> pm.Algorithm:
    ...


@overload
def _algorithm(
    algorithm: Literal["nsga3"],
    population_size: int,
    p_crossover: float,
    p_mutation: float,
    reference_directions: pm.ReferenceDirectionFactory,
) -> pm.Algorithm:
    ...


def _algorithm(
    algorithm,
    population_size,
    p_crossover,
    p_mutation,
    reference_directions=None,
) -> pm.Algorithm:
    if algorithm == "nsga2":
        return pm.NSGA2(population_size, **_operators(p_crossover, p_mutation))
    else:
        return pm.NSGA3(
            reference_directions, population_size, **_operators(p_crossover, p_mutation)
        )
