from sober._pymoo_namespace import MaximumGenerationTermination
from sober.config import config_energyplus, config_parallel, config_script
from sober.input import (
    CategoricalModifier,
    ContinuousModifier,
    DiscreteModifier,
    FunctionalModifier,
    IndexTagger,
    StringTagger,
    WeatherModifier,
)
from sober.output import RVICollector, ScriptCollector
from sober.problem import Problem

__all__ = (
    "Problem",
    "RVICollector",
    "ScriptCollector",
    "MaximumGenerationTermination",
    "config_script",
    "config_parallel",
    "config_energyplus",
    "IndexTagger",
    "StringTagger",
    "WeatherModifier",
    "DiscreteModifier",
    "ContinuousModifier",
    "FunctionalModifier",
    "CategoricalModifier",
)

"""nomenclature
- variation/uncertainty: alternative values for individual inputs at definition
- candidate/scenario: one realisation of variation/uncertainty of a given input
- candidate_vec/scenario_vec: one realisation of variation/uncertainty of all inputs in order
- duo: a pair of variation and uncertainty or candidate and scenario
- trio: a collection of three items to locate model inputs
"""
# NOTE: variation/uncertainty and candidate/scenario
#           are equivalent on the scalar level, but differ in domain
#           hence duo is a pair of either
