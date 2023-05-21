from .problem import Problem
from .results import RVICollector, ScriptCollector
from ._pymoo_namespace import MaximumGenerationTermination
from .config import config_script, config_parallel, config_energyplus
from .parameters import (
    IndexTagger,
    StringTagger,
    WeatherModifier,
    DiscreteModifier,
    ContinuousModifier,
    FunctionalModifier,
    CategoricalModifier,
)

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
- variation/uncertainty: alternative values for individual parameters at definition
- candidate/scenario: one realisation of variation/uncertainty of a given parameter
- candidate_vec/scenario_vec: one realisation of variation/uncertainty of all parameters in order
- duo: a pair of variation and uncertainty or candidate and scenario
- trio: a collection of three items to locate model parameters
"""
# NOTE: variation/uncertainty and candidate/scenario
#           are equivalent on the scalar level, but differ in domain
#           hence duo is a pair of either
