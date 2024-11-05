from sober._evolver_pymoo import MaximumGenerationTermination
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

__version__ = "0.1.0"

__all__ = (
    "CategoricalModifier",
    "ContinuousModifier",
    "DiscreteModifier",
    "FunctionalModifier",
    "IndexTagger",
    "MaximumGenerationTermination",
    "Problem",
    "RVICollector",
    "ScriptCollector",
    "StringTagger",
    "WeatherModifier",
    "config_energyplus",
    "config_parallel",
    "config_script",
)

# TODO: check all raises and asserts, and unify error messages
# TODO: py3.12: python/mypy#14072, python/mypy#15238
