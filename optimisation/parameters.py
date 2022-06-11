from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, TypeVar
from eppy.modeleditor import IDF
from uuid import uuid5, NAMESPACE_X500
from ._args_check import _checked_empty

from ._tools import DATACLASS_PARAMS

from collections.abc import Sequence

AnyModel = TypeVar("AnyModel", bound=IDF | str)


#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
@dataclass(**DATACLASS_PARAMS)
class _Tagger(ABC):
    tag: str = field(init=False)

    @classmethod
    def _uuid(cls, *descriptions: str) -> str:
        return str(uuid5(NAMESPACE_X500, cls.__name__ + "-".join(descriptions)))

    @abstractmethod
    def _tagged(self, model: AnyModel) -> AnyModel:
        ...


@dataclass(**DATACLASS_PARAMS)
class _Parameter(ABC):
    tagger: _Tagger
    low: float
    high: float


#############################################################################
#######                        TAGGER CLASSES                         #######
#############################################################################
@dataclass(**DATACLASS_PARAMS)
class IndexTagger(_Tagger):
    input_type: ClassVar[str] = "regular"
    class_name: str
    object_name: str
    field_name: str

    def __post_init__(self) -> None:
        self.tag = self._uuid(self.class_name, self.object_name, self.field_name)

    def _tagged(self, model: IDF) -> IDF:
        setattr(
            model.getobject(self.class_name, self.object_name),
            self.field_name,
            self.tag,
        )
        return model


@dataclass(**DATACLASS_PARAMS)
class StringTagger(_Tagger):
    input_type: ClassVar[str] = "macro"
    string: str
    prefix: str = None
    suffix: str = None

    def __post_init__(self) -> None:
        self.prefix, self.suffix = _checked_empty(self.prefix, self.suffix, type=str)
        if (not self.string.startswith(self.prefix)) or (
            not self.string.endswith(self.suffix)
        ):
            raise ValueError("string needs to share the prefix and the suffix.")

        self.tag = self._uuid(self.string)

    def _tagged(self, model: str) -> str:
        return model.replace(self.string, self.prefix + self.tag + self.suffix)


#############################################################################
#######                       PARAMETER CLASSES                       #######
#############################################################################
@dataclass(**DATACLASS_PARAMS)
class ContinuousParameter(_Parameter):
    ...


@dataclass(**DATACLASS_PARAMS)
class DiscreteParameter(_Parameter):
    low: float = field(init=False)
    high: float = field(init=False)
    variations: Sequence[float]
    uncertainties: Sequence[Sequence[float]]


@dataclass(**DATACLASS_PARAMS)
class CategoricalParameter(DiscreteParameter):
    variations: Sequence[str]
    uncertainties: Sequence[Sequence[str]]
