from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast, overload

from eppy.bunchhelpers import makefieldname
from eppy.modeleditor import IDF

from sober._tools import _uuid
from sober._typing import AnyIntegralDuo, AnyRealDuo, AnyStrPath

##############################  module typing  ##############################
_T = TypeVar("_T")  # AnyValueToTag # python/typing#548
_V = TypeVar("_V")  # AnyVariationValue
_U = TypeVar("_U")  # AnyUncertaintyValue
#############################################################################


#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Tagger(ABC, Generic[_T]):
    """an abstract base class for taggers"""

    _tags: tuple[str, ...]

    __slots__ = ("_tags",)

    @abstractmethod
    def __init__(self, *feature_groups: tuple[str, ...]) -> None:
        self._tags = tuple(
            _uuid(self.__class__.__name__, *item) for item in feature_groups
        )

    @abstractmethod
    def _tagged(self, model: Any) -> Any: ...

    def _detagged(self, tagged_model: str, *values: _T) -> str:
        match len(values):
            case 0:
                raise ValueError("no values for detagging.")
            case 1:
                value = values[0]
                for tag in self._tags:
                    tagged_model = tagged_model.replace(tag, str(value))
            case _:
                for tag, value in zip(self._tags, values, strict=True):
                    tagged_model = tagged_model.replace(tag, str(value))

        return tagged_model


class _IDFTagger(_Tagger):
    """an abstract base class for taggers in the IDF format"""

    @abstractmethod
    def _tagged(self, model: IDF) -> IDF: ...


class _TextTagger(_Tagger):
    """an abstract base class for taggers in the text format"""

    @abstractmethod
    def _tagged(self, model: str) -> str: ...


class _Modifier(ABC):
    """an abstract base class for input modifiers"""

    _low: float
    _high: float
    _index: int
    _label: str
    _name: str
    _is_uncertain: bool

    __slots__ = ("_low", "_high", "_index", "_label", "_name", "_is_uncertain")

    @abstractmethod
    def __init__(self, name: str) -> None:
        self._name = name

    def _check_args(self) -> None:
        pass

    @abstractmethod
    def _value(self, duo: Any, *args) -> Any:
        # *args is only used by FunctionalModifier for now

        ...


class _ModelModifierMixin(ABC, Generic[_T]):
    """an abstract base class for common functions in model modification
    (as opposed to the weather modifier)"""

    _tagger: _Tagger[_T]

    # __slots__ = ("_tagger",)  # a slots/mixin issue

    @abstractmethod
    def __init__(self, tagger: _Tagger[_T], *args, **kwargs) -> None:
        self._tagger = tagger

        super().__init__(*args, **kwargs)  # NOTE: to _RealModifier/_IntegralModifier

    def _detagged(self, tagged_model: str, *values: _T) -> str:
        return self._tagger._detagged(tagged_model, *values)


class _RealModifier(_Modifier):
    """an abstract base class for input modifiers of real variables in pymoo"""

    @abstractmethod
    def __init__(self, low: float, high: float, *, name: str) -> None:
        self._low = low
        self._high = high
        self._is_uncertain = False  # NOTE: hard-coded for now

        super().__init__(name)


class _IntegralModifier(_Modifier, Generic[_V, _U]):
    """an abstract base class for input modifiers of integral variables in pymoo"""

    _low: int
    _high: int
    _variations: tuple[_V, ...]
    _uncertainties: tuple[tuple[_U, ...], ...]
    _n_variations: int
    _ns_uncertainties: tuple[int, ...]

    __slots__ = (
        "_low",
        "_high",
        "_variations",
        "_uncertainties",
        "_n_variations",
        "_ns_uncertainties",
    )

    @abstractmethod
    def __init__(
        self, variations: Iterable[_V], *uncertainties: Iterable[_U], name: str
    ) -> None:
        # parse and count variations
        self._variations = tuple(variations)
        self._n_variations = len(self._variations)

        # parse and count uncertainties
        # NOTE: when no uncertainty, the count of uncertainties is 1
        match len(uncertainties):
            case 0:
                # no uncertainty
                # self._uncertainties = ()  # leave undefined
                self._ns_uncertainties = (1,) * self._n_variations
            case self._n_variations:
                # each variation has different uncertainties
                self._uncertainties = tuple(map(tuple, uncertainties))
                self._ns_uncertainties = tuple(map(len, self._uncertainties))
            case 1:
                # each variation has the same uncertainty
                # TODO: this is not useful for absolute uncertainty
                #                              i.e. uncertainty
                #                      but for relative uncertainty in the future
                #                              i.e. variation +- uncertainty
                raise NotImplementedError
                self._uncertainties = (tuple(uncertainties[0]),) * self._n_variations
                self._ns_uncertainties = (
                    len(self._uncertainties[0]),
                ) * self._n_variations
            case _:
                raise ValueError(
                    f"the number of uncertainties is different from that of variations: '{len(uncertainties)}', '{self._n_variations}'."
                )

        self._is_uncertain = set(self._ns_uncertainties) != {1}
        self._low = 0
        self._high = self._n_variations - 1

        super().__init__(name)

    @overload
    def __getitem__(self, index: int) -> _V: ...

    @overload
    def __getitem__(self, index: tuple[int, int]) -> _U: ...

    def __getitem__(self, index):
        match self._is_uncertain, index:
            case _, int() as i:
                return self._variations[i]
            case False, (int() as i, 0):
                return self._variations[i]
            case True, (int() as i, int() as j):
                return self._uncertainties[i][j]
            case False, (int(), int()):
                raise IndexError("no uncertainties defined.")
            case _:
                raise IndexError(f"invalid index: {index}.")


#############################################################################
#######                        TAGGER CLASSES                         #######
#############################################################################
class IndexTagger(_IDFTagger):
    """tags regular commands by indexing

    index_trio == (class_name, object_name, field_name)

    no support for nested regular commands inside macro files
    """

    _index_trios: tuple[tuple[str, str, str], ...]

    __slots__ = ("_index_trios",)

    def __init__(self, /, *index_trios: Iterable[str]) -> None:
        _index_trios = tuple(tuple(item) for item in index_trios)  # python/mypy#11682

        if any(len(item) != 3 for item in _index_trios):
            raise ValueError(
                "each index trio should contain exactly three elements: class_name, object_name, field_name."
            )

        # TODO: cast may be removed after python/mypy#1178
        _index_trios = cast(tuple[tuple[str, str, str], ...], _index_trios)

        # remove duplicate trios
        self._index_trios = tuple(set(_index_trios))

        super().__init__(*self._index_trios)

    def _tagged(self, model: IDF) -> IDF:
        for (class_name, object_name, field_name), tag in zip(
            self._index_trios, self._tags, strict=True
        ):
            obj = model.getobject(class_name, object_name)
            if obj is None:
                raise ValueError(f"object is not found in the model: '{object_name}'.")
                # eppy throws a proper error for unknown field names

            obj[makefieldname(field_name)] = tag
        return model


class StringTagger(_TextTagger):
    """tags regular and macro commands by string replacement

    string_trio == (string, prefix, suffix)

    no support for nested macro commands inside macro files
    """

    _string_trios: tuple[tuple[str, str, str], ...]

    __slots__ = ("_string_trios",)

    def __init__(self, /, *string_trios: Iterable[str]) -> None:
        _string_trios = tuple(tuple(item) for item in string_trios)  # python/mypy#11682

        if any(len(item) == 0 for item in _string_trios):
            raise ValueError(
                "each string trio should contain at least one element: string."
            )
        if any(len(item) > 3 for item in _string_trios):
            raise ValueError(
                "each string trio should contain at most three elements: string, prefix, suffix."
            )

        # assign empty string to prefix/suffix if absent
        _string_trios = tuple(item + ("",) * (3 - len(item)) for item in _string_trios)

        # TODO: cast may be removed after python/mypy#1178
        _string_trios = cast(tuple[tuple[str, str, str], ...], _string_trios)

        for string, prefix, suffix in _string_trios:
            if not (string.startswith(prefix) and string.endswith(suffix)):
                raise ValueError(
                    f"string needs to share the prefix and the suffix: '{string}, {prefix}, {suffix}'."
                )

        # remove duplicate trios
        self._string_trios = tuple(set(_string_trios))

        super().__init__(*self._string_trios)

    def _tagged(self, model: str) -> str:
        for (string, prefix, suffix), tag in zip(
            self._string_trios, self._tags, strict=True
        ):
            if string not in model:
                raise ValueError(f"string is not found in the model: '{string}'.")

            model = model.replace(string, prefix + tag + suffix)
        return model


#############################################################################
#######                       MODIFIER CLASSES                        #######
#############################################################################
class WeatherModifier(_IntegralModifier[Path | str, Path]):
    """modifies the weather input"""

    _variations: tuple[Path | str, ...]
    _uncertainties: tuple[tuple[Path, ...], ...]

    __slots__ = ("_variations", "_uncertainties")

    @overload
    def __init__(self, variations: Iterable[AnyStrPath], /, *, name: str) -> None: ...

    @overload
    def __init__(
        self,
        variations: Iterable[str],
        /,
        *uncertainties: Iterable[AnyStrPath],
        name: str,
    ) -> None: ...

    def __init__(self, variations, /, *uncertainties, name=""):
        super().__init__(variations, *uncertainties, name=name)

        # resolve weather file paths
        if self._is_uncertain:
            self._variations = tuple(str(item) for item in self._variations)
            self._uncertainties = tuple(
                tuple(Path(item).resolve(True) for item in uncertainty)
                for uncertainty in self._uncertainties
            )
        else:
            self._variations = tuple(
                Path(item).resolve(True) for item in self._variations
            )
            assert not hasattr(self, "_uncertainties")

    def _value(self, duo: AnyIntegralDuo, *args) -> Path:
        return self[duo]


class ContinuousModifier(_ModelModifierMixin[float], _RealModifier):
    """modifies continuous inputs"""

    def __init__(
        self, tagger: _Tagger, low: float, high: float, /, *, name: str = ""
    ) -> None:
        # TODO: uncertainty of continuous inputs
        super().__init__(tagger, low, high, name=name)

    def _value(self, duo: AnyRealDuo, *args) -> float:
        return duo[0]

    def _detagged(self, tagged_model: str, *values: float) -> str:
        return super()._detagged(tagged_model, *values)


class DiscreteModifier(_ModelModifierMixin[float], _IntegralModifier[float, float]):
    """modifies discrete inputs"""

    _variations: tuple[float, ...]
    _uncertainties: tuple[tuple[float, ...], ...]

    __slots__ = ("_variations", "_uncertainties")

    def __init__(
        self,
        tagger: _Tagger,
        variations: Iterable[float],
        /,
        *uncertainties: Iterable[float],
        name: str = "",
    ) -> None:
        super().__init__(tagger, variations, *uncertainties, name=name)

    def _value(self, duo: AnyIntegralDuo, *args) -> float:
        return self[duo]

    def _detagged(self, tagged_model: str, *values: float) -> str:
        return super()._detagged(tagged_model, *values)


class CategoricalModifier(_ModelModifierMixin[str], _IntegralModifier[str, str]):
    """modifies categorical inputs"""

    _variations: tuple[str, ...]
    _uncertainties: tuple[tuple[str, ...], ...]

    __slots__ = ("_variations", "_uncertainties")

    def __init__(
        self,
        tagger: _Tagger,
        variations: Iterable[str],
        /,
        *uncertainties: Iterable[str],
        name: str = "",
    ) -> None:
        super().__init__(tagger, variations, *uncertainties, name=name)

    def _value(self, duo: AnyIntegralDuo, *args) -> str:
        return self[duo]

    def _detagged(self, tagged_model: str, *values: str) -> str:
        return super()._detagged(tagged_model, *values)


class FunctionalModifier(_ModelModifierMixin[_T], _IntegralModifier[_V, _U]):
    """modifies functional inputs"""

    _func: Callable[..., _V | Iterable[_V]]
    _input_indices: tuple[int, ...]
    _extra_args: tuple[Any, ...]
    _is_scalar: bool

    __slots__ = ("_func", "_input_indices", "_extra_args", "_is_scalar")

    @overload
    def __init__(
        self,
        tagger: _Tagger,
        func: Callable[..., _V],
        input_indices: Iterable[int],
        /,
        *extra_args: Any,  # TODO: restrict this for serialisation
        is_scalar: Literal[True],
        name: str,
    ) -> None: ...

    @overload
    def __init__(
        self,
        tagger: _Tagger,
        func: Callable[..., Iterable[_V]],
        input_indices: Iterable[int],
        /,
        *extra_args: Any,
        is_scalar: Literal[False],
        name: str,
    ) -> None: ...

    def __init__(
        self, tagger, func, input_indices, /, *extra_args, is_scalar=True, name=""
    ):
        func_name = f"<function {func.__module__ + '.' + func.__code__.co_qualname}>"
        super().__init__(tagger, (func_name,), name=name)

        self._func = func
        self._input_indices = tuple(input_indices)
        self._extra_args = extra_args
        self._is_scalar = is_scalar

    def _check_args(self) -> None:
        super()._check_args()

        if any(item >= self._index for item in self._input_indices):
            raise ValueError(
                f"only previous inputs can be referred to: {self._index}, {self._input_indices}."
            )

    def _value(self, duo: AnyIntegralDuo, *args) -> _V | tuple[_V, ...]:
        # NOTE: args are values for referenced inputs
        #       args are passed as tuple whilst _extra_args unpacked
        #       this is to align their corresponding arguments in __init__

        if self._is_scalar:
            self._func = cast(Callable[..., _V], self._func)

            return self._func(args, *self._extra_args)

        else:
            self._func = cast(Callable[..., Iterable[_V]], self._func)

            return tuple(self._func(args, *self._extra_args))

    def _detagged(self, tagged_model: str, *values: _T) -> str:
        return super()._detagged(tagged_model, *values)


#############################  package typing  ##############################
# this technically belongs to _typing.py, but put here to avoid circular import
AnyIntegralModelModifier: TypeAlias = (
    DiscreteModifier | CategoricalModifier | FunctionalModifier
)
AnyRealModelModifier: TypeAlias = ContinuousModifier
AnyModelModifier: TypeAlias = AnyRealModelModifier | AnyIntegralModelModifier
# this TypeVar is defined this way
# to differ an input manager with mixed input types from one that only has integers
ModelModifier = TypeVar("ModelModifier", AnyModelModifier, AnyIntegralModelModifier)
#############################################################################
