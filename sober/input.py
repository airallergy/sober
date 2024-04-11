from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, final

from eppy.bunchhelpers import makefieldname

from sober._tools import _uuid
from sober._typing import AnyModelModifierVal, AnyModifierVal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from typing import Concatenate, Protocol, Self, TypeAlias

    from eppy.modeleditor import IDF

    from sober._typing import AnyStrPath

    class _SupportsStr(Protocol):
        __slots__ = ()

        @abstractmethod
        def __str__(self) -> str: ...

    _AnyFunc: TypeAlias = Callable[
        Concatenate[tuple[AnyModifierVal, ...], ...], AnyModelModifierVal
    ]  # TODO: consider resticting at least one previous output

##############################  module typing  ##############################
# https://github.com/python/typing/issues/60#issuecomment-869757075
# this can be removed with the new type syntax from py3.12
_MK = TypeVar("_MK", float, int)  # AnyModifierKey
_MV = TypeVar("_MV", bound=AnyModifierVal)  # AnyModifierValue
#############################################################################


@final
class _Noise(Any):
    """a helper class for _hype_ctrl_val"""

    _s: str

    __slots__ = ("_s",)

    def __new__(cls, s: str) -> Self:
        self = super().__new__(cls)
        self._s = s
        return self  # type: ignore[no-any-return]

    def __str__(self) -> str:
        """controls csv.writer"""
        return f"<noise {self._s}>"


#############################################################################
#######                     ABSTRACT BASE CLASSES                     #######
#############################################################################
class _Tagger(ABC):
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

    def _detagged(self, tagged_model: str, *values: _SupportsStr) -> str:
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

    __slots__ = ()

    @abstractmethod
    def _tagged(self, model: IDF) -> IDF: ...


class _TextTagger(_Tagger):
    """an abstract base class for taggers in the text format"""

    __slots__ = ()

    @abstractmethod
    def _tagged(self, model: str) -> str: ...


class _Modifier(ABC, Generic[_MK, _MV]):
    """an abstract base class for input modifiers"""

    _bounds: tuple[float, float]
    _is_ctrl: bool
    _is_noise: bool
    _name: str
    _index: int
    _label: str

    __slots__ = ("_bounds", "_is_ctrl", "_is_noise", "_name", "_index", "_label")

    @abstractmethod
    def __init__(self, bounds: tuple[float, float], is_noise: bool, name: str) -> None:
        self._bounds = bounds
        self._is_ctrl = not is_noise  # FunctionalModifier is neither, overwrites later
        self._is_noise = is_noise
        self._name = name

    @abstractmethod
    def __call__(self, key: _MK) -> _MV: ...

    @abstractmethod
    def _check_args(self) -> None:
        # called by _InputManager
        # as FunctionalModifier needs the index info assigned by _InputManager
        ...

    def _hype_ctrl_key(self) -> _MK:
        assert not self._is_ctrl
        return 0  # assuming the hype ctrl is an integral variable with one item

    @abstractmethod
    def _hype_ctrl_val(self) -> _MV: ...

    def _hype_ctrl_len(self) -> int:
        assert not self._is_ctrl
        return 1  # assuming the hype ctrl is an integral variable with one item


class _RealModifier(_Modifier[float, float]):
    """an abstract base class for input modifiers of real variables"""

    @abstractmethod
    def __init__(self, bounds: tuple[float, float], is_noise: bool, name: str) -> None:
        super().__init__(bounds, is_noise, name)

    def __call__(self, key: float) -> float:
        return key

    def _hype_ctrl_val(self) -> float:
        assert not self._is_ctrl
        return _Noise("(...)")


class _IntegralModifier(_Modifier[int, _MV]):
    """an abstract base class for input modifiers of integral variables"""

    _options: tuple[_MV, ...]

    __slots__ = ("_options",)

    @abstractmethod
    def __init__(self, options: tuple[_MV, ...], is_noise: bool, name: str) -> None:
        self._options = options

        bounds = (0, len(self) - 1)
        super().__init__(bounds, is_noise, name)

    def __iter__(self) -> Iterator[_MV]:
        yield from self._options

    def __len__(self) -> int:
        return len(self._options)

    def __getitem__(self, key: int) -> _MV:
        return self._options[key]

    def __call__(self, key: int) -> _MV:
        return self[key]

    def _hype_ctrl_val(self) -> _MV:
        # FunctionalModifier overwrites later
        assert not self._is_ctrl
        return _Noise("{...}")


class _ModelModifierMixin(ABC):
    """an abstract base class for common functions in model modification
    (as opposed to the weather modifier)"""

    _tagger: _Tagger
    __slots__ = ()  # [1] '_tagger' included in child classes' __slots__ to make mixin work

    @abstractmethod
    def __init__(self, tagger: _Tagger, *args, **kwargs) -> None:
        self._tagger = tagger  # type: ignore[misc]  # [1] microsoft/pyright#2039

        super().__init__(*args, **kwargs)  # NOTE: to _RealModifier/_IntegralModifier

    def _detagged(self, tagged_model: str, *values: _SupportsStr) -> str:
        return self._tagger._detagged(tagged_model, *values)


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

        # cast: python/mypy#4573, python/mypy#7853, below works already
        # >>> x = _index_trios[0]
        # >>> assert len(x) == 3
        # >>> reveal_type(x)  # x: tuple[str, str, str]
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

        # cast: python/mypy#4573, python/mypy#7853 may help, but the above assignment is too dynamic
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
class WeatherModifier(_IntegralModifier[Path]):
    """modifies the weather input"""

    __slots__ = ()

    def __init__(
        self, *options: AnyStrPath, is_noise: bool = False, name: str = ""
    ) -> None:
        super().__init__(tuple(Path(item) for item in options), is_noise, name)

    def _check_args(self) -> None:
        for item in self._options:
            # check existence
            item.resolve(True)

            # check suffix
            if item.suffix != ".epw":
                raise ValueError(f"'{item}' is no epw file.")


class ContinuousModifier(_ModelModifierMixin, _RealModifier):
    """modifies continuous inputs"""

    __slots__ = ("_tagger",)

    def __init__(
        self,
        tagger: _Tagger,
        low: float,
        high: float,
        /,
        *,
        is_noise: bool = False,
        name: str = "",
    ) -> None:
        super().__init__(tagger, (low, high), is_noise, name)

    def _check_args(self) -> None:
        low, high = self._bounds
        if low >= high:
            raise ValueError(f"the low '{low}' is not less than the high '{high}'.")


class DiscreteModifier(_ModelModifierMixin, _IntegralModifier[float]):
    """modifies discrete inputs"""

    __slots__ = ("_tagger",)

    def __init__(
        self, tagger: _Tagger, *options: float, is_noise: bool = False, name: str = ""
    ) -> None:
        super().__init__(tagger, options, is_noise, name)

    def _check_args(self) -> None:
        pass


class CategoricalModifier(_ModelModifierMixin, _IntegralModifier[str]):
    """modifies categorical inputs"""

    __slots__ = ("_tagger",)

    def __init__(
        self, tagger: _Tagger, *options: str, is_noise: bool = False, name: str = ""
    ) -> None:
        super().__init__(tagger, options, is_noise, name)

    def _check_args(self) -> None:
        pass


class FunctionalModifier(_ModelModifierMixin, _IntegralModifier[AnyModelModifierVal]):
    """modifies functional inputs"""

    _func: _AnyFunc
    _input_indices: tuple[int, ...]
    _args: tuple[Any, ...]  # TODO: restrict this for serialisation
    _kwargs: dict[str, Any]  # TODO: restrict this for serialisation

    __slots__ = ("_tagger", "_func", "_input_indices", "_args", "_kwargs")

    def __init__(
        self,
        tagger: _Tagger,
        func: _AnyFunc,
        input_indices: Iterable[int],
        *args,
        name: str = "",
        **kwargs,
    ) -> None:
        self._func = func
        self._input_indices = tuple(input_indices)
        self._args = args
        self._kwargs = kwargs

        func_name = f"<function {self._func.__module__ + '.' + self._func.__code__.co_qualname}>"
        super().__init__(tagger, (func_name,), False, name)

        self._is_ctrl = False

    def __call__(self, key: int, *input_vals: AnyModifierVal) -> AnyModelModifierVal:
        return self._func(input_vals, *self._args, **self._kwargs)

    def _check_args(self) -> None:
        if any(item >= self._index for item in self._input_indices):
            raise ValueError(
                f"only previous inputs can be referred to: {self._index}, {self._input_indices}."
            )

    def _hype_ctrl_val(self) -> AnyModelModifierVal:
        # assert not self._is_ctrl  # no need, as hardcoded in __init__
        return self._options[0]
