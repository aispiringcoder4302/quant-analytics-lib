# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for working with class and instance attributes."""

import enum
import inspect
import re
from collections.abc import Iterable
from functools import cached_property as cachedproperty

import attr
import pandas as pd
from attr.exceptions import NotAnAttrsClassError

import vectorbtpro as vbt
from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.decorators import hybrid_property, hybrid_method
from vectorbtpro.utils.hashing import Hashable

__all__ = [
    "MISSING",
    "DefineMixin",
    "define",
    "deep_getattr",
]


class _Missing(enum.Enum):
    """Class representing a sentinel for missing values."""

    MISSING = enum.auto()

    def __repr__(self):
        return "MISSING"

    def __bool__(self):
        return False


MISSING = _Missing.MISSING
"""Sentinel that represents a missing value."""

DefineMixinT = tp.TypeVar("DefineMixinT", bound="DefineMixin")


class DefineMixin(Hashable):
    """Mixin class for `define` that provides attribute field management capabilities for attrs-decorated classes.

    Args:
        *args: Positional arguments for the underlying initializer.
        **kwargs: Keyword arguments for the underlying initializer.
    """

    def __init__(self, *args, **kwargs) -> None:
        if not attr.has(type(self)):
            msg = f"{type(self)!r} is not an attrs-decorated class."
            raise NotAnAttrsClassError(msg)

        self.__attrs_init__(*args, **kwargs)

    @hybrid_property
    def fields(cls_or_self) -> tp.Optional[tp.Tuple[attr.Attribute]]:
        """Return a tuple of attribute fields for the class or instance.

        Returns:
            Optional[Tuple[Attribute]]: Tuple of attribute fields if available, otherwise None.
        """
        if isinstance(cls_or_self, type):
            cls = cls_or_self
            if not attr.has(cls):
                return None
        else:
            cls = type(cls_or_self)
        return attr.fields(cls)

    @hybrid_property
    def fields_dict(cls_or_self) -> tp.Optional[tp.Dict[str, attr.Attribute]]:
        """Return a dictionary mapping field names to attribute fields for the class or instance.

        Returns:
            Optional[Dict[str, Attribute]]: Dictionary of attribute fields if available, otherwise None.
        """
        if isinstance(cls_or_self, type):
            cls = cls_or_self
            if not attr.has(cls):
                return None
        else:
            cls = type(cls_or_self)
        return attr.fields_dict(cls)

    @hybrid_method
    def get_field(cls_or_self, field_name: str) -> attr.Attribute:
        """Return the attribute field corresponding to the specified field name.

        Args:
            field_name (str): Name of the attribute field.

        Returns:
            Attribute: The attribute field object.
        """
        return cls_or_self.fields_dict[field_name]

    @hybrid_method
    def is_field_required(cls_or_self, field_or_name: tp.Union[str, attr.Attribute]) -> bool:
        """Determine if the specified field is required.

        Args:
            field_or_name (Union[str, Attribute]): Name or object of the attribute field.

        Returns:
            bool: True if the field is required, False otherwise.
        """
        if isinstance(field_or_name, str):
            field = cls_or_self.get_field(field_or_name)
        else:
            field = field_or_name
        return field.default is MISSING and "default" not in field.metadata

    @hybrid_method
    def is_field_optional(cls_or_self, field_or_name: tp.Union[str, attr.Attribute]) -> bool:
        """Determine if the specified field is optional.

        Args:
            field_or_name (Union[str, Attribute]): Name or object of the attribute field.

        Returns:
            bool: True if the field is optional, False otherwise.
        """
        if isinstance(field_or_name, str):
            field = cls_or_self.get_field(field_or_name)
        else:
            field = field_or_name
        return field.default is MISSING and "default" in field.metadata

    def resolve_field(self, field_or_name: tp.Union[str, attr.Attribute]) -> tp.Any:
        """Resolve the value of the specified field for the instance.

        Checks the instance attribute and, if unset (i.e. equal to `MISSING`),
        returns the field's default value or the default specified in metadata if available.

        Args:
            field_or_name (Union[str, Attribute]): Name or object of the attribute field.

        Returns:
            Any: The resolved value of the field, or `MISSING` if no value is available.
        """
        if isinstance(field_or_name, str):
            if getattr(self, field_or_name) is not MISSING:
                return getattr(self, field_or_name)
            field = self.get_field(field_or_name)
        else:
            if getattr(self, field_or_name.name) is not MISSING:
                return getattr(self, field_or_name.name)
            field = field_or_name
        if field.default is not MISSING:
            return field.default
        return field.metadata.get("default", MISSING)

    def is_field_missing(self, field_or_name: tp.Union[str, attr.Attribute]) -> bool:
        """Determine if the specified field is missing.

        Args:
            field_or_name (Union[str, Attribute]): Name or object of the attribute field.

        Returns:
            bool: True if the field is missing, False otherwise.
        """
        return self.resolve_field(field_or_name) is MISSING

    def assert_field_not_missing(self, field_or_name: tp.Union[str, attr.Attribute]) -> None:
        """Assert that the specified field is not missing.

        If the field is missing, raises a ValueError indicating whether it is required or optional.

        Args:
            field_or_name (Union[str, Attribute]): Name or object of the attribute field.

        Returns:
            None
        """
        if isinstance(field_or_name, str):
            field = self.get_field(field_or_name)
        else:
            field = field_or_name
        if self.is_field_missing(field):
            if self.is_field_required(field):
                raise ValueError(f"Required field '{type(self).__name__}.{field.name}' is missing")
            elif self.is_field_optional(field):
                raise ValueError(f"Optional field '{type(self).__name__}.{field.name}' is missing")
            else:
                raise ValueError(f"Field '{type(self).__name__}.{field.name}' is missing")

    def resolve(self: DefineMixinT, assert_not_missing: bool = True) -> DefineMixinT:
        """Resolve all attribute fields for the instance.

        For each field, verifies that a value is present (if `assert_not_missing` is True)
        and replaces missing values using default values.

        Args:
            assert_not_missing (bool): If True, asserts that each field is not missing.

        Returns:
            DefineMixin: A new instance with all field values resolved.
        """
        changes = {}
        for field in self.fields:
            if assert_not_missing:
                self.assert_field_not_missing(field)
            field_value = self.resolve_field(field)
            if field_value is MISSING:
                raise ValueError(f"Field '{type(self).__name__}.{field.name}' is missing")
            changes[field.name] = field_value
        return self.replace(**changes)

    def asdict(self, full: bool = True) -> dict:
        """Convert the instance to a dictionary representation.

        If `full` is False, fields with a value of `MISSING` are omitted.

        Args:
            full (bool): If True, include all fields; otherwise, exclude fields with a missing value.

        Returns:
            dict: A dictionary mapping field names to their corresponding values.
        """
        dct = dict()
        for field in self.fields:
            k = field.name
            v = getattr(self, k)
            if full or v is not MISSING:
                dct[k] = v
        return dct

    def replace(self: DefineMixinT, **changes) -> DefineMixinT:
        """Return a new instance with updated attribute values.

        Args:
            **changes: Keyword arguments specifying the fields to update.

        Returns:
            DefineMixin: A new instance reflecting the specified changes.
        """
        return attr.evolve(self, **changes)

    def merge_with(self: DefineMixinT, other: DefineMixinT, **changes) -> DefineMixinT:
        """Return a new instance resulting from merging attribute values from this instance and another.

        In cases of overlapping fields, values from this instance and provided changes take precedence.

        Args:
            other (DefineMixin): Other instance to merge with.
            **changes: Keyword arguments specifying field updates.

        Returns:
            DefineMixin: A new instance with merged attribute values.
        """
        from vectorbtpro.utils.config import merge_dicts

        return self.replace(**merge_dicts(self.asdict(full=False), other.asdict(full=False), changes))

    def merge_over(self: DefineMixinT, other: DefineMixinT, **changes) -> DefineMixinT:
        """Return a new instance obtained by merging this instance over another.

        In cases of overlapping fields, values from this instance and provided changes override
        those from the other instance.

        Args:
            other (DefineMixin): Other instance to merge over.
            **changes: Keyword arguments specifying field updates.

        Returns:
            DefineMixin: A new instance with merged attribute values.
        """
        from vectorbtpro.utils.config import merge_dicts

        return self.replace(**merge_dicts(other.asdict(full=False), self.asdict(full=False), changes))

    def __repr__(self):
        dct = self.asdict(full=False)
        fields_dict = self.fields_dict
        attr_dct = {k: attr.field(repr=fields_dict[k].repr) for k, v in dct.items()}
        new_cls = attr.make_class(type(self).__name__, attr_dct, cmp=False)
        new_obj = new_cls(**dct)
        return new_obj.__repr__()

    @property
    def hash_key(self) -> tuple:
        return tuple(self.asdict().items())


class define(Base):
    """Class for preparing a class with `define`.

    Attaches `DefineMixin` as a base class and applies `attr.define` to configure classes as
    frozen with customized initialization.
    """

    @classmethod
    def field(cls, **kwargs) -> tp.Any:
        """Alias for `attr.field`.

        Args:
            **kwargs: Keyword arguments for `attr.field`.

        Returns:
            Any: The result from `attr.field`.
        """
        return attr.field(**kwargs)

    @classmethod
    def required_field(cls, **kwargs) -> tp.Any:
        """Return a field definition using `attr.field` with `MISSING` as the default value and
        without a default in metadata.

        Args:
            **kwargs: Keyword arguments for `attr.field`.

        Returns:
            Any: The defined field.
        """
        return cls.field(default=MISSING, **kwargs)

    @classmethod
    def optional_field(
        cls,
        *,
        default: tp.Any = MISSING,
        metadata: tp.Optional[tp.Mapping] = None,
        **kwargs,
    ) -> tp.Any:
        """Return a field definition using `attr.field` with `MISSING` as the default value and
        with the provided default stored in metadata.

        Args:
            default (Any): Default value stored in metadata.
            metadata (Optional[Mapping]): Additional metadata for the field.
            **kwargs: Keyword arguments for `attr.field`.

        Returns:
            Any: The defined field.
        """
        if metadata is None:
            metadata = {}
        else:
            metadata = dict(metadata)
        metadata["default"] = default
        return cls.field(default=MISSING, metadata=metadata, **kwargs)

    def __new__(cls, *args, **kwargs) -> tp.FlexClassWrapper:
        def wrapper(wrapped_cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
            if DefineMixin not in wrapped_cls.__bases__:
                raise TypeError("DefineMixin missing among base classes")
            return attr.define(frozen=True, slots=False, init=False, eq=False, repr=False, **kwargs)(wrapped_cls)

        if len(args) == 0:
            return wrapper
        elif len(args) == 1:
            return wrapper(args[0])
        raise ValueError("Either class or keyword arguments must be passed")


def get_dict_attr(obj: tp.Union[object, type], attr: str) -> tp.Any:
    """Retrieve an attribute directly from an object's dictionary without triggering dynamic attribute lookup.

    Args:
        obj (Union[object, type]): Object or class to inspect.
        attr (str): Name of the attribute to retrieve.

    Returns:
        Any: The attribute value if present.
    """
    if inspect.isclass(obj):
        cls = obj
    else:
        cls = type(obj)
    for obj in [obj] + cls.mro():
        if attr in obj.__dict__:
            return obj.__dict__[attr]
    raise AttributeError


def default_getattr_func(
    obj: tp.Any,
    attr: str,
    args: tp.Optional[tp.Args] = None,
    kwargs: tp.KwargsLike = None,
    call_attr: bool = True,
) -> tp.Any:
    """Retrieve an attribute from an object using default behavior.

    Args:
        obj (Any): Object from which to retrieve the attribute.
        attr (str): Attribute name to access.
        args (Args): Positional arguments passed for calling a callable attribute.
        kwargs (KwargsLike): Keyword arguments passed for calling a callable attribute.
        call_attr (bool): Indicates whether to call the attribute if it is callable.

    Returns:
        Any: The attribute's value or the result of calling it if callable.
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    out = getattr(obj, attr)
    if callable(out) and call_attr:
        return out(*args, **kwargs)
    return out


def deep_getattr(
    obj: tp.Any,
    attr_chain: tp.Union[str, tuple, Iterable],
    getattr_func: tp.Callable = default_getattr_func,
    call_last_attr: bool = True,
) -> tp.Any:
    """Retrieve attributes from an object following a specified chain.

    Args:
        obj (Any): Root object from which to retrieve attributes.
        attr_chain (Union[str, tuple, Iterable]): Chain of attributes to access.

            It can be specified as one of the following forms:

            * string: Retrieve a property or method without arguments.
            * tuple of string: Call a method without arguments.
            * tuple of string and tuple: Call a method with positional arguments.
            * tuple of string, tuple, and dict: Call a method with positional and keyword arguments.
            * Iterable: A sequence of any of the above forms.
        getattr_func (Callable): Function used for attribute retrieval, overriding default behavior
            (`default_getattr_func`).
        call_last_attr (bool): Indicates whether to call the final attribute if it is callable.

    Returns:
        Any: The resulting attribute value or the result from calling the attribute.

    !!! hint
        If your attribute chain contains only attributes and methods without arguments,
        represent it as a single string.
    """
    checks.assert_instance_of(attr_chain, (str, tuple, Iterable))

    if isinstance(attr_chain, str):
        if "." in attr_chain:
            return deep_getattr(obj, attr_chain.split("."), getattr_func=getattr_func, call_last_attr=call_last_attr)
        outer = re.compile(r"(\w+)\((.*)\)")
        match = outer.match(attr_chain)
        if isinstance(attr_chain, str) and match:
            args = ()
            kwargs = dict()
            for arg in match.group(2).split(","):
                arg = arg.strip()
                if len(arg) == 0:
                    continue
                if "=" in arg:
                    kwargs[arg.split("=")[0]] = eval(arg.split("=")[1])
                else:
                    args += (eval(arg),)
            return deep_getattr(
                obj,
                (match.group(1), args, kwargs),
                getattr_func=getattr_func,
                call_last_attr=call_last_attr,
            )
        return getattr_func(obj, attr_chain, call_attr=call_last_attr)
    if isinstance(attr_chain, tuple):
        if len(attr_chain) == 1 and isinstance(attr_chain[0], str):
            return getattr_func(obj, attr_chain[0])
        if len(attr_chain) == 2 and isinstance(attr_chain[0], str) and isinstance(attr_chain[1], tuple):
            return getattr_func(obj, attr_chain[0], args=attr_chain[1])
        if (
            len(attr_chain) == 3
            and isinstance(attr_chain[0], str)
            and isinstance(attr_chain[1], tuple)
            and isinstance(attr_chain[2], dict)
        ):
            return getattr_func(obj, attr_chain[0], args=attr_chain[1], kwargs=attr_chain[2])
    result = obj
    for i, attr in enumerate(attr_chain):
        if i < len(attr_chain) - 1:
            result = deep_getattr(result, attr, getattr_func=getattr_func, call_last_attr=True)
        else:
            result = deep_getattr(result, attr, getattr_func=getattr_func, call_last_attr=call_last_attr)
    return result


AttrResolverMixinT = tp.TypeVar("AttrResolverMixinT", bound="AttrResolverMixin")


class AttrResolverMixin(Base):
    """Class for resolving attributes of an object via getattr.

    This mixin applies resolution logic to self, properties, and methods using built-in caching.
    """

    @property
    def self_aliases(self) -> tp.Set[str]:
        """Set of alias names associated with the object.

        Returns:
            Set[str]: A set of alias names.
        """
        return {"self"}

    def resolve_self(
        self: AttrResolverMixinT,
        cond_kwargs: tp.KwargsLike = None,
        custom_arg_names: tp.Optional[tp.Set[str]] = None,
        impacts_caching: bool = True,
        silence_warnings: bool = False,
    ) -> AttrResolverMixinT:
        """Resolve self with optional keyword conditions.

        Args:
            cond_kwargs (KwargsLike): Keyword arguments for conditional resolution.
            custom_arg_names (Optional[Set[str]]): Set of custom argument names affecting caching.
            impacts_caching (bool): Flag indicating whether resolution impacts caching.
            silence_warnings (bool): Flag to suppress warning messages.

        Returns:
            AttrResolverMixin: The resolved self.

        !!! note
            `cond_kwargs` can be modified in-place.
        """
        return self

    def pre_resolve_attr(self, attr: str, final_kwargs: tp.KwargsLike = None) -> str:
        """Pre-process an attribute before resolution.

        Args:
            attr (str): Name of the attribute to process.
            final_kwargs (KwargsLike): Dictionary of keyword arguments for resolution.

        Returns:
            str: The processed attribute name.
        """
        return attr

    def post_resolve_attr(self, attr: str, out: tp.Any, final_kwargs: tp.KwargsLike = None) -> str:
        """Post-process a resolved attribute output.

        Args:
            attr (str): Original attribute name.
            out (Any): Value obtained from attribute resolution.
            final_kwargs (KwargsLike): Dictionary of keyword arguments used during resolution.

        Returns:
            str: The post-processed output.
        """
        return out

    @cachedproperty
    def cls_dir(self) -> tp.Set[str]:
        """Set of attribute names defined on the object's class.

        Returns:
            Set[str]: A set of attribute names.
        """
        return set(dir(type(self)))

    def resolve_shortcut_attr(self, attr: str, *args, **kwargs) -> tp.Any:
        """Resolve an attribute using potential shortcut properties.

        Args:
            attr (str): Attribute name to resolve.
            *args: Positional arguments passed for resolution.
            **kwargs: Keyword arguments passed for resolution.

        Returns:
            Any: The resolved attribute value.
        """
        if not attr.startswith("get_"):
            if "get_" + attr not in self.cls_dir or (len(args) == 0 and len(kwargs) == 0):
                if isinstance(getattr(type(self), attr), property):
                    return getattr(self, attr)
                return getattr(self, attr)(*args, **kwargs)
            attr = "get_" + attr

        return getattr(self, attr)(*args, **kwargs)

    def resolve_attr(
        self,
        attr: str,
        args: tp.ArgsLike = None,
        cond_kwargs: tp.KwargsLike = None,
        kwargs: tp.KwargsLike = None,
        custom_arg_names: tp.Optional[tp.Container[str]] = None,
        cache_dct: tp.KwargsLike = None,
        use_caching: bool = True,
        passed_kwargs_out: tp.KwargsLike = None,
        use_shortcuts: bool = True,
    ) -> tp.Any:
        """Resolve an attribute using keyword arguments and built-in caching.

        The resolution process follows these steps:

        * If a `get_{arg}` method exists, it is used as the attribute.
        * If the attribute is a property, its value is returned.
        * If the attribute is a method, it is invoked with `*args`, `**kwargs`, and
            `**cond_kwargs` filtered by the method's signature.
        * If `use_shortcuts` is True, `AttrResolverMixin.resolve_shortcut_attr` is applied
            to resolve potential shortcuts.

        Caching is bypassed if `use_caching` is False or if any argument name is present in `custom_arg_names`.

        Use `passed_kwargs_out` to capture the keyword arguments that were passed during resolution.

        Args:
            attr (str): Attribute name to resolve.
            args (ArgsLike): Positional arguments for the attribute.
            cond_kwargs (KwargsLike): Keyword arguments for conditional resolution.
            kwargs (KwargsLike): Keyword arguments for invoking the attribute.
            custom_arg_names (Optional[Container[str]]): Container of argument names affecting caching.
            cache_dct (KwargsLike): Dictionary used for caching resolved attributes.
            use_caching (bool): Flag indicating whether to use caching.
            passed_kwargs_out (KwargsLike): Dictionary to collect the actual keyword arguments passed.
            use_shortcuts (bool): Flag indicating whether to apply shortcut resolution.

        Returns:
            Any: The resolved attribute value.
        """
        from vectorbtpro.utils.config import merge_dicts
        from vectorbtpro.utils.parsing import get_func_arg_names

        if custom_arg_names is None:
            custom_arg_names = list()
        if cache_dct is None:
            cache_dct = {}
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        if passed_kwargs_out is None:
            passed_kwargs_out = {}
        final_kwargs = merge_dicts(cond_kwargs, kwargs)

        cls = type(self)
        _attr = self.pre_resolve_attr(attr, final_kwargs=final_kwargs)
        if "get_" + _attr in dir(cls):
            _attr = "get_" + _attr
        if inspect.ismethod(getattr(cls, _attr)) or inspect.isfunction(getattr(cls, _attr)):
            attr_func = getattr(self, _attr)
            attr_func_kwargs = dict()
            attr_func_arg_names = get_func_arg_names(attr_func)
            custom_k = False
            for k, v in final_kwargs.items():
                if k in attr_func_arg_names or k in kwargs:
                    if k in custom_arg_names:
                        custom_k = True
                    attr_func_kwargs[k] = v
                    passed_kwargs_out[k] = v
            if use_caching and not custom_k and attr in cache_dct:
                out = cache_dct[attr]
            else:
                if use_shortcuts:
                    out = self.resolve_shortcut_attr(_attr, *args, **attr_func_kwargs)
                else:
                    out = attr_func(*args, **attr_func_kwargs)
                if use_caching and not custom_k:
                    cache_dct[attr] = out
        else:
            if use_caching and attr in cache_dct:
                out = cache_dct[attr]
            else:
                if use_shortcuts:
                    out = self.resolve_shortcut_attr(_attr)
                else:
                    out = getattr(self, _attr)
                if use_caching:
                    cache_dct[attr] = out
        out = self.post_resolve_attr(attr, out, final_kwargs=final_kwargs)
        return out

    def deep_getattr(self, *args, **kwargs) -> tp.Any:
        """Retrieve a nested attribute using the `deep_getattr` function.

        Args:
            *args: Positional arguments for `deep_getattr`.
            **kwargs: Keyword arguments for `deep_getattr`.

        Returns:
            Any: The value of the nested attribute.
        """
        return deep_getattr(self, *args, **kwargs)


def parse_attrs(obj: tp.Any = None, own_only: bool = False, sort_by: tp.Optional[str] = None) -> tp.Frame:
    """Parse attributes of a class, object, or module and return a DataFrame of types and paths.

    Args:
        obj (Any): Object, class, or module whose attributes are to be parsed.

            Defaults to `vbt` if None.
        own_only (bool): If True, only include attributes defined on the object's class.
        sort_by (Optional[str]): Column name used to sort the resulting DataFrame.

    Returns:
        Frame: A DataFrame indexed by attribute names with columns for type and path information.
    """
    if obj is None:
        obj = vbt
    if inspect.isclass(obj) or inspect.ismodule(obj):
        cls = obj
    else:
        cls = type(obj)
    attr_info = {"type": {}, "path": {}}
    mro_dirs = []
    if not inspect.ismodule(cls):
        for super_cls in inspect.getmro(cls)[1:]:
            mro_dirs.append(set(dir(super_cls)))
    for k in dir(cls):
        if not k.startswith("_"):
            v = inspect.getattr_static(cls, k)
            if inspect.ismodule(v):
                if hasattr(v, "__path__"):
                    attr_info["type"][k] = "package"
                elif v.__spec__.origin in (None, "namespace"):
                    attr_info["type"][k] = "namespace"
                else:
                    attr_info["type"][k] = "module"
            else:
                attr_info["type"][k] = type(v).__name__

            if inspect.ismodule(cls):
                k_module = inspect.getmodule(v)
                if k_module is not None:
                    attr_info["path"][k] = inspect.getmodule(v).__name__
                else:
                    attr_info["path"][k] = "?"
            else:
                found_super_cls = False
                for i, super_cls in enumerate(inspect.getmro(cls)[1:]):
                    if k in mro_dirs[i]:
                        v2 = inspect.getattr_static(super_cls, k, None)
                        if v2 is not None and v == v2:
                            attr_info["path"][k] = super_cls.__module__ + "." + super_cls.__name__
                            found_super_cls = True
                if not found_super_cls:
                    attr_info["path"][k] = cls.__module__ + "." + cls.__name__

    if own_only:
        for k, v in list(attr_info["path"].items()):
            if inspect.ismodule(cls):
                if not v.startswith(cls.__name__):
                    del attr_info["type"][k]
                    del attr_info["path"][k]
            else:
                if not v.startswith(cls.__module__ + "." + cls.__name__):
                    del attr_info["type"][k]
                    del attr_info["path"][k]

    attr_info = pd.DataFrame(attr_info, dtype=object)
    if len(attr_info) > 0:
        attr_info.index.name = "attr"
        attr_info = attr_info.reset_index()
        if sort_by is not None:
            if isinstance(sort_by, str):
                if sort_by != "attr":
                    sort_by = [sort_by, "attr"]
            else:
                if "attr" not in sort_by:
                    sort_by = [*sort_by, "attr"]
            attr_info = attr_info.sort_values(sort_by)
        attr_info = attr_info.set_index("attr", drop=True)
    return attr_info
