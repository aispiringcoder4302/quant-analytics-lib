# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for validation during runtime."""

import datetime
import traceback
from collections.abc import Collection, Iterable, Sequence, Hashable, Mapping
from inspect import signature, getmro
from keyword import iskeyword
from types import FunctionType, BuiltinFunctionType, MethodType

import attr
import numba
import numpy as np
import pandas as pd
from numba.core.registry import CPUDispatcher

from vectorbtpro import _typing as tp
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.warnings_ import warn

__all__ = [
    "is_numba_enabled",
    "is_deep_equal",
]


class Comparable(Base):
    """Class for objects that support deep equality comparison."""

    def equals(self, other: tp.Any, *args, **kwargs) -> bool:
        """Return whether the current object is deeply equal to another object.

        Args:
            other (Any): The object to compare against.
            *args: Additional arguments passed to `is_deep_equal`.
            **kwargs: Additional keyword arguments passed to `is_deep_equal`.

        !!! note
            This method should accept all keyword arguments supported by `is_deep_equal`."""
        raise NotImplementedError

    def __eq__(self, other: tp.Any) -> bool:
        return self.equals(other)


# ############# Checks ############# #


def is_classic_func(arg: tp.Any) -> bool:
    """Return whether the provided argument is a classic Python function.

    Args:
        arg (Any): The object to check."""
    return isinstance(arg, FunctionType)


def is_builtin_func(arg: tp.Any) -> bool:
    """Return whether the provided argument is a built-in function.

    Args:
        arg (Any): The object to check."""
    return isinstance(arg, BuiltinFunctionType)


def is_method(arg: tp.Any) -> bool:
    """Return whether the provided argument is a method.

    Args:
        arg (Any): The object to check."""
    return isinstance(arg, MethodType)


def is_numba_enabled() -> bool:
    """Return whether Numba is globally enabled."""
    return numba.config.DISABLE_JIT != 1


def is_numba_func(arg: tp.Any) -> bool:
    """Return whether the provided argument is identified as a Numba-compiled function based on configuration.

    Args:
        arg (Any): The object to check."""
    from vectorbtpro._settings import settings

    numba_cfg = settings["numba"]

    if not numba_cfg["check_func_type"]:
        return True
    if not is_numba_enabled():
        if numba_cfg["check_func_suffix"]:
            if hasattr(arg, "__name__") and arg.__name__.endswith("_nb"):
                return True
            return False
        return False
    return isinstance(arg, CPUDispatcher)


def is_function(arg: tp.Any) -> bool:
    """Return whether the provided argument is a lambda function, built-in function, method, or Numba-compiled function.

    Args:
        arg (Any): The object to check."""
    return is_classic_func(arg) or is_builtin_func(arg) or is_method(arg) or is_numba_func(arg)


def is_bool(arg: tp.Any) -> bool:
    """Return whether the provided argument is a boolean value.

    Args:
        arg (Any): The object to check."""
    return isinstance(arg, (bool, np.bool_))


def is_int(arg: tp.Any) -> bool:
    """Return whether the provided argument is an integer (excluding booleans and timedelta values).

    Args:
        arg (Any): The object to check."""
    return isinstance(arg, (int, np.integer)) and not isinstance(arg, np.timedelta64) and not is_bool(arg)


def is_float(arg: tp.Any) -> bool:
    """Return whether the provided argument is a float.

    Args:
        arg (Any): The object to check."""
    return isinstance(arg, (float, np.floating))


def is_number(arg: tp.Any) -> bool:
    """Return whether the provided argument is a number (integer or float).

    Args:
        arg (Any): The object to check."""
    return is_int(arg) or is_float(arg)


def is_np_scalar(arg: tp.Any) -> bool:
    """Return whether the provided argument is a NumPy scalar.

    Args:
        arg (Any): The object to check."""
    return isinstance(arg, np.generic)


def is_td(arg: tp.Any) -> bool:
    """Return whether the provided argument is a timedelta object (from pandas, datetime, or NumPy).

    Args:
        arg (Any): The object to check."""
    return isinstance(arg, (pd.Timedelta, datetime.timedelta, np.timedelta64))


def is_td_like(arg: tp.Any) -> bool:
    """Return whether the provided argument is timedelta-like (i.e., a timedelta object, a number, or a string).

    Args:
        arg (Any): The object to check."""
    return is_td(arg) or is_number(arg) or isinstance(arg, str)


def is_frequency(arg: tp.Any) -> bool:
    """Return whether the provided argument is a frequency object (a timedelta or a pandas DateOffset).

    Args:
        arg (Any): The object to check."""
    return is_td(arg) or isinstance(arg, pd.DateOffset)


def is_frequency_like(arg: tp.Any) -> bool:
    """Return whether the provided argument is frequency-like (i.e., a frequency object, a number, or a string).

    Args:
        arg (Any): The object to check."""
    return is_frequency(arg) or is_number(arg) or isinstance(arg, str)


def is_dt(arg: tp.Any) -> bool:
    """Return whether the provided argument is a datetime object (from pandas, datetime, or NumPy).

    Args:
        arg (Any): The object to check."""
    return isinstance(arg, (pd.Timestamp, datetime.datetime, np.datetime64))


def is_dt_like(arg: tp.Any) -> bool:
    """Return whether the provided argument is datetime-like (i.e., a datetime object, a number, or a string).

    Args:
        arg (Any): The object to check."""
    return is_dt(arg) or is_number(arg) or isinstance(arg, str)


def is_time(arg: tp.Any) -> bool:
    """Return whether the provided argument is a time object.

    Args:
        arg (Any): The object to check."""
    return isinstance(arg, datetime.time)


def is_time_like(arg: tp.Any) -> bool:
    """Return whether the provided argument is time-like (i.e., a time object or a string).

    Args:
        arg (Any): The object to check."""
    return is_time(arg) or isinstance(arg, str)


def is_np_array(arg: tp.Any) -> bool:
    """Return whether the provided argument is a NumPy array.

    Args:
        arg (Any): The object to check."""
    return isinstance(arg, np.ndarray)


def is_record_array(arg: tp.Any) -> bool:
    """Return whether the provided argument is a structured NumPy array.

    Args:
        arg (Any): The object to check."""
    return is_np_array(arg) and arg.dtype.fields is not None


def is_series(arg: tp.Any) -> bool:
    """Return whether the provided argument is a pandas Series.

    Args:
        arg (Any): The object to check."""
    return isinstance(arg, pd.Series)


def is_index(arg: tp.Any) -> bool:
    """Return whether the provided argument is a pandas Index.

    Args:
        arg (Any): The object to check."""
    return isinstance(arg, pd.Index)


def is_multi_index(arg: tp.Any) -> bool:
    """Return whether the provided argument is a pandas MultiIndex.

    Args:
        arg (Any): The object to check."""
    return isinstance(arg, pd.MultiIndex)


def is_frame(arg: tp.Any) -> bool:
    """Return whether the provided argument is a pandas DataFrame.

    Args:
        arg (Any): The object to check."""
    return isinstance(arg, pd.DataFrame)


def is_pandas(arg: tp.Any) -> bool:
    """Return whether the provided argument is a pandas object (Series, Index, or DataFrame).

    Args:
        arg (Any): The object to check."""
    return is_series(arg) or is_index(arg) or is_frame(arg)


def is_any_array(arg: tp.Any) -> bool:
    """Return whether the provided argument is any array-like object (a NumPy array or a pandas object).

    Args:
        arg (Any): The object to check."""
    return is_pandas(arg) or isinstance(arg, np.ndarray)


def _to_any_array(arg: tp.ArrayLike) -> tp.AnyArray:
    """Convert any array-like object to an array.

    Pandas objects are kept as-is."""
    if is_any_array(arg):
        return arg
    return np.asarray(arg)


def is_collection(arg: tp.Any) -> bool:
    """Return whether the provided argument is considered a collection.

    Args:
        arg (Any): The object to check."""
    if isinstance(arg, Collection):
        return True
    try:
        len(arg)
        return True
    except TypeError:
        return False


def is_complex_collection(arg: tp.Any) -> bool:
    """Return whether the provided argument is a complex collection
    (i.e., a collection that is not a string, bytes, or bytearray).

    Args:
        arg (Any): The object to check."""
    if isinstance(arg, (str, bytes, bytearray)):
        return False
    return is_collection(arg)


def is_iterable(arg: tp.Any) -> bool:
    """Return whether the provided argument is iterable.

    Args:
        arg (Any): The object to check."""
    if isinstance(arg, Iterable):
        return True
    try:
        _ = iter(arg)
        return True
    except TypeError:
        return False


def is_complex_iterable(arg: tp.Any) -> bool:
    """Return whether the provided argument is a complex iterable
    (i.e., iterable but not a string, bytes, or bytearray).

    Args:
        arg (Any): The object to check."""
    if isinstance(arg, (str, bytes, bytearray)):
        return False
    return is_iterable(arg)


def is_sequence(arg: tp.Any) -> bool:
    """Return whether the provided argument is a sequence.

    Args:
        arg (Any): The object to check."""
    if isinstance(arg, Sequence):
        return True
    try:
        len(arg)
        arg[0:0]
        return True
    except (TypeError, IndexError, KeyError):
        return False


def is_complex_sequence(arg: tp.Any) -> bool:
    """Return whether the provided argument is a complex sequence
    (i.e., a sequence that is not a string, bytes, or bytearray).

    Args:
        arg (Any): The object to check."""
    if isinstance(arg, (str, bytes, bytearray)):
        return False
    return is_sequence(arg)


def is_hashable(arg: tp.Any) -> bool:
    """Return whether the provided argument can be hashed.

    Args:
        arg (Any): The object to check.

    !!! note
        An object with a `__hash__` method might still be unhashable if invoking `hash` raises a TypeError."""
    if not isinstance(arg, Hashable):
        return False
    # Having __hash__() method does not mean that it's hashable
    try:
        hash(arg)
    except TypeError:
        return False
    return True


def is_index_equal(arg1: tp.Any, arg2: tp.Any, check_names: bool = True) -> bool:
    """Return whether two indexes are equal.

    Args:
        arg1 (Any): The first index to compare.
        arg2 (Any): The second index to compare.
        check_names (bool): If True, compare the index names in addition to the index values.
    """
    if not check_names:
        return pd.Index.equals(arg1, arg2)
    if isinstance(arg1, pd.MultiIndex) and isinstance(arg2, pd.MultiIndex):
        if arg1.names != arg2.names:
            return False
    elif isinstance(arg1, pd.MultiIndex) or isinstance(arg2, pd.MultiIndex):
        return False
    else:
        if arg1.name != arg2.name:
            return False
    return pd.Index.equals(arg1, arg2)


def is_default_index(arg: tp.Any, check_names: bool = True) -> bool:
    """Return whether the provided index is a basic range index.

    Args:
        arg (Any): The index to check.
        check_names (bool): If True, compare the index names in addition to the index range.
    """
    return is_index_equal(arg, pd.RangeIndex(start=0, stop=len(arg), step=1), check_names=check_names)


def is_namedtuple(arg: tp.Any) -> bool:
    """Return whether the given object is an instance of a namedtuple.

    Args:
        arg (Any): The object to check.
    """
    if not isinstance(arg, type):
        arg = type(arg)
    bases = arg.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(arg, "_fields", None)
    if not isinstance(fields, tuple):
        return False
    return all(type(field) == str for field in fields)


def is_record(arg: tp.Any) -> bool:
    """Return whether the given object is a NumPy record.

    Args:
        arg (Any): The object to check.
    """
    return isinstance(arg, (np.void, np.record)) and hasattr(arg.dtype, "names") and arg.dtype.names is not None


def func_accepts_arg(func: tp.Callable, arg_name: str, arg_kind: tp.Optional[tp.MaybeTuple[int]] = None) -> bool:
    """Return whether the function accepts an argument with the specified name.

    Args:
        func (Callable): The function to inspect.
        arg_name (str): The name of the argument to verify.
        arg_kind (Optional[MaybeTuple[int]]): The kind or kinds of argument to check.
    """
    sig = signature(func)
    if arg_kind is not None and isinstance(arg_kind, int):
        arg_kind = (arg_kind,)
    if arg_kind is None:
        if arg_name.startswith("**"):
            return arg_name[2:] in [p.name for p in sig.parameters.values() if p.kind == p.VAR_KEYWORD]
        if arg_name.startswith("*"):
            return arg_name[1:] in [p.name for p in sig.parameters.values() if p.kind == p.VAR_POSITIONAL]
        return arg_name in [
            p.name for p in sig.parameters.values() if p.kind != p.VAR_POSITIONAL and p.kind != p.VAR_KEYWORD
        ]
    return arg_name in [p.name for p in sig.parameters.values() if p.kind in arg_kind]


def is_equal(
    arg1: tp.Any,
    arg2: tp.Any,
    equality_func: tp.Callable[[tp.Any, tp.Any], bool] = lambda x, y: x == y,
) -> bool:
    """Return whether two objects are equal using the provided equality function.

    Args:
        arg1 (Any): The first object for comparison.
        arg2 (Any): The second object for comparison.
        equality_func (Callable[[Any, Any], bool]): A function to evaluate equality.
    """
    try:
        return equality_func(arg1, arg2)
    except:
        pass
    return False


def is_deep_equal(
    arg1: tp.Any,
    arg2: tp.Any,
    check_exact: bool = False,
    debug: bool = False,
    only_types: bool = False,
    _key: tp.Optional[str] = None,
    **kwargs,
) -> bool:
    """Return whether two objects are deeply equal by performing a recursive comparison.

    Args:
        arg1 (Any): The first object for deep comparison.
        arg2 (Any): The second object for deep comparison.
        check_exact (bool): If True, enforce exact matching in comparisons.
        debug (bool): If True, output warning messages on mismatches.
        only_types (bool): If True, only compare the types of the objects.
        _key (Optional[str]): A key path identifier for nested comparisons.
        **kwargs: Additional keyword arguments passed to underlying comparison functions.
    """

    def _select_kwargs(_method, _kwargs):
        __kwargs = dict()
        if len(kwargs) > 0:
            for k, v in _kwargs.items():
                if func_accepts_arg(_method, k):
                    __kwargs[k] = v
        return __kwargs

    def _check_array(assert_method):
        __kwargs = _select_kwargs(assert_method, kwargs)
        if arg1.dtype != arg2.dtype:
            raise AssertionError(f"Dtypes {arg1.dtype} and {arg2.dtype} do not match")
        if arg1.dtype.fields is not None:
            for field in arg1.dtype.names:
                try:
                    assert_method(arg1[field], arg2[field], **__kwargs)
                except Exception as e:
                    raise AssertionError(f"Dtype field '{field}'") from e
        else:
            assert_method(arg1, arg2, **__kwargs)

    try:
        if only_types:
            if type(arg1) != type(arg2):
                raise AssertionError(f"Types {type(arg1)} and {type(arg2)} do not match")
            return True
        if id(arg1) == id(arg2):
            return True
        if isinstance(arg1, Comparable):
            return arg1.equals(
                arg2,
                check_exact=check_exact,
                debug=debug,
                only_types=only_types,
                _key=_key,
                **kwargs,
            )
        if type(arg1) != type(arg2):
            raise AssertionError(f"Types {type(arg1)} and {type(arg2)} do not match")
        if attr.has(type(arg1)):
            return is_deep_equal(
                attr.asdict(arg1),
                attr.asdict(arg2),
                check_exact=check_exact,
                debug=debug,
                only_types=only_types,
                _key=_key,
                **kwargs,
            )
        if isinstance(arg1, pd.Series):
            _kwargs = _select_kwargs(pd.testing.assert_series_equal, kwargs)
            pd.testing.assert_series_equal(arg1, arg2, check_exact=check_exact, **_kwargs)
        elif isinstance(arg1, pd.DataFrame):
            _kwargs = _select_kwargs(pd.testing.assert_frame_equal, kwargs)
            pd.testing.assert_frame_equal(arg1, arg2, check_exact=check_exact, **_kwargs)
        elif isinstance(arg1, pd.Index):
            _kwargs = _select_kwargs(pd.testing.assert_index_equal, kwargs)
            pd.testing.assert_index_equal(arg1, arg2, check_exact=check_exact, **_kwargs)
        elif isinstance(arg1, np.ndarray):
            try:
                _check_array(np.testing.assert_array_equal)
            except Exception as e:
                if check_exact:
                    raise e
                _check_array(np.testing.assert_allclose)
        elif isinstance(arg1, (tuple, list)):
            for i in range(len(arg1)):
                if not is_deep_equal(
                    arg1[i],
                    arg2[i],
                    check_exact=check_exact,
                    debug=debug,
                    only_types=only_types,
                    _key=f"[{i}]" if _key is None else _key + f"[{i}]",
                    **kwargs,
                ):
                    return False
        elif isinstance(arg1, dict):
            for k in arg1.keys():
                if not is_deep_equal(
                    arg1[k],
                    arg2[k],
                    check_exact=check_exact,
                    debug=debug,
                    only_types=only_types,
                    _key=f"['{k}']" if _key is None else _key + f"['{k}']",
                    **kwargs,
                ):
                    return False
        else:
            try:
                if arg1 == arg2:
                    return True
            except:
                pass
            try:
                import dill

                _kwargs = _select_kwargs(dill.dumps, kwargs)
                if dill.dumps(arg1, **_kwargs) == dill.dumps(arg2, **_kwargs):
                    return True
            except:
                pass
            if debug:
                warn(f"\n############### {_key} ###############\nObjects do not match")
            return False
    except Exception as e:
        if debug:
            if _key is None:
                warn(traceback.format_exc())
            else:
                warn(f"\n############### {_key} ###############\n" + traceback.format_exc())
        return False
    return True


def is_class(arg: type, types: tp.TypeLike) -> bool:
    """Return whether the given class matches the specified type descriptor.

    Args:
        arg (type): The class to check.
        types (TypeLike): A type, string, or `vectorbtpro.utils.parsing.Regex`
            pattern (or tuple of such) to compare against.
    """
    from vectorbtpro.utils.parsing import Regex

    if isinstance(types, str):
        return str(arg) == types or arg.__name__ == types
    if isinstance(types, Regex):
        return types.matches(str(arg)) or types.matches(arg.__name__)
    if isinstance(types, tuple):
        for t in types:
            if is_class(arg, t):
                return True
        return False
    return arg is types


def is_subclass_of(arg: tp.Any, types: tp.TypeLike) -> bool:
    """Return whether the given argument is a subclass of the specified type descriptor.

    Args:
        arg (Any): The class to verify for subclassing.
        types (TypeLike): A type, string, or `vectorbtpro.utils.parsing.Regex` pattern
            (or tuple of such) representing the superclass.
    """
    try:
        return issubclass(arg, types)
    except TypeError:
        pass
    if isinstance(types, str):
        if types.lower() == "args":
            if is_namedtuple(arg):
                return False
            return issubclass(arg, tuple)
        for base_t in getmro(arg):
            if str(base_t) == types or base_t.__name__ == types:
                return True
    from vectorbtpro.utils.parsing import Regex

    if isinstance(types, Regex):
        for base_t in getmro(arg):
            if types.matches(str(base_t)) or types.matches(base_t.__name__):
                return True
    if isinstance(types, tuple):
        for t in types:
            if is_subclass_of(arg, t):
                return True
    return False


def is_instance_of(arg: tp.Any, types: tp.TypeLike) -> bool:
    """Return True if arg is an instance of the specified type(s).

    Args:
        arg (Any): The object to check.
        types (TypeLike): A type, a tuple of types, or type name(s).
    """
    return is_subclass_of(type(arg), types)


def is_mapping(arg: tp.Any) -> bool:
    """Return True if arg is a mapping.

    Args:
        arg (Any): The object to check.
    """
    return isinstance(arg, Mapping)


def is_mapping_like(arg: tp.Any) -> bool:
    """Return True if arg is mapping-like.

    An object is considered mapping-like if it is a mapping, a Series, an Index, or a NamedTuple.

    Args:
        arg (Any): The object to check.
    """
    return is_mapping(arg) or is_series(arg) or is_index(arg) or is_namedtuple(arg)


def is_valid_variable_name(arg: str) -> bool:
    """Return True if arg is a valid variable name.

    Args:
        arg (str): The string representing the variable name.
    """
    return arg.isidentifier() and not iskeyword(arg)


def in_notebook() -> bool:
    """Return True if executing in a Jupyter notebook environment.

    This function checks the IPython configuration to determine if the code is running in a notebook.
    """
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return False
        if "IPKernelApp" not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


# ############# Asserts ############# #


def safe_assert(arg: bool, msg: tp.Optional[str] = None) -> None:
    """Assert that a condition is True.

    Raises an `AssertionError` with the provided message if the condition is False.

    Args:
        arg (bool): The condition to evaluate.
        msg (Optional[str]): The error message to use if the assertion fails.
    """
    if not arg:
        raise AssertionError(msg)


def assert_in(arg1: tp.Any, arg2: tp.Sequence, arg_name: tp.Optional[str] = None) -> None:
    """Assert that arg1 is present in arg2.

    Raises an `AssertionError` if arg1 is not found in arg2.

    Args:
        arg1 (Any): The element to search for.
        arg2 (Sequence): The sequence in which to search.
        arg_name (Optional[str]): The variable name for error messaging.
    """
    if arg_name is None:
        x = ""
    else:
        x = f"for '{arg_name}'"
    if arg1 not in arg2:
        raise AssertionError(f"{arg1} not found in {arg2}{x}")


def assert_numba_func(func: tp.Callable) -> None:
    """Assert that the function is Numba-compiled.

    Raises an `AssertionError` if func is not compiled with Numba.

    Args:
        func (Callable): The function to check for Numba compilation.
    """
    if not is_numba_func(func):
        raise AssertionError(f"Function {func} must be Numba compiled")


def assert_not_none(arg: tp.Any, arg_name: tp.Optional[str] = None) -> None:
    """Assert that the argument is not None.

    Raises an `AssertionError` if arg is None.

    Args:
        arg (Any): The value to check.
        arg_name (Optional[str]): The variable name for error messaging.
    """
    if arg_name is None:
        x = "Argument"
    else:
        x = f"Argument '{arg_name}'"
    if arg is None:
        raise AssertionError(f"{x} cannot be None")


def assert_instance_of(arg: tp.Any, types: tp.TypeLike, arg_name: tp.Optional[str] = None) -> None:
    """Assert that arg is an instance of the specified type(s).

    Raises an `AssertionError` if arg is not an instance of types.

    Args:
        arg (Any): The object to validate.
        types (TypeLike): A type, tuple of types, or type name(s) to check against.
        arg_name (Optional[str]): The variable name for error messaging.
    """
    if arg_name is None:
        x = "Argument"
    else:
        x = f"Argument '{arg_name}'"
    if not is_instance_of(arg, types):
        if isinstance(types, tuple):
            raise AssertionError(f"{x} must be of one of types {types}, not {type(arg)}")
        else:
            raise AssertionError(f"{x} must be of type {types}, not {type(arg)}")


def assert_not_instance_of(arg: tp.Any, types: tp.TypeLike, arg_name: tp.Optional[str] = None) -> None:
    """Assert that arg is not an instance of the specified type(s).

    Raises an `AssertionError` if arg is an instance of types.

    Args:
        arg (Any): The object to validate.
        types (TypeLike): A type, tuple of types, or type name(s) that are disallowed.
        arg_name (Optional[str]): The variable name for error messaging.
    """
    if arg_name is None:
        x = "Argument"
    else:
        x = f"Argument '{arg_name}'"
    if is_instance_of(arg, types):
        if isinstance(types, tuple):
            raise AssertionError(f"{x} cannot be of one of types {types}")
        else:
            raise AssertionError(f"{x} cannot be of type {types}")


def assert_subclass_of(arg: tp.Type, classes: tp.TypeLike, arg_name: tp.Optional[str] = None) -> None:
    """Assert that arg is a subclass of the specified class(es).

    Raises an `AssertionError` if arg is not a subclass of classes.

    Args:
        arg (Type): The type to check.
        classes (TypeLike): A class or tuple of classes for validation.
        arg_name (Optional[str]): The variable name for error messaging.
    """
    if arg_name is None:
        x = "Argument"
    else:
        x = f"Argument '{arg_name}'"
    if not is_subclass_of(arg, classes):
        if isinstance(classes, tuple):
            raise AssertionError(f"{x} must be a subclass of one of types {classes}")
        else:
            raise AssertionError(f"{x} must be a subclass of type {classes}")


def assert_not_subclass_of(arg: tp.Type, classes: tp.TypeLike, arg_name: tp.Optional[str] = None) -> None:
    """Assert that arg is not a subclass of the specified class(es).

    Raises an `AssertionError` if arg is a subclass of classes.

    Args:
        arg (Type): The type to check.
        classes (TypeLike): A class or tuple of classes that are disallowed.
        arg_name (Optional[str]): The variable name for error messaging.
    """
    if arg_name is None:
        x = "Argument"
    else:
        x = f"Argument '{arg_name}'"
    if is_subclass_of(arg, classes):
        if isinstance(classes, tuple):
            raise AssertionError(f"{x} cannot be a subclass of one of types {classes}")
        else:
            raise AssertionError(f"{x} cannot be a subclass of type {classes}")


def assert_type_equal(arg1: tp.Any, arg2: tp.Any) -> None:
    """Assert that arg1 and arg2 have the same type.

    Raises an `AssertionError` if the types of arg1 and arg2 do not match.

    Args:
        arg1 (Any): The first object to compare.
        arg2 (Any): The second object to compare.
    """
    if type(arg1) != type(arg2):
        raise AssertionError(f"Types {type(arg1)} and {type(arg2)} do not match")


def assert_dtype(arg: tp.ArrayLike, dtype: tp.MaybeTuple[tp.DTypeLike], arg_name: tp.Optional[str] = None) -> None:
    """Assert that the data type of arg matches the specified dtype.

    For a DataFrame, each column's data type is validated.

    Raises an `AssertionError` if argument's data type does not match dtype.

    Args:
        arg (ArrayLike): The array or DataFrame to validate.
        dtype (MaybeTuple[DTypeLike]): The expected data type or a tuple of possible data types.
        arg_name (Optional[str]): The variable name for error messaging.
    """
    if arg_name is None:
        x = "Data type"
    else:
        x = f"Data type of '{arg_name}'"
    arg = _to_any_array(arg)
    if isinstance(dtype, tuple):
        if isinstance(arg, pd.DataFrame):
            for i, col_dtype in enumerate(arg.dtypes):
                if not any([col_dtype == _dtype for _dtype in dtype]):
                    raise AssertionError(f"{x} (column {i}) must be one of {dtype}, not {col_dtype}")
        else:
            if not any([arg.dtype == _dtype for _dtype in dtype]):
                raise AssertionError(f"{x} must be one of {dtype}, not {arg.dtype}")
    else:
        if isinstance(arg, pd.DataFrame):
            for i, col_dtype in enumerate(arg.dtypes):
                if col_dtype != dtype:
                    raise AssertionError(f"{x} (column {i}) must be {dtype}, not {col_dtype}")
        else:
            if arg.dtype != dtype:
                raise AssertionError(f"{x} must be {dtype}, not {arg.dtype}")


def assert_subdtype(arg: tp.ArrayLike, dtype: tp.MaybeTuple[tp.DTypeLike], arg_name: tp.Optional[str] = None) -> None:
    """Assert that the data type of arg is a subtype of the specified dtype.

    For a DataFrame, each column's data type is validated.

    Raises an `AssertionError` if argument's data type is not a subdata type of dtype.

    Args:
        arg (ArrayLike): The array or DataFrame to validate.
        dtype (MaybeTuple[DTypeLike]): The expected data type or a tuple of data types.
        arg_name (Optional[str]): The variable name for error messaging.
    """
    if arg_name is None:
        x = "Data type"
    else:
        x = f"Data type of '{arg_name}'"
    arg = _to_any_array(arg)
    if isinstance(dtype, tuple):
        if isinstance(arg, pd.DataFrame):
            for i, col_dtype in enumerate(arg.dtypes):
                if not any([np.issubdtype(col_dtype, _dtype) for _dtype in dtype]):
                    raise AssertionError(f"{x} (column {i}) must be one of {dtype}, not {col_dtype}")
        else:
            if not any([np.issubdtype(arg.dtype, _dtype) for _dtype in dtype]):
                raise AssertionError(f"{x} must be one of {dtype}, not {arg.dtype}")
    else:
        if isinstance(arg, pd.DataFrame):
            for i, col_dtype in enumerate(arg.dtypes):
                if not np.issubdtype(col_dtype, dtype):
                    raise AssertionError(f"{x} (column {i}) must be {dtype}, not {col_dtype}")
        else:
            if not np.issubdtype(arg.dtype, dtype):
                raise AssertionError(f"{x} must be {dtype}, not {arg.dtype}")


def assert_dtype_equal(arg1: tp.ArrayLike, arg2: tp.ArrayLike) -> None:
    """Assert that the data types of arg1 and arg2 are equal.

    Raises an `AssertionError` if the data types of arg1 and arg2 do not match.

    Args:
        arg1 (ArrayLike): The first array or DataFrame to compare.
        arg2 (ArrayLike): The second array or DataFrame to compare.
    """
    arg1 = _to_any_array(arg1)
    arg2 = _to_any_array(arg2)
    if isinstance(arg1, pd.DataFrame):
        dtypes1 = arg1.dtypes.to_numpy()
    else:
        dtypes1 = np.array([arg1.dtype])
    if isinstance(arg2, pd.DataFrame):
        dtypes2 = arg2.dtypes.to_numpy()
    else:
        dtypes2 = np.array([arg2.dtype])
    if len(dtypes1) == len(dtypes2):
        if (dtypes1 == dtypes2).all():
            return
    elif len(np.unique(dtypes1)) == 1 and len(np.unique(dtypes2)) == 1:
        if np.all(np.unique(dtypes1) == np.unique(dtypes2)):
            return
    raise AssertionError(f"Data types {dtypes1} and {dtypes2} do not match")


def assert_ndim(arg: tp.ArrayLike, ndims: tp.MaybeTuple[int]) -> None:
    """Raise an `AssertionError` if `arg` does not have the expected number of dimensions.

    Args:
        arg (ArrayLike): Array-like object to be checked.
        ndims (MaybeTuple[int]): Expected number of dimensions or acceptable dimension values.
    """
    arg = _to_any_array(arg)
    if isinstance(ndims, tuple):
        if arg.ndim not in ndims:
            raise AssertionError(f"Number of dimensions must be one of {ndims}, not {arg.ndim}")
    else:
        if arg.ndim != ndims:
            raise AssertionError(f"Number of dimensions must be {ndims}, not {arg.ndim}")


def assert_len_equal(arg1: tp.Sized, arg2: tp.Sized) -> None:
    """Raise an `AssertionError` if `arg1` and `arg2` do not have the same length.

    Args:
        arg1 (Sized): The first object whose length is compared.
        arg2 (Sized): The second object whose length is compared.

    !!! note
        The arguments are not converted to NumPy arrays.
    """
    if len(arg1) != len(arg2):
        raise AssertionError(f"Lengths of {arg1} and {arg2} do not match")


def assert_shape_equal(
    arg1: tp.ArrayLike,
    arg2: tp.ArrayLike,
    axis: tp.Optional[tp.Union[int, tp.Tuple[int, int]]] = None,
) -> None:
    """Raise an `AssertionError` if the shapes of `arg1` and `arg2` do not match along the specified axis.

    If `axis` is None, the entire shapes are compared.
    If `axis` is a tuple, the first element corresponds to `arg1` and the second to `arg2`.
    If `axis` is an integer, that axis index is compared for both arrays.

    Args:
        arg1 (ArrayLike): The first array-like object.
        arg2 (ArrayLike): The second array-like object.
        axis (Optional[Union[int, Tuple[int, int]]): The axis or axes along which to compare shapes.
    """
    arg1 = _to_any_array(arg1)
    arg2 = _to_any_array(arg2)
    if axis is None:
        if arg1.shape != arg2.shape:
            raise AssertionError(f"Shapes {arg1.shape} and {arg2.shape} do not match")
    else:
        if isinstance(axis, tuple):
            if axis[0] >= arg1.ndim and axis[1] >= arg2.ndim:
                return
            if arg1.shape[axis[0]] != arg2.shape[axis[1]]:
                raise AssertionError(f"Axis {axis[0]} of {arg1.shape} and axis {axis[1]} of {arg2.shape} do not match")
        else:
            if axis >= arg1.ndim and axis >= arg2.ndim:
                return
            if arg1.shape[axis] != arg2.shape[axis]:
                raise AssertionError(f"Axis {axis} of {arg1.shape} and {arg2.shape} do not match")


def assert_index_equal(arg1: tp.Index, arg2: tp.Index, check_names: bool = True) -> None:
    """Raise an `AssertionError` if the indexes of `arg1` and `arg2` do not match.

    Args:
        arg1 (Index): The first index to compare.
        arg2 (Index): The second index to compare.
        check_names (bool): Whether to check the names of the indexes.
    """
    if not is_index_equal(arg1, arg2, check_names=check_names):
        raise AssertionError(f"Indexes {arg1} and {arg2} do not match")


def assert_columns_equal(arg1: tp.Index, arg2: tp.Index, check_names: bool = True) -> None:
    """Raise an `AssertionError` if the columns of `arg1` and `arg2` do not match.

    Args:
        arg1 (Index): The first columns index to compare.
        arg2 (Index): The second columns index to compare.
        check_names (bool): Whether to check the names of the columns.
    """
    if not is_index_equal(arg1, arg2, check_names=check_names):
        raise AssertionError(f"Columns {arg1} and {arg2} do not match")


def assert_meta_equal(arg1: tp.ArrayLike, arg2: tp.ArrayLike, axis: tp.Optional[int] = None) -> None:
    """Raise an `AssertionError` if `arg1` and `arg2` have incompatible metadata.

    The function validates type and shape equality. For pandas objects, it additionally compares
    indexes and, when applicable, columns or series names.

    Args:
        arg1 (ArrayLike): The first array-like object.
        arg2 (ArrayLike): The second array-like object.
        axis (Optional[int]): Axis along which to compare metadata.
    """
    arg1 = _to_any_array(arg1)
    arg2 = _to_any_array(arg2)
    assert_type_equal(arg1, arg2)
    if axis is not None:
        assert_shape_equal(arg1, arg2, axis=axis)
    else:
        assert_shape_equal(arg1, arg2)
    if is_pandas(arg1) and is_pandas(arg2):
        if axis is None or axis == 0:
            assert_index_equal(arg1.index, arg2.index)
        if axis is None or axis == 1:
            if is_series(arg1) and is_series(arg2):
                assert_columns_equal(pd.Index([arg1.name]), pd.Index([arg2.name]))
            else:
                assert_columns_equal(arg1.columns, arg2.columns)


def assert_array_equal(arg1: tp.ArrayLike, arg2: tp.ArrayLike) -> None:
    """Raise an `AssertionError` if `arg1` and `arg2` differ in metadata or values.

    The function first compares metadata using `assert_meta_equal`, then checks actual data equality using:

    * A pandas equality check if both objects are pandas.
    * A NumPy array equality check otherwise.

    Args:
        arg1 (ArrayLike): The first array-like object.
        arg2 (ArrayLike): The second array-like object.
    """
    arg1 = _to_any_array(arg1)
    arg2 = _to_any_array(arg2)
    assert_meta_equal(arg1, arg2)
    if is_pandas(arg1) and is_pandas(arg2):
        if arg1.equals(arg2):
            return
    elif not is_pandas(arg1) and not is_pandas(arg2):
        if np.array_equal(arg1, arg2):
            return
    raise AssertionError(f"Arrays {arg1} and {arg2} do not match")


def assert_level_not_exists(arg: tp.Index, level_name: str) -> None:
    """Raise an `AssertionError` if `arg` contains a level named `level_name`.

    Args:
        arg (Index): The index to check.
        level_name (str): The name of the level that must not exist.
    """
    if isinstance(arg, pd.MultiIndex):
        names = arg.names
    else:
        names = [arg.name]
    if level_name in names:
        raise AssertionError(f"Level {level_name} already exists in {names}")


def assert_equal(arg1: tp.Any, arg2: tp.Any, deep: bool = False) -> None:
    """Raise an `AssertionError` if `arg1` and `arg2` are not equal.

    If `deep` is True, a deep equality check is performed.

    Args:
        arg1 (Any): The first object to compare.
        arg2 (Any): The second object to compare.
        deep (bool): If True, perform a deep equality check.
    """
    if deep:
        if not is_deep_equal(arg1, arg2):
            raise AssertionError(f"{arg1} and {arg2} do not match (deep check)")
    else:
        if not is_equal(arg1, arg2):
            raise AssertionError(f"{arg1} and {arg2} do not match")


def assert_dict_valid(arg: tp.DictLike, lvl_keys: tp.Sequence[tp.MaybeSequence[str]]) -> None:
    """Raise an `AssertionError` if `arg` contains keys not present in `lvl_keys`.

    `lvl_keys` should be a sequence of sequences, each corresponding to the valid keys
    for a level of the dictionary.

    Args:
        arg (DictLike): The dictionary to validate.
        lvl_keys (Sequence[MaybeSequence[str]]): A sequence of valid key sequences for each level.
    """
    if arg is None:
        arg = {}
    if len(lvl_keys) == 0:
        return
    if isinstance(lvl_keys[0], str):
        lvl_keys = [lvl_keys]
    set1 = set(arg.keys())
    set2 = set(lvl_keys[0])
    if not set1.issubset(set2):
        raise AssertionError(f"Invalid keys {list(set1.difference(set2))}, possible keys are {list(set2)}")
    for k, v in arg.items():
        if isinstance(v, dict):
            assert_dict_valid(v, lvl_keys[1:])


def assert_dict_sequence_valid(arg: tp.DictLikeSequence, lvl_keys: tp.Sequence[tp.MaybeSequence[str]]) -> None:
    """Raise an `AssertionError` if a dictionary or any dictionary within a sequence
    contains keys not present in `lvl_keys`.

    Args:
        arg (DictLikeSequence): A dictionary or a sequence of dictionaries to validate.
        lvl_keys (Sequence[MaybeSequence[str]]): A sequence of valid key sequences for each level.
    """
    if arg is None:
        arg = {}
    if isinstance(arg, dict):
        assert_dict_valid(arg, lvl_keys)
    else:
        for _arg in arg:
            assert_dict_valid(_arg, lvl_keys)


def assert_sequence(arg: tp.Any) -> None:
    """Raise a ValueError if `arg` is not a sequence.

    Args:
        arg (Any): Object to test for sequence behavior.
    """
    if not is_sequence(arg):
        raise ValueError(f"{arg} must be a sequence")


def assert_iterable(arg: tp.Any) -> None:
    """Raise a ValueError if `arg` is not an iterable.

    Args:
        arg (Any): Object to test for iterability.
    """
    if not is_iterable(arg):
        raise ValueError(f"{arg} must be an iterable")
