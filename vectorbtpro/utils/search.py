# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for searching."""

import re
from functools import partial
from collections import deque
from pathlib import Path
from copy import copy

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import set_dict_item, ReadonlyConfig

__pdoc__ = {}


PATH_TOKEN_REGEX = re.compile(
    r"""
    \.([a-zA-Z_][a-zA-Z0-9_]*)
    |\[['"]([^'"\]]+)['"]\]
    |\[(\d+)\]
    """,
    re.VERBOSE,
)
"""Path token regex for `parse_path_str`.

Matches `.key`, `['key']`, `["key"]`, `[0]`, etc."""

FIRST_TOKEN_REGEX = re.compile(
    r"""
    ^([a-zA-Z_][a-zA-Z0-9_]*)
    |\[['"]([^'"\]]+)['"]\]
    |\[(\d+)\]
    """,
    re.VERBOSE,
)
"""First token regex for `parse_path_str`.

Matches the same as `PATH_TOKEN_REGEX` but at the start."""


def parse_path_str(path_str: str) -> tp.PathKey:
    """Parse the path string into a list of tokens."""
    if "'" not in path_str and '"' not in path_str and "[" not in path_str:
        return tuple(path_str.split(".")) if path_str != "" else ()
    tokens = []
    first_match = FIRST_TOKEN_REGEX.match(path_str)
    if not first_match:
        raise ValueError(f"Invalid path syntax: '{path_str}'")
    if first_match.group(1):
        tokens.append(first_match.group(1))
    elif first_match.group(2):
        tokens.append(first_match.group(2))
    elif first_match.group(3):
        tokens.append(int(first_match.group(3)))
    pos = first_match.end()
    for match in PATH_TOKEN_REGEX.finditer(path_str, pos):
        key_dot, key_bracket, index = match.groups()
        if key_dot:
            tokens.append(key_dot)
        elif key_bracket:
            tokens.append(key_bracket)
        elif index:
            tokens.append(int(index))
        pos = match.end()
    if pos != len(path_str):
        raise ValueError(f"Invalid path syntax at position {pos}: '{path_str}'")
    return tuple(tokens)


def combine_path_str(path_str1: str, path_str2: str) -> str:
    """Combine two path strings into one."""
    if path_str1 == "":
        return path_str2
    if path_str2 == "":
        return path_str1
    path_str1 = path_str1.rstrip()
    path_str2 = path_str2.lstrip()
    path_str1 = path_str1.rstrip(".")
    path_str2 = path_str2.lstrip(".")
    ends_with_bracket = path_str1.endswith("]")
    starts_with_bracket = path_str2.startswith("[")
    if ends_with_bracket:
        if starts_with_bracket:
            combined = path_str1 + path_str2
        else:
            combined = path_str1 + "." + path_str2
    else:
        if starts_with_bracket:
            combined = path_str1 + path_str2
        else:
            combined = path_str1 + "." + path_str2
    return combined


def minimize_pathlike_key(key: tp.PathLikeKey) -> tp.MaybePathKey:
    """Minimize a path-like key."""
    key = resolve_pathlike_key(key)
    if len(key) == 0:
        return None
    if len(key) == 1:
        return key[0]
    return key


def resolve_pathlike_key(key: tp.PathLikeKey, minimize: bool = False) -> tp.PathKey:
    """Convert a path-like key into a path key."""
    if key is None:
        key = ()
    if isinstance(key, Path):
        key = key.parts
    if isinstance(key, str):
        key = parse_path_str(key)
    if not isinstance(key, tuple):
        key = (key,)
    if minimize:
        key = minimize_pathlike_key(key)
    return key


def combine_pathlike_keys(
    key1: tp.PathLikeKey,
    key2: tp.PathLikeKey,
    resolve: bool = False,
    minimize: bool = False,
) -> tp.PathLikeKey:
    """Combine two path-like keys."""
    if not resolve:
        if isinstance(key1, Path) and isinstance(key2, Path):
            new_k = key1 / key2
            if minimize:
                return minimize_pathlike_key(new_k)
            return new_k
        if isinstance(key1, str) and isinstance(key2, str):
            new_k = combine_path_str(key1, key2)
            if minimize:
                return minimize_pathlike_key(new_k)
            return new_k
    key1 = resolve_pathlike_key(key1)
    key2 = resolve_pathlike_key(key2)
    new_k = key1 + key2
    if minimize:
        return minimize_pathlike_key(new_k)
    return new_k


def get_pathlike_key(obj: tp.Any, key: tp.PathLikeKey, keep_path: bool = False) -> tp.Any:
    """Get the value under a path-like key in an object."""
    tokens = resolve_pathlike_key(key)
    for token in tokens:
        if isinstance(obj, (set, frozenset)):
            obj = list(obj)[token]
        elif hasattr(obj, "__getitem__"):
            obj = obj[token]
        elif isinstance(token, str) and hasattr(obj, token):
            obj = getattr(obj, token)
        else:
            raise TypeError(f"Cannot navigate object of type {type(obj).__name__}")
    if not keep_path:
        return obj
    path = obj
    for token in reversed(tokens):
        path = {token: path}
    return path


def set_pathlike_key(
    obj: tp.Any,
    key: tp.PathLikeKey,
    value: tp.Any,
    make_copy: bool = True,
    prev_keys: tp.Optional[tp.PathLikeKeys] = None,
) -> tp.Any:
    """Set the value under a path-like key in an object."""
    tokens = resolve_pathlike_key(key)
    parents = []
    new_obj = obj
    for i, token in enumerate(tokens):
        parents.append((obj, token))
        if i < len(tokens) - 1:
            if isinstance(obj, (set, frozenset)):
                obj = list(obj)[token]
            elif hasattr(obj, "__getitem__"):
                obj = obj[token]
            elif isinstance(token, str) and hasattr(obj, token):
                obj = getattr(obj, token)
            else:
                raise TypeError(f"Cannot navigate object of type {type(obj).__name__}")
        elif not make_copy:
            if hasattr(obj, "__setitem__"):
                obj[token] = value
            elif hasattr(obj, "__dict__"):
                setattr(obj, token, value)
            else:
                raise TypeError(f"Cannot modify object of type {type(obj).__name__}")
    if not make_copy:
        return new_obj

    if prev_keys is None:
        prev_keys = []
    prev_key_tokens = []
    for prev_key in prev_keys:
        prev_key_tokens.append(resolve_pathlike_key(prev_key))
    new_value = value
    for i, (parent, token) in enumerate(reversed(parents)):
        i = len(parents) - 1 - i
        if make_copy:
            for prev_tokens in prev_key_tokens:
                if tokens[:i] == prev_tokens[:i]:
                    make_copy = False
        if isinstance(parent, (tuple, set, frozenset)):
            parent_list = list(parent)
            parent_list[token] = new_value
            if checks.is_namedtuple(parent):
                parent_copy = type(parent)(*parent_list)
            else:
                parent_copy = type(parent)(parent_list)
        elif hasattr(parent, "__setitem__"):
            if make_copy:
                parent_copy = copy(parent)
            else:
                parent_copy = parent
            parent_copy[token] = new_value
        elif hasattr(parent, "__dict__"):
            if make_copy:
                parent_copy = copy(parent)
            else:
                parent_copy = parent
            setattr(parent_copy, token, new_value)
        else:
            raise TypeError(f"Cannot modify object of type {type(parent).__name__}")
        new_value = parent_copy
    prev_keys.append(key)
    return new_value


def remove_pathlike_key(
    obj: tp.Any,
    key: tp.PathLikeKey,
    make_copy: bool = True,
    prev_keys: tp.Optional[tp.PathLikeKeys] = None,
) -> tp.Any:
    """Remove the value under a path-like key in an object."""
    tokens = resolve_pathlike_key(key)
    parents = []
    new_obj = obj
    for i, token in enumerate(tokens):
        parents.append((obj, token))
        if i < len(tokens) - 1:
            if isinstance(obj, (set, frozenset)):
                obj = list(obj)[token]
            elif hasattr(obj, "__getitem__"):
                obj = obj[token]
            elif isinstance(token, str) and hasattr(obj, token):
                obj = getattr(obj, token)
            else:
                raise TypeError(f"Cannot navigate object of type {type(obj).__name__}")
        elif not make_copy:
            if isinstance(obj, set):
                obj.remove(token)
            elif hasattr(obj, "__delitem__"):
                del obj[token]
            elif hasattr(obj, "__dict__"):
                delattr(obj, token)
            else:
                raise TypeError(f"Cannot modify object of type {type(obj).__name__}")
    if not make_copy:
        return new_obj

    if prev_keys is None:
        prev_keys = []
    prev_key_tokens = []
    for prev_key in prev_keys:
        prev_key_tokens.append(resolve_pathlike_key(prev_key))
    for i, (parent, token) in enumerate(reversed(parents)):
        i = len(parents) - 1 - i
        if make_copy:
            for prev_tokens in prev_key_tokens:
                if tokens[:i] == prev_tokens[:i]:
                    make_copy = False
        if isinstance(parent, (tuple, set, frozenset)):
            parent_list = list(parent)
            if i == len(parents) - 1:
                parent_list.pop(token)
            else:
                parent_list[token] = new_value
            if checks.is_namedtuple(parent):
                parent_copy = type(parent)(*parent_list)
            else:
                parent_copy = type(parent)(parent_list)
        elif hasattr(parent, "__setitem__"):
            if make_copy:
                parent_copy = copy(parent)
            else:
                parent_copy = parent
            if i == len(parents) - 1:
                del parent_copy[token]
            else:
                parent_copy[token] = new_value
        elif hasattr(parent, "__dict__"):
            if make_copy:
                parent_copy = copy(parent)
            else:
                parent_copy = parent
            if i == len(parents) - 1:
                delattr(parent_copy, token)
            else:
                setattr(parent_copy, token, new_value)
        else:
            raise TypeError(f"Cannot modify object of type {type(parent).__name__}")
        new_value = parent_copy
    prev_keys.append(key)
    return new_value


def contains_in_obj(
    obj: tp.Any,
    match_func: tp.Callable,
    traversal: tp.Optional[str] = None,
    excl_types: tp.Union[None, bool, tp.MaybeSequence[type]] = None,
    incl_types: tp.Union[None, bool, tp.MaybeSequence[type]] = None,
    max_len: tp.Optional[int] = None,
    max_depth: tp.Optional[int] = None,
    **kwargs,
) -> bool:
    """Return whether there is any match in an object in an iterative manner.

    Argument `traversal` can be "DFS" for depth-first search or "BFS" for breadth-first search.

    See `find_in_obj` for arguments."""
    from vectorbtpro._settings import settings

    search_cfg = settings["search"]

    if traversal is None:
        traversal = search_cfg["traversal"]
    if excl_types is None:
        excl_types = search_cfg["excl_types"]
    if isinstance(excl_types, bool) and excl_types:
        raise ValueError("Argument excl_types cannot be True")
    if incl_types is None:
        incl_types = search_cfg["incl_types"]
    if isinstance(incl_types, bool) and not incl_types:
        raise ValueError("Argument incl_types cannot be False")
    if max_len is None:
        max_len = search_cfg["max_len"]
    if max_depth is None:
        max_depth = search_cfg["max_depth"]

    if traversal.upper() == "DFS":
        stack = [(None, 0, obj)]
    elif traversal.upper() == "BFS":
        stack = deque([(None, 0, obj)])
    else:
        raise ValueError(f"Invalid traversal: '{traversal}'")
    while stack:
        if not isinstance(stack, deque):
            key, depth, obj = stack.pop()
        else:
            key, depth, obj = stack.popleft()
        if match_func(key, obj, **kwargs):
            return True
        if max_depth is not None and depth >= max_depth:
            continue
        if excl_types not in (None, False) and checks.is_instance_of(obj, excl_types):
            if incl_types is None or not (incl_types is True or checks.is_instance_of(obj, incl_types)):
                continue
        if isinstance(obj, dict):
            if max_len is not None and len(obj) > max_len:
                continue
            obj_items = obj.items()
            if not isinstance(stack, deque):
                obj_items = reversed(obj_items)
            for k, v in obj_items:
                new_key = combine_pathlike_keys(key, k, minimize=True)
                stack.append((new_key, depth + 1, v))
        elif isinstance(obj, (tuple, list, set, frozenset)):
            if max_len is not None and len(obj) > max_len:
                continue
            if isinstance(obj, (set, frozenset)):
                obj = list(obj)
            obj_len = len(obj)
            if not isinstance(stack, deque):
                obj = reversed(obj)
            for i, v in enumerate(obj):
                if not isinstance(stack, deque):
                    i = obj_len - 1 - i
                new_key = combine_pathlike_keys(key, i, minimize=True)
                stack.append((new_key, depth + 1, v))
    return False


def find_in_obj(
    obj: tp.Any,
    match_func: tp.Callable,
    traversal: tp.Optional[str] = None,
    excl_types: tp.Union[None, bool, tp.MaybeSequence[type]] = None,
    incl_types: tp.Union[None, bool, tp.MaybeSequence[type]] = None,
    max_len: tp.Optional[int] = None,
    max_depth: tp.Optional[int] = None,
    **kwargs,
) -> tp.PathDict:
    """Find matches in an object in an iterative manner.

    Traverses dicts, tuples, lists and (frozen-)sets. Does not look for matches in keys.

    Argument `traversal` can be "DFS" for depth-first search or "BFS" for breadth-first search.

    If `excl_types` is not None, uses `vectorbtpro.utils.checks.is_instance_of` to check whether
    the object is one of the types that are blacklisted. If so, the object is simply returned.
    Same for `incl_types` for whitelisting, but it has a priority over `excl_types`.

    If `max_len` is not None, processes any object only if it's shorter than the specified length.

    If `max_depth` is not None, processes any object only up to a certain recursion level.
    Level of 0 means dicts and other iterables are not processed, only matches are expected.

    Returns a map of keys (multiple levels get represented by a tuple) to their respective values.

    For defaults, see `vectorbtpro._settings.search`."""
    from vectorbtpro._settings import settings

    search_cfg = settings["search"]

    if traversal is None:
        traversal = search_cfg["traversal"]
    if excl_types is None:
        excl_types = search_cfg["excl_types"]
    if isinstance(excl_types, bool) and excl_types:
        raise ValueError("Argument excl_types cannot be True")
    if incl_types is None:
        incl_types = search_cfg["incl_types"]
    if isinstance(incl_types, bool) and not incl_types:
        raise ValueError("Argument incl_types cannot be False")
    if max_len is None:
        max_len = search_cfg["max_len"]
    if max_depth is None:
        max_depth = search_cfg["max_depth"]

    path_dct = {}
    if traversal.upper() == "DFS":
        stack = [(None, 0, obj)]
    elif traversal.upper() == "BFS":
        stack = deque([(None, 0, obj)])
    else:
        raise ValueError(f"Invalid traversal: '{traversal}'")
    while stack:
        if not isinstance(stack, deque):
            key, depth, obj = stack.pop()
        else:
            key, depth, obj = stack.popleft()
        if match_func(key, obj, **kwargs):
            path_dct[key] = obj
            continue
        if max_depth is not None and depth >= max_depth:
            continue
        if excl_types not in (None, False) and checks.is_instance_of(obj, excl_types):
            if incl_types is None or not (incl_types is True or checks.is_instance_of(obj, incl_types)):
                continue
        if isinstance(obj, dict):
            if max_len is not None and len(obj) > max_len:
                continue
            obj_items = obj.items()
            if not isinstance(stack, deque):
                obj_items = reversed(obj_items)
            for k, v in obj_items:
                new_key = combine_pathlike_keys(key, k, minimize=True)
                stack.append((new_key, depth + 1, v))
        elif isinstance(obj, (tuple, list, set, frozenset)):
            if max_len is not None and len(obj) > max_len:
                continue
            if isinstance(obj, (set, frozenset)):
                obj = list(obj)
            obj_len = len(obj)
            if not isinstance(stack, deque):
                obj = reversed(obj)
            for i, v in enumerate(obj):
                if not isinstance(stack, deque):
                    i = obj_len - 1 - i
                new_key = combine_pathlike_keys(key, i, minimize=True)
                stack.append((new_key, depth + 1, v))
    return path_dct


def replace_in_obj(obj: tp.Any, path_dct: tp.PathDict, _key: tp.Optional[tp.Hashable] = None) -> tp.Any:
    """Replace matches in an object in a recursive manner using a path dictionary.

    Keys in the path dictionary can be path-like keys."""
    if len(path_dct) == 0:
        return obj
    path_dct = {minimize_pathlike_key(k): v for k, v in path_dct.items()}
    if _key in path_dct:
        return path_dct[_key]

    if isinstance(obj, dict):
        new_obj = {}
        for k in obj:
            if k in path_dct:
                new_obj[k] = path_dct.pop(k)
            else:
                new_path_dct = {}
                for k2 in list(path_dct.keys()):
                    if isinstance(k2, tuple) and k2[0] == k:
                        new_k2 = k2[1:] if len(k2) > 2 else k2[1]
                        new_path_dct[new_k2] = path_dct.pop(k2)
                if len(new_path_dct) == 0:
                    new_obj[k] = obj[k]
                else:
                    new_key = combine_pathlike_keys(_key, k, minimize=True)
                    new_obj[k] = replace_in_obj(obj[k], new_path_dct, _key=new_key)
        return new_obj
    if isinstance(obj, (tuple, list, set, frozenset)):
        if isinstance(obj, list):
            obj_list = obj
        else:
            obj_list = list(obj)
        new_obj = []
        for i in range(len(obj_list)):
            if i in path_dct:
                new_obj.append(path_dct.pop(i))
            else:
                new_path_dct = {}
                for k2 in list(path_dct.keys()):
                    if isinstance(k2, tuple) and k2[0] == i:
                        new_k2 = k2[1:] if len(k2) > 2 else k2[1]
                        new_path_dct[new_k2] = path_dct.pop(k2)
                if len(new_path_dct) == 0:
                    new_obj.append(obj_list[i])
                else:
                    new_key = combine_pathlike_keys(_key, i, minimize=True)
                    new_obj.append(replace_in_obj(obj_list[i], new_path_dct, _key=new_key))
        if checks.is_namedtuple(obj):
            return type(obj)(*new_obj)
        return type(obj)(new_obj)
    return obj


def find_and_replace_in_obj(
    obj: tp.Any,
    match_func: tp.Callable,
    replace_func: tp.Callable,
    excl_types: tp.Union[None, bool, tp.MaybeSequence[type]] = None,
    incl_types: tp.Union[None, bool, tp.MaybeSequence[type]] = None,
    max_len: tp.Optional[int] = None,
    max_depth: tp.Optional[int] = None,
    make_copy: bool = True,
    check_any_first: bool = True,
    _key: tp.Optional[tp.Hashable] = None,
    _depth: int = 0,
    **kwargs,
) -> tp.Any:
    """Find and replace matches in an object in a recursive manner.

    See `find_in_obj` for arguments.

    !!! note
        If the object is deep (such as a dict or a list), creates a copy of it if any match found inside,
        thus losing the reference to the original. Make sure to do a deep or hybrid copy of the object
        before proceeding for consistent behavior, or disable `make_copy` to override the original in place.
    """
    from vectorbtpro._settings import settings

    search_cfg = settings["search"]

    if excl_types is None:
        excl_types = search_cfg["excl_types"]
    if isinstance(excl_types, bool) and excl_types:
        raise ValueError("Argument excl_types cannot be True")
    if incl_types is None:
        incl_types = search_cfg["incl_types"]
    if isinstance(incl_types, bool) and not incl_types:
        raise ValueError("Argument incl_types cannot be False")
    if max_len is None:
        max_len = search_cfg["max_len"]
    if max_depth is None:
        max_depth = search_cfg["max_depth"]

    if check_any_first and not contains_in_obj(
        obj,
        match_func,
        excl_types=excl_types,
        incl_types=incl_types,
        max_len=max_len,
        max_depth=max_depth,
        **kwargs,
    ):
        return obj

    if match_func(_key, obj, **kwargs):
        return replace_func(_key, obj, **kwargs)
    if max_depth is not None and _depth >= max_depth:
        return obj
    if excl_types not in (None, False) and checks.is_instance_of(obj, excl_types):
        if incl_types is None or not (incl_types is True or checks.is_instance_of(obj, incl_types)):
            return obj
    if isinstance(obj, dict):
        if max_len is not None and len(obj) > max_len:
            return obj
        if make_copy:
            obj = copy(obj)
        for k, v in obj.items():
            new_key = combine_pathlike_keys(_key, k, minimize=True)
            set_dict_item(
                obj,
                k,
                find_and_replace_in_obj(
                    v,
                    match_func,
                    replace_func,
                    excl_types=excl_types,
                    incl_types=incl_types,
                    max_len=max_len,
                    max_depth=max_depth,
                    make_copy=make_copy,
                    check_any_first=False,
                    _key=new_key,
                    _depth=_depth + 1,
                    **kwargs,
                ),
                force=True,
            )
        return obj
    if isinstance(obj, list):
        if max_len is not None and len(obj) > max_len:
            return obj
        if make_copy:
            obj = copy(obj)
        for i in range(len(obj)):
            new_key = combine_pathlike_keys(_key, i, minimize=True)
            obj[i] = find_and_replace_in_obj(
                obj[i],
                match_func,
                replace_func,
                excl_types=excl_types,
                incl_types=incl_types,
                max_len=max_len,
                max_depth=max_depth,
                make_copy=make_copy,
                check_any_first=False,
                _key=new_key,
                _depth=_depth + 1,
                **kwargs,
            )
        return obj
    if isinstance(obj, (tuple, set, frozenset)):
        if max_len is not None and len(obj) > max_len:
            return obj
        if isinstance(obj, list):
            obj_list = obj
        else:
            obj_list = list(obj)
        result = []
        for i, o in enumerate(obj_list):
            new_key = combine_pathlike_keys(_key, i, minimize=True)
            result.append(
                find_and_replace_in_obj(
                    o,
                    match_func,
                    replace_func,
                    excl_types=excl_types,
                    incl_types=incl_types,
                    max_len=max_len,
                    max_depth=max_depth,
                    make_copy=make_copy,
                    check_any_first=False,
                    _key=new_key,
                    _depth=_depth + 1,
                    **kwargs,
                )
            )
        if checks.is_namedtuple(obj):
            return type(obj)(*result)
        return type(obj)(result)
    return obj


def flatten_obj(
    obj: tp.Any,
    traversal: tp.Optional[str] = None,
    annotate_all: bool = False,
    excl_types: tp.Union[None, bool, tp.MaybeSequence[type]] = None,
    incl_types: tp.Union[None, bool, tp.MaybeSequence[type]] = None,
    max_len: tp.Optional[int] = None,
    max_depth: tp.Optional[int] = None,
) -> tp.PathDict:
    """Flatten object.

    Argument `traversal` can be "DFS" for depth-first search or "BFS" for breadth-first search.

    See `find_in_obj` for arguments."""
    from vectorbtpro._settings import settings

    search_cfg = settings["search"]

    if traversal is None:
        traversal = search_cfg["traversal"]
    if excl_types is None:
        excl_types = search_cfg["excl_types"]
    if isinstance(excl_types, bool) and excl_types:
        raise ValueError("Argument excl_types cannot be True")
    if incl_types is None:
        incl_types = search_cfg["incl_types"]
    if isinstance(incl_types, bool) and not incl_types:
        raise ValueError("Argument incl_types cannot be False")
    if max_len is None:
        max_len = search_cfg["max_len"]
    if max_depth is None:
        max_depth = search_cfg["max_depth"]

    path_dct = {}
    if traversal.upper() == "DFS":
        stack = [(None, 0, obj)]
    elif traversal.upper() == "BFS":
        stack = deque([(None, 0, obj)])
    else:
        raise ValueError(f"Invalid traversal: '{traversal}'")
    while stack:
        if not isinstance(stack, deque):
            key, depth, obj = stack.pop()
        else:
            key, depth, obj = stack.popleft()
        if max_depth is not None and depth >= max_depth:
            path_dct[key] = obj
            continue
        if excl_types not in (None, False) and checks.is_instance_of(obj, excl_types):
            if incl_types is None or not (incl_types is True or checks.is_instance_of(obj, incl_types)):
                path_dct[key] = obj
                continue
        if isinstance(obj, dict):
            if max_len is not None and len(obj) > max_len:
                path_dct[key] = obj
                continue
            if annotate_all:
                path_dct[key] = type(obj)
            obj_items = obj.items()
            if not isinstance(stack, deque):
                obj_items = reversed(obj_items)
            for k, v in obj_items:
                new_key = combine_pathlike_keys(key, k, minimize=True)
                stack.append((new_key, depth + 1, v))
        elif isinstance(obj, (tuple, list, set, frozenset)):
            if max_len is not None and len(obj) > max_len:
                path_dct[key] = obj
                continue
            if annotate_all or not isinstance(obj, list):
                path_dct[key] = type(obj)
            if isinstance(obj, (set, frozenset)):
                obj = list(obj)
            obj_len = len(obj)
            if not isinstance(stack, deque):
                obj = reversed(obj)
            for i, v in enumerate(obj):
                if not isinstance(stack, deque):
                    i = obj_len - 1 - i
                new_key = combine_pathlike_keys(key, i, minimize=True)
                stack.append((new_key, depth + 1, v))
        else:
            path_dct[key] = obj
    return path_dct


def unflatten_obj(path_dct: tp.PathDict) -> tp.Any:
    """Unflatten object in a recursive manner using a path dictionary.

    Keys in the path dictionary can be path-like keys."""
    path_dct = {resolve_pathlike_key(k): v for k, v in path_dct.items()}

    class _Leaf:
        def __init__(self, value):
            self.value = value

    def _build_tree(paths):
        tree = {}
        root_defined = False
        for path, value in paths.items():
            if path == ():
                if root_defined and isinstance(tree, _Leaf):
                    raise ValueError("Multiple root definitions detected")
                if isinstance(value, type):
                    tree = {"__type__": value}
                else:
                    if len(paths) > 1:
                        raise ValueError("Cannot have an empty tuple key alongside other keys")
                    tree = _Leaf(value)
                root_defined = True
                continue
            current = tree
            for key in path[:-1]:
                if not isinstance(current, dict):
                    raise ValueError(f"Conflicting path at {path[:path.index(key)+1]}")
                if key not in current:
                    current[key] = {}
                elif not isinstance(current[key], dict):
                    raise ValueError(f"Duplicate or conflicting key detected at path {path[:path.index(key)+1]}")
                current = current[key]
            last_key = path[-1]
            if last_key in current:
                raise ValueError(f"Duplicate key detected at path {path}")
            if isinstance(value, type):
                if "__type__" in current.get(last_key, {}):
                    if current[last_key]["__type__"] != value:
                        raise ValueError(f"Conflicting type specifications at path {path}")
                current.setdefault(last_key, {})["__type__"] = value
            else:
                current[last_key] = _Leaf(value)
        return tree

    def _construct(node):
        if isinstance(node, _Leaf):
            return node.value
        if not isinstance(node, dict):
            return node
        type_spec = node.pop("__type__", None)
        if not node:
            return type_spec()
        keys = node.keys()
        if all(isinstance(k, int) for k in keys):
            sorted_indices = sorted(keys)
            expected_indices = list(range(len(sorted_indices)))
            if sorted_indices != expected_indices:
                raise ValueError(f"{type_spec.__name__.capitalize()} indices must be contiguous starting from 0")
            container = [_construct(node[k]) for k in sorted(keys)]
        elif all(isinstance(k, str) for k in keys):
            container = {k: _construct(v) for k, v in node.items()}
        else:
            raise ValueError("Cannot mix integer and non-integer keys at the same level")
        if type_spec:
            return type_spec(container)
        return container

    tree = _build_tree(path_dct)
    return _construct(tree)


def contains_exact(
    string: tp.MaybeIterable[str],
    substring: str,
    ignore_case: bool = False,
) -> tp.Union[bool, tp.List[bool], tp.Series]:
    """Check if string contains a substring."""
    if not isinstance(string, str):
        if isinstance(string, pd.Series):
            return string.apply(
                partial(
                    contains_exact,
                    substring=substring,
                    ignore_case=ignore_case,
                )
            )
        return list(
            map(
                partial(
                    contains_exact,
                    substring=substring,
                    ignore_case=ignore_case,
                ),
                string,
            )
        )
    if ignore_case:
        string = string.casefold()
        substring = substring.casefold()
    return substring in string


def replace_exact(
    string: tp.MaybeIterable[str],
    substring: str,
    replacement: str,
    ignore_case: bool = False,
) -> tp.Union[str, tp.List[str], tp.Series]:
    """Replace a substring with replacement in string."""
    if not isinstance(string, str):
        if isinstance(string, pd.Series):
            return string.apply(
                partial(
                    replace_exact,
                    substring=substring,
                    replacement=replacement,
                    ignore_case=ignore_case,
                )
            )
        return list(
            map(
                partial(
                    replace_exact,
                    substring=substring,
                    replacement=replacement,
                    ignore_case=ignore_case,
                ),
                string,
            )
        )
    if ignore_case:
        pattern = re.compile(re.escape(substring), re.IGNORECASE)
        return pattern.sub(replacement, string)
    else:
        return string.replace(substring, replacement)


def contains_regex(
    string: tp.MaybeIterable[str],
    substring: str,
    ignore_case: bool = False,
    flags: int = 0,
) -> tp.Union[bool, tp.List[bool], tp.Series]:
    """Check if the string matches the given regex substring."""
    if ignore_case:
        flags = flags | re.IGNORECASE
    regex = re.compile(substring, flags=flags)

    if not isinstance(string, str):
        if isinstance(string, pd.Series):
            return string.apply(
                partial(
                    contains_regex,
                    substring=substring,
                    ignore_case=ignore_case,
                    flags=flags,
                )
            )
        return list(
            map(
                partial(
                    contains_regex,
                    substring=substring,
                    ignore_case=ignore_case,
                    flags=flags,
                ),
                string,
            )
        )
    return bool(regex.search(string))


def replace_regex(
    string: tp.MaybeIterable[str],
    pattern: str,
    replacement: str,
    ignore_case: bool = False,
    flags: int = 0,
) -> tp.Union[str, tp.List[str], tp.Series]:
    """Replace regex substring with replacement in string."""
    if ignore_case:
        flags = flags | re.IGNORECASE
    regex = re.compile(pattern, flags=flags)

    if not isinstance(string, str):
        if isinstance(string, pd.Series):
            return string.apply(
                partial(
                    replace_regex,
                    pattern=pattern,
                    replacement=replacement,
                    ignore_case=ignore_case,
                    flags=flags,
                )
            )
        return list(
            map(
                partial(
                    replace_regex,
                    pattern=pattern,
                    replacement=replacement,
                    ignore_case=ignore_case,
                    flags=flags,
                ),
                string,
            )
        )
    return regex.sub(replacement, string)


def contains_fuzzy(
    string: tp.MaybeIterable[str],
    substring: str,
    ignore_case: bool = False,
    threshold: tp.Optional[float] = 70,
    max_insertions: tp.Optional[int] = None,
    max_substitutions: tp.Optional[int] = None,
    max_deletions: tp.Optional[int] = None,
    max_l_dist: tp.Optional[int] = None,
) -> tp.Union[bool, tp.List[bool], tp.Series]:
    """Perform fuzzy matching between string and substring using fuzzysearch."""
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("fuzzysearch")
    from fuzzysearch import find_near_matches

    if not isinstance(string, str):
        if isinstance(string, pd.Series):
            return string.apply(
                partial(
                    contains_fuzzy,
                    substring=substring,
                    ignore_case=ignore_case,
                    threshold=threshold,
                    max_insertions=max_insertions,
                    max_substitutions=max_substitutions,
                    max_deletions=max_deletions,
                    max_l_dist=max_l_dist,
                )
            )
        return list(
            map(
                partial(
                    contains_fuzzy,
                    substring=substring,
                    ignore_case=ignore_case,
                    threshold=threshold,
                    max_insertions=max_insertions,
                    max_substitutions=max_substitutions,
                    max_deletions=max_deletions,
                    max_l_dist=max_l_dist,
                ),
                string,
            )
        )

    if ignore_case:
        string = string.casefold()
        substring = substring.casefold()
    if threshold is not None and max_l_dist is None:
        max_l_dist = max(1, len(substring) - int(len(substring) * (threshold / 100)))
    matches = find_near_matches(
        substring,
        string,
        max_insertions=max_insertions,
        max_substitutions=max_substitutions,
        max_deletions=max_deletions,
        max_l_dist=max_l_dist,
    )
    return len(matches) > 0


def replace_fuzzy(
    string: tp.MaybeIterable[str],
    substring: str,
    replacement: str,
    ignore_case: bool = False,
    threshold: tp.Optional[float] = 70,
    max_insertions: tp.Optional[int] = None,
    max_substitutions: tp.Optional[int] = None,
    max_deletions: tp.Optional[int] = None,
    max_l_dist: tp.Optional[int] = None,
) -> tp.Union[str, tp.List[str], tp.Series]:
    """Perform fuzzy matching and replacement between string and substring using fuzzysearch."""
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("fuzzysearch")
    from fuzzysearch import find_near_matches

    if not isinstance(string, str):
        if isinstance(string, pd.Series):
            return string.apply(
                partial(
                    replace_fuzzy,
                    substring=substring,
                    replacement=replacement,
                    ignore_case=ignore_case,
                    threshold=threshold,
                    max_insertions=max_insertions,
                    max_substitutions=max_substitutions,
                    max_deletions=max_deletions,
                    max_l_dist=max_l_dist,
                )
            )
        return list(
            map(
                partial(
                    replace_fuzzy,
                    substring=substring,
                    replacement=replacement,
                    ignore_case=ignore_case,
                    threshold=threshold,
                    max_insertions=max_insertions,
                    max_substitutions=max_substitutions,
                    max_deletions=max_deletions,
                    max_l_dist=max_l_dist,
                ),
                string,
            )
        )

    original_string = string
    if ignore_case:
        string = string.casefold()
        substring = substring.casefold()
    else:
        string = string
        substring = substring
    if threshold is not None and max_l_dist is None:
        max_l_dist = max(1, len(substring) - int(len(substring) * (threshold / 100)))
    matches = find_near_matches(
        substring,
        string,
        max_insertions=max_insertions,
        max_substitutions=max_substitutions,
        max_deletions=max_deletions,
        max_l_dist=max_l_dist,
    )
    if len(matches) == 0:
        return original_string
    matches_sorted = sorted(matches, key=lambda m: m.start)
    replaced_string = ""
    last_idx = 0
    for match in matches_sorted:
        replaced_string += original_string[match.start : match.end]
        replaced_string += replacement
        last_idx = match.end
    replaced_string += original_string[last_idx:]
    return replaced_string


def contains_rapidfuzz(
    string: tp.MaybeIterable[str],
    substring: str,
    ignore_case: bool = False,
    processor: tp.Optional[tp.Callable] = None,
    threshold: float = 70,
) -> tp.Union[bool, tp.List[bool], tp.Series]:
    """Perform fuzzy matching between string and substring using RapidFuzz."""
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("rapidfuzz")
    from rapidfuzz import fuzz

    if not isinstance(string, str):
        if isinstance(string, pd.Series):
            return string.apply(
                partial(
                    contains_rapidfuzz,
                    substring=substring,
                    ignore_case=ignore_case,
                    processor=processor,
                    threshold=threshold,
                )
            )
        return list(
            map(
                partial(
                    contains_rapidfuzz,
                    substring=substring,
                    ignore_case=ignore_case,
                    processor=processor,
                    threshold=threshold,
                ),
                string,
            )
        )
    if ignore_case:
        string = string.casefold()
        substring = substring.casefold()
    score = fuzz.partial_ratio(string, substring, processor=processor)
    return score >= threshold


def contains(
    string: tp.MaybeIterable[str],
    substring: str,
    mode: str = "exact",
    ignore_case: bool = False,
    **kwargs,
) -> tp.Union[bool, tp.List[bool], tp.Series]:
    """Search for a target string within a source string using the specified mode."""
    if mode.lower() == "exact":
        return contains_exact(string, substring, ignore_case=ignore_case, **kwargs)
    elif mode.lower() == "regex":
        return contains_regex(string, substring, ignore_case=ignore_case, **kwargs)
    elif mode.lower() == "fuzzy":
        return contains_fuzzy(string, substring, ignore_case=ignore_case, **kwargs)
    elif mode.lower() == "rapidfuzz":
        return contains_rapidfuzz(string, substring, ignore_case=ignore_case, **kwargs)
    else:
        raise ValueError(f"Invalid mode: '{mode}'")


def replace(
    string: tp.MaybeIterable[str],
    substring: str,
    replacement: str,
    mode: str = "exact",
    ignore_case: bool = False,
    **kwargs,
) -> tp.Union[bool, tp.List[bool], tp.Series]:
    """Search for a target string within a source string using the specified mode."""
    if mode.lower() == "exact":
        return replace_exact(string, substring, replacement, ignore_case=ignore_case, **kwargs)
    elif mode.lower() == "regex":
        return replace_regex(string, substring, replacement, ignore_case=ignore_case, **kwargs)
    elif mode.lower() == "fuzzy":
        return replace_fuzzy(string, substring, replacement, ignore_case=ignore_case, **kwargs)
    elif mode.lower() == "rapidfuzz":
        raise ValueError("RapidFuzz doesn't support replacement")
    else:
        raise ValueError(f"Invalid mode: '{mode}'")


search_config = ReadonlyConfig(
    {
        "contains_exact": contains_exact,
        "contains_regex": contains_regex,
        "contains_fuzzy": contains_fuzzy,
        "contains_rapidfuzz": contains_rapidfuzz,
        "contains": contains,
        "replace_exact": replace_exact,
        "replace_regex": replace_regex,
        "replace_fuzzy": replace_fuzzy,
        "replace": replace,
    }
)
"""_"""

__pdoc__[
    "search_config"
] = f"""Config of functions that can be used in searching and replacement.

```python
{search_config.prettify()}
```
"""
