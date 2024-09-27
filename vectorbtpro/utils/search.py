# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for searching."""

import re
from copy import copy
from functools import partial
from collections import deque

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import set_dict_item


def any_in_obj(
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
        raise ValueError(f"Invalid option traversal='{traversal}'")
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
            if max_len is None or len(obj) <= max_len:
                obj_items = obj.items()
                if not isinstance(stack, deque):
                    obj_items = reversed(obj_items)
                for k, v in obj_items:
                    new_key = k if key is None else (*key, k) if isinstance(key, tuple) else (key, k)
                    stack.append((new_key, depth + 1, v))
        if isinstance(obj, (tuple, list, set, frozenset)):
            if max_len is None or len(obj) <= max_len:
                if isinstance(obj, (set, frozenset)):
                    obj = list(obj)
                obj_len = len(obj)
                if not isinstance(stack, deque):
                    obj = reversed(obj)
                for i, v in enumerate(obj):
                    if not isinstance(stack, deque):
                        i = obj_len - 1 - i
                    new_key = i if key is None else (*key, i) if isinstance(key, tuple) else (key, i)
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
) -> dict:
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

    match_dct = {}
    if traversal.upper() == "DFS":
        stack = [(None, 0, obj)]
    elif traversal.upper() == "BFS":
        stack = deque([(None, 0, obj)])
    else:
        raise ValueError(f"Invalid option traversal='{traversal}'")
    while stack:
        if not isinstance(stack, deque):
            key, depth, obj = stack.pop()
        else:
            key, depth, obj = stack.popleft()
        if match_func(key, obj, **kwargs):
            match_dct[key] = obj
            continue
        if max_depth is not None and depth >= max_depth:
            continue
        if excl_types not in (None, False) and checks.is_instance_of(obj, excl_types):
            if incl_types is None or not (incl_types is True or checks.is_instance_of(obj, incl_types)):
                continue
        if isinstance(obj, dict):
            if max_len is None or len(obj) <= max_len:
                obj_items = obj.items()
                if not isinstance(stack, deque):
                    obj_items = reversed(obj_items)
                for k, v in obj_items:
                    new_key = k if key is None else (*key, k) if isinstance(key, tuple) else (key, k)
                    stack.append((new_key, depth + 1, v))
        if isinstance(obj, (tuple, list, set, frozenset)):
            if max_len is None or len(obj) <= max_len:
                if isinstance(obj, (set, frozenset)):
                    obj = list(obj)
                obj_len = len(obj)
                if not isinstance(stack, deque):
                    obj = reversed(obj)
                for i, v in enumerate(obj):
                    if not isinstance(stack, deque):
                        i = obj_len - 1 - i
                    new_key = i if key is None else (*key, i) if isinstance(key, tuple) else (key, i)
                    stack.append((new_key, depth + 1, v))
    return match_dct


def replace_in_obj(obj: tp.Any, match_dct: dict, _key: tp.Optional[tp.Hashable] = None) -> tp.Any:
    """Replace matches in an object in a recursive manner.

    See `find_in_obj` for `match_dct` (returned value)."""
    if len(match_dct) == 0:
        return obj
    if _key in match_dct:
        return match_dct[_key]
    match_dct = dict(match_dct)

    if isinstance(obj, dict):
        new_obj = {}
        for k in obj:
            if k in match_dct:
                new_obj[k] = match_dct.pop(k)
            else:
                new_match_dct = {}
                for k2 in list(match_dct.keys()):
                    if isinstance(k2, tuple) and k2[0] == k:
                        new_k2 = k2[1:] if len(k2) > 2 else k2[1]
                        new_match_dct[new_k2] = match_dct.pop(k2)
                if len(new_match_dct) == 0:
                    new_obj[k] = obj[k]
                else:
                    new_key = k if _key is None else (*_key, k) if isinstance(_key, tuple) else (_key, k)
                    new_obj[k] = replace_in_obj(obj[k], new_match_dct, _key=new_key)
        return new_obj
    if isinstance(obj, (tuple, list, set, frozenset)):
        new_obj = []
        for i in range(len(obj)):
            if i in match_dct:
                new_obj.append(match_dct.pop(i))
            else:
                new_match_dct = {}
                for k2 in list(match_dct.keys()):
                    if isinstance(k2, tuple) and k2[0] == i:
                        new_k2 = k2[1:] if len(k2) > 2 else k2[1]
                        new_match_dct[new_k2] = match_dct.pop(k2)
                if len(new_match_dct) == 0:
                    new_obj.append(obj[i])
                else:
                    new_key = i if _key is None else (*_key, i) if isinstance(_key, tuple) else (_key, i)
                    new_obj.append(replace_in_obj(obj[i], new_match_dct, _key=new_key))
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

    if check_any_first and not any_in_obj(
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
    if max_depth is None or _depth < max_depth:
        if excl_types not in (None, False) and checks.is_instance_of(obj, excl_types):
            if incl_types is None or not (incl_types is True or checks.is_instance_of(obj, incl_types)):
                return obj
        if isinstance(obj, dict):
            if max_len is None or len(obj) <= max_len:
                if make_copy:
                    obj = copy(obj)
                for k, v in obj.items():
                    new_key = k if _key is None else (*_key, k) if isinstance(_key, tuple) else (_key, k)
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
            if max_len is None or len(obj) <= max_len:
                if make_copy:
                    obj = copy(obj)
                for i in range(len(obj)):
                    new_key = i if _key is None else (*_key, i) if isinstance(_key, tuple) else (_key, i)
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
            if max_len is None or len(obj) <= max_len:
                result = []
                for i, o in enumerate(obj):
                    new_key = i if _key is None else (*_key, i) if isinstance(_key, tuple) else (_key, i)
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


def search_text(string: tp.MaybeIterable[str], query: str, ignore_case: bool = False) -> tp.MaybeList[bool]:
    """Check if query is a substring of string."""
    if not isinstance(string, str):
        if isinstance(string, pd.Series):
            return string.apply(
                partial(
                    search_text,
                    query=query,
                    ignore_case=ignore_case,
                )
            )
        return list(
            map(
                partial(
                    search_text,
                    query=query,
                    ignore_case=ignore_case,
                ),
                string,
            )
        )
    if ignore_case:
        string = string.casefold()
        query = query.casefold()
    return query in string


def search_regex(
    string: tp.MaybeIterable[str],
    pattern: str,
    ignore_case: bool = False,
    flags: int = 0,
) -> tp.MaybeList[bool]:
    """Check if the string string matches the given regex pattern."""
    if ignore_case:
        flags = flags | re.IGNORECASE
    regex = re.compile(pattern, flags=flags)
    if not isinstance(string, str):
        if isinstance(string, pd.Series):
            return string.apply(
                partial(
                    search_regex,
                    pattern=pattern,
                    ignore_case=ignore_case,
                    flags=flags,
                )
            )
        return list(
            map(
                partial(
                    search_regex,
                    pattern=pattern,
                    ignore_case=ignore_case,
                    flags=flags,
                ),
                string,
            )
        )
    return bool(regex.search(string))


def search_fuzzy(
    string: tp.MaybeIterable[str],
    query: str,
    ignore_case: bool = False,
    processor: tp.Optional[tp.Callable] = None,
    threshold: float = 70,
) -> tp.MaybeList[bool]:
    """Perform fuzzy matching between string and query using RapidFuzz."""
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("rapidfuzz")
    from rapidfuzz import fuzz

    if not isinstance(string, str):
        if isinstance(string, pd.Series):
            return string.apply(
                partial(
                    search_fuzzy,
                    query=query,
                    ignore_case=ignore_case,
                    processor=processor,
                    threshold=threshold,
                )
            )
        return list(
            map(
                partial(
                    search_fuzzy,
                    query=query,
                    ignore_case=ignore_case,
                    processor=processor,
                    threshold=threshold,
                ),
                string,
            )
        )
    if ignore_case:
        string = string.casefold()
        query = query.casefold()
    score = fuzz.partial_ratio(string, query, processor=processor)
    return score >= threshold
