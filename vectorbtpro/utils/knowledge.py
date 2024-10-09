# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for working with knowledge.

Run for the examples:

```pycon
>>> dataset = [
...     {"s": "ABC", "b": True, "d2": {"c": "red", "l": [1, 2]}},
...     {"s": "BCD", "b": True, "d2": {"c": "blue", "l": [3, 4]}},
...     {"s": "CDE", "b": False, "d2": {"c": "green", "l": [5, 6]}},
...     {"s": "DEF", "b": False, "d2": {"c": "yellow", "l": [7, 8]}},
...     {"s": "EFG", "b": False, "d2": {"c": "black", "l": [9, 10]}, "xyz": 123}
... ]
>>> asset = vbt.KnowledgeAsset(dataset)
```
"""

import sys
import ast
import os
import io
import json
from pathlib import Path
import requests
import builtins
import importlib

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks, search
from vectorbtpro.utils.config import Configured, reorder_dict, reorder_list
from vectorbtpro.utils.pickling import suggest_compression, decompress, load_bytes
from vectorbtpro.utils.path_ import check_mkdir
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.template import CustomTemplate, RepEval, RepFunc, substitute_templates
from vectorbtpro.utils.config import merge_dicts, flat_merge_dicts, deep_merge_dicts
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.execution import Task, execute, NoResult
from vectorbtpro.utils.module_ import parse_refname, get_caller_qualname, package_shortcut_config
from vectorbtpro.utils.eval_ import evaluate

__all__ = [
    "AssetFunc",
    "AssetPipeline",
    "BasicAssetPipeline",
    "ComplexAssetPipeline",
    "KnowledgeAsset",
    "ReleaseAsset",
    "MessagesAsset",
    "PagesAsset",
]


KnowledgeAssetT = tp.TypeVar("KnowledgeAssetT", bound="KnowledgeAsset")


class AssetFunc:
    """Abstract class representing a function to be applied to a data item."""

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the function to be used in expressions."""

    _wrap: tp.ClassVar[tp.Optional[str]] = None
    """Whether the results are meant to be wrapped with `KnowledgeAsset`."""

    @classmethod
    def prepare_args(cls, *args, **kwargs) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments."""
        raise NotImplementedError

    @classmethod
    def func(cls, d: tp.Any, *args, **kwargs) -> tp.Any:
        """Function to be applied to a data item."""
        raise NotImplementedError


class GetAssetFunc(AssetFunc):
    """Asset function class for `KnowledgeAsset.get`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "get"

    _wrap: tp.ClassVar[tp.Optional[str]] = False

    @classmethod
    def prepare_args(
        cls,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        source: tp.Union[None, str, tp.Callable, tp.CustomTemplate] = None,
        template_context: tp.KwargsLike = None,
        asset: tp.Optional[tp.MaybeType["KnowledgeAsset"]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            asset = KnowledgeAsset
        keep_path = asset.resolve_setting(keep_path, "keep_path")
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        template_context = asset.resolve_setting(template_context, "template_context", merge=True)

        if path is not None:
            if isinstance(path, list):
                path = [search.resolve_pathlike_key(p) for p in path]
            else:
                path = search.resolve_pathlike_key(path)
        if source is not None:
            if isinstance(source, str):
                source = RepEval(source)
            elif checks.is_function(source):
                if checks.is_builtin_func(source):
                    source = RepFunc(lambda _source=source: _source)
                else:
                    source = RepFunc(source)
            elif not isinstance(source, CustomTemplate):
                raise TypeError(f"Source must be a template")
        return (), {
            **dict(
                path=path,
                keep_path=keep_path,
                skip_missing=skip_missing,
                source=source,
                template_context=template_context,
            ),
            **kwargs,
        }

    @classmethod
    def func(
        cls,
        d: tp.Any,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        keep_path: bool = False,
        skip_missing: bool = False,
        source: tp.Optional[tp.CustomTemplate] = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.Any:
        x = d
        if path is not None:
            if isinstance(path, list):
                xs = []
                for p in path:
                    try:
                        xs.append(search.get_pathlike_key(x, p, keep_path=True))
                    except (KeyError, IndexError, AttributeError) as e:
                        if not skip_missing:
                            raise e
                        continue
                if len(xs) == 0:
                    return NoResult
                x = deep_merge_dicts(*xs)
            else:
                try:
                    x = search.get_pathlike_key(x, path, keep_path=keep_path)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    return NoResult
        if source is not None:
            _template_context = flat_merge_dicts(
                {
                    "d": d,
                    "x": x,
                    **(x if isinstance(x, dict) else {}),
                },
                template_context,
            )
            new_d = source.substitute(_template_context, eval_id="source")
            if checks.is_function(new_d):
                new_d = new_d(x)
        else:
            new_d = x
        return new_d


class SetAssetFunc(AssetFunc):
    """Asset function class for `KnowledgeAsset.set`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "set"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare_args(
        cls,
        value: tp.Any,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        asset: tp.Optional[tp.MaybeType["KnowledgeAsset"]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            asset = KnowledgeAsset
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")
        template_context = asset.resolve_setting(template_context, "template_context", merge=True)

        if checks.is_function(value):
            if checks.is_builtin_func(value):
                value = RepFunc(lambda _value=value: _value)
            else:
                value = RepFunc(value)
        if path is not None:
            if isinstance(path, list):
                paths = [search.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search.resolve_pathlike_key(path)]
        else:
            paths = [None]
        return (), {
            **dict(
                value=value,
                paths=paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
                template_context=template_context,
            ),
            **kwargs,
        }

    @classmethod
    def func(
        cls,
        d: tp.Any,
        value: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            x = d
            if p is not None:
                try:
                    x = search.get_pathlike_key(x, p[:-1])
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            _template_context = flat_merge_dicts(
                {
                    "d": d,
                    "x": x,
                    **(x if isinstance(x, dict) else {}),
                },
                template_context,
            )
            v = value.substitute(_template_context, eval_id="value", **kwargs)
            if checks.is_function(v):
                v = v(x)
            d = search.set_pathlike_key(d, p, v, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class RemoveAssetFunc(AssetFunc):
    """Asset function class for `KnowledgeAsset.remove`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "remove"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare_args(
        cls,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset: tp.Optional[tp.MaybeType["KnowledgeAsset"]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            asset = KnowledgeAsset
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")

        if isinstance(path, list):
            paths = [search.resolve_pathlike_key(p) for p in path]
        else:
            paths = [search.resolve_pathlike_key(path)]
        return (), {
            **dict(
                paths=paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
            ),
            **kwargs,
        }

    @classmethod
    def func(
        cls,
        d: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            try:
                d = search.remove_pathlike_key(d, p, make_copy=make_copy, prev_keys=prev_keys)
            except (KeyError, IndexError, AttributeError) as e:
                if not skip_missing:
                    raise e
                continue
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class MoveAssetFunc(AssetFunc):
    """Asset function class for `KnowledgeAsset.move`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "move"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare_args(
        cls,
        path: tp.Union[tp.PathMoveDict, tp.MaybeList[tp.PathLikeKey]],
        new_path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset: tp.Optional[tp.MaybeType["KnowledgeAsset"]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            asset = KnowledgeAsset
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")

        if new_path is None:
            checks.assert_instance_of(path, dict, arg_name="path")
            new_path = list(path.values())
            path = list(path.keys())
        if isinstance(path, list):
            paths = [search.resolve_pathlike_key(p) for p in path]
        else:
            paths = [search.resolve_pathlike_key(path)]
        if isinstance(new_path, list):
            new_paths = [search.resolve_pathlike_key(p) for p in new_path]
        else:
            new_paths = [search.resolve_pathlike_key(new_path)]
        if len(paths) != len(new_paths):
            raise ValueError("Number of new paths must match number of paths")
        return (), {
            **dict(
                paths=paths,
                new_paths=new_paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
            ),
            **kwargs,
        }

    @classmethod
    def func(
        cls,
        d: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        new_paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
    ) -> tp.Any:
        prev_keys = []
        for i, p in enumerate(paths):
            try:
                x = search.get_pathlike_key(d, p)
                d = search.remove_pathlike_key(d, p, make_copy=make_copy, prev_keys=prev_keys)
                d = search.set_pathlike_key(d, new_paths[i], x, make_copy=make_copy, prev_keys=prev_keys)
            except (KeyError, IndexError, AttributeError) as e:
                if not skip_missing:
                    raise e
                continue
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class RenameAssetFunc(MoveAssetFunc):
    """Asset function class for `KnowledgeAsset.rename`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "rename"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare_args(
        cls,
        path: tp.Union[tp.PathRenameDict, tp.MaybeList[tp.PathLikeKey]],
        new_token: tp.Optional[tp.MaybeList[tp.PathKeyToken]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset: tp.Optional[tp.MaybeType["KnowledgeAsset"]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            asset = KnowledgeAsset
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")

        if new_token is None:
            checks.assert_instance_of(path, dict, arg_name="path")
            new_token = list(path.values())
            path = list(path.keys())
        if isinstance(path, list):
            paths = [search.resolve_pathlike_key(p) for p in path]
        else:
            paths = [search.resolve_pathlike_key(path)]
        if isinstance(new_token, list):
            new_tokens = [search.resolve_pathlike_key(t) for t in new_token]
        else:
            new_tokens = [search.resolve_pathlike_key(new_token)]
        if len(paths) != len(new_tokens):
            raise ValueError("Number of new tokens must match number of paths")
        new_paths = []
        for i in range(len(paths)):
            if len(new_tokens[i]) != 1:
                raise ValueError("Exactly one token must be provided for each path")
            new_paths.append(paths[i][:-1] + new_tokens[i])
        return (), {
            **dict(
                paths=paths,
                new_paths=new_paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
            ),
            **kwargs,
        }


class ReorderAssetFunc(AssetFunc):
    """Asset function class for `KnowledgeAsset.reorder`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "reorder"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare_args(
        cls,
        new_order: tp.Union[str, tp.PathKeyTokens],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        asset: tp.Optional[tp.MaybeType["KnowledgeAsset"]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            asset = KnowledgeAsset
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")
        template_context = asset.resolve_setting(template_context, "template_context", merge=True)

        if isinstance(new_order, str):
            if new_order.lower() in ("asc", "ascending"):
                new_order = lambda x: (
                    sorted(x)
                    if isinstance(x, dict)
                    else sorted(
                        range(len(x)),
                        key=x.__getitem__,
                    )
                )
            elif new_order.lower() in ("desc", "descending"):
                new_order = lambda x: (
                    sorted(x)
                    if isinstance(x, dict)
                    else sorted(
                        range(len(x)),
                        key=x.__getitem__,
                        reverse=True,
                    )
                )
        if isinstance(new_order, str):
            new_order = RepEval(new_order)
        elif checks.is_function(new_order):
            if checks.is_builtin_func(new_order):
                new_order = RepFunc(lambda _new_order=new_order: _new_order)
            else:
                new_order = RepFunc(new_order)
        if path is not None:
            if isinstance(path, list):
                paths = [search.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search.resolve_pathlike_key(path)]
        else:
            paths = [None]
        return (), {
            **dict(
                new_order=new_order,
                paths=paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
                template_context=template_context,
            ),
            **kwargs,
        }

    @classmethod
    def func(
        cls,
        d: tp.Any,
        new_order: tp.Union[tp.PathKeyTokens, tp.CustomTemplate],
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            x = d
            if p is not None:
                try:
                    x = search.get_pathlike_key(x, p)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            if isinstance(new_order, CustomTemplate):
                _template_context = flat_merge_dicts(
                    {
                        "d": d,
                        "x": x,
                        **(x if isinstance(x, dict) else {}),
                    },
                    template_context,
                )
                _new_order = new_order.substitute(_template_context, eval_id="new_order", **kwargs)
                if checks.is_function(_new_order):
                    _new_order = _new_order(x)
            else:
                _new_order = new_order
            if isinstance(x, dict):
                x = reorder_dict(x, _new_order, skip_missing=skip_missing)
            else:
                if checks.is_namedtuple(x):
                    x = type(x)(*reorder_list(x, _new_order, skip_missing=skip_missing))
                else:
                    x = type(x)(reorder_list(x, _new_order, skip_missing=skip_missing))
            d = search.set_pathlike_key(d, p, x, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class QueryAssetFunc(AssetFunc):
    """Asset function class for `KnowledgeAsset.query`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "query"

    _wrap: tp.ClassVar[tp.Optional[str]] = False

    @classmethod
    def prepare_args(
        cls,
        expression: tp.Union[str, tp.Callable, tp.CustomTemplate],
        as_filter: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        asset: tp.Optional[tp.MaybeType["KnowledgeAsset"]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            asset = KnowledgeAsset
        as_filter = asset.resolve_setting(as_filter, "as_filter")
        template_context = asset.resolve_setting(template_context, "template_context", merge=True)

        if isinstance(expression, str):
            expression = RepEval(expression)
        elif checks.is_function(expression):
            if checks.is_builtin_func(expression):
                expression = RepFunc(lambda _expression=expression: _expression)
            else:
                expression = RepFunc(expression)
        elif not isinstance(expression, CustomTemplate):
            raise TypeError(f"Expression must be a template")
        return (), {
            **dict(
                expression=expression,
                as_filter=as_filter,
                template_context=template_context,
            ),
            **kwargs,
        }

    @classmethod
    def func(
        cls,
        d: tp.Any,
        expression: tp.CustomTemplate,
        as_filter: bool = True,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        _template_context = flat_merge_dicts(
            {
                "d": d,
                "x": d,
                **search.search_config,
                **(d if isinstance(d, dict) else {}),
            },
            template_context,
        )
        new_d = expression.substitute(_template_context, eval_id="expression", **kwargs)
        if checks.is_function(new_d):
            new_d = new_d(d)
        if as_filter and isinstance(new_d, bool):
            if new_d:
                return d
            return NoResult
        return new_d


class FindAssetFunc(AssetFunc):
    """Asset function class for `KnowledgeAsset.find`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "find"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare_args(
        cls,
        target: tp.MaybeList[tp.Any],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: tp.Optional[bool] = None,
        find_any: tp.Optional[bool] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        source: tp.Union[None, str, tp.Callable, tp.CustomTemplate] = None,
        in_json_dumps: tp.Optional[bool] = None,
        as_filter: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        asset: tp.Optional[tp.MaybeType["KnowledgeAsset"]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            asset = KnowledgeAsset
        per_path = asset.resolve_setting(per_path, "per_path")
        find_any = asset.resolve_setting(find_any, "find_any")
        keep_path = asset.resolve_setting(keep_path, "keep_path")
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        in_json_dumps = asset.resolve_setting(in_json_dumps, "in_json_dumps")
        as_filter = asset.resolve_setting(as_filter, "as_filter")
        template_context = asset.resolve_setting(template_context, "template_context", merge=True)

        if path is not None:
            if isinstance(path, list):
                path = [search.resolve_pathlike_key(p) for p in path]
            else:
                path = search.resolve_pathlike_key(path)
        if per_path:
            if not isinstance(target, list):
                target = [target]
            if not isinstance(path, list):
                path = [path]
            if len(target) != len(path):
                raise ValueError("Number of targets must match number of paths")
        if source is not None:
            if isinstance(source, str):
                source = RepEval(source)
            elif checks.is_function(source):
                if checks.is_builtin_func(source):
                    source = RepFunc(lambda _source=source: _source)
                else:
                    source = RepFunc(source)
            elif not isinstance(source, CustomTemplate):
                raise TypeError(f"Source must be a template")
        if "excl_types" not in kwargs:
            kwargs["excl_types"] = (tuple, set, frozenset)
        return (), {
            **dict(
                target=target,
                path=path,
                per_path=per_path,
                find_any=find_any,
                keep_path=keep_path,
                skip_missing=skip_missing,
                source=source,
                in_json_dumps=in_json_dumps,
                as_filter=as_filter,
                template_context=template_context,
            ),
            **kwargs,
        }

    @classmethod
    def match_func(
        cls,
        k: tp.Optional[tp.Hashable],
        d: tp.Any,
        target: tp.MaybeList[tp.Any],
        find_any: bool = False,
        **kwargs,
    ) -> bool:
        """Match function for `FindAssetFunc.func`.

        Uses `vectorbtpro.utils.search.contains` for text and equality checks for other types."""
        if not isinstance(target, list):
            targets = [target]
        else:
            targets = target
        for target in targets:
            if d is target:
                if find_any:
                    return True
                continue
            if d is None and target is None:
                if find_any:
                    return True
                continue
            elif checks.is_bool(d) and checks.is_bool(target):
                if d == target:
                    if find_any:
                        return True
                    continue
            elif checks.is_number(d) and checks.is_number(target):
                if d == target:
                    if find_any:
                        return True
                    continue
            elif isinstance(d, str) and isinstance(target, str):
                if search.contains(d, target, **kwargs):
                    if find_any:
                        return True
                    continue
            elif type(d) is type(target):
                if d == target:
                    if find_any:
                        return True
                    continue
            if not find_any:
                return False
        if not find_any:
            return True
        return False

    @classmethod
    def func(
        cls,
        d: tp.Any,
        target: tp.MaybeList[tp.Any],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: bool = False,
        find_any: bool = False,
        keep_path: bool = False,
        skip_missing: bool = False,
        source: tp.Union[None, str, tp.Callable, tp.CustomTemplate] = None,
        in_json_dumps: bool = False,
        as_filter: bool = True,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        if per_path:
            for i, p in enumerate(path):
                x = d
                try:
                    x = search.get_pathlike_key(x, p, keep_path=keep_path)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
                if source is not None:
                    _template_context = flat_merge_dicts(
                        {
                            "d": d,
                            "x": x,
                            **(x if isinstance(x, dict) else {}),
                        },
                        template_context,
                    )
                    _x = source.substitute(_template_context, eval_id="source")
                    if checks.is_function(_x):
                        x = _x(x)
                    else:
                        x = _x
                if not isinstance(x, str) and in_json_dumps:
                    x = json.dumps(x, ensure_ascii=False)
                if search.contains_in_obj(
                    x,
                    cls.match_func,
                    target=target[i],
                    find_any=find_any,
                    **kwargs,
                ):
                    if find_any:
                        return d if as_filter else True
                    continue
                if not find_any:
                    return NoResult if as_filter else False
            if not find_any:
                return d if as_filter else True
            return NoResult if as_filter else False
        else:
            x = d
            if path is not None:
                if isinstance(path, list):
                    xs = []
                    for p in path:
                        try:
                            xs.append(search.get_pathlike_key(x, p, keep_path=True))
                        except (KeyError, IndexError, AttributeError) as e:
                            if not skip_missing:
                                raise e
                            continue
                    if len(xs) == 0:
                        return NoResult if as_filter else False
                    x = deep_merge_dicts(*xs)
                else:
                    try:
                        x = search.get_pathlike_key(x, path, keep_path=keep_path)
                    except (KeyError, IndexError, AttributeError) as e:
                        if not skip_missing:
                            raise e
                        return NoResult if as_filter else False
            if source is not None:
                _template_context = flat_merge_dicts(
                    {
                        "d": d,
                        "x": x,
                        **(x if isinstance(x, dict) else {}),
                    },
                    template_context,
                )
                _x = source.substitute(_template_context, eval_id="source")
                if checks.is_function(_x):
                    x = _x(x)
                else:
                    x = _x
            if not isinstance(x, str) and in_json_dumps:
                x = json.dumps(x, ensure_ascii=False)
            if search.contains_in_obj(
                x,
                cls.match_func,
                target=target,
                find_any=find_any,
                **kwargs,
            ):
                return d if as_filter else True
            return NoResult if as_filter else False


class ReplaceAssetFunc(FindAssetFunc):
    """Asset function class for `KnowledgeAsset.replace_`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "replace_"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare_args(
        cls,
        target: tp.MaybeList[tp.Any],
        replacement: tp.MaybeList[tp.Any],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: tp.Optional[bool] = None,
        replace_any: tp.Optional[bool] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset: tp.Optional[tp.MaybeType["KnowledgeAsset"]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            asset = KnowledgeAsset
        per_path = asset.resolve_setting(per_path, "per_path")
        replace_any = asset.resolve_setting(replace_any, "replace_any")
        keep_path = asset.resolve_setting(keep_path, "keep_path")
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")

        if path is not None:
            if isinstance(path, list):
                paths = [search.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search.resolve_pathlike_key(path)]
        else:
            paths = [None]
        if per_path:
            if not isinstance(target, list):
                target = [target]
            if not isinstance(replacement, list):
                replacement = [replacement]
            if len(target) != len(replacement) != len(paths):
                raise ValueError("Number of targets and replacements must match number of paths")
        find_arg_names = set(get_func_arg_names(search.find_in_obj))
        find_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in find_arg_names}
        if "excl_types" not in find_kwargs:
            find_kwargs["excl_types"] = (tuple, set, frozenset)
        return (), {
            **dict(
                target=target,
                replacement=replacement,
                paths=paths,
                per_path=per_path,
                replace_any=replace_any,
                keep_path=keep_path,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
                find_kwargs=find_kwargs,
            ),
            **kwargs,
        }

    @classmethod
    def replace_func(
        cls,
        k: tp.Optional[tp.Hashable],
        d: tp.Any,
        target: tp.MaybeList[tp.Any],
        replacement: tp.MaybeList[tp.Any],
        **kwargs,
    ) -> tp.Any:
        """Replace function for `ReplaceAssetFunc.func`.

        Uses `vectorbtpro.utils.search.replace` for text and returns replacement for other types."""
        if not isinstance(target, list):
            targets = [target]
        else:
            targets = target
        if not isinstance(replacement, list):
            replacements = [replacement]
        else:
            replacements = replacement
        if len(targets) > 1 and len(replacements) == 1:
            replacements *= len(targets)
        if len(targets) != len(replacements):
            raise ValueError("Number of targets must match number of replacements")
        new_d = d
        for i, target in enumerate(targets):
            if d is target:
                return replacements[i]
            if d is None and target is None:
                return replacements[i]
            elif checks.is_bool(d) and checks.is_bool(target):
                if d == target:
                    return replacements[i]
            elif checks.is_number(d) and checks.is_number(target):
                if d == target:
                    return replacements[i]
            elif isinstance(d, str) and isinstance(target, str):
                new_d = search.replace(new_d, target, replacements[i], **kwargs)
            elif type(d) is type(target):
                if d == target:
                    return replacements[i]
        return new_d

    @classmethod
    def func(
        cls,
        d: tp.Any,
        target: tp.MaybeList[tp.Any],
        replacement: tp.MaybeList[tp.Any],
        paths: tp.List[tp.PathLikeKey],
        per_path: bool = False,
        replace_any: bool = True,
        keep_path: bool = False,
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        find_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        if find_kwargs is None:
            find_kwargs = {}
        prev_keys = []
        new_p_v_map = {}
        for i, p in enumerate(paths):
            x = d
            if p is not None:
                try:
                    x = search.get_pathlike_key(x, p, keep_path=keep_path)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            path_dct = search.find_in_obj(
                x,
                cls.match_func,
                target=target[i] if per_path else target,
                find_any=replace_any,
                **find_kwargs,
                **kwargs,
            )
            if len(path_dct) == 0 and not replace_any:
                new_p_v_map = {}
                break
            for k, v in path_dct.items():
                if p is not None and not keep_path:
                    new_p = search.combine_pathlike_keys(p, k, minimize=True)
                else:
                    new_p = k
                v = cls.replace_func(
                    k,
                    v,
                    target[i] if per_path else target,
                    replacement[i] if per_path else replacement,
                    **kwargs,
                )
                new_p_v_map[new_p] = v
        for new_p, v in new_p_v_map.items():
            d = search.set_pathlike_key(d, new_p, v, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class FlattenAssetFunc(AssetFunc):
    """Asset function class for `KnowledgeAsset.flatten`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "flatten"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare_args(
        cls,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset: tp.Optional[tp.MaybeType["KnowledgeAsset"]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            asset = KnowledgeAsset
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")

        if path is not None:
            if isinstance(path, list):
                paths = [search.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search.resolve_pathlike_key(path)]
        else:
            paths = [None]
        if "excl_types" not in kwargs:
            kwargs["excl_types"] = (tuple, set, frozenset)
        return (), {
            **dict(
                paths=paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
            ),
            **kwargs,
        }

    @classmethod
    def func(
        cls,
        d: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        **kwargs,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            x = d
            if p is not None:
                try:
                    x = search.get_pathlike_key(x, p)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            x = search.flatten_obj(x, **kwargs)
            d = search.set_pathlike_key(d, p, x, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class UnflattenAssetFunc(AssetFunc):
    """Asset function class for `KnowledgeAsset.unflatten`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "unflatten"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare_args(
        cls,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset: tp.Optional[tp.MaybeType["KnowledgeAsset"]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            asset = KnowledgeAsset
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")

        if path is not None:
            if isinstance(path, list):
                paths = [search.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search.resolve_pathlike_key(path)]
        else:
            paths = [None]
        return (), {
            **dict(
                paths=paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
            ),
            **kwargs,
        }

    @classmethod
    def func(
        cls,
        d: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        **kwargs,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            x = d
            if p is not None:
                try:
                    x = search.get_pathlike_key(x, p)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            x = search.unflatten_obj(x, **kwargs)
            d = search.set_pathlike_key(d, p, x, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class AssetPipeline:
    """Abstract class representing an asset pipeline."""

    @classmethod
    def resolve_task(
        cls,
        func: tp.AssetFuncLike,
        *args,
        prepare_args: bool = True,
        prepare_once: bool = True,
        cond_kwargs: tp.KwargsLike = None,
        asset_func_meta: tp.Union[None, dict, list] = None,
        **kwargs,
    ) -> tp.Task:
        """Resolve a task."""
        if isinstance(func, tuple):
            func = Task.from_tuple(func)
        if isinstance(func, Task):
            args = (*func.args, *args)
            kwargs = merge_dicts(func.kwargs, kwargs)
            func = func.func
        if isinstance(func, str):
            if func in globals():
                func = globals()[func]
            elif func.title() + "AssetFunc" in globals():
                func = globals()[func.title() + "AssetFunc"]
            else:
                found_func = False
                for k, v in globals().items():
                    if isinstance(v, type) and issubclass(v, AssetFunc):
                        if v._short_name is not None:
                            if func.lower() == v._short_name.lower():
                                func = v
                                found_func = True
                if not found_func:
                    raise ValueError(f"Function '{func}' not found")
        if isinstance(func, AssetFunc):
            raise TypeError("Function must be a subclass of AssetFunc, not an instance")
        if isinstance(func, type) and issubclass(func, AssetFunc):
            _asset_func_meta = {
                "_short_name": getattr(func, "_short_name"),
                "_wrap": getattr(func, "_wrap"),
            }
            if asset_func_meta is not None:
                if isinstance(asset_func_meta, dict):
                    asset_func_meta.update(_asset_func_meta)
                else:
                    asset_func_meta.append(_asset_func_meta)
            if prepare_args:
                if prepare_once:
                    if cond_kwargs is None:
                        cond_kwargs = {}
                    if len(cond_kwargs) > 0:
                        prepare_args_arg_names = get_func_arg_names(func.prepare_args)
                        for k, v in cond_kwargs.items():
                            if k in prepare_args_arg_names:
                                kwargs[k] = v
                    args, kwargs = func.prepare_args(*args, **kwargs)
                    func = func.func
                else:

                    def func(d, *args, _func=func, **kwargs):
                        new_args, new_kwargs = _func.prepare_args(*args, **kwargs)
                        return _func.func(d, *new_args, **new_kwargs)

                    args, kwargs = (), {}
            else:
                func = func.func
        if not callable(func):
            raise TypeError("Function must be callable")
        return Task(func, *args, **kwargs)

    def run(self, d: tp.Any) -> tp.Any:
        """Run the pipeline on a data item."""
        raise NotImplementedError

    def __call__(self, d: tp.Any) -> tp.Any:
        return self.run(d)


class BasicAssetPipeline(AssetPipeline):
    """Class representing a basic asset pipeline.

    Builds a composite function out of all functions.

    Usage:
        ```pycon
        >>> asset_pipeline = vbt.BasicAssetPipeline()
        >>> asset_pipeline.append("flatten")
        >>> asset_pipeline.append("query", len)
        >>> asset_pipeline.append("get")

        >>> asset_pipeline(dataset[0])
        5
        ```
    """

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 0:
            tasks = []
        else:
            tasks = args[0]
            args = args[1:]
        if not isinstance(tasks, list):
            tasks = [tasks]
        self._tasks = [self.resolve_task(task, *args, **kwargs) for task in tasks]

    @property
    def tasks(self) -> tp.List[tp.Task]:
        """Tasks."""
        return self._tasks

    def append(self, func: tp.AssetFuncLike, *args, **kwargs) -> None:
        """Append a task to the pipeline."""
        self.tasks.append(self.resolve_task(func, *args, **kwargs))

    @classmethod
    def compose_tasks(cls, tasks: tp.List[tp.Task]) -> tp.Callable:
        """Compose multiple tasks into one."""

        def composed(d):
            result = d
            for func, args, kwargs in tasks:
                result = func(result, *args, **kwargs)
            return result

        return composed

    def run(self, d: tp.Any) -> tp.Any:
        return self.compose_tasks(list(self.tasks))(d)


class ComplexAssetPipeline(AssetPipeline):
    """Class representing a complex asset pipeline.

    Takes an expression string and a context. Resolves functions inside the expression.
    Expression is evaluated with `vectorbtpro.utils.eval_.evaluate`.

    Usage:
        ```pycon
        >>> asset_pipeline = vbt.ComplexAssetPipeline("query(flatten(d), len)")

        >>> asset_pipeline(dataset[0])
        5
        ```
    """

    @classmethod
    def is_expression(cls, expression: str) -> bool:
        """Determine whether the input string is an expression."""
        return not isinstance(ast.parse(expression).body, ast.Name)

    @classmethod
    def resolve_expression_and_context(
        cls,
        expression: str,
        context: tp.KwargsLike = None,
        prepare_args: bool = True,
        prepare_once: bool = True,
        **resolve_task_kwargs,
    ) -> tp.Tuple[str, tp.Kwargs]:
        """Resolve an expression and a context.

        Parses an expression string, extracts function calls with their arguments,
        removing the first positional argument from each function, and creates a new context."""
        if context is None:
            context = {}
        for k, v in package_shortcut_config.items():
            if k not in context:
                try:
                    context[k] = importlib.import_module(v)
                except ImportError:
                    pass
        tree = ast.parse(expression)
        builtin_functions = set(dir(builtins))
        imported_functions = set()
        imported_modules = set()
        defined_functions = set()
        func_context = {}

        class _FunctionAnalyzer(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name.split(".")[0]
                    imported_modules.add(name)
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imported_functions.add(name)
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                defined_functions.add(node.name)
                self.generic_visit(node)

        analyzer = _FunctionAnalyzer()
        analyzer.visit(tree)

        class _NodeMixin:

            def get_func_name(self, func):
                attrs = []
                while isinstance(func, ast.Attribute):
                    attrs.append(func.attr)
                    func = func.value
                if isinstance(func, ast.Name):
                    attrs.append(func.id)
                return ".".join(reversed(attrs)) if attrs else "<unknown>"

            def is_function_assigned(self, func):
                func_name = self.get_func_name(func)
                if "." in func_name:
                    func_name = func_name.split(".")[0]
                return (
                    func_name in context
                    or func_name in builtin_functions
                    or func_name in imported_functions
                    or func_name in imported_modules
                    or func_name in defined_functions
                )

        class _FunctionCallVisitor(ast.NodeVisitor, _NodeMixin):

            def process_argument(self, arg):
                if isinstance(arg, ast.Constant):
                    return arg.value
                elif isinstance(arg, ast.Name):
                    var_name = arg.id
                    if var_name in context:
                        return context[var_name]
                    elif var_name in builtin_functions:
                        return getattr(builtins, var_name)
                    else:
                        raise ValueError(f"Variable '{var_name}' is not defined in the context")
                elif isinstance(arg, ast.List):
                    return [self.process_argument(elem) for elem in arg.elts]
                elif isinstance(arg, ast.Tuple):
                    return tuple(self.process_argument(elem) for elem in arg.elts)
                elif isinstance(arg, ast.Dict):
                    return {self.process_argument(k): self.process_argument(v) for k, v in zip(arg.keys, arg.values)}
                elif isinstance(arg, ast.Set):
                    return {self.process_argument(elem) for elem in arg.elts}
                elif isinstance(arg, ast.Call):
                    if self.is_function_assigned(arg.func):
                        return self.get_func_name(arg.func)
                raise ValueError(f"Unsupported or dynamic argument: {ast.dump(arg)}")

            def visit_Call(self, node):
                if self.is_function_assigned(node.func):
                    return
                self.generic_visit(node)
                func_name = self.get_func_name(node.func)
                pos_args = []
                for arg in node.args[1:]:
                    arg_value = self.process_argument(arg)
                    pos_args.append(arg_value)
                kw_args = {}
                for kw in node.keywords:
                    if kw.arg is None:
                        raise ValueError(f"Dynamic keyword argument names are not allowed in '{func_name}'")
                    kw_name = kw.arg
                    kw_value = self.process_argument(kw.value)
                    kw_args[kw_name] = kw_value
                task = cls.resolve_task(
                    func_name,
                    *pos_args,
                    **kw_args,
                    prepare_args=prepare_args,
                    prepare_once=prepare_once,
                    **resolve_task_kwargs,
                )
                if prepare_args and prepare_once:

                    def func(d, _task=task):
                        return _task.func(d, *_task.args, **_task.kwargs)

                else:
                    func = task.func

                func_context[func_name] = func

        visitor = _FunctionCallVisitor()
        visitor.visit(tree)

        if prepare_args and prepare_once:

            class _ArgumentPruner(ast.NodeTransformer, _NodeMixin):

                def visit_Call(self, node: ast.Call):
                    if self.is_function_assigned(node.func):
                        return node
                    if node.args:
                        node.args = [node.args[0]]
                    else:
                        node.args = []
                    node.keywords = []
                    self.generic_visit(node)
                    return node

            pruner = _ArgumentPruner()
            modified_tree = pruner.visit(tree)
            ast.fix_missing_locations(modified_tree)
            if sys.version_info >= (3, 9):
                new_expression = ast.unparse(modified_tree)
            else:
                import astor

                new_expression = astor.to_source(modified_tree).strip()
        else:
            new_expression = expression

        new_context = merge_dicts(func_context, context)
        return new_expression, new_context

    def __init__(
        self,
        expression: str,
        context: tp.KwargsLike = None,
        prepare_once: bool = True,
        **resolve_task_kwargs,
    ) -> None:
        self._expression, self._context = self.resolve_expression_and_context(
            expression,
            context=context,
            prepare_once=prepare_once,
            **resolve_task_kwargs,
        )

    @property
    def expression(self) -> str:
        """Expression."""
        return self._expression

    @property
    def context(self) -> tp.Kwargs:
        """Context."""
        return self._context

    def run(self, d: tp.Any) -> tp.Any:
        """Run the pipeline on a data item."""
        context = merge_dicts({"d": d, "x": d}, self.context)
        return evaluate(self.expression, context=context)


class KnowledgeAsset(Configured):
    """Class for working with a knowledge asset.

    For defaults, see `vectorbtpro._settings.knowledge`."""

    _settings_path: tp.SettingsPath = "knowledge"

    _expected_keys: tp.ExpectedKeys = (Configured._expected_keys or set()) | {"data"}

    @classmethod
    def stack(
        cls: tp.Type[KnowledgeAssetT],
        *objs: tp.MaybeTuple[KnowledgeAssetT],
        **kwargs,
    ) -> KnowledgeAssetT:
        """Stack multiple `KnowledgeAsset` instances."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, KnowledgeAsset):
                raise TypeError("Each object to be stacked must be an instance of KnowledgeAsset")
        new_data = []
        for obj in objs:
            new_data.extend(obj.data)
        return cls(data=new_data, **kwargs)

    @classmethod
    def merge(
        cls: tp.Type[KnowledgeAssetT],
        *objs: tp.MaybeTuple[KnowledgeAssetT],
        flatten_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> KnowledgeAssetT:
        """Merge multiple `KnowledgeAsset` instances."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, KnowledgeAsset):
                raise TypeError("Each object to be merged must be an instance of KnowledgeAsset")

        if flatten_kwargs is None:
            flatten_kwargs = {}
        if "annotate_all" not in flatten_kwargs:
            flatten_kwargs["annotate_all"] = True
        if "excl_types" not in flatten_kwargs:
            flatten_kwargs["excl_types"] = (tuple, set, frozenset)
        max_items = 1
        for obj in objs:
            obj_data = obj.data
            if len(obj_data) > max_items:
                max_items = len(obj_data)
        flat_data = []
        for obj in objs:
            obj_data = obj.data
            if len(obj_data) == 1:
                obj_data = [obj_data] * max_items
            flat_obj_data = list(map(lambda x: search.flatten_obj(x, **flatten_kwargs), obj_data))
            flat_data.append(flat_obj_data)
        new_data = []
        for flat_dcts in zip(*flat_data):
            merged_flat_dct = flat_merge_dicts(*flat_dcts)
            new_data.append(search.unflatten_obj(merged_flat_dct))
        return cls(data=new_data, **kwargs)

    @classmethod
    def from_json_file(
        cls,
        path: tp.PathLike,
        compression: tp.Union[None, bool, str] = None,
        decompress_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> KnowledgeAssetT:
        """Build `KnowledgeAsset` from a JSON file."""
        bytes_ = load_bytes(path, compression=compression, decompress_kwargs=decompress_kwargs)
        json_str = bytes_.decode("utf-8")
        return cls(data=json.loads(json_str), **kwargs)

    @classmethod
    def from_json_bytes(
        cls,
        bytes_: bytes,
        compression: tp.Union[None, bool, str] = None,
        decompress_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> KnowledgeAssetT:
        """Build `KnowledgeAsset` from JSON bytes."""
        if decompress_kwargs is None:
            decompress_kwargs = {}
        bytes_ = decompress(bytes_, compression=compression, **decompress_kwargs)
        json_str = bytes_.decode("utf-8")
        return cls(data=json.loads(json_str), **kwargs)

    def __init__(self, data: tp.List[tp.Any], **kwargs) -> None:
        if not isinstance(data, list):
            data = [data]
        Configured.__init__(
            self,
            data=data,
            **kwargs,
        )

        self._data = data

    @property
    def data(self) -> tp.List[tp.Any]:
        """Data."""
        return self._data

    def apply(
        self,
        func: tp.MaybeList[tp.Union[tp.AssetFuncLike, AssetPipeline]],
        *args,
        wrap: tp.Optional[bool] = None,
        execute_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Apply a function to each data item.

        Function can be either a callable, a tuple of function and its arguments,
        a `vectorbtpro.utils.execution.Task` instance, a subclass of `AssetFunc` or its prefix or full name.
        Moreover, function can be a list of the above. In such a case, `BasicAssetPipeline` will be used.
        If function is a valid expression, `ComplexAssetPipeline` will be used.

        Uses `vectorbtpro.utils.execution.execute` for execution.

        If `wrap` is True, returns a new `KnowledgeAsset` instance, otherwise raw output.

        Usage:
            ```pycon
            >>> asset.apply(["flatten", ("query", len)])
            [5, 5, 5, 5, 6]

            >>> asset.apply("query(flatten(d), len)")
            [5, 5, 5, 5, 6]
            ```
        """
        execute_kwargs = self.resolve_setting(execute_kwargs, "execute_kwargs", merge=True)
        asset_func_meta = {}

        if isinstance(func, list):
            func, args, kwargs = (
                BasicAssetPipeline(
                    func,
                    *args,
                    cond_kwargs=dict(asset=self),
                    asset_func_meta=asset_func_meta,
                    **kwargs,
                ),
                (),
                {},
            )
        elif isinstance(func, str) and ComplexAssetPipeline.is_expression(func):
            if len(args) > 0:
                raise ValueError("No more positional arguments can be applied to ComplexAssetPipeline")
            func, args, kwargs = (
                ComplexAssetPipeline(
                    func,
                    context=kwargs.get("template_context", None),
                    cond_kwargs=dict(asset=self),
                    asset_func_meta=asset_func_meta,
                    **kwargs,
                ),
                (),
                {},
            )
        elif not isinstance(func, AssetPipeline):
            func, args, kwargs = AssetPipeline.resolve_task(
                func,
                *args,
                cond_kwargs=dict(asset=self),
                asset_func_meta=asset_func_meta,
                **kwargs,
            )
        else:
            if len(args) > 0:
                raise ValueError("No more positional arguments can be applied to AssetPipeline")
            if len(kwargs) > 0:
                raise ValueError("No more keyword arguments can be applied to AssetPipeline")
        execute_kwargs = merge_dicts(
            dict(
                pbar_kwargs=dict(
                    bar_id=parse_refname(func),
                ),
            ),
            execute_kwargs,
        )

        def _get_task_generator():
            for d in self.data:
                yield Task(func, d, *args, **kwargs)

        tasks = _get_task_generator()
        new_data = execute(tasks, size=len(self.data), **execute_kwargs)
        if new_data is NoResult:
            new_data = []
        if wrap is None and asset_func_meta.get("_wrap", None) is not None:
            wrap = asset_func_meta.get("_wrap", None)
        if wrap is None:
            wrap = True
        if wrap:
            return self.replace(data=new_data)
        return new_data

    def get(
        self,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        source: tp.Union[None, str, tp.Callable, tp.CustomTemplate] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        """Get data items or parts of them.

        Uses `KnowledgeAsset.apply` on `GetAssetFunc`.

        Use argument `path` to specify what part of the data item should be got. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. If `keep_path` is True, the data item will be represented
        as a nested dictionary with path as keys. If multiple paths are provided, `keep_path` automatically
        becomes True, and they will be merged into one nested dictionary. If `skip_missing` is True
        and path is missing in the data item, will skip the data item.

        Use argument `source` instead of `path` or in addition to `path` to also preprocess the source.
        It can be a string or function (will become a template), or any custom template. In this template,
        the data item is represented by "d" and the data item under the path is represented by "x" while its
        fields (if any) are represented by their names.

        Usage:
            ```pycon
            >>> asset.get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.get("d2.l[0]")
            [1, 3, 5, 7, 9]

            >>> asset.get("d2.l", source=lambda x: sum(x))
            [3, 7, 11, 15, 19]

            >>> asset.get("d2.l[0]", keep_path=True)
            [{'d2': {'l': {0: 1}}},
             {'d2': {'l': {0: 3}}},
             {'d2': {'l': {0: 5}}},
             {'d2': {'l': {0: 7}}},
             {'d2': {'l': {0: 9}}}]

            >>> asset.get(["d2.l[0]", "d2.l[1]"])
            [{'d2': {'l': {0: 1, 1: 2}}},
             {'d2': {'l': {0: 3, 1: 4}}},
             {'d2': {'l': {0: 5, 1: 6}}},
             {'d2': {'l': {0: 7, 1: 8}}},
             {'d2': {'l': {0: 9, 1: 10}}}]

            >>> asset.get("xyz", skip_missing=True)
            [123]
            ```
        """
        if path is None and source is None:
            return self.data
        return self.apply(
            GetAssetFunc,
            path=path,
            keep_path=keep_path,
            skip_missing=skip_missing,
            source=source,
            template_context=template_context,
            **kwargs,
        )

    def select(self, *args, **kwargs) -> KnowledgeAssetT:
        """Run `KnowledgeAsset.get` and return a new `KnowledgeAsset` instance."""
        return self.replace(data=self.get(*args, **kwargs))

    def set(
        self,
        value: tp.Any,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        """Set data items or parts of them.

        Uses `KnowledgeAsset.apply` on `SetAssetFunc`.

        Argument `value` can be any value, function (will become a template), or a template. In this template,
        the data item is represented by "d" and the data item under the parent path is represented by "x"
        while its fields (if any) are represented by their names.

        Use argument `path` to specify what part of the data item should be set. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the data item, will skip the data item.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only the data items that have been changed.

        Keyword arguments are passed to template substitution in `value`.

        Usage:
            ```pycon
            >>> asset.set(lambda d: sum(d["d2"]["l"])).get()
            [3, 7, 11, 15, 19]

            >>> asset.set(lambda d: sum(d["d2"]["l"]), path="d2.sum").get()
            >>> asset.set(lambda x: sum(x["l"]), path="d2.sum").get()
            >>> asset.set(lambda l: sum(l), path="d2.sum").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2], 'sum': 3}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4], 'sum': 7}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6], 'sum': 11}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8], 'sum': 15}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10], 'sum': 19}, 'xyz': 123}]

            >>> asset.set(lambda l: sum(l), path="d2.l").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': 3}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': 7}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': 11}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': 15}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': 19}, 'xyz': 123}]
            ```
        """
        return self.apply(
            SetAssetFunc,
            value=value,
            path=path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            template_context=template_context,
            **kwargs,
        )

    def remove(
        self,
        path: tp.MaybeList[tp.PathLikeKey],
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Any:
        """Set data items or parts of them.

        Uses `KnowledgeAsset.apply` on `RemoveAssetFunc`.

        Use argument `path` to specify what part of the data item should be set. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the data item, will skip the data item.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only the data items that have been changed.

        Usage:
            ```pycon
            >>> asset.remove("d2.l[0]").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [10]}, 'xyz': 123}]

            >>> asset.remove("xyz", skip_missing=True).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}}]
            ```
        """
        return self.apply(
            RemoveAssetFunc,
            path=path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def move(
        self,
        path: tp.Union[tp.PathMoveDict, tp.MaybeList[tp.PathLikeKey]],
        new_path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Any:
        """Move data items or parts of them.

        Uses `KnowledgeAsset.apply` on `MoveAssetFunc`.

        Use argument `path` to specify what part of the data item should be renamed. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the data item, will skip the data item.

        Use argument `new_path` to specify the last part of the data item (i.e., token) that should be renamed to.
        Multiple tokens can be provided. If None, `path` must be a dictionary.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only the data items that have been changed.

        Usage:
            ```pycon
            >>> asset.move("d2.l", "l").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red'}, 'l': [1, 2]},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue'}, 'l': [3, 4]},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green'}, 'l': [5, 6]},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow'}, 'l': [7, 8]},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black'}, 'xyz': 123, 'l': [9, 10]}]

            >>> asset.move({"d2.c": "c", "b": "d2.b"}).get()
            >>> asset.move(["d2.c", "b"], ["c", "d2.b"]).get()
            [{'s': 'ABC', 'd2': {'l': [1, 2], 'b': True}, 'c': 'red'},
             {'s': 'BCD', 'd2': {'l': [3, 4], 'b': True}, 'c': 'blue'},
             {'s': 'CDE', 'd2': {'l': [5, 6], 'b': False}, 'c': 'green'},
             {'s': 'DEF', 'd2': {'l': [7, 8], 'b': False}, 'c': 'yellow'},
             {'s': 'EFG', 'd2': {'l': [9, 10], 'b': False}, 'xyz': 123, 'c': 'black'}]
            ```
        """
        return self.apply(
            MoveAssetFunc,
            path=path,
            new_path=new_path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def rename(
        self,
        path: tp.Union[tp.PathRenameDict, tp.MaybeList[tp.PathLikeKey]],
        new_token: tp.Optional[tp.MaybeList[tp.PathKeyToken]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Any:
        """Rename data items or parts of them.

        Uses `KnowledgeAsset.apply` on `RenameAssetFunc`.

        Same as `KnowledgeAsset.move` but must specify new token instead of new path.

        Usage:
            ```pycon
            >>> asset.rename("d2.l", "x").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'x': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'x': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'x': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'x': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'x': [9, 10]}, 'xyz': 123}]

            >>> asset.rename("xyz", "zyx", skip_missing=True, changed_only=True).get()
            [{'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'zyx': 123}]
            ```
        """
        return self.apply(
            RenameAssetFunc,
            path=path,
            new_token=new_token,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def reorder(
        self,
        new_order: tp.Union[str, tp.PathKeyTokens],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        """Reorder data items or parts of them.

        Uses `KnowledgeAsset.apply` on `ReorderAssetFunc`.

        Can change order in dicts based on `vectorbtpro.utils.config.reorder_dict` and
        sequences based on `vectorbtpro.utils.config.reorder_list`.

        Argument `new_order` can be a sequence of tokens. To not reorder a subset of keys, they can
        be replaced by an ellipsis (`...`). For example, `["a", ..., "z"]` puts the token "a" at the start
        and the token "z" at the end while other tokens are left in the original order. If `new_order` is
        a string, it can be "asc"/"ascending" or "desc"/"descending". Other than that, it can be a string or
        function (will become a template), or any custom template. In this template, the data item is
        represented by "d" and the data item under the path is represented by "x" while its fields (if any)
        are represented by their names.

        Use argument `path` to specify what part of the data item should be set. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the data item, will skip the data item.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only the data items that have been changed.

        Keyword arguments are passed to template substitution in `new_order`.

        Usage:
            ```pycon
            >>> asset.reorder(["xyz", ...], skip_missing=True).get()
            >>> asset.reorder(lambda x: ["xyz", ...] if "xyz" in x else [...]).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'xyz': 123, 's': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}}]

            >>> asset.reorder("descending", path="d2.l").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [2, 1]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [4, 3]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [6, 5]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [8, 7]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [10, 9]}, 'xyz': 123}]
            ```
        """
        return self.apply(
            ReorderAssetFunc,
            new_order=new_order,
            path=path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            template_context=template_context,
            **kwargs,
        )

    def query(
        self,
        expression: tp.Union[str, tp.Callable, tp.CustomTemplate],
        engine: tp.Optional[str] = None,
        as_filter: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        """Query using an engine and return the queried data item(s).

        Following engines are supported:

        * "jmespath": Evaluation with `jmespath` package
        * "jsonpath", "jsonpath-ng" or "jsonpath_ng": Evaluation with `jsonpath-ng` package
        * "jsonpath.ext", "jsonpath-ng.ext" or "jsonpath_ng.ext": Evaluation with extended `jsonpath-ng` package
        * None or "template": Evaluation of each data item as a template. The data item is represented
            by "d" or "x" while its fields (if any) are represented by their names.
            Uses `KnowledgeAsset.apply` on `QueryAssetFunc`.
        * "pandas": Same as above but variables being columns

        Templates can also use the functions defined in `vectorbtpro.utils.search.search_config`.

        They work on single values and sequences alike.

        Keyword arguments are passed to the respective search/parse/evaluation function.

        Usage:
            ```pycon
            >>> asset.query("d['s'] == 'ABC'")
            >>> asset.query("x['s'] == 'ABC'")
            >>> asset.query("s == 'ABC'")
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}}]

            >>> asset.query("x['s'] == 'ABC'", as_filter=False)
            [True, False, False, False, False]

            >>> asset.query("contains(s, 'BC')")
            >>> asset.query(lambda s: "BC" in s)
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.query("[?b == `true`].s", engine="jmespath")
            ['ABC', 'BCD']

            >>> asset.query("[?contains(s, 'BC')].s", engine="jmespath")
            ['ABC', 'BCD']

            >>> asset.query("[].d2.c", engine="jmespath")
            ['red', 'blue', 'green', 'yellow', 'black']

            >>> asset.query("[?d2.c != `blue`].d2.l", engine="jmespath")
            [[1, 2], [5, 6], [7, 8], [9, 10]]

            >>> asset.query("$[?(@.b == true)].s", engine="jsonpath.ext")
            ['ABC', 'BCD']

            >>> asset.query("$[*].d2.c", engine="jsonpath.ext")
            ['red', 'blue', 'green', 'yellow', 'black']

            >>> asset.query("$[?(@.b == false)].['s', 'd2.c']", engine="jsonpath.ext")
            ['CDE', 'DEF', 'EFG']

            >>> asset.query("s[b]", engine="pandas")
            ['ABC', 'BCD']
            ```
        """
        engine = self.resolve_setting(engine, "engine")
        as_filter = self.resolve_setting(as_filter, "as_filter")
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        if engine is None or engine.lower() == "template":
            new_obj = self.apply(
                QueryAssetFunc,
                expression=expression,
                as_filter=as_filter,
                template_context=template_context,
                **kwargs,
            )
        elif engine.lower() == "jmespath":
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("jmespath")
            import jmespath

            new_obj = jmespath.search(expression, self.data, **kwargs)
        elif engine.lower() in ("jsonpath", "jsonpath-ng", "jsonpath_ng"):
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("jsonpath_ng")
            import jsonpath_ng

            jsonpath_expr = jsonpath_ng.parse(expression)
            new_obj = [match.value for match in jsonpath_expr.find(self.data, **kwargs)]
        elif engine.lower() in ("jsonpath.ext", "jsonpath-ng.ext", "jsonpath_ng.ext"):
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("jsonpath_ng")
            import jsonpath_ng.ext

            jsonpath_expr = jsonpath_ng.ext.parse(expression)
            new_obj = [match.value for match in jsonpath_expr.find(self.data, **kwargs)]
        elif engine.lower() == "pandas":
            if isinstance(expression, str):
                expression = RepEval(expression)
            elif checks.is_function(expression):
                if checks.is_builtin_func(expression):
                    expression = RepFunc(lambda _expression=expression: _expression)
                else:
                    expression = RepFunc(expression)
            elif not isinstance(expression, CustomTemplate):
                raise TypeError(f"Expression must be a template")
            df = pd.DataFrame.from_records(self.data)
            _template_context = flat_merge_dicts(
                {
                    "d": df,
                    "x": df,
                    **search.search_config,
                    **df.to_dict(orient="series"),
                },
                template_context,
            )
            result = expression.substitute(_template_context, eval_id="expression", **kwargs)
            if checks.is_function(result):
                result = result(df)
            if as_filter and isinstance(result, pd.Series) and result.dtype == "bool":
                result = df[result]
            if isinstance(result, pd.Series):
                new_obj = result.tolist()
            elif isinstance(result, pd.DataFrame):
                new_obj = result.to_dict(orient="records")
            else:
                new_obj = result
        else:
            raise ValueError(f"Invalid engine: '{engine}'")
        return new_obj

    def filter(self, *args, **kwargs) -> KnowledgeAssetT:
        """Run `KnowledgeAsset.query` and return a new `KnowledgeAsset` instance."""
        return self.replace(data=self.query(*args, **kwargs))

    def find(
        self,
        target: tp.MaybeList[tp.Any],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: tp.Optional[bool] = None,
        find_any: tp.Optional[bool] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        source: tp.Union[None, str, tp.Callable, tp.CustomTemplate] = None,
        in_json_dumps: tp.Optional[bool] = None,
        as_filter: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> KnowledgeAssetT:
        """Find occurrences and return a new `KnowledgeAsset` instance.

        Uses `KnowledgeAsset.apply` on `FindAssetFunc`.

        Uses `vectorbtpro.utils.search.contains_in_obj` (keyword arguments are passed here)
        to find any occurrences in each data item.

        Target can be one or multiple data items. If there are multiple targets and `find_any` is True,
        the match function will return True if any of the targets have been found.

        Use argument `path` to specify what part of the data item should be searched. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. If `keep_path` is True, the data item will be represented
        as a nested dictionary with path as keys. If multiple paths are provided, `keep_path` automatically
        becomes True, and they will be merged into one nested dictionary. If `skip_missing` is True
        and path is missing in the data item, will skip the data item. If `per_path` is True, will consider
        targets to be provided per path.

        Use argument `source` instead of `path` or in addition to `path` to also preprocess the source.
        It can be a string or function (will become a template), or any custom template. In this template,
        the data item is represented by "d" and the data item under the path is represented by "x" while its
        fields (if any) are represented by their names.

        Set `in_json_dumps` to True to convert the entire data item to string and search in that string.

        Usage:
            ```pycon
            >>> asset.find("BC").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.find("BC", as_filter=False).get()
            [True, True, False, False, False]

            >>> asset.find("bc", ignore_case=True).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.find("bl", path="d2.c").get()
            [{'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.find(5, path="d2.l[0]").get()
            [{'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}}]

            >>> asset.find(True, path="d2.l", source=lambda x: sum(x) >= 10).get()
            [{'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.find(["A", "B", "C"]).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}}]

            >>> asset.find(["A", "B", "C"], find_any=True).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}}]

            >>> asset.find(["A", True], ["s", "b"], per_path=True).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}}]

            >>> asset.find(r"[ABC]+", mode="regex").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}}]

            >>> asset.find("yenlow", mode="fuzzy").get()
            [{'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}}]

            >>> asset.find("xyz", in_json_dumps=True).get()
            [{'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]
            ```
        """
        return self.apply(
            FindAssetFunc,
            target=target,
            path=path,
            per_path=per_path,
            find_any=find_any,
            keep_path=keep_path,
            skip_missing=skip_missing,
            source=source,
            in_json_dumps=in_json_dumps,
            as_filter=as_filter,
            template_context=template_context,
            **kwargs,
        )

    def replace_(
        self,
        target: tp.MaybeList[tp.Any],
        replacement: tp.MaybeList[tp.Any],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: tp.Optional[bool] = None,
        replace_any: tp.Optional[bool] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> KnowledgeAssetT:
        """Find and replace occurrences and return a new `KnowledgeAsset` instance.

        Uses `KnowledgeAsset.apply` on `ReplaceAssetFunc`.

        Uses `vectorbtpro.utils.search.find_in_obj` (keyword arguments are passed here) to find
        occurrences in each data item. Then, uses `vectorbtpro.utils.search.replace_in_obj` to replace them.

        Target can be one or multiple of data items. If there are multiple targets and `replace_any` is True,
        the match function will return True if any of the targets have been found.

        Use argument `path` to specify what part of the data item should be searched. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. If `keep_path` is True, the data item will be represented
        as a nested dictionary with path as keys. If multiple paths are provided, `keep_path` automatically
        becomes True, and they will be merged into one nested dictionary. If `skip_missing` is True
        and path is missing in the data item, will skip the data item. If `per_path` is True, will consider
        targets and replacements to be provided per path.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only the data items that have been changed.

        Usage:
            ```pycon
            >>> asset.replace_("BC", "XY").get()
            [{'s': 'AXY', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'XYD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.replace_("BC", "XY", changed_only=True).get()
            [{'s': 'AXY', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'XYD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.replace_(r"(D)E(F)", r"\1X\2", mode="regex", changed_only=True).get()
            [{'s': 'DXF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}}]

            >>> asset.replace_(True, False, changed_only=True).get()
            [{'s': 'ABC', 'b': False, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': False, 'd2': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.replace_(3, 30, path="d2.l", changed_only=True).get()
            [{'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [30, 4]}}]

            >>> asset.replace_([1, 4], [10, 40], path="d2.l", changed_only=True).get()
            >>> asset.replace_([1, 4], [10, 40], path=["d2.l[0]", "d2.l[1]"], per_path=True, changed_only=True).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [10, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 40]}}]

            >>> asset.replace_([1, 4], [10, 40], path="d2.l", replace_any=False, changed_only=True).get()
            []

            >>> asset.replace_([1, 2], [10, 20], path="d2.l", replace_any=False, changed_only=True).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [10, 20]}}]

            >>> asset.replace_("a", "X", path=["s", "d2.c"], ignore_case=True, changed_only=True).get()
            [{'s': 'XBC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'blXck', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.replace_(123, 456, path="xyz", skip_missing=True, changed_only=True).get()
            [{'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 456}]
            ```
        """
        return self.apply(
            ReplaceAssetFunc,
            target=target,
            replacement=replacement,
            path=path,
            per_path=per_path,
            replace_any=replace_any,
            keep_path=keep_path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def flatten(
        self,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Any:
        """Flatten data items or parts of them.

        Uses `KnowledgeAsset.apply` on `FlattenAssetFunc`.

        Use argument `path` to specify what part of the data item should be set. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the data item, will skip the data item.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only the data items that have been changed.

        Keyword arguments are passed to `vectorbtpro.utils.search.flatten_obj`.

        Usage:
            ```pycon
            >>> asset.flatten().get()
            [{'s': 'ABC',
              'b': True,
              ('d2', 'c'): 'red',
              ('d2', 'l', 0): 1,
              ('d2', 'l', 1): 2},
              ...
             {'s': 'EFG',
              'b': False,
              ('d2', 'c'): 'black',
              ('d2', 'l', 0): 9,
              ('d2', 'l', 1): 10,
              'xyz': 123}]
            ```
        """
        return self.apply(
            FlattenAssetFunc,
            path=path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def unflatten(
        self,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Any:
        """Unflatten data items or parts of them.

        Uses `KnowledgeAsset.apply` on `UnflattenAssetFunc`.

        Use argument `path` to specify what part of the data item should be set. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the data item, will skip the data item.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only the data items that have been changed.

        Keyword arguments are passed to `vectorbtpro.utils.search.unflatten_obj`.

        Usage:
            ```pycon
            >>> asset.flatten().unflatten().get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]
            ```
        """
        return self.apply(
            UnflattenAssetFunc,
            path=path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )


ReleaseAssetT = tp.TypeVar("ReleaseAssetT", bound="ReleaseAsset")


class ReleaseAsset(KnowledgeAsset):
    """Class for working with release assets."""

    @classmethod
    def pull(
        cls,
        asset_name: tp.Optional[str] = None,
        release_name: tp.Optional[str] = None,
        repo_owner: tp.Optional[str] = None,
        repo_name: tp.Optional[str] = None,
        token: tp.Optional[str] = None,
        token_required: tp.Optional[bool] = None,
        use_pygithub: tp.Optional[bool] = None,
        chunk_size: tp.Optional[int] = None,
        cache: tp.Optional[bool] = None,
        cache_dir: tp.Optional[tp.PathLike] = None,
        cache_mkdir_kwargs: tp.KwargsLike = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> ReleaseAssetT:
        """Build `ReleaseAsset` from a JSON asset of a release."""
        from vectorbtpro._version import __version__

        asset_name = cls.resolve_setting(asset_name, "asset_name")
        release_name = cls.resolve_setting(release_name, "release_name")
        repo_owner = cls.resolve_setting(repo_owner, "repo_owner")
        repo_name = cls.resolve_setting(repo_name, "repo_name")
        token = cls.resolve_setting(token, "token")
        token_required = cls.resolve_setting(token_required, "token_required")
        use_pygithub = cls.resolve_setting(use_pygithub, "use_pygithub")
        chunk_size = cls.resolve_setting(chunk_size, "chunk_size")
        cache = cls.resolve_setting(cache, "cache")
        cache_dir = cls.resolve_setting(cache_dir, "cache_dir")
        cache_mkdir_kwargs = cls.resolve_setting(cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True)
        show_progress = cls.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = cls.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)

        current_release = "v" + __version__
        if release_name is None:
            release_name = current_release
        template_context = cls.resolve_setting(template_context, "template_context", merge=True)
        template_context = flat_merge_dicts(
            dict(
                asset_name=asset_name,
                release_name=release_name,
                repo_owner=repo_owner,
                repo_name=repo_name,
                current_release=current_release,
            ),
            template_context,
        )
        if cache:
            cache_dir = substitute_templates(cache_dir, template_context, eval_id="cache_dir")
            cache_dir = Path(cache_dir)
            if cache_dir.exists():
                cache_file = None
                for file in cache_dir.iterdir():
                    if file.is_file() and file.name == asset_name:
                        cache_file = file
                        break
                if cache_file is not None:
                    return cls.from_json_file(cache_file, **kwargs)

        if token is None:
            token = os.environ.get("GITHUB_TOKEN", None)
        if token is None and token_required:
            raise ValueError("GitHub token is required")
        if use_pygithub is None:
            from vectorbtpro.utils.module_ import check_installed

            use_pygithub = check_installed("github")
        if use_pygithub:
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("github")
            from github import Github, Auth
            from github.GithubException import UnknownObjectException

            if token is not None:
                g = Github(auth=Auth.Token(token))
            else:
                g = Github()
            try:
                repo = g.get_repo(f"{repo_owner}/{repo_name}")
            except UnknownObjectException:
                raise Exception(f"Repository '{repo_owner}/{repo_name}' not found or access denied")
            if release_name == "latest":
                try:
                    release = repo.get_latest_release()
                except UnknownObjectException:
                    raise Exception("Latest release not found")
            else:
                releases = repo.get_releases()
                found_release = None
                for release in releases:
                    if release.title == release_name:
                        found_release = release
                if found_release is None:
                    raise Exception(f"Release '{release_name}' not found")
                release = found_release
            assets = release.get_assets()
            if asset_name is not None:
                asset = next((a for a in assets if a.name == asset_name), None)
                if asset is None:
                    raise Exception(f"Asset '{asset_name}' not found in release {release}")
            else:
                assets_list = list(assets)
                if len(assets_list) == 1:
                    asset = assets_list[0]
                else:
                    raise Exception("Please specify asset_name")
            asset_url = asset.url
        else:
            headers = {"Accept": "application/vnd.github+json"}
            if token is not None:
                headers["Authorization"] = f"token {token}"
            if release_name == "latest":
                release_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"
                response = requests.get(release_url, headers=headers)
                response.raise_for_status()
                release_info = response.json()
            else:
                releases_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases"
                response = requests.get(releases_url, headers=headers)
                response.raise_for_status()
                releases = response.json()
                release_info = None
                for release in releases:
                    if release.get("name") == release_name:
                        release_info = release
                if release_info is None:
                    raise ValueError(f"Release '{release_name}' not found")
            assets = release_info.get("assets", [])
            if asset_name is not None:
                asset = next((a for a in assets if a["name"] == asset_name), None)
                if asset is None:
                    raise Exception(f"Asset '{asset_name}' not found in release {release}")
            else:
                if len(assets) == 1:
                    asset = assets[0]
                else:
                    raise Exception("Please specify asset_name")
            asset_url = asset["url"]

        asset_headers = {"Accept": "application/octet-stream"}
        if token is not None:
            asset_headers["Authorization"] = f"token {token}"
        asset_response = requests.get(asset_url, headers=asset_headers, stream=True)
        asset_response.raise_for_status()
        file_size = int(asset_response.headers.get("Content-Length", 0))
        if file_size == 0:
            file_size = asset.get("size", 0)
        pbar_kwargs = flat_merge_dicts(
            dict(bar_id=get_caller_qualname()),
            pbar_kwargs,
        )
        pbar_kwargs = substitute_templates(pbar_kwargs, template_context, eval_id="pbar_kwargs")

        if cache:
            check_mkdir(cache_dir, **cache_mkdir_kwargs)
            cache_file = cache_dir / asset_name
            with open(cache_file, "wb") as f:
                with ProgressBar(total=file_size, show_progress=show_progress, **pbar_kwargs) as pbar:
                    for chunk in asset_response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return cls.from_json_file(cache_file, **kwargs)
        else:
            with io.BytesIO() as bytes_io:
                with ProgressBar(total=file_size, show_progress=show_progress, **pbar_kwargs) as pbar:
                    for chunk in asset_response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            bytes_io.write(chunk)
                            pbar.update(len(chunk))
                bytes_ = bytes_io.getvalue()
            compression = suggest_compression(asset_name)
            if compression is not None and "compression" not in kwargs:
                kwargs["compression"] = compression
            return cls.from_json_bytes(bytes_, **kwargs)


class MessagesAsset(ReleaseAsset):
    """Class for working with Discord messages."""

    _settings_path: tp.SettingsPath = "knowledge.assets.messages"


class PagesAsset(ReleaseAsset):
    """Class for working with website pages."""

    _settings_path: tp.SettingsPath = "knowledge.assets.pages"
