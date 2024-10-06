# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for working with knowledge.

Run for the examples:

```pycon
>>> dataset = [
...     {"s": "ABC", "b": True, "d": {"c": "red", "l": [1, 2]}},
...     {"s": "BCD", "b": True, "d": {"c": "blue", "l": [3, 4]}},
...     {"s": "CDE", "b": False, "d": {"c": "green", "l": [5, 6]}},
...     {"s": "DEF", "b": False, "d": {"c": "yellow", "l": [7, 8]}},
...     {"s": "EFG", "b": False, "d": {"c": "black", "l": [9, 10]}, "xyz": 123}
... ]
>>> asset = vbt.KnowledgeAsset(dataset)
```
"""

from abc import ABC, abstractmethod
import os
import io
import json
from pathlib import Path
import requests

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
from vectorbtpro.utils.module_ import parse_refname, get_caller_qualname

__all__ = [
    "AssetFunc",
    "AssetPipeline",
    "KnowledgeAsset",
    "ReleaseAsset",
    "MessagesAsset",
    "PagesAsset",
]


KnowledgeAssetT = tp.TypeVar("KnowledgeAssetT", bound="KnowledgeAsset")


class AssetFunc(ABC):
    """Class representing a function to be applied to a knowledge asset."""

    _wrap = None

    @classmethod
    @abstractmethod
    def prepare_args(cls, *args, **kwargs) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def func(cls, o: tp.Any, *args, **kwargs) -> tp.Any:
        """Function to be applied."""
        raise NotImplementedError


class GetAssetFunc(AssetFunc):
    """Class for `KnowledgeAsset.get`."""

    _wrap = False
    """Whether the results are meant to be wrapped with `KnowledgeAsset`."""

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
        o: tp.Any,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        keep_path: bool = False,
        skip_missing: bool = False,
        source: tp.Optional[tp.CustomTemplate] = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.Any:
        x = o
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
                    "o": o,
                    "x": x,
                    **(x if isinstance(x, dict) else {}),
                },
                template_context,
            )
            x = source.substitute(_template_context, eval_id="source")
        return x


class SetAssetFunc(AssetFunc):
    """Class for `KnowledgeAsset.set`."""

    _wrap = True

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
        o: tp.Any,
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
            x = o
            if p is not None:
                try:
                    x = search.get_pathlike_key(x, p[:-1])
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            _template_context = flat_merge_dicts(
                {
                    "o": o,
                    "x": x,
                    **(x if isinstance(x, dict) else {}),
                },
                template_context,
            )
            v = value.substitute(_template_context, eval_id="value", **kwargs)
            o = search.set_pathlike_key(o, p, v, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return o
        return NoResult


class RemoveAssetFunc(AssetFunc):
    """Class for `KnowledgeAsset.remove`."""

    _wrap = True

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
        o: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            try:
                o = search.remove_pathlike_key(o, p, make_copy=make_copy, prev_keys=prev_keys)
            except (KeyError, IndexError, AttributeError) as e:
                if not skip_missing:
                    raise e
                continue
        if not changed_only or len(prev_keys) > 0:
            return o
        return NoResult


class MoveAssetFunc(AssetFunc):
    """Class for `KnowledgeAsset.move`."""

    _wrap = True

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
        o: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        new_paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
    ) -> tp.Any:
        prev_keys = []
        for i, p in enumerate(paths):
            try:
                x = search.get_pathlike_key(o, p)
                o = search.remove_pathlike_key(o, p, make_copy=make_copy, prev_keys=prev_keys)
                o = search.set_pathlike_key(o, new_paths[i], x, make_copy=make_copy, prev_keys=prev_keys)
            except (KeyError, IndexError, AttributeError) as e:
                if not skip_missing:
                    raise e
                continue
        if not changed_only or len(prev_keys) > 0:
            return o
        return NoResult


class RenameAssetFunc(MoveAssetFunc):
    """Class for `KnowledgeAsset.rename`."""

    _wrap = True

    @classmethod
    def prepare_args(
        cls,
        path: tp.Union[tp.PathRenameDict, tp.MaybeList[tp.PathLikeKey]],
        new_token: tp.Optional[tp.MaybeList[tp.PathKeyToken]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
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
    """Class for `KnowledgeAsset.reorder`."""

    _wrap = True

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
        o: tp.Any,
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
            x = o
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
                        "o": o,
                        "x": x,
                        **(x if isinstance(x, dict) else {}),
                    },
                    template_context,
                )
                _new_order = new_order.substitute(_template_context, eval_id="new_order", **kwargs)
            else:
                _new_order = new_order
            if isinstance(x, dict):
                x = reorder_dict(x, _new_order, skip_missing=skip_missing)
            else:
                if checks.is_namedtuple(x):
                    x = type(x)(*reorder_list(x, _new_order, skip_missing=skip_missing))
                else:
                    x = type(x)(reorder_list(x, _new_order, skip_missing=skip_missing))
            o = search.set_pathlike_key(o, p, x, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return o
        return NoResult


class QueryAssetFunc(AssetFunc):
    """Class for `KnowledgeAsset.query`."""

    _wrap = False

    @classmethod
    def prepare_args(
        cls,
        expression: tp.Union[str, tp.CustomTemplate],
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
        o: tp.Any,
        expression: tp.CustomTemplate,
        as_filter: bool = True,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        _template_context = flat_merge_dicts(
            {
                "o": o,
                "x": o,
                **search.search_config,
                **(o if isinstance(o, dict) else {}),
            },
            template_context,
        )
        new_o = expression.substitute(_template_context, eval_id="expression", **kwargs)
        if as_filter and isinstance(new_o, bool):
            if new_o:
                return o
            return NoResult
        return new_o


class FindAssetFunc(AssetFunc):
    """Class for `KnowledgeAsset.find`."""

    _wrap = True

    @classmethod
    def prepare_args(
        cls,
        target: tp.MaybeList[tp.Any],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        source: tp.Union[None, str, tp.Callable, tp.CustomTemplate] = None,
        in_dumps: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        asset: tp.Optional[tp.MaybeType["KnowledgeAsset"]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            asset = KnowledgeAsset
        keep_path = asset.resolve_setting(keep_path, "keep_path")
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        in_dumps = asset.resolve_setting(in_dumps, "in_dumps")
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
                source = RepFunc(source)
            elif not isinstance(source, CustomTemplate):
                raise TypeError(f"Source must be a template")
        if "excl_types" not in kwargs:
            kwargs["excl_types"] = (tuple, set, frozenset)
        return (), {
            **dict(
                target=target,
                path=path,
                keep_path=keep_path,
                skip_missing=skip_missing,
                source=source,
                in_dumps=in_dumps,
                template_context=template_context,
            ),
            **kwargs,
        }

    @classmethod
    def match_func(
        cls,
        k: tp.Optional[tp.Hashable],
        o: tp.Any,
        target: tp.MaybeList[tp.Any],
        **kwargs,
    ) -> bool:
        """Match function for `FindAssetFunc.func`.

        Uses `vectorbtpro.utils.search.contains` for text and equality checks for other types."""
        if not isinstance(target, list):
            targets = [target]
        else:
            targets = target
        for target in targets:
            if o is target:
                return True
            if o is None and target is None:
                return True
            elif checks.is_bool(o) and checks.is_bool(target):
                if o == target:
                    return True
            elif checks.is_number(o) and checks.is_number(target):
                if o == target:
                    return True
            elif isinstance(o, str) and isinstance(target, str):
                if search.contains(o, target, **kwargs):
                    return True
            elif type(o) is type(target):
                if o == target:
                    return True
        return False

    @classmethod
    def func(
        cls,
        o: tp.Any,
        target: tp.MaybeList[tp.Any],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        keep_path: bool = False,
        skip_missing: bool = False,
        source: tp.Union[None, str, tp.Callable, tp.CustomTemplate] = None,
        in_dumps: bool = False,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        x = o
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
                    "o": o,
                    "x": x,
                    **(x if isinstance(x, dict) else {}),
                },
                template_context,
            )
            x = source.substitute(_template_context, eval_id="source")
        if not isinstance(x, str) and in_dumps:
            x = json.dumps(x, ensure_ascii=False)
        if search.contains_in_obj(
            x,
            cls.match_func,
            target=target,
            **kwargs,
        ):
            return o
        return NoResult


class ReplaceAssetFunc(FindAssetFunc):
    """Class for `KnowledgeAsset.replace_`."""

    _wrap = True

    @classmethod
    def prepare_args(
        cls,
        target: tp.MaybeList[tp.Any],
        replacement: tp.MaybeList[tp.Any],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset: tp.Optional[tp.MaybeType["KnowledgeAsset"]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            asset = KnowledgeAsset
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
        find_arg_names = set(get_func_arg_names(search.find_in_obj))
        find_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in find_arg_names}
        if "excl_types" not in find_kwargs:
            find_kwargs["excl_types"] = (tuple, set, frozenset)
        return (), {
            **dict(
                target=target,
                replacement=replacement,
                paths=paths,
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
        o: tp.Any,
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
        new_o = o
        for i, target in enumerate(targets):
            if o is target:
                return replacements[i]
            if o is None and target is None:
                return replacements[i]
            elif checks.is_bool(o) and checks.is_bool(target):
                if o == target:
                    return replacements[i]
            elif checks.is_number(o) and checks.is_number(target):
                if o == target:
                    return replacements[i]
            elif isinstance(o, str) and isinstance(target, str):
                new_o = search.replace(new_o, target, replacements[i], **kwargs)
            elif type(o) is type(target):
                if o == target:
                    return replacements[i]
        return new_o

    @classmethod
    def func(
        cls,
        o: tp.Any,
        target: tp.MaybeList[tp.Any],
        replacement: tp.MaybeList[tp.Any],
        paths: tp.List[tp.PathLikeKey],
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
        for p in paths:
            x = o
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
                target=target,
                **find_kwargs,
                **kwargs,
            )
            for k, v in path_dct.items():
                if p is not None and not keep_path:
                    new_p = search.combine_pathlike_keys(p, k, minimize=True)
                else:
                    new_p = k
                v = cls.replace_func(
                    k,
                    v,
                    target,
                    replacement,
                    **kwargs,
                )
                o = search.set_pathlike_key(o, new_p, v, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return o
        return NoResult


class FlattenAssetFunc(AssetFunc):
    """Class for `KnowledgeAsset.flatten`."""

    _wrap = True

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
        o: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        **kwargs,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            x = o
            if p is not None:
                try:
                    x = search.get_pathlike_key(x, p)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            x = search.flatten_obj(x, **kwargs)
            o = search.set_pathlike_key(o, p, x, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return o
        return NoResult


class UnflattenAssetFunc(AssetFunc):
    """Class for `KnowledgeAsset.unflatten`."""

    _wrap = True

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
        o: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        **kwargs,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            x = o
            if p is not None:
                try:
                    x = search.get_pathlike_key(x, p)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            x = search.unflatten_obj(x, **kwargs)
            o = search.set_pathlike_key(o, p, x, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return o
        return NoResult


class AssetPipeline:
    """Asset pipeline.

    Usage:
        ```pycon
        >>> asset_pipeline = vbt.AssetPipeline()
        >>> asset_pipeline.append("flatten")
        >>> asset_pipeline.append("query", "len(x)")
        >>> asset_pipeline.append("get")

        >>> asset_pipeline(dataset[0])
        5
        ```
    """

    @classmethod
    def resolve_task(
        cls,
        func: tp.AssetFuncLike,
        *args,
        prepare_args: bool = True,
        cond_kwargs: tp.KwargsLike = None,
        asset_func_meta: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Task:
        """Resolve a task."""
        func_wrap = None
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
                raise ValueError(f"Function '{func}' not found")
        if isinstance(func, AssetFunc):
            raise TypeError("Function must be a subclass of AssetFunc, not an instance")
        if isinstance(func, type) and issubclass(func, AssetFunc):
            if asset_func_meta is not None:
                if hasattr(func, "_wrap"):
                    asset_func_meta["_wrap"] = getattr(func, "_wrap")
            if prepare_args:
                if cond_kwargs is None:
                    cond_kwargs = {}
                if len(cond_kwargs) > 0:
                    prepare_args_arg_names = get_func_arg_names(func.prepare_args)
                    for k, v in cond_kwargs.items():
                        if k in prepare_args_arg_names:
                            kwargs[k] = v
                args, kwargs = func.prepare_args(*args, **kwargs)
            func = func.func
        if not callable(func):
            raise TypeError("Function must be callable")
        task = Task(func, *args, **kwargs)
        return task

    @classmethod
    def compose_tasks(cls, tasks: tp.List[tp.Task]) -> tp.Callable:
        """Compose multiple tasks into one."""

        def composed(o):
            result = o
            for func, args, kwargs in tasks:
                result = func(result, *args, **kwargs)
            return result

        return composed

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 0:
            tasks = []
        else:
            tasks = args[0]
            args = args[1:]
        if not isinstance(tasks, list):
            tasks = [tasks]
        tasks = list(map(lambda task: self.resolve_task(task, *args, **kwargs), tasks))
        self._tasks = tasks

    @property
    def tasks(self) -> tp.List[tp.Task]:
        """Tasks."""
        return self._tasks

    def append(self, func: tp.AssetFuncLike, *args, **kwargs):
        """Append a task to the pipeline."""
        self.tasks.append(self.resolve_task(func, *args, **kwargs))

    def run(self, o: tp.Any) -> tp.Any:
        """Run the pipeline on an object."""
        return self.compose_tasks(self.tasks)(o)

    def __call__(self, o: tp.Any) -> tp.Any:
        return self.run(o)


class KnowledgeAsset(Configured):
    """Class for working with a knowledge asset.

    For defaults, see `vectorbtpro._settings.knowledge`."""

    _settings_path: tp.SettingsPath = "knowledge"

    _expected_keys: tp.ExpectedKeys = (Configured._expected_keys or set()) | {"obj"}

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
        new_obj = []
        for obj in objs:
            obj = obj.obj
            if isinstance(obj, list):
                new_obj.extend(obj)
            else:
                new_obj.append(obj)
        return cls(obj=new_obj, **kwargs)

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
        must_become_list = True
        max_items = 1
        for obj in objs:
            if isinstance(obj.obj, list):
                if len(obj.obj) > max_items:
                    max_items = len(obj.obj)
        flat_objs = []
        for obj in objs:
            obj = obj.obj
            if not isinstance(obj, list) and must_become_list:
                obj = [obj] * max_items
            flat_obj = list(map(lambda x: search.flatten_obj(x, **flatten_kwargs), obj))
            flat_objs.append(flat_obj)
        if must_become_list:
            new_obj = []
            for flat_dcts in zip(*flat_objs):
                merged_flat_dct = flat_merge_dicts(*flat_dcts)
                new_obj.append(search.unflatten_obj(merged_flat_dct))
        else:
            merged_flat_dct = flat_merge_dicts(*flat_objs)
            new_obj = search.unflatten_obj(merged_flat_dct)
        return cls(obj=new_obj, **kwargs)

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
        return cls(json.loads(json_str), **kwargs)

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
        return cls(json.loads(json_str), **kwargs)

    def __init__(self, obj: tp.Any, **kwargs) -> None:
        Configured.__init__(
            self,
            obj=obj,
            **kwargs,
        )

        self._obj = obj

    @property
    def obj(self) -> tp.Any:
        """Object."""
        return self._obj

    def apply(
        self,
        func: tp.MaybeList[tp.Union[tp.AssetFuncLike, AssetPipeline]],
        *args,
        wrap: tp.Optional[bool] = None,
        execute_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Apply a function to each object.

        Function can be either a callable, a tuple of function and its arguments,
        a `vectorbtpro.utils.execution.Task` instance, a subclass of `AssetFunc` or its prefix or full name.
        Moreover, function can be a list of the above. In such a case, a composite function will be
        built and applied to each element of the asset such that the asset is processed in a single pass.

        Uses `vectorbtpro.utils.execution.execute` for execution.

        If `wrap` is True, returns a new `KnowledgeAsset` instance, otherwise raw output.

        Usage:
            ```pycon
            >>> asset.apply(["flatten", ("query", "len(x)"), "get"])
            [5, 5, 5, 5, 6]
            ```
        """
        execute_kwargs = self.resolve_setting(execute_kwargs, "execute_kwargs", merge=True)
        asset_func_meta = {}

        if isinstance(func, list):
            func, args, kwargs = (
                AssetPipeline(
                    func,
                    *args,
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
        obj = self.obj
        if not isinstance(obj, list):
            objs = [obj]
            single_obj = True
        else:
            objs = obj
            single_obj = False

        def _get_task_generator():
            for obj in objs:
                yield Task(func, obj, *args, **kwargs)

        tasks = _get_task_generator()
        execute_kwargs = flat_merge_dicts(dict(show_progress=False if single_obj else None), execute_kwargs)
        new_obj = execute(tasks, size=len(objs), **execute_kwargs)
        if new_obj is NoResult:
            if single_obj:
                new_obj = None
            else:
                new_obj = []
        elif single_obj:
            if len(new_obj) > 0:
                new_obj = new_obj[0]
            else:
                new_obj = None
        if wrap is None and asset_func_meta.get("_wrap", None) is not None:
            wrap = asset_func_meta.get("_wrap", None)
        if wrap is None:
            wrap = True
        if wrap:
            return self.replace(obj=new_obj)
        return new_obj

    def get(
        self,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        source: tp.Union[None, str, tp.Callable, tp.CustomTemplate] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        """Get objects or parts of them.

        Uses `KnowledgeAsset.apply` on `GetAssetFunc`.

        Use argument `path` to specify what part of the object should be got. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. If `keep_path` is True, object will be represented
        as a nested dictionary with path as keys. If multiple paths are provided, `keep_path` automatically
        becomes True, and they will be merged into one nested dictionary. If `skip_missing` is True
        and path is missing in the object, will skip the object.

        Use argument `source` instead of `path` or in addition to `path` to also preprocess the source.
        It can be a string or function (will become a template), or any custom template. In this template,
        the object is represented by "o" and the object under the path is represented by "x" while its
        fields (if any) are represented by their names.

        Usage:
            ```pycon
            >>> asset.get()
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.get("d.l[0]")
            [1, 3, 5, 7, 9]

            >>> asset.get("d.l", source=lambda x: sum(x))
            [3, 7, 11, 15, 19]

            >>> asset.get("d.l[0]", keep_path=True)
            [{'d': {'l': {0: 1}}},
             {'d': {'l': {0: 3}}},
             {'d': {'l': {0: 5}}},
             {'d': {'l': {0: 7}}},
             {'d': {'l': {0: 9}}}]

            >>> asset.get(["d.l[0]", "d.l[1]"])
            [{'d': {'l': {0: 1, 1: 2}}},
             {'d': {'l': {0: 3, 1: 4}}},
             {'d': {'l': {0: 5, 1: 6}}},
             {'d': {'l': {0: 7, 1: 8}}},
             {'d': {'l': {0: 9, 1: 10}}}]

            >>> asset.get("xyz", skip_missing=True)
            [123]
            ```
        """
        if path is None and source is None:
            return self.obj
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
        return self.replace(obj=self.get(*args, **kwargs))

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
        """Set objects or parts of them.

        Uses `KnowledgeAsset.apply` on `SetAssetFunc`.

        Argument `value` can be any value, function (will become a template), or a template. In this template,
        the object is represented by "o" and the object under the parent path is represented by "x" while its
        fields (if any) are represented by their names.

        Use argument `path` to specify what part of the object should be set. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the object, will skip the object.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only objects that have been changed.

        Keyword arguments are passed to template substitution in `value`.

        Usage:
            ```pycon
            >>> asset.set(lambda o: sum(o["d"]["l"])).get()
            [3, 7, 11, 15, 19]

            >>> asset.set(lambda o: sum(o["d"]["l"]), path="d.sum").get()
            >>> asset.set(lambda x: sum(x["l"]), path="d.sum").get()
            >>> asset.set(lambda l: sum(l), path="d.sum").get()
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red', 'l': [1, 2], 'sum': 3}},
             {'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'l': [3, 4], 'sum': 7}},
             {'s': 'CDE', 'b': False, 'd': {'c': 'green', 'l': [5, 6], 'sum': 11}},
             {'s': 'DEF', 'b': False, 'd': {'c': 'yellow', 'l': [7, 8], 'sum': 15}},
             {'s': 'EFG', 'b': False, 'd': {'c': 'black', 'l': [9, 10], 'sum': 19}, 'xyz': 123}]

            >>> asset.set(lambda l: sum(l), path="d.l").get()
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red', 'l': 3}},
             {'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'l': 7}},
             {'s': 'CDE', 'b': False, 'd': {'c': 'green', 'l': 11}},
             {'s': 'DEF', 'b': False, 'd': {'c': 'yellow', 'l': 15}},
             {'s': 'EFG', 'b': False, 'd': {'c': 'black', 'l': 19}, 'xyz': 123}]
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
        """Set objects or parts of them.

        Uses `KnowledgeAsset.apply` on `RemoveAssetFunc`.

        Use argument `path` to specify what part of the object should be set. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the object, will skip the object.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only objects that have been changed.

        Usage:
            ```pycon
            >>> asset.remove("d.l[0]").get()
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red', 'l': [2]}},
             {'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'l': [4]}},
             {'s': 'CDE', 'b': False, 'd': {'c': 'green', 'l': [6]}},
             {'s': 'DEF', 'b': False, 'd': {'c': 'yellow', 'l': [8]}},
             {'s': 'EFG', 'b': False, 'd': {'c': 'black', 'l': [10]}, 'xyz': 123}]

            >>> asset.remove("xyz", skip_missing=True).get()
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd': {'c': 'black', 'l': [9, 10]}}]
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
        """Move objects or parts of them.

        Uses `KnowledgeAsset.apply` on `MoveAssetFunc`.

        Use argument `path` to specify what part of the object should be renamed. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the object, will skip the object.

        Use argument `new_path` to specify the last part of the object (i.e., token) that should be renamed to.
        Multiple tokens can be provided. If None, `path` must be a dictionary.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only objects that have been changed.

        Usage:
            ```pycon
            >>> asset.move("d.l", "l").get()
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red'}, 'l': [1, 2]},
             {'s': 'BCD', 'b': True, 'd': {'c': 'blue'}, 'l': [3, 4]},
             {'s': 'CDE', 'b': False, 'd': {'c': 'green'}, 'l': [5, 6]},
             {'s': 'DEF', 'b': False, 'd': {'c': 'yellow'}, 'l': [7, 8]},
             {'s': 'EFG', 'b': False, 'd': {'c': 'black'}, 'xyz': 123, 'l': [9, 10]}]

            >>> asset.move({"d.c": "c", "b": "d.b"}).get()
            >>> asset.move(["d.c", "b"], ["c", "d.b"]).get()
            [{'s': 'ABC', 'd': {'l': [1, 2], 'b': True}, 'c': 'red'},
             {'s': 'BCD', 'd': {'l': [3, 4], 'b': True}, 'c': 'blue'},
             {'s': 'CDE', 'd': {'l': [5, 6], 'b': False}, 'c': 'green'},
             {'s': 'DEF', 'd': {'l': [7, 8], 'b': False}, 'c': 'yellow'},
             {'s': 'EFG', 'd': {'l': [9, 10], 'b': False}, 'xyz': 123, 'c': 'black'}]
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
        """Rename objects or parts of them.

        Uses `KnowledgeAsset.apply` on `RenameAssetFunc`.

        Same as `KnowledgeAsset.move` but must specify new token instead of new path.

        Usage:
            ```pycon
            >>> asset.rename("d.l", "x").get()
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red', 'x': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'x': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd': {'c': 'green', 'x': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd': {'c': 'yellow', 'x': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd': {'c': 'black', 'x': [9, 10]}, 'xyz': 123}]

            >>> asset.rename("xyz", "zyx", skip_missing=True, changed_only=True).get()
            [{'s': 'EFG', 'b': False, 'd': {'c': 'black', 'l': [9, 10]}, 'zyx': 123}]
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
        """Reorder objects or parts of them.

        Uses `KnowledgeAsset.apply` on `ReorderAssetFunc`.

        Can change order in dicts based on `vectorbtpro.utils.config.reorder_dict` and
        sequences based on `vectorbtpro.utils.config.reorder_list`.

        Argument `new_order` can be a sequence of tokens. To not reorder a subset of keys, they can
        be replaced by an ellipsis (`...`). For example, `["a", ..., "z"]` puts the token "a" at the start
        and the token "z" at the end while other tokens are left in the original order. If `new_order` is
        a string, it can be "asc"/"ascending" or "desc"/"descending". Other than that, it can be a string or
        function (will become a template), or any custom template. In this template, the object is
        represented by "o" and the object under the path is represented by "x" while its fields (if any)
        are represented by their names.

        Use argument `path` to specify what part of the object should be set. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the object, will skip the object.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only objects that have been changed.

        Keyword arguments are passed to template substitution in `new_order`.

        Usage:
            ```pycon
            >>> asset.reorder(["xyz", ...], skip_missing=True).get()
            >>> asset.reorder(lambda x: ["xyz", ...] if "xyz" in x else [...]).get()
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd': {'c': 'yellow', 'l': [7, 8]}},
             {'xyz': 123, 's': 'EFG', 'b': False, 'd': {'c': 'black', 'l': [9, 10]}}]

            >>> asset.reorder("descending", path="d.l").get()
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red', 'l': [2, 1]}},
             {'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'l': [4, 3]}},
             {'s': 'CDE', 'b': False, 'd': {'c': 'green', 'l': [6, 5]}},
             {'s': 'DEF', 'b': False, 'd': {'c': 'yellow', 'l': [8, 7]}},
             {'s': 'EFG', 'b': False, 'd': {'c': 'black', 'l': [10, 9]}, 'xyz': 123}]
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
        expression: tp.Union[str, tp.CustomTemplate],
        engine: tp.Optional[str] = None,
        as_filter: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        """Query using an engine and return the queried object(s).

        Following engines are supported:

        * "jmespath": Evaluation with `jmespath` package
        * "jsonpath", "jsonpath-ng" or "jsonpath_ng": Evaluation with `jsonpath-ng` package
        * "jsonpath.ext", "jsonpath-ng.ext" or "jsonpath_ng.ext": Evaluation with extended `jsonpath-ng` package
        * None or "template": Evaluation of each object as a template. The object is represented
            by "o" or "x" while its fields (if any) are represented by their names.
            Uses `KnowledgeAsset.apply` on `QueryAssetFunc`.
        * "pandas": Same as above but variables being columns

        Templates can also use the functions defined in `vectorbtpro.utils.search.search_config`.

        They work on single values and sequences alike.

        Keyword arguments are passed to the respective search/parse/evaluation function.

        Usage:
            ```pycon
            >>> asset.query("o['s'] == 'ABC'")
            >>> asset.query("x['s'] == 'ABC'")
            >>> asset.query("s == 'ABC'")
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red', 'l': [1, 2]}}]

            >>> asset.query("x['s'] == 'ABC'", as_filter=False)
            [True, False, False, False, False]

            >>> asset.query("contains(s, 'BC')")
            >>> asset.query(lambda s: "BC" in s)
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.query("[?b == `true`].s", engine="jmespath")
            ['ABC', 'BCD']

            >>> asset.query("[?contains(s, 'BC')].s", engine="jmespath")
            ['ABC', 'BCD']

            >>> asset.query("[].d.c", engine="jmespath")
            ['red', 'blue', 'green', 'yellow', 'black']

            >>> asset.query("[?d.c != `blue`].d.l", engine="jmespath")
            [[1, 2], [5, 6], [7, 8], [9, 10]]

            >>> asset.query("$[?(@.b == true)].s", engine="jsonpath.ext")
            ['ABC', 'BCD']

            >>> asset.query("$[*].d.c", engine="jsonpath.ext")
            ['red', 'blue', 'green', 'yellow', 'black']

            >>> asset.query("$[?(@.b == false)].['s', 'd.c']", engine="jsonpath.ext")
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

            new_obj = jmespath.search(expression, self.obj, **kwargs)
        elif engine.lower() in ("jsonpath", "jsonpath-ng", "jsonpath_ng"):
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("jsonpath_ng")
            import jsonpath_ng

            jsonpath_expr = jsonpath_ng.parse(expression)
            new_obj = [match.value for match in jsonpath_expr.find(self.obj, **kwargs)]
        elif engine.lower() in ("jsonpath.ext", "jsonpath-ng.ext", "jsonpath_ng.ext"):
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("jsonpath_ng")
            import jsonpath_ng.ext

            jsonpath_expr = jsonpath_ng.ext.parse(expression)
            new_obj = [match.value for match in jsonpath_expr.find(self.obj, **kwargs)]
        elif engine.lower() == "pandas":
            if isinstance(expression, str):
                expression = RepEval(expression)
            elif checks.is_function(expression):
                expression = RepFunc(expression)
            elif not isinstance(expression, CustomTemplate):
                raise TypeError(f"Expression must be a template")
            df = pd.DataFrame.from_records(self.obj)
            _template_context = flat_merge_dicts(
                {
                    "o": df,
                    "x": df,
                    **search.search_config,
                    **df.to_dict(orient="series"),
                },
                template_context,
            )
            result = expression.substitute(_template_context, eval_id="expression", **kwargs)
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
        return self.replace(obj=self.query(*args, **kwargs))

    def find(
        self,
        target: tp.MaybeList[tp.Any],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        source: tp.Union[None, str, tp.Callable, tp.CustomTemplate] = None,
        in_dumps: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> KnowledgeAssetT:
        """Find occurrences and return a new `KnowledgeAsset` instance.

        Uses `KnowledgeAsset.apply` on `FindAssetFunc`.

        Uses `vectorbtpro.utils.search.contains_in_obj` (keyword arguments are passed here)
        to find any occurrences in each object.

        Target can be one or multiple objects.

        Use argument `path` to specify what part of the object should be searched. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. If `keep_path` is True, object will be represented
        as a nested dictionary with path as keys. If multiple paths are provided, `keep_path` automatically
        becomes True, and they will be merged into one nested dictionary. If `skip_missing` is True
        and path is missing in the object, will skip the object.

        Use argument `source` instead of `path` or in addition to `path` to also preprocess the source.
        It can be a string or function (will become a template), or any custom template. In this template,
        the object is represented by "o" and the object under the path is represented by "x" while its
        fields (if any) are represented by their names.

        Set `in_dumps` to True to convert the entire object to string and search in that string.

        Usage:
            ```pycon
            >>> asset.find("BC").get()
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.find("bc", ignore_case=True).get()
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.find("bl", path="d.c").get()
            [{'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'EFG', 'b': False, 'd': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.find(5, path="d.l[0]").get()
            [{'s': 'CDE', 'b': False, 'd': {'c': 'green', 'l': [5, 6]}}]

            >>> asset.find(True, path="d.l", source=lambda x: sum(x) >= 10).get()
            [{'s': 'CDE', 'b': False, 'd': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.find(["A", "B", "C"]).get()
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd': {'c': 'green', 'l': [5, 6]}}]

            >>> asset.find(r"[ABC]+", mode="regex").get()
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd': {'c': 'green', 'l': [5, 6]}}]

            >>> asset.find("yenlow", mode="fuzzy").get()
            [{'s': 'DEF', 'b': False, 'd': {'c': 'yellow', 'l': [7, 8]}}]

            >>> asset.find("xyz", in_dumps=True).get()
            [{'s': 'EFG', 'b': False, 'd': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]
            ```
        """
        return self.apply(
            FindAssetFunc,
            target=target,
            path=path,
            keep_path=keep_path,
            skip_missing=skip_missing,
            source=source,
            in_dumps=in_dumps,
            template_context=template_context,
            **kwargs,
        )

    def replace_(
        self,
        target: tp.MaybeList[tp.Any],
        replacement: tp.MaybeList[tp.Any],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> KnowledgeAssetT:
        """Find and replace occurrences and return a new `KnowledgeAsset` instance.

        Uses `KnowledgeAsset.apply` on `ReplaceAssetFunc`.

        Uses `vectorbtpro.utils.search.find_in_obj` (keyword arguments are passed here) to find
        occurrences in each object. Then, uses `vectorbtpro.utils.search.replace_in_obj` to replace them.

        Target can be one or multiple of objects.

        Use argument `path` to specify what part of the object should be searched. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. If `keep_path` is True, object will be represented
        as a nested dictionary with path as keys. If multiple paths are provided, `keep_path` automatically
        becomes True, and they will be merged into one nested dictionary. If `skip_missing` is True
        and path is missing in the object, will skip the object.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only objects that have been changed.

        Usage:
            ```pycon
            >>> asset.replace_("BC", "XY").get()
            [{'s': 'AXY', 'b': True, 'd': {'c': 'red', 'l': [1, 2]}},
             {'s': 'XYD', 'b': True, 'd': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.replace_("BC", "XY", changed_only=True).get()
            [{'s': 'AXY', 'b': True, 'd': {'c': 'red', 'l': [1, 2]}},
             {'s': 'XYD', 'b': True, 'd': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.replace_(r"(D)E(F)", r"\1X\2", mode="regex", changed_only=True).get()
            [{'s': 'DXF', 'b': False, 'd': {'c': 'yellow', 'l': [7, 8]}}]

            >>> asset.replace_(True, False, changed_only=True).get()
            [{'s': 'ABC', 'b': False, 'd': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': False, 'd': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.replace_(3, 3000, path="d.l", changed_only=True).get()
            [{'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'l': [3000, 4]}}]

            >>> asset.replace_([1, 3], [1000, 3000], path="d.l", changed_only=True).get()
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red', 'l': [1000, 2]}},
             {'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'l': [3000, 4]}}]

            >>> asset.replace_("a", "X", path=["s", "d.c"], ignore_case=True, changed_only=True).get()
            [{'s': 'XBC', 'b': True, 'd': {'c': 'red', 'l': [1, 2]}},
             {'s': 'EFG', 'b': False, 'd': {'c': 'blXck', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.replace_(123, 456, path="xyz", skip_missing=True, changed_only=True).get()
            [{'s': 'EFG', 'b': False, 'd': {'c': 'black', 'l': [9, 10]}, 'xyz': 456}]
            ```
        """
        return self.apply(
            ReplaceAssetFunc,
            target=target,
            replacement=replacement,
            path=path,
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
        """Flatten objects or parts of them.

        Uses `KnowledgeAsset.apply` on `FlattenAssetFunc`.

        Use argument `path` to specify what part of the object should be set. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the object, will skip the object.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only objects that have been changed.

        Keyword arguments are passed to `vectorbtpro.utils.search.flatten_obj`.

        Usage:
            ```pycon
            >>> asset.flatten().get()
            [{'s': 'ABC',
              'b': True,
              ('d', 'c'): 'red',
              ('d', 'l', 0): 1,
              ('d', 'l', 1): 2},
              ...
             {'s': 'EFG',
              'b': False,
              ('d', 'c'): 'black',
              ('d', 'l', 0): 9,
              ('d', 'l', 1): 10,
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
        """Unflatten objects or parts of them.

        Uses `KnowledgeAsset.apply` on `UnflattenAssetFunc`.

        Use argument `path` to specify what part of the object should be set. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the object, will skip the object.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only objects that have been changed.

        Keyword arguments are passed to `vectorbtpro.utils.search.unflatten_obj`.

        Usage:
            ```pycon
            >>> asset.flatten().unflatten().get()
            [{'s': 'ABC', 'b': True, 'd': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]
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
