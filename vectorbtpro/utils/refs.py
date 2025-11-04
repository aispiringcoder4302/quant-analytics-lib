# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for working with references."""

from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.util
import inspect
import itertools
import sys
import urllib.request
import webbrowser
from collections import defaultdict
from functools import cached_property, lru_cache
from types import ModuleType

from vectorbtpro import _typing as tp
from vectorbtpro.utils.module_ import resolve_module, get_module, package_shortcut_config

__all__ = [
    "get_refname",
    "get_obj",
    "imlucky",
    "get_api_ref",
    "open_api_ref",
]


def get_caller_qualname() -> tp.Optional[str]:
    """Return the qualified name of the calling function or method.

    Returns:
        Optional[str]: Qualified name of the function or method that invoked this function.
    """
    frame = inspect.currentframe()
    try:
        caller_frame = frame.f_back
        code = caller_frame.f_code
        func_name = code.co_name
        locals_ = caller_frame.f_locals
        if "self" in locals_:
            cls = locals_["self"].__class__
            return f"{cls.__qualname__}.{func_name}"
        elif "cls" in locals_:
            cls = locals_["cls"]
            return f"{cls.__qualname__}.{func_name}"
        else:
            module = inspect.getmodule(caller_frame)
            if module:
                func = module.__dict__.get(func_name)
                if func and hasattr(func, "__qualname__"):
                    return func.__qualname__
            return func_name
    finally:
        del frame


def get_method_class(meth: tp.Callable) -> tp.Optional[tp.Type]:
    """Return the class associated with the given method, if available.

    Args:
        meth (Callable): Method or function for which to determine the associated class.

    Returns:
        Optional[type]: Class object if found, otherwise None.
    """
    if inspect.ismethod(meth) or (
        inspect.isbuiltin(meth)
        and getattr(meth, "__self__", None) is not None
        and getattr(meth.__self__, "__class__", None)
    ):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = getattr(meth, "__func__", meth)
    if inspect.isfunction(meth):
        cls = getattr(get_module(meth), meth.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0], None)
        if isinstance(cls, type):
            return cls
    return getattr(meth, "__objclass__", None)


def get_obj_refname(obj: tp.Any) -> str:
    """Return the reference name for the provided object.

    Args:
        obj (Any): Target object.

    Returns:
        str: Reference name.
    """
    from vectorbtpro.utils.decorators import (
        class_property,
        custom_property,
        hybrid_property,
    )

    if inspect.ismodule(obj):
        return obj.__name__
    if inspect.isclass(obj):
        return obj.__module__ + "." + obj.__qualname__

    if isinstance(obj, staticmethod):
        return get_obj_refname(obj.__func__)
    if isinstance(obj, classmethod):
        return get_obj_refname(obj.__func__)

    if (
        inspect.isdatadescriptor(obj)
        or inspect.ismethoddescriptor(obj)
        or inspect.isgetsetdescriptor(obj)
        or inspect.ismemberdescriptor(obj)
    ):
        cls = getattr(obj, "__objclass__", None)
        name = getattr(obj, "__name__", None)
        if cls and name:
            return get_obj_refname(cls) + "." + name

    if inspect.ismethod(obj) or inspect.isfunction(obj):
        cls = get_method_class(obj)
        if cls is not None:
            return get_obj_refname(cls) + "." + obj.__name__
        if hasattr(obj, "func"):
            return get_obj_refname(obj.func)

    if isinstance(obj, (class_property, hybrid_property, custom_property)):
        return get_obj_refname(obj.func)
    if isinstance(obj, cached_property) and hasattr(obj, "func"):
        return get_obj_refname(obj.func)
    if isinstance(obj, property):
        return get_obj_refname(obj.fget)

    if hasattr(obj, "__name__"):
        module = get_module(obj)
        if module is not None and obj.__name__ in module.__dict__:
            return get_obj_refname(module) + "." + obj.__name__

    module = get_module(obj)
    if module is not None:
        for k, v in module.__dict__.items():
            if obj is v:
                return get_obj_refname(module) + "." + k

    return get_obj_refname(type(obj))


def annotate_refname_parts(refname: str, allow_partial: bool = False) -> tp.Tuple[dict, ...]:
    """Annotate each part of a reference name with its corresponding object.

    Args:
        refname (str): Reference name.
        allow_partial (bool): Whether to allow partial resolution of the reference name.

    Returns:
        Tuple[dict, ...]: Tuple of dictionaries, each containing:

            * `name`: Reference name part.
            * `obj`: Object corresponding to the reference name part.
    """
    from vectorbtpro.utils.attr_ import get_attr

    refname_parts = refname.split(".")
    obj = None
    annotated_parts = []
    refname_so_far = None
    for i, name in enumerate(refname_parts):
        if obj is None:
            if i == 0:
                obj = importlib.import_module(name)
        else:
            try:
                obj = get_attr(obj, name)
            except AttributeError as e:
                if refname_so_far == "vectorbtpro.indicators.factory":
                    import vectorbtpro as vbt

                    obj = getattr(vbt, name)
                elif refname_so_far.startswith("vectorbtpro.indicators.factory."):
                    obj = obj(name)
                else:
                    if allow_partial:
                        obj = None
                    else:
                        raise e
        annotated_parts.append(dict(name=name, obj=obj))
        if refname_so_far is None:
            refname_so_far = name
        else:
            refname_so_far += "." + name
    return tuple(annotated_parts)


def get_refname_obj(refname: str, raise_missing: bool = False) -> tp.Any:
    """Return the object corresponding to the provided reference name.

    Args:
        refname (str): Reference name.
        raise_missing (bool): Whether to raise an error if the object cannot be found.

    Returns:
        Any: Object obtained by importing modules and accessing attributes.
    """
    refname_parts = annotate_refname_parts(refname, allow_partial=not raise_missing)
    if not refname_parts:
        return None
    return refname_parts[-1]["obj"]


def split_refname(
    refname: str,
    module: tp.Optional[tp.ModuleLike] = None,
) -> tp.Tuple[tp.Optional[ModuleType], tp.Optional[str]]:
    """Return the module and qualified name extracted from the given reference name.

    Args:
        refname (str): Reference name.
        module (Optional[ModuleLike]): Module context for extraction.

    Returns:
        Tuple[Optional[ModuleType], Optional[str]]: Tuple containing the module and qualified name.
    """
    if module is not None:
        module = resolve_module(module)
    refname_parts = refname.split(".")
    if module is None:
        module = importlib.import_module(refname_parts[0])
        refname_parts = refname_parts[1:]
        if len(refname_parts) == 0:
            return module, None
        return split_refname(".".join(refname_parts), module=module)
    elif inspect.ismodule(getattr(module, refname_parts[0])):
        module = getattr(module, refname_parts[0])
        refname_parts = refname_parts[1:]
        if len(refname_parts) == 0:
            return module, None
        return split_refname(".".join(refname_parts), module=module)
    else:
        return module, ".".join(refname_parts)


def resolve_refname(refname: str, module: tp.Optional[tp.ModuleLike] = None) -> tp.Optional[tp.MaybeList[str]]:
    """Resolve a reference name into its fully qualified form using the provided module context.

    !!! note
        This function attempts to resolve the reference name by checking the module context
        and its attributes. It may return multiple reference names if the reference name is ambiguous.

    Args:
        refname (str): Reference name to resolve.

            A reference name may be a fully qualified dotted path ("vectorbtpro.data.base.Data"),
            a library re-export ("vectorbtpro.Data"), a common alias ("vbt.Data"),
            or a simple name ("Data") that uniquely identifies an object.
        module (Optional[ModuleLike]): Module context used in reference resolution.

    Returns:
        Optional[MaybeList[str]]: Reference name(s), or None if resolution fails.
    """
    from vectorbtpro.utils.attr_ import get_attr

    if module is not None:
        module = resolve_module(module)
    if refname == "":
        if module is None:
            return None
        return module.__name__
    refname_parts = refname.split(".")
    if module is None:
        if refname_parts[0] in package_shortcut_config:
            refname_parts[0] = package_shortcut_config[refname_parts[0]]
            module = importlib.import_module(refname_parts[0])
            refname_parts = refname_parts[1:]
        else:
            try:
                module = importlib.import_module(refname_parts[0])
                refname_parts = refname_parts[1:]
            except ImportError:
                module = importlib.import_module("vectorbtpro")
    if len(refname_parts) == 0:
        return module.__name__
    if refname_parts[0] in package_shortcut_config:
        if package_shortcut_config[refname_parts[0]] == module.__name__:
            refname_parts[0] = package_shortcut_config[refname_parts[0]]
    if refname_parts[0] == module.__name__ and refname_parts[0] not in module.__dict__:
        refname_parts = refname_parts[1:]
        if len(refname_parts) == 0:
            return module.__name__

    if refname_parts[0] in module.__dict__:
        obj = module.__dict__[refname_parts[0]]
        if inspect.ismodule(obj):
            parent_module = ".".join(obj.__name__.split(".")[:-1])
        else:
            parent_module = get_module(obj)
            if parent_module is not None:
                if refname_parts[0] in parent_module.__dict__:
                    parent_module = parent_module.__name__
                else:
                    parent_module = None
        if parent_module is None or parent_module == module.__name__:
            if inspect.ismodule(obj):
                module = getattr(module, refname_parts[0])
                refname_parts = refname_parts[1:]
                return resolve_refname(".".join(refname_parts), module=module)
            if hasattr(obj, "__name__") and obj.__name__ in module.__dict__:
                obj = module.__dict__[obj.__name__]
                refname_parts[0] = obj.__name__
            if len(refname_parts) == 1:
                return module.__name__ + "." + refname_parts[0]
            if not isinstance(obj, type):
                cls = type(obj)
            else:
                cls = obj
            k = refname_parts[1]
            v = get_attr(cls, k, None)
            found_super_cls = None
            for super_cls in inspect.getmro(cls)[1:]:
                if k in dir(super_cls):
                    v2 = get_attr(super_cls, k, None)
                    if v2 is not None and v == v2:
                        found_super_cls = super_cls
            if found_super_cls is not None:
                cls_path = found_super_cls.__module__ + "." + found_super_cls.__name__
                return cls_path + "." + ".".join(refname_parts[1:])
            return module.__name__ + "." + ".".join(refname_parts)
        if inspect.ismodule(obj):
            parent_module = obj
            refname_parts = refname_parts[1:]
        return resolve_refname(".".join(refname_parts), module=parent_module)

    refnames = []
    visited_modules = set()
    for k, v in module.__dict__.items():
        if v is not module:
            if inspect.ismodule(v) and v.__name__.startswith(module.__name__) and v.__name__ not in visited_modules:
                visited_modules.add(v.__name__)
                refname = resolve_refname(".".join(refname_parts), module=v)
                if refname is not None:
                    if isinstance(refname, str):
                        refname = [refname]
                    for r in refname:
                        if r not in refnames:
                            refnames.append(r)
    if len(refnames) > 1:
        return refnames
    if len(refnames) == 1:
        return refnames[0]
    return None


def get_refname(
    obj: tp.Any,
    module: tp.Optional[tp.ModuleLike] = None,
    resolve: bool = True,
) -> tp.Optional[tp.MaybeList[str]]:
    """Return the reference name(s) for the provided object.

    Args:
        obj (Any): Object from which to extract the reference name.

            If a tuple is provided, its elements are concatenated.
            If a string is provided, it is treated as a reference name.
        module (Optional[ModuleLike]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the reference to an actual object.

    Returns:
        Optional[MaybeList[str]]: Reference name as a string, a list of strings if multiple
            reference names are found, or None.
    """
    if type(obj) is tuple:
        if len(obj) == 1:
            obj = obj[0]
        else:
            first_refname = get_obj_refname(obj[0])
            obj = first_refname + "." + ".".join(obj[1:])
    if isinstance(obj, str):
        refname = obj
    else:
        refname = get_obj_refname(obj)
    if resolve:
        return resolve_refname(refname, module=module)
    return refname


def get_obj(*args, allow_multiple: bool = False, **kwargs) -> tp.MaybeList:
    """Return the object by its reference name.

    Args:
        *args: Positional arguments for `get_refname`.
        allow_multiple (bool): Whether to allow returning multiple objects
            if more than one reference name is found.
        **kwargs: Keyword arguments for `get_refname`.

    Returns:
        MaybeList: Object or a list of objects if multiple reference names are found.
    """
    refname = get_refname(*args, **kwargs)
    if isinstance(refname, list):
        obj = None
        for _refname in refname:
            _obj = get_refname_obj(_refname)
            if obj is None:
                obj = _obj
            elif not isinstance(obj, list):
                if _obj is not obj:
                    if not allow_multiple:
                        raise ValueError("Multiple reference names found:\n\n* {}".format("\n* ".join(refname)))
                    obj = [obj, _obj]
            else:
                if _obj not in obj:
                    obj.append(_obj)
        return obj
    return get_refname_obj(refname)


def ensure_refname(
    obj: tp.Any,
    module: tp.Optional[tp.ModuleLike] = None,
    resolve: bool = True,
    vbt_only: bool = False,
    return_parts: bool = False,
    raise_error: bool = True,
) -> tp.Union[None, str, tp.Tuple[str, ModuleType, str]]:
    """Return the reference name for an object and optionally its module and qualified name.

    Args:
        obj (Any): Object from which to extract the reference name.

            If a tuple is provided, its elements are concatenated.
            If a string is provided, it is treated as a reference name.
        module (Optional[ModuleLike]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the reference to an actual object.
        vbt_only (bool): If True, limit resolution to objects within vectorbtpro.
        return_parts (bool): If True, return a tuple containing the reference name, module, and qualified name.
        raise_error (bool): Whether to raise an error if the reference name cannot be determined.

    Returns:
        Union[None, str, Tuple[str, ModuleType, str]]: Reference name as a string, or a tuple of
            (reference name, module, qualified name) if `return_parts` is True;
            or None if the reference name cannot be determined.
    """

    def _raise_error():
        raise ValueError(
            f"Couldn't find the reference name for {obj!r}, or the object is external. "
            "If the object is internal, please decompose the object or provide a string instead."
        )

    refname = get_refname(obj, module=module, resolve=resolve)
    if refname is None:
        if raise_error:
            _raise_error()
        return None
    if isinstance(refname, list):
        raise ValueError("Multiple reference names found:\n\n* {}".format("\n* ".join(refname)))
    if vbt_only or return_parts or resolve:
        module, qualname = split_refname(refname)
        if module.__name__.split(".")[0] != "vectorbtpro" and vbt_only:
            if raise_error:
                _raise_error()
            return None
        if return_parts:
            return refname, module, qualname
        if resolve:
            if qualname is None:
                return module.__name__
            return module.__name__ + "." + qualname
    return refname


def get_imlucky_url(query: str) -> str:
    """Construct a DuckDuckGo "I'm lucky" URL for a query.

    Args:
        query (str): Search query.

    Returns:
        str: DuckDuckGo "I'm lucky" URL based on the query.
    """
    return "https://duckduckgo.com/?q=!ducky+" + urllib.request.pathname2url(query)


def imlucky(query: str, **kwargs) -> bool:
    """Open a DuckDuckGo "I'm lucky" URL for a query in the web browser.

    Args:
        query (str): Search query.
        **kwargs: Keyword arguments for `webbrowser.open`.

    Returns:
        bool: True if the browser was opened successfully, False otherwise.
    """
    return webbrowser.open(get_imlucky_url(query), **kwargs)


def get_api_ref(
    obj: tp.Any,
    module: tp.Optional[tp.ModuleLike] = None,
    resolve: bool = True,
    vbt_only: bool = False,
) -> str:
    """Return the API reference URL for an object.

    Args:
        obj (Any): Object for which the API reference is constructed.
        module (Optional[ModuleLike]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the reference to an actual object.
        vbt_only (bool): If True, limit resolution to objects within vectorbtpro.

    Returns:
        str: API reference URL for the given object.
    """
    refname, module, qualname = ensure_refname(
        obj,
        module=module,
        resolve=resolve,
        vbt_only=vbt_only,
        return_parts=True,
    )
    if module.__name__.split(".")[0] == "vectorbtpro":
        api_url = "https://github.com/polakowo/vectorbt.pro/blob/pvt-links/api/"
        md_url = api_url + module.__name__ + ".md/"
        if qualname is None:
            return md_url + "#" + module.__name__.replace(".", "")
        return md_url + "#" + module.__name__.replace(".", "") + qualname.replace(".", "")
    if resolve:
        if qualname is None:
            search_query = module.__name__
        else:
            search_query = module.__name__ + "." + qualname
    else:
        search_query = refname
    return get_imlucky_url(search_query)


def open_api_ref(
    obj: tp.Any,
    module: tp.Optional[tp.ModuleLike] = None,
    resolve: bool = True,
    **kwargs,
) -> bool:
    """Open the API reference URL for an object in the web browser.

    Args:
        obj (Any): Object whose API reference is to be opened.
        module (Optional[ModuleLike]): Module context used in reference resolution.
        resolve (bool): Whether to resolve the reference to an actual object.
        **kwargs: Keyword arguments for `webbrowser.open`.

    Returns:
        bool: True if the browser was opened successfully, False otherwise.
    """
    return webbrowser.open(get_api_ref(obj, module=module, resolve=resolve), **kwargs)


def get_nonlocal_usages(
    module: tp.ModuleLike,
    include_builtins: bool = True,
    expand_star_imports: bool = False,
    return_matrix: bool = False,
    return_meta: bool = False,
) -> list:
    """Extract nonlocal name usages from the specified module.

    This analyzes name loads inside each scope (modules, functions, classes,
    lambdas, and comprehensions) and resolves them to fully qualified reference names.
    Locals of the current scope are excluded; imports and names from enclosing scopes,
    the module scope, or builtins may be included depending on the settings.

    Args:
        module (ModuleLike): Fully-qualified module name or module object.
        include_builtins (bool): If True, include builtins as `"builtins.<name>"` when no nearer binding exists.
        expand_star_imports (bool): If True, attempts to resolve `from x import *` by importing the module
            and expanding its public names.
        return_matrix (bool): If True, returns a list whose length equals the number of lines
            in the module's source; each entry contains results for that line.

            If False, returns a flat list across the whole file.
        return_meta (bool): If True, returns detailed hit dicts with keys:

            * `name`: used name.
            * `refname`: reference name of the used name.
            * `lineno`: line number (1-indexed).
            * `end_lineno`: end line number (1-indexed).
            * `col_offset`: column offset (0-indexed).
            * `end_col_offset`: end column offset (0-indexed).
            * `block`: which part of the scope the usage is in (e.g., "decorator", "head", "body", etc.).
            * `role`: syntactic role of the usage (e.g., "expr", "annotation", "default", etc.).
            * `scope_refname`: reference name of the scope where the usage occurs.
            * `source_line`: source code line where the usage occurs.

            If False, returns only reference name strings.

    Returns:
        list: When `return_matrix` is True, returns a list of per-line results
            (where lines with no hits are empty lists), otherwise a flat list.

            Also, when `return_meta` is True, each result is a list of metadata dicts.

    """
    from vectorbtpro.utils.attr_ import get_attr
    from vectorbtpro.utils.source import get_source

    module = resolve_module(module)
    source = get_source(module.__name__)
    source_lines = source.splitlines()
    n_lines = len(source_lines)
    tree = ast.parse(source, type_comments=True)

    def _is_scope(n):
        return isinstance(
            n,
            (
                ast.Module,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Lambda,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        )

    def _add_target_names(target, out):
        if isinstance(target, ast.Name):
            out.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                _add_target_names(elt, out)
        elif isinstance(target, ast.Starred):
            _add_target_names(target.value, out)

    def _add_pattern_binds(pat, out):
        if not sys.version_info >= (3, 10):
            return
        T = ast
        if isinstance(pat, getattr(T, "MatchAs", ())):
            if pat.name:
                out.add(pat.name)
            if pat.pattern:
                _add_pattern_binds(pat.pattern, out)
        elif isinstance(pat, getattr(T, "MatchStar", ())):
            if pat.name:
                out.add(pat.name)
        elif isinstance(pat, getattr(T, "MatchOr", ())):
            for p in pat.patterns:
                _add_pattern_binds(p, out)
        elif isinstance(pat, getattr(T, "MatchSequence", ())):
            for p in pat.patterns:
                _add_pattern_binds(p, out)
        elif isinstance(pat, getattr(T, "MatchMapping", ())):
            for p in pat.patterns:
                _add_pattern_binds(p, out)
            if pat.rest:
                out.add(pat.rest)
        elif isinstance(pat, getattr(T, "MatchClass", ())):
            for p in pat.patterns:
                _add_pattern_binds(p, out)
            for kp in getattr(pat, "kwd_patterns", []):
                _add_pattern_binds(kp, out)

    def _absolutize(level, name):
        if level == 0:
            return name
        if not module.__name__:
            return None
        parts = module.__name__.split(".")
        if level > len(parts):
            return None
        base_parts = parts[:-level]
        if not base_parts:
            return name or None
        base = ".".join(base_parts)
        return f"{base}.{name}" if name else base

    class ScopeInfo:
        def __init__(self, node, parent, refname):
            self._final = None

            self.node = node
            self.parent = parent
            self.locals = set()
            self.globals = set()
            self.nonlocals = set()
            self.imports = {}
            self.star_imports = []
            self.refname = refname

        def final_locals(self):
            if self._final is None:
                self._final = self.locals - self.globals - self.nonlocals
            return self._final

    scope_infos = {}

    def _scope_refname_for(node, parent):
        base = parent.refname if parent else module.__name__
        if isinstance(node, ast.Module):
            return module.__name__
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return f"{base}.{node.name}"
        tag = type(node).__name__.lower()
        return f"{base}.<{tag}>"

    class LocalCollector(ast.NodeVisitor):
        def __init__(self):
            self._stack = []

        def visit(self, node):
            if _is_scope(node):
                parent = self._stack[-1] if self._stack else None
                sc = ScopeInfo(node, parent, _scope_refname_for(node, parent))
                scope_infos[node] = sc
                self._stack.append(sc)
                self._collect_within(node, sc)
                super().visit(node)
                self._stack.pop()
            else:
                super().visit(node)

        def _collect_within(self, root, sc):
            for child in ast.iter_child_nodes(root):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    sc.locals.add(child.name)
                if hasattr(ast, "TypeAlias") and isinstance(child, getattr(ast, "TypeAlias")):
                    try:
                        sc.locals.add(child.name.id)
                    except Exception:
                        pass

            if isinstance(root, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                for gen in root.generators:
                    _add_target_names(gen.target, sc.locals)

            def _add_args_as_locals(args):
                if not args:
                    return
                for a in itertools.chain(getattr(args, "posonlyargs", ()), args.args, args.kwonlyargs):
                    sc.locals.add(a.arg)
                if getattr(args, "vararg", None):
                    sc.locals.add(args.vararg.arg)
                if getattr(args, "kwarg", None):
                    sc.locals.add(args.kwarg.arg)

            if isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                _add_args_as_locals(getattr(root, "args", None))

            def _bind_type_params_if_any(node, sc):
                tp = getattr(node, "type_params", None)
                if not tp:
                    return
                for p in getattr(tp, "params", []):
                    name = getattr(p, "name", None)
                    if isinstance(name, ast.Name):
                        sc.locals.add(name.id)
                    elif isinstance(name, str):
                        sc.locals.add(name)

            if isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                _bind_type_params_if_any(root, sc)

            def _walk_here(n):
                for child in ast.iter_child_nodes(n):
                    if _is_scope(child):
                        continue
                    if isinstance(child, ast.Global):
                        sc.globals.update(child.names)
                        continue
                    if isinstance(child, ast.Nonlocal):
                        sc.nonlocals.update(child.names)
                        continue
                    if isinstance(child, ast.Import):
                        for alias in child.names:
                            local = alias.asname or alias.name.split(".", 1)[0]
                            target = alias.name if alias.asname else alias.name.split(".", 1)[0]
                            sc.locals.add(local)
                            sc.imports[local] = target
                        continue
                    if isinstance(child, ast.ImportFrom):
                        abs_mod = _absolutize(child.level or 0, child.module)
                        for alias in child.names:
                            if alias.name == "*":
                                if abs_mod:
                                    sc.star_imports.append(abs_mod)
                                continue
                            local = alias.asname or alias.name
                            if abs_mod:
                                sc.locals.add(local)
                                sc.imports[local] = f"{abs_mod}.{alias.name}"
                            else:
                                sc.locals.add(local)
                                sc.imports[local] = alias.name
                        continue
                    if isinstance(child, ast.Assign):
                        for t in child.targets:
                            _add_target_names(t, sc.locals)
                    elif isinstance(child, ast.AugAssign):
                        _add_target_names(child.target, sc.locals)
                    elif isinstance(child, ast.AnnAssign):
                        if child.value is not None:
                            _add_target_names(child.target, sc.locals)
                    elif isinstance(child, (ast.For, ast.AsyncFor)):
                        _add_target_names(child.target, sc.locals)
                    elif isinstance(child, (ast.With, ast.AsyncWith)):
                        for item in child.items:
                            if item.optional_vars:
                                _add_target_names(item.optional_vars, sc.locals)
                    elif isinstance(child, ast.NamedExpr):
                        _add_target_names(child.target, sc.locals)
                    elif isinstance(child, ast.ExceptHandler):
                        nm = getattr(child, "name", None)
                        if sys.version_info >= (3, 11) and isinstance(nm, ast.Name):
                            sc.locals.add(nm.id)
                        elif isinstance(nm, str):
                            sc.locals.add(nm)
                    if sys.version_info >= (3, 10) and isinstance(child, getattr(ast, "Match", ())):
                        for case in child.cases:
                            _add_pattern_binds(case.pattern, sc.locals)
                    _walk_here(child)

            _walk_here(root)

    LocalCollector().visit(tree)

    if expand_star_imports:
        for sc in scope_infos.values():
            for modname in sc.star_imports:
                spec = importlib.util.find_spec(modname)
                if not spec or not (spec.origin or "").endswith((".py", ".pyc")):
                    continue
                try:
                    mod = importlib.import_module(modname)
                except Exception:
                    continue
                public = getattr(mod, "__all__", None)
                if public is None:
                    public = [n for n in dir(mod) if not n.startswith("_")]
                for n in public:
                    if n not in sc.locals:
                        sc.locals.add(n)
                        sc.imports[n] = f"{modname}.{n}"
            sc.star_imports.clear()

    builtin_names = set()
    if include_builtins:
        import builtins as _b

        builtin_names = set(dir(_b))

    module_scope = scope_infos.get(tree)

    def _is_plain_segment(seg):
        return seg and not seg.startswith("<") and seg.isidentifier()

    @lru_cache(maxsize=None)
    def _resolve_scope_object(scope_refname):
        if not isinstance(scope_refname, str):
            return None, None
        mod_name = module.__name__
        if not scope_refname.startswith(mod_name):
            return None, None
        mod_parts = mod_name.split(".")
        scope_parts = scope_refname.split(".")
        if scope_parts[: len(mod_parts)] != mod_parts:
            return None, None

        obj = module
        walked = mod_parts[:]
        for seg in scope_parts[len(mod_parts) :]:
            if not _is_plain_segment(seg):
                return None, None
            try:
                obj = get_attr(obj, seg)
            except AttributeError:
                return None, None
            walked.append(seg)
        return obj, ".".join(walked)

    @lru_cache(maxsize=None)
    def _attr_path_if_accessible(scope_refname, name):
        obj, dotted = _resolve_scope_object(scope_refname)
        if obj is None:
            return None if name is not None else None
        if name is None:
            return dotted
        if not _is_plain_segment(name):
            return None
        try:
            get_attr(obj, name)
            return f"{dotted}.{name}"
        except AttributeError:
            return None

    def _resolve_refname(name, cur):
        if name in getattr(cur, "globals", ()):
            if module_scope is not None:
                if name in module_scope.imports:
                    return module_scope.imports[name]
                if name in module_scope.final_locals():
                    dotp = _attr_path_if_accessible(module_scope.refname, name)
                    return dotp or f"{module_scope.refname}::{name}"
            dotp = _attr_path_if_accessible(module_scope.refname, name)
            return dotp or f"{module_scope.refname}::{name}"
        sc = cur
        while sc is not None:
            if name in sc.imports:
                return sc.imports[name]
            if name in sc.final_locals():
                dotp = _attr_path_if_accessible(sc.refname, name)
                return dotp or f"{sc.refname}::{name}"
            sc = sc.parent
        if include_builtins and name in builtin_names:
            return f"builtins.{name}"
        return None

    class UseCollector(ast.NodeVisitor):
        def __init__(self):
            self._stack = []
            self._block_stack = ["body"]
            self._role_stack = ["expr"]
            self._name_nodes_emitted = set()

            self.per_line_meta = defaultdict(list)

        @contextlib.contextmanager
        def _ctx(self, *, block=None, role=None):
            if block is not None:
                self._block_stack.append(block)
            if role is not None:
                self._role_stack.append(role)
            try:
                yield
            finally:
                if role is not None:
                    self._role_stack.pop()
                if block is not None:
                    self._block_stack.pop()

        def _visit_in(self, node, *, block=None, role=None):
            if node is None:
                return
            with self._ctx(block=block, role=role):
                self.visit(node)

        def _emit(self, node, refname):
            scope = self._stack[-1] if self._stack else None
            scope_refname = scope.refname if scope else module.__name__
            scope_dotted = _attr_path_if_accessible(scope_refname, None)
            hit = {
                "name": node.id,
                "refname": refname,
                "lineno": node.lineno,
                "end_lineno": getattr(node, "end_lineno", None),
                "col_offset": node.col_offset,
                "end_col_offset": getattr(node, "end_col_offset", None),
                "block": self._block_stack[-1],
                "role": self._role_stack[-1],
                "scope_refname": scope_dotted if scope_dotted else scope_refname,
                "source_line": source_lines[node.lineno - 1] if 1 <= node.lineno <= n_lines else "",
            }
            self.per_line_meta[node.lineno].append(hit)

        def visit(self, node):
            if _is_scope(node):
                self._stack.append(scope_infos[node])
                super().visit(node)
                self._stack.pop()
            else:
                super().visit(node)

        def _visit_function_like(self, node):
            for dec in getattr(node, "decorator_list", ()):
                self._visit_in(dec, block="decorator", role="expr")
            args = node.args
            for arg in itertools.chain(getattr(args, "posonlyargs", ()), args.args, args.kwonlyargs):
                if getattr(arg, "annotation", None):
                    self._visit_in(arg.annotation, block="head", role="annotation")
            if getattr(args, "vararg", None) and args.vararg.annotation:
                self._visit_in(args.vararg.annotation, block="head", role="annotation")
            if getattr(args, "kwarg", None) and args.kwarg.annotation:
                self._visit_in(args.kwarg.annotation, block="head", role="annotation")
            for default in getattr(args, "defaults", ()):
                self._visit_in(default, block="head", role="default")
            for default in getattr(args, "kw_defaults", ()):
                if default is not None:
                    self._visit_in(default, block="head", role="default")
            if getattr(node, "returns", None):
                self._visit_in(node.returns, block="head", role="returns")
            for stmt in node.body:
                self._visit_in(stmt, block="body", role="expr")

        def visit_FunctionDef(self, node):
            self._visit_function_like(node)

        def visit_AsyncFunctionDef(self, node):
            self._visit_function_like(node)

        def visit_ClassDef(self, node):
            for dec in node.decorator_list:
                self._visit_in(dec, block="decorator", role="expr")
            for base in node.bases:
                self._visit_in(base, block="head", role="base")
            for kw in node.keywords:
                role = "metaclass" if kw.arg == "metaclass" else "keyword"
                self._visit_in(kw.value, block="head", role=role)
            for stmt in node.body:
                self._visit_in(stmt, block="body", role="expr")

        def visit_TypeAlias(self, node):
            self._visit_in(node.value, block="head", role="annotation")

        def visit_AnnAssign(self, node):
            if node.annotation:
                self._visit_in(node.annotation, role="annotation")
            if node.value:
                self._visit_in(node.value)

        def visit_Attribute(self, node):
            attrs = []
            base = node
            while isinstance(base, ast.Attribute):
                attrs.append(base.attr)
                base = base.value
            if isinstance(base, ast.Name) and isinstance(base.ctx, ast.Load) and self._stack:
                cur = self._stack[-1]
                if base.id in cur.final_locals():
                    return
                refname = _resolve_refname(base.id, cur)
                if refname:
                    tail = ".".join(reversed(attrs))
                    self._emit(base, f"{refname}.{tail}")
                    self._name_nodes_emitted.add(id(base))
                return
            self.visit(node.value)

        def visit_Name(self, node):
            if id(node) in getattr(self, "_name_nodes_emitted", ()):
                return
            if self._stack and isinstance(node.ctx, ast.Load):
                cur = self._stack[-1]
                locals_final = cur.final_locals()
                if node.id not in locals_final:
                    refname = _resolve_refname(node.id, cur)
                    if refname:
                        self._emit(node, refname)

    collector = UseCollector()
    collector.visit(tree)

    if return_matrix:
        matrix = [[] for _ in range(n_lines)]
        if return_meta:
            for ln, items in collector.per_line_meta.items():
                if 1 <= ln <= n_lines:
                    matrix[ln - 1] = sorted(items, key=lambda d: d["col_offset"])
        else:
            for ln, items in collector.per_line_meta.items():
                if 1 <= ln <= n_lines:
                    items_sorted = sorted(items, key=lambda d: d["col_offset"])
                    seen, refnames = set(), []
                    for d in items_sorted:
                        if d["refname"] not in seen:
                            seen.add(d["refname"])
                            refnames.append(d["refname"])
                    matrix[ln - 1] = refnames
        return matrix
    else:
        if return_meta:
            meta_items = []
            for ln, items in collector.per_line_meta.items():
                if 1 <= ln <= n_lines:
                    meta_items.extend(items)
            return sorted(meta_items, key=lambda d: (d["lineno"], d["col_offset"]))
        else:
            seen, refnames = set(), []
            for ln, items in collector.per_line_meta.items():
                if 1 <= ln <= n_lines:
                    items_sorted = sorted(items, key=lambda d: d["col_offset"])
                    for d in items_sorted:
                        if d["refname"] not in seen:
                            seen.add(d["refname"])
                            refnames.append(d["refname"])
            return refnames
