# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for formatting."""

import inspect
import io
import re
import ast
import difflib
import tempfile
import webbrowser
from pathlib import Path
from types import ModuleType

import attr
import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils.base import Base

__all__ = [
    "prettify",
    "format_func",
    "pprint",
    "ptable",
    "phelp",
    "pdir",
    "dump",
    "refine_source",
    "refine_docstrings",
]


def camel_to_snake_case(camel_str: str) -> str:
    """Convert a camel case string to a snake case string.

    Args:
        camel_str (str): A string formatted in camel case.

    Returns:
        str: The string converted to snake case.
    """
    snake_str = re.sub(r"(?<!^)(?<![A-Z_])([A-Z])", r"_\1", camel_str).lower()
    if snake_str.startswith("_"):
        snake_str = snake_str[1:]
    return snake_str


class Prettified(Base):
    """Abstract class for objects that can be prettified."""

    def prettify(self, **kwargs) -> str:
        """Prettify the object.

        !!! warning
            Calling `prettify` can lead to an infinite recursion.
            Make sure to pre-process this object.

        Returns:
            str: A prettified representation of the object.
        """
        raise NotImplementedError

    def prettify_doc(self, **kwargs) -> str:
        """Prettify the object for documentation, equivalent to using
        `Prettified.prettify` with `repr_doc` as `repr_`."""
        return self.prettify(repr_=repr_doc, **kwargs)

    def pprint(self, **kwargs) -> None:
        """Pretty-print the object.

        Args:
            **kwargs: Additional keyword arguments passed to `prettify`.
        """
        print(self.prettify(**kwargs))

    def __str__(self) -> str:
        try:
            return self.prettify()
        except NotImplementedError:
            return repr(self)


def prettify_inited(
    cls: type,
    kwargs: tp.Any,
    replace: tp.DictLike = None,
    path: str = None,
    htchar: str = "    ",
    lfchar: str = "\n",
    indent: int = 0,
    repr_: tp.Optional[tp.Callable] = None,
) -> tp.Any:
    """Prettify an instance initialized with keyword arguments.

    Args:
        cls (type): The class of the instance.
        kwargs (Any): A dictionary of keyword arguments used for initialization.
        replace (DictLike): A mapping for value replacement.
        path (str): The current path in the object hierarchy.
        htchar (str): The string used for horizontal indentation.
        lfchar (str): The line feed character.
        indent (int): The current indentation level.
        repr_ (Optional[Callable]): A function to get the representation of an object.

            Defaults to `repr`.

    Returns:
        Any: A prettified string representation of the initialized instance.
    """
    if repr_ is None:
        repr_ = repr
    items = []
    for k, v in kwargs.items():
        if replace is None:
            replace = {}
        if path is None:
            new_path = k
        else:
            new_path = str(path) + "." + str(k)
        if new_path in replace:
            new_v = replace[new_path]
        else:
            new_v = prettify(
                v,
                replace=replace,
                path=new_path,
                htchar=htchar,
                lfchar=lfchar,
                indent=indent + 1,
                repr_=repr_,
            )
        k_repr = repr(k)
        if isinstance(k, str):
            k_repr = k_repr[1:-1]
        items.append(lfchar + htchar * (indent + 1) + k_repr + "=" + new_v)
    if len(items) == 0:
        return "%s()" % (cls.__name__,)
    return "%s(%s)" % (cls.__name__, ",".join(items) + lfchar + htchar * indent)


def prettify_dict(
    obj: tp.Any,
    replace: tp.DictLike = None,
    path: str = None,
    htchar: str = "    ",
    lfchar: str = "\n",
    indent: int = 0,
    repr_: tp.Optional[tp.Callable] = None,
) -> tp.Any:
    """Prettify a dictionary.

    Args:
        obj (Any): The dictionary to prettify.
        replace (DictLike): A mapping for value replacement.
        path (str): The current path in the object hierarchy.
        htchar (str): The string used for horizontal indentation.
        lfchar (str): The line feed character.
        indent (int): The current indentation level.
        repr_ (Optional[Callable]): A function to get the representation of an object.

            Defaults to `repr`.

    Returns:
        Any: A prettified string representation of the dictionary.
    """
    if repr_ is None:
        repr_ = repr
    if all([isinstance(k, str) and k.isidentifier() for k in obj]):
        return prettify_inited(
            type(obj),
            obj,
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
            repr_=repr_,
        )
    items = []
    for k, v in obj.items():
        if replace is None:
            replace = {}
        if path is None:
            new_path = k
        else:
            new_path = str(path) + "." + str(k)
        if new_path in replace:
            new_v = replace[new_path]
        else:
            new_v = prettify(
                v,
                replace=replace,
                path=new_path,
                htchar=htchar,
                lfchar=lfchar,
                indent=indent + 1,
                repr_=repr_,
            )
        items.append(lfchar + htchar * (indent + 1) + repr(k) + ": " + new_v)
    if type(obj) is dict:
        if len(items) == 0:
            return "{}"
        return "{%s}" % (",".join(items) + lfchar + htchar * indent)
    if len(items) == 0:
        return "%s({})" % (type(obj).__name__,)
    return "%s({%s})" % (type(obj).__name__, ",".join(items) + lfchar + htchar * indent)


def prettify(
    obj: tp.Any,
    replace: tp.DictLike = None,
    path: str = None,
    htchar: str = "    ",
    lfchar: str = "\n",
    indent: int = 0,
    repr_: tp.Optional[tp.Callable] = None,
) -> tp.Any:
    """Prettify an object.

    Unfolds regular Python data structures such as lists, tuples, and dictionaries.

    If `obj` is an instance of `Prettified`, calls its `prettify` method.

    Args:
        obj (Any): The object to prettify.
        replace (DictLike): A mapping for value replacement.
        path (str): The current path in the object hierarchy.
        htchar (str): The string used for horizontal indentation.
        lfchar (str): The line feed character.
        indent (int): The current indentation level.
        repr_ (Optional[Callable]): A function to get the representation of an object.

            Defaults to `repr`.

    Returns:
        Any: A prettified string representation of the object.
    """
    if repr_ is None:
        repr_ = repr
    if isinstance(obj, Prettified):
        return obj.prettify(
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
            repr_=repr_,
        )
    if attr.has(type(obj)):
        return prettify_inited(
            type(obj),
            attr.asdict(obj),
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
            repr_=repr_,
        )
    if isinstance(obj, dict):
        return prettify_dict(
            obj,
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
            repr_=repr_,
        )
    if isinstance(obj, tuple) and hasattr(obj, "_asdict"):
        return prettify_inited(
            type(obj),
            obj._asdict(),
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
            repr_=repr_,
        )
    if isinstance(obj, (tuple, list, set, frozenset)):
        items = []
        for v in obj:
            new_v = prettify(
                v,
                replace=replace,
                path=path,
                htchar=htchar,
                lfchar=lfchar,
                indent=indent + 1,
                repr_=repr_,
            )
            items.append(lfchar + htchar * (indent + 1) + new_v)
        if type(obj) is tuple:
            if len(items) == 0:
                return "()"
            return "(%s)" % (",".join(items) + lfchar + htchar * indent)
        if type(obj) is list:
            if len(items) == 0:
                return "[]"
            return "[%s]" % (",".join(items) + lfchar + htchar * indent)
        if type(obj) is set:
            if len(items) == 0:
                return "set()"
            return "{%s}" % (",".join(items) + lfchar + htchar * indent)
        if len(items) == 0:
            return "%s([])" % (type(obj).__name__,)
        return "%s([%s])" % (type(obj).__name__, ",".join(items) + lfchar + htchar * indent)
    if isinstance(obj, np.dtype) and hasattr(obj, "fields"):
        items = []
        for k, v in dict(obj.fields).items():
            items.append(lfchar + htchar * (indent + 1) + repr_((k, str(v[0]))))
        return "np.dtype([%s])" % (",".join(items) + lfchar + htchar * indent)
    if hasattr(obj, "shape") and isinstance(obj.shape, tuple) and len(obj.shape) > 0:
        module = type(obj).__module__
        qualname = type(obj).__qualname__
        return "<%s.%s object at %s with shape %s>" % (module, qualname, str(hex(id(obj))), obj.shape)
    if isinstance(obj, float):
        if np.isnan(obj):
            return "np.nan"
        if np.isposinf(obj):
            return "np.inf"
        if np.isneginf(obj):
            return "-np.inf"
    return repr_(obj)


def repr_doc(obj: tp.Any) -> str:
    """Representation function suited for documentation.

    Args:
        obj (Any): Object.

    Returns:
        str: Representation.
    """
    import re

    obj_repr = repr(obj)
    if obj_repr.startswith("environ({") and obj_repr.endswith("})"):
        return "os.environ"
    obj_repr = re.sub(r"\s+from\s+'[^']+'", "", obj_repr)
    obj_repr = re.sub(r"\s+at\s+0x[0-9a-fA-F]+", "", obj_repr)
    return obj_repr


def prettify_doc(*args, **kwargs):
    """Prettify for documentation, equivalent to using `prettify` with `repr_doc` as `repr_`."""
    return prettify(*args, repr_=repr_doc, **kwargs)


def pprint(*args, **kwargs) -> None:
    """Print the prettified representation of the given arguments.

    Args:
        *args: Additional positional arguments passed to `prettify`.
        **kwargs: Additional keyword arguments passed to `prettify`.
    """
    print(prettify(*args, **kwargs))


def format_array(array: tp.ArrayLike, tabulate: tp.Optional[bool] = None, html: bool = False, **kwargs) -> str:
    """Format an array for display.

    Args:
        array (ArrayLike): An array-like object to be formatted.
        tabulate (Optional[bool]): If True, use `tabulate.tabulate` for formatting;
            if False, use pandas formatting functions (`DataFrame.to_string` or `DataFrame.to_html`).

            If None, auto-detect based on the availability of the `tabulate` library and the `html` parameter.
        html (bool): Format the output in HTML if True.
        **kwargs: Additional keyword arguments for the formatting function.

    Returns:
        str: The formatted array as a string.
    """
    from vectorbtpro.base.reshaping import to_pd_array

    pd_array = to_pd_array(array)
    if tabulate is None:
        from vectorbtpro.utils.module_ import check_installed

        tabulate = check_installed("tabulate") and not html
    if tabulate:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("tabulate")
        from tabulate import tabulate

        if isinstance(pd_array, pd.Series):
            pd_array = pd_array.to_frame()
        if html:
            return tabulate(pd_array, headers="keys", tablefmt="html", **kwargs)
        return tabulate(pd_array, headers="keys", **kwargs)
    if html:
        if isinstance(pd_array, pd.Series):
            pd_array = pd_array.to_frame()
        return pd_array.to_html(**kwargs)
    return pd_array.to_string(**kwargs)


def ptable(*args, display_html: tp.Optional[bool] = None, **kwargs) -> None:
    """Print the formatted array.

    Args:
        *args: Additional arguments for `format_array`.
        display_html (Optional[bool]): Display output in HTML if True.

            If None, auto-detect if running in an IPython notebook.
        **kwargs: Additional keyword arguments for `format_array`.
    """
    from vectorbtpro.utils.checks import in_notebook

    if display_html is None:
        display_html = in_notebook()
    if display_html:
        from IPython.display import display, HTML

        display(HTML(format_array(*args, html=True, **kwargs)))
    else:
        print(format_array(*args, **kwargs))


def format_parameter(param: inspect.Parameter, annotate: bool = False) -> str:
    """Format a function parameter into a string representation.

    Args:
        param (inspect.Parameter): The parameter to format.
        annotate (bool): Include type annotation in the formatted string if True.

    Returns:
        str: The formatted parameter.
    """
    kind = param.kind
    formatted = param.name

    if annotate and param.annotation is not param.empty:
        formatted = "{}: {}".format(formatted, inspect.formatannotation(param.annotation))

    if param.default is not param.empty:
        if annotate and param.annotation is not param.empty:
            formatted = "{} = {}".format(formatted, repr(param.default))
        else:
            formatted = "{}={}".format(formatted, repr(param.default))

    if kind == param.VAR_POSITIONAL:
        formatted = "*" + formatted
    elif kind == param.VAR_KEYWORD:
        formatted = "**" + formatted

    return formatted


def format_signature(
    signature: inspect.signature,
    annotate: bool = False,
    start: str = "\n    ",
    separator: str = ",\n    ",
    end: str = "\n",
) -> str:
    """Format a function signature.

    Args:
        signature (Signature): The function signature to format.
        annotate (bool): Include type annotations if True.
        start (str): String inserted at the beginning of the parameter list.
        separator (str): String used to separate parameters.
        end (str): String appended after the parameter list.

    Returns:
        str: The formatted signature.
    """
    result = []
    render_pos_only_separator = False
    render_kw_only_separator = True

    for param in signature.parameters.values():
        formatted = format_parameter(param, annotate=annotate)

        kind = param.kind

        if kind == param.POSITIONAL_ONLY:
            render_pos_only_separator = True
        elif render_pos_only_separator:
            result.append("/")
            render_pos_only_separator = False

        if kind == param.VAR_POSITIONAL:
            render_kw_only_separator = False
        elif kind == param.KEYWORD_ONLY and render_kw_only_separator:
            result.append("*")
            render_kw_only_separator = False

        result.append(formatted)

    if render_pos_only_separator:
        result.append("/")

    if len(result) == 0:
        rendered = "()"
    else:
        rendered = "({})".format(start + separator.join(result) + end)

    if annotate and signature.return_annotation is not inspect._empty:
        anno = inspect.formatannotation(signature.return_annotation)
        rendered += " -> {}".format(anno)

    return rendered


def format_func(func: tp.Callable, incl_doc: bool = True, **kwargs) -> str:
    """Format a function or class constructor.

    Args:
        func (Callable): The function or class to format. If a class, its `__init__` method is used.
        incl_doc (bool): If True, include the function's docstring in the output if available.
        **kwargs: Additional keyword arguments for `format_signature`.

    Returns:
        str: The formatted function description, including its signature and docstring if available.
    """
    from vectorbtpro.utils.checks import is_attrs_subclass
    from vectorbtpro.utils.attr_ import DefineMixin

    doc = func.__doc__
    if is_attrs_subclass(func):
        if issubclass(func, DefineMixin):
            if func.__init__ is DefineMixin.__init__:
                func_name = func.__name__ + ".__attrs_init__"
                func = func.__attrs_init__
            else:
                func_name = func.__name__ + ".__init__"
                func = func.__init__
        else:
            if hasattr(func, "__attrs_init__"):
                func_name = func.__name__ + ".__attrs_init__"
                func = func.__attrs_init__
            else:
                func_name = func.__name__ + ".__init__"
                func = func.__init__
    elif inspect.isclass(func):
        func_name = func.__name__ + ".__init__"
        func = func.__init__
    elif inspect.ismethod(func) and hasattr(func, "__self__"):
        if isinstance(func.__self__, type):
            func_name = func.__self__.__name__ + "." + func.__name__
        else:
            func_name = type(func.__self__).__name__ + "." + func.__name__
    else:
        func_name = func.__qualname__
    if doc is None or (func.__doc__ is not None and not func.__doc__.startswith("Method generated by attrs")):
        doc = func.__doc__
    if incl_doc and doc is not None:
        return "{}{}:\n{}".format(
            func_name,
            format_signature(inspect.signature(func), **kwargs),
            "    " + "\n    ".join(inspect.cleandoc(doc).splitlines()),
        )
    return "{}{}".format(
        func_name,
        format_signature(inspect.signature(func), **kwargs),
    )


def phelp(*args, **kwargs) -> None:
    """Print the formatted representation of a function.

    Args:
        *args: Additional arguments for `format_func`.
        **kwargs: Additional keyword arguments for `format_func`.
    """
    print(format_func(*args, **kwargs))


def pdir(*args, **kwargs) -> None:
    """Print parsed attributes of an object.

    Args:
        *args: Additional arguments for `vectorbtpro.utils.attr_.parse_attrs`.
        **kwargs: Additional keyword arguments for `vectorbtpro.utils.attr_.parse_attrs`.
    """
    from vectorbtpro.utils.attr_ import parse_attrs

    ptable(parse_attrs(*args, **kwargs))


def dump(obj: tp.Any, dump_engine: str = "prettify", **kwargs) -> str:
    """Dump an object to a string using the specified dump engine.

    Args:
        obj (Any): The object to dump.
        dump_engine (str): The dump engine to use.

            Options include:

            * "repr"
            * "prettify"
            * "nestedtext"
            * "yaml"
            * "pyyaml
            * "ruamel" or "ruamel.yaml"
            * "toml"
            * "json"
        **kwargs: Additional keyword arguments for the dump engine.

    Returns:
        str: The dumped object as a string.
    """
    if isinstance(obj, str):
        return obj
    if dump_engine.lower() == "repr":
        return repr(obj)
    if dump_engine.lower() == "prettify":
        return prettify(obj, **kwargs)
    if dump_engine.lower() == "nestedtext":
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("nestedtext")
        import nestedtext as nt

        return nt.dumps(obj, **kwargs)
    if dump_engine.lower() == "yaml":
        from vectorbtpro.utils.module_ import check_installed

        if check_installed("ruamel"):
            dump_engine = "ruamel"
        else:
            dump_engine = "pyyaml"
    if dump_engine.lower() == "pyyaml":
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("yaml")
        import yaml

        def multiline_str_representer(dumper, data):
            if isinstance(data, str) and "\n" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_str(data)

        class CustomDumper(yaml.SafeDumper):
            pass

        CustomDumper.add_representer(str, multiline_str_representer)

        if "Dumper" not in kwargs:
            kwargs["Dumper"] = CustomDumper
        return yaml.dump(obj, **kwargs)
    if dump_engine.lower() in ("ruamel", "ruamel.yaml"):
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ruamel")
        from ruamel.yaml import YAML
        from ruamel.yaml.representer import RoundTripRepresenter

        def multiline_str_representer(dumper, data):
            if isinstance(data, str) and "\n" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_str(data)

        class CustomRepresenter(RoundTripRepresenter):
            pass

        CustomRepresenter.add_representer(str, multiline_str_representer)

        yaml = YAML(
            typ=kwargs.pop("typ", None),
            pure=kwargs.pop("pure", False),
            plug_ins=kwargs.pop("plug_ins", None),
        )
        if "Representer" not in kwargs:
            yaml.Representer = CustomRepresenter
        for k, v in kwargs.items():
            if not hasattr(yaml, k):
                raise AttributeError(f"Invalid YAML attribute: '{k}'")
            if isinstance(v, tuple):
                getattr(yaml, k)(*v)
            elif isinstance(v, dict):
                getattr(yaml, k)(**v)
            else:
                setattr(yaml, k, v)
        transform = kwargs.pop("transform", None)
        output = io.StringIO()
        yaml.dump(obj, output, transform=transform)
        return output.getvalue()
    if dump_engine.lower() == "toml":
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("toml")
        import toml

        return toml.dumps(obj, **kwargs)
    if dump_engine.lower() == "json":
        import json

        return json.dumps(obj, **kwargs)
    raise ValueError(f"Invalid dump engine: '{dump_engine}'")


def get_dump_language(dump_engine: str) -> str:
    """Return the language corresponding to the provided dump engine.

    Args:
        dump_engine (str): The name of the dump engine.

    Returns:
        str: The corresponding language name, or an empty string if the dump engine is unknown.
    """
    if dump_engine.lower() == "repr":
        return "python"
    if dump_engine.lower() == "prettify":
        return "python"
    if dump_engine.lower() == "nestedtext":
        return "text"
    if dump_engine.lower() == "yaml":
        return "yaml"
    if dump_engine.lower() == "pyyaml":
        return "yaml"
    if dump_engine.lower() in ("ruamel", "ruamel.yaml"):
        return "yaml"
    if dump_engine.lower() == "toml":
        return "toml"
    if dump_engine.lower() == "json":
        return "json"
    return ""


REFINE_SRC_PROMPT = """You are a code-refinement assistant. 
        
Present the refactored version of the given chunk of Python code to address 
any detected code smells or issues. You must:

1. **Return the entire code block** in your output.
2. **Do not enclose your output in triple backticks**, and return no other text or explanation."""
"""Default prompt for `refine_source`."""


def refine_source(
    source: tp.Any,
    *,
    source_name: tp.Optional[str] = None,
    as_package: bool = True,
    start_line: tp.Optional[int] = None,
    end_line: tp.Optional[int] = None,
    prompt: tp.Optional[str] = None,
    chunk_size: tp.Optional[int] = 2000,
    split_nodes: bool = True,
    split_text_kwargs: tp.KwargsLike = None,
    tokenize_kwargs: tp.KwargsLike = None,
    complete_kwargs: tp.KwargsLike = None,
    show_progress: tp.Optional[bool] = True,
    pbar_kwargs: tp.KwargsLike = None,
    mult_show_progress: tp.Optional[bool] = None,
    mult_pbar_kwargs: tp.KwargsLike = None,
    modify: bool = False,
    copy_to_clipboard: bool = False,
    show_diff: bool = True,
    open_browser: bool = True,
) -> tp.Union[tp.RefineSourceOutput, tp.RefineSourceOutputs]:
    """Refine source using chunking and completions.

    Args:
        source (Any): Source(s) or object(s) to extract the source from.

            A source can be:

            * a string (such as "import vectorbtpro as vbt ..."),
            * a path to a source file (such as "strategies/sma_crossover.py"),
            * any Python object (such as `pipeline_nb`), or
            * an iterable of such.

            If a path to a directory or (sub-)package is passed, it will be considered as multiple sources.
        source_name (Optional[str]): Name of the source to be displayed in the progress bar.
        as_package (bool): Whether to treat a (sub-)package as multiple sources.
        start_line (Optional[int]): Inclusive start line in the source.

            !!! note
                Counting starts with 1.
        end_line (Optional[int]): Inclusive end line in the source.

            !!! note
                Counting starts with 1.
        prompt (Optional[str]): System prompt.
        chunk_size (Optional[int]): The maximum number of tokens in each chunk.

            If None, feeds the entire source.
        split_nodes (bool): Whether to split nodes that exceed the maximum chunk size.
        split_text_kwargs (KwargsLike): Keyword arguments passed to
            `vectorbtpro.utils.knowledge.chatting.split_text`.
        tokenize_kwargs (KwargsLike): Keyword arguments passed to
            `vectorbtpro.utils.knowledge.chatting.tokenize`.
        complete_kwargs (KwargsLike): Keyword arguments passed to
            `vectorbtpro.utils.knowledge.chatting.complete_content`.
        show_progress (Optional[bool]): Whether to show the progress bar iterating over chunks.
        pbar_kwargs (KwargsLike): Keyword arguments passed to `vectorbtpro.utils.pbar.ProgressBar`.
        mult_show_progress (Optional[bool]): The same as `show_progress` but for iterating over multiple sources.

            If None, defaults to `show_progress`.
        mult_pbar_kwargs (KwargsLike): The same as `pbar_kwargs` but for iterating over multiple sources.

            Gets merged over `pbar_kwargs`.
        modify (bool): Whether to modify the source file.
        copy_to_clipboard (bool): Whether to copy the refined source to clipboard.

            Doesn't work for multiple sources.
        show_diff (bool): Whether to show the delta HTML file using `difflib`.

            Doesn't work for multiple sources.
        open_browser (bool): Whether to open the browser.

            Doesn't work for multiple sources.

    Returns:
        Union[RefineSourceOutput, RefineSourceOutputs]:

            * The refined source if `modify` and `copy_to_clipboard` are False.
            * The path to the source file if `modify` is True.
            * The path to the delta HTML file if `show_diff` is True.
            * Zipped list of sources and outputs if there are multiple sources.
    """
    from vectorbtpro.utils.checks import is_numba_func, is_complex_iterable
    from vectorbtpro.utils.config import merge_dicts
    from vectorbtpro.utils.module_ import assert_can_import
    from vectorbtpro.utils.path_ import get_common_prefix
    from vectorbtpro.utils.pbar import ProgressBar
    from vectorbtpro.utils.knowledge.chatting import split_text, tokenize, complete_content

    pbar_kwargs = merge_dicts(
        dict(desc_kwargs=dict(refresh=True)),
        pbar_kwargs,
    )

    if isinstance(source, str):
        try:
            if Path(source).exists():
                source = Path(source)
        except Exception as e:
            pass
    if isinstance(source, ModuleType) and hasattr(source, "__path__") and as_package:
        source = Path(getattr(source, "__path__"))
    if isinstance(source, Path) and source.is_dir():
        source = list(source.rglob("*.py"))
    if is_complex_iterable(source):
        sources = source
        source_names = []
        paths = []
        all_paths = True
        for i, source in enumerate(sources):
            if isinstance(source, str):
                try:
                    if Path(source).exists():
                        source = Path(source)
                except Exception as e:
                    pass
            if isinstance(source, str):
                source_name = f"<string>"
                all_paths = False
            elif isinstance(source, Path):
                source_name = source.name
                paths.append(source.resolve())
            else:
                if is_numba_func(source):
                    source = source.py_func
                source_path = inspect.getsourcefile(source)
                if source_path:
                    source_path = Path(source_path)
                if not source_path or not source_path.is_file():
                    raise ValueError(f"Cannot determine a valid source file for object {source}")
                source_name = source_path.name
                paths.append(source.resolve())
            source_names.append(source_name)
        if all_paths:
            common_path = Path(get_common_prefix(paths)).resolve()
            source_names = [str(path.relative_to(common_path)) for path in paths]

        outputs = []
        if mult_show_progress is None:
            mult_show_progress = show_progress
        mult_pbar_kwargs = merge_dicts(pbar_kwargs, mult_pbar_kwargs)
        with ProgressBar(total=len(source_names), show_progress=mult_show_progress, **mult_pbar_kwargs) as pbar:
            for i, source in enumerate(sources):
                pbar.set_description(dict(source=source_names[i]))
                output = refine_source(
                    source=source,
                    source_name=source_names[i],
                    as_package=False,
                    start_line=start_line,
                    end_line=end_line,
                    prompt=prompt,
                    chunk_size=chunk_size,
                    split_nodes=split_nodes,
                    split_text_kwargs=split_text_kwargs,
                    tokenize_kwargs=tokenize_kwargs,
                    complete_kwargs=complete_kwargs,
                    show_progress=show_progress,
                    pbar_kwargs=pbar_kwargs,
                    modify=modify,
                    copy_to_clipboard=False,
                    show_diff=False,
                    open_browser=False,
                )
                outputs.append(output)
                pbar.update()
        return list(zip(sources, outputs))

    if isinstance(source, str):
        source_path = None
        source_lines = source.splitlines(keepends=True)
        source_start_line = 1
        if source_name is None:
            source_name = "<string>"
    elif isinstance(source, Path):
        source_path = source
        with source_path.open("r", encoding="utf-8") as f:
            source_lines = f.readlines()
        source_start_line = 1
        if source_name is None:
            source_name = source_path.name
    else:
        if is_numba_func(source):
            source = source.py_func
        source_path = inspect.getsourcefile(source)
        if source_path:
            source_path = Path(source_path)
        if not source_path or not source_path.is_file():
            raise ValueError(f"Cannot determine a valid source file for object {source}")
        source_lines, source_start_line = inspect.getsourcelines(source)
        if source_name is None:
            source_name = source_path.name

    if start_line is None:
        start_line = source_start_line
    else:
        start_line = source_start_line + start_line - 1
    if end_line is None:
        end_line = source_start_line + len(source_lines) - 1
    else:
        end_line = source_start_line + end_line - 1
    start_index = start_line - 1
    end_index = end_line
    source_lines = source_lines[start_index:end_index]
    source = "".join(source_lines)
    source_name = f"{source_name}#L{start_line}-L{end_line}"

    if prompt is None:
        prompt = REFINE_SRC_PROMPT
    split_text_kwargs = merge_dicts(
        dict(
            text_splitter="segment",
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=[[r"(?=\n\s*(?:@[^\n]+\n\s*)*def\s)"], [r"\n\s*\n"], None],
            chunk_template="$chunk_text",
        ),
        split_text_kwargs,
    )
    tokenize_kwargs = merge_dicts(
        dict(),
        tokenize_kwargs,
    )
    complete_kwargs = merge_dicts(
        dict(
            model="o3-mini",
            reasoning_effort="high",
            system_as_user=True,
            system_prompt=prompt,
        ),
        complete_kwargs,
    )
    if copy_to_clipboard:
        assert_can_import("pyperclip")

    if chunk_size is not None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None
        spans = []
        if tree is not None and hasattr(tree, "body"):
            for node in tree.body:
                if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                    spans.append((node.lineno - 1, node.end_lineno))
        spans.sort(key=lambda x: x[0])

        blocks = []
        prev = 0
        for start, end in spans:
            if prev < start:
                blocks.append("".join(source_lines[prev:start]))
            blocks.append("".join(source_lines[start:end]))
            prev = end
        if prev < len(source_lines):
            blocks.append("".join(source_lines[prev:]))

        if split_nodes:
            final_chunks = []
            buf = []
            buf_tokens = 0
            for block in blocks:
                sub_blocks = split_text(block, **split_text_kwargs)
                for sb in sub_blocks:
                    sb_tokens = len(tokenize(sb, **tokenize_kwargs))
                    if buf_tokens + sb_tokens > chunk_size:
                        if buf:
                            final_chunks.append("".join(buf))
                        buf = [sb]
                        buf_tokens = sb_tokens
                    else:
                        buf.append(sb)
                        buf_tokens += sb_tokens
            if buf:
                final_chunks.append("".join(buf))
        else:
            final_chunks = blocks
    else:
        final_chunks = [source]

    processed = []
    chunk_start_line = start_line
    with ProgressBar(total=len(final_chunks), show_progress=show_progress, **pbar_kwargs) as pbar:
        for i in range(len(final_chunks)):
            chunk = final_chunks[i]
            chunk_lines = len(chunk.splitlines(keepends=True))
            pbar.set_description(dict(lines="{}..{}".format(chunk_start_line, chunk_start_line + chunk_lines - 1)))
            leading_len = len(chunk) - len(chunk.lstrip())
            leading = chunk[:leading_len]
            trailing_len = len(chunk) - len(chunk.rstrip())
            trailing = chunk[-trailing_len:] if trailing_len > 0 else ""
            middle = chunk[leading_len : len(chunk) - trailing_len]
            processed_middle = complete_content(leading + middle + trailing, **complete_kwargs).strip()
            processed.append(leading + processed_middle + trailing)
            chunk_start_line += chunk_lines
            pbar.update()
    new_source = "".join(processed)

    if modify and source_path:
        with source_path.open("r", encoding="utf-8") as f:
            file_contents = f.readlines()
        new_source_lines = new_source.splitlines(keepends=True)
        if not new_source_lines or not new_source_lines[-1].endswith("\n"):
            new_source_lines.append("\n")
        file_contents[start_index:end_index] = new_source_lines
        with source_path.open("w", encoding="utf-8") as f:
            f.writelines(file_contents)

    if copy_to_clipboard:
        import pyperclip

        pyperclip.copy(new_source)

    if show_diff:
        differ = difflib.HtmlDiff()
        html_diff = differ.make_file(
            source.splitlines(),
            new_source.splitlines(),
            fromdesc="Original",
            todesc="Modified",
        )
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            prefix=re.sub(r"[\W_]+", "_", source_name).strip("_"),
            suffix=".html",
        ) as f:
            f.write(html_diff)
            file_path = Path(f.name)
        if open_browser:
            webbrowser.open("file://" + str(file_path.resolve()))
        if modify and source_path:
            return source_path, file_path
        if copy_to_clipboard:
            return file_path
        return new_source, file_path

    if modify and source_path:
        return source_path
    if copy_to_clipboard:
        return None
    return new_source



REFINE_DOCSTR_PROMPT = """You are a code-refinement assistant. 

Your goal is to refine (rewrite for clarity, correctness, consistent format and wording) 
**only** the docstrings of the given chunk of Python code. You must:

1. **Modify only existing docstrings**.
2. **Return the entire code block** in your output.
3. **Do not enclose your output in triple backticks**, and return no other text or explanation.
4. **Retain all non-docstring parts of the code** exactly as they are.

### 1. Scope of Edits

- **Identify existing docstrings** in functions, classes, and methods.
- **Edit only those docstrings**; do not create new ones.
- **Do not document** functions or methods whose names begin with one or two underscores 
    (e.g., `_preprocess`, `__eq__`).
- **Keep the `__init__` docstring empty**. Instead, document its parameters in the class 
    docstring inside the "Args" section.
- **Keep the license text**.

### 2. Content Requirements

- **Use relevant details from the code** to make each docstring clear and self-explanatory 
    for someone who cannot see the source code.
- **Keep docstrings concise and correct** in grammar and content, but do not remove any 
    valuable information unless it's duplicate.
- **Omit sections** such as "Raises," "Attributes," "Methods," or default values.
- **Do not extra mention the default value of an argument**.
- **Do not extra mention that an argument is optional**.
    - For example, `x: tp.Optional[int] = None` becomes `x (Optional[int]): ...` and 
        not `x (int, optional): ...`.
    - Do not omit the `Optional[...]` type hint.
- **Retain any admonitions** like `"!!! note"` or `"!!! warning"` exactly as they are, for example:
    ```
    !!! note
      Here comes the note.
    ```
- Instead of adding the section "Note", use the admonition "!!! note".
- **Preserve indentation, whitespace, and formatting** in lists or multi-line text 
    unless it is incorrect.
- **Do not change code examples** in the "Usage" section. Use only the name "Usage", 
    not "Examples" or any other name.
- If a function returns `None` or bool, **do not add a "Returns" section**.
- **Do not add your own "Usage" section**.
- **Make sure to list all arguments, their types, and descriptions**, apart from `self`, `cls`, and `cls_or_self`

### 3. Style and Format

- **Use Markdown format**.
- **Follow PEP 257 guidelines**.
- **Use Google-style docstrings** for arguments and return values, for example:
    ```
    Args:
        arg_name (type): Description of the argument.
    
    Returns:
        return_type: Description of the returned value.
    ```
- If the description of an argument has multiple sentences, **separate them by an empty line**. 
    For example:
    ```
    x (Optional[int]): First integer.
      
        Refer to `prepare_x` for further details.
    y (Optional[int]): Second integer.
  
        If not provided, uses `x`.
  
        Refer to `prepare_y` for further details.
    ```
    instead of
    ```
    x (Optional[int]): First integer. Refer to `prepare_x` for further details.
    y (Optional[int]): Second integer. If not provided, uses `x`. Refer to `prepare_y` for further details.
    ```
- **Preserve type hints** but remove module prefixes such as "tp." and the suffix "T".
    - For example, `x: tp.Union[None, int, tp.DatetimeLike] = ...` becomes 
        `x (Union[None, int, DatetimeLike]): ...`.
    - Also, `x: tp.MaybeType[KnowledgeAssetT] = ...` becomes `x (MaybeType[KnowledgeAsset]): ...`.
- Treat classes decorated with `@define` **as if they were decorated with `@attr.s`**, adjusting 
    docstrings accordingly.
- For module docstrings, retain the phrasing that **identifies them as a module**, 
    such as "Module providing X".
- For class docstrings, retain the phrasing that **identifies them as a class**, 
    such as "Class for X".
- **Begin method docstrings with imperative verbs** (e.g., "Return," "Fetch," "Create") 
    rather than "Does X...".
- **Properties should describe the object rather than the action**, for example "Context." 
    instead of "Return the context."
- Bullet points must be at the **same indentation level as the parent sentence** and should 
    have **one empty line before the list**.
- **Use a consistent style for bullet points**, such as "*"
- For `*args` begin the description with `Additional arguments passed to/for`.
- For `**kwargs` begin the description with `Additional keyword arguments passed to/for`.
- When dealing with named tuples and enums, replace "Attributes:" by "Fields:" in their docstrings
- Make sure that docstrings have three double quotes (\""") at the start and the end
- Make sure that **docstrings are properly indented** relative to the first three double quotes (\""")

### 4. Referencing and Usage

- ***Refer to Python objects** (classes, functions, methods, or any code reference) **with backticks** and, 
    if relevant, fully qualified names.
    - For example, `Data.get` (if in the same module) or `vectorbtpro.data.base.Data.get`.
    - Do not shorten fully qualified names, such as from `vectorbtpro.utils.datetime_.get_rangebreaks` 
        to `dt.get_rangebreaks`.
- If the reference is known to belong to the same class (for instance, when in the code the object 
    is being accessed from `cls`, `self`, or `cls_or_self`), prepend the class to the reference if known.
    - For example, `FigureMixin.show` instead of `show`.
- **Always use backticks around code references**, e.g., "Keyword arguments for `make_subplots`" 
    instead of "Keyword arguments for make_subplots".
- **If a "Usage" section exists**, place it **at the end** of the docstring.

### 5. Special Arguments

- **Document the function(s)** that will receive `*args` or `**kwargs`, if known.

### 6. Miscellaneous

- **Do not start any docstring with a blank line**.
- **Do not end a docstring with a blank line** unless it contains multiple lines.
- **Only fix indentation or whitespace if it is incorrect**.
"""
"""Prompt for `refine_docstrings`."""


def refine_docstrings(source: tp.Any, **kwargs) -> tp.RefineSourceOutput:
    """Call `refine_source` with `prompt=REFINE_DOCSTR_PROMPT`."""
    return refine_source(source, prompt=REFINE_DOCSTR_PROMPT, **kwargs)
