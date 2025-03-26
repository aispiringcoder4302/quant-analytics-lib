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

    Returns:
        Any: A prettified string representation of the initialized instance.
    """
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
            new_v = prettify(v, replace=replace, path=new_path, htchar=htchar, lfchar=lfchar, indent=indent + 1)
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
) -> tp.Any:
    """Prettify a dictionary.

    Args:
        obj (Any): The dictionary to prettify.
        replace (DictLike): A mapping for value replacement.
        path (str): The current path in the object hierarchy.
        htchar (str): The string used for horizontal indentation.
        lfchar (str): The line feed character.
        indent (int): The current indentation level.

    Returns:
        Any: A prettified string representation of the dictionary.
    """
    if all([isinstance(k, str) and k.isidentifier() for k in obj]):
        return prettify_inited(
            type(obj),
            obj,
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
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
            new_v = prettify(v, replace=replace, path=new_path, htchar=htchar, lfchar=lfchar, indent=indent + 1)
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

    Returns:
        Any: A prettified string representation of the object.
    """
    if isinstance(obj, Prettified):
        return obj.prettify(replace=replace, path=path, htchar=htchar, lfchar=lfchar, indent=indent)
    if attr.has(type(obj)):
        return prettify_inited(
            type(obj),
            attr.asdict(obj),
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
        )
    if isinstance(obj, dict):
        return prettify_dict(obj, replace=replace, path=path, htchar=htchar, lfchar=lfchar, indent=indent)
    if isinstance(obj, tuple) and hasattr(obj, "_asdict"):
        return prettify_inited(
            type(obj),
            obj._asdict(),
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
        )
    if isinstance(obj, (tuple, list, set, frozenset)):
        items = []
        for v in obj:
            new_v = prettify(v, replace=replace, path=path, htchar=htchar, lfchar=lfchar, indent=indent + 1)
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
            items.append(lfchar + htchar * (indent + 1) + repr((k, str(v[0]))))
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
    return repr(obj)


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
    if inspect.isclass(func):
        func_name = func.__name__ + ".__init__"
        func = func.__init__
    elif inspect.ismethod(func) and hasattr(func, "__self__"):
        if isinstance(func.__self__, type):
            func_name = func.__self__.__name__ + "." + func.__name__
        else:
            func_name = type(func.__self__).__name__ + "." + func.__name__
    else:
        func_name = func.__qualname__
    if incl_doc and func.__doc__ is not None:
        return "{}{}:\n{}".format(
            func_name,
            format_signature(inspect.signature(func), **kwargs),
            "    " + "\n    ".join(inspect.cleandoc(func.__doc__).splitlines()),
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
        *args: Additional arguments for `parse_attrs`.
        **kwargs: Additional keyword arguments for `parse_attrs`.
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
