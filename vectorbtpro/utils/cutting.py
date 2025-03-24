# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for extracting annotated code sections from code and saving them to files."""

import importlib
import inspect
from pathlib import Path
from types import ModuleType, FunctionType

from vectorbtpro import _typing as tp
from vectorbtpro.utils.path_ import check_mkdir
from vectorbtpro.utils.template import CustomTemplate, RepEval

__all__ = [
    "cut_and_save_module",
    "cut_and_save_func",
]


def collect_blocks(lines: tp.Iterable[str]) -> tp.Dict[str, tp.List[str]]:
    """Collect block sections from lines.

    Args:
        lines (Iterable[str]): An iterable of strings representing lines of code.

    Returns:
        Dict[str, List[str]]: A dictionary mapping block names to lists of lines contained within each block.
    """
    blocks = {}
    block_name = None

    for line in lines:
        sline = line.strip()

        if sline.startswith("# % <block") and sline.endswith(">"):
            block_name = sline[len("# % <block") : -1].strip()
            if len(block_name) == 0:
                raise ValueError("Missing block name")
            blocks[block_name] = []
        elif sline.startswith("# % </block>"):
            block_name = None
        elif block_name is not None:
            blocks[block_name].append(line)

    return blocks


def cut_from_code(
    code: str,
    section_name: str,
    prepend_lines: tp.Optional[tp.Iterable[str]] = None,
    append_lines: tp.Optional[tp.Iterable[str]] = None,
    out_lines_callback: tp.Union[None, tp.Callable, CustomTemplate] = None,
    return_lines: bool = False,
    **kwargs,
) -> tp.Union[str, tp.List[str]]:
    """Extract an annotated code section from the source code.

    This function processes a code string to extract a specific section defined by markers.
    The section must begin with `# % <section section_name>` and end with `# % </section>`.

    Within the section, blocks can be defined using markers `# % <block block_name>` and `# % </block>`,
    and these blocks are collected for use within expressions.

    Lines placed between `# % <skip [expression]>` and `# % </skip>` are skipped, while lines between
    `# % <uncomment [expression]>` and `# % </uncomment>` are uncommented.

    Any line containing `# %` outside these blocks is treated as a Python expression.
    The evaluation result dictates the output:

    * None: The line is skipped.
    * str: A single line to insert.
    * Iterable[str]: Multiple lines to insert into the processing queue.

    Expressions are evaluated in strict mode, raising errors on failure, unless prefixed
    with `?` to evaluate softly.

    Args:
        code (str): The source code to process.
        section_name (str): The name of the section to extract.
        prepend_lines (Optional[Iterable[str]]): Lines to prepend to the output.
        append_lines (Optional[Iterable[str]]): Lines to append to the output.
        out_lines_callback (Union[None, Callable, CustomTemplate]): A callback or template
            to transform the output lines.
        return_lines (bool): If True, returns the output as a list of lines.
        **kwargs: Additional context variables for expression evaluation.

    Returns:
        Union[str, List[str]]: The extracted and processed code section as a cleaned string,
            or as a list of lines if 'return_lines' is True.
    """
    lines = code.split("\n")
    blocks = collect_blocks(lines)

    out_lines = []
    if prepend_lines is not None:
        out_lines.extend(list(prepend_lines))
    section_found = False
    uncomment = False
    skip = False
    i = 0

    while i < len(lines):
        line = lines[i]
        sline = line.strip()

        if sline.startswith("# % <section") and sline.endswith(">"):
            if section_found:
                raise ValueError("Missing </section>")
            found_name = sline[len("# % <section") : -1].strip()
            if len(found_name) == 0:
                raise ValueError("Missing section name")
            section_found = found_name == section_name
        elif section_found:
            context = {
                "lines": lines,
                "blocks": blocks,
                "section_name": section_name,
                "line": line,
                "out_lines": out_lines,
                **kwargs,
            }
            if sline.startswith("# % </section>"):
                if append_lines is not None:
                    out_lines.extend(list(append_lines))
                if out_lines_callback is not None:
                    if isinstance(out_lines_callback, CustomTemplate):
                        out_lines_callback = out_lines_callback.substitute(context=context, strict=True)
                    out_lines = out_lines_callback(out_lines)
                if return_lines:
                    return out_lines
                return inspect.cleandoc("\n".join(out_lines))
            if sline.startswith("# % <skip") and sline.endswith(">"):
                if skip:
                    raise ValueError("Missing </skip>")
                expression = sline[len("# % <skip") : -1].strip()
                if len(expression) == 0:
                    skip = True
                else:
                    if expression.startswith("?"):
                        expression = expression[1:]
                        strict = False
                    else:
                        strict = True
                    eval_skip = RepEval(expression).substitute(context=context, strict=strict)
                    if not isinstance(eval_skip, RepEval):
                        skip = eval_skip
            elif sline.startswith("# % </skip>"):
                skip = False
            elif not skip:
                if sline.startswith("# % <uncomment") and sline.endswith(">"):
                    if uncomment:
                        raise ValueError("Missing </uncomment>")
                    expression = sline[len("# % <uncomment") : -1].strip()
                    if len(expression) == 0:
                        uncomment = True
                    else:
                        if expression.startswith("?"):
                            expression = expression[1:]
                            strict = False
                        else:
                            strict = True
                        eval_uncomment = RepEval(expression).substitute(context=context, strict=strict)
                        if not isinstance(eval_uncomment, RepEval):
                            uncomment = eval_uncomment
                elif sline.startswith("# % </uncomment>"):
                    uncomment = False
                elif "# %" in line:
                    expression = line.split("# %")[1].strip()
                    if expression.startswith("?"):
                        expression = expression[1:]
                        strict = False
                    else:
                        strict = True
                    line_woc = line.split("# %")[0].rstrip()
                    context["line"] = line_woc
                    eval_line = RepEval(expression).substitute(context=context, strict=strict)
                    if eval_line is not None:
                        if not isinstance(eval_line, RepEval):
                            if isinstance(eval_line, str):
                                out_lines.append(eval_line)
                            else:
                                lines[i + 1 : i + 1] = eval_line
                        else:
                            out_lines.append(line)
                elif uncomment:
                    if sline.startswith("# "):
                        out_lines.append(sline[2:])
                    elif sline.startswith("#"):
                        out_lines.append(sline[1:])
                    else:
                        out_lines.append(line)
                else:
                    out_lines.append(line)

        i += 1
    if section_found:
        raise ValueError(f"Code section '{section_name}' not closed")
    raise ValueError(f"Code section '{section_name}' not found")


def suggest_module_path(
    section_name: str,
    path: tp.Optional[tp.PathLike] = None,
    mkdir_kwargs: tp.KwargsLike = None,
) -> Path:
    """Suggest a file path for the target module.

    Determines a suitable file path based on the provided section name and optional path.
    If the supplied path is a directory or lacks a file extension, the section name is used to
    form the file name with a `.py` extension. This function also ensures that the target directory exists.

    Args:
        section_name (str): The name of the code section.
        path (Optional[PathLike]): The base path or file path.
        mkdir_kwargs (KwargsLike): Additional keyword arguments for directory creation.

    Returns:
        Path: The suggested file path.
    """
    if path is None:
        path = Path(".")
    else:
        path = Path(path)
    if not path.is_file() and path.suffix == "":
        path = (path / section_name).with_suffix(".py")
    if mkdir_kwargs is None:
        mkdir_kwargs = {}
    check_mkdir(path.parent, **mkdir_kwargs)
    return path


def cut_and_save(
    code: str,
    section_name: str,
    path: tp.Optional[tp.PathLike] = None,
    mkdir_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> Path:
    """Extract an annotated section from the code and save it to a file.

    This function cuts a specified section from the provided source code using `cut_from_code`
    and then saves the processed code to a file determined by `suggest_module_path`.

    Args:
        code (str): The source code containing the annotated section.
        section_name (str): The name of the section to extract.
        path (Optional[PathLike]): The file path or directory for saving the extracted code.
        mkdir_kwargs (KwargsLike): Additional keyword arguments for directory creation.
        **kwargs: Additional keyword arguments passed to `cut_from_code`.

    Returns:
        Path: The file path where the extracted code section was saved.
    """
    parsed_code = cut_from_code(code, section_name, **kwargs)
    path = suggest_module_path(section_name, path=path, mkdir_kwargs=mkdir_kwargs)
    with open(path, "w") as f:
        f.write(parsed_code)
    return path


def cut_and_save_module(module: tp.Union[str, ModuleType], *args, **kwargs) -> Path:
    """Extract an annotated section from a module's source code and save it to a file.

    If the module is provided as a string representing its import path, it is first imported
    before processing. The source code is then retrieved using `inspect.getsource`, after which
    the annotated section is extracted and saved via `cut_and_save`.

    Args:
        module (Union[str, ModuleType]): The target module or its import path.
        *args: Additional positional arguments for `cut_and_save`.
        **kwargs: Additional keyword arguments for `cut_and_save`.

    Returns:
        Path: The file path where the extracted module section was saved.
    """
    if isinstance(module, str):
        module = importlib.import_module(module)
    code = inspect.getsource(module)
    return cut_and_save(code, *args, **kwargs)


def cut_and_save_func(func: tp.Union[str, FunctionType], *args, **kwargs) -> Path:
    """Cut a function's annotated code section from its module and save it to a file.

    Args:
        func (Union[str, FunctionType]): A function reference or its fully qualified name as a string.

            If a string is provided, the module is imported and the function is retrieved.
        *args: Additional arguments passed to `cut_and_save`.
        **kwargs: Additional keyword arguments passed to `cut_and_save`.

    Returns:
        Path: The file path where the extracted code section is saved.
    """
    if isinstance(func, str):
        module = importlib.import_module(".".join(func.split(".")[:-1]))
        func = getattr(module, func.split(".")[-1])
    else:
        module = inspect.getmodule(func)
    code = inspect.getsource(module)
    return cut_and_save(code, section_name=func.__name__, *args, **kwargs)
