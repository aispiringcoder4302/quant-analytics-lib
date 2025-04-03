# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for working with source code."""

import re
import ast
import importlib
import inspect
import difflib
import tempfile
import webbrowser
from pathlib import Path
from types import ModuleType, FunctionType

from vectorbtpro import _typing as tp
from vectorbtpro.utils.path_ import check_mkdir
from vectorbtpro.utils.template import CustomTemplate, RepEval

__all__ = [
    "cut_and_save_module",
    "cut_and_save_func",
    "refine_source",
    "refine_docstrings",
]


def collect_blocks(lines: tp.Iterable[str]) -> tp.Dict[str, tp.List[str]]:
    """Collect block sections from source code lines.

    Scans through the provided lines and groups lines into blocks defined by
    markers starting with `# % <block block_name>` and ending with `# % </block>`.

    Args:
        lines (Iterable[str]): Lines of source code.

    Returns:
        Dict[str, List[str]]: A mapping from block names to lists of lines for each block.
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


def cut_from_source(
    source: str,
    section_name: str,
    prepend_lines: tp.Optional[tp.Iterable[str]] = None,
    append_lines: tp.Optional[tp.Iterable[str]] = None,
    out_lines_callback: tp.Union[None, tp.Callable, CustomTemplate] = None,
    return_lines: bool = False,
    **kwargs,
) -> tp.Union[str, tp.List[str]]:
    """Extract an annotated section from the source code.

    Processes the source code string to extract a section defined by markers. The section is delimited
    by a starting marker `# % <section section_name>` and an ending marker `# % </section>`. Within the
    section, block subsections can be defined using markers `# % <block block_name>` and `# % </block>`.

    The function also handles skip and uncomment operations:

    * Lines between `# % <skip [expression]>` and `# % </skip>` are omitted.
    * Lines between `# % <uncomment [expression]>` and `# % </uncomment>` have their comment prefix removed.

    Any line containing `# %` outside these blocks is interpreted as a Python expression. The evaluation
    result of the expression directs the output as follows:

    * `None`: Skip the line.
    * `str`: Insert a single line.
    * `Iterable[str]`: Insert multiple lines into the output.

    Args:
        source (str): The source code to process.
        section_name (str): The name of the section to extract.
        prepend_lines (Optional[Iterable[str]]): Lines to prepend to the extracted section.
        append_lines (Optional[Iterable[str]]): Lines to append to the extracted section.
        out_lines_callback (Union[None, Callable, CustomTemplate]): A callback or template
            to process the output lines.
        return_lines (bool): If True, return the output as a list of lines.
        **kwargs: Additional context variables for expression evaluation.

    Returns:
        Union[str, List[str]]: The processed section as a cleaned string, or as a list
            of lines if `return_lines` is True.
    """
    lines = source.split("\n")
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

    Determines a suitable file path using the given section name and an optional base path.
    If the provided path is a directory or lacks a file extension, uses the section name to form a filename
    with a `.py` extension. This function also ensures that the target directory exists.

    Args:
        section_name (str): The name of the section.
        path (Optional[PathLike]): A base directory or file path.
        mkdir_kwargs (KwargsLike): Additional keyword arguments for directory creation.

    Returns:
        Path: The determined file path.
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
    source: str,
    section_name: str,
    path: tp.Optional[tp.PathLike] = None,
    mkdir_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> Path:
    """Extract an annotated section from the source code and save it to a file.

    Extracts a section, identified by an annotation, from the given source code using `cut_from_source`
    and saves it to a file determined by `suggest_module_path`.

    Args:
        source (str): The source code containing the annotated section.
        section_name (str): The name of the section to extract.
        path (Optional[PathLike]): File path or directory in which to save the extracted section.
        mkdir_kwargs (KwargsLike): Keyword arguments for directory creation.
        **kwargs: Additional keyword arguments passed to `cut_from_source`.

    Returns:
        Path: The file path where the extracted section is saved.
    """
    parsed_source = cut_from_source(source, section_name, **kwargs)
    path = suggest_module_path(section_name, path=path, mkdir_kwargs=mkdir_kwargs)
    with open(path, "w") as f:
        f.write(parsed_source)
    return path


def cut_and_save_module(module: tp.Union[str, ModuleType], *args, **kwargs) -> Path:
    """Extract an annotated section from a module's source code and save it to a file.

    If a module is provided as an import path string, it is imported prior to processing.
    The source code is retrieved using `inspect.getsource`, after which the specified section
    is extracted and saved using `cut_and_save`.

    Args:
        module (Union[str, ModuleType]): The target module or its import path.
        *args: Additional positional arguments passed to `cut_and_save`.
        **kwargs: Additional keyword arguments passed to `cut_and_save`.

    Returns:
        Path: The file path where the extracted module section is saved.
    """
    if isinstance(module, str):
        module = importlib.import_module(module)
    source = inspect.getsource(module)
    return cut_and_save(source, *args, **kwargs)


def cut_and_save_func(func: tp.Union[str, FunctionType], *args, **kwargs) -> Path:
    """Extract a function's annotated section from its module and save it to a file.

    If `func` is provided as a fully qualified name string, the containing module is imported and
    the function is retrieved before extraction. The source code is then obtained and processed
    using `cut_and_save`.

    Args:
        func (Union[str, FunctionType]): A function or its fully qualified name.

            If provided as a string, the module will be imported and the function will be retrieved.
        *args: Additional positional arguments passed to `cut_and_save`.
        **kwargs: Additional keyword arguments passed to `cut_and_save`.

    Returns:
        Path: The file path where the extracted function section is saved.
    """
    if isinstance(func, str):
        module = importlib.import_module(".".join(func.split(".")[:-1]))
        func = getattr(module, func.split(".")[-1])
    else:
        module = inspect.getmodule(func)
    source = inspect.getsource(module)
    return cut_and_save(source, section_name=func.__name__, *args, **kwargs)


def split_source(
    source: str,
    should_split: tp.Optional[tp.Callable[[ast.AST, int, int, int], bool]] = None,
    return_span: bool = False,
    return_level: bool = False,
) -> tp.SourceChunks:
    """Split the source code into definition-based chunks, optionally returning spans and nesting levels.

    The source code is divided into chunks based on code definitions such that the concatenation
    of the chunks reconstructs the original source code exactly, with no lines duplicated or lost.

    Args:
        source (str): The source code to split.
        should_split (Optional[Callable]): A callback `should_split(node, start: int, end: int, level: int) -> bool`
            to determine whether a node should be split into a header (with docstring) and body.

            By default, nodes are not split.

            !!! note
                `start` and `end` are 1-based line numbers.
        return_span (bool): Whether to also return the start and end line of each chunk.
        return_level (bool): Whether to also return the nesting level of each chunk.

    Returns:
        SourceChunks: A list of chunk source codes or tuples of chunk source codes and
            their start line, end line, and/or nesting level.

    """
    if should_split is None:

        def _should_split(node, start, end, level):
            return False

        should_split = _should_split

    lines = source.splitlines(keepends=True)
    tree = ast.parse(source, type_comments=True)

    def _compute_end_lineno(node):
        max_lineno = getattr(node, "lineno", 0)
        for child in ast.iter_child_nodes(node):
            child_end = _compute_end_lineno(child)
            if child_end > max_lineno:
                max_lineno = child_end
        return max_lineno

    def _get_stmt_range(stmt):
        start = stmt.lineno
        end = getattr(stmt, "end_lineno", _compute_end_lineno(stmt))
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and getattr(
            stmt, "decorator_list", None
        ):
            dec_starts = [getattr(d, "lineno", start) for d in stmt.decorator_list]
            if dec_starts:
                start = min(start, min(dec_starts))
        return start, end

    def _find_header_end(def_start, lines):
        header_end = def_start
        for i in range(def_start - 1, len(lines)):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith("#"):
                continue
            no_comment = stripped.split("#")[0].rstrip()
            if no_comment.endswith(":"):
                header_end = i + 1
                break
        return header_end

    def _extract_docstring_chunk(node, start, lines: tp.List[str]):
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            doc_stmt = node.body[0]
            chunk_end = getattr(doc_stmt, "end_lineno", _compute_end_lineno(doc_stmt))
            return (start, chunk_end), chunk_end + 1, sorted(node.body[1:], key=lambda s: s.lineno)
        return None

    def _split_def_node(node, start, end, level, lines):
        if not node.body:
            return [(start, end, level)]

        doc_info = _extract_docstring_chunk(node, start, lines)
        if doc_info is not None:
            (header_start, header_end), body_start, remaining_stmts = doc_info
            header_chunk = (header_start, header_end, level)
        else:
            header_end = _find_header_end(start, lines)
            header_chunk = (start, header_end, level)
            body_start = header_end + 1
            remaining_stmts = sorted(node.body, key=lambda s: s.lineno)
        chunks = [header_chunk]
        subchunks = _split_body_into_chunks(
            parent_start=body_start,
            parent_end=end,
            level=level,
            body=remaining_stmts,
            lines=lines,
        )
        chunks.extend(subchunks)
        return chunks

    def _split_body_into_chunks(parent_start, parent_end, level, body, lines):
        stmts_sorted = sorted(body, key=lambda st: st.lineno)
        chunks: tp.List[tp.Tuple[int, int, int]] = []
        current_line = parent_start

        for stmt in stmts_sorted:
            stmt_start, stmt_end = _get_stmt_range(stmt)
            if stmt_start > current_line:
                chunks.append((current_line, stmt_start - 1, level))
            if should_split(stmt, stmt_start, stmt_end, level):
                def_chunks = _split_def_node(
                    node=stmt,
                    start=stmt_start,
                    end=stmt_end,
                    level=level + 1,
                    lines=lines,
                )
                chunks.extend(def_chunks)
            else:
                chunks.append((stmt_start, stmt_end, level))
            current_line = stmt_end + 1
        if current_line <= parent_end:
            chunks.append((current_line, parent_end, level))

        return chunks

    top_chunks = _split_body_into_chunks(
        parent_start=1,
        parent_end=len(lines),
        level=0,
        body=tree.body,
        lines=lines,
    )
    if return_span and return_level:
        return [("".join(lines[start - 1 : end]), start, end, lvl) for (start, end, lvl) in top_chunks]
    if return_span:
        return [("".join(lines[start - 1 : end]), start, end) for (start, end, lvl) in top_chunks]
    if return_level:
        return [("".join(lines[start - 1 : end]), lvl) for (start, end, lvl) in top_chunks]
    else:
        return ["".join(lines[start - 1 : end]) for (start, end, lvl) in top_chunks]


def get_source_indent(source: str) -> int:
    """Return the minimum indentation, in spaces, of all non-empty lines in the source code.

    Tabs are treated as 4 spaces.

    Args:
        source (str): The source code to analyze.

    Returns:
        int: The minimum indentation in spaces.
    """
    lines = source.splitlines(keepends=True)
    indentations = []
    for line in lines:
        if line.strip():
            line_expanded = line.replace("\t", " " * 4)
            match = re.match(r"^( *)", line_expanded)
            if match:
                indentations.append(len(match.group(1)))
    return min(indentations) if indentations else 0


def remove_source_indent(source: str, indent: int) -> str:
    """Remove a fixed number of leading spaces from all non-empty lines in the source code.

    Tabs are treated as 4 spaces.

    Args:
        source (str): The source code to process.
        indent (int): The number of leading spaces to remove from each non-empty line.

    Returns:
        str: The source code with the specified indentation removed.
    """
    dedented_lines = []
    for line in source.splitlines(keepends=True):
        line_expanded = line.replace("\t", " " * 4)
        if line.strip():
            dedented_lines.append(line_expanded[indent:])
        else:
            dedented_lines.append(line_expanded)
    return "".join(dedented_lines)


def add_source_indent(source: str, indent: int) -> str:
    """Add spaces to each non-empty line in a source string.

    Args:
        source (str): The source code to modify.
        indent (int): The number of spaces to add as indentation to each non-empty line.

    Returns:
        str: The resulting source code with added indentation.
    """
    indent_str = " " * indent
    indented_lines = []
    for line in source.splitlines(keepends=True):
        if line.strip():
            indented_lines.append(indent_str + line)
        else:
            indented_lines.append(line)
    return "".join(indented_lines)


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
    split: bool = True,
    split_classes: bool = True,
    split_functions: bool = False,
    max_split_level: tp.Optional[int] = None,
    uniform_chunks: bool = True,
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
    """Refine the source code by splitting it into manageable chunks and applying completion methods.

    Args:
        source (Any): Source(s) or object(s) from which to extract the source code. A source may be:

            * a string containing code (e.g. "import vectorbtpro as vbt ..."),
            * a file path (e.g. "strategies/sma_crossover.py"),
            * a Python object (e.g. `pipeline_nb`), or
            * an iterable of the above.

            When a directory or package is provided, all contained Python files are processed.
        source_name (Optional[str]): Name displayed in the progress bar and/or HTML file name.
        as_package (bool): Whether to process a package as multiple sources.
        start_line (Optional[int]): Inclusive starting line number in the source code.

            !!! note
                Counting starts at 1.
        end_line (Optional[int]): Inclusive ending line number in the source code.

            !!! note
                Counting starts at 1.
        prompt (Optional[str]): System prompt used for generating completions.
        chunk_size (Optional[int]): Maximum token count for each chunk.

            If None, processes the entire source as a single chunk.
        split (bool): Whether to split the source code into chunks.
        split_classes (bool): Whether to split class definitions that exceed the chunk size.
        split_functions (bool): Whether to split function definitions that exceed the chunk size.
        max_split_level (Optional[int]): Maximum nesting level when splitting.
        uniform_chunks (bool): Whether to each chunk should start and end at the same base level.

            If nested chunks (with level > base) are present, includes them only if they fit as a whole.
        tokenize_kwargs (KwargsLike): Additional keyword arguments for
            `vectorbtpro.utils.knowledge.chatting.tokenize`.
        complete_kwargs (KwargsLike): Additional keyword arguments for
            `vectorbtpro.utils.knowledge.chatting.completed`.
        show_progress (Optional[bool]): Whether to display progress during chunk processing.
        pbar_kwargs (KwargsLike): Additional keyword arguments for `vectorbtpro.utils.pbar.ProgressBar`.
        mult_show_progress (Optional[bool]): Whether to display progress when processing multiple sources.

            If not provided, defaults to `show_progress`.
        mult_pbar_kwargs (KwargsLike): Additional keyword arguments for the progress bar
            when processing multiple sources.

            These are merged with `pbar_kwargs`.
        modify (bool): Whether to update the source file with the refined code.
        copy_to_clipboard (bool): Whether to copy the refined source code to the clipboard.

            Does not apply when processing multiple sources.
        show_diff (bool): Whether to generate and display an HTML diff file using `difflib`.

            Does not apply when processing multiple sources.
        open_browser (bool): Whether to open the HTML diff in a web browser.

            Does not apply when processing multiple sources.

    Returns:
        Union[RefineSourceOutput, RefineSourceOutputs]: Result of the refinement process.

            * Returns the refined source code if neither `modify` nor `copy_to_clipboard` is True.
            * Returns the path to the updated source file if `modify` is True.
            * Returns the path to the HTML diff file if `show_diff` is True.
            * For multiple sources, returns a zipped list of sources and their corresponding outputs.
    """
    from vectorbtpro.utils.checks import is_numba_func, is_complex_iterable
    from vectorbtpro.utils.config import merge_dicts
    from vectorbtpro.utils.module_ import assert_can_import
    from vectorbtpro.utils.path_ import get_common_prefix
    from vectorbtpro.utils.pbar import ProgressBar
    from vectorbtpro.utils.knowledge.chatting import tokenize, completed

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
        package_path = getattr(source, "__path__")
        if isinstance(package_path, list):
            source = [Path(path) for path in package_path]
        else:
            source = Path(package_path)
    if isinstance(source, Path) and source.is_dir():
        source = list(source.rglob("*.py"))
    if is_complex_iterable(source):
        sources = source
        new_sources = []
        for source in sources:
            if isinstance(source, Path) and source.is_dir():
                new_sources.extend(list(source.rglob("*.py")))
            else:
                new_sources.append(source)
        sources = new_sources
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
                paths.append(source_path.resolve())
            source_names.append(source_name)
        if all_paths:
            common_path = Path(get_common_prefix(paths)).resolve()
            same_file = True
            for path in paths:
                if path.relative_to(common_path) != Path():
                    same_file = False
                    break
            if not same_file:
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
                    split=split,
                    split_classes=split_classes,
                    split_functions=split_functions,
                    max_split_level=max_split_level,
                    uniform_chunks=uniform_chunks,
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
        if source_start_line == 0:
            source_start_line = 1
        if source_name is None:
            source_name = source_path.name

    if start_line is None:
        start_line = 1
    if end_line is None:
        end_line = len(source_lines)
    start_index = start_line - 1
    end_index = end_line
    source_lines = source_lines[start_index:end_index]

    source_end_line = source_start_line + end_line - 1
    source_start_line = source_start_line + start_line - 1
    source_start_index = source_start_line - 1
    source_end_index = source_end_line
    source = "".join(source_lines)
    source_name = f"{source_name}#L{source_start_line}-L{source_end_line}"

    if prompt is None:
        prompt = REFINE_SRC_PROMPT
    if tokenize_kwargs is None:
        tokenize_kwargs = {}
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

    if split:

        def _should_split(node, start, end, level):
            if max_split_level is None or level <= max_split_level:
                if (isinstance(node, ast.ClassDef) and split_classes) or (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and split_functions
                ):
                    node_source = "".join(source_lines[start - 1 : end])
                    if len(tokenize(node_source, **tokenize_kwargs)) > chunk_size:
                        return True
            return False

        chunks_with_level = split_source(source, should_split=_should_split, return_level=True)
        if uniform_chunks:
            source_chunks = []
            buffer = []
            buffer_tokens = 0
            buffer_level = None
            i = 0

            def _flush():
                nonlocal buffer, buffer_tokens, buffer_level
                if buffer:
                    source_chunks.append("".join(chunk for chunk, _ in buffer))
                    buffer = []
                    buffer_tokens = 0
                    buffer_level = None

            while i < len(chunks_with_level):
                chunk, level = chunks_with_level[i]
                chunk_tokens = len(tokenize(chunk, **tokenize_kwargs))
                if buffer_level is None:
                    if chunk_tokens > chunk_size:
                        source_chunks.append(chunk)
                        i += 1
                        continue
                    buffer_level = level
                if level < buffer_level:
                    _flush()
                    continue
                if level > buffer_level:
                    nested_group = []
                    nested_tokens = 0
                    j = i
                    while j < len(chunks_with_level) and chunks_with_level[j][1] > buffer_level:
                        nested_chunk, _ = chunks_with_level[j]
                        tks = len(tokenize(nested_chunk, **tokenize_kwargs))
                        nested_group.append((nested_chunk, level))
                        nested_tokens += tks
                        j += 1
                    if buffer_tokens + nested_tokens <= chunk_size:
                        buffer.extend(nested_group)
                        buffer_tokens += nested_tokens
                        i = j
                        continue
                    else:
                        _flush()
                        continue
                if buffer_tokens + chunk_tokens > chunk_size:
                    _flush()
                    continue
                else:
                    buffer.append((chunk, level))
                    buffer_tokens += chunk_tokens
                    i += 1
            _flush()
        else:
            source_chunks = []
            buffer = []
            buffer_tokens = 0
            for chunk, level in chunks_with_level:
                chunk_tokens = len(tokenize(chunk, **tokenize_kwargs))
                if buffer_tokens + chunk_tokens > chunk_size:
                    if buffer:
                        source_chunks.append("".join(buffer))
                    buffer = [chunk]
                    buffer_tokens = chunk_tokens
                else:
                    buffer.append(chunk)
                    buffer_tokens += chunk_tokens
            if buffer:
                source_chunks.append("".join(buffer))
    else:
        source_chunks = [source]

    processed = []
    chunk_start_line = source_start_line
    with ProgressBar(total=len(source_chunks), show_progress=show_progress, **pbar_kwargs) as pbar:
        for i in range(len(source_chunks)):
            chunk = source_chunks[i]
            chunk_lines = chunk.splitlines(keepends=True)
            pbar.set_description(
                dict(
                    lines="{}..{}".format(
                        chunk_start_line,
                        chunk_start_line + len(chunk_lines) - 1,
                    )
                )
            )
            indent = get_source_indent(chunk)
            chunk = remove_source_indent(chunk, indent)
            leading_len = len(chunk) - len(chunk.lstrip())
            leading = chunk[:leading_len]
            trailing_len = len(chunk) - len(chunk.rstrip())
            trailing = chunk[-trailing_len:] if trailing_len > 0 else ""
            middle = chunk[leading_len : len(chunk) - trailing_len]
            new_middle = completed(middle, **complete_kwargs)
            new_middle = add_source_indent(new_middle, indent)
            new_chunk = leading + new_middle + trailing
            processed.append(new_chunk)
            chunk_start_line += len(chunk_lines)
            pbar.update()
    new_source = "".join(processed)

    if modify and source_path:
        with source_path.open("r", encoding="utf-8") as f:
            file_contents = f.readlines()
        new_source_lines = new_source.splitlines(keepends=True)
        if not new_source_lines or not new_source_lines[-1].endswith("\n"):
            new_source_lines.append("\n")
        file_contents[source_start_index:source_end_index] = new_source_lines
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
- If a function returns `None`, **do not add a "Returns" section**.
- **Do not add your own "Usage" section**.
- **Make sure to list all arguments, their types, and descriptions**, apart from `self`, `cls`, and `cls_or_self`
- Do not mention type `dict` when describing variable keyword arguments.

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
    - Do not unfold type hints prefixed with `Maybe` into a union. Keep them as they are.
    - Change type hints in docstring only, **do not change type hints in function signatures**.
- Treat classes decorated with `@define` **as if they were decorated with `@attr.s`**, adjusting 
    docstrings accordingly.
    - **Do not duplicate fields and their descriptions** in the "Args" section.
    - Create an "Args" section only if the class defines its own `__init__` method.
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
- Make sure that docstrings have three double quotes (\"\"\") at the start and the end
- Make sure that **docstrings are properly indented** relative to the first three double quotes (\"\"\")

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
- **Do not change any existing indentation or spacing**.
- **Do not change usage examples or their formatting**.
- **Use the name "vectorbtpro" instead of "VectorBT® PRO"** in docstrings.
"""
"""Prompt for `refine_docstrings`."""


def refine_docstrings(source: tp.Any, **kwargs) -> tp.RefineSourceOutput:
    """Call `refine_source` with the prompt from `REFINE_DOCSTR_PROMPT` to refine
    docstrings in the given source code.

    Args:
        source (Any): The source code to be refined.
        **kwargs: Additional keyword arguments passed to `refine_source`.

    Returns:
        RefineSourceOutput: The result of the refinement process.
    """
    return refine_source(source, prompt=REFINE_DOCSTR_PROMPT, **kwargs)
