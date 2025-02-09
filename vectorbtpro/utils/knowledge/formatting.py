# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Classes for content formatting.

See `vectorbtpro.utils.knowledge` for the toy dataset."""

import re
import inspect
import time
import sys
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import Configured, merge_dicts, flat_merge_dicts
from vectorbtpro.utils.module_ import get_caller_qualname
from vectorbtpro.utils.path_ import check_mkdir
from vectorbtpro.utils.template import CustomTemplate, Sub, RepFunc

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from IPython.display import DisplayHandle as DisplayHandleT
except ImportError:
    DisplayHandleT = "DisplayHandle"

__all__ = [
    "ContentFormatter",
    "PlainFormatter",
    "IPythonFormatter",
    "IPythonMarkdownFormatter",
    "IPythonHTMLFormatter",
    "HTMLFileFormatter",
]


def to_markdown(
    text: str,
    remove_code_title: tp.Optional[bool] = None,
    even_indentation: tp.Optional[bool] = None,
) -> str:
    """Convert text to Markdown.

    If `remove_code_title` is True, removes `title` attribute from a code block and puts it above it.

    If `even_indentation` is True, makes leading spaces even. For example, 3 leading spaces become 4."""
    from vectorbtpro._settings import settings

    formatting_cfg = settings["knowledge"]["formatting"]
    if remove_code_title is None:
        remove_code_title = formatting_cfg["remove_code_title"]
    if even_indentation is None:
        even_indentation = formatting_cfg["even_indentation"]

    markdown = text
    if remove_code_title:

        def _replace_code_block(match):
            language = match.group(1)
            title = match.group(2)
            code = match.group(3)
            if title:
                title_md = f"**{title}**\n\n"
            else:
                title_md = ""
            code_md = f"```{language}\n{code}\n```"
            return title_md + code_md

        code_block_pattern = re.compile(r'```(\w+)\s+title="([^"]*)"\s*\n(.*?)\n```', re.DOTALL)
        markdown = code_block_pattern.sub(_replace_code_block, markdown)

    if even_indentation:
        leading_spaces_pattern = re.compile(r"^( +)(?=\S|$|\n)")
        fixed_lines = []
        for line in markdown.splitlines(keepends=True):
            match = leading_spaces_pattern.match(line)
            if match and len(match.group(0)) % 2 != 0:
                line = " " + line
            fixed_lines.append(line)
        markdown = "".join(fixed_lines)

    return markdown


def to_html(
    markdown: str,
    resolve_extensions: tp.Optional[bool] = None,
    make_links: tp.Optional[bool] = None,
    **markdown_kwargs,
) -> str:
    """Convert Markdown to HTML.

    If `resolve_extensions` is True, resolves Markdown extensions. Uses `pymdownx` extensions over
    native extensions if installed.

    If `make_links` is True, detects raw URLs in HTML text (p and span elements only) and convert them to links.

    Keyword arguments are passed to `markdown.markdown`."""
    from vectorbtpro._settings import settings
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("markdown")
    import markdown as md

    formatting_cfg = settings["knowledge"]["formatting"]
    if resolve_extensions is None:
        resolve_extensions = formatting_cfg["resolve_extensions"]
    if make_links is None:
        make_links = formatting_cfg["make_links"]
    markdown_kwargs = merge_dicts(formatting_cfg["markdown_kwargs"], markdown_kwargs)

    extensions = markdown_kwargs.pop("extensions", [])
    if resolve_extensions:
        from vectorbtpro.utils.module_ import check_installed

        filtered_extensions = [ext for ext in extensions if "." not in ext or check_installed(ext.partition(".")[0])]
        ext_set = set(filtered_extensions)
        remove_fenced_code = "fenced_code" in ext_set and "pymdownx.superfences" in ext_set
        remove_codehilite = "codehilite" in ext_set and "pymdownx.highlight" in ext_set
        if remove_fenced_code or remove_codehilite:
            filtered_extensions = [
                ext
                for ext in filtered_extensions
                if not ((ext == "fenced_code" and remove_fenced_code) or (ext == "codehilite" and remove_codehilite))
            ]
        extensions = filtered_extensions
    html = md.markdown(markdown, extensions=extensions, **markdown_kwargs)
    if make_links:
        tag_pattern = re.compile(r"<(p|span)(\s[^>]*)?>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
        url_pattern = re.compile(r'(https?://[^\s<>"\'`]+?)(?=[.,;:!?)\]]*(?:\s|$))', re.IGNORECASE)

        def _replace_urls(match, _url_pattern=url_pattern):
            tag = match.group(1)
            attributes = match.group(2) if match.group(2) else ""
            content = match.group(3)
            parts = re.split(r"(<a\b[^>]*>.*?</a>)", content, flags=re.DOTALL | re.IGNORECASE)
            for i, part in enumerate(parts):
                if not re.match(r"<a\b[^>]*>.*?</a>", part, re.DOTALL | re.IGNORECASE):
                    part = _url_pattern.sub(r'<a href="\1">\1</a>', part)
                    parts[i] = part
            new_content = "".join(parts)
            return f"<{tag}{attributes}>{new_content}</{tag}>"

        html = tag_pattern.sub(_replace_urls, html)
    return html.strip()


def format_html(
    html_template: tp.Optional[str] = None,
    title: tp.Optional[str] = None,
    html_metadata: tp.Optional[str] = None,
    html_content: tp.Optional[str] = None,
    style_extras: tp.Optional[tp.MaybeList[str]] = None,
    head_extras: tp.Optional[tp.MaybeList[str]] = None,
    body_extras: tp.Optional[tp.MaybeList[str]] = None,
    use_pygments: tp.Optional[bool] = None,
    pygments_kwargs: tp.KwargsLike = None,
    template_context: tp.KwargsLike = None,
) -> str:
    """Format HTML template.

    If `use_pygments` is True, uses Pygments package for code highlighting. Arguments in
    `pygments_kwargs` are then passed to `pygments.formatters.HtmlFormatter`.

    Use `style_extras` to inject additional CSS rules outside the predefined ones.
    Use `head_extras` to inject additional HTML elements into the `<head>` section, such as meta tags,
    links to external stylesheets, or scripts. Use `body_extras` to inject JavaScript files or inline
    scripts at the end of the `<body>`. All of these arguments can be lists.

    HTML template is a template that can use all the arguments except those related to pygments.
    It can be either a custom template, or string or function that will become one."""
    from vectorbtpro._settings import settings
    from vectorbtpro.utils.module_ import check_installed, assert_can_import

    formatting_cfg = settings["knowledge"]["formatting"]

    if html_template is None:
        html_template = formatting_cfg["html_template"]
    if style_extras is None:
        style_extras = []
    style_extras = formatting_cfg["style_extras"] + style_extras
    if head_extras is None:
        head_extras = []
    head_extras = formatting_cfg["head_extras"] + head_extras
    if body_extras is None:
        body_extras = []
    body_extras = formatting_cfg["body_extras"] + body_extras
    if use_pygments is None:
        use_pygments = formatting_cfg["use_pygments"]
    pygments_kwargs = merge_dicts(formatting_cfg["pygments_kwargs"], pygments_kwargs)

    if title is None:
        title = ""
    if html_metadata is None:
        html_metadata = ""
    if html_content is None:
        html_content = ""
    if style_extras is None:
        style_extras = []
    if isinstance(style_extras, str):
        style_extras = [style_extras]
    if not isinstance(style_extras, list):
        style_extras = list(style_extras)
    style_extras = "\n".join(style_extras)
    if head_extras is None:
        head_extras = []
    if isinstance(head_extras, str):
        head_extras = [head_extras]
    if not isinstance(head_extras, list):
        head_extras = list(head_extras)
    head_extras = "\n".join(head_extras)
    if body_extras is None:
        body_extras = []
    if isinstance(body_extras, str):
        body_extras = [body_extras]
    if not isinstance(body_extras, list):
        body_extras = list(body_extras)
    body_extras = "\n".join(body_extras)
    if use_pygments is None:
        use_pygments = check_installed("pygments")
    if use_pygments:
        assert_can_import("pygments")
        from pygments.formatters import HtmlFormatter

        formatter = HtmlFormatter(**pygments_kwargs)
        highlight_css = formatter.get_style_defs(".highlight")
        if style_extras == "":
            style_extras = highlight_css
        else:
            style_extras = highlight_css + "\n" + style_extras
    if isinstance(html_template, str):
        html_template = Sub(html_template)
    elif checks.is_function(html_template):
        html_template = RepFunc(html_template)
    elif not isinstance(html_template, CustomTemplate):
        raise TypeError(f"HTML template must be a string, function, or template")
    return html_template.substitute(
        flat_merge_dicts(
            dict(
                title=title,
                html_metadata=html_metadata,
                html_content=html_content,
                style_extras=style_extras,
                head_extras=head_extras,
                body_extras=body_extras,
            ),
            template_context,
        ),
        eval_id="context_prompt",
    )


class ContentFormatter(Configured):
    """Class for formatting content.

    For defaults, see `formatting` in `vectorbtpro._settings.knowledge`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.formatting", "knowledge.formatting.formatter_config"]

    def __init__(
        self,
        output_to: tp.Optional[tp.Union[str, tp.TextIO]] = None,
        flush_output: tp.Optional[bool] = None,
        buffer_output: tp.Optional[bool] = None,
        close_output: tp.Optional[bool] = None,
        update_interval: tp.Optional[float] = None,
        minimal_format: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            output_to=output_to,
            flush_output=flush_output,
            buffer_output=buffer_output,
            close_output=close_output,
            update_interval=update_interval,
            minimal_format=minimal_format,
            template_context=template_context,
            **kwargs,
        )

        output_to = self.resolve_setting(output_to, "output_to")
        flush_output = self.resolve_setting(flush_output, "flush_output")
        buffer_output = self.resolve_setting(buffer_output, "buffer_output")
        close_output = self.resolve_setting(close_output, "close_output")
        update_interval = self.resolve_setting(update_interval, "update_interval")
        minimal_format = self.resolve_setting(minimal_format, "minimal_format")
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        if isinstance(output_to, (str, Path)):
            output_to = Path(output_to).open("w")
            if close_output is None:
                close_output = True
        else:
            if close_output is None:
                close_output = False

        self._output_to = output_to
        self._flush_output = flush_output
        self._buffer_output = buffer_output
        self._close_output = close_output
        self._update_interval = update_interval
        self._minimal_format = minimal_format
        self._template_context = template_context

        self._last_update = None
        self._lines = []
        self._current_line = []
        self._in_code_block = False
        self._code_block_indent = ""
        self._buffer = []
        self._content = ""

    @property
    def output_to(self) -> tp.Optional[tp.Union[str, tp.TextIO]]:
        """Redirect output to a file or stream."""
        return self._output_to

    @property
    def flush_output(self) -> bool:
        """Whether to flush output."""
        return self._flush_output

    @property
    def buffer_output(self) -> bool:
        """Whether to buffer output."""
        return self._buffer_output

    @property
    def close_output(self) -> bool:
        """Whether to close output."""
        return self._close_output

    @property
    def update_interval(self) -> tp.Optional[float]:
        """Update interval (in seconds)."""
        return self._update_interval

    @property
    def minimal_format(self) -> bool:
        """Whether input is minimally-formatted."""
        return self._minimal_format

    @property
    def template_context(self) -> tp.Kwargs:
        """Context used to substitute templates."""
        return self._template_context

    @property
    def last_update(self) -> tp.Optional[int]:
        """Last update time."""
        return self._last_update

    @property
    def lines(self) -> tp.List[str]:
        """List of lines."""
        return self._lines

    @property
    def current_line(self) -> tp.List[str]:
        """List of strings representing the current line."""
        return self._current_line

    @property
    def in_code_block(self) -> bool:
        """Whether currently in a code block."""
        return self._in_code_block

    @property
    def code_block_indent(self) -> str:
        """Indentation of the code block."""
        return self._code_block_indent

    @property
    def buffer(self) -> tp.List[str]:
        """List of strings in the buffer."""
        return self._buffer

    @property
    def content(self) -> str:
        """Content."""
        return self._content

    def initialize(self) -> None:
        """Initialize."""
        self._last_update = time.time()

    def format_line(self, line: str) -> str:
        """Format line."""
        start = 0
        while True:
            idx = line.find("```", start)
            if idx == -1:
                break
            if not self.in_code_block:
                self._in_code_block = True
                if line[:idx].strip() == "":
                    self._code_block_indent = line[:idx]
                else:
                    self._code_block_indent = ""
            else:
                self._in_code_block = False
            start = idx + 3
        return line

    def flush(self, final: bool = False) -> None:
        """Flush buffer and final line."""
        new_content = "".join(self.buffer)
        self.buffer.clear()

        lines = new_content.splitlines(keepends=True)
        for line in lines:
            if final or line.endswith("\n") or line.endswith("\r\n"):
                stripped_line = line.rstrip("\r\n")
                self.current_line.append(stripped_line)
                complete_line = "".join(self.current_line)
                formatted_line = self.format_line(complete_line)
                self.lines.append(formatted_line + line[len(stripped_line) :])
                self.current_line.clear()
            else:
                self.current_line.append(line)
        if final and self.current_line:
            complete_line = "".join(self.current_line)
            formatted_line = self.format_line(complete_line)
            self.lines.append(formatted_line)
            self.current_line.clear()

        if final:
            self._content = "".join(self.lines)
        else:
            content = self.lines.copy()
            if self.current_line:
                content.extend(self.current_line)
                if self.in_code_block:
                    content.append("\n" + self.code_block_indent + "```")
            else:
                if self.in_code_block:
                    content.append(self.code_block_indent + "```")
            self._content = "".join(content)

    def buffer_update(self) -> None:
        """Update based on buffer content."""
        if self.buffer_output and self.output_to is not None:
            print("".join(self.buffer), end="", file=self.output_to, flush=self.flush_output)

    def update(self, final: bool = False) -> None:
        """Update content."""
        self._last_update = time.time()
        if self.buffer:
            self.buffer_update()
        if self.buffer or (final and self.current_line):
            self.flush(final=final)

    def append(self, new_content: str, final: bool = False) -> None:
        """Append new content to buffer and update."""
        if not self.buffer_output and self.output_to is not None:
            print(new_content, end="", file=self.output_to, flush=self.flush_output)
        self.buffer.append(new_content)
        if (
            final
            or self.last_update is None
            or self.update_interval is None
            or (time.time() - self.last_update >= self.update_interval)
        ):
            self.update(final=final)

    def append_once(self, content: str) -> None:
        """Append final content and finalize."""
        if self.last_update is None:
            self.initialize()
        self.append(content, final=True)
        self.finalize(update=False)

    def finalize(self, update: bool = True) -> None:
        """Update for the last time and close the stream."""
        if update:
            self.update(final=True)
        if self.close_output and self.output_to is not None:
            self.output_to.close()

    def __enter__(self) -> tp.Self:
        self.initialize()
        return self

    def __exit__(self, *args) -> None:
        self.finalize()


class PlainFormatter(ContentFormatter):
    """Class for formatting plain content.

    For defaults, see `formatting.formatter_configs.plain` in `vectorbtpro._settings.knowledge`."""

    _short_name = "plain"

    _settings_path: tp.SettingsPath = "knowledge.formatting.formatter_configs.plain"

    def buffer_update(self) -> None:
        print("".join(self.buffer), end="")


class IPythonFormatter(ContentFormatter):
    """Class for formatting plain content in IPython.

    For defaults, see `formatting.formatter_configs.ipython` in `vectorbtpro._settings.knowledge`."""

    _short_name = "ipython"

    _settings_path: tp.SettingsPath = "knowledge.formatting.formatter_configs.ipython"

    def __init__(self, *args, **kwargs) -> None:
        ContentFormatter.__init__(self, *args, **kwargs)

        self._display_handle = None

    @property
    def display_handle(self) -> tp.Optional[DisplayHandleT]:
        """Display handle."""
        return self._display_handle

    def initialize(self) -> None:
        ContentFormatter.initialize(self)

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("IPython")
        from IPython.display import display

        self._display_handle = display("", display_id=True)

    def update_display(self) -> None:
        """Update display with content."""
        self.display_handle.update(self.content)

    def update(self, final: bool = False) -> None:
        ContentFormatter.update(self, final=final)
        self.update_display()


class IPythonMarkdownFormatter(IPythonFormatter):
    """Class for formatting Markdown content in IPython.

    For defaults, see `formatting.formatter_configs.ipython_markdown` in `vectorbtpro._settings.knowledge`."""

    _short_name = "ipython_markdown"

    _settings_path: tp.SettingsPath = "knowledge.formatting.formatter_configs.ipython_markdown"

    def __init__(self, *args, to_markdown_kwargs: tp.KwargsLike = None, **kwargs) -> None:
        IPythonFormatter.__init__(
            self,
            *args,
            to_markdown_kwargs=to_markdown_kwargs,
            **kwargs,
        )

        if self.minimal_format:
            to_markdown_kwargs = self.resolve_setting(
                to_markdown_kwargs, "to_markdown_kwargs", sub_path="minimal_format_config", merge=True
            )
        else:
            to_markdown_kwargs = self.resolve_setting(to_markdown_kwargs, "to_markdown_kwargs", merge=True)

        self._to_markdown_kwargs = to_markdown_kwargs

    @property
    def to_markdown_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `to_markdown`."""
        return self._to_markdown_kwargs

    def update_display(self) -> None:
        from IPython.display import Markdown

        markdown_content = to_markdown(self.content, **self.to_markdown_kwargs)
        self.display_handle.update(Markdown(markdown_content))


class IPythonHTMLFormatter(IPythonFormatter):
    """Class for formatting HTML content in IPython.

    For defaults, see `formatting.formatter_configs.ipython_html` in `vectorbtpro._settings.knowledge`."""

    _short_name = "ipython_html"

    _settings_path: tp.SettingsPath = "knowledge.formatting.formatter_configs.ipython_html"

    def __init__(
        self,
        *args,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        IPythonFormatter.__init__(
            self,
            *args,
            to_markdown_kwargs=to_markdown_kwargs,
            to_html_kwargs=to_html_kwargs,
            **kwargs,
        )

        if self.minimal_format:
            to_markdown_kwargs = self.resolve_setting(
                to_markdown_kwargs, "to_markdown_kwargs", sub_path="minimal_format_config", merge=True
            )
            to_html_kwargs = self.resolve_setting(
                to_html_kwargs, "to_html_kwargs", sub_path="minimal_format_config", merge=True
            )
        else:
            to_markdown_kwargs = self.resolve_setting(to_markdown_kwargs, "to_markdown_kwargs", merge=True)
            to_html_kwargs = self.resolve_setting(to_html_kwargs, "to_html_kwargs", merge=True)

        self._to_markdown_kwargs = to_markdown_kwargs
        self._to_html_kwargs = to_html_kwargs

    @property
    def to_markdown_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `to_markdown`."""
        return self._to_markdown_kwargs

    @property
    def to_html_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `to_html`."""
        return self._to_html_kwargs

    def update_display(self) -> None:
        from IPython.display import HTML

        markdown_content = to_markdown(self.content, **self.to_markdown_kwargs)
        html_content = to_html(markdown_content, **self.to_html_kwargs)
        self.display_handle.update(HTML(html_content))


class HTMLFileFormatter(ContentFormatter):
    """Class for formatting static HTML files.

    For defaults, see `formatting.formatter_configs.html` in `vectorbtpro._settings.knowledge`."""

    _short_name = "html"

    _settings_path: tp.SettingsPath = "knowledge.formatting.formatter_configs.html"

    def __init__(
        self,
        *args,
        page_title: str = "",
        refresh_page: tp.Optional[bool] = None,
        dir_path: tp.Optional[tp.PathLike] = None,
        mkdir_kwargs: tp.KwargsLike = None,
        temp_files: tp.Optional[bool] = None,
        file_prefix_len: tp.Optional[int] = None,
        file_suffix_len: tp.Optional[int] = None,
        open_browser: tp.Optional[bool] = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        format_html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        ContentFormatter.__init__(
            self,
            *args,
            page_title=page_title,
            refresh_page=refresh_page,
            dir_path=dir_path,
            mkdir_kwargs=mkdir_kwargs,
            temp_files=temp_files,
            file_prefix_len=file_prefix_len,
            file_suffix_len=file_suffix_len,
            open_browser=open_browser,
            to_markdown_kwargs=to_markdown_kwargs,
            to_html_kwargs=to_html_kwargs,
            format_html_kwargs=format_html_kwargs,
            **kwargs,
        )

        refresh_page = self.resolve_setting(refresh_page, "refresh_page")
        dir_path = self.resolve_setting(dir_path, "dir_path")
        mkdir_kwargs = self.resolve_setting(mkdir_kwargs, "mkdir_kwargs", merge=True)
        temp_files = self.resolve_setting(temp_files, "temp_files")
        file_prefix_len = self.resolve_setting(file_prefix_len, "file_prefix_len")
        file_suffix_len = self.resolve_setting(file_suffix_len, "file_suffix_len")
        open_browser = self.resolve_setting(open_browser, "open_browser")

        if self.minimal_format:
            to_markdown_kwargs = self.resolve_setting(
                to_markdown_kwargs, "to_markdown_kwargs", sub_path="minimal_format_config", merge=True
            )
            to_html_kwargs = self.resolve_setting(
                to_html_kwargs, "to_html_kwargs", sub_path="minimal_format_config", merge=True
            )
            format_html_kwargs = self.resolve_setting(
                format_html_kwargs, "format_html_kwargs", sub_path="minimal_format_config", merge=True
            )
        else:
            to_markdown_kwargs = self.resolve_setting(to_markdown_kwargs, "to_markdown_kwargs", merge=True)
            to_html_kwargs = self.resolve_setting(to_html_kwargs, "to_html_kwargs", merge=True)
            format_html_kwargs = self.resolve_setting(format_html_kwargs, "format_html_kwargs", merge=True)

        dir_path = self.resolve_setting(dir_path, "dir_path")
        template_context = self.template_context
        if isinstance(dir_path, CustomTemplate):
            cache_dir = self.get_setting("cache_dir", default=None)
            if cache_dir is not None:
                if isinstance(cache_dir, CustomTemplate):
                    cache_dir = cache_dir.substitute(template_context, eval_id="cache_dir")
                template_context = flat_merge_dicts(dict(cache_dir=cache_dir), template_context)
            release_dir = self.get_setting("release_dir", default=None)
            if release_dir is not None:
                if isinstance(release_dir, CustomTemplate):
                    release_dir = release_dir.substitute(template_context, eval_id="release_dir")
                template_context = flat_merge_dicts(dict(release_dir=release_dir), template_context)
            dir_path = dir_path.substitute(template_context, eval_id="dir_path")

        self._page_title = page_title
        self._refresh_page = refresh_page
        self._dir_path = dir_path
        self._mkdir_kwargs = mkdir_kwargs
        self._temp_files = temp_files
        self._file_prefix_len = file_prefix_len
        self._file_suffix_len = file_suffix_len
        self._open_browser = open_browser
        self._to_markdown_kwargs = to_markdown_kwargs
        self._to_html_kwargs = to_html_kwargs
        self._format_html_kwargs = format_html_kwargs

        self._file_handle = None

    @property
    def page_title(self) -> str:
        """Page title."""
        return self._page_title

    @property
    def refresh_page(self) -> bool:
        """Whether to refresh the HTML page."""
        return self._refresh_page

    @property
    def dir_path(self) -> tp.Optional[tp.Path]:
        """Path to the directory."""
        return self._dir_path

    @property
    def mkdir_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.path_.check_mkdir`."""
        return self._mkdir_kwargs

    @property
    def temp_files(self) -> bool:
        """Whether to save as temporary files

        Otherwise, will save under `HTMLFileFormatter.dir_path`."""
        return self._temp_files

    @property
    def file_prefix_len(self) -> int:
        """Number of chars of a truncated title as a prefix."""
        return self._file_prefix_len

    @property
    def file_suffix_len(self) -> int:
        """Number of chars of a random hash as a suffix."""
        return self._file_suffix_len

    @property
    def open_browser(self) -> bool:
        """Whether to open the default browser."""
        return self._open_browser

    @property
    def to_markdown_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `to_markdown`."""
        return self._to_markdown_kwargs

    @property
    def to_html_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `to_html`."""
        return self._to_html_kwargs

    @property
    def format_html_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `format_html`."""
        return self._format_html_kwargs

    @property
    def file_handle(self) -> tp.Optional[tp.TextIO]:
        """File handle."""
        return self._file_handle

    def initialize(self) -> None:
        ContentFormatter.initialize(self)

        if not self.temp_files:
            import secrets
            import string

            check_mkdir(self.dir_path, **self.mkdir_kwargs)
            page_title = self.page_title.lower().replace(" ", "-")
            if len(page_title) > self.file_prefix_len:
                words = page_title.split("-")
                truncated_page_title = ""
                for word in words:
                    if len(truncated_page_title) + len(word) + 1 <= self.file_prefix_len:
                        truncated_page_title += word + "-"
                    else:
                        break
                truncated_page_title = truncated_page_title.rstrip("-")
            else:
                truncated_page_title = page_title
            suffix_chars = string.ascii_lowercase + string.digits
            random_suffix = "".join(secrets.choice(suffix_chars) for _ in range(self.file_suffix_len))
            if truncated_page_title:
                short_filename = f"{truncated_page_title}-{random_suffix}.html"
            else:
                short_filename = f"{random_suffix}.html"
            file_path = self.dir_path / short_filename
            self._file_handle = open(str(file_path.resolve()), "w", encoding="utf-8")
        else:
            import tempfile

            self._file_handle = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                prefix=get_caller_qualname() + "_",
                suffix=".html",
                delete=False,
            )
        if self.refresh_page:
            refresh_content = max(1, int(self.update_interval)) if self.update_interval is not None else 1
            html = f'<!DOCTYPE html><html><head><meta http-equiv="refresh" content="{refresh_content}"></head></html>'
            self.file_handle.write(html)
            self.file_handle.flush()
        if self.open_browser:
            import webbrowser

            webbrowser.open("file://" + str(Path(self.file_handle.name).resolve()))

    def update(self, final: bool = False) -> None:
        """Update the HTML file with newly-updated content."""
        ContentFormatter.update(self, final=final)

        markdown_content = to_markdown(self.content, **self.to_markdown_kwargs)
        html_content = to_html(markdown_content, **self.to_html_kwargs)
        if not final and self.refresh_page:
            refresh_content = max(1, int(self.update_interval)) if self.update_interval is not None else 1
            _format_html_kwargs = dict(self.format_html_kwargs)
            head_extras = list(_format_html_kwargs.get("head_extras", []))
            if head_extras is None:
                head_extras = []
            if isinstance(head_extras, str):
                head_extras = [head_extras]
            else:
                head_extras = list(head_extras)
            head_extras.insert(0, f'<meta http-equiv="refresh" content="{refresh_content}">')
            _format_html_kwargs["head_extras"] = head_extras
            html_content = '<div id="overlay" class="overlay"></div>\n' + html_content
        else:
            _format_html_kwargs = self.format_html_kwargs
        html = format_html(
            title=self.page_title,
            html_content=html_content,
            **_format_html_kwargs,
        )
        self.file_handle.seek(0)
        self.file_handle.write(html)
        self.file_handle.truncate()
        self.file_handle.flush()


def resolve_formatter(formatter: tp.ContentFormatterLike) -> tp.MaybeType[ContentFormatter]:
    """Resolve a subclass or an instance of `ContentFormatter`.

    The following values are supported:

    * "plain" (`PlainFormatter`): Prints the raw output
    * "ipython" (`IPythonFormatter`): Renders an unformatted text in a notebook environment
    * "ipython_markdown" (`IPythonMarkdownFormatter`): Renders a Markdown in a notebook environment
    * "ipython_html" (`IPythonHTMLFormatter`): Renders an HTML in a notebook environment
    * "ipython_auto": Decides between using "ipython_html" or "plain" depending on the environment
    * "html" (`HTMLFileFormatter`): Writes a static HTML page and displays it in the browser
    * A subclass or an instance of `ContentFormatter`
    """
    if formatter is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["formatting"]
        formatter = chat_cfg["formatter"]
    if isinstance(formatter, str):
        if formatter.lower() == "ipython_auto":
            if checks.in_notebook():
                formatter = "ipython_html"
            else:
                formatter = "plain"
        current_module = sys.modules[__name__]
        found_formatter = None
        for name, cls in inspect.getmembers(current_module, inspect.isclass):
            if name.endswith("Formatter"):
                _short_name = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == formatter.lower():
                    found_formatter = cls
                    break
        if found_formatter is None:
            raise ValueError(f"Invalid formatter: '{formatter}'")
        formatter = found_formatter
    if isinstance(formatter, type):
        checks.assert_subclass_of(formatter, ContentFormatter, arg_name="formatter")
    else:
        checks.assert_instance_of(formatter, ContentFormatter, arg_name="formatter")
    return formatter
