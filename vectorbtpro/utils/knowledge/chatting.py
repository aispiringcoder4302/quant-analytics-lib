# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Classes for chatting.

See `vectorbtpro.utils.knowledge` for the toy dataset."""

import inspect
import time
import sys
import warnings
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, flat_merge_dicts, deep_merge_dicts, Configured, HasSettings
from vectorbtpro.utils.decorators import hybrid_method
from vectorbtpro.utils.module_ import get_caller_qualname
from vectorbtpro.utils.parsing import get_func_arg_names, get_func_kwargs
from vectorbtpro.utils.path_ import check_mkdir, remove_dir
from vectorbtpro.utils.template import CustomTemplate, RepFunc, Sub

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from tiktoken import Encoding as EncodingT
except ImportError:
    EncodingT = "Encoding"
try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from openai import OpenAI as OpenAIT, Stream as StreamT
    from openai.types.chat.chat_completion import ChatCompletion as ChatCompletionT
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as ChatCompletionChunkT
except ImportError:
    OpenAIT = "OpenAI"
    StreamT = "Stream"
    ChatCompletionT = "ChatCompletion"
    ChatCompletionChunkT = "ChatCompletionChunk"
try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from litellm import ModelResponse as ModelResponseT, CustomStreamWrapper as CustomStreamWrapperT
except ImportError:
    ModelResponseT = "ModelResponse"
    CustomStreamWrapperT = "CustomStreamWrapper"
try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from llama_index.core.llms import LLM as LLMT, ChatMessage as ChatMessageT, ChatResponse as ChatResponseT
except ImportError:
    LLMT = "LLM"
    ChatMessageT = "ChatMessage"
    ChatResponseT = "ChatResponse"
try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from IPython.display import DisplayHandle as DisplayHandleT
except ImportError:
    DisplayHandleT = "DisplayHandle"

__all__ = [
    "ChatEngine",
    "OpenAIEngine",
    "LiteLLMEngine",
    "LlamaIndexEngine",
    "ContentFormatter",
    "PlainFormatter",
    "IPythonFormatter",
    "IPythonMarkdownFormatter",
    "IPythonHTMLFormatter",
    "HTMLFileFormatter",
    "Contextable",
]


# ############# Chat engines ############# #


class ChatEngine(Configured):
    """Abstract class representing a chat engine."""

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the engine."""

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    def __init__(self, **kwargs) -> None:
        Configured.__init__(self, **kwargs)

    @property
    def model(self) -> tp.Optional[str]:
        """Model."""
        return None

    def get_chat_response(self, messages: tp.Union[str, tp.List[dict]]) -> tp.Any:
        """Get chat response to messages."""
        raise NotImplementedError

    def get_message_content(self, response: tp.Any) -> tp.Optional[str]:
        """Get content from a chat response."""
        raise NotImplementedError

    def get_stream_response(self, messages: tp.Union[str, tp.List[dict]]) -> tp.Any:
        """Get streaming response to messages."""
        raise NotImplementedError

    def get_delta_content(self, response: tp.Any) -> tp.Optional[str]:
        """Get content from a streaming response chunk."""
        raise NotImplementedError


class OpenAIEngine(ChatEngine):
    """Chat engine class for OpenAI."""

    _short_name = "openai"

    _settings_path: tp.SettingsPath = "knowledge.chat.engines.openai"

    def __init__(self, **kwargs) -> None:
        ChatEngine.__init__(self, **kwargs)

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("openai")
        from openai import OpenAI

        openai_config = merge_dicts(self.get_settings(), kwargs)
        model = openai_config.pop("model", None)
        if model is None:
            raise ValueError("Must provide a model")
        client_arg_names = set(get_func_arg_names(OpenAI.__init__))
        client_kwargs = {}
        completion_kwargs = {}
        for k, v in openai_config.items():
            if k in client_arg_names:
                client_kwargs[k] = v
            else:
                completion_kwargs[k] = v
        client = OpenAI(**client_kwargs)

        self._model = model
        self._client = client
        self._completion_kwargs = completion_kwargs

    @property
    def model(self) -> tp.Optional[str]:
        return self._model

    @property
    def client(self) -> OpenAIT:
        """Client."""
        return self._client

    @property
    def completion_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `openai.resources.chat.completions.Completions.create`."""
        return self._completion_kwargs

    @classmethod
    def prepare_messages(cls, messages: tp.Union[str, tp.List[dict]]) -> tp.List[dict]:
        """Prepare messages."""
        if isinstance(messages, str):
            messages = [dict(role="user", content=messages)]
        return messages

    def get_chat_response(self, messages: tp.Union[str, tp.List[dict]]) -> ChatCompletionT:
        return self.client.chat.completions.create(
            messages=self.prepare_messages(messages),
            model=self.model,
            stream=False,
            **self.completion_kwargs,
        )

    def get_message_content(self, response: ChatCompletionT) -> tp.Optional[str]:
        return response.choices[0].message.content

    def get_stream_response(self, messages: tp.Union[str, tp.List[dict]]) -> StreamT:
        return self.client.chat.completions.create(
            messages=self.prepare_messages(messages),
            model=self.model,
            stream=True,
            **self.completion_kwargs,
        )

    def get_delta_content(self, response_chunk: ChatCompletionChunkT) -> tp.Optional[str]:
        return response_chunk.choices[0].delta.content


class LiteLLMEngine(ChatEngine):
    """Chat engine class for LiteLLM."""

    _short_name = "litellm"

    _settings_path: tp.SettingsPath = "knowledge.chat.engines.litellm"

    def __init__(self, **kwargs) -> None:
        ChatEngine.__init__(self, **kwargs)

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("litellm")

        completion_kwargs = merge_dicts(self.get_settings(), kwargs)
        model = completion_kwargs.pop("model", None)
        if model is None:
            raise ValueError("Must provide a model")

        self._model = model
        self._completion_kwargs = completion_kwargs

    @property
    def model(self) -> tp.Optional[str]:
        return self._model

    @property
    def completion_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `litellm.completion`."""
        return self._completion_kwargs

    @classmethod
    def prepare_messages(cls, messages: tp.Union[str, tp.List[dict]]) -> tp.List[dict]:
        """Prepare messages."""
        if isinstance(messages, str):
            messages = [dict(role="user", content=messages)]
        return messages

    def get_chat_response(self, messages: tp.Union[str, tp.List[dict]]) -> ModelResponseT:
        from litellm import completion

        return completion(
            messages=self.prepare_messages(messages),
            model=self.model,
            stream=False,
            **self.completion_kwargs,
        )

    def get_message_content(self, response: ModelResponseT) -> tp.Optional[str]:
        return response.choices[0].message.content

    def get_stream_response(self, messages: tp.Union[str, tp.List[dict]]) -> CustomStreamWrapperT:
        from litellm import completion

        return completion(
            messages=self.prepare_messages(messages),
            model=self.model,
            stream=True,
            **self.completion_kwargs,
        )

    def get_delta_content(self, response_chunk: ModelResponseT) -> tp.Optional[str]:
        return response_chunk.choices[0].delta.content


class LlamaIndexEngine(ChatEngine):
    """Chat engine class for LlamaIndex."""

    _short_name = "llama_index"

    _settings_path: tp.SettingsPath = "knowledge.chat.engines.llama_index"

    def __init__(self, **kwargs) -> None:
        ChatEngine.__init__(self, **kwargs)

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("llama_index")
        from llama_index.core.llms import LLM

        llama_index_config = merge_dicts(self.get_settings(), kwargs)
        llm = llama_index_config.pop("llm", None)
        if llm is None:
            raise ValueError("Must provide an LLM name or path")
        if isinstance(llm, str):
            from importlib import import_module
            from vectorbtpro.utils.module_ import search_package

            if "." in llm:
                path_parts = llm.split(".")
                llm = path_parts[-1]
                if len(path_parts) == 2:
                    module_name = "llama_index.llms." + path_parts[0]
                else:
                    module_name = ".".join(path_parts[:-1])
                module = import_module(module_name)
            else:
                import llama_index.llms

                module = llama_index.llms
            match_func = lambda k, v: isinstance(v, type) and issubclass(v, LLM)
            candidates = search_package(module, match_func)
            class_found = False
            for k, v in candidates.items():
                if llm.lower().replace("_", "") == k.lower():
                    llm = v
                    class_found = True
                    break
            if not class_found:
                raise ValueError(f"LLM '{llm}' not found")
        if isinstance(llm, type):
            llm_name = llm.__name__.lower()
            module_name = llm.__module__
        else:
            checks.assert_instance_of(llm, LLM, arg_name="llm")
            llm_name = type(llm).__name__.lower()
            module_name = type(llm).__module__
        llm_configs = llama_index_config.pop("llm_configs", {})
        if llm_name in llm_configs:
            llama_index_config = deep_merge_dicts(llama_index_config, llm_configs[llm_name])
        elif module_name in llm_configs:
            llama_index_config = deep_merge_dicts(llama_index_config, llm_configs[module_name])
        if isinstance(llm, type):
            llm = llm(**llama_index_config)
        elif len(kwargs) > 0:
            raise ValueError("Cannot apply config to already initialized LLM")
        model = llama_index_config.get("model", None)
        if model is None:
            func_kwargs = get_func_kwargs(type(llm).__init__)
            model = func_kwargs.get("model", None)

        self._model = model
        self._llm = llm

    @property
    def model(self) -> tp.Optional[str]:
        return self._model

    @property
    def llm(self) -> LLMT:
        """LLM."""
        return self._llm

    @classmethod
    def prepare_messages(cls, messages: tp.Union[str, tp.List[dict]]) -> tp.List[ChatMessageT]:
        """Prepare messages."""
        from llama_index.core.llms import ChatMessage

        if isinstance(messages, str):
            messages = [dict(role="user", content=messages)]
        return list(map(lambda x: ChatMessage(**dict(x)), messages))

    def get_chat_response(self, messages: tp.Union[str, tp.List[dict]]) -> ChatResponseT:
        return self.llm.chat(self.prepare_messages(messages))

    def get_message_content(self, response: ChatResponseT) -> tp.Optional[str]:
        return response.message.content

    def get_stream_response(self, messages: tp.Union[str, tp.List[dict]]) -> tp.Generator[ChatResponseT, None, None]:
        return self.llm.stream_chat(self.prepare_messages(messages))

    def get_delta_content(self, response_chunk: ChatResponseT) -> tp.Optional[str]:
        return response_chunk.delta


# ############# Content formatters ############# #


ContentFormatterT = tp.TypeVar("ContentFormatterT", bound="ContentFormatter")


class ContentFormatter(Configured):
    """Class for formatting content."""

    _display_format: tp.ClassVar[tp.Optional[str]] = None
    """Display format."""

    _settings_path: tp.SettingsPath = "knowledge"

    def __init__(
        self,
        output_to: tp.Optional[tp.Union[str, tp.TextIO]] = None,
        flush_output: tp.Optional[bool] = None,
        buffer_output: tp.Optional[bool] = None,
        close_output: tp.Optional[bool] = None,
        update_interval: tp.Optional[float] = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            output_to=output_to,
            flush_output=flush_output,
            buffer_output=buffer_output,
            close_output=close_output,
            update_interval=update_interval,
            **kwargs,
        )

        output_to = self.resolve_setting(output_to, "output_to", sub_path="chat")
        flush_output = self.resolve_setting(flush_output, "flush_output", sub_path="chat")
        buffer_output = self.resolve_setting(buffer_output, "buffer_output", sub_path="chat")
        close_output = self.resolve_setting(close_output, "close_output", sub_path="chat")
        update_interval = self.resolve_setting(update_interval, "update_interval", sub_path="chat")

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

    def __enter__(self: ContentFormatterT) -> ContentFormatterT:
        self.initialize()
        return self

    def __exit__(self, *args) -> None:
        self.finalize()


class PlainFormatter(ContentFormatter):
    """Class for formatting plain content.

    Used as `display_format="plain"` in `Contextable.chat`."""

    _display_format = "plain"

    def buffer_update(self) -> None:
        print("".join(self.buffer), end="")


class IPythonFormatter(ContentFormatter):
    """Class for formatting plain content in IPython.

    Used as `display_format="ipython"` in `Contextable.chat`."""

    _display_format = "ipython"

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

    Used as `display_format="ipython_markdown"` in `Contextable.chat`."""

    _display_format = "ipython_markdown"

    def __init__(self, *args, to_markdown_kwargs: tp.KwargsLike = None, **kwargs) -> None:
        IPythonFormatter.__init__(
            self,
            *args,
            to_markdown_kwargs=to_markdown_kwargs,
            **kwargs,
        )

        to_markdown_kwargs = self.resolve_setting(to_markdown_kwargs, "to_markdown_kwargs", merge=True, sub_path="chat")

        self._to_markdown_kwargs = to_markdown_kwargs

    @property
    def to_markdown_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToMarkdownAssetFunc.to_markdown`."""
        return self._to_markdown_kwargs

    def update_display(self) -> None:
        from vectorbtpro.utils.knowledge.custom_asset_funcs import ToMarkdownAssetFunc
        from IPython.display import Markdown

        markdown_content = ToMarkdownAssetFunc.to_markdown(self.content, **self.to_markdown_kwargs)
        self.display_handle.update(Markdown(markdown_content))


class IPythonHTMLFormatter(IPythonFormatter):
    """Class for formatting HTML content in IPython.

    Used as `display_format="ipython_html"` in `Contextable.chat`."""

    _display_format = "ipython_html"

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

        to_markdown_kwargs = self.resolve_setting(to_markdown_kwargs, "to_markdown_kwargs", merge=True, sub_path="chat")
        to_html_kwargs = self.resolve_setting(to_html_kwargs, "to_html_kwargs", merge=True, sub_path="chat")

        self._to_markdown_kwargs = to_markdown_kwargs
        self._to_html_kwargs = to_html_kwargs

    @property
    def to_markdown_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToMarkdownAssetFunc.to_markdown`."""
        return self._to_markdown_kwargs

    @property
    def to_html_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc.to_html`."""
        return self._to_html_kwargs

    def update_display(self) -> None:
        from vectorbtpro.utils.knowledge.custom_asset_funcs import ToMarkdownAssetFunc, ToHTMLAssetFunc
        from IPython.display import HTML

        markdown_content = ToMarkdownAssetFunc.to_markdown(self.content, **self.to_markdown_kwargs)
        html_content = ToHTMLAssetFunc.to_html(markdown_content, **self.to_html_kwargs)
        self.display_handle.update(HTML(html_content))


class HTMLFileFormatter(ContentFormatter):
    """Class for formatting static HTML files.

    Used as `display_format="html"` in `Contextable.chat`."""

    _display_format = "html"

    def __init__(
        self,
        *args,
        page_title: str = "",
        refresh_page: tp.Optional[bool] = None,
        cache: tp.Optional[bool] = None,
        cache_dir: tp.Optional[tp.PathLike] = None,
        cache_mkdir_kwargs: tp.KwargsLike = None,
        clear_cache: tp.Optional[bool] = None,
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
            cache=cache,
            cache_dir=cache_dir,
            cache_mkdir_kwargs=cache_mkdir_kwargs,
            clear_cache=clear_cache,
            file_prefix_len=file_prefix_len,
            file_suffix_len=file_suffix_len,
            open_browser=open_browser,
            to_markdown_kwargs=to_markdown_kwargs,
            to_html_kwargs=to_html_kwargs,
            format_html_kwargs=format_html_kwargs,
            **kwargs,
        )

        refresh_page = self.resolve_setting(refresh_page, "refresh_page", sub_path="chat")
        cache = self.resolve_setting(cache, "cache", sub_path="chat")
        cache_dir = self.resolve_setting(cache_dir, "cache_dir", sub_path="chat")
        cache_mkdir_kwargs = self.resolve_setting(cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True, sub_path="chat")
        clear_cache = self.resolve_setting(clear_cache, "clear_cache", sub_path="chat")
        file_prefix_len = self.resolve_setting(file_prefix_len, "file_prefix_len", sub_path="chat")
        file_suffix_len = self.resolve_setting(file_suffix_len, "file_suffix_len", sub_path="chat")
        open_browser = self.resolve_setting(open_browser, "open_browser", sub_path="chat")
        to_markdown_kwargs = self.resolve_setting(to_markdown_kwargs, "to_markdown_kwargs", merge=True, sub_path="chat")
        to_html_kwargs = self.resolve_setting(to_html_kwargs, "to_html_kwargs", merge=True, sub_path="chat")
        format_html_kwargs = self.resolve_setting(format_html_kwargs, "format_html_kwargs", merge=True, sub_path="chat")

        self._page_title = page_title
        self._refresh_page = refresh_page
        self._cache = cache
        self._cache_dir = cache_dir
        self._cache_mkdir_kwargs = cache_mkdir_kwargs
        self._clear_cache = clear_cache
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
    def cache(self) -> bool:
        """Whether to store the HTML file in the cache directory or as a temporary file."""
        return self._cache

    @property
    def cache_dir(self) -> tp.PathLike:
        """Path to the cache directory.

        Files will be stored in the subdirectory "chat"."""
        return self._cache_dir

    @property
    def cache_mkdir_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.path_.check_mkdir`."""
        return self._cache_mkdir_kwargs

    @property
    def clear_cache(self) -> bool:
        """Whether to remove the cache directory."""
        return self._clear_cache

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
        """Keyword arguments passed to
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToMarkdownAssetFunc.to_markdown`."""
        return self._to_markdown_kwargs

    @property
    def to_html_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc.to_html`."""
        return self._to_html_kwargs

    @property
    def format_html_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc.format_html`."""
        return self._format_html_kwargs

    @property
    def file_handle(self) -> tp.Optional[tp.TextIO]:
        """File handle."""
        return self._file_handle

    def initialize(self) -> None:
        ContentFormatter.initialize(self)

        if self.cache:
            import secrets
            import string

            html_dir = Path(self.cache_dir) / "chat"
            if html_dir.exists():
                if self.clear_cache:
                    remove_dir(html_dir, missing_ok=True, with_contents=True)
            check_mkdir(html_dir, **self.cache_mkdir_kwargs)
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
            short_filename = f"{truncated_page_title}-{random_suffix}.html"
            file_path = html_dir / short_filename
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

        from vectorbtpro.utils.knowledge.custom_asset_funcs import ToMarkdownAssetFunc, ToHTMLAssetFunc

        markdown_content = ToMarkdownAssetFunc.to_markdown(self.content, **self.to_markdown_kwargs)
        html_content = ToHTMLAssetFunc.to_html(markdown_content, **self.to_html_kwargs)
        if not final and self.refresh_page:
            refresh_content = max(1, int(self.update_interval)) if self.update_interval is not None else 1
            _format_html_kwargs = dict(self.format_html_kwargs)
            head_extras = _format_html_kwargs["head_extras"]
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
        html = ToHTMLAssetFunc.format_html(
            title=self.page_title,
            html_content=html_content,
            **_format_html_kwargs,
        )
        self.file_handle.seek(0)
        self.file_handle.write(html)
        self.file_handle.truncate()
        self.file_handle.flush()


# ############# Contextable class ############# #


class Contextable(HasSettings):
    """Abstract class that can be converted into a context."""

    _settings_path: tp.SettingsPath = "knowledge"

    def to_context(self, *args, **kwargs) -> str:
        """Convert to a context."""
        raise NotImplementedError

    @classmethod
    def resolve_encoding(
        cls,
        tokenizer: tp.Union[None, str, EncodingT] = None,
        model: tp.Optional[str] = None,
    ) -> EncodingT:
        """Resolve the encoding."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("tiktoken")
        from tiktoken import Encoding

        tokenizer = cls.resolve_setting(tokenizer, "tokenizer", sub_path="chat")
        if isinstance(tokenizer, str):
            from tiktoken import get_encoding, encoding_for_model

            if tokenizer.startswith("model_or_"):
                try:
                    if model is None:
                        raise KeyError
                    encoding = encoding_for_model(model)
                except KeyError:
                    tokenizer = tokenizer[len("model_or_") :]
                    encoding = get_encoding(tokenizer) if "k_base" in tokenizer else encoding_for_model(tokenizer)
            elif isinstance(tokenizer, str):
                encoding = get_encoding(tokenizer) if "k_base" in tokenizer else encoding_for_model(tokenizer)
            else:
                encoding = tokenizer
        else:
            encoding = tokenizer
        checks.assert_instance_of(encoding, Encoding, arg_name="tokenizer")
        return encoding

    @classmethod
    def count_tokens_in_context(
        cls,
        context: str,
        tokenizer: tp.Union[None, str, EncodingT] = None,
        model: tp.Optional[str] = None,
    ) -> int:
        """Count the number of tokens in a context."""
        encoding = cls.resolve_encoding(tokenizer=tokenizer, model=model)
        return len(encoding.encode(context))

    def count_tokens(
        self,
        to_context_kwargs: tp.KwargsLike = None,
        tokenizer: tp.Union[None, str, EncodingT] = None,
        model: tp.Optional[str] = None,
    ) -> int:
        """Count the number of tokens in the context."""
        to_context_kwargs = self.resolve_setting(to_context_kwargs, "to_context_kwargs", merge=True, sub_path="chat")
        context = self.to_context(**to_context_kwargs)
        encoding = self.resolve_encoding(tokenizer=tokenizer, model=model)
        return len(encoding.encode(context))

    @classmethod
    def count_tokens_in_messages(
        cls,
        messages: tp.Union[str, tp.List[dict]],
        tokenizer: tp.Union[None, str, EncodingT] = None,
        tokens_per_message: tp.Optional[int] = None,
        tokens_per_name: tp.Optional[int] = None,
        model: tp.Optional[str] = None,
    ) -> int:
        """Count the number of tokens in messages."""
        encoding = cls.resolve_encoding(tokenizer=tokenizer, model=model)
        tokens_per_message = cls.resolve_setting(tokens_per_message, "tokens_per_message", sub_path="chat")
        tokens_per_name = cls.resolve_setting(tokens_per_name, "tokens_per_name", sub_path="chat")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def build_messages(
        self,
        message: str,
        chat_history: tp.ChatHistory = None,
        to_context_kwargs: tp.KwargsLike = None,
        max_tokens: tp.Optional[int] = None,
        tokenizer: tp.Union[None, str, EncodingT] = None,
        tokens_per_message: tp.Optional[int] = None,
        tokens_per_name: tp.Optional[int] = None,
        model: tp.Optional[str] = None,
        system_prompt: tp.Optional[str] = None,
        system_as_user: tp.Optional[bool] = None,
        context_prompt: tp.Optional[str] = None,
        template_context: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.Union[str, tp.List[dict]]:
        """Build messages for chatting."""
        if chat_history is None:
            chat_history = []
        to_context_kwargs = self.resolve_setting(to_context_kwargs, "to_context_kwargs", merge=True, sub_path="chat")
        max_tokens = self.resolve_setting(max_tokens, "max_tokens", sub_path="chat")
        system_prompt = self.resolve_setting(system_prompt, "system_prompt", sub_path="chat")
        system_as_user = self.resolve_setting(system_as_user, "system_as_user", sub_path="chat")
        context_prompt = self.resolve_setting(context_prompt, "context_prompt", sub_path="chat")
        template_context = self.resolve_setting(template_context, "template_context", merge=True, sub_path="chat")
        silence_warnings = self.resolve_setting(silence_warnings, "silence_warnings", sub_path="chat")

        if isinstance(context_prompt, str):
            context_prompt = Sub(context_prompt)
        elif checks.is_function(context_prompt):
            context_prompt = RepFunc(context_prompt)
        elif not isinstance(context_prompt, CustomTemplate):
            raise TypeError(f"Context prompt must be a string, function, or template")
        context = self.to_context(**to_context_kwargs)
        if max_tokens is not None:
            empty_context_prompt = context_prompt.substitute(
                flat_merge_dicts(dict(context=""), template_context),
                eval_id="context_prompt",
            )
            empty_messages = [
                dict(role="user" if system_as_user else "system", content=system_prompt),
                dict(role="user", content=empty_context_prompt),
                *chat_history,
                dict(role="user", content=message),
            ]
            num_tokens = self.count_tokens_in_messages(
                empty_messages,
                tokenizer=tokenizer,
                tokens_per_message=tokens_per_message,
                tokens_per_name=tokens_per_name,
                model=model,
            )
            max_context_tokens = max(0, max_tokens - num_tokens)
            encoding = self.resolve_encoding(tokenizer=tokenizer, model=model)
            encoded_context = encoding.encode(context)
            if len(encoded_context) > max_context_tokens:
                context = encoding.decode(encoded_context[:max_context_tokens])
                if not silence_warnings:
                    warnings.warn(
                        f"Context is too long ({len(encoded_context)}). "
                        f"Truncating to {max_context_tokens} tokens. "
                        f"Pass silence_warnings=True to silence this warning.",
                        stacklevel=2,
                    )
        template_context = flat_merge_dicts(dict(context=context), template_context)
        context_prompt = context_prompt.substitute(template_context, eval_id="context_prompt")
        return [
            dict(role="user" if system_as_user else "system", content=system_prompt),
            dict(role="user", content=context_prompt),
            *chat_history,
            dict(role="user", content=message),
        ]

    @hybrid_method
    def chat(
        cls_or_self,
        message: str,
        chat_history: tp.Optional[tp.MutableSequence[str]] = None,
        stream: tp.Optional[bool] = None,
        engine: tp.Union[None, str, ChatEngine] = None,
        to_context_kwargs: tp.KwargsLike = None,
        max_tokens: tp.Optional[int] = None,
        tokenizer: tp.Union[None, str, EncodingT] = None,
        tokens_per_message: tp.Optional[int] = None,
        tokens_per_name: tp.Optional[int] = None,
        system_prompt: tp.Optional[str] = None,
        system_as_user: tp.Optional[bool] = None,
        context_prompt: tp.Optional[str] = None,
        display_format: tp.Union[None, str, ContentFormatter] = None,
        output_to: tp.Optional[tp.Union[str, tp.TextIO]] = None,
        flush_output: tp.Optional[bool] = None,
        buffer_output: tp.Optional[bool] = None,
        close_output: tp.Optional[bool] = None,
        update_interval: tp.Optional[float] = None,
        refresh_page: tp.Optional[bool] = None,
        cache: tp.Optional[bool] = None,
        cache_dir: tp.Optional[tp.PathLike] = None,
        cache_mkdir_kwargs: tp.KwargsLike = None,
        clear_cache: tp.Optional[bool] = None,
        file_prefix_len: tp.Optional[int] = None,
        file_suffix_len: tp.Optional[int] = None,
        open_browser: tp.Optional[bool] = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        format_html_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
        return_response: bool = False,
        **kwargs,
    ) -> tp.ChatOutput:
        """Chat with an LLM using LlamaIndex and the dumped asset as a context.

        The following engines are supported:

        * "openai" (`OpenAIEngine`)
        * "litellm" (`LiteLLMEngine`)
        * "llama_index" (`LlamaIndexEngine`)
        * A subclass of `ChatEngine`

        For LlamaIndex, LLM can be provided via `llm`, which can be either the name of the class,
        the path to the class (accepted are both "llama_index.xxx.yyy" and "xxx.yyy"), or a subclass
        or an instance of `llama_index.core.llms.LLM`. Case of strings doesn't matter.

        Uses `context_prompt` as a prompt template requiring the variable "context".
        The prompt can be either a custom template, or string or function that will become one.
        The context itself is generated by dumping and joining data items with `Contextable.to_context`
        with `to_context_kwargs` as keyword arguments. Once the prompt is evaluated, it becomes a system message.
        Uses `system_prompt` as a system prompt preceding the context prompt. Enable `system_as_user`
        to use the user role for the system message (for experimental models where the system role is not available).

        Use `max_context_chars` to limit the number of characters in the context. Use `max_context_tokens`
        to limit the number of tokens in the context (requires tiktoken). Use `tokenizer` to provide
        a model name, an encoding name, or an encoding object for tokenization.

        Pass `chat_history` as a mutable sequence (for example, list) to keep track of chat history.
        After generating a response, the output will be appended to this sequence as an assistant message.

        Argument `message` becomes a user message.

        If the output is a stream (`stream=True`), appends chunks one by one and displays the
        intermediate result. Otherwise, displays the entire message.

        The following display formats are supported:

        * "plain" (`PlainFormatter`): Prints the raw output
        * "ipython" (`IPythonFormatter`): Renders an unformatted text in a notebook environment
        * "ipython_markdown" (`IPythonMarkdownFormatter`): Renders a Markdown in a notebook environment
        * "ipython_html" (`IPythonHTMLFormatter`): Renders an HTML in a notebook environment
        * "ipython_auto": Decides between using "ipython_html" or "plain" depending on the environment
        * "html" (`HTMLFileFormatter`): Writes a static HTML page and displays it in the browser
        * A subclass of `ContentFormatter`

        If `update_interval` is None, will update as soon as the next chunk arrives (apart from refreshing
        HTML pages, here the minimum refresh rate is once per second). Otherwise, will collect chunks and
        update once in `update_interval` seconds.

        To convert to Markdown, uses
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToMarkdownAssetFunc.to_markdown`
        along with `to_markdown_kwargs`. To convert to HTML, uses
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc.to_html`
        along with `to_html_kwargs`. To create and format the HTML page, uses
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc.format_html`
        along with `format_html_kwargs`.

        The HTML file is stored either in the cache directory (`cache=True`)
        under "chat" and with truncated title as prefix and random hash as suffix, or as a
        temporary file (`cache=False`). If `clear_cache` is True, deletes any existing directory
        before creating a new one. Returns the path of the directory where HTML file is stored.

        The raw output can be also redirected to a file (or any IO) specified in `output_to`
        and flushed with each chunk if `flush_output` is True. If `buffer_output` is True, will
        respect `update_interval`, otherwise, will push any new content right away.

        Keyword arguments are passed to the engine.

        For defaults, see `chat` in `vectorbtpro._settings.knowledge`.

        Usage:
            ```pycon
            >>> asset.chat("What's the value under 'xyz'?")
            The value under 'xyz' is 123.

            >>> chat_history = []
            >>> asset.chat("What's the value under 'xyz'?", chat_history=chat_history)
            The value under 'xyz' is 123.

            >>> asset.chat("Are you sure?", chat_history=chat_history)
            Yes, I am sure. The value under 'xyz' is 123 for the entry where `s` is "EFG".
            ```
        """
        if isinstance(cls_or_self, type):
            from vectorbtpro.utils.parsing import get_forward_args

            args, kwargs = get_forward_args(super().chat, locals())
            return super().chat(*args, **kwargs)

        stream = cls_or_self.resolve_setting(stream, "stream", sub_path="chat")
        engine = cls_or_self.resolve_setting(engine, "engine", sub_path="chat")
        display_format = cls_or_self.resolve_setting(display_format, "display_format", sub_path="chat")
        output_to = cls_or_self.resolve_setting(output_to, "output_to", sub_path="chat")
        flush_output = cls_or_self.resolve_setting(flush_output, "flush_output", sub_path="chat")
        buffer_output = cls_or_self.resolve_setting(buffer_output, "buffer_output", sub_path="chat")
        close_output = cls_or_self.resolve_setting(close_output, "close_output", sub_path="chat")
        update_interval = cls_or_self.resolve_setting(update_interval, "update_interval", sub_path="chat")
        refresh_page = cls_or_self.resolve_setting(refresh_page, "refresh_page", sub_path="chat")
        cache = cls_or_self.resolve_setting(cache, "cache", sub_path="chat")
        cache_dir = cls_or_self.resolve_setting(cache_dir, "cache_dir", sub_path="chat")
        cache_mkdir_kwargs = cls_or_self.resolve_setting(
            cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True, sub_path="chat"
        )
        file_prefix_len = cls_or_self.resolve_setting(file_prefix_len, "file_prefix_len", sub_path="chat")
        file_suffix_len = cls_or_self.resolve_setting(file_suffix_len, "file_suffix_len", sub_path="chat")
        open_browser = cls_or_self.resolve_setting(open_browser, "open_browser", sub_path="chat")
        to_markdown_kwargs = cls_or_self.resolve_setting(
            to_markdown_kwargs, "to_markdown_kwargs", merge=True, sub_path="chat"
        )
        to_html_kwargs = cls_or_self.resolve_setting(to_html_kwargs, "to_html_kwargs", merge=True, sub_path="chat")
        format_html_kwargs = cls_or_self.resolve_setting(
            format_html_kwargs, "format_html_kwargs", merge=True, sub_path="chat"
        )
        template_context = cls_or_self.resolve_setting(
            template_context, "template_context", merge=True, sub_path="chat"
        )
        silence_warnings = cls_or_self.resolve_setting(silence_warnings, "silence_warnings", sub_path="chat")

        if isinstance(engine, str):
            if engine.lower() == "auto":
                from vectorbtpro.utils.module_ import check_installed

                if check_installed("openai"):
                    engine = "openai"
                elif check_installed("litellm"):
                    engine = "litellm"
                elif check_installed("llama_index"):
                    engine = "llama_index"
                else:
                    raise ValueError("No LLM packages installed")
            current_module = sys.modules[__name__]
            found_engine = False
            for name, cls in inspect.getmembers(current_module, inspect.isclass):
                if name.endswith("Engine"):
                    short_name = getattr(cls, "_short_name", None)
                    if short_name is not None and short_name.lower() == engine.lower():
                        engine = cls
                        found_engine = True
                        break
            if not found_engine:
                raise ValueError(f"Invalid chat engine: '{engine}'")
        if isinstance(engine, type):
            checks.assert_subclass_of(engine, ChatEngine, arg_name="engine")
            if engine._short_name is not None:
                kwargs = cls_or_self.resolve_setting(
                    kwargs, "engines." + engine._short_name, merge=True, sub_path="chat"
                )
            engine = engine(**kwargs)
        else:
            checks.assert_instance_of(engine, ChatEngine, arg_name="engine")
        if chat_history is None:
            chat_history = []
        messages = cls_or_self.build_messages(
            message,
            chat_history=chat_history,
            to_context_kwargs=to_context_kwargs,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
            tokens_per_message=tokens_per_message,
            tokens_per_name=tokens_per_name,
            model=engine.model,
            system_prompt=system_prompt,
            system_as_user=system_as_user,
            context_prompt=context_prompt,
            template_context=template_context,
            silence_warnings=silence_warnings,
        )
        if stream:
            response = engine.get_stream_response(messages)
        else:
            response = engine.get_chat_response(messages)

        if isinstance(display_format, str):
            if display_format.lower() == "ipython_auto":
                if checks.in_notebook():
                    display_format = "ipython_html"
                else:
                    display_format = "plain"
            current_module = sys.modules[__name__]
            found_format = False
            for name, cls in inspect.getmembers(current_module, inspect.isclass):
                if name.endswith("Formatter"):
                    _display_format = getattr(cls, "_display_format", None)
                    if _display_format is not None and _display_format.lower() == display_format.lower():
                        display_format = cls
                        found_format = True
                        break
            if not found_format:
                raise ValueError(f"Invalid display format: '{display_format}'")
        if isinstance(display_format, type):
            checks.assert_subclass_of(display_format, ContentFormatter, arg_name="display_format")
            method_locals = locals()
            init_kwargs = {}
            for k in display_format._expected_keys:
                if k in method_locals:
                    init_kwargs[k] = method_locals[k]
            display_format = display_format(**init_kwargs)
        else:
            checks.assert_instance_of(display_format, ContentFormatter, "display_format")

        if stream:
            with display_format:
                for i, response_chunk in enumerate(response):
                    new_content = engine.get_delta_content(response_chunk)
                    if new_content is not None:
                        display_format.append(new_content)
                content = display_format.content
        else:
            content = engine.get_message_content(response)
            if content is None:
                content = ""
            display_format.append_once(content)
        chat_history.append(dict(role="user", content=message))
        chat_history.append(dict(role="assistant", content=content))

        if isinstance(display_format, HTMLFileFormatter) and display_format.file_handle is not None:
            file_path = Path(display_format.file_handle.name)
        else:
            file_path = None
        if return_response:
            return response, file_path
        return file_path
