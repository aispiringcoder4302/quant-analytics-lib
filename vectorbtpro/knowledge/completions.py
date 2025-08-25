# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes and utilities for completions."""

import inspect
import sys
import time
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.knowledge.formatting import ContentFormatter, HTMLFileFormatter, resolve_formatter
from vectorbtpro.knowledge.tokenization import Tokenizer, TikTokenizer, resolve_tokenizer
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, flat_merge_dicts, Configured
from vectorbtpro.utils.parsing import get_func_arg_names, get_func_kwargs
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.template import CustomTemplate, SafeSub, RepFunc
from vectorbtpro.utils.warnings_ import warn

if tp.TYPE_CHECKING:
    from openai import OpenAI as OpenAIT, Stream as StreamT
    from openai.types.chat.chat_completion import ChatCompletion as ChatCompletionT
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as ChatCompletionChunkT
    from openai.types.responses import Response as ResponseT, ResponseStreamEvent as ResponseStreamEventT
else:
    OpenAIT = "openai.OpenAI"
    StreamT = "openai.Stream"
    ChatCompletionT = "openai.types.chat.chat_completion.ChatCompletion"
    ChatCompletionChunkT = "openai.types.chat.chat_completion_chunk.ChatCompletionChunk"
    ResponseT = "openai.types.responses.Response"
    ResponseStreamEventT = "openai.types.responses.ResponseStreamEvent"
if tp.TYPE_CHECKING:
    from anthropic import Client as AnthropicClientT, Stream as AnthropicStreamT
    from anthropic.types import Message as AnthropicMessageT, MessageStreamEvent as AnthropicMessageStreamEventT
else:
    AnthropicClientT = "anthropic.Client"
    AnthropicStreamT = "anthropic.Stream"
    AnthropicMessageT = "anthropic.types.Message"
    AnthropicMessageStreamEventT = "anthropic.types.MessageStreamEvent"
if tp.TYPE_CHECKING:
    from google.genai import Client as GenAIClientT
    from google.genai.types import Content as ContentT, GenerateContentResponse as GenerateContentResponseT
else:
    GenAIClientT = "google.genai.Client"
    ContentT = "google.genai.types.Content"
    GenerateContentResponseT = "google.genai.types.GenerateContentResponse"
if tp.TYPE_CHECKING:
    from huggingface_hub import (
        InferenceClient as InferenceClientT,
        ChatCompletionOutput as ChatCompletionOutputT,
        ChatCompletionStreamOutput as ChatCompletionStreamOutputT,
    )
else:
    InferenceClientT = "huggingface_hub.InferenceClient"
    ChatCompletionOutputT = "huggingface_hub.ChatCompletionOutput"
    ChatCompletionStreamOutputT = "huggingface_hub.ChatCompletionStreamOutput"
if tp.TYPE_CHECKING:
    from litellm import ModelResponse as ModelResponseT, CustomStreamWrapper as CustomStreamWrapperT
else:
    ModelResponseT = "litellm.ModelResponse"
    CustomStreamWrapperT = "litellm.CustomStreamWrapper"
if tp.TYPE_CHECKING:
    from llama_index.core.llms import LLM as LLMT, ChatResponse as ChatResponseT
else:
    LLMT = "llama_index.core.llms.LLM"
    ChatResponseT = "llama_index.core.llms.ChatResponse"
if tp.TYPE_CHECKING:
    from ollama import Client as OllamaClientT, ChatResponse as OllamaChatResponseT
else:
    OllamaClientT = "ollama.Client"
    OllamaChatResponseT = "ollama.ChatResponse"

__all__ = [
    "Completions",
    "OpenAICompletions",
    "AnthropicCompletions",
    "GeminiCompletions",
    "HFInferenceCompletions",
    "LiteLLMCompletions",
    "LlamaIndexCompletions",
    "OllamaCompletions",
    "complete",
    "completed",
]


class ThoughtProcessor:
    """Processes content that may include `<think>...</think>` segments.

    Works for both streaming and non-streaming inputs. When `include_thoughts` is
    True, tags and their content pass through unchanged. When False, thought regions
    and tags are removed.

    Args:
        include_thoughts (bool): Whether to keep or remove thought regions.
    """

    OPEN_TAG = "<think>"
    """Opening tag for thought segments."""

    CLOSE_TAG = "</think>"
    """Closing tag for thought segments."""

    def __init__(self, include_thoughts: bool = True) -> None:
        self._include_thoughts = include_thoughts
        self._thinking = False
        self._thinking_source = None

    @property
    def include_thoughts(self) -> bool:
        """Whether to include thinking messages (wrapped in `<think>` tags) in output.

        Returns:
            bool: True if thinking messages should be included, False otherwise.
        """
        return self._include_thoughts

    def reset_thought_state(self) -> None:
        """Resets the internal streaming state.

        Clears any active thought segment without emitting a closing tag. Call this
        before starting a new, unrelated stream if reusing the same instance.
        """
        self._thinking = False
        self._thinking_source = None

    def process_thought(
        self,
        *,
        thought: tp.Optional[str] = None,
        content: tp.Optional[str] = None,
        flush: bool = False,
    ) -> tp.Optional[str]:
        """Processes a unit of input for both streaming and non-streaming use.

        Accepts `thought`, `content`, both, or neither. When both are provided
        and no thought is currently open, the output equals `<think>{thought}</think>{content}`
        if `include_thoughts` is True, or just `{content}` if False.

        Stateful behavior:

        * Explicit thoughts (`thought`) open an explicit segment on first call,
        and close automatically when a plain `content` chunk without tags
        is processed, or when `ThoughtProcessor.process_thought` is called with neither argument.
        * Inline `<think>`/`</think>` tokens inside `content` are handled
        even if they span multiple calls.

        Args:
            thought (Optional[str]): Reasoning content from a separate channel (explicit thoughts).
            content (Optional[str]): Natural language content that may contain inline tags.
            flush (bool): If True, flushes the current thought state, closing any open thought segment.

        Returns:
            Optional[str]: The output string for this call, or None if nothing should be emitted.
        """
        out = None

        if thought is not None:
            if not self._thinking:
                self._thinking = True
                self._thinking_source = "explicit"
                if self.include_thoughts:
                    out = (out or "") + self.OPEN_TAG + (thought or "")
            else:
                if self.include_thoughts:
                    out = (out or "") + (thought or "")

        if content is not None:
            s = content or ""
            if (
                self._thinking
                and self._thinking_source == "explicit"
                and s != ""
                and (self.OPEN_TAG not in s and self.CLOSE_TAG not in s)
            ):
                self._thinking = False
                self._thinking_source = None
                piece = (self.CLOSE_TAG + s) if self.include_thoughts else s
                out = (out or "") + piece if piece else out
            else:

                def _parse_inline(chunk):
                    i = 0
                    parts = []
                    while i < len(chunk):
                        if not self._thinking:
                            j = chunk.find(self.OPEN_TAG, i)
                            if j == -1:
                                parts.append(chunk[i:])
                                break
                            pre = chunk[i:j]
                            if pre:
                                parts.append(pre)
                            self._thinking = True
                            self._thinking_source = "inline"
                            i = j + len(self.OPEN_TAG)
                            if self.include_thoughts:
                                parts.append(self.OPEN_TAG)
                        else:
                            k = chunk.find(self.CLOSE_TAG, i)
                            if k == -1:
                                seg = chunk[i:]
                                if self.include_thoughts and seg:
                                    parts.append(seg)
                                break
                            thought_piece = chunk[i:k]
                            if self.include_thoughts and thought_piece:
                                parts.append(thought_piece)
                            self._thinking = False
                            self._thinking_source = None
                            i = k + len(self.CLOSE_TAG)
                            if self.include_thoughts:
                                parts.append(self.CLOSE_TAG)
                    return "".join(parts)

                piece = _parse_inline(s)
                out = (out or "") + piece if piece else out

        if ((thought is None and content is None) or flush) and self._thinking:
            self._thinking = False
            self._thinking_source = None
            if self.include_thoughts:
                out = (out or "") + self.CLOSE_TAG

        return out if out not in ("", None) else None

    def flush_thought(self) -> tp.Optional[str]:
        """Flushes the current thought state, closing any open thought segment.

        Returns:
            Optional[str]: The output string for this call, or None if nothing should be emitted.
        """
        return self.process_thought(flush=True)


class Completions(ThoughtProcessor, Configured):
    """Abstract class for completion providers.

    Args:
        context (str): Context string to be used as a user message.
        chat_history (Optional[ChatHistory]): Chat history, a list of dictionaries with defined roles.

            After a response is generated, the assistant message is appended to this history.
        stream (Optional[bool]): Boolean indicating whether responses are streamed.

            In streaming mode, chunks are appended and displayed incrementally; otherwise,
            the entire message is displayed.
        max_tokens (Union[None, bool, int]): Maximum token limit configured for messages.

            If False, the limit is disabled.
        tokenizer (TokenizerLike): Identifier, subclass, or instance of `vectorbtpro.knowledge.tokenization.Tokenizer`.

            Resolved using `vectorbtpro.knowledge.tokenization.resolve_tokenizer`.
        tokenizer_kwargs (KwargsLike): Keyword arguments to initialize or update `tokenizer`.
        system_prompt (Optional[str]): System prompt that precedes the context prompt.

            This prompt is used to set the system's behavior or context for the conversation.
        system_as_user (Optional[bool]): Boolean indicating whether to use the user role for the system message.

            This is mainly used for experimental models where a dedicated system role is not available.
        context_template (Optional[str]): Context template requiring a 'context' variable.

            The template can be a string, a function, or an instance of `vectorbtpro.utils.template.CustomTemplate`.
        formatter (ContentFormatterLike): Identifier, subclass, or instance of
            `vectorbtpro.knowledge.formatting.ContentFormatter`.

            Resolved using `vectorbtpro.knowledge.formatting.resolve_formatter`.

            This formatter is used to format the content of the response.
        formatter_kwargs (KwargsLike): Keyword arguments to initialize or update `formatter`.
        minimal_format (Optional[bool]): Boolean indicating if the input is minimally formatted.
        quick_mode (Optional[bool]): Boolean indicating whether quick mode is enabled.
        silence_warnings (Optional[bool]): Flag to suppress warning messages.
        show_progress (Optional[bool]): Flag indicating whether to display the progress bar.
        pbar_kwargs (Kwargs): Keyword arguments for configuring the progress bar.
        template_context (KwargsLike): Additional context for template substitution.
        include_thoughts (Optional[bool]): Whether to keep or remove thought regions.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and
        its sub-configurations `chat` and `chat.completions_config`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat", "knowledge.chat.completions_config"]

    def __init__(
        self,
        context: str = "",
        chat_history: tp.Optional[tp.ChatHistory] = None,
        stream: tp.Optional[bool] = None,
        max_tokens: tp.Union[None, bool, int] = None,
        tokenizer: tp.TokenizerLike = None,
        tokenizer_kwargs: tp.KwargsLike = None,
        system_prompt: tp.Optional[str] = None,
        system_as_user: tp.Optional[bool] = None,
        context_template: tp.Optional[str] = None,
        formatter: tp.ContentFormatterLike = None,
        formatter_kwargs: tp.KwargsLike = None,
        minimal_format: tp.Optional[bool] = None,
        quick_mode: tp.Optional[bool] = None,
        silence_warnings: tp.Optional[bool] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        include_thoughts: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            context=context,
            chat_history=chat_history,
            stream=stream,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            system_prompt=system_prompt,
            system_as_user=system_as_user,
            context_template=context_template,
            formatter=formatter,
            formatter_kwargs=formatter_kwargs,
            minimal_format=minimal_format,
            quick_mode=quick_mode,
            silence_warnings=silence_warnings,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            template_context=template_context,
            include_thoughts=include_thoughts,
            **kwargs,
        )

        if chat_history is None:
            chat_history = []
        stream = self.resolve_setting(stream, "stream")
        max_tokens_set = max_tokens is not None
        max_tokens = self.resolve_setting(max_tokens, "max_tokens")
        tokenizer = self.resolve_setting(tokenizer, "tokenizer", default=None)
        tokenizer_kwargs = self.resolve_setting(tokenizer_kwargs, "tokenizer_kwargs", default=None, merge=True)
        system_prompt = self.resolve_setting(system_prompt, "system_prompt")
        system_as_user = self.resolve_setting(system_as_user, "system_as_user")
        context_template = self.resolve_setting(context_template, "context_template")
        formatter = self.resolve_setting(formatter, "formatter", default=None)
        formatter_kwargs = self.resolve_setting(formatter_kwargs, "formatter_kwargs", default=None, merge=True)
        minimal_format = self.resolve_setting(minimal_format, "minimal_format", default=None)
        quick_mode = self.resolve_setting(quick_mode, "quick_mode")
        silence_warnings = self.resolve_setting(silence_warnings, "silence_warnings")
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        template_context = self.resolve_setting(template_context, "template_context", merge=True)
        include_thoughts = self.resolve_setting(include_thoughts, "include_thoughts")

        tokenizer = resolve_tokenizer(tokenizer)
        formatter = resolve_formatter(formatter)

        self._context = context
        self._chat_history = chat_history
        self._stream = stream
        self._max_tokens_set = max_tokens_set
        self._max_tokens = max_tokens
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs
        self._system_prompt = system_prompt
        self._system_as_user = system_as_user
        self._context_template = context_template
        self._formatter = formatter
        self._formatter_kwargs = formatter_kwargs
        self._minimal_format = minimal_format
        self._quick_mode = quick_mode
        self._silence_warnings = silence_warnings
        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs
        self._template_context = template_context

        ThoughtProcessor.__init__(self, include_thoughts=include_thoughts)

    @property
    def context(self) -> str:
        """Context string to be used as a user message.

        Returns:
            str: Context string used for expression evaluation.
        """
        return self._context

    @property
    def chat_history(self) -> tp.ChatHistory:
        """Chat history, a list of dictionaries with defined roles.

        After a response is generated, the assistant message is appended to this history.

        Returns:
            ChatHistory: List of dictionaries representing the chat history.
        """
        return self._chat_history

    @property
    def stream(self) -> bool:
        """Boolean indicating whether responses are streamed.

        In streaming mode, chunks are appended and displayed incrementally; otherwise,
        the entire message is displayed.

        Returns:
            bool: True if streaming is enabled, False otherwise.
        """
        return self._stream

    @property
    def max_tokens_set(self) -> tp.Optional[int]:
        """Boolean indicating if `Completions.max_tokens` was explicitly provided by the user.

        Returns:
            Optional[int]: Maximum token limit set by the user; None if not set.
        """
        return self._max_tokens_set

    @property
    def max_tokens(self) -> tp.Union[bool, int]:
        """Maximum token limit configured for messages.

        Returns:
            Union[bool, int]: Maximum token limit; False if disabled.
        """
        return self._max_tokens

    @property
    def tokenizer(self) -> tp.MaybeType[Tokenizer]:
        """Subclass or instance of `vectorbtpro.knowledge.tokenization.Tokenizer`.

        Resolved using `vectorbtpro.knowledge.tokenization.resolve_tokenizer`.

        Returns:
            MaybeType[Tokenizer]: Resolved tokenizer instance or subclass.
        """
        return self._tokenizer

    @property
    def tokenizer_kwargs(self) -> tp.Kwargs:
        """Keyword arguments to initialize or update `Completions.tokenizer`.

        Returns:
            Kwargs: Keyword arguments for tokenizer initialization or update.
        """
        return self._tokenizer_kwargs

    @property
    def system_prompt(self) -> str:
        """System prompt that precedes the context prompt.

        This prompt is used to set the system's behavior or context for the conversation.

        Returns:
            str: System prompt.
        """
        return self._system_prompt

    @property
    def system_as_user(self) -> bool:
        """Boolean indicating whether to use the user role for the system message.

        This is mainly used for experimental models where a dedicated system role is not available.

        Returns:
            bool: True if the system message is treated as a user message, False otherwise.
        """
        return self._system_as_user

    @property
    def context_template(self) -> str:
        """Context prompt template requiring a 'context' variable.

        The template can be a string, a function, or an instance of `vectorbtpro.utils.template.CustomTemplate`.

        This prompt is used to provide context for the conversation.

        Returns:
            str: Context prompt template.
        """
        return self._context_template

    @property
    def formatter(self) -> tp.MaybeType[ContentFormatter]:
        """Content formatter subclass or instance.

        Resolved using `vectorbtpro.knowledge.formatting.resolve_formatter`.

        This formatter is used to format the content of the response.

        Returns:
            MaybeType[ContentFormatter]: Resolved content formatter instance or subclass.
        """
        return self._formatter

    @property
    def formatter_kwargs(self) -> tp.Kwargs:
        """Keyword arguments to initialize or update `Completions.formatter`.

        Returns:
            Kwargs: Keyword arguments for the content formatter.
        """
        return self._formatter_kwargs

    @property
    def minimal_format(self) -> bool:
        """Boolean indicating if the input is minimally formatted.

        Returns:
            bool: True if the input is minimally formatted, False otherwise.
        """
        return self._minimal_format

    @property
    def quick_mode(self) -> bool:
        """Boolean indicating whether quick mode is enabled.

        Returns:
            bool: True if quick mode is enabled, False otherwise.
        """
        return self._quick_mode

    @property
    def silence_warnings(self) -> bool:
        """Boolean indicating whether warnings are suppressed.

        Returns:
            bool: True if warnings are suppressed, False otherwise.
        """
        return self._silence_warnings

    @property
    def show_progress(self) -> tp.Optional[bool]:
        """Whether to display a progress bar.

        Returns:
            Optional[bool]: True if progress bar is shown, False otherwise.
        """
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `vectorbtpro.utils.pbar.ProgressBar`.

        Returns:
            Kwargs: Keyword arguments for the progress bar.
        """
        return self._pbar_kwargs

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    @property
    def model(self) -> tp.Optional[str]:
        """Model name.

        Returns:
            Optional[str]: Model name if specified; otherwise, None.
        """
        return None

    def get_chat_response(self, messages: tp.ChatMessages, **kwargs) -> tp.Any:
        """Return a chat response based on the provided messages

        Args:
            messages (ChatMessages): List of dictionaries representing the conversation history.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Chat response generated from the provided messages.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def get_message_content(self, response: tp.Any) -> tp.Optional[str]:
        """Return the content extracted from a chat response.

        Args:
            response (Any): Chat response object.

        Returns:
            Optional[str]: Content extracted from the chat response.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def get_stream_response(self, messages: tp.ChatMessages, **kwargs) -> tp.Any:
        """Return a streaming response generated from the provided messages.

        Args:
            messages (ChatMessages): List of dictionaries representing the conversation history.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Streaming response generated from the provided messages.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def get_delta_content(self, response_chunk: tp.Any) -> tp.Optional[str]:
        """Return the content extracted from a streaming response chunk.

        Args:
            response (Any): Streaming response object.

        Returns:
            Optional[str]: Content extracted from the streaming response chunk.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def prepare_messages(self, message: str) -> tp.ChatMessages:
        """Return a list of chat messages formatted for a completion request.

        Args:
            message (str): User message to process.

        Returns:
            ChatMessages: List of dictionaries representing the conversation history.
        """
        context = self.context
        chat_history = self.chat_history
        max_tokens_set = self.max_tokens_set
        max_tokens = self.max_tokens
        tokenizer = self.tokenizer
        tokenizer_kwargs = self.tokenizer_kwargs
        system_prompt = self.system_prompt
        system_as_user = self.system_as_user
        context_template = self.context_template
        template_context = self.template_context
        silence_warnings = self.silence_warnings

        if isinstance(tokenizer, type):
            tokenizer_kwargs = dict(tokenizer_kwargs)
            tokenizer_kwargs["template_context"] = merge_dicts(
                template_context, tokenizer_kwargs.get("template_context", None)
            )
            if issubclass(tokenizer, TikTokenizer) and "model" not in tokenizer_kwargs:
                tokenizer_kwargs["model"] = self.model
            tokenizer = tokenizer(**tokenizer_kwargs)
        elif tokenizer_kwargs:
            tokenizer = tokenizer.replace(**tokenizer_kwargs)

        if context:
            if isinstance(context_template, str):
                context_template = SafeSub(context_template)
            elif checks.is_function(context_template):
                context_template = RepFunc(context_template)
            elif not isinstance(context_template, CustomTemplate):
                raise TypeError("Context prompt must be a string, function, or template")
            if max_tokens not in (None, False):
                if max_tokens is True:
                    raise ValueError("max_tokens cannot be True")
                empty_context_template = context_template.substitute(
                    flat_merge_dicts(dict(context=""), template_context),
                    eval_id="context_template",
                )
                empty_messages = [
                    dict(role="user" if system_as_user else "system", content=system_prompt),
                    dict(role="user", content=empty_context_template),
                    *chat_history,
                    dict(role="user", content=message),
                ]
                num_tokens = tokenizer.count_tokens_in_messages(empty_messages)
                max_context_tokens = max(0, max_tokens - num_tokens)
                encoded_context = tokenizer.encode(context)
                if len(encoded_context) > max_context_tokens:
                    context = tokenizer.decode(encoded_context[:max_context_tokens])
                    if not max_tokens_set and not silence_warnings:
                        warn(
                            f"Context is too long ({len(encoded_context)}). "
                            f"Truncating to {max_context_tokens} tokens."
                        )
            template_context = flat_merge_dicts(dict(context=context), template_context)
            context_template = context_template.substitute(template_context, eval_id="context_template")
            return [
                dict(role="user" if system_as_user else "system", content=system_prompt),
                dict(role="user", content=context_template),
                *chat_history,
                dict(role="user", content=message),
            ]
        else:
            return [
                dict(role="user" if system_as_user else "system", content=system_prompt),
                *chat_history,
                dict(role="user", content=message),
            ]

    def get_completion(
        self,
        message: str,
        return_response: bool = False,
    ) -> tp.ChatOutput:
        """Return the formatted completion output for a provided message.

        Args:
            message (str): User message to generate a completion for.
            return_response (bool): Flag to return the raw response along with the file path.

        Returns:
            ChatOutput: File path for the formatted output; if `return_response` is True,
                a tuple containing the file path and raw response.
        """
        chat_history = self.chat_history
        stream = self.stream
        formatter = self.formatter
        formatter_kwargs = self.formatter_kwargs
        template_context = self.template_context

        messages = self.prepare_messages(message)
        if self.stream:
            response = self.get_stream_response(messages)
        else:
            response = self.get_chat_response(messages)
        self.reset_thought_state()

        if isinstance(formatter, type):
            formatter_kwargs = dict(formatter_kwargs)
            if "minimal_format" not in formatter_kwargs:
                formatter_kwargs["minimal_format"] = self.minimal_format
            formatter_kwargs["template_context"] = merge_dicts(
                template_context, formatter_kwargs.get("template_context", None)
            )
            if issubclass(formatter, HTMLFileFormatter):
                if "page_title" not in formatter_kwargs:
                    formatter_kwargs["page_title"] = message
                if "cache_dir" not in formatter_kwargs:
                    chat_dir = self.get_setting("chat_dir", default=None)
                    if isinstance(chat_dir, CustomTemplate):
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
                        chat_dir = chat_dir.substitute(template_context, eval_id="chat_dir")
                    chat_dir = Path(chat_dir) / "html"
                    formatter_kwargs["dir_path"] = chat_dir
            formatter = formatter(**formatter_kwargs)
        elif formatter_kwargs:
            formatter = formatter.replace(**formatter_kwargs)
        if stream:
            with formatter:
                for i, response_chunk in enumerate(response):
                    new_content = self.get_delta_content(response_chunk)
                    if new_content is not None:
                        formatter.append(new_content)
                content = formatter.content
                flushed_content = self.flush_thought()
                if flushed_content:
                    content += flushed_content
        else:
            content = self.get_message_content(response) or ""
            flushed_content = self.flush_thought()
            if flushed_content:
                content += flushed_content
            formatter.append_once(content)

        chat_history.append(dict(role="user", content=message))
        chat_history.append(dict(role="assistant", content=content))
        if isinstance(formatter, HTMLFileFormatter) and formatter.file_handle is not None:
            file_path = Path(formatter.file_handle.name)
        else:
            file_path = None
        if return_response:
            return file_path, response
        return file_path

    def get_completion_content(self, message: str) -> str:
        """Return the text content of a completion for a given message.

        Args:
            message (str): User message to complete.

        Returns:
            str: Generated completion text.
        """
        chat_history = self.chat_history

        messages = self.prepare_messages(message)
        response = self.get_chat_response(messages)
        content = self.get_message_content(response)
        if content is None:
            content = ""
        chat_history.append(dict(role="user", content=message))
        chat_history.append(dict(role="assistant", content=content))
        return content


class OpenAICompletions(Completions):
    """Completions class for OpenAI.

    Args:
        model (Optional[str]): Identifier for the model to use.
        client_kwargs (KwargsLike): Keyword arguments for `openai.OpenAI`.
        completions_kwargs (KwargsLike): Keyword arguments for `openai.Completions.create`.
        responses_kwargs (KwargsLike): Keyword arguments for `openai.Responses.create`.
        use_responses (bool): Whether to use the Responses API instead of the Completions API.

            Note that thought summarization is not supported in the Completions API.
        **kwargs: Keyword arguments for `Completions` or used as `client_kwargs`,
            `completions_kwargs`, or `responses_kwargs`.

    !!! info
        For default settings, see `chat.completions_configs.openai` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "openai"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.openai"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        completions_kwargs: tp.KwargsLike = None,
        responses_kwargs: tp.KwargsLike = None,
        use_responses: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        Completions.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            completions_kwargs=completions_kwargs,
            responses_kwargs=responses_kwargs,
            use_responses=use_responses,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("openai")
        from openai import OpenAI

        openai_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = openai_config.pop("model", None)
        def_quick_model = openai_config.pop("quick_model", None)
        def_client_kwargs = openai_config.pop("client_kwargs", None)
        def_completions_kwargs = openai_config.pop("completions_kwargs", None)
        def_responses_kwargs = openai_config.pop("responses_kwargs", None)
        def_use_responses = openai_config.pop("use_responses", None)

        if model is None:
            model = def_quick_model if self.quick_mode else def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = set(get_func_arg_names(Completions.__init__)) | set(get_func_arg_names(type(self).__init__))
        for k in list(openai_config.keys()):
            if k in init_arg_names:
                openai_config.pop(k)

        client_arg_names = set(get_func_arg_names(OpenAI.__init__))
        _client_kwargs = {}
        _completions_kwargs = {}
        _responses_kwargs = {}
        for k, v in openai_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _completions_kwargs[k] = v
                _responses_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        completions_kwargs = merge_dicts(_completions_kwargs, def_completions_kwargs, completions_kwargs)
        responses_kwargs = merge_dicts(_responses_kwargs, def_responses_kwargs, responses_kwargs)
        client = OpenAI(**client_kwargs)

        if use_responses is None:
            use_responses = def_use_responses

        self._model = model
        self._client = client
        self._completions_kwargs = completions_kwargs
        self._responses_kwargs = responses_kwargs
        self._use_responses = use_responses

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> OpenAIT:
        """OpenAI client instance used for API calls.

        Returns:
            OpenAI: OpenAI client instance.
        """
        return self._client

    @property
    def completions_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `openai.Completions.create`.

        Returns:
            Kwargs: Keyword arguments for the completion API call.
        """
        return self._completions_kwargs

    @property
    def responses_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `openai.Responses.create`.

        Returns:
            Kwargs: Keyword arguments for the responses API call.
        """
        return self._responses_kwargs

    @property
    def use_responses(self) -> bool:
        """Whether to use the Responses API instead of the Completions API.

        Returns:
            bool: Whether to use the Responses API.
        """
        return self._use_responses

    def format_messages(self, messages: tp.ChatMessages) -> tp.Tuple[tp.List[tp.Dict[str, str]], str]:
        """Format messages to Responses API format.

        Args:
            messages (ChatMessages): List of dictionaries representing the conversation history.

        Returns:
            Tuple[List[Dict[str, str]], str]: List of message dictionaries and system instructions.
        """
        input = []
        instructions = []
        for message in messages:
            if isinstance(message, dict):
                role = message.pop("role", "user")
                content = message.pop("content", "")
                if len(message) > 0:
                    raise ValueError(f"Unsupported message format: {message}. Expected dict with 'role' and 'content'.")

                if role == "system":
                    instructions.append(content)
                    continue
                input.append(dict(role=role, content=content))
            elif isinstance(message, str):
                input.append(dict(role="user", content=message))
            else:
                raise TypeError(f"Unsupported message type: {type(message)}. Expected dict or str.")
        if instructions:
            instructions = "\n".join(instructions)
        else:
            instructions = None
        return input, instructions

    def get_chat_response(self, messages: tp.ChatMessages) -> ChatCompletionT:
        if self.use_responses:
            input, instructions = self.format_messages(messages)
            return self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=input,
                stream=False,
                **self.responses_kwargs,
            )
        else:
            return self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                stream=False,
                **self.completions_kwargs,
            )

    def get_message_content(self, response: tp.Union[ChatCompletionT, ResponseT]) -> tp.Optional[str]:
        if self.use_responses:
            out = None
            for output in response.output:
                if output.type == "reasoning":
                    for summary in output.summary:
                        if summary.type == "summary_text":
                            thought = self.process_thought(thought=summary.text, flush=True)
                            if thought is not None:
                                if out is None:
                                    out = ""
                                out += thought
                if output.type == "message":
                    for content in output.content:
                        if content.type == "output_text":
                            text = self.process_thought(content=content.text, flush=True)
                            if text is not None:
                                if out is None:
                                    out = ""
                                out += text
            return out
        return response.choices[0].message.content

    def get_stream_response(self, messages: tp.ChatMessages) -> StreamT:
        if self.use_responses:
            input, instructions = self.format_messages(messages)
            return self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=input,
                stream=True,
                **self.responses_kwargs,
            )
        else:
            return self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                stream=True,
                **self.completions_kwargs,
            )

    def get_delta_content(
        self,
        response_chunk: tp.Union[ChatCompletionChunkT, ResponseStreamEventT],
    ) -> tp.Optional[str]:
        if self.use_responses:
            if response_chunk.type == "response.reasoning_summary_text.delta":
                return self.process_thought(thought=response_chunk.delta)
            if response_chunk.type == "response.output_text.delta":
                return self.process_thought(content=response_chunk.delta)
            return self.flush_thought()
        return response_chunk.choices[0].delta.content


class AnthropicCompletions(Completions):
    """Completions class for Anthropic (Claude).

    Args:
        model (Optional[str]): Anthropic model identifier.
        client_type (Union[None, str, type]): Anthropic client type.

            Supported values:

            * "anthropic": `anthropic.Anthropic`
            * "bedrock": `anthropic.AnthropicBedrock`
            * "vertex": `anthropic.AnthropicVertex`
            * type: Custom Anthropic client class
        client_kwargs (KwargsLike): Keyword arguments for `client_type`
        messages_kwargs (KwargsLike): Keyword arguments for `anthropic.Client.messages.create`.
        **kwargs: Keyword arguments for `Completions` or used as `client_kwargs` or `messages_kwargs`.

    !!! info
        For default settings, see `chat.completions_configs.anthropic` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "anthropic"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.anthropic"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_type: tp.Union[None, str, type] = None,
        client_kwargs: tp.KwargsLike = None,
        messages_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Completions.__init__(
            self,
            model=model,
            client_type=client_type,
            client_kwargs=client_kwargs,
            messages_kwargs=messages_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("anthropic")

        anthropic_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = anthropic_config.pop("model", None)
        def_client_type = anthropic_config.pop("client_type", None)
        def_quick_model = anthropic_config.pop("quick_model", None)
        def_client_kwargs = anthropic_config.pop("client_kwargs", None)
        def_messages_kwargs = anthropic_config.pop("messages_kwargs", None)

        if model is None:
            model = def_quick_model if self.quick_mode else def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = set(get_func_arg_names(Completions.__init__)) | set(get_func_arg_names(type(self).__init__))
        for k in list(anthropic_config.keys()):
            if k in init_arg_names:
                anthropic_config.pop(k)

        if client_type is None:
            client_type = def_client_type
        if isinstance(client_type, str) and client_type.lower() == "anthropic":
            from anthropic import Anthropic

            client_type = Anthropic
        elif isinstance(client_type, str) and client_type.lower() == "bedrock":
            from anthropic import AnthropicBedrock

            client_type = AnthropicBedrock
        elif isinstance(client_type, str) and client_type.lower() == "vertex":
            from anthropic import AnthropicVertex

            client_type = AnthropicVertex
        elif not isinstance(client_type, type):
            raise ValueError(f"Invalid client_type: {client_type!r}")

        client_arg_names = set(get_func_arg_names(client_type.__init__))
        _client_kwargs = {}
        _messages_kwargs = {}
        for k, v in anthropic_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _messages_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        messages_kwargs = merge_dicts(_messages_kwargs, def_messages_kwargs, messages_kwargs)

        client = client_type(**client_kwargs)

        self._model = model
        self._client = client
        self._messages_kwargs = messages_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> AnthropicClientT:
        """Anthropic client instance.

        Returns:
            Anthropic: Anthropic client instance.
        """
        return self._client

    @property
    def messages_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `anthropic.Client.messages.create`.

        Returns:
            Kwargs: Keyword arguments for message creation.
        """
        return self._messages_kwargs

    def format_messages(self, messages: tp.ChatMessages) -> tp.Tuple[tp.List[tp.Dict[str, str]], str]:
        """Format messages to Anthropic format.

        Args:
            messages (ChatMessages): List of dictionaries representing the conversation history.

        Returns:
            Tuple[List[Dict[str, str]], str]: List of message dictionaries and system message.
        """
        formatted_messages = []
        system_message = ""

        for message in messages:
            if isinstance(message, dict):
                role = message.pop("role", "user")
                content = message.pop("content", "")
                if len(message) > 0:
                    raise ValueError(f"Unsupported message format: {message}. Expected dict with 'role' and 'content'.")

                if role == "system":
                    system_message = content
                elif role in ["user", "assistant"]:
                    formatted_messages.append({"role": role, "content": content})
            elif isinstance(message, str):
                formatted_messages.append({"role": "user", "content": message})
            else:
                raise TypeError(f"Unsupported message type: {type(message)}. Expected dict or str.")

        return formatted_messages, system_message

    def get_chat_response(self, messages: tp.ChatMessages) -> AnthropicMessageT:
        formatted_messages, system_message = self.format_messages(messages)
        messages_kwargs = dict(self.messages_kwargs)
        if system_message:
            messages_kwargs["system"] = system_message

        return self.client.messages.create(
            model=self.model,
            messages=formatted_messages,
            stream=False,
            **messages_kwargs,
        )

    def get_message_content(self, response: AnthropicMessageT) -> tp.Optional[str]:
        from anthropic.types import ThinkingBlock, TextBlock

        content = None
        for block in response.content:
            if isinstance(block, ThinkingBlock):
                thinking = self.process_thought(thought=block.thinking, flush=True)
                if thinking is not None:
                    if content is None:
                        content = ""
                    content += thinking
            elif isinstance(block, TextBlock):
                text = self.process_thought(content=block.text, flush=True)
                if text is not None:
                    if content is None:
                        content = ""
                    content += text
            else:
                out = self.flush_thought()
                if out is not None:
                    if content is None:
                        content = ""
                    content += out
        return content

    def get_stream_response(self, messages: tp.ChatMessages) -> AnthropicStreamT:
        formatted_messages, system_message = self.format_messages(messages)
        messages_kwargs = dict(self.messages_kwargs)
        if system_message:
            messages_kwargs["system"] = system_message

        return self.client.messages.create(
            model=self.model,
            messages=formatted_messages,
            stream=True,
            **messages_kwargs,
        )

    def get_delta_content(self, response_chunk: AnthropicMessageStreamEventT) -> tp.Optional[str]:
        from anthropic.types import RawContentBlockDeltaEvent, ThinkingDelta, TextDelta

        if isinstance(response_chunk, RawContentBlockDeltaEvent) and isinstance(response_chunk.delta, ThinkingDelta):
            return self.process_thought(thought=response_chunk.delta.thinking)
        if isinstance(response_chunk, RawContentBlockDeltaEvent) and isinstance(response_chunk.delta, TextDelta):
            return self.process_thought(content=response_chunk.delta.text)
        return self.flush_thought()


class GeminiCompletions(Completions):
    """Completions class for Google GenAI (Gemini).

    Args:
        model (Optional[str]): Gemini model identifier.
        client_kwargs (KwargsLike): Keyword arguments for `google.genai.Client`.
        completions_kwargs (KwargsLike): Keyword arguments for `google.genai.Client.models.generate_content`.
        **kwargs: Keyword arguments for `Completions` or used as `client_kwargs` or `completions_kwargs`.

    !!! info
        For default settings, see `chat.completions_configs.gemini` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "gemini"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.gemini"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        completions_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Completions.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            completions_kwargs=completions_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("google.genai")
        from google.genai import Client

        gemini_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = gemini_config.pop("model", None)
        def_quick_model = gemini_config.pop("quick_model", None)
        def_client_kwargs = gemini_config.pop("client_kwargs", None)
        def_completions_kwargs = gemini_config.pop("completions_kwargs", None)

        if model is None:
            model = def_quick_model if self.quick_mode else def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = set(get_func_arg_names(Completions.__init__)) | set(get_func_arg_names(type(self).__init__))
        for k in list(gemini_config.keys()):
            if k in init_arg_names:
                gemini_config.pop(k)

        client_arg_names = set(get_func_arg_names(Client.__init__))
        _client_kwargs = {}
        _completions_kwargs = {}
        for k, v in gemini_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _completions_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        completions_kwargs = merge_dicts(_completions_kwargs, def_completions_kwargs, completions_kwargs)

        client = Client(**client_kwargs)

        self._model = model
        self._client = client
        self._completions_kwargs = completions_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> GenAIClientT:
        """Gemini client instance.

        Returns:
            Client: Gemini client instance.
        """
        return self._client

    @property
    def completions_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `google.genai.Client.models.generate_content`.

        Returns:
            Kwargs: Keyword arguments for content generation.
        """
        return self._completions_kwargs

    def format_messages(self, messages: tp.ChatMessages) -> tp.Tuple[tp.List[ContentT], tp.List[str]]:
        """Format messages to Gemini format.

        Args:
            messages (ChatMessages): List of dictionaries representing the conversation history.

        Returns:
            Union[List[Content], List[str]]: List of `google.genai.types.Content` objects and system instructions.
        """
        from google.genai.types import Content, Part

        contents = []
        system_instruction = []
        for message in messages:
            if isinstance(message, dict):
                role = message.pop("role", "user")
                text = message.pop("content", "")
                if len(message) > 0:
                    raise ValueError(f"Unsupported message format: {message}. Expected dict with 'role' and 'content'.")

                if role == "assistant":
                    role = "model"
                elif role == "system":
                    system_instruction.append(text)
                    continue
                content = Content(role=role, parts=[Part.from_text(text=text)])
                contents.append(content)
            elif isinstance(message, str):
                content = Content(role="user", parts=[Part.from_text(text=message)])
                contents.append(content)
            else:
                raise TypeError(f"Unsupported message type: {type(message)}. Expected dict or str.")
        return contents, system_instruction

    def get_chat_response(self, messages: tp.ChatMessages) -> GenerateContentResponseT:
        from google.genai.types import GenerateContentConfig
        from google.genai.errors import ClientError

        formatted_messages, system_instruction = self.format_messages(messages)
        completions_kwargs = dict(self.completions_kwargs)
        config = dict(completions_kwargs.pop("config", {}))
        if system_instruction:
            config["system_instruction"] = system_instruction

        attempted = False
        while True:
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=formatted_messages,
                    config=GenerateContentConfig(**config),
                    **completions_kwargs,
                )
            except ClientError as e:
                if e.code == 429 and not attempted:
                    time.sleep(60)
                    attempted = True
                else:
                    raise e

    def get_message_content(self, response: GenerateContentResponseT) -> tp.Optional[str]:
        content = None
        for part in response.candidates[0].content.parts:
            if getattr(part, "thought", False):
                text = self.process_thought(thought=part.text, flush=True)
                if text is not None:
                    if content is None:
                        content = ""
                    content += text
            else:
                text = self.process_thought(content=part.text, flush=True)
                if text is not None:
                    if content is None:
                        content = ""
                    content += text
        return content

    def get_stream_response(self, messages: tp.ChatMessages) -> tp.Iterator[GenerateContentResponseT]:
        from google.genai.types import GenerateContentConfig
        from google.genai.errors import ClientError

        formatted_messages, system_instruction = self.format_messages(messages)
        completions_kwargs = dict(self.completions_kwargs)
        config = dict(completions_kwargs.pop("config", {}))
        if system_instruction:
            config["system_instruction"] = system_instruction

        attempted = False
        while True:
            try:
                return self.client.models.generate_content_stream(
                    model=self.model,
                    contents=formatted_messages,
                    config=GenerateContentConfig(**config),
                    **completions_kwargs,
                )
            except ClientError as e:
                if e.code == 429 and not attempted:
                    time.sleep(60)
                    attempted = True
                else:
                    raise e

    def get_delta_content(self, response_chunk: GenerateContentResponseT) -> tp.Optional[str]:
        content = None
        for part in response_chunk.candidates[0].content.parts:
            if getattr(part, "thought", False):
                text = self.process_thought(thought=part.text)
                if text is not None:
                    if content is None:
                        content = ""
                    content += text
            else:
                text = self.process_thought(content=part.text)
                if text is not None:
                    if content is None:
                        content = ""
                    content += text
        return content


class HFInferenceCompletions(Completions):
    """Completions class for HuggingFace Inference.

    Args:
        model (Optional[str]): HuggingFace model identifier.
        client_kwargs (KwargsLike): Keyword arguments for `huggingface_hub.InferenceClient`.
        chat_completion_kwargs (KwargsLike): Keyword arguments for `huggingface_hub.InferenceClient.chat_completion`.
        **kwargs: Keyword arguments for `Completions` or used as `client_kwargs` or `chat_completion_kwargs`.

    !!! info
        For default settings, see `chat.completions_configs.hf_inference` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "hf_inference"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.hf_inference"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        chat_completion_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Completions.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            chat_completion_kwargs=chat_completion_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("huggingface_hub")
        from huggingface_hub import InferenceClient

        hf_inference_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = hf_inference_config.pop("model", None)
        def_quick_model = hf_inference_config.pop("quick_model", None)
        def_client_kwargs = hf_inference_config.pop("client_kwargs", None)
        def_chat_completion_kwargs = hf_inference_config.pop("chat_completion_kwargs", None)

        if model is None:
            model = def_quick_model if self.quick_mode else def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = set(get_func_arg_names(Completions.__init__)) | set(get_func_arg_names(type(self).__init__))
        for k in list(hf_inference_config.keys()):
            if k in init_arg_names:
                hf_inference_config.pop(k)

        client_arg_names = set(get_func_arg_names(InferenceClient.__init__))
        _client_kwargs = {}
        _chat_completion_kwargs = {}
        for k, v in hf_inference_config.items():
            if k in client_arg_names:
                _client_kwargs[k] = v
            else:
                _chat_completion_kwargs[k] = v
        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        chat_completion_kwargs = merge_dicts(
            _chat_completion_kwargs, def_chat_completion_kwargs, chat_completion_kwargs
        )
        client = InferenceClient(model=model, **client_kwargs)

        self._model = model
        self._client = client
        self._chat_completion_kwargs = chat_completion_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> InferenceClientT:
        """HuggingFace Inference client instance.

        Returns:
            InferenceClient: HuggingFace Inference client instance.
        """
        return self._client

    @property
    def chat_completion_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `huggingface_hub.InferenceClient.chat_completion`.

        Returns:
            Kwargs: Keyword arguments for chat completion.
        """
        return self._chat_completion_kwargs

    def get_chat_response(self, messages: tp.ChatMessages) -> ChatCompletionOutputT:
        return self.client.chat_completion(
            messages=messages,
            model=self.model,
            stream=False,
            **self.chat_completion_kwargs,
        )

    def get_message_content(self, response: ChatCompletionOutputT) -> tp.Optional[str]:
        message = response.choices[0].message
        if hasattr(message, "thinking"):
            thought = message.thinking
        elif hasattr(message, "reasoning"):
            thought = message.reasoning
        elif hasattr(message, "reasoning_content"):
            thought = message.reasoning_content
        else:
            thought = None
        content = message.content
        return self.process_thought(thought=thought, content=content, flush=True)

    def get_stream_response(self, messages: tp.ChatMessages) -> ChatCompletionStreamOutputT:
        return self.client.chat_completion(
            messages=messages,
            model=self.model,
            stream=True,
            **self.chat_completion_kwargs,
        )

    def get_delta_content(self, response_chunk: ChatCompletionStreamOutputT) -> tp.Optional[str]:
        delta = response_chunk.choices[0].delta
        if hasattr(delta, "thinking"):
            thought = delta.thinking
        elif hasattr(delta, "reasoning"):
            thought = delta.reasoning
        elif hasattr(delta, "reasoning_content"):
            thought = delta.reasoning_content
        else:
            thought = None
        content = delta.content
        return self.process_thought(thought=thought, content=content)


class LiteLLMCompletions(Completions):
    """Completions class for LiteLLM.

    Args:
        model (Optional[str]): Identifier for the model to use.
        completion_kwargs (KwargsLike): Keyword arguments for `litellm.completion`.
        **kwargs: Keyword arguments for `Completions` or used as `completion_kwargs`.

    !!! info
        For default settings, see `chat.completions_configs.litellm` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "litellm"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.litellm"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        completion_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Completions.__init__(
            self,
            model=model,
            completion_kwargs=completion_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("litellm")

        super_arg_names = set(get_func_arg_names(Completions.__init__))
        for k in list(kwargs.keys()):
            if k in super_arg_names:
                kwargs.pop(k)
        litellm_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = litellm_config.pop("model", None)
        def_quick_model = litellm_config.pop("quick_model", None)
        def_completion_kwargs = litellm_config.pop("completion_kwargs", None)

        if model is None:
            model = def_quick_model if self.quick_mode else def_model
        if model is None:
            raise ValueError("Must provide a model")
        completion_kwargs = merge_dicts(litellm_config, def_completion_kwargs, completion_kwargs)

        self._model = model
        self._completion_kwargs = completion_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def completion_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `litellm.completion`.

        Returns:
            Kwargs: Keyword arguments for the completion API call.
        """
        return self._completion_kwargs

    def get_chat_response(self, messages: tp.ChatMessages) -> ModelResponseT:
        from litellm import completion

        return completion(
            messages=messages,
            model=self.model,
            stream=False,
            **self.completion_kwargs,
        )

    def get_message_content(self, response: ModelResponseT) -> tp.Optional[str]:
        message = response.choices[0].message
        reasoning_content = getattr(message, "reasoning_content", None)
        return self.process_thought(thought=reasoning_content, content=message.content, flush=True)

    def get_stream_response(self, messages: tp.ChatMessages) -> CustomStreamWrapperT:
        from litellm import completion

        return completion(
            messages=messages,
            model=self.model,
            stream=True,
            **self.completion_kwargs,
        )

    def get_delta_content(self, response_chunk: ModelResponseT) -> tp.Optional[str]:
        delta = response_chunk.choices[0].delta
        reasoning_content = getattr(delta, "reasoning_content", None)
        return self.process_thought(thought=reasoning_content, content=delta.content)


class LlamaIndexCompletions(Completions):
    """Completions class for LlamaIndex.

    LLM can be provided via `llm`, which can be either the name of the class (case doesn't matter),
    the path or its suffix to the class (case matters), or a subclass or an instance of
    `llama_index.core.llms.LLM`.

    Args:
        llm (Union[None, str, MaybeType[LLM]]): Identifier, class path, subclass, or instance of
            `llama_index.core.llms.LLM`.
        llm_kwargs (KwargsLike): Additional parameters for LLM initialization.
        **kwargs: Keyword arguments for `Completions` or used as `llm_kwargs`.

    !!! info
        For default settings, see `chat.completions_configs.llama_index` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "llama_index"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.llama_index"

    def __init__(
        self,
        llm: tp.Union[None, str, tp.MaybeType[LLMT]] = None,
        llm_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Completions.__init__(
            self,
            llm=llm,
            llm_kwargs=llm_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("llama_index")
        from llama_index.core.llms import LLM

        llama_index_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_llm = llama_index_config.pop("llm", None)
        def_llm_kwargs = llama_index_config.pop("llm_kwargs", None)

        if llm is None:
            llm = def_llm
        if llm is None:
            raise ValueError("Must provide an LLM name or path")
        init_arg_names = set(get_func_arg_names(Completions.__init__)) | set(get_func_arg_names(type(self).__init__))
        for k in list(llama_index_config.keys()):
            if k in init_arg_names:
                llama_index_config.pop(k)

        if isinstance(llm, str):
            import llama_index.llms
            from vectorbtpro.utils.module_ import search_package

            def _match_func(k, v):
                if isinstance(v, type) and issubclass(v, LLM):
                    if "." in llm:
                        if k.endswith(llm):
                            return True
                    else:
                        if k.split(".")[-1].lower() == llm.lower():
                            return True
                        if k.split(".")[-1].replace("LLM", "").lower() == llm.lower().replace("_", ""):
                            return True
                return False

            found_llm = search_package(
                llama_index.llms,
                _match_func,
                path_attrs=True,
                return_first=True,
            )
            if found_llm is None:
                raise ValueError(f"LLM {llm!r} not found")
            llm = found_llm
        if isinstance(llm, type):
            checks.assert_subclass_of(llm, LLM, arg_name="llm")
            llm_name = llm.__name__.replace("LLM", "").lower()
            module_name = llm.__module__
        else:
            checks.assert_instance_of(llm, LLM, arg_name="llm")
            llm_name = type(llm).__name__.replace("LLM", "").lower()
            module_name = type(llm).__module__
        llm_configs = llama_index_config.pop("llm_configs", {})
        if llm_name in llm_configs:
            llama_index_config = merge_dicts(llama_index_config, llm_configs[llm_name])
        elif module_name in llm_configs:
            llama_index_config = merge_dicts(llama_index_config, llm_configs[module_name])
        llm_kwargs = merge_dicts(llama_index_config, def_llm_kwargs, llm_kwargs)
        def_model = llm_kwargs.pop("model", None)
        quick_model = llm_kwargs.pop("quick_model", None)
        model = quick_model if self.quick_mode else def_model
        if model is None:
            func_kwargs = get_func_kwargs(type(llm).__init__)
            model = func_kwargs.get("model", None)
        else:
            llm_kwargs["model"] = model
        if isinstance(llm, type):
            llm = llm(**llm_kwargs)
        elif len(kwargs) > 0:
            raise ValueError("Cannot apply config to already initialized LLM")

        self._model = model
        self._llm = llm

    @property
    def model(self) -> tp.Optional[str]:
        return self._model

    @property
    def llm(self) -> LLMT:
        """Initialized LLM instance used for generating completions.

        Returns:
            LLM: Initialized LLM instance.
        """
        return self._llm

    def get_chat_response(self, messages: tp.ChatMessages) -> ChatResponseT:
        from llama_index.core.llms import ChatMessage

        return self.llm.chat(list(map(lambda x: ChatMessage(**dict(x)), messages)))

    def get_message_content(self, response: ChatResponseT) -> tp.Optional[str]:
        return response.message.content

    def get_stream_response(self, messages: tp.ChatMessages) -> tp.Iterator[ChatResponseT]:
        from llama_index.core.llms import ChatMessage

        return self.llm.stream_chat(list(map(lambda x: ChatMessage(**dict(x)), messages)))

    def get_delta_content(self, response_chunk: ChatResponseT) -> tp.Optional[str]:
        return response_chunk.delta


class OllamaCompletions(Completions):
    """Completions class for Ollama.

    Args:
        model (Optional[str]): Ollama model identifier.

            Pulls the model if not already available locally.
        client_kwargs (KwargsLike): Keyword arguments for `ollama.Client`.
        chat_kwargs (KwargsLike): Keyword arguments for `ollama.Client.chat`.
        **kwargs: Keyword arguments for `Completions` or used as `client_kwargs` or `chat_kwargs`.

    !!! info
        For default settings, see `chat.completions_configs.ollama` in `vectorbtpro._settings.knowledge`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = "ollama"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.ollama"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        client_kwargs: tp.KwargsLike = None,
        chat_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Completions.__init__(
            self,
            model=model,
            client_kwargs=client_kwargs,
            chat_kwargs=chat_kwargs,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ollama")
        from ollama import Client

        ollama_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = ollama_config.pop("model", None)
        def_quick_model = ollama_config.pop("quick_model", None)
        def_client_kwargs = ollama_config.pop("client_kwargs", None)
        def_chat_kwargs = ollama_config.pop("chat_kwargs", None)

        if model is None:
            model = def_quick_model if self.quick_mode else def_model
        if model is None:
            raise ValueError("Must provide a model")
        init_arg_names = set(get_func_arg_names(Completions.__init__)) | set(get_func_arg_names(type(self).__init__))
        for k in list(ollama_config.keys()):
            if k in init_arg_names:
                ollama_config.pop(k)

        _client_kwargs = {}
        _chat_kwargs = {}
        for k, v in ollama_config.items():
            _chat_kwargs[k] = v

        client_kwargs = merge_dicts(_client_kwargs, def_client_kwargs, client_kwargs)
        chat_kwargs = merge_dicts(_chat_kwargs, def_chat_kwargs, chat_kwargs)

        client = Client(**client_kwargs)
        model_installed = False
        for installed_model in client.list().models:
            if installed_model.model == model:
                model_installed = True
                break
        if not model_installed:
            pbar = None
            status = None
            for response in client.pull(model, stream=True):
                if pbar is not None and status is not None and response.status != status:
                    pbar.refresh()
                    pbar.exit()
                    pbar = None
                    status = None
                if response.completed is not None:
                    status = response.status
                    if pbar is None:
                        pbar = ProgressBar(total=response.total, show_progress=self.show_progress, **self.pbar_kwargs)
                        pbar.enter()
                    pbar.set_prefix(status)
                    pbar.update_to(response.completed)

        self._model = model
        self._client = client
        self._chat_kwargs = chat_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> OllamaClientT:
        """Ollama client instance.

        Returns:
            Client: Ollama client instance.
        """
        return self._client

    @property
    def chat_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `ollama.Client.chat`.

        Returns:
            Kwargs: Keyword arguments for chat completion.
        """
        return self._chat_kwargs

    def get_chat_response(self, messages: tp.ChatMessages) -> OllamaChatResponseT:
        return self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            **self.chat_kwargs,
        )

    def get_message_content(self, response: OllamaChatResponseT) -> tp.Optional[str]:
        message = response["message"]
        if hasattr(message, "thinking"):
            thought = message.thinking
        elif hasattr(message, "reasoning"):
            thought = message.reasoning
        elif hasattr(message, "reasoning_content"):
            thought = message.reasoning_content
        else:
            thought = None
        content = message.content
        return self.process_thought(thought=thought, content=content, flush=True)

    def get_stream_response(self, messages: tp.ChatMessages) -> tp.Iterator[OllamaChatResponseT]:
        return self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            **self.chat_kwargs,
        )

    def get_delta_content(self, response_chunk: OllamaChatResponseT) -> tp.Optional[str]:
        message = response_chunk["message"]
        if hasattr(message, "thinking"):
            thought = message.thinking
        elif hasattr(message, "reasoning"):
            thought = message.reasoning
        elif hasattr(message, "reasoning_content"):
            thought = message.reasoning_content
        else:
            thought = None
        content = message.content
        return self.process_thought(thought=thought, content=content)


def resolve_completions(completions: tp.CompletionsLike = None) -> tp.MaybeType[Completions]:
    """Resolve and return a `Completions` subclass or instance.

    Args:
        completions (CompletionsLike): Identifier, subclass, or instance of `Completions`.

            Supported identifiers:

            * "openai" for `OpenAICompletions`
            * "anthropic" for `AnthropicCompletions`
            * "gemini" for `GeminiCompletions`
            * "hf_inference" for `HFInferenceCompletions`
            * "litellm" for `LiteLLMCompletions`
            * "llama_index" for `LlamaIndexCompletions`
            * "ollama" for `OllamaCompletions`
            * "auto" to select the first available option

    Returns:
        Completions: Resolved completions class or instance.

    !!! info
        For default settings, see `chat` in `vectorbtpro._settings.knowledge`.
    """
    if completions is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        completions = chat_cfg["completions"]
    if isinstance(completions, str):
        if completions.lower() == "auto":
            import os
            from vectorbtpro.utils.module_ import check_installed

            if check_installed("openai") and os.getenv("OPENAI_API_KEY"):
                completions = "openai"
            elif check_installed("anthropic") and os.getenv("ANTHROPIC_API_KEY"):
                completions = "anthropic"
            elif check_installed("google.genai") and os.getenv("GEMINI_API_KEY"):
                completions = "gemini"
            elif check_installed("huggingface_hub") and os.getenv("HF_TOKEN"):
                completions = "hf_inference"
            elif check_installed("litellm"):
                completions = "litellm"
            elif check_installed("llama_index"):
                completions = "llama_index"
            elif check_installed("ollama"):
                completions = "ollama"
            else:
                raise ValueError(
                    "No completions available. "
                    "Please install one of the supported packages: "
                    "openai, "
                    "litellm, "
                    "llama-index, "
                    "huggingface-hub, "
                    "google-genai, "
                    "anthropic, "
                    "ollama."
                )
        curr_module = sys.modules[__name__]
        found_completions = None
        for name, cls in inspect.getmembers(curr_module, inspect.isclass):
            if name.endswith("Completions"):
                _short_name: tp.ClassVar[tp.Optional[str]] = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == completions.lower():
                    found_completions = cls
                    break
        if found_completions is None:
            raise ValueError(f"Invalid completions: {completions!r}")
        completions = found_completions
    if isinstance(completions, type):
        checks.assert_subclass_of(completions, Completions, arg_name="completions")
    else:
        checks.assert_instance_of(completions, Completions, arg_name="completions")
    return completions


def complete(message: str, completions: tp.CompletionsLike = None, **kwargs) -> tp.ChatOutput:
    """Get and return the chat completion for a provided message.

    Args:
        message (str): Input message for which to generate a completion.
        completions (CompletionsLike): Identifier, subclass, or instance of `Completions`.

            Resolved using `resolve_completions`.
        **kwargs: Keyword arguments to initialize or update `completions`.

    Returns:
        ChatOutput: Completion output generated by the resolved completions.
    """
    completions = resolve_completions(completions=completions)
    if isinstance(completions, type):
        completions = completions(**kwargs)
    elif kwargs:
        completions = completions.replace(**kwargs)
    return completions.get_completion(message)


def completed(message: str, completions: tp.CompletionsLike = None, **kwargs) -> str:
    """Return completion content for a given message using the provided completions configuration.

    Args:
        message (str): Input message.
        completions (CompletionsLike): Identifier, subclass, or instance of `Completions`.

            Resolved using `resolve_completions`.
        **kwargs: Keyword arguments to initialize or update `completions`.

    Returns:
        str: Completion content based on the input message.
    """
    completions = resolve_completions(completions=completions)
    if isinstance(completions, type):
        completions = completions(**kwargs)
    elif kwargs:
        completions = completions.replace(**kwargs)
    return completions.get_completion_content(message)
