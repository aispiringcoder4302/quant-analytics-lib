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
import sys
import warnings
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, flat_merge_dicts, deep_merge_dicts, Configured, HasSettings
from vectorbtpro.utils.decorators import hybrid_method
from vectorbtpro.utils.knowledge.formatting import ContentFormatter, HTMLFileFormatter, resolve_formatter
from vectorbtpro.utils.parsing import get_func_arg_names, get_func_kwargs
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
    from llama_index.core.embeddings import BaseEmbedding as BaseEmbeddingT
    from llama_index.core.llms import LLM as LLMT, ChatMessage as ChatMessageT, ChatResponse as ChatResponseT
except ImportError:
    BaseEmbeddingT = "BaseEmbedding"
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
    "Tokenizer",
    "TikTokenizer",
    "Embeddings",
    "OpenAIEmbeddings",
    "LiteLLMEmbeddings",
    "LlamaIndexEmbeddings",
    "embed",
    "Completions",
    "OpenAICompletions",
    "LiteLLMCompletions",
    "LlamaIndexCompletions",
    "complete",
    "Contextable",
]


# ############# Tokenizers ############# #


class Tokenizer(Configured):
    """Abstract class for tokenizers.

    For defaults, see `chat` in `vectorbtpro._settings.knowledge`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    _settings_path: tp.SettingsPath = "knowledge"

    def __init__(self, **kwargs) -> None:
        Configured.__init__(self, **kwargs)

    def encode(self, text: str) -> tp.List[int]:
        """Encode text into a list of tokens."""
        raise NotImplementedError

    def decode(self, tokens: tp.List[int]) -> str:
        """Decode a list of tokens into text."""
        raise NotImplementedError

    def count_tokens_in_messages(self, messages: tp.ChatMessages) -> int:
        """Count tokens in messages."""
        raise NotImplementedError


class TikTokenizer(Tokenizer):
    """Tokenizer class for tiktoken.

    Encoding can be a model name, an encoding name, or an encoding object for tokenization.

    For defaults, see `chat.tokenizer_configs.tiktoken` in `vectorbtpro._settings.knowledge`."""

    _short_name = "tiktoken"

    _settings_path: tp.SettingsPath = "knowledge.chat.tokenizer_configs.tiktoken"

    def __init__(
        self,
        encoding: tp.Union[None, str, EncodingT] = None,
        model: tp.Optional[str] = None,
        tokens_per_message: tp.Optional[int] = None,
        tokens_per_name: tp.Optional[int] = None,
        **kwargs,
    ) -> None:
        Tokenizer.__init__(
            self,
            encoding=encoding,
            model=model,
            tokens_per_message=tokens_per_message,
            tokens_per_name=tokens_per_name,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("tiktoken")
        from tiktoken import Encoding, get_encoding, encoding_for_model

        encoding = self.resolve_setting(encoding, "encoding", sub_path="chat")
        model = self.resolve_setting(model, "model", sub_path="chat")
        tokens_per_message = self.resolve_setting(tokens_per_message, "tokens_per_message", sub_path="chat")
        tokens_per_name = self.resolve_setting(tokens_per_name, "tokens_per_name", sub_path="chat")

        if isinstance(encoding, str):
            if encoding.startswith("model_or_"):
                try:
                    if model is None:
                        raise KeyError
                    encoding = encoding_for_model(model)
                except KeyError:
                    encoding = encoding[len("model_or_") :]
                    encoding = get_encoding(encoding) if "k_base" in encoding else encoding_for_model(encoding)
            elif isinstance(encoding, str):
                encoding = get_encoding(encoding) if "k_base" in encoding else encoding_for_model(encoding)
        checks.assert_instance_of(encoding, Encoding, arg_name="encoding")

        self._encoding = encoding
        self._tokens_per_message = tokens_per_message
        self._tokens_per_name = tokens_per_name

    @property
    def encoding(self) -> EncodingT:
        """Encoding."""
        return self._encoding

    @property
    def tokens_per_message(self) -> int:
        """Tokens per message."""
        return self._tokens_per_message

    @property
    def tokens_per_name(self) -> int:
        """Tokens per name."""
        return self._tokens_per_name

    def encode(self, text: str) -> tp.List[int]:
        return self.encoding.encode(text)

    def decode(self, tokens: tp.List[int]) -> str:
        return self.encoding.decode(tokens)

    def count_tokens_in_messages(self, messages: tp.ChatMessages) -> int:
        num_tokens = 0
        for message in messages:
            num_tokens += self.tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.encode(value))
                if key == "name":
                    num_tokens += self.tokens_per_name
        num_tokens += 3
        return num_tokens


def resolve_tokenizer(tokenizer: tp.TokenizerLike = None) -> tp.MaybeType[Tokenizer]:
    """Resolve a subclass or an instance of `Tokenizer`.

    The following strings are supported:

    * "tiktoken" (`TikTokenizer`)
    * A subclass or an instance of `Tokenizer`
    """
    if tokenizer is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        tokenizer = chat_cfg["tokenizer"]
    if isinstance(tokenizer, str):
        current_module = sys.modules[__name__]
        found_tokenizer = None
        for name, cls in inspect.getmembers(current_module, inspect.isclass):
            if name.endswith("Tokenizer"):
                _short_name = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == tokenizer.lower():
                    found_tokenizer = cls
                    break
        if found_tokenizer is None:
            raise ValueError(f"Invalid tokenizer: '{tokenizer}'")
        tokenizer = found_tokenizer
    if isinstance(tokenizer, type):
        checks.assert_subclass_of(tokenizer, Tokenizer, arg_name="tokenizer")
    else:
        checks.assert_instance_of(tokenizer, Tokenizer, arg_name="tokenizer")
    return tokenizer


# ############# Embeddings ############# #


class Embeddings(Configured):
    """Abstract class for embeddings.

    For defaults, see `chat` in `vectorbtpro._settings.knowledge`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    _settings_path: tp.SettingsPath = "knowledge"

    def __init__(self, **kwargs) -> None:
        Configured.__init__(self, **kwargs)

    @property
    def model(self) -> tp.Optional[str]:
        """Model."""
        return None

    def get_embedding(self, query: str) -> tp.List[float]:
        """Get embedding for a query."""
        raise NotImplementedError

    def get_embeddings(self, queries: tp.List[str]) -> tp.List[tp.List[float]]:
        """Get embeddings for multiple queries."""
        return [self.get_embedding(query) for query in queries]


class OpenAIEmbeddings(Embeddings):
    """Embeddings class for OpenAI.

    For defaults, see `chat.embeddings_configs.openai` in `vectorbtpro._settings.knowledge`."""

    _short_name = "openai"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.openai"

    def __init__(self, **kwargs) -> None:
        Embeddings.__init__(self, **kwargs)

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("openai")
        from openai import OpenAI

        openai_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        model = openai_config.pop("model", None)
        if model is None:
            raise ValueError("Must provide a model")
        client_arg_names = set(get_func_arg_names(OpenAI.__init__))
        client_kwargs = {}
        embeddings_kwargs = {}
        for k, v in openai_config.items():
            if k in client_arg_names:
                client_kwargs[k] = v
            else:
                embeddings_kwargs[k] = v
        client = OpenAI(**client_kwargs)

        self._model = model
        self._client = client
        self._embeddings_kwargs = embeddings_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> OpenAIT:
        """Client."""
        return self._client

    @property
    def embeddings_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `openai.resources.embeddings.Embeddings.create`."""
        return self._embeddings_kwargs

    def get_embedding(self, query: str) -> tp.List[float]:
        response = self.client.embeddings.create(input=query, model=self.model, **self.embeddings_kwargs)
        return response.data[0].embedding

    def get_embeddings(self, queries: tp.List[str]) -> tp.List[tp.List[float]]:
        response = self.client.embeddings.create(input=queries, model=self.model, **self.embeddings_kwargs)
        return [embedding.embedding for embedding in response.data]


class LiteLLMEmbeddings(Embeddings):
    """Embeddings class for LiteLLM.

    For defaults, see `chat.embeddings_configs.litellm` in `vectorbtpro._settings.knowledge`."""

    _short_name = "litellm"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.litellm"

    def __init__(self, **kwargs) -> None:
        Embeddings.__init__(self, **kwargs)

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("litellm")

        embedding_kwargs = merge_dicts(self.get_settings(inherit=False), kwargs)
        model = embedding_kwargs.pop("model", None)
        if model is None:
            raise ValueError("Must provide a model")

        self._model = model
        self._embedding_kwargs = embedding_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def embedding_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `litellm.embedding`."""
        return self._embedding_kwargs

    def get_embedding(self, query: str) -> tp.List[float]:
        from litellm import embedding

        response = embedding(self.model, input=query, **self.embedding_kwargs)
        return response.data[0]["embedding"]

    def get_embeddings(self, queries: tp.List[str]) -> tp.List[tp.List[float]]:
        from litellm import embedding

        response = embedding(self.model, input=queries, **self.embedding_kwargs)
        return [embedding["embedding"] for embedding in response.data]


class LlamaIndexEmbeddings(Embeddings):
    """Embeddings class for LlamaIndex.

    For defaults, see `chat.embeddings_configs.llama_index` in `vectorbtpro._settings.knowledge`."""

    _short_name = "llama_index"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.llama_index"

    def __init__(self, **kwargs) -> None:
        Embeddings.__init__(self, **kwargs)

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("llama_index")
        from llama_index.core.embeddings import BaseEmbedding

        llama_index_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        embedding = llama_index_config.pop("embedding", None)
        if embedding is None:
            raise ValueError("Must provide an embedding name or path")
        if isinstance(embedding, str):
            from importlib import import_module
            from vectorbtpro.utils.module_ import search_package

            if "." in embedding:
                path_parts = embedding.split(".")
                embedding = path_parts[-1]
                if len(path_parts) == 2:
                    module_name = "llama_index.embeddings." + path_parts[0]
                else:
                    module_name = ".".join(path_parts[:-1])
                module = import_module(module_name)
            else:
                import llama_index.embeddings

                module = llama_index.embeddings
            match_func = lambda k, v: isinstance(v, type) and issubclass(v, BaseEmbedding)
            candidates = search_package(module, match_func)
            class_found = False
            for k, v in candidates.items():
                if embedding.lower().replace("_", "") == k.rstrip("Embedding").lower():
                    embedding = v
                    class_found = True
                    break
            if not class_found:
                raise ValueError(f"Embedding '{embedding}' not found")
        if isinstance(embedding, type):
            embedding_name = embedding.__name__.lower()
            module_name = embedding.__module__
        else:
            checks.assert_instance_of(embedding, BaseEmbedding, arg_name="embedding")
            embedding_name = type(embedding).__name__.lower()
            module_name = type(embedding).__module__
        embedding_configs = llama_index_config.pop("embedding_configs", {})
        if embedding_name in embedding_configs:
            llama_index_config = deep_merge_dicts(llama_index_config, embedding_configs[embedding_name])
        elif module_name in embedding_configs:
            llama_index_config = deep_merge_dicts(llama_index_config, embedding_configs[module_name])
        if isinstance(embedding, type):
            embedding = embedding(**llama_index_config)
        elif len(kwargs) > 0:
            raise ValueError("Cannot apply config to already initialized embedding")
        model_name = llama_index_config.get("model_name", None)
        if model_name is None:
            func_kwargs = get_func_kwargs(type(embedding).__init__)
            model_name = func_kwargs.get("model_name", None)

        self._model = model_name
        self._embedding = embedding

    @property
    def model(self) -> tp.Optional[str]:
        return self._model

    @property
    def embedding(self) -> BaseEmbeddingT:
        """Embedding."""
        return self._embedding

    def get_embedding(self, query: str) -> tp.List[float]:
        return self.embedding.get_text_embedding(query)

    def get_embeddings(self, queries: tp.List[str]) -> tp.List[tp.List[float]]:
        return self.embedding.get_text_embedding_batch(queries)


def resolve_embeddings(embeddings: tp.EmbeddingsLike = None) -> tp.MaybeType[Embeddings]:
    """Resolve a subclass or an instance of `Embeddings`.

    The following strings are supported:

    * "openai" (`OpenAIEmbeddings`)
    * "litellm" (`LiteLLMEmbeddings`)
    * "llama_index" (`LlamaIndexEmbeddings`)
    * "auto": Any installed from above, in the same order
    * A subclass or an instance of `Embeddings`
    """
    if embeddings is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        embeddings = chat_cfg["embeddings"]
    if isinstance(embeddings, str):
        if embeddings.lower() == "auto":
            from vectorbtpro.utils.module_ import check_installed

            if check_installed("openai"):
                embeddings = "openai"
            elif check_installed("litellm"):
                embeddings = "litellm"
            elif check_installed("llama_index"):
                embeddings = "llama_index"
            else:
                raise ValueError("No packages for embeddings installed")
        current_module = sys.modules[__name__]
        found_embeddings = None
        for name, cls in inspect.getmembers(current_module, inspect.isclass):
            if name.endswith("Embeddings"):
                _short_name = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == embeddings.lower():
                    found_embeddings = cls
                    break
        if found_embeddings is None:
            raise ValueError(f"Invalid embeddings: '{embeddings}'")
        embeddings = found_embeddings
    if isinstance(embeddings, type):
        checks.assert_subclass_of(embeddings, Embeddings, arg_name="embeddings")
    else:
        checks.assert_instance_of(embeddings, Embeddings, arg_name="embeddings")
    return embeddings


def embed(query: tp.MaybeList[str], embeddings: tp.EmbeddingsLike = None, **kwargs) -> tp.MaybeEmbeddingsOutput:
    """Get embedding(s) for one or more queries.

    Resolves `embeddings` with `resolve_embeddings`. Keyword arguments are passed to either
    initialize a class or replace an instance of `Embeddings`."""
    embeddings = resolve_embeddings(embeddings=embeddings)
    if isinstance(embeddings, type):
        embeddings = embeddings(**kwargs)
    elif kwargs:
        embeddings = embeddings.replace(**kwargs)
    if isinstance(query, str):
        return embeddings.get_embedding(query)
    return embeddings.get_embeddings(query)


# ############# Completions ############# #


class Completions(Configured):
    """Abstract class for completions.

    For argument descriptions, see their properties, like `Completions.chat_history`.

    For defaults, see `chat` in `vectorbtpro._settings.knowledge`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    _settings_path: tp.SettingsPath = "knowledge"

    def __init__(
        self,
        context: str = "",
        chat_history: tp.Optional[tp.ChatHistory] = None,
        stream: tp.Optional[bool] = None,
        max_tokens: tp.Optional[int] = None,
        tokenizer: tp.TokenizerLike = None,
        tokenizer_kwargs: tp.KwargsLike = None,
        system_prompt: tp.Optional[str] = None,
        system_as_user: tp.Optional[bool] = None,
        context_prompt: tp.Optional[str] = None,
        formatter: tp.ContentFormatterLike = None,
        formatter_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
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
            context_prompt=context_prompt,
            formatter=formatter,
            formatter_kwargs=formatter_kwargs,
            template_context=template_context,
            silence_warnings=silence_warnings,
            **kwargs,
        )

        if chat_history is None:
            chat_history = []
        stream = self.resolve_setting(stream, "stream", sub_path="chat")
        max_tokens = self.resolve_setting(max_tokens, "max_tokens", sub_path="chat")
        tokenizer = self.resolve_setting(tokenizer, "tokenizer", default=None, sub_path="chat", sub_path_only=True)
        tokenizer_kwargs = self.resolve_setting(
            tokenizer_kwargs, "tokenizer_kwargs", default=None, sub_path="chat", sub_path_only=True, merge=True
        )
        system_prompt = self.resolve_setting(system_prompt, "system_prompt", sub_path="chat")
        system_as_user = self.resolve_setting(system_as_user, "system_as_user", sub_path="chat")
        context_prompt = self.resolve_setting(context_prompt, "context_prompt", sub_path="chat")
        formatter = self.resolve_setting(formatter, "formatter", default=None, sub_path="chat", sub_path_only=True)
        formatter_kwargs = self.resolve_setting(
            formatter_kwargs, "formatter_kwargs", default=None, sub_path="chat", merge=True, sub_path_only=True
        )
        template_context = self.resolve_setting(template_context, "template_context", sub_path="chat", merge=True)
        silence_warnings = self.resolve_setting(silence_warnings, "silence_warnings", sub_path="chat")

        tokenizer = resolve_tokenizer(tokenizer)
        formatter = resolve_formatter(formatter)

        self._context = context
        self._chat_history = chat_history
        self._stream = stream
        self._max_tokens = max_tokens
        self._tokenizer = tokenizer
        self._tokenizer_kwargs = tokenizer_kwargs
        self._system_prompt = system_prompt
        self._system_as_user = system_as_user
        self._context_prompt = context_prompt
        self._formatter = formatter
        self._formatter_kwargs = formatter_kwargs
        self._template_context = template_context
        self._silence_warnings = silence_warnings

    @property
    def context(self) -> str:
        """Context.

        Becomes a user message."""
        return self._context

    @property
    def chat_history(self) -> tp.ChatHistory:
        """Chat history.

        Must be list of dictionaries with proper roles.

        After generating a response, the output will be appended to this sequence as an assistant message."""
        return self._chat_history

    @property
    def stream(self) -> bool:
        """Whether to stream the response.

        When streaming, appends chunks one by one and displays the intermediate result.
        Otherwise, displays the entire message."""
        return self._stream

    @property
    def max_tokens(self) -> tp.Optional[int]:
        """Maximum number of tokens in messages."""
        return self._max_tokens

    @property
    def tokenizer(self) -> tp.MaybeType[Tokenizer]:
        """A subclass or an instance of `Tokenizer`.

        Resolved with `resolve_tokenizer`."""
        return self._tokenizer

    @property
    def tokenizer_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `Completions.tokenizer`.

        Used either to initialize a class or replace an instance of `Tokenizer`."""
        return self._tokenizer_kwargs

    @property
    def system_prompt(self) -> str:
        """System prompt.

        Precedes the context prompt."""
        return self._system_prompt

    @property
    def system_as_user(self) -> bool:
        """Whether to use the user role for the system message.

        Mainly for experimental models where the system role is not available."""
        return self._system_as_user

    @property
    def context_prompt(self) -> str:
        """Context prompt.

        A prompt template requiring the variable "context". The prompt can be either a custom template,
        or string or function that will become one. Once the prompt is evaluated, it becomes a user message."""
        return self._context_prompt

    @property
    def formatter(self) -> tp.MaybeType[ContentFormatter]:
        """A subclass or an instance of `vectorbtpro.utils.knowledge.formatting.ContentFormatter`.

        Resolved with `vectorbtpro.utils.knowledge.formatting.resolve_formatter`."""
        return self._formatter

    @property
    def formatter_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `Completions.formatter`.

        Used either to initialize a class or replace an instance of
        `vectorbtpro.utils.knowledge.formatting.ContentFormatter`."""
        return self._formatter_kwargs

    @property
    def template_context(self) -> tp.Kwargs:
        """Context used to substitute templates."""
        return self._template_context

    @property
    def silence_warnings(self) -> bool:
        """Whether to silence warnings."""
        return self._silence_warnings

    @property
    def model(self) -> tp.Optional[str]:
        """Model."""
        return None

    def get_chat_response(self, messages: tp.ChatMessages, **kwargs) -> tp.Any:
        """Get chat response to messages."""
        raise NotImplementedError

    def get_message_content(self, response: tp.Any) -> tp.Optional[str]:
        """Get content from a chat response."""
        raise NotImplementedError

    def get_stream_response(self, messages: tp.ChatMessages, **kwargs) -> tp.Any:
        """Get streaming response to messages."""
        raise NotImplementedError

    def get_delta_content(self, response: tp.Any) -> tp.Optional[str]:
        """Get content from a streaming response chunk."""
        raise NotImplementedError

    def prepare_messages(self, message: str) -> tp.ChatMessages:
        """Prepare messages for a completion."""
        context = self.context
        chat_history = self.chat_history
        max_tokens = self.max_tokens
        tokenizer = self.tokenizer
        tokenizer_kwargs = self.tokenizer_kwargs
        system_prompt = self.system_prompt
        system_as_user = self.system_as_user
        context_prompt = self.context_prompt
        template_context = self.template_context
        silence_warnings = self.silence_warnings

        if isinstance(tokenizer, type):
            if issubclass(tokenizer, TikTokenizer) and "page_title" not in tokenizer_kwargs:
                tokenizer_kwargs = dict(tokenizer_kwargs)
                tokenizer_kwargs["model"] = self.model
            tokenizer = tokenizer(**tokenizer_kwargs)
        elif tokenizer_kwargs:
            tokenizer = tokenizer.replace(**tokenizer_kwargs)

        if context:
            if isinstance(context_prompt, str):
                context_prompt = Sub(context_prompt)
            elif checks.is_function(context_prompt):
                context_prompt = RepFunc(context_prompt)
            elif not isinstance(context_prompt, CustomTemplate):
                raise TypeError(f"Context prompt must be a string, function, or template")
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
                num_tokens = tokenizer.count_tokens_in_messages(empty_messages)
                max_context_tokens = max(0, max_tokens - num_tokens)
                encoded_context = tokenizer.encode(context)
                if len(encoded_context) > max_context_tokens:
                    context = tokenizer.decode(encoded_context[:max_context_tokens])
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
        """Get completion for a message."""
        chat_history = self.chat_history
        stream = self.stream
        formatter = self.formatter
        formatter_kwargs = self.formatter_kwargs

        messages = self.prepare_messages(message)
        if self.stream:
            response = self.get_stream_response(messages)
        else:
            response = self.get_chat_response(messages)

        if isinstance(formatter, type):
            if issubclass(formatter, HTMLFileFormatter) and "page_title" not in formatter_kwargs:
                formatter_kwargs = dict(formatter_kwargs)
                formatter_kwargs["page_title"] = message
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
        else:
            content = self.get_message_content(response)
            if content is None:
                content = ""
            formatter.append_once(content)

        chat_history.append(dict(role="user", content=message))
        chat_history.append(dict(role="assistant", content=content))
        if isinstance(formatter, HTMLFileFormatter) and formatter.file_handle is not None:
            file_path = Path(formatter.file_handle.name)
        else:
            file_path = None
        if return_response:
            return response, file_path
        return file_path


class OpenAICompletions(Completions):
    """Completions class for OpenAI.

    Keyword arguments are distributed between the client call and the completion call.

    For defaults, see `chat.completions_configs.openai` in `vectorbtpro._settings.knowledge`."""

    _short_name = "openai"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.openai"

    def __init__(
        self,
        context: str = "",
        chat_history: tp.Optional[tp.ChatHistory] = None,
        stream: tp.Optional[bool] = None,
        max_tokens: tp.Optional[int] = None,
        tokenizer: tp.TokenizerLike = None,
        tokenizer_kwargs: tp.KwargsLike = None,
        system_prompt: tp.Optional[str] = None,
        system_as_user: tp.Optional[bool] = None,
        context_prompt: tp.Optional[str] = None,
        formatter: tp.ContentFormatterLike = None,
        formatter_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        Completions.__init__(
            self,
            context=context,
            chat_history=chat_history,
            stream=stream,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            system_prompt=system_prompt,
            system_as_user=system_as_user,
            context_prompt=context_prompt,
            formatter=formatter,
            formatter_kwargs=formatter_kwargs,
            template_context=template_context,
            silence_warnings=silence_warnings,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("openai")
        from openai import OpenAI

        openai_config = merge_dicts(self.get_settings(inherit=False), kwargs)
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
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> OpenAIT:
        """Client."""
        return self._client

    @property
    def completion_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `openai.resources.chat.completions_configs.Completions.create`."""
        return self._completion_kwargs

    def get_chat_response(self, messages: tp.ChatMessages) -> ChatCompletionT:
        return self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=False,
            **self.completion_kwargs,
        )

    def get_message_content(self, response: ChatCompletionT) -> tp.Optional[str]:
        return response.choices[0].message.content

    def get_stream_response(self, messages: tp.ChatMessages) -> StreamT:
        return self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=True,
            **self.completion_kwargs,
        )

    def get_delta_content(self, response_chunk: ChatCompletionChunkT) -> tp.Optional[str]:
        return response_chunk.choices[0].delta.content


class LiteLLMCompletions(Completions):
    """Completions class for LiteLLM.

    Keyword arguments are passed to the completion call.

    For defaults, see `chat.completions_configs.litellm` in `vectorbtpro._settings.knowledge`."""

    _short_name = "litellm"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.litellm"

    def __init__(
        self,
        context: str = "",
        chat_history: tp.Optional[tp.ChatHistory] = None,
        stream: tp.Optional[bool] = None,
        max_tokens: tp.Optional[int] = None,
        tokenizer: tp.TokenizerLike = None,
        tokenizer_kwargs: tp.KwargsLike = None,
        system_prompt: tp.Optional[str] = None,
        system_as_user: tp.Optional[bool] = None,
        context_prompt: tp.Optional[str] = None,
        formatter: tp.ContentFormatterLike = None,
        formatter_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        Completions.__init__(
            self,
            context=context,
            chat_history=chat_history,
            stream=stream,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            system_prompt=system_prompt,
            system_as_user=system_as_user,
            context_prompt=context_prompt,
            formatter=formatter,
            formatter_kwargs=formatter_kwargs,
            template_context=template_context,
            silence_warnings=silence_warnings,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("litellm")

        completion_kwargs = merge_dicts(self.get_settings(inherit=False), kwargs)
        model = completion_kwargs.pop("model", None)
        if model is None:
            raise ValueError("Must provide a model")

        self._model = model
        self._completion_kwargs = completion_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def completion_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `litellm.completion`."""
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
        return response.choices[0].message.content

    def get_stream_response(self, messages: tp.ChatMessages) -> CustomStreamWrapperT:
        from litellm import completion

        return completion(
            messages=messages,
            model=self.model,
            stream=True,
            **self.completion_kwargs,
        )

    def get_delta_content(self, response_chunk: ModelResponseT) -> tp.Optional[str]:
        return response_chunk.choices[0].delta.content


class LlamaIndexCompletions(Completions):
    """Completions class for LlamaIndex.

    LLM can be provided via `llm`, which can be either the name of the class, the path to the class
    (accepted are both "llama_index.xxx.yyy" and "xxx.yyy"), or a subclass or an instance of
    `llama_index.core.llms.LLM`. Case of strings doesn't matter.

    Keyword arguments are passed to the resolved LLM.

    For defaults, see `chat.completions_configs.llama_index` in `vectorbtpro._settings.knowledge`."""

    _short_name = "llama_index"

    _settings_path: tp.SettingsPath = "knowledge.chat.completions_configs.llama_index"

    def __init__(
        self,
        context: str = "",
        chat_history: tp.Optional[tp.ChatHistory] = None,
        stream: tp.Optional[bool] = None,
        max_tokens: tp.Optional[int] = None,
        tokenizer: tp.TokenizerLike = None,
        tokenizer_kwargs: tp.KwargsLike = None,
        system_prompt: tp.Optional[str] = None,
        system_as_user: tp.Optional[bool] = None,
        context_prompt: tp.Optional[str] = None,
        formatter: tp.ContentFormatterLike = None,
        formatter_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        Completions.__init__(
            self,
            context=context,
            chat_history=chat_history,
            stream=stream,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            system_prompt=system_prompt,
            system_as_user=system_as_user,
            context_prompt=context_prompt,
            formatter=formatter,
            formatter_kwargs=formatter_kwargs,
            template_context=template_context,
            silence_warnings=silence_warnings,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("llama_index")
        from llama_index.core.llms import LLM

        llama_index_config = merge_dicts(self.get_settings(inherit=False), kwargs)
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

    def get_chat_response(self, messages: tp.ChatMessages) -> ChatResponseT:
        from llama_index.core.llms import ChatMessage

        return self.llm.chat(list(map(lambda x: ChatMessage(**dict(x)), messages)))

    def get_message_content(self, response: ChatResponseT) -> tp.Optional[str]:
        return response.message.content

    def get_stream_response(self, messages: tp.ChatMessages) -> tp.Generator[ChatResponseT, None, None]:
        from llama_index.core.llms import ChatMessage

        return self.llm.stream_chat(list(map(lambda x: ChatMessage(**dict(x)), messages)))

    def get_delta_content(self, response_chunk: ChatResponseT) -> tp.Optional[str]:
        return response_chunk.delta


def resolve_completions(completions: tp.CompletionsLike = None) -> tp.MaybeType[Completions]:
    """Resolve a subclass or an instance of `Completions`.

    The following strings are supported:

    * "openai" (`OpenAICompletions`)
    * "litellm" (`LiteLLMCompletions`)
    * "llama_index" (`LlamaIndexCompletions`)
    * "auto": Any installed from above, in the same order
    * A subclass or an instance of `Completions`
    """
    if completions is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        completions = chat_cfg["completions"]
    if isinstance(completions, str):
        if completions.lower() == "auto":
            from vectorbtpro.utils.module_ import check_installed

            if check_installed("openai"):
                completions = "openai"
            elif check_installed("litellm"):
                completions = "litellm"
            elif check_installed("llama_index"):
                completions = "llama_index"
            else:
                raise ValueError("No packages for completions installed")
        current_module = sys.modules[__name__]
        found_completions = None
        for name, cls in inspect.getmembers(current_module, inspect.isclass):
            if name.endswith("Completions"):
                _short_name = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == completions.lower():
                    found_completions = cls
                    break
        if found_completions is None:
            raise ValueError(f"Invalid completions: '{completions}'")
        completions = found_completions
    if isinstance(completions, type):
        checks.assert_subclass_of(completions, Completions, arg_name="completions")
    else:
        checks.assert_instance_of(completions, Completions, arg_name="completions")
    return completions


def complete(message: str, completions: tp.CompletionsLike = None, **kwargs) -> tp.ChatOutput:
    """Get completion for a message.

    Resolves `completions` with `resolve_completions`. Keyword arguments are passed to either
    initialize a class or replace an instance of `Completions`."""
    completions = resolve_completions(completions=completions)
    if isinstance(completions, type):
        completions = completions(**kwargs)
    elif kwargs:
        completions = completions.replace(**kwargs)
    return completions.get_completion(message)


# ############# Contextable ############# #


class Contextable(HasSettings):
    """Abstract class that can be converted into a context."""

    _settings_path: tp.SettingsPath = "knowledge"

    def to_context(self, *args, **kwargs) -> str:
        """Convert to a context."""
        raise NotImplementedError

    def count_tokens(
        self,
        to_context_kwargs: tp.KwargsLike = None,
        tokenizer: tp.TokenizerLike = None,
        tokenizer_kwargs: tp.KwargsLike = None,
    ) -> int:
        """Count the number of tokens in the context."""
        to_context_kwargs = self.resolve_setting(to_context_kwargs, "to_context_kwargs", sub_path="chat", merge=True)
        tokenizer = self.resolve_setting(tokenizer, "tokenizer", default=None, sub_path="chat", sub_path_only=True)
        tokenizer_kwargs = self.resolve_setting(
            tokenizer_kwargs, "tokenizer_kwargs", default=None, sub_path="chat", sub_path_only=True, merge=True
        )

        context = self.to_context(**to_context_kwargs)
        tokenizer = resolve_tokenizer(tokenizer)
        if isinstance(tokenizer, type):
            tokenizer = tokenizer(**tokenizer_kwargs)
        elif tokenizer_kwargs:
            tokenizer = tokenizer.replace(**tokenizer_kwargs)
        return len(tokenizer.encode(context))

    @hybrid_method
    def chat(
        cls_or_self,
        message: str,
        chat_history: tp.Optional[tp.ChatHistory] = None,
        *,
        to_context_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.ChatOutput:
        """Chat with an LLM while using the instance as a context.

        Uses `complete`.

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

        to_context_kwargs = cls_or_self.resolve_setting(
            to_context_kwargs, "to_context_kwargs", sub_path="chat", merge=True
        )
        context = cls_or_self.to_context(**to_context_kwargs)
        return complete(message, context=context, chat_history=chat_history, **kwargs)
