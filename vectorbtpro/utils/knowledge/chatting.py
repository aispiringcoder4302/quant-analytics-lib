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

import hashlib
import inspect
import re
import sys
import warnings
from pathlib import Path

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.config import merge_dicts, flat_merge_dicts, deep_merge_dicts, Configured, HasSettings
from vectorbtpro.utils.decorators import memoized_method, hybrid_method
from vectorbtpro.utils.knowledge.formatting import ContentFormatter, HTMLFileFormatter, resolve_formatter
from vectorbtpro.utils.parsing import get_func_arg_names, get_func_kwargs
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.pickling import dumps
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
    from llama_index.core.node_parser import NodeParser as NodeParserT
except ImportError:
    BaseEmbeddingT = "BaseEmbedding"
    LLMT = "LLM"
    ChatMessageT = "ChatMessage"
    ChatResponseT = "ChatResponse"
    NodeParserT = "NodeParser"
try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from IPython.display import DisplayHandle as DisplayHandleT
except ImportError:
    DisplayHandleT = "DisplayHandle"

__all__ = [
    "Tokenizer",
    "TikTokenizer",
    "tokenize",
    "detokenize",
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
    "TextSplitter",
    "TokenSplitter",
    "SegmentSplitter",
    "LlamaIndexSplitter",
    "IndexDocument",
    "KnowledgeDocument",
    "IndexNode",
    "NodeIndex",
    "LocalIndex",
    "split_text",
    "Contextable",
]


# ############# Tokenizers ############# #


class Tokenizer(Configured):
    """Abstract class for tokenizers.

    For defaults, see `chat` in `vectorbtpro._settings.knowledge`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat"]

    def __init__(self, **kwargs) -> None:
        Configured.__init__(self, **kwargs)

    def encode(self, text: str) -> tp.Tokens:
        """Encode text into a list of tokens."""
        raise NotImplementedError

    def decode(self, tokens: tp.Tokens) -> str:
        """Decode a list of tokens into text."""
        raise NotImplementedError

    @memoized_method
    def encode_single(self, text: str) -> tp.Token:
        """Encode text into a single token."""
        tokens = self.encode(text)
        if len(tokens) > 1:
            raise ValueError("Text contains multiple tokens")
        return tokens[0]

    @memoized_method
    def decode_single(self, token: tp.Token) -> str:
        """Decode a single token into text."""
        return self.decode([token])

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text."""
        return len(self.encode(text))

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

        encoding = self.resolve_setting(encoding, "encoding")
        model = self.resolve_setting(model, "model")
        tokens_per_message = self.resolve_setting(tokens_per_message, "tokens_per_message")
        tokens_per_name = self.resolve_setting(tokens_per_name, "tokens_per_name")

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

    def encode(self, text: str) -> tp.Tokens:
        return self.encoding.encode(text)

    def decode(self, tokens: tp.Tokens) -> str:
        return self.encoding.decode(tokens)

    def count_tokens_in_messages(self, messages: tp.ChatMessages) -> int:
        num_tokens = 0
        for message in messages:
            num_tokens += self.tokens_per_message
            for key, value in message.items():
                num_tokens += self.count_tokens(value)
                if key == "name":
                    num_tokens += self.tokens_per_name
        num_tokens += 3
        return num_tokens


def resolve_tokenizer(tokenizer: tp.TokenizerLike = None) -> tp.MaybeType[Tokenizer]:
    """Resolve a subclass or an instance of `Tokenizer`.

    The following values are supported:

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


def tokenize(text: str, tokenizer: tp.TokenizerLike = None, **kwargs) -> tp.Tokens:
    """Tokenize text.

    Resolves `tokenizer` with `resolve_tokenizer`. Keyword arguments are passed to either
    initialize a class or replace an instance of `Tokenizer`."""
    tokenizer = resolve_tokenizer(tokenizer=tokenizer)
    if isinstance(tokenizer, type):
        tokenizer = tokenizer(**kwargs)
    elif kwargs:
        tokenizer = tokenizer.replace(**kwargs)
    return tokenizer.encode(text)


def detokenize(tokens: tp.Tokens, tokenizer: tp.TokenizerLike = None, **kwargs) -> str:
    """Detokenize text.

    Resolves `tokenizer` with `resolve_tokenizer`. Keyword arguments are passed to either
    initialize a class or replace an instance of `Tokenizer`."""
    tokenizer = resolve_tokenizer(tokenizer=tokenizer)
    if isinstance(tokenizer, type):
        tokenizer = tokenizer(**kwargs)
    elif kwargs:
        tokenizer = tokenizer.replace(**kwargs)
    return tokenizer.decode(tokens)


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

    def __init__(
        self,
        model: tp.Optional[str] = None,
        batch_size: tp.Optional[int] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(self, model=model, **kwargs)

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("openai")
        from openai import OpenAI

        openai_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = openai_config.pop("model", None)
        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        def_batch_size = openai_config.pop("batch_size", None)
        if batch_size is None:
            batch_size = def_batch_size
        def_show_progress = openai_config.pop("show_progress", None)
        if show_progress is None:
            show_progress = def_show_progress
        def_pbar_kwargs = openai_config.pop("pbar_kwargs", {})
        pbar_kwargs = merge_dicts(dict(prefix="get_embeddings"), def_pbar_kwargs, pbar_kwargs)

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
        self._batch_size = batch_size
        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs

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

    @property
    def batch_size(self) -> tp.Optional[int]:
        """Batch size.

        Set to None to disable batching."""
        return self._batch_size

    @property
    def show_progress(self) -> tp.Optional[bool]:
        """Whether to show progress bar."""
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.pbar.ProgressBar`."""
        return self._pbar_kwargs

    def get_embedding(self, query: str) -> tp.List[float]:
        response = self.client.embeddings.create(input=query, model=self.model, **self.embeddings_kwargs)
        return response.data[0].embedding

    def get_embeddings(self, queries: tp.List[str]) -> tp.List[tp.List[float]]:
        if self.batch_size is not None:
            batches = [queries[i:i + self.batch_size] for i in range(0, len(queries), self. batch_size)]
        else:
            batches = [queries]
        embeddings = []
        with ProgressBar(total=len(queries), show_progress=self.show_progress, **self.pbar_kwargs) as pbar:
            for batch in batches:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                    **self.embeddings_kwargs
                )
                embeddings.extend([embedding.embedding for embedding in response.data])
                pbar.update(len(batch))
        return embeddings


class LiteLLMEmbeddings(Embeddings):
    """Embeddings class for LiteLLM.

    For defaults, see `chat.embeddings_configs.litellm` in `vectorbtpro._settings.knowledge`."""

    _short_name = "litellm"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.litellm"

    def __init__(
        self,
        model: tp.Optional[str] = None,
        batch_size: tp.Optional[int] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(self, model=model, **kwargs)

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("litellm")

        litellm_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        def_model = litellm_config.pop("model", None)
        if model is None:
            model = def_model
        if model is None:
            raise ValueError("Must provide a model")
        def_batch_size = litellm_config.pop("batch_size", None)
        if batch_size is None:
            batch_size = def_batch_size
        def_show_progress = litellm_config.pop("show_progress", None)
        if show_progress is None:
            show_progress = def_show_progress
        def_pbar_kwargs = litellm_config.pop("pbar_kwargs", {})
        pbar_kwargs = merge_dicts(dict(prefix="get_embeddings"), def_pbar_kwargs, pbar_kwargs)

        self._model = model
        self._embedding_kwargs = litellm_config
        self._batch_size = batch_size
        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs

    @property
    def model(self) -> str:
        return self._model

    @property
    def embedding_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `litellm.embedding`."""
        return self._embedding_kwargs

    @property
    def batch_size(self) -> tp.Optional[int]:
        """Batch size.

        Set to None to disable batching."""
        return self._batch_size

    @property
    def show_progress(self) -> tp.Optional[bool]:
        """Whether to show progress bar."""
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.pbar.ProgressBar`."""
        return self._pbar_kwargs

    def get_embedding(self, query: str) -> tp.List[float]:
        from litellm import embedding

        response = embedding(self.model, input=query, **self.embedding_kwargs)
        return response.data[0]["embedding"]

    def get_embeddings(self, queries: tp.List[str]) -> tp.List[tp.List[float]]:
        from litellm import embedding

        if self.batch_size is not None:
            batches = [queries[i:i + self.batch_size] for i in range(0, len(queries), self. batch_size)]
        else:
            batches = [queries]
        embeddings = []
        with ProgressBar(total=len(queries), show_progress=self.show_progress, **self.pbar_kwargs) as pbar:
            for batch in batches:
                response = embedding(self.model, input=batch, **self.embedding_kwargs)
                embeddings.extend([embedding["embedding"] for embedding in response.data])
                pbar.update(len(batch))
        return embeddings


class LlamaIndexEmbeddings(Embeddings):
    """Embeddings class for LlamaIndex.

    For defaults, see `chat.embeddings_configs.llama_index` in `vectorbtpro._settings.knowledge`."""

    _short_name = "llama_index"

    _settings_path: tp.SettingsPath = "knowledge.chat.embeddings_configs.llama_index"

    def __init__(self, embedding: tp.Union[None, str, tp.MaybeType[BaseEmbeddingT]] = None, **kwargs) -> None:
        Embeddings.__init__(self, embedding=embedding, **kwargs)

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("llama_index")
        from llama_index.core.embeddings import BaseEmbedding

        llama_index_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        if embedding is None:
            embedding = llama_index_config.pop("embedding", None)
        if embedding is None:
            raise ValueError("Must provide an embedding name or path")
        if isinstance(embedding, str):
            import llama_index.embeddings
            from vectorbtpro.utils.module_ import search_package

            def _match_func(k, v):
                if isinstance(v, type) and issubclass(v, BaseEmbedding):
                    if "." in embedding:
                        if k.endswith(embedding):
                            return True
                    else:
                        if k.split(".")[-1].lower() == embedding.lower():
                            return True
                        if k.split(".")[-1].replace("Embedding", "").lower() == embedding.lower().replace("_", ""):
                            return True
                return False

            found_embedding = search_package(
                llama_index.embeddings,
                _match_func,
                path_attrs=True,
                return_first=True,
            )
            if found_embedding is None:
                raise ValueError(f"Embedding '{embedding}' not found")
            embedding = found_embedding
        if isinstance(embedding, type):
            checks.assert_subclass_of(embedding, BaseEmbedding, arg_name="embedding")
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

    The following values are supported:

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


def embed(query: tp.MaybeList[str], embeddings: tp.EmbeddingsLike = None, **kwargs) -> tp.MaybeList[tp.List[float]]:
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

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat"]

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
        stream = self.resolve_setting(stream, "stream")
        max_tokens = self.resolve_setting(max_tokens, "max_tokens")
        tokenizer = self.resolve_setting(tokenizer, "tokenizer", default=None)
        tokenizer_kwargs = self.resolve_setting(tokenizer_kwargs, "tokenizer_kwargs", default=None, merge=True)
        system_prompt = self.resolve_setting(system_prompt, "system_prompt")
        system_as_user = self.resolve_setting(system_as_user, "system_as_user")
        context_prompt = self.resolve_setting(context_prompt, "context_prompt")
        formatter = self.resolve_setting(formatter, "formatter", default=None)
        formatter_kwargs = self.resolve_setting(formatter_kwargs, "formatter_kwargs", default=None, merge=True)
        template_context = self.resolve_setting(template_context, "template_context", merge=True)
        silence_warnings = self.resolve_setting(silence_warnings, "silence_warnings")

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
            if issubclass(formatter, HTMLFileFormatter):
                formatter_kwargs = dict(formatter_kwargs)
                if "page_title" not in formatter_kwargs:
                    formatter_kwargs["page_title"] = message
                if "def_cache_suffix" not in formatter_kwargs:
                    formatter_kwargs["def_cache_suffix"] = "chat"
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
        model: tp.Optional[str] = None,
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
            model=model,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("openai")
        from openai import OpenAI

        openai_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        if model is None:
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
        model: tp.Optional[str] = None,
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
            model=model,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("litellm")

        completion_kwargs = merge_dicts(self.get_settings(inherit=False), kwargs)
        if model is None:
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

    LLM can be provided via `llm`, which can be either the name of the class (case doesn't matter),
    the path or its suffix to the class (case matters), or a subclass or an instance of
    `llama_index.core.llms.LLM`.

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
        llm: tp.Union[None, str, tp.MaybeType[LLMT]] = None,
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
            llm=llm,
            **kwargs,
        )

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("llama_index")
        from llama_index.core.llms import LLM

        llama_index_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        if llm is None:
            llm = llama_index_config.pop("llm", None)
        if llm is None:
            raise ValueError("Must provide an LLM name or path")
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
                        if k.split(".")[-1].lower() == llm.lower().replace("_", ""):
                            return True
                return False

            found_llm = search_package(
                llama_index.llms,
                _match_func,
                path_attrs=True,
                return_first=True,
            )
            if found_llm is None:
                raise ValueError(f"LLM '{llm}' not found")
            llm = found_llm
        if isinstance(llm, type):
            checks.assert_subclass_of(llm, LLM, arg_name="llm")
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

    def get_stream_response(self, messages: tp.ChatMessages) -> tp.Iterator[ChatResponseT]:
        from llama_index.core.llms import ChatMessage

        return self.llm.stream_chat(list(map(lambda x: ChatMessage(**dict(x)), messages)))

    def get_delta_content(self, response_chunk: ChatResponseT) -> tp.Optional[str]:
        return response_chunk.delta


def resolve_completions(completions: tp.CompletionsLike = None) -> tp.MaybeType[Completions]:
    """Resolve a subclass or an instance of `Completions`.

    The following values are supported:

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


# ############# Splitting ############# #


class TextSplitter(Configured):
    """Abstract class for text splitters.

    For defaults, see `chat` in `vectorbtpro._settings.knowledge`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat"]

    def __init__(self, **kwargs) -> None:
        Configured.__init__(self, **kwargs)

    def split(self, text: str) -> tp.TSRangeChunks:
        """Split text and yield start character and end character position of each chunk."""
        raise NotImplementedError

    def split_text(self, text: str) -> tp.TSTextChunks:
        """Split text and return text chunks."""
        for start, end in self.split(text):
            yield text[start:end]


class TokenSplitter(TextSplitter):
    """Splitter class for tokens.

    For defaults, see `chat.text_splitter_configs.token` in `vectorbtpro._settings.knowledge`."""

    _short_name = "token"

    _settings_path: tp.SettingsPath = "knowledge.chat.text_splitter_configs.token"

    def __init__(
        self,
        chunk_size: tp.Optional[int] = None,
        chunk_overlap: tp.Union[None, int, float] = None,
        tokenizer: tp.TokenizerLike = None,
        tokenizer_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        TextSplitter.__init__(
            self,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs,
        )

        chunk_size = self.resolve_setting(chunk_size, "chunk_size")
        chunk_overlap = self.resolve_setting(chunk_overlap, "chunk_overlap")
        tokenizer = self.resolve_setting(tokenizer, "tokenizer", default=None)
        tokenizer_kwargs = self.resolve_setting(tokenizer_kwargs, "tokenizer_kwargs", default=None, merge=True)

        tokenizer = resolve_tokenizer(tokenizer)
        if isinstance(tokenizer, type):
            tokenizer = tokenizer(**tokenizer_kwargs)
        elif tokenizer_kwargs:
            tokenizer = tokenizer.replace(**tokenizer_kwargs)
        if checks.is_float(chunk_overlap):
            if 0 <= abs(chunk_overlap) <= 1:
                chunk_overlap = chunk_overlap * chunk_size
            elif not chunk_overlap.is_integer():
                raise TypeError("Floating number for chunk_overlap must be between 0 and 1")
            chunk_overlap = int(chunk_overlap)

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._tokenizer = tokenizer

    @property
    def chunk_size(self) -> int:
        """Maximum number of tokens per chunk."""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Number of overlapping tokens between chunks.

        Can also be provided as a floating number relative to `SegmentSplitter.chunk_size`."""
        return self._chunk_overlap

    @property
    def tokenizer(self) -> Tokenizer:
        """An instance of `Tokenizer`."""
        return self._tokenizer

    def split_into_tokens(self, text: str) -> tp.TSRangeChunks:
        """Split text into tokens."""
        tokens = self.tokenizer.encode(text)
        last_end = 0
        for token in tokens:
            _text = self.tokenizer.decode_single(token)
            start = last_end
            end = start + len(_text)
            yield start, end
            last_end = end

    def split(self, text: str) -> tp.TSRangeChunks:
        return self.split_into_tokens(text)


class SegmentSplitter(TokenSplitter):
    """Splitter class for segments based on separators.

    If a segment is too big, the next separator within the same layer is taken to split the segment
    into smaller segments. If a segment is too big and there are no segments previously added
    to the chunk, or, if the number of tokens is less than the minimal count, the next layer is taken.
    To split into tokens, set any separator to None. To split into characters, use an empty string.

    For defaults, see `chat.text_splitter_configs.segment` in `vectorbtpro._settings.knowledge`."""

    _short_name = "segment"

    _settings_path: tp.SettingsPath = "knowledge.chat.text_splitter_configs.segment"

    def __init__(
        self,
        separators: tp.MaybeList[tp.MaybeList[tp.Optional[str]]] = None,
        min_chunk_size: tp.Union[None, int, float] = None,
        **kwargs,
    ) -> None:
        TokenSplitter.__init__(
            self,
            separators=separators,
            min_chunk_size=min_chunk_size,
            **kwargs,
        )

        separators = self.resolve_setting(separators, "separators")
        min_chunk_size = self.resolve_setting(min_chunk_size, "min_chunk_size")

        if not isinstance(separators, list):
            separators = [separators]
        else:
            separators = list(separators)
        for layer in range(len(separators)):
            if not isinstance(separators[layer], list):
                separators[layer] = [separators[layer]]
            else:
                separators[layer] = list(separators[layer])
        if checks.is_float(min_chunk_size):
            if 0 <= abs(min_chunk_size) <= 1:
                min_chunk_size = min_chunk_size * self.chunk_size
            elif not min_chunk_size.is_integer():
                raise TypeError("Floating number for min_chunk_size must be between 0 and 1")
            min_chunk_size = int(min_chunk_size)

        self._separators = separators
        self._min_chunk_size = min_chunk_size

    @property
    def separators(self) -> tp.List[tp.List[tp.Optional[str]]]:
        """Nested list of separators grouped into layers."""
        return self._separators

    @property
    def min_chunk_size(self) -> int:
        """Minimum number of tokens per chunk.

        Can also be provided as a floating number relative to `SegmentSplitter.chunk_size`."""
        return self._min_chunk_size

    def split_into_segments(self, text: str, separator: tp.Optional[str] = None) -> tp.TSRangeChunks:
        """Split text into segments."""
        if not separator:
            if separator is None:
                for start, end in self.split_into_tokens(text):
                    yield start, end
            else:
                for i in range(len(text)):
                    yield i, i + 1
        else:
            last_end = 0

            for match in re.finditer(separator, text):
                start, end = match.span()
                if start > last_end:
                    _text = text[last_end:start]
                    yield last_end, start

                _text = text[start:end]
                yield start, end
                last_end = end

            if last_end < len(text):
                _text = text[last_end:]
                yield last_end, len(text)

    def split(self, text: str) -> tp.TSRangeChunks:
        if not text:
            yield 0, 0
            return None
        if self.tokenizer.count_tokens(text) <= self.chunk_size:
            yield 0, len(text)
            return None

        current_layer = 0
        chunk_start = 0
        chunk_tokens = []
        stable_token_count = 0
        stable_char_count = 0
        remaining_text = text
        overlap_segments = []

        while remaining_text:
            current_separators = self.separators[current_layer]
            current_start = chunk_start
            current_text = remaining_text
            current_segments = overlap_segments
            overlap_segments = []

            for separator in current_separators:
                segments = self.split_into_segments(current_text, separator=separator)
                current_text = ""
                finished = False

                for segment in segments:
                    segment_start = current_start + segment[0]
                    segment_end = current_start + segment[1]

                    if not chunk_tokens:
                        segment_text = text[segment_start:segment_end]
                        new_chunk_tokens = self.tokenizer.encode(segment_text)
                        new_stable_token_count = 0
                        new_stable_char_count = 0
                    elif not stable_token_count:
                        chunk_text = text[chunk_start:segment_end]
                        new_chunk_tokens = self.tokenizer.encode(chunk_text)
                        new_stable_token_count = 0
                        new_stable_char_count = 0
                        min_token_count = min(len(chunk_tokens), len(new_chunk_tokens))
                        for i in range(min_token_count):
                            if chunk_tokens[i] == new_chunk_tokens[i]:
                                new_stable_token_count += 1
                                new_stable_char_count += len(self.tokenizer.decode_single(chunk_tokens[i]))
                            else:
                                break
                    else:
                        stable_tokens = chunk_tokens[:stable_token_count]
                        unstable_start = chunk_start + stable_char_count
                        partial_text = text[unstable_start:segment_end]
                        partial_tokens = self.tokenizer.encode(partial_text)
                        new_chunk_tokens = stable_tokens + partial_tokens
                        new_stable_token_count = stable_token_count
                        new_stable_char_count = stable_char_count
                        min_token_count = min(len(chunk_tokens), len(new_chunk_tokens))
                        for i in range(stable_token_count, min_token_count):
                            if chunk_tokens[i] == new_chunk_tokens[i]:
                                new_stable_token_count += 1
                                new_stable_char_count += len(self.tokenizer.decode_single(chunk_tokens[i]))
                            else:
                                break

                    if len(new_chunk_tokens) > self.chunk_size:
                        current_text = text[segment_start:segment_end]
                        current_start = segment_start
                        finished = False
                        break
                    else:
                        current_segments.append((segment_start, segment_end))
                        chunk_tokens = new_chunk_tokens
                        stable_token_count = new_stable_token_count
                        stable_char_count = new_stable_char_count
                        finished = True

                if finished:
                    break

            if current_segments and len(chunk_tokens) >= self.min_chunk_size:
                chunk_start = current_segments[0][0]
                chunk_end = current_segments[-1][1]
                yield chunk_start, chunk_end

                if self.chunk_overlap:
                    if len(chunk_tokens) <= self.chunk_overlap:
                        raise ValueError(
                            "Chunk overlap is equal to or greater than the total number of tokens in the chunk. "
                            "Reduce chunk_overlap, or increase min_chunk_size or the separator granularity."
                        )

                    chunk_tokens = chunk_tokens[-self.chunk_overlap :]
                    chunk_start = chunk_end - len(self.tokenizer.decode(chunk_tokens))
                    overlap_segments = [(chunk_start, chunk_end)]
                else:
                    chunk_tokens = []
                    chunk_start = chunk_end

                stable_token_count = 0
                stable_char_count = 0
                remaining_text = text[chunk_start:]
                current_layer = 0

            else:
                current_layer += 1
                if current_layer == len(self.separators):
                    remaining_tokens = self.tokenizer.encode(remaining_text)
                    if len(remaining_tokens) < self.chunk_size:
                        if current_segments:
                            chunk_start = current_segments[0][0]
                            chunk_end = current_segments[-1][1]
                            yield chunk_start, chunk_end
                        break
                    if len(chunk_tokens) < self.min_chunk_size:
                        raise ValueError(
                            "Total number of tokens in the chunk is less than the minimal chunk size. "
                            "Increase min_chunk_size or the separator granularity."
                        )
                    break


class LlamaIndexSplitter(TextSplitter):
    """Splitter class based on a node parser from LlamaIndex.

    For defaults, see `chat.text_splitter_configs.llama_index` in `vectorbtpro._settings.knowledge`."""

    _short_name = "llama_index"

    _settings_path: tp.SettingsPath = "knowledge.chat.text_splitter_configs.llama_index"

    def __init__(self, node_parser: tp.Union[None, str, NodeParserT] = None, **kwargs) -> None:
        TextSplitter.__init__(self, **kwargs)

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("llama_index")
        from llama_index.core.node_parser import NodeParser

        llama_index_config = merge_dicts(self.get_settings(inherit=False), kwargs)
        if node_parser is None:
            node_parser = llama_index_config.pop("node_parser", None)

        if isinstance(node_parser, str):
            import llama_index.core.node_parser
            from vectorbtpro.utils.module_ import search_package

            def _match_func(k, v):
                if isinstance(v, type) and issubclass(v, NodeParser):
                    if "." in node_parser:
                        if k.endswith(node_parser):
                            return True
                    else:
                        if k.split(".")[-1].lower() == node_parser.lower():
                            return True
                        if k.split(".")[-1].replace("Splitter", "").replace(
                            "NodeParser", ""
                        ).lower() == node_parser.lower().replace("_", ""):
                            return True
                return False

            found_node_parser = search_package(
                llama_index.core.node_parser,
                _match_func,
                path_attrs=True,
                return_first=True,
            )
            if found_node_parser is None:
                raise ValueError(f"Node parser '{node_parser}' not found")
            node_parser = found_node_parser
        if isinstance(node_parser, type):
            checks.assert_subclass_of(node_parser, NodeParser, arg_name="node_parser")
            node_parser_name = node_parser.__name__.lower()
            module_name = node_parser.__module__
        else:
            checks.assert_instance_of(node_parser, NodeParser, arg_name="node_parser")
            node_parser_name = type(node_parser).__name__.lower()
            module_name = type(node_parser).__module__
        node_parser_configs = llama_index_config.pop("node_parser_configs", {})
        if node_parser_name in node_parser_configs:
            llama_index_config = deep_merge_dicts(llama_index_config, node_parser_configs[node_parser_name])
        elif module_name in node_parser_configs:
            llama_index_config = deep_merge_dicts(llama_index_config, node_parser_configs[module_name])
        if isinstance(node_parser, type):
            node_parser = node_parser(**llama_index_config)
        elif len(kwargs) > 0:
            raise ValueError("Cannot apply config to already initialized node parser")
        model_name = llama_index_config.get("model_name", None)
        if model_name is None:
            func_kwargs = get_func_kwargs(type(node_parser).__init__)
            model_name = func_kwargs.get("model_name", None)

        self._model = model_name
        self._node_parser = node_parser

    @property
    def node_parser(self) -> NodeParserT:
        """An instance of `llama_index.core.node_parser.interface.NodeParser`."""
        return self._node_parser

    def split(self, text: str) -> tp.TSRangeChunks:
        for text_chunk in self.split_text(text):
            start = text.find(text_chunk)
            if start == -1:
                end = -1
            else:
                end = start + len(text_chunk)
            yield start, end

    def split_text(self, text: str) -> tp.TSTextChunks:
        from llama_index.core.schema import Document

        nodes = self.node_parser.get_nodes_from_documents([Document(text=text)])
        for node in nodes:
            yield node.text


def resolve_text_splitter(text_splitter: tp.TextSplitterLike = None) -> tp.MaybeType[TextSplitter]:
    """Resolve a subclass or an instance of `TextSplitter`.

    The following values are supported:

    * "token" (`TokenSplitter`)
    * "segment" (`SegmentSplitter`)
    * "llama_index" (`LlamaIndexSplitter`)
    * A subclass or an instance of `TextSplitter`
    """
    if text_splitter is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        text_splitter = chat_cfg["text_splitter"]
    if isinstance(text_splitter, str):
        current_module = sys.modules[__name__]
        found_text_splitter = None
        for name, cls in inspect.getmembers(current_module, inspect.isclass):
            if name.endswith("Splitter"):
                _short_name = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == text_splitter.lower():
                    found_text_splitter = cls
                    break
        if found_text_splitter is None:
            raise ValueError(f"Invalid text splitter: '{text_splitter}'")
        text_splitter = found_text_splitter
    if isinstance(text_splitter, type):
        checks.assert_subclass_of(text_splitter, TextSplitter, arg_name="text_splitter")
    else:
        checks.assert_instance_of(text_splitter, TextSplitter, arg_name="text_splitter")
    return text_splitter


def split_text(text: str, text_splitter: tp.TextSplitterLike = None, **kwargs) -> tp.List[str]:
    """Split text.

    Resolves `text_splitter` with `resolve_text_splitter`. Keyword arguments are passed to either
    initialize a class or replace an instance of `TextSplitter`."""
    text_splitter = resolve_text_splitter(text_splitter=text_splitter)
    if isinstance(text_splitter, type):
        text_splitter = text_splitter(**kwargs)
    elif kwargs:
        text_splitter = text_splitter.replace(**kwargs)
    return list(text_splitter.split_text(text))


# ############# Indexing ############# #

IndexDocumentT = tp.TypeVar("IndexDocumentT", bound="IndexDocument")


@define
class IndexDocument(DefineMixin):
    """Abstract class for index documents.

    An index document stores data and an identifier, and exposes methods for getting and splitting content.
    If an identifier wasn't provided, generates the MD5 hash of the data."""

    data: tp.Any = define.field()
    """Data."""

    id_: str = define.field(default=None)
    """Document identifier."""

    def get_text(self) -> tp.Optional[str]:
        """Get text.

        Returns None if no text."""
        raise NotImplementedError

    def get_metadata(self) -> tp.Optional[tp.Any]:
        """Get metadata.

        Returns None if no metadata."""
        raise NotImplementedError

    def get_metadata_content(self) -> tp.Optional[str]:
        """Get metadata content.

        Returns None if no metadata."""
        raise NotImplementedError

    def get_content(self) -> tp.Optional[str]:
        """Get content (text + metadata).

        Returns None if no text or metadata."""
        raise NotImplementedError

    def split(self: IndexDocumentT) -> tp.List[IndexDocumentT]:
        """Split document into multiple documents."""
        raise NotImplementedError

    def __attrs_post_init__(self):
        if self.id_ is None:
            new_id = self.generate_id()
            object.__setattr__(self, "id_", new_id)

    def generate_id(self) -> str:
        """Generate a unique identifier."""
        return hashlib.md5(dumps(self.data)).hexdigest()

    @property
    def hash_key(self) -> tuple:
        return (self.id_,)


@define
class KnowledgeDocument(IndexDocument, DefineMixin):
    """Class for knowledge documents."""

    text_path: tp.Optional[tp.PathLikeKey] = define.field(default=None)
    """Path to the text field."""

    metadata_path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = define.field(default=None)
    """Path(s) to metadata fields.
    
    To not include metadata, set it to empty list."""

    skip_missing: bool = define.field(default=True)
    """Set missing text or metadata to None rather than raise an error."""

    split_text_kwargs: tp.KwargsLike = define.field(factory=dict)
    """Keyword arguments passed to `split_text`."""

    dump_kwargs: tp.KwargsLike = define.field(factory=dict)
    """Keyword arguments passed to `vectorbtpro.utils.formatting.dump`."""

    metadata_template: str = define.field(default="---\n{metadata_content}\n---\n\n")
    """Metadata template."""

    content_template: str = define.field(default="{metadata_content}{text}")
    """Content template."""

    def get_text(self) -> tp.Optional[str]:
        from vectorbtpro.utils.search import get_pathlike_key

        if self.text_path is not None:
            try:
                text = get_pathlike_key(self.data, self.text_path, keep_path=False)
            except (KeyError, IndexError, AttributeError) as e:
                if not self.skip_missing:
                    raise e
                return None
            if text is None:
                return None
            if not isinstance(text, str):
                raise TypeError(f"Text field must be a string, not {type(text)}")
            return text
        if self.data is None:
            return None
        if not isinstance(self.data, str):
            raise TypeError(f"If text path is not provided, data item must be a string, not {type(self.data)}")
        return self.data

    def get_metadata(self) -> tp.Optional[tp.Any]:
        from vectorbtpro.utils.search import get_pathlike_key, remove_pathlike_key

        if self.metadata_path is not None:
            if isinstance(self.metadata_path, list):
                xs = []
                for p in self.metadata_path:
                    try:
                        xs.append(get_pathlike_key(self.data, p, keep_path=True))
                    except (KeyError, IndexError, AttributeError) as e:
                        if not self.skip_missing:
                            raise e
                        continue
                if len(xs) == 0:
                    return None
                return deep_merge_dicts(*xs)
            else:
                try:
                    return get_pathlike_key(self.data, self.metadata_path, keep_path=False)
                except (KeyError, IndexError, AttributeError) as e:
                    if not self.skip_missing:
                        raise e
                    return None
        if self.text_path is not None:
            try:
                return remove_pathlike_key(self.data, self.text_path, make_copy=True)
            except (KeyError, IndexError, AttributeError) as e:
                if not self.skip_missing:
                    raise e
                return self.data
        return None

    def get_metadata_content(self) -> tp.Optional[str]:
        from vectorbtpro.utils.formatting import dump

        metadata = self.get_metadata()
        if metadata is None:
            return None
        return dump(metadata, **self.dump_kwargs)

    def get_content(self) -> tp.Optional[str]:
        text = self.get_text()
        metadata_content = self.get_metadata_content()
        if text is None and metadata_content is None:
            return None
        if text is None:
            text = ""
        if metadata_content is None:
            metadata_content = ""
        if metadata_content:
            metadata_content = self.metadata_template.format(metadata_content=metadata_content)
        return self.content_template.format(metadata_content=metadata_content, text=text)

    def split(self: IndexDocumentT) -> tp.List[IndexDocumentT]:
        from vectorbtpro.utils.search import set_pathlike_key

        text = self.get_text()
        if text is None:
            return [self]
        text_chunks = split_text(text, **self.split_text_kwargs)
        document_chunks = []
        prev_keys = []
        for text_chunk in text_chunks:
            if self.text_path is not None:
                data_chunk = set_pathlike_key(
                    self.data,
                    self.text_path,
                    text_chunk,
                    make_copy=True,
                    prev_keys=prev_keys,
                )
            else:
                data_chunk = text_chunk
            document_chunks.append(self.replace(data=data_chunk, id_=None))
        return document_chunks


@define
class IndexNode(DefineMixin):
    """Class for index nodes.

    An index node stores the identifier of a document, its relationships to other documents,
    as well as the embedding (if provided). It's a different entity from a document to avoid
    persisting document's data."""

    id_: str = define.field()
    """Node identifier."""

    parent_id: tp.Optional[str] = define.field(default=None)
    """Parent node identifier."""

    child_ids: tp.List[str] = define.field(factory=list)
    """Child node identifiers."""

    embedding: tp.Optional[tp.List[int]] = define.field(default=None, repr=lambda x: f"List[{len(x)}]" if x else None)
    """Embedding."""

    @property
    def hash_key(self) -> tuple:
        return (self.id_,)


def embedding_similarity(
    emb1: tp.Union[tp.List[float], np.ndarray],
    emb2: tp.Union[tp.MaybeList[tp.List[float]], np.ndarray],
    metric: tp.Union[str, tp.Callable] = "cosine",
) -> tp.Union[float, np.ndarray]:
    """Compute similarity scores between two embeddings, which can be either single or multiple.

    Supported metrics are 'cosine', 'euclidean', and 'dot'. A metric can also be a callable that should
    take two and return one 2-dim NumPy array."""
    emb1 = np.asarray(emb1)
    emb2 = np.asarray(emb2)
    emb1_single = emb1.ndim == 1
    emb2_single = emb2.ndim == 1
    if emb1_single:
        emb1 = emb1.reshape(1, -1)
    if emb2_single:
        emb2 = emb2.reshape(1, -1)

    if metric.lower() == "cosine":
        emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
        emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
        emb1_norm = np.nan_to_num(emb1_norm)
        emb2_norm = np.nan_to_num(emb2_norm)
        similarity_matrix = np.dot(emb1_norm, emb2_norm.T)
    elif metric.lower() == "euclidean":
        diff = emb1[:, np.newaxis, :] - emb2[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        similarity_matrix = 1 / (distances + 1e-10)
    elif metric.lower() == "dot":
        similarity_matrix = np.dot(emb1, emb2.T)
    elif callable(metric):
        similarity_matrix = metric(emb1, emb2)
    else:
        raise ValueError(f"Invalid metric: '{metric}'")

    if emb1_single and emb2_single:
        return float(similarity_matrix[0, 0])
    if emb1_single or emb2_single:
        return similarity_matrix.flatten()
    return similarity_matrix


class NodeIndex(Configured):
    """Abstract class for node indexes.

    For defaults, see `chat` in `vectorbtpro._settings.knowledge`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat"]

    def __init__(self, **kwargs) -> None:
        Configured.__init__(self, **kwargs)

        self._index = {}

    @property
    def index(self) -> tp.Dict[str, IndexNode]:
        """Dictionary with index nodes keyed by their ids."""
        return self._index

    def save_index(self) -> tp.Path:
        """Save index."""
        raise NotImplementedError

    def load_index(self) -> tp.Dict[str, IndexNode]:
        """Load and return index."""
        raise NotImplementedError

    def load_update_index(self) -> None:
        """Load and update index."""
        self._index = self.load_index()

    def add_node(self, node: IndexNode) -> None:
        """Add node to the index."""
        self.index[node.id_] = node

    def embed_documents(self, documents: tp.MaybeIterable[IndexDocument], **kwargs) -> None:
        """Convert document(s) to nodes, embed them, and add them to the index.

        Keyword arguments are passed to `embed`."""
        node_id_contents = {}
        if isinstance(documents, IndexDocument):
            documents = [documents]
        for document in documents:
            if document.id_ not in self.index:
                document_chunks = document.split()
                child_ids = []
                parent_node = IndexNode(document.id_, child_ids=child_ids)
                self.index[parent_node.id_] = parent_node
                for document_chunk in document_chunks:
                    if document_chunk.id_ != document.id_:
                        if document_chunk.id_ not in self.index:
                            child_node = IndexNode(document_chunk.id_, parent_id=document.id_)
                            self.index[child_node.id_] = child_node
                            child_ids.append(child_node.id_)
                            node_id_contents[child_node.id_] = document_chunk.get_content()
                if not parent_node.child_ids:
                    node_id_contents[parent_node.id_] = document.get_content()

        if node_id_contents:
            embeddings = embed(list(node_id_contents.values()), **kwargs)
            node_id_embeddings = dict(zip(node_id_contents.keys(), embeddings))
            for node_id, embedding in node_id_embeddings.items():
                new_node = self.index[node_id].replace(embedding=embedding)
                self.index[node_id] = new_node


class LocalIndex(NodeIndex):
    """Index class based locally.

    For defaults, see `chat.node_index_configs.local` in `vectorbtpro._settings.knowledge`."""

    _short_name = "local"

    _settings_path: tp.SettingsPath = "knowledge.chat.node_index_configs.local"

    def __init__(
        self,
        path: tp.Optional[tp.PathLike] = None,
        compression: tp.Union[None, bool, str] = None,
        save_kwargs: tp.KwargsLike = None,
        load_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        NodeIndex.__init__(
            self,
            path=path,
            compression=compression,
            save_kwargs=save_kwargs,
            load_kwargs=load_kwargs,
            **kwargs,
        )

        path = self.resolve_setting(path, "path")
        compression = self.resolve_setting(compression, "compression")
        save_kwargs = self.resolve_setting(save_kwargs, "save_kwargs", merge=True)
        load_kwargs = self.resolve_setting(load_kwargs, "load_kwargs", merge=True)

        self._path = path
        self._compression = compression
        self._save_kwargs = save_kwargs
        self._load_kwargs = load_kwargs

    @property
    def path(self) -> tp.Optional[tp.Path]:
        """Path to the index file."""
        return self._path

    @property
    def compression(self) -> tp.CompressionLike:
        """Compression."""
        return self._compression

    @property
    def save_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.path_.save`."""
        return self._save_kwargs

    @property
    def load_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.path_.load`."""
        return self._load_kwargs

    def save_index(self) -> tp.Path:
        from vectorbtpro.utils.pickling import save

        return save(
            self.index,
            path=self.path,
            compression=self.compression,
            **self.save_kwargs,
        )

    def load_index(self) -> tp.Dict[int, IndexNode]:
        from vectorbtpro.utils.pickling import load

        return load(
            path=self.path,
            compression=self.compression,
            **self.load_kwargs,
        )


# ############# Contexting ############# #


class Contextable(HasSettings):
    """Abstract class that can be converted into a context."""

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat"]

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
        to_context_kwargs = self.resolve_setting(to_context_kwargs, "to_context_kwargs", merge=True)
        tokenizer = self.resolve_setting(tokenizer, "tokenizer", default=None)
        tokenizer_kwargs = self.resolve_setting(tokenizer_kwargs, "tokenizer_kwargs", default=None, merge=True)

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

        to_context_kwargs = cls_or_self.resolve_setting(to_context_kwargs, "to_context_kwargs", merge=True)
        context = cls_or_self.to_context(**to_context_kwargs)
        return complete(message, context=context, chat_history=chat_history, **kwargs)
