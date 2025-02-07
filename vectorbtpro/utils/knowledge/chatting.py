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
from collections.abc import MutableMapping

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.config import merge_dicts, flat_merge_dicts, Configured, HasSettings, ExtSettingsPath
from vectorbtpro.utils.decorators import memoized_method, hybrid_method
from vectorbtpro.utils.knowledge.formatting import ContentFormatter, HTMLFileFormatter, resolve_formatter
from vectorbtpro.utils.parsing import get_func_arg_names, get_func_kwargs
from vectorbtpro.utils.template import CustomTemplate, Sub, RepFunc, substitute_templates

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
try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from lmdbm import Lmdb as LmdbT
except ImportError:
    LmdbT = "Lmdb"

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
    "split_text",
    "StoreObject",
    "StoreDocument",
    "TextDocument",
    "StoreEmbedding",
    "ObjectStore",
    "DictStore",
    "MemoryStore",
    "FileStore",
    "LMDBStore",
    "EmbeddedDocument",
    "ScoredDocument",
    "DocumentRanker",
    "embed_documents",
    "rank_documents",
    "Rankable",
    "Contextable",
    "RankContextable",
]


# ############# Tokenizers ############# #


class Tokenizer(Configured):
    """Abstract class for tokenizers.

    For defaults, see `chat` in `vectorbtpro._settings.knowledge`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat"]

    def __init__(self, template_context: tp.KwargsLike = None, **kwargs) -> None:
        Configured.__init__(self, template_context=template_context, **kwargs)

        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        self._template_context = template_context

    @property
    def template_context(self) -> tp.Kwargs:
        """Context used to substitute templates."""
        return self._template_context

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

    def __init__(self, template_context: tp.KwargsLike = None, **kwargs) -> None:
        Configured.__init__(self, template_context=template_context, **kwargs)

        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        self._template_context = template_context

    @property
    def template_context(self) -> tp.Kwargs:
        """Context used to substitute templates."""
        return self._template_context

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
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            batch_size=batch_size,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            template_context=template_context,
            **kwargs,
        )

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
        pbar_kwargs = merge_dicts(def_pbar_kwargs, pbar_kwargs)

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
        from vectorbtpro.utils.pbar import ProgressBar

        if self.batch_size is not None:
            batches = [queries[i : i + self.batch_size] for i in range(0, len(queries), self.batch_size)]
        else:
            batches = [queries]
        embeddings = []
        pbar_kwargs = merge_dicts(dict(prefix="get_embeddings"), self.pbar_kwargs)
        with ProgressBar(total=len(queries), show_progress=self.show_progress, **pbar_kwargs) as pbar:
            for batch in batches:
                response = self.client.embeddings.create(input=batch, model=self.model, **self.embeddings_kwargs)
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
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            model=model,
            batch_size=batch_size,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            template_context=template_context,
            **kwargs,
        )

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
        pbar_kwargs = merge_dicts(def_pbar_kwargs, pbar_kwargs)

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
        from vectorbtpro.utils.pbar import ProgressBar

        if self.batch_size is not None:
            batches = [queries[i : i + self.batch_size] for i in range(0, len(queries), self.batch_size)]
        else:
            batches = [queries]
        embeddings = []
        pbar_kwargs = merge_dicts(dict(prefix="get_embeddings"), self.pbar_kwargs)
        with ProgressBar(total=len(queries), show_progress=self.show_progress, **pbar_kwargs) as pbar:
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

    def __init__(
        self,
        embedding: tp.Union[None, str, tp.MaybeType[BaseEmbeddingT]] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Embeddings.__init__(
            self,
            embedding=embedding,
            template_context=template_context,
            **kwargs,
        )

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
            llama_index_config = merge_dicts(llama_index_config, embedding_configs[embedding_name])
        elif module_name in embedding_configs:
            llama_index_config = merge_dicts(llama_index_config, embedding_configs[module_name])
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
        silence_warnings: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
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
            silence_warnings=silence_warnings,
            template_context=template_context,
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
        silence_warnings = self.resolve_setting(silence_warnings, "silence_warnings")
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

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
        self._silence_warnings = silence_warnings
        self._template_context = template_context

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
    def silence_warnings(self) -> bool:
        """Whether to silence warnings."""
        return self._silence_warnings

    @property
    def template_context(self) -> tp.Kwargs:
        """Context used to substitute templates."""
        return self._template_context

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
        template_context = self.template_context

        messages = self.prepare_messages(message)
        if self.stream:
            response = self.get_stream_response(messages)
        else:
            response = self.get_chat_response(messages)

        if isinstance(formatter, type):
            formatter_kwargs = dict(formatter_kwargs)
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
        silence_warnings: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
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
            silence_warnings=silence_warnings,
            template_context=template_context,
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
        silence_warnings: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
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
            silence_warnings=silence_warnings,
            template_context=template_context,
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
        silence_warnings: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
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
            silence_warnings=silence_warnings,
            template_context=template_context,
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
            llama_index_config = merge_dicts(llama_index_config, llm_configs[llm_name])
        elif module_name in llm_configs:
            llama_index_config = merge_dicts(llama_index_config, llm_configs[module_name])
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

    def __init__(self, template_context: tp.KwargsLike = None, **kwargs) -> None:
        Configured.__init__(self, template_context=template_context, **kwargs)

        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        self._template_context = template_context

    @property
    def template_context(self) -> tp.Kwargs:
        """Context used to substitute templates."""
        return self._template_context

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
            tokenizer_kwargs = dict(tokenizer_kwargs)
            tokenizer_kwargs["template_context"] = merge_dicts(
                self.template_context, tokenizer_kwargs.get("template_context", None)
            )
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

    def __init__(
        self,
        node_parser: tp.Union[None, str, NodeParserT] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        TextSplitter.__init__(self, template_context=template_context, **kwargs)

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
            llama_index_config = merge_dicts(llama_index_config, node_parser_configs[node_parser_name])
        elif module_name in node_parser_configs:
            llama_index_config = merge_dicts(llama_index_config, node_parser_configs[module_name])
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


# ############# Storing ############# #


StoreObjectT = tp.TypeVar("StoreObjectT", bound="StoreObject")


@define
class StoreObject(DefineMixin):
    """Class for objects to be managed by a store."""

    id_: str = define.field()
    """Object identifier."""

    @property
    def hash_key(self) -> tuple:
        return (self.id_,)


StoreDocumentT = tp.TypeVar("StoreDocumentT", bound="StoreDocument")


@define
class StoreDocument(StoreObject, DefineMixin):
    """Abstract class for documents to be stored."""

    data: tp.Any = define.field()
    """Data."""

    template_context: tp.KwargsLike = define.field(factory=dict)
    """Context used to substitute templates."""

    @classmethod
    def id_from_data(cls, data: tp.Any) -> str:
        """Generate a unique identifier from data."""
        from vectorbtpro.utils.pickling import dumps

        return hashlib.md5(dumps(data)).hexdigest()

    @classmethod
    def from_data(
        cls: tp.Type[StoreDocumentT],
        data: tp.Any,
        id_: tp.Optional[str] = None,
        **kwargs,
    ) -> StoreDocumentT:
        """Create an instance of `StoreDocument` from data."""
        if id_ is None:
            id_ = cls.id_from_data(data)
        return cls(id_, data, **kwargs)

    def __attrs_post_init__(self):
        if self.id_ is None:
            new_id = self.id_from_data(self.data)
            object.__setattr__(self, "id_", new_id)

    def get_content(self, for_embed: bool = False) -> tp.Optional[str]:
        """Get content.

        Returns None if there's no content."""
        raise NotImplementedError

    def split(self: StoreDocumentT) -> tp.List[StoreDocumentT]:
        """Split document into multiple documents."""
        raise NotImplementedError


TextDocumentT = tp.TypeVar("TextDocumentT", bound="TextDocument")


def def_metadata_template(metadata_content: str) -> str:
    """Default metadata template"""
    if metadata_content.endswith("\n"):
        return "---\n{metadata_content}---\n\n".format(metadata_content=metadata_content)
    return "---\n{metadata_content}\n---\n\n".format(metadata_content=metadata_content)


@define
class TextDocument(StoreDocument, HasSettings, DefineMixin):
    """Class for text documents."""

    text_path: tp.Optional[tp.PathLikeKey] = define.field(default=None)
    """Path to the text field."""

    split_text_kwargs: tp.KwargsLike = define.field(factory=dict)
    """Keyword arguments passed to `split_text`."""

    excl_metadata: tp.Union[bool, tp.MaybeList[tp.PathLikeKey]] = define.field(default=False)
    """Whether to exclude metadata and which fields to exclude.
    
    If False, metadata becomes everything except text."""

    excl_embed_metadata: tp.Union[None, bool, tp.MaybeList[tp.PathLikeKey]] = define.field(default=None)
    """Whether to exclude metadata and which fields to exclude for embeddings.

    If None, becomes `TextDocument.excl_metadata`."""

    skip_missing: bool = define.field(default=True)
    """Set missing text or metadata to None rather than raise an error."""

    dump_kwargs: tp.KwargsLike = define.field(factory=dict)
    """Keyword arguments passed to `vectorbtpro.utils.formatting.dump`."""

    metadata_template: tp.CustomTemplateLike = define.field(
        default=RepFunc(def_metadata_template, eval_id="metadata_template")
    )
    """Metadata template.
    
    Must be suitable for formatting via the `format()` method."""

    content_template: tp.CustomTemplateLike = define.field(
        default=Sub("${metadata_content}${text}", eval_id="content_template")
    )
    """Content template.
    
    Must be suitable for formatting via the `format()` method."""

    def get_text(self) -> tp.Optional[str]:
        """Get text.

        Returns None if no text."""
        from vectorbtpro.utils.search_ import get_pathlike_key

        if self.data is None:
            return None
        if isinstance(self.data, str):
            return self.data
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
        raise TypeError(f"If text path is not provided, data item must be a string, not {type(self.data)}")

    def get_metadata(self, for_embed: bool = False) -> tp.Optional[tp.Any]:
        """Get metadata.

        Returns None if no metadata."""
        from vectorbtpro.utils.search_ import remove_pathlike_key

        if self.data is None or isinstance(self.data, str) or self.text_path is None:
            return None
        prev_keys = []
        data = self.data
        try:
            data = remove_pathlike_key(data, self.text_path, make_copy=True, prev_keys=prev_keys)
        except (KeyError, IndexError, AttributeError) as e:
            if not self.skip_missing:
                raise e
        excl_metadata = self.excl_metadata
        if for_embed:
            excl_embed_metadata = self.excl_embed_metadata
            if excl_embed_metadata is None:
                excl_embed_metadata = excl_metadata
            excl_metadata = excl_embed_metadata
        if isinstance(excl_metadata, bool):
            if excl_metadata:
                return None
            return data
        if not excl_metadata:
            return data
        if not isinstance(excl_metadata, list):
            excl_metadata = [excl_metadata]
        for p in excl_metadata:
            try:
                data = remove_pathlike_key(data, p, make_copy=True, prev_keys=prev_keys)
            except (KeyError, IndexError, AttributeError) as e:
                continue
        return data

    def get_metadata_content(self, for_embed: bool = False) -> tp.Optional[str]:
        """Get metadata content.

        Returns None if no metadata."""
        from vectorbtpro.utils.formatting import dump

        metadata = self.get_metadata(for_embed=for_embed)
        if metadata is None:
            return None
        return dump(metadata, **self.dump_kwargs)

    def get_content(self, for_embed: bool = False) -> tp.Optional[str]:
        text = self.get_text()
        metadata_content = self.get_metadata_content(for_embed=for_embed)
        if text is None and metadata_content is None:
            return None
        if text is None:
            text = ""
        if metadata_content is None:
            metadata_content = ""
        if metadata_content:
            metadata_template = self.metadata_template
            if isinstance(metadata_template, str):
                metadata_template = Sub(metadata_template)
            elif checks.is_function(metadata_template):
                metadata_template = RepFunc(metadata_template)
            elif not isinstance(metadata_template, CustomTemplate):
                raise TypeError(f"Metadata template must be a string, function, or template")
            template_context = flat_merge_dicts(
                dict(metadata_content=metadata_content),
                self.template_context,
            )
            metadata_content = metadata_template.substitute(template_context, eval_id="metadata_template")
        content_template = self.content_template
        if isinstance(content_template, str):
            content_template = Sub(content_template)
        elif checks.is_function(content_template):
            content_template = RepFunc(content_template)
        elif not isinstance(content_template, CustomTemplate):
            raise TypeError(f"Content template must be a string, function, or template")
        template_context = flat_merge_dicts(
            dict(metadata_content=metadata_content, text=text),
            self.template_context,
        )
        return content_template.substitute(template_context, eval_id="content_template")

    def split(self: TextDocumentT) -> tp.List[TextDocumentT]:
        from vectorbtpro.utils.search_ import set_pathlike_key

        text = self.get_text()
        if text is None:
            return [self]
        text_chunks = split_text(text, **self.split_text_kwargs)
        document_chunks = []
        for text_chunk in text_chunks:
            if not isinstance(self.data, str) and self.text_path is not None:
                data_chunk = set_pathlike_key(
                    self.data,
                    self.text_path,
                    text_chunk,
                    make_copy=True,
                )
            else:
                data_chunk = text_chunk
            document_chunks.append(self.replace(data=data_chunk, id_=None))
        return document_chunks


@define
class StoreEmbedding(StoreObject, DefineMixin):
    """Class for embeddings to be stored."""

    parent_id: tp.Optional[str] = define.field(default=None)
    """Parent object identifier."""

    child_ids: tp.List[str] = define.field(factory=list)
    """Child object identifiers."""

    embedding: tp.Optional[tp.List[int]] = define.field(default=None, repr=lambda x: f"List[{len(x)}]" if x else None)
    """Embedding."""


class MetaObjectStore(type(Configured), type(MutableMapping)):
    """Metaclass for `ObjectStore`."""

    pass


class ObjectStore(Configured, MutableMapping, metaclass=MetaObjectStore):
    """Abstract class for managing an object store.

    For defaults, see `chat` in `vectorbtpro._settings.knowledge`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the class."""

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat"]

    def __init__(
        self,
        store_id: tp.Optional[str] = None,
        purge_on_open: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            store_id=store_id,
            template_context=template_context,
            **kwargs,
        )

        store_id = self.resolve_setting(store_id, "store_id")
        purge_on_open = self.resolve_setting(purge_on_open, "purge_on_open")
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        self._store_id = store_id
        self._purge_on_open = purge_on_open
        self._template_context = template_context

        self._opened = False
        self._enter_calls = 0

    @property
    def store_id(self) -> str:
        """Store id."""
        return self._store_id

    @property
    def purge_on_open(self) -> bool:
        """Whether to purge on open."""
        return self._purge_on_open

    @property
    def template_context(self) -> tp.Kwargs:
        """Context used to substitute templates."""
        return self._template_context

    @property
    def opened(self) -> bool:
        """Whether the store has been opened."""
        return self._opened

    @property
    def enter_calls(self) -> int:
        """Number of enter calls."""
        return self._enter_calls

    def open(self) -> None:
        """Open the store."""
        if self.opened:
            self.close()
        if self.purge_on_open:
            self.purge()
        self._opened = True

    def commit(self) -> None:
        """Commit changes."""
        pass

    def close(self) -> None:
        """Close the store."""
        self.commit()
        self._opened = False

    def purge(self) -> None:
        """Purge the store."""
        self.close()

    def __getitem__(self, id_: str) -> StoreObjectT:
        raise NotImplementedError

    def __setitem__(self, id_: str, obj: StoreObjectT) -> None:
        raise NotImplementedError

    def __delitem__(self, id_: str) -> None:
        raise NotImplementedError

    def __iter__(self) -> tp.Iterator[str]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __enter__(self) -> tp.Self:
        if not self.opened:
            self.open()
        self._enter_calls += 1
        return self

    def __exit__(self, *args) -> None:
        if self.enter_calls == 1:
            self.close()
            self._close_on_exit = False
        self._enter_calls -= 1
        if self.enter_calls < 0:
            self._enter_calls = 0


class DictStore(ObjectStore):
    """Store class based on a dictionary.

    For defaults, see `chat.obj_store_configs.memory` in `vectorbtpro._settings.knowledge`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "dict"

    _settings_path: tp.SettingsPath = "knowledge.chat.obj_store_configs.dict"

    def __init__(self, **kwargs) -> None:
        ObjectStore.__init__(self, **kwargs)

        self._store = {}

    @property
    def store(self) -> tp.Dict[str, StoreObjectT]:
        """Store dictionary."""
        return self._store

    def __getitem__(self, id_: str) -> StoreObjectT:
        return self.store[id_]

    def __setitem__(self, id_: str, obj: StoreObjectT) -> None:
        self.store[id_] = obj

    def __delitem__(self, id_: str) -> None:
        del self.store[id_]

    def __iter__(self) -> tp.Iterator[str]:
        return iter(self.store)

    def __len__(self) -> int:
        return len(self.store)


memory_store: tp.Dict[str, tp.Dict[str, StoreObjectT]] = {}
"""Object store by store id for `MemoryStore`."""


class MemoryStore(DictStore):
    """Store class based in memory.

    Commits changes to `memory_store`.

    For defaults, see `chat.obj_store_configs.memory` in `vectorbtpro._settings.knowledge`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "memory"

    _settings_path: tp.SettingsPath = "knowledge.chat.obj_store_configs.memory"

    def __init__(self, **kwargs) -> None:
        DictStore.__init__(self, **kwargs)

    @property
    def store(self) -> tp.Dict[str, StoreObjectT]:
        """Store dictionary."""
        return self._store

    def store_exists(self) -> bool:
        """Whether store exists."""
        return self.store_id in memory_store

    def open(self) -> None:
        ObjectStore.open(self)
        if self.store_exists():
            self._store = dict(memory_store[self.store_id])

    def commit(self) -> None:
        ObjectStore.commit(self)
        memory_store[self.store_id] = dict(self.store)

    def purge(self) -> None:
        ObjectStore.purge(self)
        if self.store_exists():
            del memory_store[self.store_id]


class FileStore(DictStore):
    """Store class based on files.

    Either commits changes to a single file (with index id being the file name), or commits the initial
    changes to the base file and any other change to patch file(s) (with index id being the directory name).
    Also, can mirror the store to `memory_store` to avoid repeated disk I/O operations within a single session.

    For defaults, see `chat.obj_store_configs.file` in `vectorbtpro._settings.knowledge`."""

    _short_name = "file"

    _settings_path: tp.SettingsPath = "knowledge.chat.obj_store_configs.file"

    def __init__(
        self,
        dir_path: tp.Optional[tp.PathLike] = None,
        compression: tp.Union[None, bool, str] = None,
        save_kwargs: tp.KwargsLike = None,
        load_kwargs: tp.KwargsLike = None,
        use_patching: tp.Optional[bool] = None,
        consolidate: tp.Optional[bool] = None,
        memory_mirror: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        DictStore.__init__(
            self,
            dir_path=dir_path,
            compression=compression,
            save_kwargs=save_kwargs,
            load_kwargs=load_kwargs,
            use_patching=use_patching,
            consolidate=consolidate,
            memory_mirror=memory_mirror,
            **kwargs,
        )

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
        compression = self.resolve_setting(compression, "compression")
        save_kwargs = self.resolve_setting(save_kwargs, "save_kwargs", merge=True)
        load_kwargs = self.resolve_setting(load_kwargs, "load_kwargs", merge=True)
        use_patching = self.resolve_setting(use_patching, "use_patching")
        consolidate = self.resolve_setting(consolidate, "consolidate")
        memory_mirror = self.resolve_setting(memory_mirror, "memory_mirror")

        self._dir_path = dir_path
        self._compression = compression
        self._save_kwargs = save_kwargs
        self._load_kwargs = load_kwargs
        self._use_patching = use_patching
        self._consolidate = consolidate
        self._memory_mirror = memory_mirror

        self._store_changes = {}
        self._new_keys = set()
        self._memory_store = MemoryStore(store_id=str(self.store_path.resolve()))

    @property
    def dir_path(self) -> tp.Optional[tp.Path]:
        """Path to the directory."""
        return self._dir_path

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

    @property
    def use_patching(self) -> bool:
        """Whether to use directory with patch files or create a single file."""
        return self._use_patching

    @property
    def consolidate(self) -> bool:
        """Whether to consolidate patch files."""
        return self._consolidate

    @property
    def memory_mirror(self) -> bool:
        """Whether to mirror the store in memory for the next instance."""
        return self._memory_mirror

    @property
    def store_changes(self) -> tp.Dict[str, StoreObjectT]:
        """Store with new or modified objects only."""
        return self._store_changes

    @property
    def new_keys(self) -> tp.Set[str]:
        """Keys that haven't been added to the store."""
        return self._new_keys

    @property
    def memory_store(self) -> MemoryStore:
        """Memory store."""
        return self._memory_store

    @property
    def store_path(self) -> tp.Path:
        """Path to the directory with patch files or a single file."""
        dir_path = self.dir_path
        if dir_path is None:
            dir_path = "."
        dir_path = Path(dir_path)
        return dir_path / self.store_id

    def get_next_patch_path(self) -> tp.Path:
        """Get path to the next patch file to be saved."""
        indices = []
        for file in self.store_path.glob("patch_*"):
            indices.append(int(file.stem.split("_")[1]))
        next_index = max(indices) + 1 if indices else 0
        return self.store_path / f"patch_{next_index}"

    def store_exists(self) -> bool:
        if self.memory_mirror and self.memory_store.store_exists():
            return True
        return self.store_path.exists()

    def open(self) -> None:
        DictStore.open(self)
        if self.memory_mirror and self.memory_store.store_exists():
            self.memory_store.open()
            self.store.clear()
            self.store.update(self.memory_store.store)
        else:
            if self.store_path.exists():
                from vectorbtpro.utils.pickling import load

                if self.store_path.is_dir():
                    store = {}
                    store.update(
                        load(
                            path=self.store_path / "base",
                            compression=self.compression,
                            **self.load_kwargs,
                        )
                    )
                    patch_paths = sorted(self.store_path.glob("patch_*"), key=lambda f: int(f.stem.split("_")[1]))
                    for patch_path in patch_paths:
                        store.update(
                            load(
                                path=patch_path,
                                compression=self.compression,
                                **self.load_kwargs,
                            )
                        )
                else:
                    store = load(
                        path=self.store_path,
                        compression=self.compression,
                        **self.load_kwargs,
                    )
                self._store = store
                if self.memory_mirror:
                    self.memory_store.store.clear()
                    self.memory_store.store.update(store)

        self._store_changes = {}
        self._new_keys = set()
        self._consolidate = False

    def commit(self) -> tp.Optional[tp.Path]:
        DictStore.commit(self)
        if self.memory_mirror:
            self.memory_store.commit()
        from vectorbtpro.utils.pickling import save

        file_path = None
        if self.use_patching:
            base_path = self.store_path / "base"
            if self.consolidate:
                self.purge()
                file_path = save(
                    self.store,
                    path=base_path,
                    compression=self.compression,
                    **self.save_kwargs,
                )
            elif self.store_changes:
                if self.store_path.exists() and self.store_path.is_file():
                    self.purge()
                if not base_path.exists():
                    file_path = save(
                        self.store_changes,
                        path=base_path,
                        compression=self.compression,
                        **self.save_kwargs,
                    )
                else:
                    file_path = save(
                        self.store_changes,
                        path=self.get_next_patch_path(),
                        compression=self.compression,
                        **self.save_kwargs,
                    )
        else:
            if self.consolidate or self.store_changes:
                if self.store_path.exists() and self.store_path.is_dir():
                    self.purge()
                file_path = save(
                    self.store,
                    path=self.store_path,
                    compression=self.compression,
                    **self.save_kwargs,
                )

        self._store_changes = {}
        self._new_keys = set()
        self._consolidate = False
        return file_path

    def purge(self) -> None:
        DictStore.purge(self)
        if self.memory_mirror:
            self.memory_store.purge()
        from vectorbtpro.utils.path_ import remove_file, remove_dir

        if self.store_path.exists():
            if self.store_path.is_dir():
                remove_dir(self.store_path, with_contents=True)
            else:
                remove_file(self.store_path)

        self._store_changes = {}
        self._new_keys = set()
        self._consolidate = False

    def __setitem__(self, id_: str, obj: StoreObjectT) -> None:
        if obj.id_ not in self:
            self.new_keys.add(obj.id_)
        self.store_changes[obj.id_] = obj
        DictStore.__setitem__(self, id_, obj)
        if self.memory_mirror:
            self.memory_store[id_] = obj

    def __delitem__(self, id_: str) -> None:
        if id_ in self.new_keys:
            del self.store_changes[id_]
            self.new_keys.remove(id_)
        else:
            if id_ in self.store_changes:
                del self.store_changes[id_]
        DictStore.__delitem__(self, id_)
        if self.memory_mirror:
            del self.memory_store[id_]


class LMDBStore(ObjectStore):
    """Store class based on LMDB (Lightning Memory-Mapped Database).

    Uses [lmdbm](https://pypi.org/project/lmdbm/) package.

    For defaults, see `chat.obj_store_configs.lmdb` in `vectorbtpro._settings.knowledge`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "lmdb"

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    _settings_path: tp.SettingsPath = "knowledge.chat.obj_store_configs.lmdb"

    def __init__(
        self,
        dir_path: tp.Optional[tp.PathLike] = None,
        mkdir_kwargs: tp.KwargsLike = None,
        dumps_kwargs: tp.KwargsLike = None,
        loads_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        ObjectStore.__init__(self, dir_path=dir_path, mkdir_kwargs=mkdir_kwargs, **kwargs)

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("lmdbm")

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
        mkdir_kwargs = self.resolve_setting(mkdir_kwargs, "mkdir_kwargs", merge=True)
        dumps_kwargs = self.resolve_setting(dumps_kwargs, "dumps_kwargs", merge=True)
        loads_kwargs = self.resolve_setting(loads_kwargs, "loads_kwargs", merge=True)
        open_kwargs = merge_dicts(self.get_settings(inherit=False), kwargs)
        for arg_name in get_func_arg_names(ObjectStore.__init__) + get_func_arg_names(type(self).__init__):
            if arg_name in open_kwargs:
                del open_kwargs[arg_name]

        self._dir_path = dir_path
        self._mkdir_kwargs = mkdir_kwargs
        self._dumps_kwargs = dumps_kwargs
        self._loads_kwargs = loads_kwargs
        self._open_kwargs = open_kwargs

        self._db = None

    @property
    def dir_path(self) -> tp.Optional[tp.Path]:
        """Path to the directory."""
        return self._dir_path

    @property
    def mkdir_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.path_.check_mkdir`."""
        return self._mkdir_kwargs

    @property
    def dumps_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.pickling.dumps`."""
        return self._dumps_kwargs

    @property
    def loads_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.pickling.loads`."""
        return self._loads_kwargs

    @property
    def open_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `lmdbm.lmdbm.Lmdb.open`."""
        return self._open_kwargs

    @property
    def db_path(self) -> tp.Path:
        """Path to the database."""
        dir_path = self.dir_path
        if dir_path is None:
            dir_path = "."
        dir_path = Path(dir_path)
        return dir_path / self.store_id

    @property
    def db(self) -> tp.Optional[LmdbT]:
        """Database."""
        return self._db

    def open(self) -> None:
        ObjectStore.open(self)
        from lmdbm import Lmdb
        from vectorbtpro.utils.path_ import check_mkdir

        check_mkdir(self.db_path.parent, **self.mkdir_kwargs)
        self._db = Lmdb.open(str(self.db_path.resolve()), **self.open_kwargs)

    def commit(self) -> None:
        self.db.sync()

    def close(self) -> None:
        ObjectStore.close(self)
        if self.db:
            self.db.close()
        self._db = None

    def purge(self) -> None:
        ObjectStore.purge(self)
        from vectorbtpro.utils.path_ import remove_dir

        remove_dir(self.db_path, missing_ok=True, with_contents=True)

    def encode(self, obj: StoreObjectT) -> bytes:
        """Encode an object."""
        from vectorbtpro.utils.pickling import dumps

        return dumps(obj, **self.dumps_kwargs)

    def decode(self, bytes_: bytes) -> StoreObjectT:
        """Decode an object."""
        from vectorbtpro.utils.pickling import loads

        return loads(bytes_, **self.loads_kwargs)

    def __getitem__(self, id_: str) -> StoreObjectT:
        return self.decode(self.db[id_])

    def __setitem__(self, id_: str, obj: StoreObjectT) -> None:
        self.db[id_] = self.encode(obj)

    def __delitem__(self, id_: str) -> None:
        del self.db[id_]

    def __iter__(self) -> tp.Iterator[str]:
        return iter(self.db)

    def __len__(self) -> int:
        return len(self.db)


def resolve_obj_store(obj_store: tp.ObjectStoreLike = None) -> tp.MaybeType[ObjectStore]:
    """Resolve a subclass or an instance of `ObjectStore`.

    The following values are supported:

    * "dict" (`DictStore`)
    * "memory" (`MemoryStore`)
    * "file" (`FileStore`)
    * "lmdb" (`LMDBStore`)
    * A subclass or an instance of `ObjectStore`
    """
    if obj_store is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["chat"]
        obj_store = chat_cfg["obj_store"]
    if isinstance(obj_store, str):
        current_module = sys.modules[__name__]
        found_obj_store = None
        for name, cls in inspect.getmembers(current_module, inspect.isclass):
            if name.endswith("Store"):
                _short_name = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == obj_store.lower():
                    found_obj_store = cls
                    break
        if found_obj_store is None:
            raise ValueError(f"Invalid object store: '{obj_store}'")
        obj_store = found_obj_store
    if isinstance(obj_store, type):
        checks.assert_subclass_of(obj_store, ObjectStore, arg_name="obj_store")
    else:
        checks.assert_instance_of(obj_store, ObjectStore, arg_name="obj_store")
    return obj_store


# ############# Ranking ############# #


@define
class EmbeddedDocument(DefineMixin):
    """Abstract class for embedded documents."""

    document: StoreDocument = define.field()
    """Document."""

    embedding: tp.Optional[tp.List[float]] = define.field(default=None)
    """Embedding."""

    child_documents: tp.List["EmbeddedDocument"] = define.field(factory=list)
    """Embedded child documents."""


@define
class ScoredDocument(DefineMixin):
    """Abstract class for scored documents."""

    document: StoreDocument = define.field()
    """Document."""

    score: float = define.field(default=float("nan"))
    """Score."""

    child_documents: tp.List["ScoredDocument"] = define.field(factory=list)
    """Scored child documents."""


class DocumentRanker(Configured):
    """Class for embedding, scoring, and ranking documents.

    For defaults, see `chat` in `vectorbtpro._settings.knowledge`."""

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat"]

    def __init__(
        self,
        embeddings: tp.EmbeddingsLike = None,
        embeddings_kwargs: tp.KwargsLike = None,
        doc_store: tp.TokenizerLike = None,
        doc_store_kwargs: tp.KwargsLike = None,
        emb_store: tp.TokenizerLike = None,
        emb_store_kwargs: tp.KwargsLike = None,
        score_func: tp.Union[None, str, tp.Callable] = None,
        score_agg_func: tp.Union[None, str, tp.Callable] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            embeddings=embeddings,
            embeddings_kwargs=embeddings_kwargs,
            doc_store=doc_store,
            doc_store_kwargs=doc_store_kwargs,
            emb_store=emb_store,
            emb_store_kwargs=emb_store_kwargs,
            score_func=score_func,
            score_agg_func=score_agg_func,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            template_context=template_context,
            **kwargs,
        )

        embeddings = self.resolve_setting(embeddings, "embeddings", default=None)
        embeddings_kwargs = self.resolve_setting(embeddings_kwargs, "embeddings_kwargs", default=None, merge=True)
        doc_store = self.resolve_setting(doc_store, "doc_store", default=None)
        doc_store_kwargs = self.resolve_setting(doc_store_kwargs, "doc_store_kwargs", default=None, merge=True)
        emb_store = self.resolve_setting(emb_store, "emb_store", default=None)
        emb_store_kwargs = self.resolve_setting(emb_store_kwargs, "emb_store_kwargs", default=None, merge=True)
        score_func = self.resolve_setting(score_func, "score_func")
        score_agg_func = self.resolve_setting(score_agg_func, "score_agg_func")
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        obj_store = self.get_setting("obj_store", default=None)
        obj_store_kwargs = self.get_setting("obj_store_kwargs", default=None, merge=True)
        if doc_store is None:
            doc_store = obj_store
        doc_store_kwargs = merge_dicts(obj_store_kwargs, doc_store_kwargs)
        if emb_store is None:
            emb_store = obj_store
        emb_store_kwargs = merge_dicts(obj_store_kwargs, emb_store_kwargs)

        embeddings = resolve_embeddings(embeddings)
        if isinstance(embeddings, type):
            embeddings_kwargs = dict(embeddings_kwargs)
            embeddings_kwargs["template_context"] = merge_dicts(
                template_context, embeddings_kwargs.get("template_context", None)
            )
            embeddings = embeddings(**embeddings_kwargs)
        elif embeddings_kwargs:
            embeddings = embeddings.replace(**embeddings_kwargs)

        doc_store = resolve_obj_store(doc_store)
        if not isinstance(doc_store._settings_path, str):
            raise TypeError("_settings_path for ObjectStore and its subclasses should be a string")
        doc_store_cls = doc_store if isinstance(doc_store, type) else type(doc_store)
        with ExtSettingsPath([(doc_store_cls, doc_store._settings_path.replace("obj_store", "doc_store"))]):
            if isinstance(doc_store, type):
                doc_store_kwargs = dict(doc_store_kwargs)
                doc_store_kwargs["template_context"] = merge_dicts(
                    template_context, doc_store_kwargs.get("template_context", None)
                )
                doc_store = doc_store(**doc_store_kwargs)
            elif doc_store_kwargs:
                doc_store = doc_store.replace(**doc_store_kwargs)

        emb_store = resolve_obj_store(emb_store)
        if not isinstance(emb_store._settings_path, str):
            raise TypeError("_settings_path for ObjectStore and its subclasses should be a string")
        emb_store_cls = emb_store if isinstance(emb_store, type) else type(emb_store)
        with ExtSettingsPath([(emb_store_cls, emb_store._settings_path.replace("obj_store", "emb_store"))]):
            if isinstance(emb_store, type):
                emb_store_kwargs = dict(emb_store_kwargs)
                emb_store_kwargs["template_context"] = merge_dicts(
                    template_context, emb_store_kwargs.get("template_context", None)
                )
                emb_store = emb_store(**emb_store_kwargs)
            elif emb_store_kwargs:
                emb_store = emb_store.replace(**emb_store_kwargs)

        if isinstance(score_agg_func, str):
            score_agg_func = getattr(np, score_agg_func)

        self._embeddings = embeddings
        self._doc_store = doc_store
        self._emb_store = emb_store
        self._score_func = score_func
        self._score_agg_func = score_agg_func
        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs
        self._template_context = template_context

    @property
    def embeddings(self) -> Embeddings:
        """An instance of `Embeddings`."""
        return self._embeddings

    @property
    def doc_store(self) -> ObjectStore:
        """An instance of `ObjectStore` for documents."""
        return self._doc_store

    @property
    def emb_store(self) -> ObjectStore:
        """An instance of `ObjectStore` for embeddings."""
        return self._emb_store

    @property
    def score_func(self) -> tp.Union[str, tp.Callable]:
        """Score function.

        See `DocumentRanker.compute_score`."""
        return self._score_func

    @property
    def score_agg_func(self) -> tp.Callable:
        """Score aggregation function."""
        return self._score_agg_func

    @property
    def show_progress(self) -> tp.Optional[bool]:
        """Whether to show progress bar."""
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.pbar.ProgressBar`."""
        return self._pbar_kwargs

    @property
    def template_context(self) -> tp.Kwargs:
        """Context used to substitute templates."""
        return self._template_context

    def embed_documents(
        self,
        documents: tp.Iterable[StoreDocument],
        return_embeddings: bool = False,
        return_documents: bool = False,
    ) -> tp.Optional[tp.EmbeddedDocuments]:
        """Embed documents.

        If `return_embeddings` and `return_documents` are both False, returns nothing.
        If `return_embeddings` and `return_documents` are both True, for each document,
        returns the document and either an embedding or a list of document chunks and their embeddings.
        If `return_documents` is False, returns only embeddings."""
        with self.doc_store, self.emb_store:
            documents = list(documents)
            documents_to_split = []
            document_splits = {}
            for document in documents:
                if document.id_ not in self.emb_store:
                    documents_to_split.append(document)
            if documents_to_split:
                from vectorbtpro.utils.pbar import ProgressBar

                pbar_kwargs = merge_dicts(dict(prefix="split_documents"), self.pbar_kwargs)
                with ProgressBar(total=len(documents), show_progress=self.show_progress, **pbar_kwargs) as pbar:
                    for document in documents_to_split:
                        document_splits[document.id_] = document.split()
                        pbar.update()

            obj_contents = {}
            for document in documents:
                if document.id_ not in self.doc_store:
                    self.doc_store[document.id_] = document
                if document.id_ in document_splits:
                    document_chunks = document_splits[document.id_]
                    obj = StoreEmbedding(document.id_)
                    for document_chunk in document_chunks:
                        if document_chunk.id_ != document.id_:
                            if document_chunk.id_ not in self.doc_store:
                                self.doc_store[document_chunk.id_] = document_chunk
                            if document_chunk.id_ not in self.emb_store:
                                child_obj = StoreEmbedding(document_chunk.id_, parent_id=document.id_)
                                self.emb_store[child_obj.id_] = child_obj
                            else:
                                child_obj = self.emb_store[document_chunk.id_]
                            obj.child_ids.append(child_obj.id_)
                            if not child_obj.embedding:
                                content = document_chunk.get_content(for_embed=True)
                                if content:
                                    obj_contents[child_obj.id_] = content
                    self.emb_store[obj.id_] = obj
                else:
                    obj = self.emb_store[document.id_]
                if not obj.child_ids and not obj.embedding:
                    content = document.get_content(for_embed=True)
                    if content:
                        obj_contents[obj.id_] = content

            if obj_contents:
                embeddings = self.embeddings.get_embeddings(list(obj_contents.values()))
                obj_embeddings = dict(zip(obj_contents.keys(), embeddings))
                for obj_id, embedding in obj_embeddings.items():
                    obj = self.emb_store[obj_id]
                    new_obj = obj.replace(embedding=embedding)
                    self.emb_store[new_obj.id_] = new_obj

            if return_embeddings or return_documents:
                embeddings = []
                for document in documents:
                    obj = self.emb_store[document.id_]
                    if obj.embedding:
                        if return_documents:
                            embeddings.append(EmbeddedDocument(document, embedding=obj.embedding))
                        else:
                            embeddings.append(obj.embedding)
                    elif obj.child_ids:
                        child_embeddings = []
                        for child_id in obj.child_ids:
                            child_obj = self.emb_store[child_id]
                            if child_obj.embedding:
                                if return_documents:
                                    child_document = self.doc_store[child_id]
                                    child_embeddings.append(
                                        EmbeddedDocument(child_document, embedding=child_obj.embedding)
                                    )
                                else:
                                    child_embeddings.append(child_obj.embedding)
                            else:
                                if return_documents:
                                    child_document = self.doc_store[child_id]
                                    child_embeddings.append(EmbeddedDocument(child_document))
                                else:
                                    child_embeddings.append(None)
                        if return_documents:
                            embeddings.append(EmbeddedDocument(document, child_documents=child_embeddings))
                        else:
                            embeddings.append(child_embeddings)
                    else:
                        if return_documents:
                            embeddings.append(EmbeddedDocument(document))
                        else:
                            embeddings.append(None)

                return embeddings

    def compute_score(
        self,
        emb1: tp.Union[tp.MaybeIterable[tp.List[float]], np.ndarray],
        emb2: tp.Union[tp.MaybeIterable[tp.List[float]], np.ndarray],
    ) -> tp.Union[float, np.ndarray]:
        """Compute scores between embeddings, which can be either single or multiple.

        Supported distance functions are 'cosine', 'euclidean', and 'dot'. A metric can also be a callable that should
        take two and return one 2-dim NumPy array."""
        emb1 = np.asarray(emb1)
        emb2 = np.asarray(emb2)
        emb1_single = emb1.ndim == 1
        emb2_single = emb2.ndim == 1
        if emb1_single:
            emb1 = emb1.reshape(1, -1)
        if emb2_single:
            emb2 = emb2.reshape(1, -1)

        if isinstance(self.score_func, str):
            if self.score_func.lower() == "cosine":
                emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
                emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
                emb1_norm = np.nan_to_num(emb1_norm)
                emb2_norm = np.nan_to_num(emb2_norm)
                score_matrix = np.dot(emb1_norm, emb2_norm.T)
            elif self.score_func.lower() == "euclidean":
                diff = emb1[:, np.newaxis, :] - emb2[np.newaxis, :, :]
                distances = np.linalg.norm(diff, axis=2)
                score_matrix = np.divide(1, distances, where=distances != 0, out=np.full_like(distances, np.inf))
            elif self.score_func.lower() == "dot":
                score_matrix = np.dot(emb1, emb2.T)
            else:
                raise ValueError(f"Invalid distance function: '{self.score_func}'")
        else:
            score_matrix = self.score_func(emb1, emb2)

        if emb1_single and emb2_single:
            return float(score_matrix[0, 0])
        if emb1_single or emb2_single:
            return score_matrix.flatten()
        return score_matrix

    def score_documents(
        self,
        query: str,
        documents: tp.Optional[tp.Iterable[StoreDocument]] = None,
        return_chunks: bool = False,
        return_documents: bool = False,
    ) -> tp.ScoredDocuments:
        """Score documents by relevance to a query."""
        with self.doc_store, self.emb_store:
            if documents is None:
                if self.doc_store is None:
                    raise ValueError("Must provide at least documents or doc_store")
                documents = self.doc_store.values()
                documents_provided = False
            else:
                documents_provided = True
            documents = list(documents)
            if not documents:
                return []
            self.embed_documents(documents)
            if return_chunks:
                document_chunks = []
                for document in documents:
                    obj = self.emb_store[document.id_]
                    if obj.child_ids:
                        for child_id in obj.child_ids:
                            document_chunk = self.doc_store[child_id]
                            document_chunks.append(document_chunk)
                    elif not obj.parent_id or obj.parent_id not in self.doc_store:
                        document_chunk = self.doc_store[obj.id_]
                        document_chunks.append(document_chunk)
                documents = document_chunks
            elif not documents_provided:
                document_parents = []
                for document in documents:
                    obj = self.emb_store[document.id_]
                    if not obj.parent_id or obj.parent_id not in self.doc_store:
                        document_parent = self.doc_store[obj.id_]
                        document_parents.append(document_parent)
                documents = document_parents

            obj_embeddings = {}
            for document in documents:
                obj = self.emb_store[document.id_]
                if obj.embedding:
                    obj_embeddings[obj.id_] = obj.embedding
                elif obj.child_ids:
                    for child_id in obj.child_ids:
                        child_obj = self.emb_store[child_id]
                        if child_obj.embedding:
                            obj_embeddings[child_id] = child_obj.embedding
            if obj_embeddings:
                query_embedding = self.embeddings.get_embedding(query)
                scores = self.compute_score(query_embedding, list(obj_embeddings.values()))
                obj_scores = dict(zip(obj_embeddings.keys(), scores))
            else:
                obj_scores = {}

            scores = []
            for document in documents:
                obj = self.emb_store[document.id_]
                child_scores = []
                if obj.child_ids:
                    for child_id in obj.child_ids:
                        if child_id in obj_scores:
                            child_score = obj_scores[child_id]
                            if return_documents:
                                child_document = self.doc_store[child_id]
                                child_scores.append(ScoredDocument(child_document, score=child_score))
                            else:
                                child_scores.append(child_score)
                    if child_scores:
                        if return_documents:
                            doc_score = self.score_agg_func([document.score for document in child_scores])
                        else:
                            doc_score = self.score_agg_func(child_scores)
                    else:
                        doc_score = float("nan")
                else:
                    if obj.id_ in obj_scores:
                        doc_score = obj_scores[obj.id_]
                    else:
                        doc_score = float("nan")
                if return_documents:
                    scores.append(ScoredDocument(document, score=doc_score, child_documents=child_scores))
                else:
                    scores.append(doc_score)
            return scores

    @classmethod
    def resolve_top_k(
        cls,
        scores: tp.Iterable[float],
        top_k: tp.TopKLike = None,
        cutoff: tp.Optional[float] = None,
    ) -> tp.Optional[int]:
        """Resolve `top_k` based on _sorted_ scores.

        Supported values are integers (top number), floats (top %), strings (supported methods are
        'elbow' and 'kmeans'), as well as callables that should take a 1-dim NumPy array and return
        an integer or a float. Filters out NaN before computation (requires them to be at the tail)."""
        if top_k is None and cutoff is None:
            return None
        scores = np.asarray(scores)
        scores = scores[~np.isnan(scores)]
        if cutoff is not None:
            scores = scores[scores >= cutoff]
            if top_k is None:
                return len(scores)

        if isinstance(top_k, str):
            if top_k.lower() == "elbow":
                if scores.size == 0:
                    return 0
                diffs = np.diff(scores)
                top_k = np.argmax(-diffs) + 1
            elif top_k.lower() == "kmeans":
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=2, random_state=0).fit(scores.reshape(-1, 1))
                high_score_cluster = np.argmax(kmeans.cluster_centers_)
                top_k_indices = np.where(kmeans.labels_ == high_score_cluster)[0]
                top_k = max(top_k_indices) + 1
            else:
                raise ValueError(f"Invalid top_k method: '{top_k}'")
        elif callable(top_k):
            top_k = top_k(scores)
        if checks.is_float(top_k):
            top_k = int(top_k * len(scores))
        if cutoff is not None:
            return min(top_k, len(scores))
        return top_k

    def rank_documents(
        self,
        query: str,
        documents: tp.Optional[tp.Iterable[StoreDocument]] = None,
        top_k: tp.TopKLike = None,
        cutoff: tp.Optional[float] = None,
        return_chunks: bool = False,
        return_scores: bool = False,
    ) -> tp.RankedDocuments:
        """Sort documents by relevance to a query."""
        scored_documents = self.score_documents(
            query,
            documents=documents,
            return_chunks=return_chunks,
            return_documents=True,
        )
        scored_documents = sorted(scored_documents, key=lambda x: (not np.isnan(x.score), x.score), reverse=True)
        scores = [document.score for document in scored_documents]
        top_k = self.resolve_top_k(scores, top_k=top_k, cutoff=cutoff)
        if top_k is not None:
            if top_k == 0:
                raise ValueError("No documents selected after ranking. Change top_k or cutoff.")
            scored_documents = scored_documents[:top_k]
        if return_scores:
            return scored_documents
        return [document.document for document in scored_documents]


def embed_documents(
    documents: tp.Iterable[StoreDocument],
    return_embeddings: bool = False,
    return_documents: bool = False,
    document_ranker: tp.Optional[tp.MaybeType[DocumentRanker]] = None,
    **kwargs,
) -> tp.Optional[tp.EmbeddedDocuments]:
    """Embed documents.

    Keyword arguments are passed to either initialize a class or replace an
    instance of `DocumentRanker`."""
    if document_ranker is None:
        document_ranker = DocumentRanker
    if isinstance(document_ranker, type):
        checks.assert_subclass_of(document_ranker, DocumentRanker, "document_ranker")
        document_ranker = document_ranker(**kwargs)
    else:
        checks.assert_instance_of(document_ranker, DocumentRanker, "document_ranker")
        if kwargs:
            document_ranker = document_ranker.replace(**kwargs)
    return document_ranker.embed_documents(
        documents,
        return_embeddings=return_embeddings,
        return_documents=return_documents,
    )


def rank_documents(
    query: str,
    documents: tp.Optional[tp.Iterable[StoreDocument]] = None,
    top_k: tp.TopKLike = None,
    cutoff: tp.Optional[float] = None,
    return_chunks: bool = False,
    return_scores: bool = False,
    document_ranker: tp.Optional[tp.MaybeType[DocumentRanker]] = None,
    **kwargs,
) -> tp.RankedDocuments:
    """Rank documents by their relevance to a query.

    Keyword arguments are passed to either initialize a class or replace an
    instance of `DocumentRanker`."""
    if document_ranker is None:
        document_ranker = DocumentRanker
    if isinstance(document_ranker, type):
        checks.assert_subclass_of(document_ranker, DocumentRanker, "document_ranker")
        document_ranker = document_ranker(**kwargs)
    else:
        checks.assert_instance_of(document_ranker, DocumentRanker, "document_ranker")
        if kwargs:
            document_ranker = document_ranker.replace(**kwargs)
    return document_ranker.rank_documents(
        query,
        documents=documents,
        top_k=top_k,
        cutoff=cutoff,
        return_chunks=return_chunks,
        return_scores=return_scores,
    )


RankableT = tp.TypeVar("RankableT", bound="Rankable")


class Rankable(HasSettings):
    """Abstract class that can be ranked."""

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.chat"]

    def embed(
        self: RankableT,
        return_embeddings: bool = False,
        return_documents: bool = False,
        **kwargs,
    ) -> tp.Optional[RankableT]:
        """Embed documents."""
        raise NotImplementedError

    def rank(
        self: RankableT,
        query: str,
        top_k: tp.TopKLike = None,
        cutoff: tp.Optional[float] = None,
        return_chunks: bool = False,
        return_scores: bool = False,
        **kwargs,
    ) -> RankableT:
        """Rank documents by their relevance to a query."""
        raise NotImplementedError


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

    def create_chat(
        self,
        to_context_kwargs: tp.KwargsLike = None,
        completions: tp.CompletionsLike = None,
        **kwargs,
    ) -> Completions:
        """Create a chat by returning an instance of `Completions`.

        Uses `Contextable.to_context` to turn this instance to a context.

        Usage:
            ```pycon
            >>> chat = asset.create_chat()

            >>> chat.get_completion("What's the value under 'xyz'?")
            The value under 'xyz' is 123.

            >>> chat.get_completion("Are you sure?")
            Yes, I am sure. The value under 'xyz' is 123 for the entry where `s` is "EFG".
            ```"""
        to_context_kwargs = self.resolve_setting(to_context_kwargs, "to_context_kwargs", merge=True)
        context = self.to_context(**to_context_kwargs)
        completions = resolve_completions(completions=completions)
        if isinstance(completions, type):
            completions = completions(context=context, **kwargs)
        else:
            completions = completions.replace(context=context, **kwargs)
        return completions

    @hybrid_method
    def chat(
        cls_or_self,
        message: str,
        chat_history: tp.Optional[tp.ChatHistory] = None,
        **kwargs,
    ) -> tp.ChatOutput:
        """Chat with an LLM while using the instance as a context.

        Uses `Contextable.create_chat` and then `Completions.get_completion`.

        !!! note
            Context is recalculated each time this method is invoked. For multiple turns,
            it's more efficient to use `Contextable.create_chat`.

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

        completions = cls_or_self.create_chat(chat_history=chat_history, **kwargs)
        return completions.get_completion(message)


class RankContextable(Rankable, Contextable):
    """Abstract class that combines both `Rankable` and `Contextable` to rank a context."""

    @hybrid_method
    def chat(
        cls_or_self,
        message: str,
        *args,
        rank: tp.Optional[bool] = None,
        top_k: tp.TopKLike = None,
        cutoff: tp.Optional[float] = None,
        return_chunks: tp.Optional[bool] = None,
        rank_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.ChatOutput:
        """See `Contextable.chat`.

        If `rank` is True, or `rank` is None and any of `top_k`, `cutoff`, or `return_chunks` is set,
        will rank the documents with `Rankable.rank` first."""
        if isinstance(cls_or_self, type):
            from vectorbtpro.utils.parsing import get_forward_args

            args, kwargs = get_forward_args(super().chat, locals())
            return super().chat(*args, **kwargs)

        rank = cls_or_self.resolve_setting(rank, "rank")
        rank_kwargs = cls_or_self.resolve_setting(rank_kwargs, "rank_kwargs", merge=True)
        def_top_k = rank_kwargs.pop("top_k")
        if top_k is None:
            top_k = def_top_k
        def_cutoff = rank_kwargs.pop("cutoff")
        if cutoff is None:
            cutoff = def_cutoff
        def_return_chunks = rank_kwargs.pop("return_chunks")
        if return_chunks is None:
            return_chunks = def_return_chunks
        if rank or (rank is None and (top_k or cutoff or return_chunks)):
            _cls_or_self = cls_or_self.rank(
                message,
                top_k=top_k,
                cutoff=cutoff,
                return_chunks=return_chunks,
                **rank_kwargs,
            )
        else:
            _cls_or_self = cls_or_self
        return Contextable.chat.__func__(_cls_or_self, message, *args, **kwargs)
