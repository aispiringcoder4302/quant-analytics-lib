# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Base class."""

from vectorbtpro import _typing as tp

__all__ = ["Base"]


class Base:
    """Base class for all VBT classes."""

    @classmethod
    def find_api(cls, *, attr: tp.Optional[str] = None, **kwargs) -> tp.MaybePagesAsset:
        """Find API pages and headings relevant to this class or one of its attributes.

        Uses `vectorbtpro.utils.knowledge.custom_assets.find_api`."""
        from vectorbtpro.utils.checks import assert_instance_of
        from vectorbtpro.utils.knowledge.custom_assets import find_api

        if attr is None:
            obj = cls
        else:
            assert_instance_of(attr, str, arg_name="attr")
            obj = (cls, attr)
        return find_api(obj, **kwargs)

    @classmethod
    def find_docs(cls, *, attr: tp.Optional[str] = None, **kwargs) -> tp.MaybePagesAsset:
        """Find documentation pages and headings relevant to this class or one of its attributes.

        Uses `vectorbtpro.utils.knowledge.custom_assets.find_docs`."""
        from vectorbtpro.utils.checks import assert_instance_of
        from vectorbtpro.utils.knowledge.custom_assets import find_docs

        if attr is None:
            obj = cls
        else:
            assert_instance_of(attr, str, arg_name="attr")
            obj = (cls, attr)
        return find_docs(obj, **kwargs)

    @classmethod
    def find_messages(cls, *, attr: tp.Optional[str] = None, **kwargs) -> tp.MaybeMessagesAsset:
        """Find messages relevant to this class or one of its attributes.

        Uses `vectorbtpro.utils.knowledge.custom_assets.find_messages`."""
        from vectorbtpro.utils.checks import assert_instance_of
        from vectorbtpro.utils.knowledge.custom_assets import find_messages

        if attr is None:
            obj = cls
        else:
            assert_instance_of(attr, str, arg_name="attr")
            obj = (cls, attr)
        return find_messages(obj, **kwargs)

    @classmethod
    def find_examples(cls, *, attr: tp.Optional[str] = None, **kwargs) -> tp.MaybeVBTAsset:
        """Find (code) examples relevant to this class or one of its attributes.

        Uses `vectorbtpro.utils.knowledge.custom_assets.find_examples`."""
        from vectorbtpro.utils.checks import assert_instance_of
        from vectorbtpro.utils.knowledge.custom_assets import find_examples

        if attr is None:
            obj = cls
        else:
            assert_instance_of(attr, str, arg_name="attr")
            obj = (cls, attr)
        return find_examples(obj, **kwargs)

    @classmethod
    def find_assets(cls, *, attr: tp.Optional[str] = None, **kwargs) -> tp.MaybeDict[tp.VBTAsset]:
        """Find all assets relevant to this class or one of its attributes.

        Uses `vectorbtpro.utils.knowledge.custom_assets.find_assets`."""
        from vectorbtpro.utils.checks import assert_instance_of
        from vectorbtpro.utils.knowledge.custom_assets import find_assets

        if attr is None:
            obj = cls
        else:
            assert_instance_of(attr, str, arg_name="attr")
            obj = (cls, attr)
        return find_assets(obj, **kwargs)

    @classmethod
    def chat(
        cls,
        message: str,
        chat_history: tp.Optional[tp.MutableSequence[str]] = None,
        *,
        attr: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.ChatOutput:
        """Chat this class or one of its attributes.

        Uses `vectorbtpro.utils.knowledge.custom_assets.chat_about`."""
        from vectorbtpro.utils.checks import assert_instance_of
        from vectorbtpro.utils.knowledge.custom_assets import chat_about

        if attr is None:
            obj = cls
        else:
            assert_instance_of(attr, str, arg_name="attr")
            obj = (cls, attr)
        return chat_about(obj, message, chat_history=chat_history, **kwargs)
