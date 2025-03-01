# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Base class."""

from vectorbtpro import _typing as tp

__all__ = ["Base"]


class Base:
    """Base class for all VBT classes."""

    @classmethod
    def find_api(cls, **kwargs) -> tp.MaybePagesAsset:
        """Find API pages and headings relevant to this class or one of its attributes.

        Uses `vectorbtpro.utils.knowledge.custom_assets.find_api`."""
        from vectorbtpro.utils.knowledge.custom_assets import find_api

        return find_api(cls, **kwargs)

    @classmethod
    def find_docs(cls, **kwargs) -> tp.MaybePagesAsset:
        """Find documentation pages and headings relevant to this class or one of its attributes.

        Uses `vectorbtpro.utils.knowledge.custom_assets.find_docs`."""
        from vectorbtpro.utils.knowledge.custom_assets import find_docs

        return find_docs(cls, **kwargs)

    @classmethod
    def find_messages(cls, **kwargs) -> tp.MaybeMessagesAsset:
        """Find messages relevant to this class or one of its attributes.

        Uses `vectorbtpro.utils.knowledge.custom_assets.find_messages`."""
        from vectorbtpro.utils.knowledge.custom_assets import find_messages

        return find_messages(cls, **kwargs)

    @classmethod
    def find_examples(cls, **kwargs) -> tp.MaybeVBTAsset:
        """Find (code) examples relevant to this class or one of its attributes.

        Uses `vectorbtpro.utils.knowledge.custom_assets.find_examples`."""
        from vectorbtpro.utils.knowledge.custom_assets import find_examples

        return find_examples(cls, **kwargs)

    @classmethod
    def find_assets(cls, **kwargs) -> tp.MaybeDict[tp.VBTAsset]:
        """Find all assets relevant to this class or one of its attributes.

        Uses `vectorbtpro.utils.knowledge.custom_assets.find_assets`."""
        from vectorbtpro.utils.knowledge.custom_assets import find_assets

        return find_assets(cls, **kwargs)

    @classmethod
    def chat(cls, message: str, chat_history: tp.ChatHistory = None, **kwargs) -> tp.ChatOutput:
        """Chat this class or one of its attributes.

        Uses `vectorbtpro.utils.knowledge.custom_assets.chat_about`."""
        from vectorbtpro.utils.knowledge.custom_assets import chat_about

        return chat_about(cls, message, chat_history=chat_history, **kwargs)
