# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Base class."""

from vectorbtpro import _typing as tp

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from vectorbtpro.utils.knowledge.custom_assets import (
        VBTAsset as VBTAssetT,
        PagesAsset as PagesAssetT,
        MessagesAsset as MessagesAssetT,
    )
except ImportError:
    VBTAssetT = tp.Any
    PagesAssetT = tp.Any
    MessagesAssetT = tp.Any

__all__ = ["Base"]


class Base:
    """Base class for all VBT classes."""

    @classmethod
    def find_relevant_api(
        cls,
        attr: tp.Optional[str] = None,
        *,
        pages_asset: tp.Optional[tp.MaybeType[PagesAssetT]] = None,
        pull_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[PagesAssetT, tp.Tuple[PagesAssetT, dict]]:
        """Find API pages and headings relevant to an object.

        Based on `vectorbtpro.utils.knowledge.custom_assets.PagesAsset.find_relevant_api`.

        Use `pages_asset` to provide a custom subclass or instance of
        `vectorbtpro.utils.knowledge.custom_assets.PagesAsset`."""
        from vectorbtpro.utils.checks import assert_subclass_of, assert_instance_of
        from vectorbtpro.utils.knowledge.custom_assets import PagesAsset

        if pages_asset is None:
            pages_asset = PagesAsset
        if isinstance(pages_asset, type):
            assert_subclass_of(pages_asset, PagesAsset, arg_name="pages_asset")
            if pull_kwargs is None:
                pull_kwargs = {}
            pages_asset = pages_asset.pull(**pull_kwargs)
        assert_instance_of(pages_asset, PagesAsset, arg_name="pages_asset")
        if attr is None:
            obj = cls
        else:
            assert_instance_of(attr, str, arg_name="attr")
            obj = (cls, attr)
        return pages_asset.find_relevant_api(obj, **kwargs)

    @classmethod
    def find_relevant_docs(
        cls,
        attr: tp.Optional[str] = None,
        *,
        pages_asset: tp.Optional[tp.MaybeType[PagesAssetT]] = None,
        pull_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[PagesAssetT, tp.Tuple[PagesAssetT, dict]]:
        """Find documentation pages and headings relevant to an object.

        Based on `vectorbtpro.utils.knowledge.custom_assets.PagesAsset.find_relevant_docs`.

        Use `pages_asset` to provide a custom subclass or instance of
        `vectorbtpro.utils.knowledge.custom_assets.PagesAsset`."""
        from vectorbtpro.utils.checks import assert_subclass_of, assert_instance_of
        from vectorbtpro.utils.knowledge.custom_assets import PagesAsset

        if pages_asset is None:
            pages_asset = PagesAsset
        if isinstance(pages_asset, type):
            assert_subclass_of(pages_asset, PagesAsset, arg_name="pages_asset")
            if pull_kwargs is None:
                pull_kwargs = {}
            pages_asset = pages_asset.pull(**pull_kwargs)
        assert_instance_of(pages_asset, PagesAsset, arg_name="pages_asset")
        if attr is None:
            obj = cls
        else:
            assert_instance_of(attr, str, arg_name="attr")
            obj = (cls, attr)
        return pages_asset.find_relevant_docs(obj, **kwargs)

    @classmethod
    def find_relevant_messages(
        cls,
        attr: tp.Optional[str] = None,
        *,
        messages_asset: tp.Optional[tp.MaybeType[MessagesAssetT]] = None,
        pull_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[MessagesAssetT, tp.Tuple[MessagesAssetT, dict]]:
        """Find messages relevant to an object.

        Based on `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset.find_relevant_messages`.

        Use `messages_asset` to provide a custom subclass or instance of
        `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset`."""
        from vectorbtpro.utils.checks import assert_subclass_of, assert_instance_of
        from vectorbtpro.utils.knowledge.custom_assets import MessagesAsset

        if messages_asset is None:
            messages_asset = MessagesAsset
        if isinstance(messages_asset, type):
            assert_subclass_of(messages_asset, MessagesAsset, arg_name="messages_asset")
            if pull_kwargs is None:
                pull_kwargs = {}
            messages_asset = messages_asset.pull(**pull_kwargs)
        assert_instance_of(messages_asset, MessagesAsset, arg_name="messages_asset")
        if attr is None:
            obj = cls
        else:
            assert_instance_of(attr, str, arg_name="attr")
            obj = (cls, attr)
        return messages_asset.find_relevant_messages(obj, **kwargs)

    @classmethod
    def find_relevant_examples(
        cls,
        attr: tp.Optional[str] = None,
        *,
        as_code: bool = True,
        return_type: tp.Optional[str] = "field",
        pages_asset: tp.Optional[tp.MaybeType[PagesAssetT]] = None,
        messages_asset: tp.Optional[tp.MaybeType[MessagesAssetT]] = None,
        pull_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[VBTAssetT, tp.Tuple[VBTAssetT, dict]]:
        """Find (code) examples relevant to an object in pages and messages.

        By default, extracts code with text. Use `return_type="match"` to extract code without text,
        or, for instance, `return_type="item"` to also get links.

        Based on `vectorbtpro.utils.knowledge.custom_assets.VBTAsset.find_obj_mentions`.

        Use `pages_asset` to provide a custom subclass or instance of
        `vectorbtpro.utils.knowledge.custom_assets.PagesAsset`.
        Use `messages_asset` to provide a custom subclass or instance of
        `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset`."""
        from vectorbtpro.utils.checks import assert_subclass_of, assert_instance_of
        from vectorbtpro.utils.knowledge.custom_assets import PagesAsset, MessagesAsset

        if pages_asset is None:
            pages_asset = PagesAsset
        if isinstance(pages_asset, type):
            assert_subclass_of(pages_asset, PagesAsset, arg_name="pages_asset")
            if pull_kwargs is None:
                pull_kwargs = {}
            pages_asset = pages_asset.pull(**pull_kwargs)
        assert_instance_of(pages_asset, PagesAsset, arg_name="pages_asset")
        if messages_asset is None:
            messages_asset = MessagesAsset
        if isinstance(messages_asset, type):
            assert_subclass_of(messages_asset, MessagesAsset, arg_name="messages_asset")
            if pull_kwargs is None:
                pull_kwargs = {}
            messages_asset = messages_asset.pull(**pull_kwargs)
        assert_instance_of(messages_asset, MessagesAsset, arg_name="messages_asset")
        if attr is None:
            obj = cls
        else:
            assert_instance_of(attr, str, arg_name="attr")
            obj = (cls, attr)
        mentions_in_pages = pages_asset.find_obj_mentions(
            obj,
            as_code=as_code,
            return_type=return_type,
            **kwargs,
        )
        mentions_in_messages = messages_asset.find_obj_mentions(
            obj,
            as_code=as_code,
            return_type=return_type,
            **kwargs,
        )
        mentions_asset = mentions_in_pages + mentions_in_messages
        if (
            len(mentions_asset) > 0
            and isinstance(mentions_asset[0], list)
            and return_type.lower() in ("field", "match")
        ):
            mentions_asset = mentions_asset.merge_lists()
        return mentions_asset
