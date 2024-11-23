# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Base class."""

from vectorbtpro import _typing as tp

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from vectorbtpro.utils.knowledge.custom_assets import (
        PagesAsset as PagesAssetT,
        MessagesAsset as MessagesAssetT,
    )
except ImportError:
    PagesAssetT = tp.Any
    MessagesAssetT = tp.Any

__all__ = ["Base"]


class Base:
    """Base class for all VBT classes."""

    @classmethod
    def get_api_asset(
        cls,
        *args,
        pages_asset: tp.Optional[tp.MaybeType[PagesAssetT]] = None,
        pull_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[PagesAssetT, tp.Tuple[PagesAssetT, dict]]:
        """Find relevant API for this class or attribute.

        Uses `vectorbtpro.utils.knowledge.custom_assets.PagesAsset.find_relevant_api`.

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
        return pages_asset.find_relevant_api(cls, *args, **kwargs)

    @classmethod
    def get_doc_asset(cls) -> tp.KnowledgeAsset:
        """Get knowledge asset for documentation mentioning this class or instance."""
        raise NotImplementedError

    @classmethod
    def get_message_asset(cls) -> tp.KnowledgeAsset:
        """Get knowledge asset for messages mentioning this class or instance."""
        raise NotImplementedError
