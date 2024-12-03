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
    def find_api(
        cls,
        *,
        attr: tp.Optional[str] = None,
        pages_asset: tp.Optional[tp.MaybeType[PagesAssetT]] = None,
        pull_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[PagesAssetT, list, dict]:
        """Find API pages and headings relevant to this class or one of its attributes.

        Based on `vectorbtpro.utils.knowledge.custom_assets.PagesAsset.find_obj_api`.

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
        return pages_asset.find_obj_api(obj, **kwargs)

    @classmethod
    def find_docs(
        cls,
        *,
        attr: tp.Optional[str] = None,
        pages_asset: tp.Optional[tp.MaybeType[PagesAssetT]] = None,
        pull_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[PagesAssetT, list, dict]:
        """Find documentation pages and headings relevant to this class or one of its attributes.

        Based on `vectorbtpro.utils.knowledge.custom_assets.PagesAsset.find_obj_docs`.

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
        return pages_asset.find_obj_docs(obj, **kwargs)

    @classmethod
    def find_messages(
        cls,
        *,
        attr: tp.Optional[str] = None,
        messages_asset: tp.Optional[tp.MaybeType[MessagesAssetT]] = None,
        pull_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[MessagesAssetT, list, dict]:
        """Find messages relevant to this class or one of its attributes.

        Based on `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset.find_obj_messages`.

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
        return messages_asset.find_obj_messages(obj, **kwargs)

    @classmethod
    def find_examples(
        cls,
        *,
        attr: tp.Optional[str] = None,
        as_code: bool = True,
        return_type: tp.Optional[str] = "field",
        pages_asset: tp.Optional[tp.MaybeType[PagesAssetT]] = None,
        messages_asset: tp.Optional[tp.MaybeType[MessagesAssetT]] = None,
        pull_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[VBTAssetT, list, dict]:
        """Find (code) examples relevant to this class or one of its attributes.

        Based on `vectorbtpro.utils.knowledge.custom_assets.VBTAsset.find_obj_obj_mentions`.

        By default, extracts code with text. Use `return_type="match"` to extract code without text,
        or, for instance, `return_type="item"` to also get links.

        Use `pages_asset` to provide a custom subclass or instance of
        `vectorbtpro.utils.knowledge.custom_assets.PagesAsset`. Use `messages_asset` to provide a
        custom subclass or instance of `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset`."""
        from vectorbtpro.utils.checks import assert_subclass_of, assert_instance_of
        from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset
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
            isinstance(mentions_asset, KnowledgeAsset)
            and len(mentions_asset) > 0
            and isinstance(mentions_asset[0], list)
            and return_type.lower() in ("field", "match")
        ):
            mentions_asset = mentions_asset.merge_lists()
        return mentions_asset

    @classmethod
    def find_assets(
        cls,
        asset_names: tp.Optional[tp.MaybeIterable[str]] = None,
        *,
        attr: tp.Optional[str] = None,
        minimize: tp.Optional[bool] = None,
        minimize_pages: tp.Optional[bool] = None,
        minimize_messages: tp.Optional[bool] = None,
        stack: bool = True,
        pages_asset: tp.Optional[tp.MaybeType[PagesAssetT]] = None,
        messages_asset: tp.Optional[tp.MaybeType[MessagesAssetT]] = None,
        pull_kwargs: tp.KwargsLike = None,
        api_kwargs: tp.KwargsLike = None,
        docs_kwargs: tp.KwargsLike = None,
        messages_kwargs: tp.KwargsLike = None,
        examples_kwargs: tp.KwargsLike = None,
        minimize_kwargs: tp.KwargsLike = None,
        minimize_pages_kwargs: tp.KwargsLike = None,
        minimize_messages_kwargs: tp.KwargsLike = None,
        stack_kwargs: tp.KwargsLike = None,
    ) -> tp.Union[tp.Dict[str, VBTAssetT], VBTAssetT]:
        """Find all assets relevant to this class or one of its attributes.

        Argument `asset_names` can be a list of asset names in any order. It defaults to "api", "docs",
        and "messages", It can also include ellipsis (`...`). For example, `["messages", ...]` puts
        "messages" at the beginning and all other assets in their usual order at the end.
        The following asset names are supported:

        * "api": `Base.find_api` with `api_kwargs`
        * "docs": `Base.find_docs` with `docs_kwargs`
        * "messages": `Base.find_messages` with `messages_kwargs`
        * "examples": `Base.find_examples` with `examples_kwargs`

        !!! note
            Examples usually overlap with other assets, thus they are excluded by default.

        Use `pages_asset` to provide a custom subclass or instance of
        `vectorbtpro.utils.knowledge.custom_assets.PagesAsset`. Use `messages_asset` to provide a
        custom subclass or instance of `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset`.
        Both assets are reused among "find" calls.

        Set `stack` to True to stack all assets into a single asset. Uses
        `vectorbtpro.utils.knowledge.base.KnowledgeAsset.stack` with `stack_kwargs`.

        Set `minimize` to True (or `minimize_pages` for pages and `minimize_messages` for messages)
        in order to minimize to remove fields that aren't relevant for chatting.
        It defaults to True if `stack` is True, otherwise, it defaults to False.
        Uses `vectorbtpro.utils.knowledge.custom_assets.VBTAsset.minimize` with `minimize_kwargs`,
        `vectorbtpro.utils.knowledge.custom_assets.PagesAsset.minimize` with `minimize_pages_kwargs`,
        and `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset.minimize` with `minimize_messages_kwargs`.
        Arguments `minimize_pages_kwargs` and `minimize_messages_kwargs` are merged over `minimize_kwargs`."""
        from vectorbtpro.utils.checks import assert_subclass_of, assert_instance_of
        from vectorbtpro.utils.config import merge_dicts, reorder_list
        from vectorbtpro.utils.knowledge.custom_assets import VBTAsset, PagesAsset, MessagesAsset

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

        assets = []
        if asset_names is not None:
            if isinstance(asset_names, (str, type(Ellipsis))):
                asset_names = [asset_names]
            all_asset_names = ["api", "docs", "messages", "examples"]
            asset_keys = []
            for asset_name in asset_names:
                if asset_name is not Ellipsis:
                    asset_key = all_asset_names.index(asset_name.lower())
                    if asset_key == -1:
                        raise ValueError(f"Invalid asset name: '{asset_name}'")
                    asset_keys.append(asset_key)
                else:
                    asset_keys.append(Ellipsis)
            new_asset_names = reorder_list(all_asset_names, asset_keys, skip_missing=True)
            if "examples" not in asset_names and "examples" in new_asset_names:
                new_asset_names.remove("examples")
            asset_names = new_asset_names
        else:
            asset_names = ["api", "docs", "messages"]
        for asset_name in asset_names:
            if asset_name == "api":
                if api_kwargs is None:
                    api_kwargs = {}
                asset = cls.find_api(
                    attr=attr,
                    pages_asset=pages_asset,
                    **api_kwargs,
                )
                if len(asset) > 0:
                    assets.append(asset)
            elif asset_name == "docs":
                if docs_kwargs is None:
                    docs_kwargs = {}
                asset = cls.find_docs(
                    attr=attr,
                    pages_asset=pages_asset,
                    **docs_kwargs,
                )
                if len(asset) > 0:
                    assets.append(asset)
            elif asset_name == "messages":
                if messages_kwargs is None:
                    messages_kwargs = {}
                asset = cls.find_messages(
                    attr=attr,
                    messages_asset=messages_asset,
                    **messages_kwargs,
                )
                if len(asset) > 0:
                    assets.append(asset)
            elif asset_name == "examples":
                if examples_kwargs is None:
                    examples_kwargs = {}
                asset = cls.find_examples(
                    attr=attr,
                    pages_asset=pages_asset,
                    messages_asset=messages_asset,
                    **examples_kwargs,
                )
                if len(asset) > 0:
                    assets.append(asset)

        if minimize is None:
            minimize = stack
        if minimize:
            if minimize_kwargs is None:
                minimize_kwargs = {}
            for i in range(len(assets)):
                if isinstance(assets[i], VBTAsset) and not isinstance(assets[i], (PagesAsset, MessagesAsset)):
                    assets[i] = assets[i].minimize(**minimize_kwargs)
        if minimize_pages is None:
            minimize_pages = minimize
        if minimize_pages:
            minimize_pages_kwargs = merge_dicts(minimize_kwargs, minimize_pages_kwargs)
            for i in range(len(assets)):
                if isinstance(assets[i], PagesAsset):
                    assets[i] = assets[i].minimize(**minimize_pages_kwargs)
        if minimize_messages is None:
            minimize_messages = minimize
        if minimize_messages:
            minimize_messages_kwargs = merge_dicts(minimize_kwargs, minimize_messages_kwargs)
            for i in range(len(assets)):
                if isinstance(assets[i], MessagesAsset):
                    assets[i] = assets[i].minimize(**minimize_messages_kwargs)
        if stack:
            if len(assets) >= 2:
                if stack_kwargs is None:
                    stack_kwargs = {}
                return VBTAsset.stack(*assets, **stack_kwargs)
            if len(assets) == 1:
                return assets[0]
            return VBTAsset([])
        return dict(zip(asset_names, assets))
