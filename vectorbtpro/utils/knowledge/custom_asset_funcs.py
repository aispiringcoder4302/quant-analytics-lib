# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Custom asset function classes."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.knowledge.base_asset_funcs import AssetFunc


class AggAttachAssetFunc(AssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset.aggregate_attachments`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "agg_attach"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare(
        cls,
        clear_metadata: tp.Optional[bool] = None,
        metadata_clear_kwargs: tp.KwargsLike = None,
        metadata_dump_kwargs: tp.KwargsLike = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc, DumpAssetFunc

        if asset is None:
            from vectorbtpro.utils.knowledge.custom_assets import MessagesAsset

            asset = MessagesAsset
        clear_metadata = asset.resolve_setting(clear_metadata, "clear_metadata")
        metadata_clear_kwargs = asset.resolve_setting(metadata_clear_kwargs, "metadata_clear_kwargs", merge=True)
        metadata_dump_kwargs = asset.resolve_setting(metadata_dump_kwargs, "metadata_dump_kwargs", merge=True)

        metadata_clear_kwargs = merge_dicts(dict(target=FindRemoveAssetFunc.is_empty_func), metadata_clear_kwargs)
        _, metadata_clear_kwargs = FindRemoveAssetFunc.prepare(**metadata_clear_kwargs)
        _, metadata_dump_kwargs = DumpAssetFunc.prepare(**metadata_dump_kwargs)
        return (), {
            **dict(
                clear_metadata=clear_metadata,
                metadata_clear_kwargs=metadata_clear_kwargs,
                metadata_dump_kwargs=metadata_dump_kwargs,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        clear_metadata: bool = True,
        metadata_clear_kwargs: tp.KwargsLike = None,
        metadata_dump_kwargs: tp.KwargsLike = None,
        link_map: tp.Optional[tp.Dict[str, dict]] = None,
    ) -> tp.Any:
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc, DumpAssetFunc

        if not isinstance(d, dict):
            raise TypeError("Data item must be a dict")
        if metadata_clear_kwargs is None:
            metadata_clear_kwargs = {}
        if metadata_dump_kwargs is None:
            metadata_dump_kwargs = {}

        new_d = dict(d)
        new_d["content"] = new_d["content"].strip()
        attachments = new_d.pop("attachments", [])
        for attachment in attachments:
            metadata = dict(attachment)
            content = metadata.pop("content").strip()
            if metadata and clear_metadata:
                metadata = FindRemoveAssetFunc.call(metadata, **metadata_clear_kwargs)
            if not metadata and not content:
                continue
            elif not metadata:
                metadata = None
            metadata = DumpAssetFunc.call({"attachment": metadata}, **metadata_dump_kwargs).strip()
            if new_d["content"]:
                new_d["content"] += "\n\n"
            new_d["content"] += "---\n" + metadata + "\n---\n\n" + content
        return new_d


class AggBlockAssetFunc(AssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset.aggregate_blocks`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "agg_block"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare(
        cls,
        aggregate_fields: tp.Union[None, bool, tp.MaybeSet[str]] = None,
        block_links_only: tp.Optional[bool] = None,
        clear_metadata: tp.Optional[bool] = None,
        metadata_clear_kwargs: tp.KwargsLike = None,
        metadata_dump_kwargs: tp.KwargsLike = None,
        link_map: tp.Optional[tp.Dict[str, dict]] = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc, DumpAssetFunc

        if asset is None:
            from vectorbtpro.utils.knowledge.custom_assets import MessagesAsset

            asset = MessagesAsset
        aggregate_fields = asset.resolve_setting(aggregate_fields, "aggregate_fields")
        block_links_only = asset.resolve_setting(block_links_only, "block_links_only")
        clear_metadata = asset.resolve_setting(clear_metadata, "clear_metadata")
        metadata_clear_kwargs = asset.resolve_setting(metadata_clear_kwargs, "metadata_clear_kwargs", merge=True)
        metadata_dump_kwargs = asset.resolve_setting(metadata_dump_kwargs, "metadata_dump_kwargs", merge=True)

        metadata_clear_kwargs = merge_dicts(dict(target=FindRemoveAssetFunc.is_empty_func), metadata_clear_kwargs)
        _, metadata_clear_kwargs = FindRemoveAssetFunc.prepare(**metadata_clear_kwargs)
        _, metadata_dump_kwargs = DumpAssetFunc.prepare(**metadata_dump_kwargs)
        return (), {
            **dict(
                aggregate_fields=aggregate_fields,
                block_links_only=block_links_only,
                clear_metadata=clear_metadata,
                metadata_clear_kwargs=metadata_clear_kwargs,
                metadata_dump_kwargs=metadata_dump_kwargs,
                link_map=link_map,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        aggregate_fields: tp.Union[bool, tp.MaybeSet[str]] = False,
        block_links_only: bool = True,
        clear_metadata: bool = True,
        metadata_clear_kwargs: tp.KwargsLike = None,
        metadata_dump_kwargs: tp.KwargsLike = None,
        link_map: tp.Optional[tp.Dict[str, dict]] = None,
    ) -> tp.Any:
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc, DumpAssetFunc

        if not isinstance(d, dict):
            raise TypeError("Data item must be a dict")
        if isinstance(aggregate_fields, bool):
            if aggregate_fields:
                aggregate_fields = {"mentions", "attachments", "reactions"}
            else:
                aggregate_fields = set()
        elif isinstance(aggregate_fields, str):
            aggregate_fields = {aggregate_fields}
        elif not isinstance(aggregate_fields, set):
            aggregate_fields = set(aggregate_fields)
        if metadata_clear_kwargs is None:
            metadata_clear_kwargs = {}
        if metadata_dump_kwargs is None:
            metadata_dump_kwargs = {}

        new_d = {}
        metadata_keys = []
        for k, v in d.items():
            if k == "link":
                new_d[k] = d["block"][0]
            if k == "block":
                continue
            if k in {"thread", "channel", "author"}:
                new_d[k] = v[0]
                continue
            if k == "reference" and link_map is not None:
                found_missing = False
                new_v = []
                for _v in v:
                    if _v:
                        if _v in link_map:
                            _v = link_map[_v]["block"]
                        else:
                            found_missing = True
                            break
                    if _v not in new_v:
                        new_v.append(_v)
                if found_missing or len(new_v) > 1:
                    new_d[k] = "?"
                else:
                    new_d[k] = new_v[0]
            if k == "replies" and link_map is not None:
                new_v = []
                for _v in v:
                    for __v in _v:
                        if __v and __v in link_map:
                            __v = link_map[__v]["block"]
                            if __v not in new_v:
                                new_v.append(__v)
                        else:
                            new_v.append("?")
                new_d[k] = new_v
            if k == "content":
                new_d[k] = ""
                continue
            if k in aggregate_fields and isinstance(v[0], list):
                new_v = []
                for _v in new_v:
                    for __v in _v:
                        if __v not in new_v:
                            new_v.append(__v)
                new_d[k] = new_v
                continue
            if k == "reactions" and k in aggregate_fields:
                new_d[k] = sum(v)
                continue
            if block_links_only:
                if k in ("link", "block", "thread", "reference", "replies"):
                    continue
            metadata_keys.append(k)
        if len(metadata_keys) > 0:
            for i in range(len(d[metadata_keys[0]])):
                content = d["content"][i].strip()
                metadata = {}
                for k in metadata_keys:
                    metadata[k] = d[k][i]
                if metadata and clear_metadata:
                    metadata = FindRemoveAssetFunc.call(metadata, **metadata_clear_kwargs)
                if not metadata and not content:
                    continue
                elif not metadata:
                    metadata = None
                metadata = DumpAssetFunc.call({"message": metadata}, **metadata_dump_kwargs).strip()
                if new_d["content"]:
                    new_d["content"] += "\n\n"
                new_d["content"] += "---\n" + metadata + "\n---\n\n" + content
        return new_d


class AggThreadAssetFunc(AssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.custom_assets.MessagesAsset.aggregate_threads`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "agg_thread"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare(
        cls,
        aggregate_fields: tp.Union[None, bool, tp.MaybeSet[str]] = None,
        thread_links_only: tp.Optional[bool] = None,
        clear_metadata: tp.Optional[bool] = None,
        metadata_clear_kwargs: tp.KwargsLike = None,
        metadata_dump_kwargs: tp.KwargsLike = None,
        link_map: tp.Optional[tp.Dict[str, dict]] = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc, DumpAssetFunc

        if asset is None:
            from vectorbtpro.utils.knowledge.custom_assets import MessagesAsset

            asset = MessagesAsset
        aggregate_fields = asset.resolve_setting(aggregate_fields, "aggregate_fields")
        thread_links_only = asset.resolve_setting(thread_links_only, "thread_links_only")
        clear_metadata = asset.resolve_setting(clear_metadata, "clear_metadata")
        metadata_clear_kwargs = asset.resolve_setting(metadata_clear_kwargs, "metadata_clear_kwargs", merge=True)
        metadata_dump_kwargs = asset.resolve_setting(metadata_dump_kwargs, "metadata_dump_kwargs", merge=True)

        metadata_clear_kwargs = merge_dicts(dict(target=FindRemoveAssetFunc.is_empty_func), metadata_clear_kwargs)
        _, metadata_clear_kwargs = FindRemoveAssetFunc.prepare(**metadata_clear_kwargs)
        _, metadata_dump_kwargs = DumpAssetFunc.prepare(**metadata_dump_kwargs)
        return (), {
            **dict(
                aggregate_fields=aggregate_fields,
                thread_links_only=thread_links_only,
                clear_metadata=clear_metadata,
                metadata_clear_kwargs=metadata_clear_kwargs,
                metadata_dump_kwargs=metadata_dump_kwargs,
                link_map=link_map,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        aggregate_fields: tp.Union[bool, tp.MaybeSet[str]] = False,
        thread_links_only: bool = True,
        clear_metadata: bool = True,
        metadata_clear_kwargs: tp.KwargsLike = None,
        metadata_dump_kwargs: tp.KwargsLike = None,
        link_map: tp.Optional[tp.Dict[str, dict]] = None,
    ) -> tp.Any:
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc, DumpAssetFunc

        if not isinstance(d, dict):
            raise TypeError("Data item must be a dict")
        if isinstance(aggregate_fields, bool):
            if aggregate_fields:
                aggregate_fields = {"mentions", "attachments", "reactions"}
            else:
                aggregate_fields = set()
        elif isinstance(aggregate_fields, str):
            aggregate_fields = {aggregate_fields}
        elif not isinstance(aggregate_fields, set):
            aggregate_fields = set(aggregate_fields)
        if metadata_clear_kwargs is None:
            metadata_clear_kwargs = {}
        if metadata_dump_kwargs is None:
            metadata_dump_kwargs = {}

        new_d = {}
        metadata_keys = []
        for k, v in d.items():
            if k == "link":
                new_d[k] = d["thread"][0]
            if k == "thread":
                continue
            if k == "channel":
                new_d[k] = v[0]
                continue
            if k == "content":
                new_d[k] = ""
                continue
            if k in aggregate_fields and isinstance(v[0], list):
                new_v = []
                for _v in new_v:
                    for __v in _v:
                        if __v not in new_v:
                            new_v.append(__v)
                new_d[k] = new_v
                continue
            if k == "reactions" and k in aggregate_fields:
                new_d[k] = sum(v)
                continue
            if thread_links_only:
                if k in ("link", "block", "thread", "reference", "replies"):
                    continue
            metadata_keys.append(k)
        if len(metadata_keys) > 0:
            for i in range(len(d[metadata_keys[0]])):
                content = d["content"][i].strip()
                metadata = {}
                for k in metadata_keys:
                    metadata[k] = d[k][i]
                if metadata and clear_metadata:
                    metadata = FindRemoveAssetFunc.call(metadata, **metadata_clear_kwargs)
                if not metadata and not content:
                    continue
                elif not metadata:
                    metadata = None
                metadata = DumpAssetFunc.call({"message": metadata}, **metadata_dump_kwargs).strip()
                if new_d["content"]:
                    new_d["content"] += "\n\n"
                new_d["content"] += "---\n" + metadata + "\n---\n\n" + content
        return new_d
