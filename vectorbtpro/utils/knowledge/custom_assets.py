# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Custom asset classes."""

import os
import re
from types import ModuleType
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import flat_merge_dicts, deep_merge_dicts
from vectorbtpro.utils.module_ import prepare_refname, get_caller_qualname
from vectorbtpro.utils.path_ import check_mkdir, remove_dir
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.template import RepFunc
from vectorbtpro.utils.knowledge.base_assets import ReleaseAsset

__all__ = [
    "VBTAsset",
    "MessagesAsset",
    "PagesAsset",
]


VBTAssetT = tp.TypeVar("VBTAssetT", bound="VBTAsset")


class VBTAsset(ReleaseAsset):
    """Class for working with VBT content."""

    def find_by_link(
        self: VBTAssetT,
        target: tp.MaybeList[str],
        mode: str = "exact",
        single_item: bool = False,
        **kwargs,
    ) -> tp.Union[VBTAssetT, tp.Any]:
        """Find the item(s) corresponding to link(s)."""
        found = self.find_items(target, path="link", mode=mode, single_item=single_item, **kwargs)
        if len(found) == 0:
            raise ValueError(f"No item matching '{target}'")
        if single_item and len(found) > 1:
            links_block = "\n".join(found.get("link"))
            raise ValueError(f"Multiple items matching '{target}':\n\n{links_block}")
        return found

    def minimize_links(self: VBTAssetT) -> VBTAssetT:
        """Minimize links."""
        return self.find_replace(
            {
                r"(https://vectorbt\.pro/pvt_[a-zA-Z0-9]+)": "$pvt_site",
                r"(https://vectorbt\.pro)": "$pub_site",
                r"(https://discord\.com/channels/[0-9]+)": "$discord",
                r"(https://github\.com/polakowo/vectorbt\.pro)": "$github",
            },
            mode="regex",
        )

    def minimize(self: VBTAssetT, minimize_links: tp.Optional[bool] = None) -> VBTAssetT:
        """Minimize by keeping the most useful information.'

        If `minimize_links` is True, replaces redundant URL prefixes by templates that can
        be easily substituted later."""
        minimize_links = self.resolve_setting(minimize_links, "minimize_links")

        new_instance = self.find_remove_empty()
        if minimize_links:
            return new_instance.minimize_links()
        return new_instance

    def to_markdown(
        self: VBTAssetT,
        root_metadata_key: tp.Optional[tp.Key] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[VBTAssetT, tp.Any]:
        """Convert to Markdown.

        Uses `VBTAsset.apply` on `vectorbtpro.utils.knowledge.custom_asset_funcs.ToMarkdownAssetFunc`.

        Use `root_metadata_key` to provide the root key for the metadata markdown.

        If `clear_metadata` is True, removes empty fields from the metadata. Arguments in
        `clear_metadata_kwargs` are passed to `vectorbtpro.utils.knowledge.base_asset_funcs.FindRemoveAssetFunc`,
        while `dump_metadata_kwargs` are passed to `vectorbtpro.utils.knowledge.base_asset_funcs.DumpAssetFunc`."""
        return self.apply(
            "to_markdown",
            root_metadata_key=root_metadata_key,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            **kwargs,
        )

    @classmethod
    def urls_to_paths(cls, urls: tp.Iterable[str], extension: str) -> tp.List[str]:
        """Convert a list of URLs to corresponding paths."""
        from urllib.parse import urlparse

        url_paths = []
        for url in urls:
            parsed = urlparse(url)
            path_parts = [parsed.netloc]
            url_path = parsed.path.strip("/")
            if url_path:
                parts = url_path.split("/")
                if parsed.fragment:
                    path_parts.extend(parts)
                    file_name = f"{parsed.fragment}.{extension}"
                    path_parts.append(file_name)
                else:
                    if len(parts) > 1:
                        path_parts.extend(parts[:-1])
                    last_part = parts[-1]
                    file_name = f"{last_part}.{extension}"
                    path_parts.append(file_name)
            else:
                if parsed.fragment:
                    file_name = f"{parsed.fragment}.{extension}"
                    path_parts.append(file_name)
                else:
                    path_parts.append(f"index.{extension}")
            file_path = os.path.join(*path_parts)
            url_paths.append(file_path)
        return url_paths

    def save_to_markdown(
        self,
        cache: tp.Optional[bool] = None,
        cache_dir: tp.Optional[tp.PathLike] = None,
        cache_mkdir_kwargs: tp.KwargsLike = None,
        clear_cache: bool = False,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> Path:
        """Save to Markdown files.

        If `cache` is True, uses the cache directory. Otherwise, creates a temporary directory.
        If `clear_cache` is True, deletes any existing directory before creating a new one.
        Returns the path of the directory where Markdown files are stored.

        Keyword arguments are passed to `vectorbtpro.utils.knowledge.custom_asset_funcs.ToMarkdownAssetFunc`."""
        import tempfile
        from vectorbtpro.utils.knowledge.custom_asset_funcs import ToMarkdownAssetFunc

        cache = self.resolve_setting(cache, "cache")
        cache_dir = self.resolve_setting(cache_dir, "cache_dir")
        cache_mkdir_kwargs = self.resolve_setting(cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True)
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)

        if cache:
            markdown_dir = Path(cache_dir) / "markdown"
            if markdown_dir.exists():
                if clear_cache:
                    remove_dir(markdown_dir, missing_ok=True, with_contents=True)
            check_mkdir(markdown_dir, **cache_mkdir_kwargs)
        else:
            markdown_dir = Path(tempfile.mkdtemp(prefix=get_caller_qualname() + "_"))
        link_map = {d["link"]: dict(d) for d in self.data}
        url_paths = self.urls_to_paths(link_map.keys(), "md")
        url_file_map = dict(zip(link_map.keys(), [markdown_dir / p for p in url_paths]))
        _, to_html_kwargs = ToMarkdownAssetFunc.prepare(**kwargs)

        if show_progress is None:
            show_progress = not self.single_item
        prefix = get_caller_qualname().split(".")[-1]
        pbar_kwargs = flat_merge_dicts(
            dict(
                bar_id=get_caller_qualname(),
                prefix=prefix,
            ),
            pbar_kwargs,
        )
        with ProgressBar(total=len(self.data), show_progress=show_progress, **pbar_kwargs) as pbar:
            for d in self.data:
                if not url_file_map[d["link"]].exists():
                    markdown_content = ToMarkdownAssetFunc.call(d, **to_html_kwargs)
                    check_mkdir(url_file_map[d["link"]].parent, mkdir=True)
                    with open(url_file_map[d["link"]], "w", encoding="utf-8") as f:
                        f.write(markdown_content)
                pbar.update()

        return markdown_dir

    def to_html(
        self: VBTAssetT,
        root_metadata_key: tp.Optional[tp.Key] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        html_kwargs: tp.KwargsLike = None,
        use_pygments: tp.Optional[bool] = None,
        formatter_kwargs: tp.KwargsLike = None,
        css_style: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.Union[VBTAssetT, tp.Any]:
        """Convert to HTML.

        Uses `VBTAsset.apply` on `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc`.

        Arguments in `html_kwargs` are distributed among two functions:
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc.preprocess_html`
        and `markdown.markdown`.

        If `use_pygments` is True, uses Pygments package for code highlighting. Arguments in
        `formatter_kwargs` are then passed to `pygments.formatters.HtmlFormatter.

        Use `css_style` to set additional CSS style or override the existing one.

        For other arguments, see `VBTAsset.to_markdown`."""
        return self.apply(
            "to_html",
            root_metadata_key=root_metadata_key,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            html_kwargs=html_kwargs,
            use_pygments=use_pygments,
            formatter_kwargs=formatter_kwargs,
            css_style=css_style,
            **kwargs,
        )

    def get_top_parent_links(self) -> tp.List[str]:
        """Get links of top parents."""
        link_map = {d["link"]: dict(d) for d in self.data}
        top_parents = []
        for d in self.data:
            if d.get("parent", None) is None or d["parent"] not in link_map:
                top_parents.append(d["link"])
        return top_parents

    @classmethod
    def replace_urls_in_html(cls, html: str, url_map: dict) -> str:
        """Replace URLs in <a href="..."> attributes based on a provided mapping."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("bs4")

        from bs4 import BeautifulSoup
        from urllib.parse import urlparse, urlunparse

        soup = BeautifulSoup(html, "html.parser")

        for a_tag in soup.find_all("a", href=True):
            original_href = a_tag["href"]
            if original_href in url_map:
                a_tag["href"] = url_map[original_href]
            else:
                try:
                    parsed_href = urlparse(original_href)
                    base_url = urlunparse(parsed_href._replace(fragment=""))
                    if base_url in url_map:
                        new_base_url = url_map[base_url]
                        new_parsed = urlparse(new_base_url)
                        new_parsed = new_parsed._replace(fragment=parsed_href.fragment)
                        new_href = urlunparse(new_parsed)
                        a_tag["href"] = new_href
                except ValueError:
                    pass
        return str(soup)

    def save_to_html(
        self,
        cache: tp.Optional[bool] = None,
        cache_dir: tp.Optional[tp.PathLike] = None,
        cache_mkdir_kwargs: tp.KwargsLike = None,
        clear_cache: bool = False,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        return_url_map: bool = False,
        **kwargs,
    ) -> tp.Union[Path, dict]:
        """Save to HTML files.

        In addition, if there are multiple top-level parents, creates an index page.

        If `cache` is True, uses the cache directory. Otherwise, creates a temporary directory.
        If `clear_cache` is True, deletes any existing directory before creating a new one.
        Returns the path of the directory where HTML files are stored.

        Keyword arguments are passed to `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc`."""
        import tempfile
        from vectorbtpro.utils.knowledge.custom_asset_funcs import ToHTMLAssetFunc

        cache = self.resolve_setting(cache, "cache")
        cache_dir = self.resolve_setting(cache_dir, "cache_dir")
        cache_mkdir_kwargs = self.resolve_setting(cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True)
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)

        if cache:
            html_dir = Path(cache_dir) / "html"
            if html_dir.exists():
                if clear_cache:
                    remove_dir(html_dir, missing_ok=True, with_contents=True)
            check_mkdir(html_dir, **cache_mkdir_kwargs)
        else:
            html_dir = Path(tempfile.mkdtemp(prefix=get_caller_qualname() + "_"))
        link_map = {d["link"]: dict(d) for d in self.data}
        top_parents = self.get_top_parent_links()
        if len(top_parents) > 1:
            link_map["/"] = {}
        url_paths = self.urls_to_paths(link_map.keys(), "html")
        url_file_map = dict(zip(link_map.keys(), [html_dir / p for p in url_paths]))
        url_map = {k: "file://" + str(v.resolve()) for k, v in url_file_map.items()}
        _, to_html_kwargs = ToHTMLAssetFunc.prepare(**kwargs)

        if len(top_parents) > 1:
            entry_link = "/"
            if not url_file_map[entry_link].exists():
                html = ToHTMLAssetFunc.call([link_map[link] for link in top_parents], **to_html_kwargs)
                html = self.replace_urls_in_html(html, url_map)
                check_mkdir(url_file_map[entry_link].parent, mkdir=True)
                with open(url_file_map[entry_link], "w", encoding="utf-8") as f:
                    f.write(html)

        if show_progress is None:
            show_progress = not self.single_item
        prefix = get_caller_qualname().split(".")[-1]
        pbar_kwargs = flat_merge_dicts(
            dict(
                bar_id=get_caller_qualname(),
                prefix=prefix,
            ),
            pbar_kwargs,
        )
        with ProgressBar(total=len(self.data), show_progress=show_progress, **pbar_kwargs) as pbar:
            for d in self.data:
                if not url_file_map[d["link"]].exists():
                    html = ToHTMLAssetFunc.call(d, **to_html_kwargs)
                    html = self.replace_urls_in_html(html, url_map)
                    check_mkdir(url_file_map[d["link"]].parent, mkdir=True)
                    with open(url_file_map[d["link"]], "w", encoding="utf-8") as f:
                        f.write(html)
                pbar.update()

        if return_url_map:
            return url_map
        return html_dir

    def browse(
        self,
        entry_link: tp.Optional[tp.MaybeList[str]] = None,
        entry_only: bool = False,
        **kwargs,
    ) -> None:
        """Browse one or more pages.

        Use `entry_link` to specify the link of the page that should be displayed first.
        If `entry_link` is None and there are multiple top-level parents, displays them as an index.

        If `entry_only` is True, saves to HTML and displays only those links that are in `entry_link`.
        Otherwise, stores **all** links and then displays only those in `entry_link`.

        Keyword arguments are passed to `PagesAsset.save_to_html`."""
        import webbrowser

        top_parents = self.get_top_parent_links()
        if entry_link is None:
            if self.single_item:
                entry_link = self.data[0]["link"]
            else:
                if len(top_parents) == 1:
                    entry_link = top_parents[0]
                else:
                    entry_link = "/"
        if entry_only:
            if entry_link == "/":
                new_instance = self.find_by_link(top_parents)
            else:
                new_instance = self.find_by_link(entry_link, single_item=True)
        else:
            new_instance = self
        url_map = new_instance.save_to_html(return_url_map=True, **kwargs)
        webbrowser.open(url_map[entry_link])


MessagesAssetT = tp.TypeVar("MessagesAssetT", bound="MessagesAsset")


class MessagesAsset(VBTAsset):
    """Class for working with Discord messages.

    For defaults, see `assets.messages` in `vectorbtpro._settings.knowledge`."""

    _settings_path: tp.SettingsPath = "knowledge.assets.messages"

    def aggregate_messages(
        self: MessagesAssetT,
        metadata_format: tp.Optional[str] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[MessagesAssetT, tp.Any]:
        """Aggregate attachments by message.

        Argument `metadata_format` can be either "markdown" or "html". For keyword arguments, see
        `MessagesAsset.to_markdown` and `MessagesAsset.to_html` respectively.

        Uses `MessagesAsset.apply` on `vectorbtpro.utils.knowledge.custom_asset_funcs.AggMessageAssetFunc`."""
        return self.apply(
            "agg_message",
            metadata_format=metadata_format,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            html_kwargs=html_kwargs,
            **kwargs,
        )

    def aggregate_blocks(
        self: MessagesAssetT,
        collect_kwargs: tp.KwargsLike = None,
        aggregate_fields: tp.Union[None, bool, tp.MaybeSet[str]] = None,
        parent_links_only: tp.Optional[bool] = None,
        metadata_format: tp.Optional[str] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[MessagesAssetT, tp.Any]:
        """Aggregate messages by block.

        First, uses `MessagesAsset.reduce` on `vectorbtpro.utils.knowledge.base_asset_funcs.CollectAssetFunc`
        to collect data items by the field "block". Keyword arguments in `collect_kwargs` are passed here.
        Argument `uniform_groups` is True by default. Then, uses `MessagesAsset.apply` on
        `vectorbtpro.utils.knowledge.custom_asset_funcs.AggBlockAssetFunc` to aggregate each collected data item.

        Use `aggregate_fields` to provide a set of fields to be aggregated rather than used in child metadata.
        It can be True to aggregate all lists and False to aggregate none.

        If `parent_links_only` is True, doesn't include links in the metadata of each message.

        Argument `metadata_format` can be either "markdown" or "html". For other keyword arguments, see
        `MessagesAsset.to_markdown` and `MessagesAsset.to_html` respectively."""
        if collect_kwargs is None:
            collect_kwargs = {}
        if "uniform_groups" not in collect_kwargs:
            collect_kwargs["uniform_groups"] = True
        instance = self.collect(by_path="block", wrap=True, **collect_kwargs)
        return instance.apply(
            "agg_block",
            aggregate_fields=aggregate_fields,
            parent_links_only=parent_links_only,
            metadata_format=metadata_format,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            html_kwargs=html_kwargs,
            link_map={d["link"]: dict(d) for d in self.data},
            **kwargs,
        )

    def aggregate_threads(
        self: MessagesAssetT,
        collect_kwargs: tp.KwargsLike = None,
        aggregate_fields: tp.Union[None, bool, tp.MaybeSet[str]] = None,
        parent_links_only: tp.Optional[bool] = None,
        metadata_format: tp.Optional[str] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[MessagesAssetT, tp.Any]:
        """Aggregate messages by thread.

        Same as `MessagesAsset.aggregate_blocks` but for threads.

        Uses `vectorbtpro.utils.knowledge.custom_asset_funcs.AggThreadAssetFunc`."""
        if collect_kwargs is None:
            collect_kwargs = {}
        if "uniform_groups" not in collect_kwargs:
            collect_kwargs["uniform_groups"] = True
        instance = self.collect(by_path="thread", wrap=True, **collect_kwargs)
        return instance.apply(
            "agg_thread",
            aggregate_fields=aggregate_fields,
            parent_links_only=parent_links_only,
            metadata_format=metadata_format,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            html_kwargs=html_kwargs,
            link_map={d["link"]: dict(d) for d in self.data},
            **kwargs,
        )

    def aggregate_channels(
        self: MessagesAssetT,
        collect_kwargs: tp.KwargsLike = None,
        aggregate_fields: tp.Union[None, bool, tp.MaybeSet[str]] = None,
        parent_links_only: tp.Optional[bool] = None,
        metadata_format: tp.Optional[str] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[MessagesAssetT, tp.Any]:
        """Aggregate messages by channel.

        Same as `MessagesAsset.aggregate_threads` but for channels.

        Uses `vectorbtpro.utils.knowledge.custom_asset_funcs.AggChannelAssetFunc`."""
        if collect_kwargs is None:
            collect_kwargs = {}
        if "uniform_groups" not in collect_kwargs:
            collect_kwargs["uniform_groups"] = True
        instance = self.collect(by_path="channel", wrap=True, **collect_kwargs)
        return instance.apply(
            "agg_channel",
            aggregate_fields=aggregate_fields,
            parent_links_only=parent_links_only,
            metadata_format=metadata_format,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            html_kwargs=html_kwargs,
            link_map={d["link"]: dict(d) for d in self.data},
            **kwargs,
        )

    def aggregate_servers(
        self: MessagesAssetT,
        collect_kwargs: tp.KwargsLike = None,
        aggregate_fields: tp.Union[None, bool, tp.MaybeSet[str]] = None,
        parent_links_only: tp.Optional[bool] = None,
        metadata_format: tp.Optional[str] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[MessagesAssetT, tp.Any]:
        """Aggregate messages by server.

        Same as `MessagesAsset.aggregate_channels` but for servers.

        Uses `vectorbtpro.utils.knowledge.custom_asset_funcs.AggServerAssetFunc`."""
        from vectorbtpro.utils.knowledge.custom_asset_funcs import AggServerAssetFunc

        server_link_template = RepFunc(lambda x: AggServerAssetFunc.get_server_link(x))
        collect_kwargs = deep_merge_dicts(
            dict(get_kwargs=dict(source=server_link_template)),
            collect_kwargs,
        )
        instance = self.collect(by_path="link", wrap=True, **collect_kwargs)
        return instance.apply(
            "agg_server",
            aggregate_fields=aggregate_fields,
            parent_links_only=parent_links_only,
            metadata_format=metadata_format,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            html_kwargs=html_kwargs,
            link_map={d["link"]: dict(d) for d in self.data},
            **kwargs,
        )

    def aggregate(self, by: str, *args, **kwargs) -> tp.Union[MessagesAssetT, tp.Any]:
        """Aggregate by "message" (attachments), "block", "channel", "thread", or "server"."""
        if not by.lower().endswith("s"):
            by += "s"
        return getattr(self, "aggregate_" + by.lower())(*args, **kwargs)


PagesAssetT = tp.TypeVar("PagesAssetT", bound="PagesAsset")


class PagesAsset(VBTAsset):
    """Class for working with website pages.

    For defaults, see `assets.pages` in `vectorbtpro._settings.knowledge`."""

    _settings_path: tp.SettingsPath = "knowledge.assets.pages"

    def find_page_for(
        self,
        obj: tp.Any,
        module: tp.Union[None, str, ModuleType] = None,
        resolve: bool = True,
        **kwargs,
    ) -> PagesAssetT:
        """Find the page or heading corresponding to an (internal) object or reference name."""
        refname = prepare_refname(obj, module=module, resolve=resolve)
        return self.find_by_link(f"#({re.escape(refname)})$", mode="regex", single_item=True, **kwargs)

    def aggregate(
        self: PagesAssetT,
        append_obj_type: tp.Optional[bool] = None,
        append_github_link: tp.Optional[bool] = None,
    ) -> PagesAssetT:
        """Aggregate pages and headings.

        Content of each heading will be converted into markdown and concatenated into the content
        of the parent heading or page. Only regular pages and headings without parents will be left.

        If `append_obj_type` is True, will also append object type to the heading name.
        If `append_github_link` is True, will also append GitHub link to the heading name."""
        append_obj_type = self.resolve_setting(append_obj_type, "append_obj_type")
        append_github_link = self.resolve_setting(append_github_link, "append_github_link")

        link_map = {d["link"]: dict(d) for d in self.data}
        top_parents = self.get_top_parent_links()
        aggregated_links = set()

        def _aggregate_content(link):
            node = link_map[link]
            content = node["content"]
            if content is None:
                content = ""
            if node["type"].startswith("heading"):
                level = int(node["type"].split(" ")[1])
                heading_markdown = "#" * level + " " + node["name"]
                if append_obj_type and node.get("obj_type", None) is not None:
                    heading_markdown += f" | {node['obj_type']}"
                if append_github_link and node.get("github_link", None) is not None:
                    heading_markdown += f" | [source]({node['github_link']})"
                if content == "":
                    content = heading_markdown
                else:
                    content = f"{heading_markdown}\n\n{content}"

            children = list(node["children"])
            for child in list(children):
                child_node = link_map[child]
                child_content = _aggregate_content(child)
                if child_node["type"].startswith("heading"):
                    if child_content.startswith("# "):
                        content = child_content
                    else:
                        content += f"\n\n{child_content}"
                    children.remove(child)
                    aggregated_links.add(child)

            if content != "":
                node["content"] = content
            node["children"] = children
            return content

        for top_parent in top_parents:
            _aggregate_content(top_parent)

        new_data = [link_map[link] for link in link_map if link not in aggregated_links]
        return self.replace(data=new_data)
