# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Custom asset classes."""

import os
import re
from types import ModuleType
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.utils.path_ import check_mkdir, remove_dir
from vectorbtpro.utils.module_ import prepare_refname, get_caller_qualname
from vectorbtpro.utils.knowledge.base_assets import ReleaseAsset

__all__ = [
    "MessagesAsset",
    "PagesAsset",
]


MinimizerMixinT = tp.TypeVar("MinimizerMixinT", bound="MinimizerMixin")


class MinimizerMixin:
    """Mixin for minimizing knowledge assets."""

    def minimize_links(self: MinimizerMixinT) -> MinimizerMixinT:
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

    def minimize(self: MinimizerMixinT, minimize_links: tp.Optional[bool] = None) -> MinimizerMixinT:
        """Minimize by keeping the most useful information."""
        minimize_links = self.resolve_setting(minimize_links, "minimize_links")

        new_instance = self.find_remove_empty()
        if minimize_links:
            return new_instance.minimize_links()
        return new_instance


class ConverterMixin:
    """Mixin for converting knowledge assets to different formats."""

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

    @classmethod
    def preprocess_markdown(cls, content: str) -> str:
        """Preprocess Markdown."""

        def _replace_code_block(match):
            language = match.group(1)
            title = match.group(2)
            code = match.group(3)
            if title:
                title_md = f"**{title}**\n\n"
            else:
                title_md = ""
            code_md = f"```{language}\n{code}\n```"
            return title_md + code_md

        code_block_pattern = re.compile(r'```(\w+)\s+title="([^"]*)"\s*\n(.*?)\n```', re.DOTALL)
        content = code_block_pattern.sub(_replace_code_block, content)
        return content.strip()

    @classmethod
    def get_markdown_metadata(cls, d: dict, metadata_dump_kwargs: tp.KwargsLike = None) -> str:
        """Get metadata of a page or heading in Markdown format."""
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc, DumpAssetFunc

        metadata_dump_kwargs = cls.resolve_setting(metadata_dump_kwargs, "metadata_dump_kwargs", merge=True)

        d = dict(d)
        del d["content"]
        metadata = FindRemoveAssetFunc.prepare_and_call(d, target=FindRemoveAssetFunc.is_empty_func)
        metadata = DumpAssetFunc.prepare_and_call({"metadata": metadata}, **metadata_dump_kwargs)
        metadata = cls.preprocess_markdown(metadata)
        return metadata

    @classmethod
    def get_markdown_content(cls, d: dict) -> str:
        """Get content of a page or heading in Markdown format."""
        if d["content"] is None:
            return ""
        content = cls.preprocess_markdown(d["content"])
        return content

    def save_to_markdown(
        self,
        metadata_dump_kwargs: tp.KwargsLike = None,
        cache: tp.Optional[bool] = None,
        cache_dir: tp.Optional[tp.PathLike] = None,
        cache_mkdir_kwargs: tp.KwargsLike = None,
        clear_cache: bool = False,
    ) -> Path:
        """Save pages and headings to Markdown files.

        Returns the path of the directory where Markdown files are stored.

        If `cache` is True, uses the cache directory. Otherwise, creates a temporary directory."""
        import tempfile

        cache = self.resolve_setting(cache, "cache")
        cache_dir = self.resolve_setting(cache_dir, "cache_dir")
        cache_mkdir_kwargs = self.resolve_setting(cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True)

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

        for d in self.data:
            if not url_file_map[d["link"]].exists():
                markdown_metadata = self.get_markdown_metadata(d, metadata_dump_kwargs=metadata_dump_kwargs)
                if markdown_metadata:
                    markdown_metadata = "---\n" + markdown_metadata + "\n---"
                markdown_content = self.get_markdown_content(d)
                if markdown_metadata and markdown_content:
                    markdown_content = markdown_metadata + "\n\n" + markdown_content
                elif markdown_metadata:
                    markdown_content = markdown_metadata
                check_mkdir(url_file_map[d["link"]].parent, mkdir=True)
                with open(url_file_map[d["link"]], "w", encoding="utf-8") as f:
                    f.write(markdown_content)

        return markdown_dir

    @classmethod
    def get_html_metadata(
        cls,
        d: dict,
        metadata_dump_kwargs: tp.KwargsLike = None,
        markdown_kwargs: tp.KwargsLike = None,
    ) -> str:
        """Get metadata of a page or heading in HTML format."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("markdown")
        import markdown

        metadata = cls.get_markdown_metadata(d, metadata_dump_kwargs=metadata_dump_kwargs)
        metadata = re.compile(r"(https?://[^\s]+)").sub(r'<a href="\1">\1</a>', metadata)
        if markdown_kwargs is None:
            markdown_kwargs = {}
        return markdown.markdown(metadata, **markdown_kwargs)

    @classmethod
    def get_html_content(cls, d: dict, markdown_kwargs: tp.KwargsLike = None) -> str:
        """Get content of a page or heading in HTML format."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("markdown")
        import markdown

        content = cls.get_markdown_content(d)
        if markdown_kwargs is None:
            markdown_kwargs = {}
        return markdown.markdown(content, **markdown_kwargs)

    @classmethod
    def format_html_template(
        cls,
        title: str = "",
        html_metadata: tp.MaybeList[str] = "",
        html_content: str = "",
        use_pygments: tp.Optional[bool] = None,
        formatter_kwargs: tp.KwargsLike = None,
        css_style: tp.Optional[str] = None,
    ) -> str:
        """Format HTML template."""
        from vectorbtpro.utils.module_ import check_installed, assert_can_import

        use_pygments = cls.resolve_setting(use_pygments, "use_pygments")
        formatter_kwargs = cls.resolve_setting(formatter_kwargs, "formatter_kwargs")
        css_style = cls.resolve_setting(css_style, "css_style")

        if use_pygments is None:
            use_pygments = check_installed("pygments")
        if use_pygments:
            assert_can_import("pygments")
            from pygments.formatters import HtmlFormatter

            formatter = HtmlFormatter(**formatter_kwargs)
            more_css_style = formatter.get_style_defs(".codehilite")
            if css_style == "":
                css_style = more_css_style
            else:
                css_style = more_css_style + "\n" + css_style

        if html_metadata:
            if isinstance(html_metadata, list):
                html_metadata = "\n".join([f'<div class="metadata">{m}</div>' for m in html_metadata])
            else:
                html_metadata = f'<div class="metadata">{html_metadata}</div>'
        if html_content:
            html_content = f'<div class="content">{html_content}</div>'
        return f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="utf-8">
                        <title>{title}</title>
                        <style>
                            .metadata {{
                                background-color: #f8f8f8;
                                margin: 10px;
                                padding: 0 40px;
                                font-family: monospace;
                                white-space: pre;
                                border: 1px solid #ddd;
                                border-radius: 4px;
                            }}
                            .content {{
                                font-family: Arial, sans-serif;
                                margin: 40px;
                                line-height: 1.6;
                            }}
                            h1, h2, h3, h4, h5, h6 {{
                                color: #333;
                            }}
                            pre {{
                                background-color: #f8f8f8;
                                padding: 10px;
                                border: 1px solid #ddd;
                                border-radius: 4px;
                            }}
                            .admonition {{
                                background-color: #f9f9f9;
                                margin: 20px 0;
                                padding: 10px 20px;
                                border-left: 5px solid #ccc;
                                border-radius: 4px;
                            }}
                            .admonition > p:first-child {{
                                font-weight: bold;
                                margin-bottom: 5px;
                            }}
                            {css_style}
                        </style>
                    </head>
                    <body>
                        {html_metadata}
                        {html_content}
                    </body>
                    </html>
                    """

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
                parsed_href = urlparse(original_href)
                base_url = urlunparse(parsed_href._replace(fragment=""))
                if base_url in url_map:
                    new_base_url = url_map[base_url]
                    new_parsed = urlparse(new_base_url)
                    new_parsed = new_parsed._replace(fragment=parsed_href.fragment)
                    new_href = urlunparse(new_parsed)
                    a_tag["href"] = new_href
        return str(soup)

    def get_top_parent_links(self) -> tp.List[str]:
        """Get links of top parents."""
        link_map = {d["link"]: dict(d) for d in self.data}
        top_parents = []
        for d in self.data:
            if d.get("parent", None) is None or d["parent"] not in link_map:
                top_parents.append(d["link"])
        return top_parents

    def save_to_html(
        self,
        metadata_dump_kwargs: tp.KwargsLike = None,
        markdown_kwargs: tp.KwargsLike = None,
        use_pygments: tp.Optional[bool] = None,
        formatter_kwargs: tp.KwargsLike = None,
        css_style: tp.Optional[str] = None,
        cache: tp.Optional[bool] = None,
        cache_dir: tp.Optional[tp.PathLike] = None,
        cache_mkdir_kwargs: tp.KwargsLike = None,
        clear_cache: bool = False,
        return_url_map: bool = False,
    ) -> tp.Union[Path, dict]:
        """Save pages and headings to HTML files.

        Returns the path of the directory where HTML files are stored.

        In addition, if there are multiple top-level parents, creates an index page.

        If `cache` is True, uses the cache directory. Otherwise, creates a temporary directory."""
        import tempfile

        cache = self.resolve_setting(cache, "cache")
        cache_dir = self.resolve_setting(cache_dir, "cache_dir")
        cache_mkdir_kwargs = self.resolve_setting(cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True)

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

        if len(top_parents) > 1:
            entry_link = "/"
            if not url_file_map[entry_link].exists():
                html_metadata = []
                for link in top_parents:
                    html_metadata.append(
                        self.get_html_metadata(
                            link_map[link],
                            metadata_dump_kwargs=metadata_dump_kwargs,
                            markdown_kwargs=markdown_kwargs,
                        )
                    )
                html = self.format_html_template(
                    title=entry_link,
                    html_metadata=html_metadata,
                    use_pygments=use_pygments,
                    formatter_kwargs=formatter_kwargs,
                    css_style=css_style,
                )
                html = self.replace_urls_in_html(html, url_map)
                check_mkdir(url_file_map[entry_link].parent, mkdir=True)
                with open(url_file_map[entry_link], "w", encoding="utf-8") as f:
                    f.write(html)
        for d in self.data:
            if not url_file_map[d["link"]].exists():
                html_metadata = self.get_html_metadata(
                    d,
                    metadata_dump_kwargs=metadata_dump_kwargs,
                    markdown_kwargs=markdown_kwargs,
                )
                html_content = self.get_html_content(d, markdown_kwargs=markdown_kwargs)
                html = self.format_html_template(
                    title=d["link"],
                    html_metadata=html_metadata,
                    html_content=html_content,
                    use_pygments=use_pygments,
                    formatter_kwargs=formatter_kwargs,
                    css_style=css_style,
                )
                html = self.replace_urls_in_html(html, url_map)
                check_mkdir(url_file_map[d["link"]].parent, mkdir=True)
                with open(url_file_map[d["link"]], "w", encoding="utf-8") as f:
                    f.write(html)

        if return_url_map:
            return url_map
        return html_dir

    def browse(
        self,
        entry_link: tp.Optional[tp.MaybeList[str]] = None,
        entry_only: bool = False,
        **kwargs,
    ) -> None:
        """Browse one or more pages or headings.

        Use `entry_link` to specify the link of the page that should be displayed first.
        If `entry_link` is None and there are multiple top-level parents, displays them as an index.

        Other keyword arguments are passed to `PagesAsset.to_html`."""
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
                new_instance = self.find_pages(top_parents)
            else:
                new_instance = self.find_page(entry_link)
        else:
            new_instance = self
        url_map = new_instance.save_to_html(return_url_map=True, **kwargs)
        webbrowser.open(url_map[entry_link])


MessagesAssetT = tp.TypeVar("MessagesAssetT", bound="MessagesAsset")


class MessagesAsset(ReleaseAsset, MinimizerMixin, ConverterMixin):
    """Class for working with Discord messages.

    For defaults, see `assets.messages` in `vectorbtpro._settings.knowledge`."""

    _settings_path: tp.SettingsPath = "knowledge.assets.messages"

    def find_messages(
        self,
        target: tp.MaybeList[tp.Any],
        path: tp.MaybeList[tp.PathLikeKey] = "link",
        mode: str = "exact",
        **kwargs,
    ) -> MessagesAssetT:
        """Find one to multiple messages (by its link by default)."""
        found = self.find(target, path=path, mode=mode, **kwargs)
        if len(found) == 0:
            raise ValueError(f"No message with '{path}' matching '{target}'")
        return found

    def find_message(
        self,
        target: tp.MaybeList[tp.Any],
        path: tp.MaybeList[tp.PathLikeKey] = "link",
        mode: str = "exact",
        **kwargs,
    ) -> MessagesAssetT:
        """Find one message (by its link by default)."""
        found = self.find_messages(target, path=path, mode=mode, single_item=True, **kwargs)
        if len(found) > 1:
            links_block = "\n".join(found.get("link"))
            raise ValueError(f"Multiple messages with '{path}' matching '{target}':\n\n{links_block}")
        return found

    def aggregate_attachments(
        self: MessagesAssetT,
        clear_metadata: tp.Optional[bool] = None,
        metadata_clear_kwargs: tp.KwargsLike = None,
        metadata_dump_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> MessagesAssetT:
        """Aggregate attachments.

        Uses `MessagesAsset.apply` on `vectorbtpro.utils.knowledge.custom_asset_funcs.AggAttachAssetFunc`."""
        return self.apply(
            "agg_attach",
            clear_metadata=clear_metadata,
            metadata_clear_kwargs=metadata_clear_kwargs,
            metadata_dump_kwargs=metadata_dump_kwargs,
            **kwargs,
        )

    def aggregate_blocks(
        self: MessagesAssetT,
        collect_kwargs: tp.KwargsLike = None,
        aggregate_fields: tp.Union[None, bool, tp.MaybeSet[str]] = None,
        block_links_only: tp.Optional[bool] = None,
        clear_metadata: tp.Optional[bool] = None,
        metadata_clear_kwargs: tp.KwargsLike = None,
        metadata_dump_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> MessagesAssetT:
        """Aggregate blocks.

        If `block_links_only` is True, doesn't include links in the metadata of each message.

        Argument `uniform_groups` is True by default.

        Uses `MessagesAsset.apply` on `vectorbtpro.utils.knowledge.custom_asset_funcs.AggBlockAssetFunc`."""
        if collect_kwargs is None:
            collect_kwargs = {}
        if "uniform_groups" not in collect_kwargs:
            collect_kwargs["uniform_groups"] = True
        instance = self.collect(by_path="block", wrap=True, **collect_kwargs)
        return instance.apply(
            "agg_block",
            aggregate_fields=aggregate_fields,
            block_links_only=block_links_only,
            clear_metadata=clear_metadata,
            metadata_clear_kwargs=metadata_clear_kwargs,
            metadata_dump_kwargs=metadata_dump_kwargs,
            link_map={d["link"]: dict(d) for d in self.data},
            **kwargs,
        )

    def aggregate_threads(
        self: MessagesAssetT,
        collect_kwargs: tp.KwargsLike = None,
        aggregate_fields: tp.Union[None, bool, tp.MaybeSet[str]] = None,
        thread_links_only: tp.Optional[bool] = None,
        clear_metadata: tp.Optional[bool] = None,
        metadata_clear_kwargs: tp.KwargsLike = None,
        metadata_dump_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> MessagesAssetT:
        """Aggregate threads.

        If `thread_links_only` is True, doesn't include links in the metadata of each message.

        Argument `uniform_groups` is True by default.

        Uses `MessagesAsset.apply` on `vectorbtpro.utils.knowledge.custom_asset_funcs.AggThreadAssetFunc`."""
        if collect_kwargs is None:
            collect_kwargs = {}
        if "uniform_groups" not in collect_kwargs:
            collect_kwargs["uniform_groups"] = True
        instance = self.collect(by_path="thread", wrap=True, **collect_kwargs)
        return instance.apply(
            "agg_thread",
            aggregate_fields=aggregate_fields,
            thread_links_only=thread_links_only,
            clear_metadata=clear_metadata,
            metadata_clear_kwargs=metadata_clear_kwargs,
            metadata_dump_kwargs=metadata_dump_kwargs,
            link_map={d["link"]: dict(d) for d in self.data},
            **kwargs,
        )


PagesAssetT = tp.TypeVar("PagesAssetT", bound="PagesAsset")


class PagesAsset(ReleaseAsset, MinimizerMixin, ConverterMixin):
    """Class for working with website pages.

    For defaults, see `assets.pages` in `vectorbtpro._settings.knowledge`."""

    _settings_path: tp.SettingsPath = "knowledge.assets.pages"

    def find_pages(
        self,
        target: tp.MaybeList[tp.Any],
        path: tp.MaybeList[tp.PathLikeKey] = "link",
        mode: str = "exact",
        **kwargs,
    ) -> PagesAssetT:
        """Find one to multiple pages or headings (by its link by default)."""
        found = self.find(target, path=path, mode=mode, **kwargs)
        if len(found) == 0:
            raise ValueError(f"No page or heading with '{path}' matching '{target}'")
        return found

    def find_page(
        self,
        target: tp.MaybeList[tp.Any],
        path: tp.MaybeList[tp.PathLikeKey] = "link",
        mode: str = "exact",
        **kwargs,
    ) -> PagesAssetT:
        """Find one page or heading (by its link by default)."""
        found = self.find_pages(target, path=path, mode=mode, single_item=True, **kwargs)
        if len(found) > 1:
            links_block = "\n".join(found.get("link"))
            raise ValueError(f"Multiple pages or headings with '{path}' matching '{target}':\n\n{links_block}")
        return found

    def find_page_for(
        self,
        obj: tp.Any,
        module: tp.Union[None, str, ModuleType] = None,
        resolve: bool = True,
        **kwargs,
    ) -> PagesAssetT:
        """Find the page or heading corresponding to an (internal) object or reference name."""
        refname = prepare_refname(obj, module=module, resolve=resolve)
        return self.find_page(f"#({re.escape(refname)})$", mode="regex", **kwargs)

    def aggregate(
        self: PagesAssetT,
        append_obj_type: tp.Optional[bool] = None,
        append_github_link: tp.Optional[bool] = None,
    ) -> PagesAssetT:
        """Aggregate pages and headings.

        Content of each heading will be converted into markdown and concatenated into the content
        of the parent heading or page. Only regular pages and headings without parents will be left."""
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
