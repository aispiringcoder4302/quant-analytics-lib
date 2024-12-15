# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Custom asset classes."""

import inspect
import io
import os
import pkgutil
import re
from collections import defaultdict, deque
from pathlib import Path
from types import ModuleType

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, flat_merge_dicts, reorder_list, HybridConfig
from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset
from vectorbtpro.utils.module_ import prepare_refname, get_caller_qualname
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.path_ import check_mkdir, remove_dir, get_common_prefix, dir_tree_from_paths
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.pickling import suggest_compression
from vectorbtpro.utils.search import find

__all__ = [
    "VBTAsset",
    "PagesAsset",
    "MessagesAsset",
    "find_api",
    "find_docs",
    "find_messages",
    "find_examples",
    "find_assets",
    "chat_about",
]


__pdoc__ = {}

class_abbr_config = HybridConfig(
    dict(
        Accessor={"acc"},
        Array={"arr"},
        ArrayWrapper={"wrapper"},
        Benchmark={"bm"},
        Cacheable={"ca"},
        Chunkable={"ch"},
        Drawdowns={"dd"},
        Jitable={"jit"},
        Figure={"fig"},
        MappedArray={"ma"},
        NumPy={"np"},
        Numba={"nb"},
        Optimizer={"opt"},
        Pandas={"pd"},
        Portfolio={"pf"},
        ProgressBar={"pbar"},
        Registry={"reg"},
        Returns_={"ret"},
        Returns={"rets"},
        QuantStats={"qs"},
        Signals_={"sig"},
    )
)
"""_"""

__pdoc__[
    "class_abbr_config"
] = f"""Config for class name (part) abbreviations.

```python
{class_abbr_config.prettify()}
```
"""


class NoItemFoundError(Exception):
    """Exception raised when no data item was found."""


class MultipleItemsFoundError(Exception):
    """Exception raised when multiple data items were found."""


VBTAssetT = tp.TypeVar("VBTAssetT", bound="VBTAsset")


class VBTAsset(KnowledgeAsset):
    """Class for working with VBT content.

    For defaults, see `assets.vbt` in `vectorbtpro._settings.knowledge`."""

    _settings_path: tp.SettingsPath = "knowledge.assets.vbt"

    @classmethod
    def pull(
        cls: tp.Type[VBTAssetT],
        asset_name: tp.Optional[str] = None,
        release_name: tp.Optional[str] = None,
        repo_owner: tp.Optional[str] = None,
        repo_name: tp.Optional[str] = None,
        token: tp.Optional[str] = None,
        token_required: tp.Optional[bool] = None,
        use_pygithub: tp.Optional[bool] = None,
        chunk_size: tp.Optional[int] = None,
        cache: tp.Optional[bool] = None,
        cache_dir: tp.Optional[tp.PathLike] = None,
        cache_mkdir_kwargs: tp.KwargsLike = None,
        clear_cache: bool = False,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> VBTAssetT:
        """Build `VBTAsset` from a JSON asset of a release."""
        from vectorbtpro._version import __version__
        import requests

        asset_name = cls.resolve_setting(asset_name, "asset_name")
        release_name = cls.resolve_setting(release_name, "release_name")
        repo_owner = cls.resolve_setting(repo_owner, "repo_owner")
        repo_name = cls.resolve_setting(repo_name, "repo_name")
        token = cls.resolve_setting(token, "token")
        token_required = cls.resolve_setting(token_required, "token_required")
        use_pygithub = cls.resolve_setting(use_pygithub, "use_pygithub")
        chunk_size = cls.resolve_setting(chunk_size, "chunk_size")
        cache = cls.resolve_setting(cache, "cache")
        cache_dir_none = cache_dir is None
        cache_dir = cls.resolve_setting(cache_dir, "cache_dir")
        cache_mkdir_kwargs = cls.resolve_setting(cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True)
        show_progress = cls.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = cls.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)

        current_release = "v" + __version__
        if release_name is None:
            release_name = current_release
        release_dir = Path(cache_dir)
        if cache_dir_none:
            release_dir /= "releases"
            release_dir /= release_name
        if cache:
            if release_dir.exists():
                if clear_cache:
                    remove_dir(release_dir, missing_ok=True, with_contents=True)
                else:
                    cache_file = None
                    for file in release_dir.iterdir():
                        if file.is_file() and file.name == asset_name:
                            cache_file = file
                            break
                    if cache_file is not None:
                        return cls.from_json_file(cache_file, **kwargs)

        if token is None:
            token = os.environ.get("GITHUB_TOKEN", None)
        if token is None and token_required:
            raise ValueError("GitHub token is required")
        if use_pygithub is None:
            from vectorbtpro.utils.module_ import check_installed

            use_pygithub = check_installed("github")
        if use_pygithub:
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("github")
            from github import Github, Auth
            from github.GithubException import UnknownObjectException

            if token is not None:
                g = Github(auth=Auth.Token(token))
            else:
                g = Github()
            try:
                repo = g.get_repo(f"{repo_owner}/{repo_name}")
            except UnknownObjectException:
                raise Exception(f"Repository '{repo_owner}/{repo_name}' not found or access denied")
            if release_name == "latest":
                try:
                    release = repo.get_latest_release()
                except UnknownObjectException:
                    raise Exception("Latest release not found")
            else:
                releases = repo.get_releases()
                found_release = None
                for release in releases:
                    if release.title == release_name:
                        found_release = release
                if found_release is None:
                    raise Exception(f"Release '{release_name}' not found")
                release = found_release
            assets = release.get_assets()
            if asset_name is not None:
                asset = next((a for a in assets if a.name == asset_name), None)
                if asset is None:
                    raise Exception(f"Asset '{asset_name}' not found in release {release}")
            else:
                assets_list = list(assets)
                if len(assets_list) == 1:
                    asset = assets_list[0]
                else:
                    raise Exception("Please specify asset_name")
            asset_url = asset.url
        else:
            headers = {"Accept": "application/vnd.github+json"}
            if token is not None:
                headers["Authorization"] = f"token {token}"
            if release_name == "latest":
                release_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"
                response = requests.get(release_url, headers=headers)
                response.raise_for_status()
                release_info = response.json()
            else:
                releases_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases"
                response = requests.get(releases_url, headers=headers)
                response.raise_for_status()
                releases = response.json()
                release_info = None
                for release in releases:
                    if release.get("name") == release_name:
                        release_info = release
                if release_info is None:
                    raise ValueError(f"Release '{release_name}' not found")
            assets = release_info.get("assets", [])
            if asset_name is not None:
                asset = next((a for a in assets if a["name"] == asset_name), None)
                if asset is None:
                    raise Exception(f"Asset '{asset_name}' not found in release {release}")
            else:
                if len(assets) == 1:
                    asset = assets[0]
                else:
                    raise Exception("Please specify asset_name")
            asset_url = asset["url"]

        asset_headers = {"Accept": "application/octet-stream"}
        if token is not None:
            asset_headers["Authorization"] = f"token {token}"
        asset_response = requests.get(asset_url, headers=asset_headers, stream=True)
        asset_response.raise_for_status()
        file_size = int(asset_response.headers.get("Content-Length", 0))
        if file_size == 0:
            file_size = asset.get("size", 0)
        if show_progress is None:
            show_progress = True
        pbar_kwargs = flat_merge_dicts(
            dict(
                bar_id=get_caller_qualname(),
                unit="iB",
                unit_scale=True,
                prefix=f"Downloading {asset_name}",
            ),
            pbar_kwargs,
        )

        if cache:
            check_mkdir(release_dir, **cache_mkdir_kwargs)
            cache_file = release_dir / asset_name
            with open(cache_file, "wb") as f:
                with ProgressBar(total=file_size, show_progress=show_progress, **pbar_kwargs) as pbar:
                    for chunk in asset_response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return cls.from_json_file(cache_file, **kwargs)
        else:
            with io.BytesIO() as bytes_io:
                with ProgressBar(total=file_size, show_progress=show_progress, **pbar_kwargs) as pbar:
                    for chunk in asset_response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            bytes_io.write(chunk)
                            pbar.update(len(chunk))
                bytes_ = bytes_io.getvalue()
            compression = suggest_compression(asset_name)
            if compression is not None and "compression" not in kwargs:
                kwargs["compression"] = compression
            return cls.from_json_bytes(bytes_, **kwargs)

    def find_link(
        self: VBTAssetT,
        link: tp.MaybeList[str],
        mode: str = "end",
        per_path: bool = False,
        single_item: bool = True,
        consolidate: bool = True,
        allow_empty: bool = False,
        **kwargs,
    ) -> tp.MaybeVBTAsset:
        """Find item(s) corresponding to link(s)."""

        def _extend_link(link):
            from urllib.parse import urlparse

            if not urlparse(link).fragment:
                if link.endswith("/"):
                    return [link, link[:-1]]
                return [link, link + "/"]
            return [link]

        links = link
        if mode.lower() in ("exact", "end"):
            if isinstance(link, str):
                links = _extend_link(link)
            elif isinstance(link, list):
                from itertools import chain

                links = list(chain(*map(_extend_link, link)))
            else:
                raise TypeError("Link must be either string or list")
        found = self.find(links, path="link", mode=mode, per_path=per_path, single_item=single_item, **kwargs)
        if isinstance(found, (type(self), list)):
            if len(found) == 0:
                if allow_empty:
                    return found
                raise NoItemFoundError(f"No item matching '{link}'")
            if single_item and len(found) > 1:
                if consolidate:
                    top_parents = self.get_top_parent_links(list(found))
                    if len(top_parents) == 1:
                        for i, d in enumerate(found):
                            if d["link"] == top_parents[0]:
                                if isinstance(found, type(self)):
                                    return found.replace(data=[d], single_item=True)
                                return d
                links_block = "\n".join([d["link"] for d in found])
                raise MultipleItemsFoundError(f"Multiple items matching '{link}':\n\n{links_block}")
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

    def select_previous(self: VBTAssetT, link: str, **kwargs) -> VBTAssetT:
        """Select the previous data item."""
        d = self.find_link(link, wrap=False, **kwargs)
        d_index = self.index(d)
        new_data = []
        if d_index > 0:
            new_data.append(self.data[d_index - 1])
        return self.replace(data=new_data, single_item=True)

    def select_next(self: VBTAssetT, link: str, **kwargs) -> VBTAssetT:
        """Select the next data item."""
        d = self.find_link(link, wrap=False, **kwargs)
        d_index = self.index(d)
        new_data = []
        if d_index < len(self.data) - 1:
            new_data.append(self.data[d_index + 1])
        return self.replace(data=new_data, single_item=True)

    def to_markdown(
        self: VBTAssetT,
        root_metadata_key: tp.Optional[tp.Key] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeVBTAsset:
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
    def links_to_paths(
        cls,
        urls: tp.Iterable[str],
        extension: tp.Optional[str] = None,
        allow_fragments: bool = True,
    ) -> tp.List[Path]:
        """Convert links to corresponding paths."""
        from urllib.parse import urlparse

        url_paths = []
        for url in urls:
            parsed = urlparse(url, allow_fragments=allow_fragments)
            path_parts = [parsed.netloc]
            url_path = parsed.path.strip("/")
            if url_path:
                parts = url_path.split("/")
                if parsed.fragment:
                    path_parts.extend(parts)
                    if extension is not None:
                        file_name = parsed.fragment + "." + extension
                    else:
                        file_name = parsed.fragment
                    path_parts.append(file_name)
                else:
                    if len(parts) > 1:
                        path_parts.extend(parts[:-1])
                    last_part = parts[-1]
                    if extension is not None:
                        file_name = last_part + "." + extension
                    else:
                        file_name = last_part
                    path_parts.append(file_name)
            else:
                if parsed.fragment:
                    if extension is not None:
                        file_name = parsed.fragment + "." + extension
                    else:
                        file_name = parsed.fragment
                    path_parts.append(file_name)
                else:
                    if extension is not None:
                        path_parts.append("index." + extension)
                    else:
                        path_parts.append("index")
            url_paths.append(Path(os.path.join(*path_parts)))
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

        Keyword arguments are passed to `vectorbtpro.utils.knowledge.custom_asset_funcs.ToMarkdownAssetFunc`.

        Last keyword arguments in `kwargs` are forwarded down to
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToMarkdownAssetFunc.to_markdown`."""
        import tempfile
        from vectorbtpro.utils.knowledge.custom_asset_funcs import ToMarkdownAssetFunc

        cache = self.resolve_setting(cache, "cache")
        cache_dir_none = cache_dir is None
        cache_dir = self.resolve_setting(cache_dir, "cache_dir")
        cache_mkdir_kwargs = self.resolve_setting(cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True)
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)

        if cache:
            markdown_dir = Path(cache_dir)
            if cache_dir_none:
                markdown_dir /= "markdown"
            if markdown_dir.exists():
                if clear_cache:
                    remove_dir(markdown_dir, missing_ok=True, with_contents=True)
            check_mkdir(markdown_dir, **cache_mkdir_kwargs)
        else:
            markdown_dir = Path(tempfile.mkdtemp(prefix=get_caller_qualname() + "_"))
        link_map = {d["link"]: dict(d) for d in self.data}
        url_paths = self.links_to_paths(link_map.keys(), extension="md")
        url_file_map = dict(zip(link_map.keys(), [markdown_dir / p for p in url_paths]))
        _, kwargs = ToMarkdownAssetFunc.prepare(**kwargs)

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
                    markdown_content = ToMarkdownAssetFunc.call(d, **kwargs)
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
        to_markdown_kwargs: tp.KwargsLike = None,
        format_html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeVBTAsset:
        """Convert to HTML.

        Uses `VBTAsset.apply` on `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc`.

        Arguments in `format_html_kwargs` are passed to
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc.format_html`.

        Last keyword arguments in `kwargs` are forwarded down to
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc.to_html`.

        For other arguments, see `VBTAsset.to_markdown`."""
        return self.apply(
            "to_html",
            root_metadata_key=root_metadata_key,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            to_markdown_kwargs=to_markdown_kwargs,
            format_html_kwargs=format_html_kwargs,
            **kwargs,
        )

    @classmethod
    def get_top_parent_links(cls, data: tp.List[tp.Any]) -> tp.List[str]:
        """Get links of top parents in data."""
        link_map = {d["link"]: dict(d) for d in data}
        top_parents = []
        for d in data:
            if d.get("parent", None) is None or d["parent"] not in link_map:
                top_parents.append(d["link"])
        return top_parents

    @property
    def top_parent_links(self) -> tp.List[str]:
        """Get links of top parents."""
        return self.get_top_parent_links(self.data)

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
    ) -> tp.Union[Path, tp.Tuple[Path, dict]]:
        """Save to HTML files.

        Opens the web browser. Also, returns the path of the directory where HTML files are stored,
        and if `return_url_map` is True also returns the link->file map.

        In addition, if there are multiple top-level parents, creates an index page.

        If `cache` is True, uses the cache directory. Otherwise, creates a temporary directory.
        If `clear_cache` is True, deletes any existing directory before creating a new one.

        Keyword arguments are passed to `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc`."""
        import tempfile
        from vectorbtpro.utils.knowledge.custom_asset_funcs import ToHTMLAssetFunc

        cache = self.resolve_setting(cache, "cache")
        cache_dir_none = cache_dir is None
        cache_dir = self.resolve_setting(cache_dir, "cache_dir")
        cache_mkdir_kwargs = self.resolve_setting(cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True)
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)

        if cache:
            html_dir = Path(cache_dir)
            if cache_dir_none:
                html_dir /= "html"
            if html_dir.exists():
                if clear_cache:
                    remove_dir(html_dir, missing_ok=True, with_contents=True)
            check_mkdir(html_dir, **cache_mkdir_kwargs)
        else:
            html_dir = Path(tempfile.mkdtemp(prefix=get_caller_qualname() + "_"))
        link_map = {d["link"]: dict(d) for d in self.data}
        top_parents = self.top_parent_links
        if len(top_parents) > 1:
            link_map["/"] = {}
        url_paths = self.links_to_paths(link_map.keys(), extension="html")
        url_file_map = dict(zip(link_map.keys(), [html_dir / p for p in url_paths]))
        url_map = {k: "file://" + str(v.resolve()) for k, v in url_file_map.items()}
        _, kwargs = ToHTMLAssetFunc.prepare(**kwargs)

        if len(top_parents) > 1:
            entry_link = "/"
            if not url_file_map[entry_link].exists():
                html = ToHTMLAssetFunc.call([link_map[link] for link in top_parents], **kwargs)
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
                    html = ToHTMLAssetFunc.call(d, **kwargs)
                    html = self.replace_urls_in_html(html, url_map)
                    check_mkdir(url_file_map[d["link"]].parent, mkdir=True)
                    with open(url_file_map[d["link"]], "w", encoding="utf-8") as f:
                        f.write(html)
                pbar.update()

        if return_url_map:
            return html_dir, url_map
        return html_dir

    def browse(
        self,
        entry_link: tp.Optional[str] = None,
        find_kwargs: tp.KwargsLike = None,
        open_browser: tp.Optional[bool] = None,
        **kwargs,
    ) -> Path:
        """Browse one or more HTML pages.

        Opens the web browser. Also, returns the path of the directory where HTML files are stored.

        Use `entry_link` to specify the link of the page that should be displayed first.
        If `entry_link` is None and there are multiple top-level parents, displays them as an index.
        If it's not None, it will be matched using `VBTAsset.find_link` and `find_kwargs`.

        Keyword arguments are passed to `PagesAsset.save_to_html`."""
        open_browser = self.resolve_setting(open_browser, "open_browser")

        if entry_link is None:
            if len(self.data) == 1:
                entry_link = self.data[0]["link"]
            else:
                top_parents = self.top_parent_links
                if len(top_parents) == 1:
                    entry_link = top_parents[0]
                else:
                    entry_link = "/"
        else:
            if find_kwargs is None:
                find_kwargs = {}
            d = self.find_link(entry_link, wrap=False, **find_kwargs)
            entry_link = d["link"]
        html_dir, url_map = self.save_to_html(return_url_map=True, **kwargs)
        if open_browser:
            import webbrowser

            webbrowser.open(url_map[entry_link])
        return html_dir

    def display(
        self,
        link: tp.Optional[str] = None,
        find_kwargs: tp.KwargsLike = None,
        open_browser: tp.Optional[bool] = None,
        **kwargs,
    ) -> Path:
        """Display as an HTML page.

        Opens the web browser. Also, returns the path of the temporary HTML file."""
        import tempfile

        open_browser = self.resolve_setting(open_browser, "open_browser")

        if link is not None:
            if find_kwargs is None:
                find_kwargs = {}
            single_instance = self.find_link(link, **find_kwargs)
        else:
            if len(self.data) != 1:
                raise ValueError("Must provide link")
            single_instance = self
        html = single_instance.to_html(wrap=False, single_item=True, **kwargs)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            prefix=get_caller_qualname() + "_",
            suffix=".html",
            delete=False,
        ) as f:
            f.write(html)
            file_path = Path(f.name)
        if open_browser:
            import webbrowser

            webbrowser.open("file://" + str(file_path.resolve()))
        return file_path

    @classmethod
    def prepare_mention_target(
        cls,
        target: str,
        as_code: bool = False,
        as_regex: bool = True,
        allow_prefix: bool = False,
        allow_suffix: bool = False,
    ) -> str:
        """Prepare a mention target."""
        if as_regex:
            escaped_target = re.escape(target)
            new_target = ""
            if not allow_prefix and re.match(r"\w", target[0]):
                new_target += r"(?<!\w)"
            new_target += escaped_target
            if not allow_suffix and re.match(r"\w", target[-1]):
                new_target += r"(?!\w)"
            elif not as_code and target[-1] == ".":
                new_target += r"(?=\w)"
            return new_target
        return target

    @classmethod
    def split_class_name(cls, name: str) -> tp.List[str]:
        """Split a class name constituent parts."""
        return re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z][a-z]+", name)

    @classmethod
    def get_class_abbrs(cls, name: str) -> tp.List[str]:
        """Convert a class name to snake case and its abbreviated versions."""
        from itertools import product

        parts = cls.split_class_name(name)

        replacement_lists = []
        for i, part in enumerate(parts):
            replacements = [part.lower()]
            if i == 0 and f"{part}_" in class_abbr_config:
                replacements.extend(class_abbr_config[f"{part}_"])
            if part in class_abbr_config:
                replacements.extend(class_abbr_config[part])
            replacement_lists.append(replacements)
        all_combinations = list(product(*replacement_lists))
        snake_case_names = ["_".join(combo) for combo in all_combinations]

        return snake_case_names

    @classmethod
    def generate_refname_targets(
        cls,
        refname: str,
        resolve: bool = True,
        incl_shortcuts: tp.Optional[bool] = None,
        incl_shortcut_access: tp.Optional[bool] = None,
        incl_shortcut_call: tp.Optional[bool] = None,
        incl_instances: tp.Optional[bool] = None,
        as_code: tp.Optional[bool] = None,
        as_regex: tp.Optional[bool] = None,
        allow_prefix: tp.Optional[bool] = None,
        allow_suffix: tp.Optional[bool] = None,
    ) -> tp.List[str]:
        """Generate reference name targets.

        If `incl_shortcuts` is True, includes shortcuts found in `import vectorbtpro as vbt`.
        In addition, if `incl_shortcut_access` is True and the object is a class or module, includes a version
        with attribute access, and if `incl_shortcut_call` is True and the object is callable, includes a version
        that is being called.

        If `incl_instances` is True, includes typical short names of classes, which
        include the snake-cased class name and mapped name parts found in `class_abbr_config`.

        Prepares each mention target with `VBTAsset.prepare_mention_target`."""
        from vectorbtpro.utils.module_ import annotate_refname_parts
        import vectorbtpro as vbt

        incl_shortcuts = cls.resolve_setting(incl_shortcuts, "incl_shortcuts")
        incl_shortcut_access = cls.resolve_setting(incl_shortcut_access, "incl_shortcut_access")
        incl_shortcut_call = cls.resolve_setting(incl_shortcut_call, "incl_shortcut_call")
        incl_instances = cls.resolve_setting(incl_instances, "incl_instances")
        as_code = cls.resolve_setting(as_code, "as_code")
        as_regex = cls.resolve_setting(as_regex, "as_regex")
        allow_prefix = cls.resolve_setting(allow_prefix, "allow_prefix")
        allow_suffix = cls.resolve_setting(allow_suffix, "allow_suffix")

        def _prepare_target(
            target,
            _as_code=as_code,
            _as_regex=as_regex,
            _allow_prefix=allow_prefix,
            _allow_suffix=allow_suffix,
        ):
            return cls.prepare_mention_target(
                target,
                as_code=_as_code,
                as_regex=_as_regex,
                allow_prefix=_allow_prefix,
                allow_suffix=_allow_suffix,
            )

        targets = set()
        new_target = _prepare_target(refname)
        targets.add(new_target)
        refname_parts = refname.split(".")
        if resolve:
            annotated_parts = annotate_refname_parts(refname)
            if len(annotated_parts) >= 2 and isinstance(annotated_parts[-2]["obj"], type):
                cls_refname = ".".join(refname_parts[:-1])
                cls_aliases = {annotated_parts[-2]["name"]}
                attr_aliases = set()
                for k, v in vbt.__dict__.items():
                    v_refname = prepare_refname(v, raise_error=False)
                    if v_refname is not None:
                        if v_refname == cls_refname:
                            cls_aliases.add(k)
                        elif v_refname == refname:
                            attr_aliases.add(k)
                            if incl_shortcuts:
                                new_target = _prepare_target("vbt." + k)
                                targets.add(new_target)
                if incl_shortcuts:
                    for cls_alias in cls_aliases:
                        new_target = _prepare_target(cls_alias + "." + annotated_parts[-1]["name"])
                        targets.add(new_target)
                    for attr_alias in attr_aliases:
                        if incl_shortcut_call and callable(annotated_parts[-1]["obj"]):
                            new_target = _prepare_target(attr_alias + "(")
                            targets.add(new_target)
                if incl_instances:
                    for cls_alias in cls_aliases:
                        for class_abbr in cls.get_class_abbrs(cls_alias):
                            new_target = _prepare_target(class_abbr + "." + annotated_parts[-1]["name"])
                            targets.add(new_target)
            else:
                if len(refname_parts) >= 2:
                    module_name = ".".join(refname_parts[:-1])
                    attr_name = refname_parts[-1]
                    new_target = _prepare_target("from {} import {}".format(module_name, attr_name))
                    targets.add(new_target)
                aliases = {annotated_parts[-1]["name"]}
                for k, v in vbt.__dict__.items():
                    v_refname = prepare_refname(v, raise_error=False)
                    if v_refname is not None:
                        if v_refname == refname:
                            aliases.add(k)
                            if incl_shortcuts:
                                new_target = _prepare_target("vbt." + k)
                                targets.add(new_target)
                if incl_shortcuts:
                    for alias in aliases:
                        if incl_shortcut_access and isinstance(annotated_parts[-1]["obj"], (type, ModuleType)):
                            new_target = _prepare_target(alias + ".")
                            targets.add(new_target)
                        if incl_shortcut_call and callable(annotated_parts[-1]["obj"]):
                            new_target = _prepare_target(alias + "(")
                            targets.add(new_target)
                if incl_instances and isinstance(annotated_parts[-1]["obj"], type):
                    for alias in aliases:
                        for class_abbr in cls.get_class_abbrs(alias):
                            new_target = _prepare_target(class_abbr + " =")
                            targets.add(new_target)
                            new_target = _prepare_target(class_abbr + ".")
                            targets.add(new_target)
        return sorted(targets)

    def generate_mention_targets(
        self,
        obj: tp.MaybeList,
        *,
        attr: tp.Optional[str] = None,
        module: tp.Union[None, str, ModuleType] = None,
        resolve: bool = True,
        incl_base_attr: tp.Optional[bool] = None,
        incl_shortcuts: tp.Optional[bool] = None,
        incl_shortcut_access: tp.Optional[bool] = None,
        incl_shortcut_call: tp.Optional[bool] = None,
        incl_instances: tp.Optional[bool] = None,
        as_code: tp.Optional[bool] = None,
        as_regex: tp.Optional[bool] = None,
        allow_prefix: tp.Optional[bool] = None,
        allow_suffix: tp.Optional[bool] = None,
    ) -> tp.List[str]:
        """Generate mention targets.

        Prepares the object reference with `vectorbtpro.utils.module_.prepare_refname`.
        If an attribute is provided, checks whether the attribute is defined by the object itself
        or by one of its base classes. If the latter and `incl_base_attr` is True, generates
        reference name targets for both the object attribute and the base class attribute.

        Generates reference name targets with `VBTAsset.generate_refname_targets`."""
        from vectorbtpro.utils.module_ import prepare_refname

        incl_base_attr = self.resolve_setting(incl_base_attr, "incl_base_attr")

        targets = []
        if not isinstance(obj, list):
            objs = [obj]
        else:
            objs = obj
        for obj in objs:
            obj_refname = prepare_refname(obj, module=module, resolve=resolve)
            if attr is not None:
                checks.assert_instance_of(attr, str, arg_name="attr")
                if isinstance(obj, tuple):
                    attr_obj = (*obj, attr)
                else:
                    attr_obj = (obj, attr)
                base_attr_refname = prepare_refname(attr_obj, module=module, resolve=resolve)
                obj_refname += "." + attr
                if base_attr_refname == obj_refname:
                    obj_refname = base_attr_refname
                    base_attr_refname = None
            else:
                base_attr_refname = None
            targets.extend(
                self.generate_refname_targets(
                    obj_refname,
                    resolve=resolve,
                    incl_shortcuts=incl_shortcuts,
                    incl_shortcut_access=incl_shortcut_access,
                    incl_shortcut_call=incl_shortcut_call,
                    incl_instances=incl_instances,
                    as_code=as_code,
                    as_regex=as_regex,
                    allow_prefix=allow_prefix,
                    allow_suffix=allow_suffix,
                )
            )
            if incl_base_attr and base_attr_refname is not None:
                targets.extend(
                    self.generate_refname_targets(
                        base_attr_refname,
                        resolve=resolve,
                        incl_shortcuts=incl_shortcuts,
                        incl_shortcut_access=incl_shortcut_access,
                        incl_shortcut_call=incl_shortcut_call,
                        incl_instances=incl_instances,
                        as_code=as_code,
                        as_regex=as_regex,
                        allow_prefix=allow_prefix,
                        allow_suffix=allow_suffix,
                    )
                )
        seen = set()
        targets = [x for x in targets if not (x in seen or seen.add(x))]
        return targets

    @classmethod
    def merge_mention_targets(cls, targets: tp.List[str], as_regex: bool = True) -> str:
        """Merge mention targets into a single regular expression."""
        if as_regex:
            prefixed_targets = []
            non_prefixed_targets = []
            common_prefix = r"(?<!\w)"
            for target in targets:
                if target.startswith(common_prefix):
                    prefixed_targets.append(target[len(common_prefix) :])
                else:
                    non_prefixed_targets.append(target)
            combined_targets = []
            if prefixed_targets:
                combined_prefixed = "|".join(f"(?:{p})" for p in prefixed_targets)
                combined_targets.append(f"{common_prefix}(?:{combined_prefixed})")
            if non_prefixed_targets:
                combined_non_prefixed = "|".join(f"(?:{p})" for p in non_prefixed_targets)
                combined_targets.append(f"(?:{combined_non_prefixed})")
            if len(combined_targets) == 1:
                return combined_targets[0]
        else:
            combined_targets = [re.escape(target) for target in targets]
        combined_target = "|".join(combined_targets)
        return f"(?:{combined_target})"

    def find_obj_mentions(
        self,
        obj: tp.MaybeList,
        *,
        attr: tp.Optional[str] = None,
        module: tp.Union[None, str, ModuleType] = None,
        resolve: bool = True,
        incl_shortcuts: tp.Optional[bool] = None,
        incl_shortcut_access: tp.Optional[bool] = None,
        incl_shortcut_call: tp.Optional[bool] = None,
        incl_instances: tp.Optional[bool] = None,
        incl_custom: tp.Optional[tp.MaybeList[str]] = None,
        is_custom_regex: bool = False,
        as_code: tp.Optional[bool] = None,
        as_regex: tp.Optional[bool] = None,
        allow_prefix: tp.Optional[bool] = None,
        allow_suffix: tp.Optional[bool] = None,
        merge_targets: tp.Optional[bool] = None,
        per_path: bool = False,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = "content",
        return_type: tp.Optional[str] = "item",
        **kwargs,
    ) -> tp.MaybeVBTAsset:
        """Find mentions of a VBT object.

        Generates mention targets with `VBTAsset.generate_mention_targets`.

        Provide custom mentions in `incl_custom`. If regular expressions are provided,
        set `is_custom_regex` to True.

        If `as_code` is True, uses `VBTAsset.find_code`, otherwise, uses `VBTAsset.find`.

        If `as_regex` is True, search is refined by using regular expressions. For instance,
        `vbt.PF` may match `vbt.PFO` if RegEx is not used.

        If `merge_targets`, uses `VBTAsset.merge_mention_targets` to reduce the number of targets.
        Sets `as_regex` to True if False (but after the targets were generated)."""
        as_code = self.resolve_setting(as_code, "as_code")
        as_regex = self.resolve_setting(as_regex, "as_regex")
        allow_prefix = self.resolve_setting(allow_prefix, "allow_prefix")
        allow_suffix = self.resolve_setting(allow_suffix, "allow_suffix")
        merge_targets = self.resolve_setting(merge_targets, "merge_targets")

        mention_targets = self.generate_mention_targets(
            obj,
            attr=attr,
            module=module,
            resolve=resolve,
            incl_shortcuts=incl_shortcuts,
            incl_shortcut_access=incl_shortcut_access,
            incl_shortcut_call=incl_shortcut_call,
            incl_instances=incl_instances,
            as_code=as_code,
            as_regex=as_regex,
            allow_prefix=allow_prefix,
            allow_suffix=allow_suffix,
        )
        if incl_custom:

            def _prepare_target(
                target,
                _as_code=as_code,
                _as_regex=as_regex,
                _allow_prefix=allow_prefix,
                _allow_suffix=allow_suffix,
            ):
                return self.prepare_mention_target(
                    target,
                    as_code=_as_code,
                    as_regex=_as_regex,
                    allow_prefix=_allow_prefix,
                    allow_suffix=_allow_suffix,
                )

            if isinstance(incl_custom, str):
                incl_custom = [incl_custom]
            for custom in incl_custom:
                new_target = _prepare_target(custom, _as_regex=is_custom_regex)
                if new_target not in mention_targets:
                    mention_targets.append(new_target)
        if merge_targets:
            mention_targets = self.merge_mention_targets(mention_targets, as_regex=as_regex)
            as_regex = True
        if as_code:
            mentions_asset = self.find_code(
                mention_targets,
                escape_target=not as_regex,
                path=path,
                per_path=per_path,
                return_type=return_type,
                **kwargs,
            )
        elif as_regex:
            mentions_asset = self.find(
                mention_targets,
                mode="regex",
                path=path,
                per_path=per_path,
                return_type=return_type,
                **kwargs,
            )
        else:
            mentions_asset = self.find(
                mention_targets,
                path=path,
                per_path=per_path,
                return_type=return_type,
                **kwargs,
            )
        return mentions_asset


PagesAssetT = tp.TypeVar("PagesAssetT", bound="PagesAsset")


class PagesAsset(VBTAsset):
    """Class for working with website pages.

    Has the following fields:

    * link: URL of the page (without fragment), such as "https://vectorbt.pro/features/data/", or
        URL of the heading (with fragment), such as "https://vectorbt.pro/features/data/#trading-view"
    * parent: URL of the parent page or heading. For example, a heading 1 is a parent of a heading 2.
    * children: List of URLs of the child pages and/or headings. For example, a heading 2 is a child of a heading 1.
    * name: Name of the page or heading. Within the API, the name of the object that the heading represents,
        such as "Portfolio.from_signals".
    * type: Type of the page or heading, such as "page", "heading 1", "heading 2", etc.
    * icon: Icon, such as "material-brain"
    * tags: List of tags, such as ["portfolio", "records"]
    * content: String content of the page or heading. Can be None in pages that solely redirect.
    * obj_type: Within the API, the type of the object that the heading represents, such as "property"
    * github_link: Within the API, the URL to the source code of the object that the heading represents

    For defaults, see `assets.pages` in `vectorbtpro._settings.knowledge`."""

    _settings_path: tp.SettingsPath = "knowledge.assets.pages"

    def minimize(self: PagesAssetT, minimize_links: tp.Optional[bool] = None) -> PagesAssetT:
        new_instance = VBTAsset.minimize(self, minimize_links=minimize_links)
        new_instance = new_instance.remove(
            [
                "parent",
                "children",
                "type",
                "icon",
                "tags",
            ],
            skip_missing=True,
        )
        return new_instance

    def descend_links(self: PagesAssetT, links: tp.List[str]) -> PagesAssetT:
        """Descend links by removing redundant ones.

        Only headings are descended."""
        redundant_links = set()
        new_data = {}
        for link in links:
            if link in redundant_links:
                continue
            descendant_headings = self.select_descendant_headings(link, incl_link=True)
            for d in descendant_headings:
                if d["link"] != link:
                    redundant_links.add(d["link"])
                new_data[d["link"]] = d
        for link in links:
            if link in redundant_links and link in new_data:
                del new_data[link]
        return self.replace(data=list(new_data.values()))

    def aggregate_links(self: PagesAssetT, links: tp.List[str], aggregate_kwargs: tp.KwargsLike = None) -> PagesAssetT:
        """Aggregate links by removing redundant ones.

        Only headings are aggregated."""
        if aggregate_kwargs is None:
            aggregate_kwargs = {}
        redundant_links = set()
        new_data = {}
        for link in links:
            if link in redundant_links:
                continue
            descendant_headings = self.select_descendant_headings(link, incl_link=True)
            for d in descendant_headings:
                if d["link"] != link:
                    redundant_links.add(d["link"])
            descendant_headings = descendant_headings.aggregate(**aggregate_kwargs)
            new_data[link] = descendant_headings[0]
        for link in links:
            if link in redundant_links and link in new_data:
                del new_data[link]
        return self.replace(data=list(new_data.values()))

    def find_page(
        self: PagesAssetT,
        link: tp.MaybeList[str],
        aggregate: bool = False,
        aggregate_kwargs: tp.KwargsLike = None,
        incl_descendants: bool = False,
        **kwargs,
    ) -> tp.MaybePagesAsset:
        """Find the page(s) corresponding to link(s).

        Keyword arguments are passed to `VBTAsset.find_link`."""
        found = self.find_link(link, **kwargs)
        if not isinstance(found, (type(self), list)):
            return found
        if aggregate:
            return self.aggregate_links([d["link"] for d in found], aggregate_kwargs=aggregate_kwargs)
        if incl_descendants:
            return self.descend_links([d["link"] for d in found])
        return found

    def find_refname(
        self,
        refname: tp.MaybeList[str],
        **kwargs,
    ) -> tp.MaybePagesAsset:
        """Find the page corresponding to a reference."""
        if isinstance(refname, list):
            link = list(map(lambda x: f"#({re.escape(x)})$", refname))
        else:
            link = f"#({re.escape(refname)})$"
        return self.find_page(link, mode="regex", **kwargs)

    def find_obj(
        self,
        obj: tp.Any,
        *,
        attr: tp.Optional[str] = None,
        module: tp.Union[None, str, ModuleType] = None,
        resolve: bool = True,
        **kwargs,
    ) -> tp.MaybePagesAsset:
        """Find the page corresponding a single (internal) object or reference name.

        Prepares the reference with `vectorbtpro.utils.module_.prepare_refname`."""
        if attr is not None:
            checks.assert_instance_of(attr, str, arg_name="attr")
            if isinstance(obj, tuple):
                obj = (*obj, attr)
            else:
                obj = (obj, attr)
        refname = prepare_refname(obj, module=module, resolve=resolve)
        return self.find_refname(refname, **kwargs)

    @classmethod
    def parse_content_links(cls, content: str) -> tp.List[str]:
        """Parse all links from a content."""
        link_pattern = r'(?<!\!)\[[^\]]+\]\((\S+?)(?:\s+(?:"[^"]*"|\'[^\']*\'))?\)'
        return re.findall(link_pattern, content)

    @classmethod
    def parse_link_refname(cls, link: str) -> tp.Optional[str]:
        """Parse the reference name from a link."""
        if "/api/" not in link:
            return None
        if "#" in link:
            refname = link.split("#")[1]
            if refname.startswith("vectorbtpro"):
                return refname
            return None
        return "vectorbtpro." + ".".join(link.split("/api/")[1].strip("/").split("/"))

    @classmethod
    def is_link_module(cls, link: str) -> bool:
        """Return whether a link is a module."""
        if "/api/" not in link:
            return False
        if "#" not in link:
            return True
        refname = link.split("#")[1]
        if "/".join(refname.split(".")) in link:
            return True
        return False

    def find_obj_api(
        self,
        obj: tp.MaybeList,
        *,
        attr: tp.Optional[str] = None,
        module: tp.Union[None, str, ModuleType] = None,
        resolve: bool = True,
        use_parent: tp.Optional[bool] = None,
        use_base_parents: tp.Optional[bool] = None,
        use_ref_parents: tp.Optional[bool] = None,
        incl_bases: tp.Union[None, bool, int] = None,
        incl_ancestors: tp.Union[None, bool, int] = None,
        incl_base_ancestors: tp.Union[None, bool, int] = None,
        incl_refs: tp.Union[None, bool, int] = None,
        incl_descendants: tp.Optional[bool] = None,
        incl_ancestor_descendants: tp.Optional[bool] = None,
        incl_ref_descendants: tp.Optional[bool] = None,
        aggregate: tp.Optional[bool] = None,
        aggregate_ancestors: tp.Optional[bool] = None,
        aggregate_refs: tp.Optional[bool] = None,
        aggregate_kwargs: tp.KwargsLike = None,
        topo_sort: tp.Optional[bool] = None,
        return_refname_graph: bool = False,
    ) -> tp.Union[PagesAssetT, tp.Tuple[PagesAssetT, dict]]:
        """Find API pages and headings relevant to object(s).

        Prepares the object reference with `vectorbtpro.utils.module_.prepare_refname`.

        If `incl_bases` is True, extends the asset with the base classes/attributes if the object is
        a class/attribute. For instance, `vectorbtpro.portfolio.base.Portfolio` has
        `vectorbtpro.generic.analyzable.Analyzable` as one of its base classes. It can also be an
        integer indicating the maximum inheritance level. If `obj` is a module, then bases are sub-modules.

        If `incl_ancestors` is True, extends the asset with the ancestors of the object.
        For instance, `vectorbtpro.portfolio.base.Portfolio` has `vectorbtpro.portfolio.base` as its ancestor.
        It can also be an integer indicating the maximum inheritance level. Provide `incl_base_ancestors`
        to override `incl_ancestors` for base classes/attributes.

        If `incl_refs` is True, extends the asset with the references found in the content of the object.
        It can also be an integer indicating the maximum reference level. Defaults to False for modules
        and classes, and True otherwise. If resolution of reference names is disabled, defaults to False.

        If `incl_descendants` is True, extends the asset page or heading with any descendant headings.
        Provide `incl_ancestor_descendants` and `incl_ref_descendants` to override `incl_descendants`
        for ancestors and references respectively.

        If `aggregate` is True, aggregates any descendant headings into pages for this object
        and all base classes/attributes. Provide `aggregate_ancestors` and `aggregate_refs` to
        override `aggregate` for ancestors and references respectively.

        If `topo_sort` is True, creates a topological graph from all reference names and sorts pages
        and headings based on this graph. Use `return_refname_graph` to True to also return the graph."""
        from vectorbtpro.utils.module_ import prepare_refname, annotate_refname_parts

        incl_bases = self.resolve_setting(incl_bases, "incl_bases")
        incl_ancestors = self.resolve_setting(incl_ancestors, "incl_ancestors")
        incl_base_ancestors = self.resolve_setting(incl_base_ancestors, "incl_base_ancestors")
        incl_refs = self.resolve_setting(incl_refs, "incl_refs")
        incl_descendants = self.resolve_setting(incl_descendants, "incl_descendants")
        incl_ancestor_descendants = self.resolve_setting(incl_ancestor_descendants, "incl_ancestor_descendants")
        incl_ref_descendants = self.resolve_setting(incl_ref_descendants, "incl_ref_descendants")
        aggregate = self.resolve_setting(aggregate, "aggregate")
        aggregate_ancestors = self.resolve_setting(aggregate_ancestors, "aggregate_ancestors")
        aggregate_refs = self.resolve_setting(aggregate_refs, "aggregate_refs")
        topo_sort = self.resolve_setting(topo_sort, "topo_sort")

        base_refnames = []
        base_refnames_set = set()
        if not isinstance(obj, list):
            objs = [obj]
        else:
            objs = obj
        for obj in objs:
            if attr is not None:
                checks.assert_instance_of(attr, str, arg_name="attr")
                if isinstance(obj, tuple):
                    obj = (*obj, attr)
                else:
                    obj = (obj, attr)
            obj_refname = prepare_refname(obj, module=module, resolve=resolve)
            refname_graph = defaultdict(list)
            if resolve:
                annotated_parts = annotate_refname_parts(obj_refname)
                if isinstance(annotated_parts[-1]["obj"], ModuleType):
                    _module = annotated_parts[-1]["obj"]
                    _cls = None
                    _attr = None
                elif isinstance(annotated_parts[-1]["obj"], type):
                    _module = None
                    _cls = annotated_parts[-1]["obj"]
                    _attr = None
                elif len(annotated_parts) >= 2 and isinstance(annotated_parts[-2]["obj"], type):
                    _module = None
                    _cls = annotated_parts[-2]["obj"]
                    _attr = annotated_parts[-1]["name"]
                else:
                    _module = None
                    _cls = None
                    _attr = None
                if use_parent is None:
                    use_parent = _cls is not None and _attr is None
                if not aggregate and not incl_descendants:
                    use_parent = False
                    use_base_parents = False
                if incl_refs is None:
                    incl_refs = _module is None and _cls is None
                if _cls is not None and incl_bases:
                    level_classes = defaultdict(set)
                    visited = set()
                    queue = deque([(_cls, 0)])
                    while queue:
                        current_cls, current_level = queue.popleft()
                        if current_cls in visited:
                            continue
                        visited.add(current_cls)
                        level_classes[current_level].add(current_cls)
                        for base in current_cls.__bases__:
                            queue.append((base, current_level + 1))
                    mro = inspect.getmro(_cls)
                    classes = []
                    levels = list(level_classes.keys())
                    if not isinstance(incl_bases, bool):
                        if isinstance(incl_bases, int):
                            levels = levels[: incl_bases + 1]
                        else:
                            raise TypeError(f"Invalid incl_bases: {incl_bases}")
                    for level in levels:
                        classes.extend([_cls for _cls in mro if _cls in level_classes[level]])
                    for c in classes:
                        if c.__module__.split(".")[0] != "vectorbtpro":
                            continue
                        if _attr is not None:
                            if not hasattr(c, _attr):
                                continue
                            refname = prepare_refname((c, _attr))
                        else:
                            refname = prepare_refname(c)
                        if (use_parent and refname == obj_refname) or use_base_parents:
                            refname = ".".join(refname.split(".")[:-1])
                        if refname not in base_refnames_set:
                            base_refnames.append(refname)
                            base_refnames_set.add(refname)
                            for b in c.__bases__:
                                if b.__module__.split(".")[0] == "vectorbtpro":
                                    if _attr is not None:
                                        if not hasattr(b, _attr):
                                            continue
                                        b_refname = prepare_refname((b, _attr))
                                    else:
                                        b_refname = prepare_refname(b)
                                    if use_base_parents:
                                        b_refname = ".".join(b_refname.split(".")[:-1])
                                    if refname != b_refname:
                                        refname_graph[refname].append(b_refname)
                elif _module is not None and hasattr(_module, "__path__") and incl_bases:
                    base_refnames.append(_module.__name__)
                    base_refnames_set.add(_module.__name__)
                    refname_level = {}
                    refname_level[_module.__name__] = 0
                    for _, refname, _ in pkgutil.walk_packages(_module.__path__, prefix=f"{_module.__name__}."):
                        if refname not in base_refnames_set:
                            parent_refname = ".".join(refname.split(".")[:-1])
                            if not isinstance(incl_bases, bool):
                                if isinstance(incl_bases, int):
                                    if refname_level[parent_refname] + 1 > incl_bases:
                                        continue
                                else:
                                    raise TypeError(f"Invalid incl_bases: {incl_bases}")
                            base_refnames.append(refname)
                            base_refnames_set.add(refname)
                            refname_level[refname] = refname_level[parent_refname] + 1
                            if parent_refname != refname:
                                refname_graph[parent_refname].append(refname)
                else:
                    base_refnames.append(obj_refname)
                    base_refnames_set.add(obj_refname)
            else:
                if incl_refs is None:
                    incl_refs = False
                base_refnames.append(obj_refname)
                base_refnames_set.add(obj_refname)
        api_asset = self.find_refname(
            base_refnames,
            single_item=False,
            incl_descendants=incl_descendants,
            aggregate=aggregate,
            aggregate_kwargs=aggregate_kwargs,
            allow_empty=True,
            wrap=True,
        )
        if len(api_asset) == 0:
            return api_asset
        if not topo_sort:
            refname_indices = {refname: [] for refname in base_refnames}
            remaining_indices = []
            for i, d in enumerate(api_asset):
                refname = self.parse_link_refname(d["link"])
                if refname is not None:
                    while refname not in refname_indices:
                        if not refname:
                            break
                        refname = ".".join(refname.split(".")[:-1])
                if refname:
                    refname_indices[refname].append(i)
                else:
                    remaining_indices.append(i)
            get_indices = [i for v in refname_indices.values() for i in v] + remaining_indices
            api_asset = api_asset.get_items(get_indices)

        if incl_ancestors or incl_refs:
            refnames_aggregated = {}
            for d in api_asset:
                refname = self.parse_link_refname(d["link"])
                if refname is not None:
                    refnames_aggregated[refname] = aggregate
            to_ref_api_asset = api_asset
            if incl_ancestors:
                anc_refnames = []
                anc_refnames_set = set(refnames_aggregated.keys())
                for d in api_asset:
                    child_refname = refname = self.parse_link_refname(d["link"])
                    if refname is not None:
                        if incl_base_ancestors or refname == obj_refname:
                            refname = ".".join(refname.split(".")[:-1])
                            anc_level = 1
                            while refname:
                                if isinstance(incl_base_ancestors, bool) or refname == obj_refname:
                                    if not isinstance(incl_ancestors, bool):
                                        if isinstance(incl_ancestors, int):
                                            if anc_level > incl_ancestors:
                                                break
                                        else:
                                            raise TypeError(f"Invalid incl_ancestors: {incl_ancestors}")
                                else:
                                    if not isinstance(incl_base_ancestors, bool):
                                        if isinstance(incl_base_ancestors, int):
                                            if anc_level > incl_base_ancestors:
                                                break
                                        else:
                                            raise TypeError(f"Invalid incl_base_ancestors: {incl_base_ancestors}")
                                if refname not in anc_refnames_set:
                                    anc_refnames.append(refname)
                                    anc_refnames_set.add(refname)
                                    if refname != child_refname:
                                        refname_graph[refname].append(child_refname)
                                child_refname = refname
                                refname = ".".join(refname.split(".")[:-1])
                                anc_level += 1
                anc_api_asset = self.find_refname(
                    anc_refnames,
                    single_item=False,
                    incl_descendants=incl_ancestor_descendants,
                    aggregate=aggregate_ancestors,
                    aggregate_kwargs=aggregate_kwargs,
                    allow_empty=True,
                    wrap=True,
                )
                if aggregate_ancestors or incl_ancestor_descendants:
                    obj_index = None
                    for i, d in enumerate(api_asset):
                        d_refname = self.parse_link_refname(d["link"])
                        if d_refname == obj_refname:
                            obj_index = i
                            break
                    if obj_index is not None:
                        del api_asset[obj_index]
                for d in anc_api_asset:
                    refname = self.parse_link_refname(d["link"])
                    if refname is not None:
                        refnames_aggregated[refname] = aggregate_ancestors
                api_asset = anc_api_asset + api_asset

            if incl_refs:
                if not aggregate and not incl_descendants:
                    use_ref_parents = False
                main_ref_api_asset = None
                ref_api_asset = to_ref_api_asset
                while incl_refs:
                    content_refnames = []
                    content_refnames_set = set()
                    for d in ref_api_asset:
                        d_refname = self.parse_link_refname(d["link"])
                        if d_refname is not None:
                            for link in self.parse_content_links(d["content"]):
                                if "/api/" in link:
                                    refname = self.parse_link_refname(link)
                                    if refname is not None:
                                        if use_ref_parents and not self.is_link_module(link):
                                            refname = ".".join(refname.split(".")[:-1])
                                        if refname not in content_refnames_set:
                                            content_refnames.append(refname)
                                            content_refnames_set.add(refname)
                                            if d_refname != refname and refname not in refname_graph:
                                                refname_graph[d_refname].append(refname)
                    ref_refnames = []
                    ref_refnames_set = set(refnames_aggregated.keys()) | content_refnames_set
                    for refname in content_refnames:
                        if refname in refnames_aggregated and (refnames_aggregated[refname] or not aggregate_refs):
                            continue
                        _refname = refname
                        while _refname:
                            _refname = ".".join(_refname.split(".")[:-1])
                            if _refname in ref_refnames_set and refnames_aggregated.get(_refname, aggregate_refs):
                                break
                        if not _refname:
                            ref_refnames.append(refname)
                    if len(ref_refnames) == 0:
                        break
                    ref_api_asset = self.find_refname(
                        ref_refnames,
                        single_item=False,
                        incl_descendants=incl_ref_descendants,
                        aggregate=aggregate_refs,
                        aggregate_kwargs=aggregate_kwargs,
                        allow_empty=True,
                        wrap=True,
                    )
                    for d in ref_api_asset:
                        refname = self.parse_link_refname(d["link"])
                        if refname is not None:
                            refnames_aggregated[refname] = aggregate_refs
                    if main_ref_api_asset is None:
                        main_ref_api_asset = ref_api_asset
                    else:
                        main_ref_api_asset += ref_api_asset
                    incl_refs -= 1
                if main_ref_api_asset is not None:
                    api_asset += main_ref_api_asset
                    aggregated_refnames_set = set()
                    for refname, aggregated in refnames_aggregated.items():
                        if aggregated:
                            aggregated_refnames_set.add(refname)
                    delete_indices = []
                    for i, d in enumerate(api_asset):
                        refname = self.parse_link_refname(d["link"])
                        if refname is not None:
                            if not refnames_aggregated[refname] and refname in aggregated_refnames_set:
                                delete_indices.append(i)
                                continue
                            while refname:
                                refname = ".".join(refname.split(".")[:-1])
                                if refname in aggregated_refnames_set:
                                    break
                        if refname:
                            delete_indices.append(i)
                    if len(delete_indices) > 0:
                        api_asset.delete_items(delete_indices, inplace=True)

        if topo_sort:
            from graphlib import TopologicalSorter

            refname_topo_graph = defaultdict(set)
            refname_topo_sorter = TopologicalSorter(refname_topo_graph)
            for parent_node, child_nodes in refname_graph.items():
                for child_node in child_nodes:
                    refname_topo_sorter.add(child_node, parent_node)
            refname_topo_order = refname_topo_sorter.static_order()
            refname_indices = {refname: [] for refname in refname_topo_order}
            remaining_indices = []
            for i, d in enumerate(api_asset):
                refname = self.parse_link_refname(d["link"])
                if refname is not None:
                    while refname not in refname_indices:
                        if not refname:
                            break
                        refname = ".".join(refname.split(".")[:-1])
                    if refname:
                        refname_indices[refname].append(i)
                    else:
                        remaining_indices.append(i)
                else:
                    remaining_indices.append(i)
            get_indices = [i for v in refname_indices.values() for i in v] + remaining_indices
            api_asset = api_asset.get_items(get_indices)

        if return_refname_graph:
            return api_asset, refname_graph
        return api_asset

    def find_obj_docs(
        self,
        obj: tp.MaybeList,
        *,
        attr: tp.Optional[str] = None,
        module: tp.Union[None, str, ModuleType] = None,
        resolve: bool = True,
        incl_pages: tp.Optional[tp.MaybeIterable[str]] = None,
        excl_pages: tp.Optional[tp.MaybeIterable[str]] = None,
        page_find_mode: tp.Optional[str] = None,
        up_aggregate: tp.Optional[bool] = None,
        up_aggregate_th: tp.Union[None, int, float] = None,
        up_aggregate_pages: tp.Optional[bool] = None,
        aggregate: tp.Optional[bool] = None,
        aggregate_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybePagesAsset:
        """Find documentation relevant to object(s).

        If a link matches one of the links or link parts in `incl_pages`, it will be included,
        otherwise, it will be excluded if `incl_pages` is not empty. If a link matches one of the links
        or link parts in `excl_pages`, it will be excluded, otherwise, it will be included. Matching is
        done using `vectorbtpro.utils.search.find` with `page_find_mode` used as `mode`.
        For example, using `excl_pages=["release-notes"]` won't search in release notes.

        If `up_aggregate` is True, will aggregate each set of headings into their parent if their number
        is greater than some threshold `up_aggregate_th`, which depends on the total number of headings
        in the parent. It can be an integer for absolute number or float for relative number.
        For example, `up_aggregate_th=2/3` means this method must find 2 headings out of 3 in order to
        replace it by the full parent heading/page. If `up_aggregate_pages` is True, does the same
        to pages. For example, if 2 tutorial pages out of 3 are matched, the whole tutorial series is used.

        If `aggregate` is True, aggregates any descendant headings into pages for this object
        and all base classes/attributes using `PagesAsset.aggregate_links`.

        Uses `PagesAsset.find_obj_mentions`."""
        incl_pages = self.resolve_setting(incl_pages, "incl_pages")
        excl_pages = self.resolve_setting(excl_pages, "excl_pages")
        page_find_mode = self.resolve_setting(page_find_mode, "page_find_mode")
        up_aggregate = self.resolve_setting(up_aggregate, "up_aggregate")
        up_aggregate_th = self.resolve_setting(up_aggregate_th, "up_aggregate_th")
        up_aggregate_pages = self.resolve_setting(up_aggregate_pages, "up_aggregate_pages")
        aggregate = self.resolve_setting(aggregate, "aggregate")

        if incl_pages is None:
            incl_pages = ()
        elif isinstance(incl_pages, str):
            incl_pages = (incl_pages,)
        if excl_pages is None:
            excl_pages = ()
        elif isinstance(excl_pages, str):
            excl_pages = (excl_pages,)

        def _filter_func(x):
            if "link" not in x:
                return False
            if "/api/" in x["link"]:
                return False
            if excl_pages:
                for page in excl_pages:
                    if find(page, x["link"], mode=page_find_mode):
                        return False
            if incl_pages:
                for page in incl_pages:
                    if find(page, x["link"], mode=page_find_mode):
                        return True
                return False
            return True

        docs_asset = self.filter(_filter_func)
        mentions_asset = docs_asset.find_obj_mentions(
            obj,
            attr=attr,
            module=module,
            resolve=resolve,
            **kwargs,
        )
        if (
            isinstance(mentions_asset, PagesAsset)
            and len(mentions_asset) > 0
            and isinstance(mentions_asset[0], dict)
            and "link" in mentions_asset[0]
        ):
            if up_aggregate:
                link_map = {d["link"]: dict(d) for d in docs_asset.data}
                new_links = {d["link"] for d in mentions_asset}
                while True:
                    parent_map = defaultdict(list)
                    without_parent = set()
                    for link in new_links:
                        if link_map[link]["parent"] is not None:
                            parent_map[link_map[link]["parent"]].append(link)
                        else:
                            without_parent.add(link)
                    _new_links = set()
                    for parent, children in parent_map.items():
                        headings = set()
                        non_headings = set()
                        for child in children:
                            if link_map[child]["type"].startswith("heading"):
                                headings.add(child)
                            else:
                                non_headings.add(child)
                        if up_aggregate_pages:
                            _children = children
                        else:
                            _children = headings
                        if checks.is_float(up_aggregate_th) and 0 <= abs(up_aggregate_th) <= 1:
                            _up_aggregate_th = int(up_aggregate_th * len(link_map[parent]["children"]))
                        elif checks.is_number(up_aggregate_th):
                            if checks.is_float(up_aggregate_th) and not up_aggregate_th.is_integer():
                                raise TypeError(f"Up-aggregation threshold ({up_aggregate_th}) must be between 0 and 1")
                            _up_aggregate_th = int(up_aggregate_th)
                        else:
                            raise TypeError(f"Up-aggregation threshold must be a number")
                        if 0 < len(_children) >= _up_aggregate_th:
                            _new_links.add(parent)
                        else:
                            _new_links |= headings
                        _new_links |= non_headings
                    if _new_links == new_links:
                        break
                    new_links = _new_links | without_parent
                return docs_asset.find_page(
                    list(new_links),
                    single_item=False,
                    aggregate=aggregate,
                    aggregate_kwargs=aggregate_kwargs,
                )
            if aggregate:
                return docs_asset.aggregate_links(
                    [d["link"] for d in mentions_asset],
                    aggregate_kwargs=aggregate_kwargs,
                )
        return mentions_asset

    def browse(
        self,
        entry_link: tp.Optional[str] = None,
        descendants_only: bool = False,
        aggregate: bool = False,
        aggregate_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> Path:
        new_instance = self
        if entry_link is not None and entry_link != "/" and descendants_only:
            new_instance = new_instance.select_descendants(entry_link, incl_link=True)
        if aggregate:
            if aggregate_kwargs is None:
                aggregate_kwargs = {}
            new_instance = new_instance.aggregate(**aggregate_kwargs)
        return VBTAsset.browse(new_instance, entry_link=entry_link, **kwargs)

    def display(
        self,
        link: tp.Optional[str] = None,
        aggregate: bool = False,
        aggregate_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> Path:
        new_instance = self
        if link is not None:
            new_instance = new_instance.find_page(
                link,
                aggregate=aggregate,
                aggregate_kwargs=aggregate_kwargs,
            )
        elif aggregate:
            if aggregate_kwargs is None:
                aggregate_kwargs = {}
            new_instance = new_instance.aggregate(**aggregate_kwargs)
        return VBTAsset.display(new_instance, **kwargs)

    def aggregate(
        self: PagesAssetT,
        append_obj_type: tp.Optional[bool] = None,
        append_github_link: tp.Optional[bool] = None,
    ) -> PagesAssetT:
        """Aggregate pages.

        Content of each heading will be converted into markdown and concatenated into the content
        of the parent heading or page. Only regular pages and headings without parents will be left.

        If `append_obj_type` is True, will also append object type to the heading name.
        If `append_github_link` is True, will also append GitHub link to the heading name."""
        append_obj_type = self.resolve_setting(append_obj_type, "append_obj_type")
        append_github_link = self.resolve_setting(append_github_link, "append_github_link")

        link_map = {d["link"]: dict(d) for d in self.data}
        top_parents = self.top_parent_links
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
                if child in link_map:
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

    def select_parent(self: PagesAssetT, link: str, incl_link: bool = False, **kwargs) -> PagesAssetT:
        """Select the parent page of a link."""
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if incl_link:
            new_data.append(d)
        if d.get("parent", None):
            if d["parent"] in link_map:
                new_data.append(link_map[d["parent"]])
        return self.replace(data=new_data, single_item=True)

    def select_children(self, link: str, incl_link: bool = False, **kwargs) -> PagesAssetT:
        """Select the child pages of a link."""
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if incl_link:
            new_data.append(d)
        if d.get("children", []):
            for child in d["children"]:
                if child in link_map:
                    new_data.append(link_map[child])
        return self.replace(data=new_data, single_item=False)

    def select_siblings(self, link: str, incl_link: bool = False, **kwargs) -> PagesAssetT:
        """Select the sibling pages of a link."""
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if incl_link:
            new_data.append(d)
        if d.get("parent", None):
            if d["parent"] in link_map:
                parent_d = link_map[d["parent"]]
                if parent_d.get("children", []):
                    for child in parent_d["children"]:
                        if incl_link or child != d["link"]:
                            if child in link_map:
                                new_data.append(link_map[child])
        return self.replace(data=new_data, single_item=False)

    def select_descendants(self, link: str, incl_link: bool = False, **kwargs) -> PagesAssetT:
        """Select all descendant pages of a link."""
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if incl_link:
            new_data.append(d)
        descendants = set()
        stack = [d]
        while stack:
            d = stack.pop()
            children = d.get("children", [])
            for child in children:
                if child in link_map and child not in descendants:
                    descendants.add(child)
                    new_data.append(link_map[child])
                    stack.append(link_map[child])
        return self.replace(data=new_data, single_item=False)

    def select_branch(self, link: str, **kwargs) -> PagesAssetT:
        """Select all descendant pages of a link including the link."""
        return self.select_descendants(link, incl_link=True, **kwargs)

    def select_ancestors(self, link: str, incl_link: bool = False, **kwargs) -> PagesAssetT:
        """Select all ancestor pages of a link."""
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if incl_link:
            new_data.append(d)
        ancestors = set()
        parent = d.get("parent", None)
        while parent and parent in link_map:
            if parent in ancestors:
                break
            ancestors.add(parent)
            new_data.append(link_map[parent])
            parent = link_map[parent].get("parent", None)
        return self.replace(data=new_data, single_item=False)

    def select_parent_page(self, link: str, incl_link: bool = False, **kwargs) -> PagesAssetT:
        """Select parent page."""
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if incl_link:
            new_data.append(d)
        ancestors = set()
        parent = d.get("parent", None)
        while parent and parent in link_map:
            if parent in ancestors:
                break
            ancestors.add(parent)
            new_data.append(link_map[parent])
            if link_map[parent]["type"] == "page":
                break
            parent = link_map[parent].get("parent", None)
        return self.replace(data=new_data, single_item=False)

    def select_descendant_headings(self, link: str, incl_link: bool = False, **kwargs) -> PagesAssetT:
        """Select descendant headings."""
        d = self.find_page(link, wrap=False, **kwargs)
        link_map = {d["link"]: dict(d) for d in self.data}
        new_data = []
        if incl_link:
            new_data.append(d)
        descendants = set()
        stack = [d]
        while stack:
            d = stack.pop()
            children = d.get("children", [])
            for child in children:
                if child in link_map and child not in descendants:
                    if link_map[child]["type"].startswith("heading"):
                        descendants.add(child)
                        new_data.append(link_map[child])
                        stack.append(link_map[child])
        return self.replace(data=new_data, single_item=False)

    def print_site_schema(
        self,
        append_type: bool = False,
        append_obj_type: bool = False,
        structure_fragments: bool = True,
        split_fragments: bool = True,
        **dir_tree_kwargs,
    ) -> None:
        """Print site schema.

        If `structure_fragments` is True, builds a hierarchy of fragments. Otherwise,
        displays them on the same level.

        If `split_fragments` is True, displays fragments as continuation of their parents.
        Otherwise, displays them in full length.

        Keyword arguments are split between `KnowledgeAsset.describe` and
        `vectorbtpro.utils.path_.dir_tree_from_paths`."""
        link_map = {d["link"]: dict(d) for d in self.data}
        links = []
        for link, d in link_map.items():
            if not structure_fragments:
                links.append(link)
                continue
            x = d
            link_base = None
            link_fragments = []
            while x["type"].startswith("heading") and "#" in x["link"]:
                link_parts = x["link"].split("#")
                if link_base is None:
                    link_base = link_parts[0]
                link_fragments.append("#" + link_parts[1])
                if not x.get("parent", None) or x["parent"] not in link_map:
                    if x["type"].startswith("heading"):
                        level = int(x["type"].split()[1])
                        for i in range(level - 1):
                            link_fragments.append("?")
                    break
                x = link_map[x["parent"]]
            if link_base is None:
                links.append(link)
            else:
                if split_fragments and len(link_fragments) > 1:
                    link_fragments = link_fragments[::-1]
                    new_link_fragments = [link_fragments[0]]
                    for i in range(1, len(link_fragments)):
                        link_fragment1 = link_fragments[i - 1]
                        link_fragment2 = link_fragments[i]
                        if link_fragment2.startswith(link_fragment1 + "."):
                            new_link_fragments.append("." + link_fragment2[len(link_fragment1 + ".") :])
                        else:
                            new_link_fragments.append(link_fragment2)
                    link_fragments = new_link_fragments
                links.append(link_base + "/".join(link_fragments))
        paths = self.links_to_paths(links, allow_fragments=not structure_fragments)

        path_names = []
        for i, d in enumerate(link_map.values()):
            path_name = paths[i].name
            brackets = []
            if append_type:
                brackets.append(d["type"])
            if append_obj_type and d["obj_type"]:
                brackets.append(d["obj_type"])
            if brackets:
                path_name += f" [{', '.join(brackets)}]"
            path_names.append(path_name)
        if "root_name" not in dir_tree_kwargs:
            root_name = get_common_prefix(link_map.keys())
            if not root_name:
                root_name = "/"
            dir_tree_kwargs["root_name"] = root_name
        if "sort" not in dir_tree_kwargs:
            dir_tree_kwargs["sort"] = False
        if "path_names" not in dir_tree_kwargs:
            dir_tree_kwargs["path_names"] = path_names
        if "length_limit" not in dir_tree_kwargs:
            dir_tree_kwargs["length_limit"] = None
        print(dir_tree_from_paths(paths, **dir_tree_kwargs))


MessagesAssetT = tp.TypeVar("MessagesAssetT", bound="MessagesAsset")


class MessagesAsset(VBTAsset):
    """Class for working with Discord messages.

    Each message has the following fields:

    link: URL of the message, such as "https://discord.com/channels/918629562441695344/919715148896301067/923327319882485851"
    block: URL of the first message in the block. A block is a bunch of messages of the same author
        that either reference a message of another author, or don't reference any message at all.
    thread: URL of the first message in the thread. A thread is a bunch of blocks that reference each other
        in a chain, such as questions, answers, follow-up questions, etc.
    reference: URL of the message that the message references. Can be None.
    replies: List of URLs of the messages that reference the message
    channel: Channel of the message, such as "support"
    timestamp: Timestamp of the message, such as "2024-01-01 00:00:00"
    author: Author of the message, such as "@polakowo"
    content: String content of the message
    mentions: List of Discord usernames that this message mentions, such as ["@polakowo"]
    attachments: List of attachments. Each attachment has two fields: "file_name", such as "some_image.png",
        and "content" containing the string content extracted from the file.
    reactions: Total number of reactions that this message has received

    For defaults, see `assets.messages` in `vectorbtpro._settings.knowledge`."""

    _settings_path: tp.SettingsPath = "knowledge.assets.messages"

    def minimize(self: MessagesAssetT, minimize_links: tp.Optional[bool] = None) -> MessagesAssetT:
        new_instance = VBTAsset.minimize(self, minimize_links=minimize_links)
        new_instance = new_instance.remove(
            [
                "block",
                "thread",
                "replies",
                "mentions",
                "reactions",
            ],
            skip_missing=True,
        )
        return new_instance

    def aggregate_messages(
        self: MessagesAssetT,
        metadata_format: tp.Optional[str] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeMessagesAsset:
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
            to_markdown_kwargs=to_markdown_kwargs,
            to_html_kwargs=to_html_kwargs,
            **kwargs,
        )

    def aggregate_blocks(
        self: MessagesAssetT,
        collect_kwargs: tp.KwargsLike = None,
        aggregate_fields: tp.Union[None, bool, tp.MaybeIterable[str]] = None,
        parent_links_only: tp.Optional[bool] = None,
        metadata_format: tp.Optional[str] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeMessagesAsset:
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
        instance = self.collect(by="block", wrap=True, **collect_kwargs)
        return instance.apply(
            "agg_block",
            aggregate_fields=aggregate_fields,
            parent_links_only=parent_links_only,
            metadata_format=metadata_format,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            to_markdown_kwargs=to_markdown_kwargs,
            to_html_kwargs=to_html_kwargs,
            link_map={d["link"]: dict(d) for d in self.data},
            **kwargs,
        )

    def aggregate_threads(
        self: MessagesAssetT,
        collect_kwargs: tp.KwargsLike = None,
        aggregate_fields: tp.Union[None, bool, tp.MaybeIterable[str]] = None,
        parent_links_only: tp.Optional[bool] = None,
        metadata_format: tp.Optional[str] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeMessagesAsset:
        """Aggregate messages by thread.

        Same as `MessagesAsset.aggregate_blocks` but for threads.

        Uses `vectorbtpro.utils.knowledge.custom_asset_funcs.AggThreadAssetFunc`."""
        if collect_kwargs is None:
            collect_kwargs = {}
        if "uniform_groups" not in collect_kwargs:
            collect_kwargs["uniform_groups"] = True
        instance = self.collect(by="thread", wrap=True, **collect_kwargs)
        return instance.apply(
            "agg_thread",
            aggregate_fields=aggregate_fields,
            parent_links_only=parent_links_only,
            metadata_format=metadata_format,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            to_markdown_kwargs=to_markdown_kwargs,
            to_html_kwargs=to_html_kwargs,
            link_map={d["link"]: dict(d) for d in self.data},
            **kwargs,
        )

    def aggregate_channels(
        self: MessagesAssetT,
        collect_kwargs: tp.KwargsLike = None,
        aggregate_fields: tp.Union[None, bool, tp.MaybeIterable[str]] = None,
        parent_links_only: tp.Optional[bool] = None,
        metadata_format: tp.Optional[str] = None,
        clear_metadata: tp.Optional[bool] = None,
        clear_metadata_kwargs: tp.KwargsLike = None,
        dump_metadata_kwargs: tp.KwargsLike = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeMessagesAsset:
        """Aggregate messages by channel.

        Same as `MessagesAsset.aggregate_threads` but for channels.

        Uses `vectorbtpro.utils.knowledge.custom_asset_funcs.AggChannelAssetFunc`."""
        if collect_kwargs is None:
            collect_kwargs = {}
        if "uniform_groups" not in collect_kwargs:
            collect_kwargs["uniform_groups"] = True
        instance = self.collect(by="channel", wrap=True, **collect_kwargs)
        return instance.apply(
            "agg_channel",
            aggregate_fields=aggregate_fields,
            parent_links_only=parent_links_only,
            metadata_format=metadata_format,
            clear_metadata=clear_metadata,
            clear_metadata_kwargs=clear_metadata_kwargs,
            dump_metadata_kwargs=dump_metadata_kwargs,
            to_markdown_kwargs=to_markdown_kwargs,
            to_html_kwargs=to_html_kwargs,
            link_map={d["link"]: dict(d) for d in self.data},
            **kwargs,
        )

    @property
    def lowest_aggregate_by(self) -> str:
        """Get the lowest level that aggregates all messages."""
        if len(self) == 1 and self[0].get("attachments", []):
            return "message"
        try:
            if len(set(self.get("block"))) == 1:
                return "block"
        except KeyError:
            pass
        try:
            if len(set(self.get("thread"))) == 1:
                return "thread"
        except KeyError:
            pass
        try:
            if len(set(self.get("channel"))) == 1:
                return "channel"
        except KeyError:
            pass
        raise ValueError("Must provide by")

    def aggregate(self, by: tp.Optional[str] = None, *args, **kwargs) -> tp.MaybeMessagesAsset:
        """Aggregate by "message" (attachments), "block", "thread", or "channel".

        If `by` is None, uses `MessagesAsset.lowest_aggregate_by`."""
        if by is None:
            by = self.lowest_aggregate_by
        if not by.lower().endswith("s"):
            by += "s"
        return getattr(self, "aggregate_" + by.lower())(*args, **kwargs)

    def select_reference(self: MessagesAssetT, link: str, **kwargs) -> MessagesAssetT:
        """Select the reference message."""
        d = self.find_link(link, wrap=False, **kwargs)
        reference = d.get("reference", None)
        new_data = []
        if reference:
            for d2 in self.data:
                if d2["reference"] == reference:
                    new_data.append(d2)
                    break
        return self.replace(data=new_data, single_item=True)

    def select_replies(self: MessagesAssetT, link: str, **kwargs) -> MessagesAssetT:
        """Select the reply messages."""
        d = self.find_link(link, wrap=False, **kwargs)
        replies = d.get("replies", [])
        new_data = []
        if replies:
            reply_data = {reply: None for reply in replies}
            replies_found = 0
            for d2 in self.data:
                if d2["link"] in reply_data:
                    reply_data[d2["link"]] = d2
                    replies_found += 1
                    if replies_found == len(replies):
                        break
            new_data = list(reply_data.values())
        return self.replace(data=new_data, single_item=True)

    def select_block(self: MessagesAssetT, link: str, incl_link: bool = True, **kwargs) -> MessagesAssetT:
        """Select the messages that belong to the block of a link."""
        d = self.find_link(link, wrap=False, **kwargs)
        new_data = []
        for d2 in self.data:
            if d2["block"] == d["block"] and (incl_link or d2["link"] != d["link"]):
                new_data.append(d2)
        return self.replace(data=new_data, single_item=False)

    def select_thread(self: MessagesAssetT, link: str, incl_link: bool = True, **kwargs) -> MessagesAssetT:
        """Select the messages that belong to the thread of a link."""
        d = self.find_link(link, wrap=False, **kwargs)
        new_data = []
        for d2 in self.data:
            if d2["thread"] == d["thread"] and (incl_link or d2["link"] != d["link"]):
                new_data.append(d2)
        return self.replace(data=new_data, single_item=False)

    def select_channel(self: MessagesAssetT, link: str, incl_link: bool = True, **kwargs) -> MessagesAssetT:
        """Select the messages that belong to the channel of a link."""
        d = self.find_link(link, wrap=False, **kwargs)
        new_data = []
        for d2 in self.data:
            if d2["channel"] == d["channel"] and (incl_link or d2["link"] != d["link"]):
                new_data.append(d2)
        return self.replace(data=new_data, single_item=False)

    def find_obj_messages(
        self,
        obj: tp.MaybeList,
        *,
        attr: tp.Optional[str] = None,
        module: tp.Union[None, str, ModuleType] = None,
        resolve: bool = True,
        **kwargs,
    ) -> tp.MaybeMessagesAsset:
        """Find messages relevant to object(s).

        Uses `MessagesAsset.find_obj_mentions`."""
        return self.find_obj_mentions(obj, attr=attr, module=module, resolve=resolve, **kwargs)


def find_api(
    obj: tp.MaybeList,
    *,
    attr: tp.Optional[str] = None,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
    pages_asset: tp.Optional[tp.MaybeType[PagesAssetT]] = None,
    pull_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.MaybePagesAsset:
    """Find API pages and headings relevant to object(s).

    Based on `PagesAsset.find_obj_api`.

    Use `pages_asset` to provide a custom subclass or instance of `PagesAsset`."""
    if pages_asset is None:
        pages_asset = PagesAsset
    if isinstance(pages_asset, type):
        checks.assert_subclass_of(pages_asset, PagesAsset, arg_name="pages_asset")
        if pull_kwargs is None:
            pull_kwargs = {}
        pages_asset = pages_asset.pull(**pull_kwargs)
    checks.assert_instance_of(pages_asset, PagesAsset, arg_name="pages_asset")
    return pages_asset.find_obj_api(obj, attr=attr, module=module, resolve=resolve, **kwargs)


def find_docs(
    obj: tp.MaybeList,
    *,
    attr: tp.Optional[str] = None,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
    pages_asset: tp.Optional[tp.MaybeType[PagesAssetT]] = None,
    pull_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.MaybePagesAsset:
    """Find documentation pages and headings relevant to object(s).

    Based on `PagesAsset.find_obj_docs`.

    Use `pages_asset` to provide a custom subclass or instance of `PagesAsset`."""
    if pages_asset is None:
        pages_asset = PagesAsset
    if isinstance(pages_asset, type):
        checks.assert_subclass_of(pages_asset, PagesAsset, arg_name="pages_asset")
        if pull_kwargs is None:
            pull_kwargs = {}
        pages_asset = pages_asset.pull(**pull_kwargs)
    checks.assert_instance_of(pages_asset, PagesAsset, arg_name="pages_asset")
    return pages_asset.find_obj_docs(obj, attr=attr, module=module, resolve=resolve, **kwargs)


def find_messages(
    obj: tp.MaybeList,
    *,
    attr: tp.Optional[str] = None,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
    messages_asset: tp.Optional[tp.MaybeType[MessagesAssetT]] = None,
    pull_kwargs: tp.KwargsLike = None,
    aggregate_messages: bool = True,
    aggregate_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.MaybeMessagesAsset:
    """Find messages relevant to object(s).

    Based on `MessagesAsset.find_obj_messages`.

    Use `messages_asset` to provide a custom subclass or instance of `MessagesAsset`.

    Set `aggregate_messages` to True to aggregate attachments into message content."""
    if messages_asset is None:
        messages_asset = MessagesAsset
    if isinstance(messages_asset, type):
        checks.assert_subclass_of(messages_asset, MessagesAsset, arg_name="messages_asset")
        if pull_kwargs is None:
            pull_kwargs = {}
        messages_asset = messages_asset.pull(**pull_kwargs)
    checks.assert_instance_of(messages_asset, MessagesAsset, arg_name="messages_asset")
    if aggregate_messages:
        if aggregate_kwargs is None:
            aggregate_kwargs = {}
        messages_asset = messages_asset.aggregate_messages(**aggregate_kwargs)
    return messages_asset.find_obj_messages(obj, attr=attr, module=module, resolve=resolve, **kwargs)


def find_examples(
    obj: tp.MaybeList,
    *,
    attr: tp.Optional[str] = None,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
    as_code: bool = True,
    return_type: tp.Optional[str] = "field",
    pages_asset: tp.Optional[tp.MaybeType[PagesAssetT]] = None,
    messages_asset: tp.Optional[tp.MaybeType[MessagesAssetT]] = None,
    pull_kwargs: tp.KwargsLike = None,
    aggregate_messages: bool = True,
    aggregate_kwargs: tp.KwargsLike = None,
    shuffle_messages: bool = False,
    **kwargs,
) -> tp.MaybeVBTAsset:
    """Find (code) examples relevant to object(s).

    Based on `VBTAsset.find_obj_mentions`.

    By default, extracts code with text. Use `return_type="match"` to extract code without text,
    or, for instance, `return_type="item"` to also get links.

    Use `pages_asset` to provide a custom subclass or instance of `PagesAsset`. Use `messages_asset`
    to provide a custom subclass or instance of `MessagesAsset`.

    Set `aggregate_messages` to True to aggregate attachments into message content.

    Set `shuffle_messages` to True to shuffle messages. Useful in chatting to increase diversity
    when context is too big."""
    if pages_asset is None:
        pages_asset = PagesAsset
    if isinstance(pages_asset, type):
        checks.assert_subclass_of(pages_asset, PagesAsset, arg_name="pages_asset")
        if pull_kwargs is None:
            pull_kwargs = {}
        pages_asset = pages_asset.pull(**pull_kwargs)
    checks.assert_instance_of(pages_asset, PagesAsset, arg_name="pages_asset")
    if messages_asset is None:
        messages_asset = MessagesAsset
    if isinstance(messages_asset, type):
        checks.assert_subclass_of(messages_asset, MessagesAsset, arg_name="messages_asset")
        if pull_kwargs is None:
            pull_kwargs = {}
        messages_asset = messages_asset.pull(**pull_kwargs)
    checks.assert_instance_of(messages_asset, MessagesAsset, arg_name="messages_asset")
    if aggregate_messages:
        if aggregate_kwargs is None:
            aggregate_kwargs = {}
        messages_asset = messages_asset.aggregate_messages(**aggregate_kwargs)
    if shuffle_messages:
        messages_asset = messages_asset.shuffle()
    combined_asset = pages_asset + messages_asset
    return combined_asset.find_obj_mentions(
        obj,
        attr=attr,
        module=module,
        resolve=resolve,
        as_code=as_code,
        return_type=return_type,
        **kwargs,
    )


def find_assets(
    obj: tp.MaybeList,
    *,
    attr: tp.Optional[str] = None,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
    asset_names: tp.Optional[tp.MaybeIterable[str]] = None,
    pages_asset: tp.Optional[tp.MaybeType[PagesAssetT]] = None,
    messages_asset: tp.Optional[tp.MaybeType[MessagesAssetT]] = None,
    pull_kwargs: tp.KwargsLike = None,
    aggregate_messages: bool = True,
    aggregate_kwargs: tp.KwargsLike = None,
    shuffle_messages: bool = False,
    minimize: tp.Optional[bool] = None,
    minimize_pages: tp.Optional[bool] = None,
    minimize_messages: tp.Optional[bool] = None,
    combine: bool = True,
    api_kwargs: tp.KwargsLike = None,
    docs_kwargs: tp.KwargsLike = None,
    messages_kwargs: tp.KwargsLike = None,
    examples_kwargs: tp.KwargsLike = None,
    minimize_kwargs: tp.KwargsLike = None,
    minimize_pages_kwargs: tp.KwargsLike = None,
    minimize_messages_kwargs: tp.KwargsLike = None,
    combine_kwargs: tp.KwargsLike = None,
    **find_kwargs,
) -> tp.MaybeDict[tp.VBTAsset]:
    """Find all assets relevant to object(s).

    Argument `asset_names` can be a list of asset names in any order. It defaults to "api", "docs",
    and "messages", It can also include ellipsis (`...`). For example, `["messages", ...]` puts
    "messages" at the beginning and all other assets in their usual order at the end.
    The following asset names are supported:

    * "api": `find_api` with `api_kwargs`
    * "docs": `find_docs` with `docs_kwargs`
    * "messages": `find_messages` with `messages_kwargs`
    * "examples": `find_examples` with `examples_kwargs`
    * "all": All of the above

    !!! note
        Examples usually overlap with other assets, thus they are excluded by default.

    Use `pages_asset` to provide a custom subclass or instance of `PagesAsset`. Use `messages_asset`
    to provide a custom subclass or instance of `MessagesAsset`. Both assets are reused among "find" calls.

    Set `aggregate_messages` to True to aggregate attachments into message content.

    Set `shuffle_messages` to True to shuffle messages. Useful in chatting to increase diversity
    when context is too big.

    Set `combine` to True to combine all assets into a single asset. Uses
    `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.combine` with `combine_kwargs`.

    Set `minimize` to True (or `minimize_pages` for pages and `minimize_messages` for messages)
    in order to minimize to remove fields that aren't relevant for chatting.
    It defaults to True if `combine` is True, otherwise, it defaults to False. Uses `VBTAsset.minimize`
    with `minimize_kwargs`, `PagesAsset.minimize` with `minimize_pages_kwargs`, and `MessagesAsset.minimize`
    with `minimize_messages_kwargs`. Arguments `minimize_pages_kwargs` and `minimize_messages_kwargs`
    are merged over `minimize_kwargs`."""
    if pages_asset is None:
        pages_asset = PagesAsset
    if isinstance(pages_asset, type):
        checks.assert_subclass_of(pages_asset, PagesAsset, arg_name="pages_asset")
        if pull_kwargs is None:
            pull_kwargs = {}
        pages_asset = pages_asset.pull(**pull_kwargs)
    checks.assert_instance_of(pages_asset, PagesAsset, arg_name="pages_asset")
    if messages_asset is None:
        messages_asset = MessagesAsset
    if isinstance(messages_asset, type):
        checks.assert_subclass_of(messages_asset, MessagesAsset, arg_name="messages_asset")
        if pull_kwargs is None:
            pull_kwargs = {}
        messages_asset = messages_asset.pull(**pull_kwargs)
    checks.assert_instance_of(messages_asset, MessagesAsset, arg_name="messages_asset")
    if aggregate_messages:
        if aggregate_kwargs is None:
            aggregate_kwargs = {}
        messages_asset = messages_asset.aggregate_messages(**aggregate_kwargs)
    if shuffle_messages:
        messages_asset = messages_asset.shuffle()

    if api_kwargs is None:
        api_kwargs = {}
    docs_kwargs = merge_dicts(find_kwargs, docs_kwargs)
    messages_kwargs = merge_dicts(find_kwargs, messages_kwargs)
    examples_kwargs = merge_dicts(find_kwargs, examples_kwargs)

    asset_dict = {}
    all_asset_names = ["api", "docs", "messages", "examples"]
    if asset_names is not None:
        if isinstance(asset_names, str) and asset_names.lower() == "all":
            asset_names = all_asset_names
        else:
            if isinstance(asset_names, (str, type(Ellipsis))):
                asset_names = [asset_names]
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
            asset = find_api(
                obj,
                attr=attr,
                module=module,
                resolve=resolve,
                pages_asset=pages_asset,
                **api_kwargs,
            )
            if len(asset) > 0:
                asset_dict[asset_name] = asset
        elif asset_name == "docs":
            asset = find_docs(
                obj,
                attr=attr,
                module=module,
                resolve=resolve,
                pages_asset=pages_asset,
                **docs_kwargs,
            )
            if len(asset) > 0:
                asset_dict[asset_name] = asset
        elif asset_name == "messages":
            asset = find_messages(
                obj,
                attr=attr,
                module=module,
                resolve=resolve,
                messages_asset=messages_asset,
                aggregate_messages=False,
                aggregate_kwargs=aggregate_kwargs,
                **messages_kwargs,
            )
            if len(asset) > 0:
                asset_dict[asset_name] = asset
        elif asset_name == "examples":
            if examples_kwargs is None:
                examples_kwargs = {}
            asset = find_examples(
                obj,
                attr=attr,
                module=module,
                resolve=resolve,
                pages_asset=pages_asset,
                messages_asset=messages_asset,
                aggregate_messages=False,
                aggregate_kwargs=aggregate_kwargs,
                shuffle_messages=False,
                **examples_kwargs,
            )
            if len(asset) > 0:
                asset_dict[asset_name] = asset

    if minimize is None:
        minimize = combine
    if minimize:
        if minimize_kwargs is None:
            minimize_kwargs = {}
        for k, v in asset_dict.items():
            if isinstance(v, VBTAsset) and not isinstance(v, (PagesAsset, MessagesAsset)):
                asset_dict[k] = v.minimize(**minimize_kwargs)
    if minimize_pages is None:
        minimize_pages = minimize
    if minimize_pages:
        minimize_pages_kwargs = merge_dicts(minimize_kwargs, minimize_pages_kwargs)
        for k, v in asset_dict.items():
            if isinstance(v, PagesAsset):
                asset_dict[k] = v.minimize(**minimize_pages_kwargs)
    if minimize_messages is None:
        minimize_messages = minimize
    if minimize_messages:
        minimize_messages_kwargs = merge_dicts(minimize_kwargs, minimize_messages_kwargs)
        for k, v in asset_dict.items():
            if isinstance(v, MessagesAsset):
                asset_dict[k] = v.minimize(**minimize_messages_kwargs)
    if combine:
        if len(asset_dict) >= 2:
            if combine_kwargs is None:
                combine_kwargs = {}
            return VBTAsset.combine(*asset_dict.values(), **combine_kwargs)
        if len(asset_dict) == 1:
            return list(asset_dict.values())[0]
        return VBTAsset([])
    return asset_dict


def chat_about(
    obj: tp.MaybeList,
    message: str,
    chat_history: tp.ChatHistory = None,
    *,
    asset_names: tp.Optional[tp.MaybeIterable[str]] = "examples",
    shuffle_messages: tp.Optional[bool] = None,
    shuffle: tp.Optional[bool] = None,
    find_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> tp.ChatOutput:
    """Chat about object(s).

    Uses `find_assets` with `combine=True` and `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.chat`.
    Keyword arguments are distributed among these two methods automatically, unless some keys cannot be
    found in both signatures. In such a case, the key will be used for chatting. If this is not wanted,
    specify the `find_assets`-related arguments explicitly with `find_kwargs`.

    If `shuffle` is True, shuffles the combined asset. By default, shuffles only messages (`shuffle=False`
    and `shuffle_messages=True`). If `shuffle` is False, shuffles neither messages nor combined asset."""
    if shuffle is not None:
        if shuffle_messages is None:
            shuffle_messages = False
    else:
        shuffle = False
        if shuffle_messages is None:
            shuffle_messages = True
    find_arg_names = set(get_func_arg_names(find_assets))
    if find_kwargs is None:
        find_kwargs = {}
    else:
        find_kwargs = dict(find_kwargs)
    chat_kwargs = {}
    for k, v in kwargs.items():
        if k in find_arg_names:
            if k not in find_kwargs:
                find_kwargs[k] = v
        else:
            chat_kwargs[k] = v

    asset = find_assets(
        obj,
        asset_names=asset_names,
        combine=True,
        shuffle_messages=shuffle_messages,
        **find_kwargs,
    )
    if shuffle:
        asset = asset.shuffle()
    return asset.chat(message, chat_history, **chat_kwargs)
