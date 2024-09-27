# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for working with knowledge."""

import os
import io
import json
import requests
from pathlib import Path

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import Configured
from vectorbtpro.utils.pickling import suggest_compression, decompress, load_bytes
from vectorbtpro.utils.path_ import check_mkdir
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.template import CustomTemplate, RepEval, RepFunc, substitute_templates
from vectorbtpro.utils.config import flat_merge_dicts
from vectorbtpro.utils.search import any_in_obj, search_text, search_regex, search_fuzzy

__all__ = [
    "JSONAsset",
    "ReleaseAsset",
    "MessagesAsset",
    "PagesAsset",
]


JSONAssetT = tp.TypeVar("JSONAssetT", bound="JSONAsset")


class JSONAsset(Configured):
    """Class for working with a JSON asset.

    For defaults, see `vectorbtpro._settings.knowledge`."""

    _settings_path: tp.SettingsPath = "knowledge"

    _expected_keys: tp.ExpectedKeys = (Configured._expected_keys or set()) | {"json_obj"}

    @classmethod
    def from_file(
        cls,
        path: tp.PathLike,
        compression: tp.Union[None, bool, str] = None,
        decompress_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> JSONAssetT:
        """Build `JSONAsset` from a (compressed or not) JSON file."""
        bytes_ = load_bytes(path, compression=compression, decompress_kwargs=decompress_kwargs)
        json_str = bytes_.decode("utf-8")
        return cls(json.loads(json_str), **kwargs)

    @classmethod
    def from_bytes(
        cls,
        bytes_: bytes,
        compression: tp.Union[None, bool, str] = None,
        decompress_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> JSONAssetT:
        """Build `JSONAsset` from a (compressed or not) bytes."""
        if decompress_kwargs is None:
            decompress_kwargs = {}
        bytes_ = decompress(bytes_, compression=compression, **decompress_kwargs)
        json_str = bytes_.decode("utf-8")
        return cls(json.loads(json_str), **kwargs)

    def __init__(self, json_obj: tp.JSON, **kwargs) -> None:
        Configured.__init__(
            self,
            json_obj=json_obj,
            **kwargs,
        )

        self._json_obj = json_obj

    @property
    def json_obj(self) -> tp.JSON:
        """JSON object."""
        return self._json_obj

    def query(
        self,
        expression: tp.Union[str, CustomTemplate],
        engine: tp.Optional[str] = None,
        as_filter: bool = True,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        """Query JSON object using an engine.

        Following engines are supported:

        * "jmespath": Evaluation with `jmespath` package
        * "jsonpath", "jsonpath-ng" or "jsonpath_ng": Evaluation with `jsonpath-ng` package
        * "jsonpath.ext", "jsonpath-ng.ext" or "jsonpath_ng.ext": Evaluation with extended `jsonpath-ng` package
        * None or "template": Evaluation of each object as a template. The object is represented
            as "x" while its fields (if any) are represented by their names.
        * "pandas": Same as above but variables being columns

        Templates can also use the following functions:

        * `vectorbtpro.utils.search.search_text`
        * `vectorbtpro.utils.search.search_regex`
        * `vectorbtpro.utils.search.search_fuzzy`

        They work on single values and sequences alike.

        Keyword arguments are passed to the respective search/parse/evaluation function.

        Usage:
            ```pycon
            >>> json_obj = [
            ...     {"name": "Seattle", "state": "WA"},
            ...     {"name": "New York", "state": "NY"},
            ...     {"name": "Bellevue", "state": "WA"},
            ...     {"name": "Olympia", "state": "WA"}
            ... ]
            >>> json_asset = vbt.JSONAsset(json_obj)

            >>> json_asset.query("x['state'] == 'NY'")
            [{'name': 'New York', 'state': 'NY'}]

            >>> json_asset.query("x['state'] == 'NY'", as_filter=False)
            [False, True, False, False]

            >>> json_asset.query("state == 'NY'")
            [{'name': 'New York', 'state': 'NY'}]

            >>> json_asset.query("search_text(name, 'York')")
            [{'name': 'New York', 'state': 'NY'}]

            >>> json_asset.query(lambda state: state == "NY")
            [{'name': 'New York', 'state': 'NY'}]

            >>> json_asset.query(lambda state: state == "NY")
            [{'name': 'New York', 'state': 'NY'}]

            >>> json_asset.query("[?state == `NY`]", engine="jmespath")
            [{'name': 'New York', 'state': 'NY'}]

            >>> json_asset.query("[?state == `NY`].name", engine="jmespath")
            ['New York']

            >>> json_asset.query("$[?state == 'NY'].name", engine="jsonpath.ext")
            ['New York']

            >>> json_asset.query("name[state == 'NY']", engine="pandas")
            ['New York']
            ```

        """
        engine = self.resolve_setting(engine, "engine")

        if engine is None or engine.lower() in ("template", "pandas"):
            if isinstance(expression, str):
                expression = RepEval(expression)
            elif callable(expression):
                expression = RepFunc(expression)
            elif not isinstance(expression, CustomTemplate):
                raise TypeError(f"Expression must be a template")
        if engine is None or engine.lower() == "template":
            template_context = self.resolve_setting(template_context, "template_context", merge=True)
            json_obj = self.json_obj
            if not isinstance(json_obj, list):
                json_obj = [json_obj]
                single_obj = True
            else:
                single_obj = False
            new_json_obj = []
            for obj in json_obj:
                _template_context = flat_merge_dicts(
                    {
                        "x": obj,
                        "search_text": search_text,
                        "search_regex": search_regex,
                        "search_fuzzy": search_fuzzy,
                        **(obj if isinstance(obj, dict) else {}),
                    },
                    template_context,
                )
                new_obj = expression.substitute(_template_context, eval_id="expression", **kwargs)
                if as_filter and isinstance(new_obj, bool):
                    if new_obj:
                        new_json_obj.append(obj)
                else:
                    new_json_obj.append(new_obj)
            if single_obj:
                if len(new_json_obj) > 0:
                    new_json_obj = new_json_obj[0]
                else:
                    new_json_obj = None
        else:
            if engine.lower() == "jmespath":
                from vectorbtpro.utils.module_ import assert_can_import

                assert_can_import("jmespath")
                import jmespath

                new_json_obj = jmespath.search(expression, self.json_obj, **kwargs)
            elif engine.lower() in ("jsonpath", "jsonpath-ng", "jsonpath_ng"):
                from vectorbtpro.utils.module_ import assert_can_import

                assert_can_import("jsonpath_ng")
                import jsonpath_ng

                jsonpath_expr = jsonpath_ng.parse(expression)
                new_json_obj = [match.value for match in jsonpath_expr.find(self.json_obj, **kwargs)]
            elif engine.lower() in ("jsonpath.ext", "jsonpath-ng.ext", "jsonpath_ng.ext"):
                from vectorbtpro.utils.module_ import assert_can_import

                assert_can_import("jsonpath_ng")
                import jsonpath_ng.ext

                jsonpath_expr = jsonpath_ng.ext.parse(expression)
                new_json_obj = [match.value for match in jsonpath_expr.find(self.json_obj, **kwargs)]
            elif engine.lower() == "pandas":
                df = pd.DataFrame.from_records(self.json_obj)
                _template_context = flat_merge_dicts(
                    {
                        "x": df,
                        "search_text": search_text,
                        "search_regex": search_regex,
                        "search_fuzzy": search_fuzzy,
                        **df.to_dict(orient="series"),
                    },
                    template_context,
                )
                result = expression.substitute(_template_context, eval_id="expression", **kwargs)
                if as_filter and isinstance(result, pd.Series) and result.dtype == "bool":
                    result = df[result]
                if isinstance(result, pd.Series):
                    new_json_obj = result.tolist()
                elif isinstance(result, pd.DataFrame):
                    new_json_obj = result.to_dict(orient="records")
                else:
                    new_json_obj = result
            else:
                raise ValueError(f"Invalid option engine='{engine}'")
        return new_json_obj

    def filter(self, *args, **kwargs) -> JSONAssetT:
        """Build a new instance from `JSONAsset.query`."""
        new_json_obj = self.query(*args, **kwargs)
        return self.replace(json_obj=new_json_obj)

    def match_func(
        self,
        key: tp.Optional[tp.Hashable],
        obj: object,
        query: tp.MaybeIterable[tp.JSONPrimitive],
        search_method: str = "text",
        **kwargs,
    ) -> bool:
        """Match function for `JSONAsset.find`."""
        if query is None or isinstance(query, (str, int, float, bool)):
            queries = [query]
        else:
            queries = query
        for query in queries:
            if obj is None and query is None:
                return True
            elif isinstance(obj, str) and isinstance(query, str):
                if search_method.lower() == "text":
                    if search_text(obj, query, **kwargs):
                        return True
                elif search_method.lower() == "regex":
                    if search_regex(obj, query, **kwargs):
                        return True
                elif search_method.lower() == "fuzzy":
                    if search_fuzzy(obj, query, **kwargs):
                        return True
                else:
                    raise ValueError(f"Invalid option search_method='{search_method}'")
            elif isinstance(obj, (int, float)) and isinstance(query, (int, float)):
                return True
            elif isinstance(obj, bool) and isinstance(query, bool):
                return True
        return False

    def find(
        self,
        query: tp.MaybeIterable[tp.JSONPrimitive],
        where: tp.Optional[str] = None,
        search_method: str = "text",
        stringify: bool = False,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> JSONAssetT:
        """Find occurrences and build a new `JSONAsset` instance.

        Uses `vectorbtpro.utils.search.any_in_obj` (keyword arguments are passed here)
        on `JSONAsset.match_func` to find occurrences in each object.

        Query can be one or multiple of JSON primitives.

        Following search methods are supported:

        * "text": `vectorbtpro.utils.search.search_text`
        * "regex": `vectorbtpro.utils.search.search_regex`
        * "fuzzy": `vectorbtpro.utils.search.search_fuzzy`

        If `where` is provided, it becomes a template. In this template, the object is represented
        as "x" while its fields (if any) are represented by their names.
        """
        if where is not None:
            if isinstance(where, str):
                where = RepEval(where)
            elif callable(where):
                where = RepFunc(where)
            elif not isinstance(where, CustomTemplate):
                raise TypeError(f"Expression 'where' must be a template")

        json_obj = self.json_obj
        if not isinstance(json_obj, list):
            json_obj = [json_obj]
            single_obj = True
        else:
            single_obj = False
        new_json_obj = []
        for obj in json_obj:
            if where is not None:
                _template_context = flat_merge_dicts({
                    "x": obj,
                    **(obj if isinstance(obj, dict) else {}),
                }, template_context)
                _obj = where.substitute(_template_context, eval_id="where", **kwargs)
            else:
                _obj = obj
            if not isinstance(_obj, str) and stringify:
                _obj = json.dumps(_obj, ensure_ascii=False)
            if any_in_obj(
                _obj,
                self.match_func,
                incl_types=list,
                query=query,
                search_method=search_method,
                **kwargs,
            ):
                new_json_obj.append(obj)
        if single_obj:
            if len(new_json_obj) > 0:
                new_json_obj = new_json_obj[0]
            else:
                new_json_obj = None
        return self.replace(json_obj=new_json_obj)


ReleaseAssetT = tp.TypeVar("ReleaseAssetT", bound="ReleaseAsset")


class ReleaseAsset(JSONAsset):
    """Class for working with release assets."""

    @classmethod
    def pull(
        cls,
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
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> ReleaseAssetT:
        """Build `ReleaseAsset` from a JSON file that is an asset of a release."""
        from vectorbtpro._version import __version__

        asset_name = cls.resolve_setting(asset_name, "asset_name")
        release_name = cls.resolve_setting(release_name, "release_name")
        repo_owner = cls.resolve_setting(repo_owner, "repo_owner")
        repo_name = cls.resolve_setting(repo_name, "repo_name")
        token = cls.resolve_setting(token, "token")
        token_required = cls.resolve_setting(token_required, "token_required")
        use_pygithub = cls.resolve_setting(use_pygithub, "use_pygithub")
        chunk_size = cls.resolve_setting(chunk_size, "chunk_size")
        cache = cls.resolve_setting(cache, "cache")
        cache_dir = cls.resolve_setting(cache_dir, "cache_dir")
        cache_mkdir_kwargs = cls.resolve_setting(cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True)
        show_progress = cls.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = cls.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)

        current_release = "v" + __version__
        if release_name is None:
            release_name = current_release
        template_context = cls.resolve_setting(template_context, "template_context", merge=True)
        template_context = flat_merge_dicts(
            dict(
                asset_name=asset_name,
                release_name=release_name,
                repo_owner=repo_owner,
                repo_name=repo_name,
                current_release=current_release,
            ),
            template_context,
        )
        if cache:
            cache_dir = substitute_templates(cache_dir, template_context, eval_id="cache_dir")
            cache_dir = Path(cache_dir)
            if cache_dir.exists():
                cache_file = None
                for file in cache_dir.iterdir():
                    if file.is_file() and file.name == asset_name:
                        cache_file = file
                        break
                if cache_file is not None:
                    return cls.from_file(cache_file, **kwargs)

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
        pbar_kwargs = substitute_templates(pbar_kwargs, template_context, eval_id="pbar_kwargs")

        if cache:
            check_mkdir(cache_dir, **cache_mkdir_kwargs)
            cache_file = cache_dir / asset_name
            with open(cache_file, "wb") as f:
                with ProgressBar(total=file_size, show_progress=show_progress, **pbar_kwargs) as pbar:
                    for chunk in asset_response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return cls.from_file(cache_file, **kwargs)
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
            return cls.from_bytes(bytes_, **kwargs)


class MessagesAsset(ReleaseAsset):
    """Class for working with Discord messages."""

    _settings_path: tp.SettingsPath = "knowledge.assets.messages"


class PagesAsset(ReleaseAsset):
    """Class for working with website pages."""

    _settings_path: tp.SettingsPath = "knowledge.assets.pages"
