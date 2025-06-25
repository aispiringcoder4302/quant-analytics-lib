# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing MCP server tools.

This module is meant to be executed as a script to run the MCP server."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.module_ import assert_can_import

__all__ = []


def auto_cast(value: tp.Any) -> tp.Any:
    """Automatically cast a string to an appropriate Python literal type."""
    import ast

    if value is not None and isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value


def search(
    query: str,
    asset_names: tp.Union[str, tp.List[str]] = "all",
    search_method: str = "hybrid",
    return_chunks: bool = True,
    return_metadata: str = "none",
    n: int = 5,
    page: int = 1,
) -> str:
    """Search for VectorBT PRO (vectorbtpro, VBT) assets relevant
    to the provided (natural language) query and return the results as a context string.

    !!! note
        This tool is designed to search for general information about VBT.
        For specific information about a specific object (such as `vbt.Portfolio`),
        use tools that take a reference name. They operate on the actual objects.

        Also, running this tool on any combination of assets for the first time may take a while,
        as it prepares and caches the documents. If the tool times out repeatedly,
        it's recommended to call `vbt.search()` directly in your code to build the index
        and then use the MCP tool to search the index.

    Args:
        query (str): Search query.

            Do not reinstate the name "VectorBT PRO" in the query, as it is already implied.
        asset_names (Union[str, List[str]]): One or more asset names to search. Supported names:

            * "api": API reference. Best for specific API queries.
            * "docs": Regular documentation, including getting started, features, tutorials, guides,
                recipes, and legal information. Best for general queries.
            * "messages": Discord messages and discussions. Best for support queries.
            * "examples": Code examples across all assets. Best for practical implementation queries.
            * "all": All of the above. Best for comprehensive queries.

            Do not use "examples" with other assets, such as ["examples", "docs"], as examples are already
            included in those assets. Use them separately to get only examples, such as ["examples"].

            Order doesn't matter.

            Defaults to "all".
        search_method (str): Strategy for document search. Supported strategies:

            * "bm25": Uses BM25 for lexical search. Best for specific keywords.
            * "embeddings": Uses embeddings for semantic search. Best for general queries.
            * "hybrid": Combines both embeddings and BM25. Best for balanced search.

            Defaults to "hybrid".
        return_chunks (bool): Whether to return the chunks of the results; otherwise, returns the full results.

            Defaults to True.
        return_metadata (str): Metadata to return with the results. Supported options:

            * "none": No metadata.
            * "minimal": Minimal metadata, such as title and URL.
            * "full": Full metadata, including hierarchy and relationships.

            Defaults to "none".
        n (int): Number of results to return per page.

            Defaults to 5.
        page (int): Page number to return (1-indexed).

            Use to paginate results. For example, if `n=5` and `page=2`, it will return the
            6th to 10th results. Defaults to 1.

    Returns:
        str: Context string containing the search results.
    """
    from vectorbtpro.utils.knowledge.custom_assets import search

    query = auto_cast(query)
    asset_names = auto_cast(asset_names)
    search_method = auto_cast(search_method)
    return_chunks = auto_cast(return_chunks)
    return_metadata = auto_cast(return_metadata)
    n = auto_cast(n)
    page = auto_cast(page)

    if search_method == "embeddings":
        search_method = "embeddings_fallback"
    elif search_method == "hybrid":
        search_method = "hybrid_fallback"

    results = search(
        query,
        search_method=search_method,
        return_chunks=return_chunks,
        find_assets_kwargs=dict(
            asset_names=asset_names,
            minimize=False,
        ),
        display=False,
    )
    if page < 1:
        raise ValueError("Page number must be greater than or equal to 1")
    results = results[(page - 1) * n : page * n]
    if return_metadata.lower() == "minimal":
        results = results.minimize_metadata()
    elif return_metadata.lower() == "none":
        results = results.remove_metadata()
    elif return_metadata.lower() != "full":
        raise ValueError(f"Invalid return_metadata: '{return_metadata}'")
    return results.to_context()


def find(
    refname: tp.Union[str, tp.List[str]],
    module: tp.Optional[str] = None,
    resolve: bool = True,
    asset_names: tp.Union[str, tp.List[str]] = "all",
    aggregate_api: bool = False,
    aggregate_messages: bool = False,
    return_metadata: str = "none",
    n: int = 5,
    page: int = 1,
) -> str:
    """Find VectorBT PRO (vectorbtpro, VBT) assets relevant to a specific object and
    return the results as a context string.

    This can be used to find assets mentioning specific VBT objects, such as modules, classes,
    functions, and instances. For example, searching for "vbt.Portfolio" will generate a list
    of the following targets that include some of the most common portfolio usage patterns:

    ```
    ['vbt.PF',
     'vbt.Portfolio',
     'from vectorbtpro.portfolio.base import Portfolio',
     'vectorbtpro.portfolio.base.Portfolio',
     ...
     'Portfolio(',
     'Portfolio.',
     'pf =',
     'pf.']
    ```

    If any of the mentioned targets are found in an asset, it will be returned.

    Args:
        refname (Union[str, List[str]]): One or more references to the object(s).

            A reference can be a fully-qualified dotted name (e.g., "vectorbtpro.data.base.Data")
            or a short name (e.g., "Data", "vbt.Portfolio") that uniquely identifies the object.

            If multiple references are provided, returns a code example if any of the references
            are found in the code example.
        module (Optional[str]): Module name to resolve the reference.

            By default, the module is inferred from the reference name.
        resolve (bool): Whether to resolve the object's reference name.

            Set to False to find any string, not just VBT objects, such as "SQLAlchemy".
            In this case, `refname` becomes a simple string to match against.
            Defaults to True.
        asset_names (Union[str, List[str]]): One or more asset names to search. Supported names:

            * "api": API reference.
            * "docs": Regular documentation, including getting started, features, tutorials, guides,
                recipes, and legal information.
            * "messages": Discord messages and discussions.
            * "examples": Code examples across all assets.
            * "all": All of the above, in the order specified above.

            Do not use "examples" with other assets, as examples are already included in those assets.
            Use them separately to get only examples.

            Order matters. May also include ellipsis. For example, `["messages", "..."]` puts
            "messages" at the beginning and all other assets in their usual order at the end.

            Defaults to "all".
        aggregate_api (bool): Whether to aggregate all children of the object into a single context.

            If True, the context will contain all the children of the object, such as methods,
            properties, and attributes, in a single context string. Note that this might result in
            a large context string, especially for modules and classes. If False, the context
            will contain only the object description.

            Applies only to API documentation. Defaults to False.
        aggregate_messages (bool): Whether to aggregate messages belonging to the same thread (question-reply chain).

            If True, finding an object in a message will return the question and all replies
            in a single context string, not just the isolated message containing the object.

            Applies only to Discord messages. Defaults to False.
        return_metadata (str): Metadata to return with the results. Supported options:

            * "none": No metadata.
            * "minimal": Minimal metadata, such as title and URL.
            * "full": Full metadata, including hierarchy and relationships.

            Defaults to "none".
        n (int): Number of results to return per page.

            Defaults to 5.
        page (int): Page number to return (1-indexed).

            Use to paginate results. For example, if `n=5` and `page=2`, it will return the
            6th to 10th results. Defaults to 1.

    Returns:
        str: Context string containing the search results.
    """
    from vectorbtpro.utils.knowledge.custom_assets import find_assets

    refname = auto_cast(refname)
    module = auto_cast(module)
    resolve = auto_cast(resolve)
    asset_names = auto_cast(asset_names)
    aggregate_api = auto_cast(aggregate_api)
    aggregate_messages = auto_cast(aggregate_messages)
    return_metadata = auto_cast(return_metadata)
    n = auto_cast(n)
    page = auto_cast(page)

    results = find_assets(
        refname,
        module=module,
        resolve=resolve,
        asset_names=asset_names,
        api_kwargs=dict(
            only_obj=True,
            aggregate=aggregate_api,
        ),
        docs_kwargs=dict(
            aggregate=False,
            up_aggregate=False,
        ),
        messages_kwargs=dict(
            aggregate="threads" if aggregate_messages else "messages",
            latest_first=True,
        ),
        examples_kwargs=dict(
            return_type="match" if return_metadata.lower() == "none" else "item",
            latest_messages_first=True,
        ),
        minimize=False,
    )
    if page < 1:
        raise ValueError("Page number must be greater than or equal to 1")
    results = results[(page - 1) * n : page * n]
    if return_metadata.lower() == "minimal":
        results = results.minimize_metadata()
    elif return_metadata.lower() == "none":
        results = results.remove_metadata()
    elif return_metadata.lower() != "full":
        raise ValueError(f"Invalid return_metadata: '{return_metadata}'")
    return results.to_context()


def get_source(refname: tp.Union[str, tp.List[str]], module: tp.Optional[str] = None) -> str:
    """Get the source code of any object.

    This can be used to inspect the implementation of VectorBT PRO (vectorbtpro, VBT) objects,
    such as modules, classes, functions, and instances. It uses AST parsing to retrieve the source code
    of any object, including named tuples, class variables, dataclasses, and other objects that
    may not have a traditional source code representation.

    Args:
        refname (Union[str, List[str]]): One or more references to the object(s).

            A reference can be a fully-qualified dotted name (e.g., "vectorbtpro.data.base.Data")
            or a short name (e.g., "Data", "vbt.Portfolio") that uniquely identifies the object.
        module (Optional[str]): Module name to resolve the reference.

            By default, the module is inferred from the reference name.

    Returns:
        str: Source code of the object.

            Multiple references can be provided, in which case the source code of each object
            is concatenated together, separated by two newlines.
    """
    from vectorbtpro.utils.source import get_source
    from vectorbtpro.utils.module_ import resolve_refname

    refname = auto_cast(refname)
    module = auto_cast(module)
    if isinstance(refname, str):
        refname = [refname]
    sources = []
    for name in refname:
        resolved_name = resolve_refname(name, module=module)
        if not resolved_name:
            raise ValueError(f"Reference name '{name}' cannot be resolved to an object")
        sources.append(get_source(resolved_name))
    return "\n\n".join(sources)


def attr_tree(refname: str, module: tp.Optional[str] = None, own_only: bool = False) -> str:
    """Get a visual tree of an object's attributes in YAML format.

    Can be used to discover the API of VectorBT PRO (vectorbtpro, VBT). For example, use it to
    find out what methods and properties are available on a specific class, or to explore the
    objects defined in a module.

    Each attribute is represented as a leaf in the tree, whereas the tree structure
    represents the hierarchy of modules and classes from which the attributes are inherited.

    Each leaf in the tree is formatted as:

    ```
    <attr_name> [<attr_type>] (@ <qualname>)
    ```

    where the "@ <qualname>" suffix is shown only when the attribute's `__qualname__` either

    * differs from the attribute's own name (indicating an alias or re-export), or
    * is shared by multiple attributes (true duplicates).

    Args:
        refname (str): Reference to the object.

            A reference can be a fully-qualified dotted name (e.g., "vectorbtpro.data.base.Data")
            or a short name (e.g., "Data", "vbt.Portfolio") that uniquely identifies the object.

            Pass "vbt" to get all the attributes of the `vectorbtpro` module.
        own_only (bool): If True, include only attributes that are defined directly on
            object's class/module; inherited members are omitted.
        module (Optional[str]): Module name to resolve the reference.

            By default, the module is inferred from the reference name.

    Returns:
        str: Printable, newline-separated string representing the attribute hierarchy.
    """
    from vectorbtpro.utils.attr_ import attr_tree
    from vectorbtpro.utils.module_ import resolve_refname, get_refname_obj

    def _yaml_name_formatter(name, is_directory):
        if is_directory:
            return f"{name}:"
        return name

    refname = auto_cast(refname)
    module = auto_cast(module)
    resolved_refname = resolve_refname(refname, module=module)
    if not resolved_refname:
        raise ValueError(f"Reference name '{refname}' cannot be resolved to an object")
    obj = get_refname_obj(resolved_refname)
    return (
        "```yaml\n"
        + attr_tree(
            obj,
            own_only=own_only,
            root_as_item=True,
            name_formatter=_yaml_name_formatter,
            space="  ",
            branch="  ",
            tee="- ",
            last="- ",
        )
        + "\n```"
    )


if __name__ == "__main__":
    assert_can_import("mcp")
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("VectorBT PRO")
    mcp.tool()(search)
    mcp.tool()(find)
    mcp.tool()(get_source)
    mcp.tool()(attr_tree)
    mcp.run()
