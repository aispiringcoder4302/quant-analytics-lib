import ast
from typing import Optional, Union, List, Any
import vectorbtpro as vbt

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("VectorBT PRO")


def auto_cast(value: Any) -> Any:
    """Automatically cast a string to an appropriate Python literal type."""
    if value is not None and isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value


@mcp.tool()
def search(
    query: str,
    n: int = 5,
    page: int = 1,
    assets: Optional[Union[str, List[str]]] = "all",
    search_method: Optional[str] = "bm25",
    return_chunks: bool = True,
) -> str:
    """Search for VectorBT PRO (vectorbtpro, VBT) assets relevant
    to the provided (natural language) query and return the results as a context string.

    !!! note
        This tool is designed to search for general information about VectorBT PRO.
        For specific information about a specific object (such as `vbt.Portfolio`),
        use tools that take a reference name. They operate on the actual objects.

    Args:
        query (str): Search query.

            Do not reinstate the name "VectorBT PRO" in the query, as it is already implied.
        n (int): Number of top documents to return per page.

            Defaults to 5.
        page (int): Page number to return (1-indexed).

            Use to paginate results. For example, if `n=5` and `page=2`, it will return the
            6th to 10th results. Defaults to 1.
        assets (Optional[Union[str, List[str]]]): One or more assets to search. Supported assets:

            * "api": API reference. Best for specific API queries.
            * "docs": Regular documentation, including getting started, features, tutorials, guides,
                recipes, and legal information. Best for general queries.
            * "messages": Discord messages and discussions. Best for support queries.
            * "examples": Code examples across all assets. Best for practical implementation queries.
            * "all": All of the above. Best for comprehensive queries.

            Defaults to "all".
        search_method (Optional[str]): Strategy for document search. Supported strategies:

            * "bm25": Uses BM25 for lexical search. Best for specific keywords.
            * "embeddings": Uses embeddings for semantic search. Best for general queries.
            * "hybrid": Combines both embeddings and BM25. Best for balanced search.

            Defaults to "bm25".
        return_chunks (bool): Whether to return the chunks of the results; otherwise, returns the full results.

            Defaults to True.

    Returns:
        str: Context string containing the search results.
    """
    results = vbt.search(
        query,
        search_method=auto_cast(search_method),
        return_chunks=auto_cast(return_chunks),
        find_assets_kwargs=dict(asset_names=auto_cast(assets)),
        display=False,
    )
    n = auto_cast(n)
    page = auto_cast(page)
    if page < 1:
        raise ValueError("Page number must be greater than or equal to 1")
    results = results[(page - 1) * n : page * n]
    return results.to_context()


@mcp.tool()
def get_source(refname: Union[str, List[str]], module: Optional[str] = None) -> str:
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


@mcp.tool()
def attr_tree(refname: str, module: Optional[str] = None, own_only: bool = False) -> str:
    """Get a visual tree of an object's attributes.

    Can be used to discover the API of VectorBT PRO (vectorbtpro, VBT). For example, use it to
    find out what methods and properties are available on a specific class, or to explore the
    objects defined in a module.

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

    refname = auto_cast(refname)
    module = auto_cast(module)
    resolved_refname = resolve_refname(refname, module=module)
    if not resolved_refname:
        raise ValueError(f"Reference name '{refname}' cannot be resolved to an object")
    obj = get_refname_obj(resolved_refname)
    return attr_tree(obj, own_only=own_only)


if __name__ == "__main__":
    mcp.run()
