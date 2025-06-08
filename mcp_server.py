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
    """Search for VectorBT PRO (also called vectorbtpro or simply VBT) assets relevant
    to the provided (natural language) query and return the results as a context string.

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
def get_source(refname: str, module: Optional[str] = None) -> str:
    """Get the source code of any object.

    This can be used to inspect the implementation of VectorBT PRO objects, such as modules,
    classes, functions, and instances. It uses AST parsing to retrieve the source code
    of any object, including named tuples, class variables, dataclasses, and other objects that
    may not have a traditional source code representation.

    Args:
        refname (str): Reference to the object.

            A reference can be a fully-qualified dotted name (e.g., "vectorbtpro.data.base.Data")
            or a short unambiguous name (e.g., "Data", "vbt.Portfolio").
        module (Optional[str]): Module name to resolve the reference.

            By default, the module is inferred from the reference name.

    Returns:
        str: Source code of the object.
    """
    from vectorbtpro.utils.source import get_source
    from vectorbtpro.utils.module_ import resolve_refname

    refname = resolve_refname(refname, module=module)
    if not refname:
        raise ValueError("Reference name cannot be resolved to an object")
    return get_source(refname)


@mcp.tool()
def get_attrs(refname: str, module: Optional[str] = None, own_only: bool = False) -> List[dict]:
    """Get attributes of any object, such as module, class, function, or instance.

    Can be used to discover the API of VectorBT PRO. For example, you can use it to
    find out what attributes are available in a module or class.

    Args:
        refname (str): Reference to the object.

            A reference can be a fully-qualified dotted name (e.g., "vectorbtpro.data.base.Data")
            or a short unambiguous name (e.g., "Data", "vbt.Portfolio").
        module (Optional[str]): Module name to resolve the reference.

            By default, the module is inferred from the reference name.
        own_only (bool): If True, only include attributes defined on the object's class.

            When False, include all attributes, including those inherited from parent classes.

    Returns:
        List[dict]: List of dictionaries containing the attributes and their metadata.
    """
    from vectorbtpro.utils.attr_ import get_attrs
    from vectorbtpro.utils.module_ import resolve_refname, get_refname_obj

    refname = resolve_refname(refname, module=module)
    if not refname:
        raise ValueError("Reference name cannot be resolved to an object")
    obj = get_refname_obj(refname)
    df = get_attrs(obj, own_only=own_only)
    return df.to_dict(orient="records")


if __name__ == "__main__":
    mcp.run()
