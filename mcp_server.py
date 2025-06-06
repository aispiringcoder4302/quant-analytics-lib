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
    to the provided query and return the results as a context string.

    Args:
        query (str): The search query.

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
        search_method (Optional[str]): Strategy for document search. Supported strategies:

            * "embeddings": Uses embeddings for semantic search. Use for general queries.
                Requires downloading embeddings when first used.
            * "bm25": Uses BM25 for lexical search. Use for specific queries.
                Very fast and does not require downloading embeddings.
            * "hybrid": Combines both embeddings and BM25. Use for balanced search.

            Defaults to "bm25".
        return_chunks (bool): Whether to return the chunks of the results; otherwise, returns the full results.

            Defaults to True.

    Returns:
        str: A context string containing the search results.
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


if __name__ == "__main__":
    mcp.run()
