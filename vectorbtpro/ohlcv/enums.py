"""Module providing named tuples and enumerated types for OHLC(V) data."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.formatting import prettify_doc

__pdoc__all__ = __all__ = [
    "PriceFeature",
]

__pdoc__ = {}


# ############# Enums ############# #


class PriceFeatureT(tp.NamedTuple):
    Open: int = 0
    High: int = 1
    Low: int = 2
    Close: int = 3


PriceFeature = PriceFeatureT()
"""_"""

__pdoc__[
    "PriceFeature"
] = f"""Price feature enumeration.

Fields:
    Open: Index for the open price.
    High: Index for the high price.
    Low: Index for the low price.
    Close: Index for the close price.

```python
{prettify_doc(PriceFeature)}
```
"""
