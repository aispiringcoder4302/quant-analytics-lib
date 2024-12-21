# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Named tuples and enumerated types for OHLC(V) data.

Defines enums and other schemas for `vectorbtpro.ohlcv`."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.formatting import prettify

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
] = f"""Price feature.

```python
{prettify(PriceFeature)}
```
"""
