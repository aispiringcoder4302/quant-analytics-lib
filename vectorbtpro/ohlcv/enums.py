# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Named tuples and enumerated types for OHLC(V) data.

Defines enums and other schemas for `vectorbtpro.ohlcv`."""

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
] = f"""Price feature.

```python
{prettify_doc(PriceFeature)}
```
"""
