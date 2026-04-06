"""Package for working with returns.

Provides common financial risk and performance metrics modeled after
[empyrical](https://github.com/quantopian/empyrical), an adapter for quantstats,
and additional return-based features.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.returns.accessors import *
    from vectorbtpro.returns.nb import *
    from vectorbtpro.returns.qs_adapter import *

__exclude_from__all__ = [
    "enums",
]

__import_if_installed__ = dict()
__import_if_installed__["qs_adapter"] = "quantstats"
