"""Package providing classes and utilities for portfolio optimization.

!!! info
    For default settings, see `vectorbtpro._settings.pfopt`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.portfolio.pfopt.base import *
    from vectorbtpro.portfolio.pfopt.nb import *
    from vectorbtpro.portfolio.pfopt.records import *
