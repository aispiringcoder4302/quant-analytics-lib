"""Package providing classes and utilities for resampling.

!!! info
    For default settings, see `vectorbtpro._settings.resampling`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.base.resampling.base import *
    from vectorbtpro.base.resampling.nb import *
