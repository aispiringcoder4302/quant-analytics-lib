# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Numba-compiled functions for generic data.

Provides an arsenal of Numba-compiled functions that are used by accessors
and in many other parts of a backtesting pipeline, such as technical indicators.
These only accept NumPy arrays and other Numba-compatible types.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    Rolling functions with `minp=None` have `min_periods` set to the window size.

    All functions passed as argument must be Numba-compiled.

    Records must retain the order they were created in.

!!! warning
    Make sure to use `parallel=True` only if your columns are independent.
"""

from vectorbtpro.generic.nb.apply_reduce import *
from vectorbtpro.generic.nb.base import *
from vectorbtpro.generic.nb.iter_ import *
from vectorbtpro.generic.nb.patterns import *
from vectorbtpro.generic.nb.records import *
from vectorbtpro.generic.nb.rolling import *
from vectorbtpro.generic.nb.sim_range import *

__all__ = []
