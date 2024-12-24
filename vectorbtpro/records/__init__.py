# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Modules for working with records.

Records are the second form of data representation in vectorbtpro. They allow storing sparse event data
such as drawdowns, orders, trades, and positions, without converting them back to the matrix form and
occupying the user's memory."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.records.base import *
    from vectorbtpro.records.chunking import *
    from vectorbtpro.records.col_mapper import *
    from vectorbtpro.records.mapped_array import *
    from vectorbtpro.records.nb import *
