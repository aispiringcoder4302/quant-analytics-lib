# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing default data types.

!!! info
    For default settings, see `vectorbtpro._settings.numpy`.
"""

from vectorbtpro._settings import settings

int_ = settings["numpy"]["int_"]
"""Default integer data type retrieved from `vectorbtpro._settings.numpy`."""

float_ = settings["numpy"]["float_"]
"""Default floating point data type retrieved from `vectorbtpro._settings.numpy`."""

__all__ = [
    "int_",
    "float_",
]
