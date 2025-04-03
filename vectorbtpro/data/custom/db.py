# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `DBData` class for retrieving database data."""

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.local import LocalData

__all__ = [
    "DBData",
]

__pdoc__ = {}


class DBData(LocalData):
    """Class for retrieving database data.

    This class extends `vectorbtpro.data.custom.local.LocalData` and utilizes custom settings
    specified under the `data.custom.db` path.
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.db")
