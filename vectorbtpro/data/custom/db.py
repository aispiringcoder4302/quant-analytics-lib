"""Module providing the `DBData` class for retrieving database data."""

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.local import LocalData

__all__ = [
    "DBData",
]

__pdoc__ = {}


class DBData(LocalData):
    """Data class for retrieving database data.

    This class extends `vectorbtpro.data.custom.local.LocalData`.

    !!! info
        For default settings, see `custom.db` in `vectorbtpro._settings.data`.
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.db")
