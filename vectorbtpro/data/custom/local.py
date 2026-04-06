"""Module providing the `LocalData` class for fetching local data."""

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.custom import CustomData

__all__ = [
    "LocalData",
]

__pdoc__ = {}


class LocalData(CustomData):
    """Data class for fetching local data.

    !!! info
        For default settings, see `custom.local` in `vectorbtpro._settings.data`.
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.local")
