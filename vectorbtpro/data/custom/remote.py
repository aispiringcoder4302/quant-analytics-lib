"""Module providing the `RemoteData` class for fetching remote data."""

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.custom import CustomData

__all__ = [
    "RemoteData",
]

__pdoc__ = {}


class RemoteData(CustomData):
    """Data class for fetching remote data.

    !!! info
        For default settings, see `custom.remote` in `vectorbtpro._settings.data`.
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.remote")
