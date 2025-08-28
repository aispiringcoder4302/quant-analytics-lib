# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `GBMTickData` class for generating synthetic tick data."""

from vectorbtpro import _typing as tp
from vectorbtpro.base.accessors import BaseIDXAccessor
from vectorbtpro.data.custom.gbm import GBMData

__all__ = [
    "GBMTickData",
]

__pdoc__ = {}


class GBMTickData(GBMData):
    """Data class for synthetic tick data generation via geometric Brownian motion with randomized timestamps.

    This class extends `vectorbtpro.data.custom.random.GBMData` to generate data with non-uniform
    time intervals between data points, simulating realistic tick data timing.

    See:
        * `GBMTickData.fetch_key` for argument details.

    !!! info
        For default settings, see `custom.gbm_tick` in `vectorbtpro._settings.data`.
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.gbm_tick")

    @classmethod
    def fetch_key(
        cls,
        *args,
        randomness: tp.Optional[float] = None,
        step: tp.Union[None, int, tp.TimedeltaLike] = None,
        inclusive: tp.Optional[str] = None,
        seed: tp.Optional[int] = None,
        **kwargs,
    ) -> tp.KeyData:
        """Generate synthetic tick data for a given key (feature or symbol) with randomized timestamps.

        First calls the parent `fetch_key` method to generate a regular datetime index and data,
        then randomizes the index to create non-uniform time intervals using
        `vectorbtpro.base.accessors.BaseIDXAccessor.randomize`.

        Args:
            args: Positional arguments for the parent `SyntheticData.fetch_key`.
            randomness (float): Factor between 0 and 1 controlling the amount of randomization.

                0 means no change, 1 means full randomization.
            step (Union[int, TimedeltaLike]): Step for the randomization (e.g., '5m' for 5 minutes).

                Displacements will be multiples of this step. Should be integer if the index is
                numeric and timedelta-like if the index is datetime.
            inclusive (str): Which endpoints to keep fixed in the randomization.

                One of "both", "left", "right", or "neither".
            seed (int): GBM seed for deterministic output.

                !!! note
                    When using a seed, pass a unique seed per feature or symbol via
                    `vectorbtpro.data.base.feature_dict`, `vectorbtpro.data.base.symbol_dict`,
                    or generally `vectorbtpro.data.base.key_dict`.
            **kwargs: Keyword arguments for the parent `SyntheticData.fetch_key`.

        Returns:
            KeyData: Fetched data with randomized timestamps and metadata dictionary.
        """
        randomness = cls.resolve_custom_setting(randomness, "randomness")
        step = cls.resolve_custom_setting(step, "step")
        inclusive = cls.resolve_custom_setting(inclusive, "inclusive")
        seed = cls.resolve_custom_setting(seed, "seed")

        data, metadata = super().fetch_key(*args, inclusive="both", seed=seed, **kwargs)
        randomized_index = BaseIDXAccessor(data.index).randomize(
            randomness=randomness,
            step=step,
            inclusive=inclusive,
            seed=seed,
        )
        randomized_data = data.copy()
        randomized_data.index = randomized_index
        return randomized_data, metadata
