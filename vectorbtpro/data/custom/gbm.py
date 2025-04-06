# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing `GBMData` for generating synthetic geometric Brownian motion data."""

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_1d_array
from vectorbtpro.data import nb
from vectorbtpro.data.custom.synthetic import SyntheticData
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.random_ import set_seed

__all__ = [
    "GBMData",
]

__pdoc__ = {}


class GBMData(SyntheticData):
    """Class for synthetic data generated via geometric Brownian motion using
    `vectorbtpro.data.nb.generate_gbm_data_nb`."""

    _settings_path: tp.SettingsPath = dict(custom="data.custom.gbm")

    @classmethod
    def generate_key(
        cls,
        key: tp.Key,
        index: tp.Index,
        columns: tp.Union[tp.Hashable, tp.IndexLike] = None,
        start_value: tp.Optional[float] = None,
        mean: tp.Optional[float] = None,
        std: tp.Optional[float] = None,
        dt: tp.Optional[float] = None,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        **kwargs,
    ) -> tp.KeyData:
        """Generate synthetic data for a feature or symbol using geometric Brownian motion.

        Args:
            key (Hashable): Feature or symbol identifier.
            index (pd.Index): Pandas index representing time.
            columns (Hashable or IndexLike): Column names.

                Provide a single hashable value to create a Series.
            start_value (float): Initial value at time 0.

                Note that this value does not appear as the first data point.
            mean (float): Drift or mean of the percentage change.
            std (float): Standard deviation of the percentage change.
            dt (float): Time increment for one period.
            seed (int): Seed to ensure deterministic output.
            jitted (JittedOption): Option to control JIT compilation.

        Returns:
            KeyData: The generated data and a metadata dictionary.

        For defaults, refer to `custom.gbm` in `vectorbtpro._settings.data`.

        !!! note
            When setting a seed, pass a seed per feature or symbol using
            `vectorbtpro.data.base.feature_dict`/`vectorbtpro.data.base.symbol_dict` or, more generally,
            `vectorbtpro.data.base.key_dict`.
        """
        if checks.is_hashable(columns):
            columns = [columns]
            make_series = True
        else:
            make_series = False
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        start_value = cls.resolve_custom_setting(start_value, "start_value")
        mean = cls.resolve_custom_setting(mean, "mean")
        std = cls.resolve_custom_setting(std, "std")
        dt = cls.resolve_custom_setting(dt, "dt")
        seed = cls.resolve_custom_setting(seed, "seed")
        if seed is not None:
            set_seed(seed)

        func = jit_reg.resolve_option(nb.generate_gbm_data_nb, jitted)
        out = func(
            (len(index), len(columns)),
            start_value=to_1d_array(start_value),
            mean=to_1d_array(mean),
            std=to_1d_array(std),
            dt=to_1d_array(dt),
        )
        if make_series:
            return pd.Series(out[:, 0], index=index, name=columns[0])
        return pd.DataFrame(out, index=index, columns=columns)

    def update_key(self, key: tp.Key, key_is_feature: bool = False, **kwargs) -> tp.KeyData:
        fetch_kwargs = self.select_fetch_kwargs(key)
        fetch_kwargs["start"] = self.select_last_index(key)
        _ = fetch_kwargs.pop("start_value", None)
        start_value = self.data[key].iloc[-2]
        fetch_kwargs["seed"] = None
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        if key_is_feature:
            return self.fetch_feature(key, start_value=start_value, **kwargs)
        return self.fetch_symbol(key, start_value=start_value, **kwargs)
