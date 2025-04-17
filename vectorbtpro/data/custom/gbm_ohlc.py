# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `GBMOHLCData` class for synthetic OHLC data generation using
geometric Brownian motion simulation."""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import broadcast_array_to
from vectorbtpro.data import nb
from vectorbtpro.data.custom.synthetic import SyntheticData
from vectorbtpro.ohlcv import nb as ohlcv_nb
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.template import substitute_templates

__all__ = [
    "GBMOHLCData",
]

__pdoc__ = {}


class GBMOHLCData(SyntheticData):
    """Class for generating synthetic OHLC data using geometric Brownian motion.

    Uses `vectorbtpro.data.nb.generate_gbm_data_1d_nb` to simulate price ticks and
    `vectorbtpro.ohlcv.nb.ohlc_every_1d_nb` to aggregate ticks into OHLC bars.
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.gbm_ohlc")

    @classmethod
    def generate_symbol(
        cls,
        symbol: tp.Symbol,
        index: tp.Index,
        n_ticks: tp.Optional[tp.ArrayLike] = None,
        start_value: tp.Optional[float] = None,
        mean: tp.Optional[float] = None,
        std: tp.Optional[float] = None,
        dt: tp.Optional[float] = None,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SymbolData:
        """Generate synthetic OHLC data for a given symbol using geometric Brownian motion.

        Args:
            symbol (hashable): The symbol identifier.
            index (pd.Index): The Pandas index representing time periods.
            n_ticks (Optional[ArrayLike]): The number of ticks per bar.

                Can be substituted using a template with a context containing `symbol` and `index`.
            start_value (float): The initial value at time 0.

                Note that this value does not appear as the first value in the resulting data.
            mean (float): The drift representing the mean percentage change.
            std (float): The standard deviation of the percentage change.
            dt (float): The time increment per period.
            seed (Optional[int]): A seed for deterministic output.
            jitted (any): Jitting option; refer to `vectorbtpro.utils.jitting.resolve_jitted_option`.
            template_context (KwargsLike): Additional context for template substitution.

        Returns:
            SymbolData: The generated data and a metadata dictionary.

        For defaults, see `custom.gbm` in `vectorbtpro._settings.data`.

        !!! note
            When setting a seed, provide a seed per symbol using `vectorbtpro.data.base.symbol_dict`.
        """
        n_ticks = cls.resolve_custom_setting(n_ticks, "n_ticks")
        template_context = merge_dicts(dict(symbol=symbol, index=index), template_context)
        n_ticks = substitute_templates(n_ticks, template_context, eval_id="n_ticks")
        n_ticks = broadcast_array_to(n_ticks, len(index))
        start_value = cls.resolve_custom_setting(start_value, "start_value")
        mean = cls.resolve_custom_setting(mean, "mean")
        std = cls.resolve_custom_setting(std, "std")
        dt = cls.resolve_custom_setting(dt, "dt")
        seed = cls.resolve_custom_setting(seed, "seed")
        if seed is not None:
            set_seed(seed)

        func = jit_reg.resolve_option(nb.generate_gbm_data_1d_nb, jitted)
        ticks = func(
            np.sum(n_ticks),
            start_value=start_value,
            mean=mean,
            std=std,
            dt=dt,
        )
        func = jit_reg.resolve_option(ohlcv_nb.ohlc_every_1d_nb, jitted)
        out = func(ticks, n_ticks)
        return pd.DataFrame(out, index=index, columns=["Open", "High", "Low", "Close"])

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        fetch_kwargs["start"] = self.select_last_index(symbol)
        _ = fetch_kwargs.pop("start_value", None)
        start_value = self.data[symbol]["Open"].iloc[-1]
        fetch_kwargs["seed"] = None
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, start_value=start_value, **kwargs)
