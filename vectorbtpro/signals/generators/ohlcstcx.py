# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module with `OHLCSTCX`."""

from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.generators.ohlcstx import ohlcstx_config, ohlcstx_func_config, _bind_ohlcstx_plot

__all__ = [
    "OHLCSTCX",
]

__pdoc__ = {}

OHLCSTCX = SignalFactory(
    **ohlcstx_config.merge_with(
        dict(
            class_name="OHLCSTCX",
            module_name=__name__,
            short_name="ohlcstcx",
            mode="chain",
        )
    ),
).with_place_func(
    **ohlcstx_func_config,
)


class _OHLCSTCX(OHLCSTCX):
    """Exit signal generator based on OHLC and stop values.

    Generates chain of `new_entries` and `exits` based on `entries` and
    `vectorbtpro.signals.nb.ohlc_stop_place_nb`.

    See `OHLCSTX` for notes on parameters."""

    plot = _bind_ohlcstx_plot(OHLCSTCX, "new_entries")


setattr(OHLCSTCX, "__doc__", _OHLCSTCX.__doc__)
setattr(OHLCSTCX, "plot", _OHLCSTCX.plot)
OHLCSTCX.fix_docstrings(__pdoc__)
