# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `STX` signal generator."""

import numpy as np

from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import stop_place_nb
from vectorbtpro.utils.config import ReadonlyConfig

__all__ = [
    "STX",
]

__pdoc__ = {}

stx_config = ReadonlyConfig(
    dict(
        class_name="STX",
        module_name=__name__,
        short_name="stx",
        mode="exits",
        input_names=["entry_ts", "ts", "follow_ts"],
        in_output_names=["stop_ts"],
        param_names=["stop", "trailing"],
    )
)
"""Factory configuration for creating a `STX` signal generator instance."""

stx_func_config = ReadonlyConfig(
    dict(
        exit_place_func_nb=stop_place_nb,
        exit_settings=dict(
            pass_inputs=["entry_ts", "ts", "follow_ts"],
            pass_in_outputs=["stop_ts"],
            pass_params=["stop", "trailing"],
        ),
        param_settings=dict(
            stop=flex_elem_param_config,
            trailing=flex_elem_param_config,
        ),
        trailing=False,
        ts=np.nan,
        follow_ts=np.nan,
        stop_ts=np.nan,
    )
)
"""Configuration for the exit function of the `STX` signal generator, specifying mappings 
for inputs, outputs, and parameters."""

STX = SignalFactory(**stx_config).with_place_func(**stx_func_config)


class _STX(STX):
    """Exit signal generator based on stop values.

    Generates exit signals from entry signals using the `vectorbtpro.signals.nb.stop_place_nb` function.

    !!! hint
        All parameter values can be specified as a single value (per frame) or as a NumPy array
        (per row, column, or element). To generate multiple combinations, pass them as lists.
    """
    pass


setattr(STX, "__doc__", _STX.__doc__)
STX.fix_docstrings(__pdoc__)
