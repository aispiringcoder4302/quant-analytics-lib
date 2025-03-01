# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module with `RPROBX`."""

from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import rand_by_prob_place_nb
from vectorbtpro.utils.config import ReadonlyConfig

__all__ = [
    "RPROBX",
]

__pdoc__ = {}

rprobx_config = ReadonlyConfig(
    dict(
        class_name="RPROBX",
        module_name=__name__,
        short_name="rprobx",
        mode="exits",
        param_names=["prob"],
    ),
)
"""Factory config for `RPROBX`."""

rprobx_func_config = ReadonlyConfig(
    dict(
        exit_place_func_nb=rand_by_prob_place_nb,
        exit_settings=dict(
            pass_params=["prob"],
            pass_kwargs=["pick_first"],
        ),
        param_settings=dict(
            prob=flex_elem_param_config,
        ),
        seed=None,
    )
)
"""Exit function config for `RPROBX`."""

RPROBX = SignalFactory(**rprobx_config).with_place_func(**rprobx_func_config)


class _RPROBX(RPROBX):
    """Random exit signal generator based on probabilities.

    Generates `exits` based on `entries` and `vectorbtpro.signals.nb.rand_by_prob_place_nb`.

    See `RPROB` for notes on parameters."""

    pass


setattr(RPROBX, "__doc__", _RPROBX.__doc__)
RPROBX.fix_docstrings(__pdoc__)
