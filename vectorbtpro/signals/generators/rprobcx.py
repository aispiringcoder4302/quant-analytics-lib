# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module with `RPROBCX`."""

from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.generators.rprobx import rprobx_config, rprobx_func_config

__all__ = [
    "RPROBCX",
]

__pdoc__ = {}

RPROBCX = SignalFactory(
    **rprobx_config.merge_with(
        dict(
            class_name="RPROBCX",
            short_name="rprobcx",
            mode="chain",
        )
    ),
).with_place_func(**rprobx_func_config)


class _RPROBCX(RPROBCX):
    """Random exit signal generator based on probabilities.

    Generates chain of `new_entries` and `exits` based on `entries` and
    `vectorbtpro.signals.nb.rand_by_prob_place_nb`.

    See `RPROB` for notes on parameters."""

    pass


setattr(RPROBCX, "__doc__", _RPROBCX.__doc__)
RPROBCX.fix_docstrings(__pdoc__)
