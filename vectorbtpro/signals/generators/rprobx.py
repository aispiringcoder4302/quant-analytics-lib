"""Module providing the `RPROBX` class for generating random exit signals based on probabilities."""

from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import rand_by_prob_place_nb
from vectorbtpro.utils.config import ReadonlyConfig, merge_dicts

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
"""Readonly configuration for the `RPROBX` factory."""

rprobx_func_config = ReadonlyConfig(
    dict(
        exit_place_func_nb=rand_by_prob_place_nb,
        exit_settings=dict(
            pass_params=["prob"],
            pass_kwargs=["pick_first"],
        ),
        param_settings=dict(
            prob=merge_dicts(
                flex_elem_param_config,
                dict(
                    doc="Probability of placing an exit, as a scalar or an array.",
                ),
            ),
        ),
        seed=None,
    )
)
"""Readonly configuration for the exit function of `RPROBX`."""

RPROBX = SignalFactory(**rprobx_config).with_place_func(**rprobx_func_config)


class _RPROBX(RPROBX):
    """Class representing a random exit signal generator based on probabilities.

    See:
        * `RPROBX.run` for the main entry point.
        * `vectorbtpro.signals.nb.rand_by_prob_place_nb` for details on the exit placement.
        * `vectorbtpro.signals.generators.rprob.RPROB` for parameter details.
    """

    pass


RPROBX.clone_docstring(_RPROBX)
