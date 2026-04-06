"""Module providing the `RPROBCX` class for generating random exit signals based on probabilities."""

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
            module_name=__name__,
            short_name="rprobcx",
            mode="chain",
        )
    ),
).with_place_func(**rprobx_func_config)


class _RPROBCX(RPROBCX):
    """Class representing a random exit signal generator based on probabilities.

    Generates a chain of `new_entries` and `exits` derived from input `entries`.

    See:
        * `RPROBCX.run` for the main entry point.
        * `vectorbtpro.signals.nb.rand_by_prob_place_nb` for details on the exit placement.
        * `vectorbtpro.signals.generators.rprob.RPROB` for parameter arguments.
    """

    pass


RPROBCX.clone_docstring(_RPROBCX)
