"""Module providing the `OHLCSTCX` class for generating stop signals based on OHLC data."""

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
    """Class representing a chained exit signal generator based on OHLC data and stop values.

    Generates a chain of `new_entries` and `exits` derived from input `entries`.

    See:
        * `OHLCSTCX.run` for the main entry point.
        * `vectorbtpro.signals.nb.ohlc_stop_place_nb` for details on the exit placement.
        * `vectorbtpro.signals.generators.ohlcstx.OHLCSTX` for parameter details.
    """

    plot = _bind_ohlcstx_plot(OHLCSTCX, "new_entries")


OHLCSTCX.clone_docstring(_OHLCSTCX)
OHLCSTCX.clone_method(_OHLCSTCX.plot)
