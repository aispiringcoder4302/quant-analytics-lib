# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `MACD` class for calculating the Moving Average Convergence Divergence indicator."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_2d_array
from vectorbtpro.generic import nb as generic_nb, enums as generic_enums
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.colors import adjust_opacity
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "MACD",
]

__pdoc__ = {}

MACD = IndicatorFactory(
    class_name="MACD",
    module_name=__name__,
    input_names=["close"],
    param_names=["fast_window", "slow_window", "signal_window", "wtype", "macd_wtype", "signal_wtype"],
    output_names=["macd", "signal"],
    lazy_outputs=dict(
        hist=lambda self: self.wrapper.wrap(
            nb.macd_hist_nb(
                to_2d_array(self.macd),
                to_2d_array(self.signal),
            ),
        ),
    ),
).with_apply_func(
    nb.macd_nb,
    kwargs_as_args=["minp", "macd_minp", "signal_minp", "adjust", "macd_adjust", "signal_adjust"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            dtype_kwargs=dict(enum_unkval=None),
            post_index_func=lambda index: index.str.lower(),
        ),
        macd_wtype=dict(
            dtype=generic_enums.WType,
            dtype_kwargs=dict(enum_unkval=None),
            post_index_func=lambda index: index.str.lower(),
        ),
        signal_wtype=dict(
            dtype=generic_enums.WType,
            dtype_kwargs=dict(enum_unkval=None),
            post_index_func=lambda index: index.str.lower(),
        ),
    ),
    fast_window=12,
    slow_window=26,
    signal_window=9,
    wtype="exp",
    macd_wtype=None,
    signal_wtype=None,
    minp=None,
    macd_minp=None,
    signal_minp=None,
    adjust=False,
    macd_adjust=None,
    signal_adjust=None,
)


class _MACD(MACD):
    """Class representing the Moving Average Convergence Divergence (MACD) indicator.

    This trend-following momentum indicator illustrates the relationship between
    two moving averages of price data.

    See:
        * https://www.investopedia.com/terms/m/macd.asp the definition of MACD.
        * `vectorbtpro.indicators.nb.macd_nb` for the underlying implementation.
        * `vectorbtpro.indicators.nb.macd_hist_nb` for the underlying implementation of `MACD.hist`.
        * `vectorbtpro.indicators.nb.macd_signal_nb` for the underlying implementation of `MACD.signal`.
    """

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        macd_trace_kwargs: tp.KwargsLike = None,
        signal_trace_kwargs: tp.KwargsLike = None,
        hist_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the `MACD.macd`, `MACD.signal`, and `MACD.hist` values of the indicator.

        Args:
            column (Optional[Label]): Name of the column to plot.
            macd_trace_kwargs (KwargsLike): Keyword arguments passed to
                `plotly.graph_objects.Scatter` for plotting the `MACD.macd` line.
            signal_trace_kwargs (KwargsLike): Keyword arguments passed to
                `plotly.graph_objects.Scatter` for plotting the `MACD.signal` line.
            hist_trace_kwargs (KwargsLike): Keyword arguments passed to
                `plotly.graph_objects.Bar` for plotting the `MACD.hist` as a histogram.
            add_trace_kwargs (KwargsLike): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for configuring the figure layout.

        Returns:
            BaseFigure: The updated figure with the MACD indicator plots.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.MACD.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/MACD.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/MACD.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
            fig.update_layout(bargap=0)
        fig.update_layout(**layout_kwargs)

        if macd_trace_kwargs is None:
            macd_trace_kwargs = {}
        if signal_trace_kwargs is None:
            signal_trace_kwargs = {}
        if hist_trace_kwargs is None:
            hist_trace_kwargs = {}
        macd_trace_kwargs = merge_dicts(
            dict(name="MACD", line=dict(color=plotting_cfg["color_schema"]["lightblue"])), macd_trace_kwargs
        )
        signal_trace_kwargs = merge_dicts(
            dict(name="Signal", line=dict(color=plotting_cfg["color_schema"]["lightpurple"])), signal_trace_kwargs
        )

        fig = self_col.macd.vbt.lineplot(
            trace_kwargs=macd_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.signal.vbt.lineplot(
            trace_kwargs=signal_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        # Plot hist
        hist = self_col.hist.values
        hist_diff = generic_nb.diff_1d_nb(hist)
        marker_colors = np.full(hist.shape, adjust_opacity("silver", 0.75), dtype=object)
        marker_colors[(hist > 0) & (hist_diff > 0)] = adjust_opacity("green", 0.75)
        marker_colors[(hist > 0) & (hist_diff <= 0)] = adjust_opacity("lightgreen", 0.75)
        marker_colors[(hist < 0) & (hist_diff < 0)] = adjust_opacity("red", 0.75)
        marker_colors[(hist < 0) & (hist_diff >= 0)] = adjust_opacity("lightcoral", 0.75)

        _hist_trace_kwargs = merge_dicts(
            dict(
                name="Histogram",
                x=self_col.hist.index,
                y=self_col.hist.values,
                marker_color=marker_colors,
                marker_line_width=0,
            ),
            hist_trace_kwargs,
        )
        hist_bar = go.Bar(**_hist_trace_kwargs)
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        fig.add_trace(hist_bar, **add_trace_kwargs)

        return fig


setattr(MACD, "__doc__", _MACD.__doc__)
setattr(MACD, "plot", _MACD.plot)
MACD.fix_docstrings(__pdoc__)
