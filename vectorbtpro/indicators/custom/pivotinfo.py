# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `PIVOTINFO` indicator class for analyzing pivot points."""

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_2d_array
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.indicators.enums import Pivot, TrendMode
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "PIVOTINFO",
]

__pdoc__ = {}

PIVOTINFO = IndicatorFactory(
    class_name="PIVOTINFO",
    module_name=__name__,
    short_name="pivotinfo",
    input_names=["high", "low"],
    param_names=["up_th", "down_th"],
    output_names=["conf_pivot", "conf_idx", "last_pivot", "last_idx"],
    lazy_outputs=dict(
        conf_value=lambda self: self.wrapper.wrap(
            nb.pivot_value_nb(
                to_2d_array(self.high),
                to_2d_array(self.low),
                to_2d_array(self.conf_pivot),
                to_2d_array(self.conf_idx),
            )
        ),
        last_value=lambda self: self.wrapper.wrap(
            nb.pivot_value_nb(
                to_2d_array(self.high),
                to_2d_array(self.low),
                to_2d_array(self.last_pivot),
                to_2d_array(self.last_idx),
            )
        ),
        pivots=lambda self: self.wrapper.wrap(
            nb.pivots_nb(
                to_2d_array(self.conf_pivot),
                to_2d_array(self.conf_idx),
                to_2d_array(self.last_pivot),
            )
        ),
        modes=lambda self: self.wrapper.wrap(
            nb.modes_nb(
                to_2d_array(self.pivots),
            )
        ),
    ),
    attr_settings=dict(
        conf_pivot=dict(dtype=Pivot, enum_unkval=0),
        last_pivot=dict(dtype=Pivot, enum_unkval=0),
        pivots=dict(dtype=Pivot, enum_unkval=0),
        modes=dict(dtype=TrendMode, enum_unkval=0),
    ),
).with_apply_func(
    nb.pivot_info_nb,
    param_settings=dict(
        up_th=flex_elem_param_config,
        down_th=flex_elem_param_config,
    ),
)


class _PIVOTINFO(PIVOTINFO):
    """Class representing the indicator returning various pivot analysis metrics based on predefined thresholds.

    Fields:
        `conf_pivot` (Pivot): Type of the latest confirmed pivot (running).
        `conf_idx` (int): Index of the latest confirmed pivot (running).
        `conf_value` (float): High/low value under the latest confirmed pivot (running).
        `last_pivot` (Pivot): Type of the latest pivot (running).
        `last_idx` (int): Index of the latest pivot (running).
        `last_value` (float): High/low value under the latest pivot (running).
        `pivots` (Pivot): Confirmed pivots stored by their indices
            (looking ahead - use only for plotting).
        `modes` (TrendMode): Trend modes between confirmed pivot points
            (looking ahead - use only for plotting).

    See:
        * `vectorbtpro.indicators.nb.pivot_info_nb` for the underlying implementation.
        * `vectorbtpro.indicators.nb.pivot_value_nb` for the underlying implementation of
            the `PIVOTINFO.conf_value` and `PIVOTINFO.last_value` properties.
        * `vectorbtpro.indicators.nb.pivots_nb` for the underlying implementation of
            the `PIVOTINFO.pivots` property.
        * `vectorbtpro.indicators.nb.modes_nb` for the underlying implementation of
            the `PIVOTINFO.modes` property.
    """

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        conf_value_trace_kwargs: tp.KwargsLike = None,
        last_value_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the confirmed and last pivot value lines on a figure.

        Args:
            column (Optional[Label]): Name of the column to plot.
            conf_value_trace_kwargs (KwargsLike): Keyword arguments passed to
                `plotly.graph_objects.Scatter` for the `PIVOTINFO.conf_value` line.
            last_value_trace_kwargs (KwargsLike): Keyword arguments passed to
                `plotly.graph_objects.Scatter` for the `PIVOTINFO.last_value` line.
            add_trace_kwargs (KwargsLike): Keyword arguments passed to 
                `fig.add_trace` when adding each trace.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for configuring the figure layout.

        Returns:
            BaseFigure: The updated figure with the plotted confirmed and last pivot values.

        Examples:
            ```pycon
            >>> fig = ohlcv.vbt.ohlcv.plot()
            >>> vbt.PIVOTINFO.run(ohlcv['High'], ohlcv['Low'], 0.1, 0.1).plot(fig=fig).show()
            ```

            ![](/assets/images/api/PIVOTINFO.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/PIVOTINFO.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if conf_value_trace_kwargs is None:
            conf_value_trace_kwargs = {}
        if last_value_trace_kwargs is None:
            last_value_trace_kwargs = {}
        conf_value_trace_kwargs = merge_dicts(
            dict(name="Confirmed value", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            conf_value_trace_kwargs,
        )
        last_value_trace_kwargs = merge_dicts(
            dict(name="Last value", line=dict(color=plotting_cfg["color_schema"]["lightpurple"])),
            last_value_trace_kwargs,
        )

        fig = self_col.conf_value.vbt.lineplot(
            trace_kwargs=conf_value_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.last_value.vbt.lineplot(
            trace_kwargs=last_value_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig

    def plot_zigzag(
        self,
        column: tp.Optional[tp.Label] = None,
        zigzag_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the zigzag line based on pivot data.

        Args:
            column (Optional[Label]): Name of the column to plot.

            zigzag_trace_kwargs (KwargsLike): Keyword arguments passed to 
                `plotly.graph_objects.Scatter` for the zigzag line.
            add_trace_kwargs (KwargsLike): Keyword arguments passed to 
                `fig.add_trace` when adding each trace.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for configuring the figure layout.

        Returns:
            BaseFigure: The updated figure with the plotted zigzag line.

        Examples:
            ```pycon
            >>> fig = ohlcv.vbt.ohlcv.plot()
            >>> vbt.PIVOTINFO.run(ohlcv['High'], ohlcv['Low'], 0.1, 0.1).plot_zigzag(fig=fig).show()
            ```

            ![](/assets/images/api/PIVOTINFO_zigzag.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/PIVOTINFO_zigzag.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if zigzag_trace_kwargs is None:
            zigzag_trace_kwargs = {}
        zigzag_trace_kwargs = merge_dicts(
            dict(name="ZigZag", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            zigzag_trace_kwargs,
        )

        pivots = self_col.pivots
        highs = self_col.high[pivots == Pivot.Peak]
        lows = self_col.low[pivots == Pivot.Valley]
        fig = (
            pd.concat((highs, lows))
            .sort_index()
            .vbt.lineplot(
                trace_kwargs=zigzag_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        )

        return fig


setattr(PIVOTINFO, "__doc__", _PIVOTINFO.__doc__)
setattr(PIVOTINFO, "plot", _PIVOTINFO.plot)
setattr(PIVOTINFO, "plot_zigzag", _PIVOTINFO.plot_zigzag)
PIVOTINFO.fix_docstrings(__pdoc__)
