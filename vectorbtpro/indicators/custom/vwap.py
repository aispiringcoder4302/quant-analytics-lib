# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `VWAP` class for calculating the Volume-Weighted Average Price indicator."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.template import RepFunc

__all__ = [
    "VWAP",
]

__pdoc__ = {}


def substitute_anchor(wrapper: ArrayWrapper, anchor: tp.Optional[tp.FrequencyLike]) -> tp.Array1d:
    """Substitute the reset frequency with group lengths.

    Computes group lengths based on the provided `anchor`. If `anchor` is None, returns an array
    with the number of rows in the wrapper; otherwise, calculates group lengths using the
    wrapper's index grouper.

    Args:
        wrapper (ArrayWrapper): Wrapper instance.
        anchor (Optional[FrequencyLike]): The reset frequency for grouping.

    Returns:
        Array1d: An array containing the group lengths.
    """
    if anchor is None:
        return np.array([wrapper.shape[0]])
    return wrapper.get_index_grouper(anchor).get_group_lens()


VWAP = IndicatorFactory(
    class_name="VWAP",
    module_name=__name__,
    short_name="vwap",
    input_names=["high", "low", "close", "volume"],
    param_names=["anchor"],
    output_names=["vwap"],
).with_apply_func(
    nb.vwap_nb,
    param_settings=dict(
        anchor=dict(template=RepFunc(substitute_anchor)),
    ),
    anchor="D",
)


class _VWAP(VWAP):
    """Class representing the Volume-Weighted Average Price (VWAP) indicator.

    Calculates the volume-weighted average price commonly used in intraday charts.
    The calculation resets at the beginning of each trading session.

    The `anchor` parameter specifies the grouping for when the VWAP resets and can be any valid index grouper.

    See:
        * https://www.investopedia.com/terms/v/vwap.asp for the definition of VWAP.
        * `vectorbtpro.indicators.nb.vwap_nb` for the underlying implementation.
    """

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        vwap_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `VWAP.vwap` against `VWAP.close` values.

        Args:
            column (Optional[Label]): The name of the column to plot.
            plot_close (bool): Whether to include the `VWAP.close` values in the plot.
            close_trace_kwargs (KwargsLike): Keyword arguments for
                `plotly.graph_objects.Scatter` used to plot `VWAP.close`.
            vwap_trace_kwargs (KwargsLike): Keyword arguments for
                `plotly.graph_objects.Scatter` used to plot `VWAP.vwap`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` when adding each trace.
            fig (Optional[BaseFigure]): The figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for configuring the figure layout.
        
        Returns:
            BaseFigure: The updated figure containing the plotted traces.
        
        Examples:
            ```pycon
            >>> vbt.VWAP.run(
            ...    ohlcv['High'],
            ...    ohlcv['Low'],
            ...    ohlcv['Close'],
            ...    ohlcv['Volume'],
            ...    anchor="W"
            ... ).plot().show()
            ```
            
            ![](/assets/images/api/VWAP.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/VWAP.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        if vwap_trace_kwargs is None:
            vwap_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        vwap_trace_kwargs = merge_dicts(
            dict(name="VWAP", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            vwap_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.vwap.vbt.lineplot(
            trace_kwargs=vwap_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(VWAP, "__doc__", _VWAP.__doc__)
setattr(VWAP, "plot", _VWAP.plot)
VWAP.fix_docstrings(__pdoc__)
