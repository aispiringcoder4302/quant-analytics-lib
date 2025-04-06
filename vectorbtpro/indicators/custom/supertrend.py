# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module with `SUPERTREND`."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "SUPERTREND",
]

__pdoc__ = {}

SUPERTREND = IndicatorFactory(
    class_name="SUPERTREND",
    module_name=__name__,
    short_name="supertrend",
    input_names=["high", "low", "close"],
    param_names=["period", "multiplier"],
    output_names=["trend", "direction", "long", "short"],
).with_apply_func(nb.supertrend_nb, period=7, multiplier=3)


class _SUPERTREND(SUPERTREND):
    """Supertrend indicator."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        superl_trace_kwargs: tp.KwargsLike = None,
        supers_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `SUPERTREND.long` and `SUPERTREND.short` against `SUPERTREND.close`.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `SUPERTREND.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `SUPERTREND.close`.
            superl_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `SUPERTREND.long`.
            supers_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `SUPERTREND.short`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for configuring the figure layout.

        Usage:
            ```pycon
            >>> vbt.SUPERTREND.run(ohlcv['High'], ohlcv['Low'], ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/SUPERTREND.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/SUPERTREND.dark.svg#only-dark){: .iimg loading=lazy }
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
        if superl_trace_kwargs is None:
            superl_trace_kwargs = {}
        if supers_trace_kwargs is None:
            supers_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        superl_trace_kwargs = merge_dicts(
            dict(name="Long", line=dict(color=plotting_cfg["color_schema"]["green"])),
            superl_trace_kwargs,
        )
        supers_trace_kwargs = merge_dicts(
            dict(name="Short", line=dict(color=plotting_cfg["color_schema"]["red"])),
            supers_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.long.vbt.lineplot(
            trace_kwargs=superl_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.short.vbt.lineplot(
            trace_kwargs=supers_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(SUPERTREND, "__doc__", _SUPERTREND.__doc__)
setattr(SUPERTREND, "plot", _SUPERTREND.plot)
SUPERTREND.fix_docstrings(__pdoc__)
