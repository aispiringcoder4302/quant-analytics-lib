# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `FMAX` indicator."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "FMAX",
]

__pdoc__ = {}

FMAX = IndicatorFactory(
    class_name="FMAX",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wait"],
    output_names=["fmax"],
).with_apply_func(
    nb.future_max_nb,
    window=14,
    wait=1,
)


class _FMAX(FMAX):
    """Class representing a look-ahead indicator that computes the future maximum value using
    `vectorbtpro.labels.nb.future_max_nb`."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        fmax_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `FMAX.fmax` against `FMAX.close`.

        Args:
            column (Optional[Label]): Name of the column to plot.
            plot_close (bool): Flag indicating whether to plot `FMAX.close`.
            close_trace_kwargs (KwargsLike): Keyword arguments passed to
                `plotly.graph_objects.Scatter` for plotting `FMAX.close`.
            fmax_trace_kwargs (KwargsLike): Keyword arguments passed to
                `plotly.graph_objects.Scatter` for plotting `FMAX.fmax`.
            add_trace_kwargs (KwargsLike): Keyword arguments passed to
                `fig.add_trace` for adding each trace.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Additional keyword arguments for configuring the figure layout.

        Returns:
            BaseFigure: The figure with the indicator traces plotted.

        Examples:
            ```pycon
            >>> vbt.FMAX.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/FMAX.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/FMAX.dark.svg#only-dark){: .iimg loading=lazy }
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
        if fmax_trace_kwargs is None:
            fmax_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        fmax_trace_kwargs = merge_dicts(
            dict(name="Future max", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            fmax_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.fmax.vbt.lineplot(
            trace_kwargs=fmax_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(FMAX, "__doc__", _FMAX.__doc__)
setattr(FMAX, "plot", _FMAX.plot)
FMAX.fix_docstrings(__pdoc__)
