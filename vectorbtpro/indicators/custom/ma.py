# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `MA` class for calculating the moving average."""

from vectorbtpro import _typing as tp
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "MA",
]

__pdoc__ = {}

MA = IndicatorFactory(
    class_name="MA",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wtype"],
    output_names=["ma"],
).with_apply_func(
    nb.ma_nb,
    kwargs_as_args=["minp", "adjust"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            dtype_kwargs=dict(enum_unkval=None),
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    window=14,
    wtype="simple",
    minp=None,
    adjust=False,
)


class _MA(MA):
    """Class representing the Moving Average (MA) indicator.

    A moving average is a popular technical analysis indicator that smooths out price
    fluctuations by filtering out short-term noise from the price data.

    See:
        * https://www.investopedia.com/terms/m/movingaverage.asp for the definition of MA.
        * `vectorbtpro.indicators.nb.ma_nb` for the underlying implementation.
    """

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        ma_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the moving average alongside the close price.

        Args:
            column (Optional[Label]): Name of the column to select for plotting.
            plot_close (bool): Whether to include the close price trace in the plot.
            close_trace_kwargs (KwargsLike): Keyword arguments for the `MA.close` scatter trace.
            ma_trace_kwargs (KwargsLike): Keyword arguments for the `MA.ma` scatter trace.
            add_trace_kwargs (KwargsLike): Keyword arguments for adding a trace using `fig.add_trace`.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for configuring the figure layout.

        Returns:
            BaseFigure: The figure with the plotted moving average and close price traces.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.MA.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/MA.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/MA.dark.svg#only-dark){: .iimg loading=lazy }
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
        if ma_trace_kwargs is None:
            ma_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        ma_trace_kwargs = merge_dicts(
            dict(name="MA", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            ma_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.ma.vbt.lineplot(
            trace_kwargs=ma_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(MA, "__doc__", _MA.__doc__)
setattr(MA, "plot", _MA.plot)
MA.fix_docstrings(__pdoc__)
