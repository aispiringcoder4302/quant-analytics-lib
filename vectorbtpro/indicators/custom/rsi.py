# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `RSI` class for calculating the Relative Strength Index indicator."""

from vectorbtpro import _typing as tp
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "RSI",
]

__pdoc__ = {}

RSI = IndicatorFactory(
    class_name="RSI",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wtype"],
    output_names=["rsi"],
).with_apply_func(
    nb.rsi_nb,
    kwargs_as_args=["minp", "adjust"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            dtype_kwargs=dict(enum_unkval=None),
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    window=14,
    wtype="wilder",
    minp=None,
    adjust=False,
)


class _RSI(RSI):
    """Class representing the Relative Strength Index (RSI) indicator.

    Represents the Relative Strength Index indicator, which measures the speed
    and change of price movements of a security by comparing the magnitude of
    recent gains and losses over a specified period. It is primarily used to
    identify overbought or oversold conditions.

    See [Relative Strength Index (RSI)](https://www.investopedia.com/terms/r/rsi.asp).

    See:
        * https://www.investopedia.com/terms/r/rsi.asp for the definition of RSI.
        * `vectorbtpro.indicators.nb.rsi_nb` for the underlying implementation.
    """

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        limits: tp.Tuple[float, float] = (30, 70),
        rsi_trace_kwargs: tp.KwargsLike = None,
        add_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the `RSI.rsi` output of the RSI indicator.

        Args:
            column (Optional[Label]): Label of the column to plot.
            limits (Tuple[float, float]): A tuple representing the lower and upper boundaries for the RSI plot.
            rsi_trace_kwargs (KwargsLike): Keyword arguments for
                configuring the `RSI.rsi` trace, passed to `plotly.graph_objects.Scatter`.
            add_shape_kwargs (KwargsLike): Keyword arguments for `fig.add_shape` for each shape.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: A figure object containing the RSI line plot and
                the shaded area between the specified limits.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.RSI.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/RSI.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/RSI.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if rsi_trace_kwargs is None:
            rsi_trace_kwargs = {}
        rsi_trace_kwargs = merge_dicts(
            dict(name="RSI", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            rsi_trace_kwargs,
        )

        fig = self_col.rsi.vbt.lineplot(
            trace_kwargs=rsi_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        xaxis = "xaxis" + xref[1:]
        yaxis = "yaxis" + yref[1:]
        default_layout = dict()
        default_layout[yaxis] = dict(range=[-5, 105])
        fig.update_layout(**default_layout)
        fig.update_layout(**layout_kwargs)

        # Fill void between limits
        add_shape_kwargs = merge_dicts(
            dict(
                type="rect",
                xref=xref,
                yref=yref,
                x0=self_col.wrapper.index[0],
                y0=limits[0],
                x1=self_col.wrapper.index[-1],
                y1=limits[1],
                fillcolor="mediumslateblue",
                opacity=0.2,
                layer="below",
                line_width=0,
            ),
            add_shape_kwargs,
        )
        fig.add_shape(**add_shape_kwargs)

        return fig


setattr(RSI, "__doc__", _RSI.__doc__)
setattr(RSI, "plot", _RSI.plot)
RSI.fix_docstrings(__pdoc__)
