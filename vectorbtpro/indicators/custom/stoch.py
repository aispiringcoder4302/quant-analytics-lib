# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `STOCH` indicator."""

from vectorbtpro import _typing as tp
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "STOCH",
]

__pdoc__ = {}

STOCH = IndicatorFactory(
    class_name="STOCH",
    module_name=__name__,
    input_names=["high", "low", "close"],
    param_names=["fast_k_window", "slow_k_window", "slow_d_window", "wtype", "slow_k_wtype", "slow_d_wtype"],
    output_names=["fast_k", "slow_k", "slow_d"],
).with_apply_func(
    nb.stoch_nb,
    kwargs_as_args=["minp", "fast_k_minp", "slow_k_minp", "slow_d_minp", "adjust", "slow_k_adjust", "slow_d_adjust"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            dtype_kwargs=dict(enum_unkval=None),
            post_index_func=lambda index: index.str.lower(),
        ),
        slow_k_wtype=dict(
            dtype=generic_enums.WType,
            dtype_kwargs=dict(enum_unkval=None),
            post_index_func=lambda index: index.str.lower(),
        ),
        slow_d_wtype=dict(
            dtype=generic_enums.WType,
            dtype_kwargs=dict(enum_unkval=None),
            post_index_func=lambda index: index.str.lower(),
        ),
    ),
    fast_k_window=14,
    slow_k_window=3,
    slow_d_window=3,
    wtype="simple",
    slow_k_wtype=None,
    slow_d_wtype=None,
    minp=None,
    fast_k_minp=None,
    slow_k_minp=None,
    slow_d_minp=None,
    adjust=False,
    slow_k_adjust=None,
    slow_d_adjust=None,
)


class _STOCH(STOCH):
    """Stochastic Oscillator (STOCH).

    Implements a stochastic oscillator, a momentum indicator that compares a security's closing
    price to its price range over a specified period. It is used to identify overbought and oversold
    conditions with values typically bounded between 0 and 100.

    See [Investopedia – Stochastic Oscillator](https://www.investopedia.com/terms/s/stochasticoscillator.asp).
    """

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        limits: tp.Tuple[float, float] = (20, 80),
        fast_k_trace_kwargs: tp.KwargsLike = None,
        slow_k_trace_kwargs: tp.KwargsLike = None,
        slow_d_trace_kwargs: tp.KwargsLike = None,
        add_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `STOCH.slow_k` and `STOCH.slow_d`.

        Args:
            column (str): The name of the column to select for plotting.
            limits (Tuple[float, float]): Lower and upper y-axis limits for the filled range.
            fast_k_trace_kwargs (dict): Additional keyword arguments for
                the `plotly.graph_objects.Scatter` trace of `STOCH.fast_k`.
            slow_k_trace_kwargs (dict): Additional keyword arguments for
                the `plotly.graph_objects.Scatter` trace of `STOCH.slow_k`.
            slow_d_trace_kwargs (dict): Additional keyword arguments for
                the `plotly.graph_objects.Scatter` trace of `STOCH.slow_d`.
            add_shape_kwargs (dict): Additional keyword arguments for
                `fig.add_shape` when adding the filled range.
            add_trace_kwargs (dict): Additional keyword arguments for
                `fig.add_trace` when adding each trace.
            fig (Optional[BaseFigure]): The figure to update; if None, a new figure is created.
            **layout_kwargs: Additional keyword arguments for configuring the figure layout.

        Returns:
            BaseFigure: The updated or newly created figure with the plotted STOCH traces and filled range.

        Examples:
            ```pycon
            >>> vbt.STOCH.run(ohlcv['High'], ohlcv['Low'], ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/STOCH.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/STOCH.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fast_k_trace_kwargs is None:
            fast_k_trace_kwargs = {}
        if slow_k_trace_kwargs is None:
            slow_k_trace_kwargs = {}
        if slow_d_trace_kwargs is None:
            slow_d_trace_kwargs = {}
        fast_k_trace_kwargs = merge_dicts(
            dict(name="Fast %K", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            fast_k_trace_kwargs,
        )
        slow_k_trace_kwargs = merge_dicts(
            dict(name="Slow %K", line=dict(color=plotting_cfg["color_schema"]["lightpurple"])),
            slow_k_trace_kwargs,
        )
        slow_d_trace_kwargs = merge_dicts(
            dict(name="Slow %D", line=dict(color=plotting_cfg["color_schema"]["lightpink"])),
            slow_d_trace_kwargs,
        )

        fig = self_col.fast_k.vbt.lineplot(
            trace_kwargs=fast_k_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.slow_k.vbt.lineplot(
            trace_kwargs=slow_k_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.slow_d.vbt.lineplot(
            trace_kwargs=slow_d_trace_kwargs,
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


setattr(STOCH, "__doc__", _STOCH.__doc__)
setattr(STOCH, "plot", _STOCH.plot)
STOCH.fix_docstrings(__pdoc__)
