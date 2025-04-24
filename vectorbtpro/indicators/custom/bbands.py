# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `BBANDS` class for Bollinger Bands indicator."""

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_2d_array
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.colors import adjust_opacity
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "BBANDS",
]

__pdoc__ = {}

BBANDS = IndicatorFactory(
    class_name="BBANDS",
    module_name=__name__,
    short_name="bb",
    input_names=["close"],
    param_names=["window", "wtype", "alpha"],
    output_names=["upper", "middle", "lower"],
    lazy_outputs=dict(
        percent_b=lambda self: self.wrapper.wrap(
            nb.bbands_percent_b_nb(
                to_2d_array(self.close),
                to_2d_array(self.upper),
                to_2d_array(self.lower),
            ),
        ),
        bandwidth=lambda self: self.wrapper.wrap(
            nb.bbands_bandwidth_nb(
                to_2d_array(self.upper),
                to_2d_array(self.middle),
                to_2d_array(self.lower),
            ),
        ),
    ),
).with_apply_func(
    nb.bbands_nb,
    kwargs_as_args=["minp", "adjust", "ddof"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            dtype_kwargs=dict(enum_unkval=None),
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    window=14,
    wtype="simple",
    alpha=2,
    minp=None,
    adjust=False,
    ddof=0,
)


class _BBANDS(BBANDS):
    """Class representing the Bollinger Bands (BBANDS) indicator.

    Bollinger Bands is a technical analysis tool that plots three lines:

    * The upper band (calculated a specified number of standard deviations above the middle band).
    * The middle band (a simple moving average of the security's price).
    * The lower band (calculated a specified number of standard deviations below the middle band).

    These bands help identify market volatility and potential overbought or oversold conditions.

    See:
        * https://www.investopedia.com/terms/b/bollingerbands.asp for the definition of BBANDS.
        * `vectorbtpro.indicators.nb.bbands_nb` for the underlying implementation.
        * `vectorbtpro.indicators.nb.bbands_percent_b_nb` for the underlying implementation of `BBANDS.percent_b`.
        * `vectorbtpro.indicators.nb.bbands_bandwidth_nb` for the underlying implementation of `BBANDS.bandwidth`.
    """

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        upper_trace_kwargs: tp.KwargsLike = None,
        middle_trace_kwargs: tp.KwargsLike = None,
        lower_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot the BBANDS traces.

        Return the updated figure with plotted `BBANDS.upper`, `BBANDS.middle`, `BBANDS.lower`,
        and optionally `BBANDS.close`.

        Args:
            column (Optional[Label]): Name of the column to plot.
            plot_close (bool): Flag indicating whether to include `BBANDS.close` in the plot.
            close_trace_kwargs (KwargsLike): Keyword arguments for
                `plotly.graph_objects.Scatter` for `BBANDS.close`.
            upper_trace_kwargs (KwargsLike): Keyword arguments for
                `plotly.graph_objects.Scatter` for `BBANDS.upper`.
            middle_trace_kwargs (KwargsLike): Keyword arguments for
                `plotly.graph_objects.Scatter` for `BBANDS.middle`.
            lower_trace_kwargs (KwargsLike): Keyword arguments for
                `plotly.graph_objects.Scatter` for `BBANDS.lower`.
            add_trace_kwargs (KwargsLike): Keyword arguments for `fig.add_trace` for each trace.
            fig (Optional[BaseFigure]): Figure to update; if None, a new figure is created.
            **layout_kwargs: Keyword arguments for `fig.update_layout`.

        Returns:
            BaseFigure: The updated figure with BBANDS traces.

        !!! info
            For default settings, see `vectorbtpro._settings.plotting`.

        Examples:
            ```pycon
            >>> vbt.BBANDS.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/BBANDS.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/BBANDS.dark.svg#only-dark){: .iimg loading=lazy }
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
        if upper_trace_kwargs is None:
            upper_trace_kwargs = {}
        if middle_trace_kwargs is None:
            middle_trace_kwargs = {}
        if lower_trace_kwargs is None:
            lower_trace_kwargs = {}
        lower_trace_kwargs = merge_dicts(
            dict(
                name="Lower band",
                line=dict(color=adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.5)),
            ),
            lower_trace_kwargs,
        )
        upper_trace_kwargs = merge_dicts(
            dict(
                name="Upper band",
                line=dict(color=adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.5)),
                fill="tonexty",
                fillcolor="rgba(128, 128, 128, 0.2)",
            ),
            upper_trace_kwargs,
        )  # default kwargs
        middle_trace_kwargs = merge_dicts(
            dict(name="Middle band", line=dict(color=plotting_cfg["color_schema"]["lightblue"])), middle_trace_kwargs
        )
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )

        fig = self_col.lower.vbt.lineplot(
            trace_kwargs=lower_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.upper.vbt.lineplot(
            trace_kwargs=upper_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.middle.vbt.lineplot(
            trace_kwargs=middle_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        return fig


setattr(BBANDS, "__doc__", _BBANDS.__doc__)
setattr(BBANDS, "plot", _BBANDS.plot)
BBANDS.fix_docstrings(__pdoc__)
