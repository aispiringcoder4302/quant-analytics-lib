# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `PIVOTLB` indicator."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb

__all__ = [
    "PIVOTLB",
]

__pdoc__ = {}

PIVOTLB = IndicatorFactory(
    class_name="PIVOTLB",
    module_name=__name__,
    input_names=["high", "low"],
    param_names=["up_th", "down_th"],
    output_names=["labels"],
).with_apply_func(
    nb.pivots_nb,
    param_settings=dict(
        up_th=flex_elem_param_config,
        down_th=flex_elem_param_config,
    ),
)


class _PIVOTLB(PIVOTLB):
    """Label generator using `vectorbtpro.labels.nb.pivots_nb`.

    This class generates labels based on pivot thresholds applied to high and low values.
    """

    def plot(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:
        """Plot the median of `PIVOTLB.high` and `PIVOTLB.low` with a heatmap overlay of `PIVOTLB.labels`.

        Calculates the median of the high and low values and overlays it with a heatmap
        representation of the generated labels.

        Args:
            column (Optional[Label]): Column to select from the indicator output.
            **kwargs: Additional keyword arguments passed to
                `vectorbtpro.generic.accessors.GenericAccessor.overlay_with_heatmap`.

        Returns:
            BaseFigure: A figure object displaying the plot.

        Usage:
            ```pycon
            >>> vbt.PIVOTLB.run(ohlcv['High'], ohlcv['Low'], up_th=0.2, down_th=0.2).plot().show()
            ```

            ![](/assets/images/api/PIVOTLB.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/PIVOTLB.dark.svg#only-dark){: .iimg loading=lazy }
        """
        self_col = self.select_col(column=column, group_by=False)
        median = (self_col.high + self_col.low) / 2
        return median.rename("Median").vbt.overlay_with_heatmap(self_col.labels.rename("Labels"), **kwargs)


setattr(PIVOTLB, "__doc__", _PIVOTLB.__doc__)
setattr(PIVOTLB, "plot", _PIVOTLB.plot)
PIVOTLB.fix_docstrings(__pdoc__)
