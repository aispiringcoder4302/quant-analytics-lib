# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module defining the `FIXLB` generator class for fixed labels."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb

__all__ = [
    "FIXLB",
]

__pdoc__ = {}

FIXLB = IndicatorFactory(
    class_name="FIXLB",
    module_name=__name__,
    input_names=["close"],
    param_names=["n"],
    output_names=["labels"],
).with_apply_func(
    nb.fixed_labels_nb,
    n=1,
)


class _FIXLB(FIXLB):
    """Class representing the look-ahead fixed label generator.

    See:
        * `vectorbtpro.labels.nb.fixed_labels_nb` for the underlying implementation.
    """

    def plot(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:
        """Plot `FIXLB.close` and overlay it with a heatmap of `FIXLB.labels`.

        Args:
            column (Optional[Label]): Column to plot.
            **kwargs: Keyword arguments for `vectorbtpro.generic.accessors.GenericAccessor.overlay_with_heatmap`.

        Returns:
            BaseFigure: The resulting figure object.

        Examples:
            ```pycon
            >>> vbt.FIXLB.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/FIXLB.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/FIXLB.dark.svg#only-dark){: .iimg loading=lazy }
        """
        self_col = self.select_col(column=column, group_by=False)
        return self_col.close.rename("Close").vbt.overlay_with_heatmap(self_col.labels.rename("Labels"), **kwargs)


setattr(FIXLB, "__doc__", _FIXLB.__doc__)
setattr(FIXLB, "plot", _FIXLB.plot)
FIXLB.fix_docstrings(__pdoc__)
