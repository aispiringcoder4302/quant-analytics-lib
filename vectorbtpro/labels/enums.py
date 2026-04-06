"""Module providing named tuples and enumerated types for label generation."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.formatting import prettify_doc

__pdoc__all__ = __all__ = ["TrendLabelMode"]

__pdoc__ = {}


class TrendLabelModeT(tp.NamedTuple):
    Binary: int = 0
    BinaryCont: int = 1
    BinaryContSat: int = 2
    PctChange: int = 3
    PctChangeNorm: int = 4


TrendLabelMode = TrendLabelModeT()
"""_"""

__pdoc__[
    "TrendLabelMode"
] = f"""Trend label mode enumeration.

```python
{prettify_doc(TrendLabelMode)}
```

Fields:
    Binary: See `vectorbtpro.labels.nb.bin_trend_labels_nb`.
    BinaryCont: See `vectorbtpro.labels.nb.binc_trend_labels_nb`.
    BinaryContSat: See `vectorbtpro.labels.nb.bincs_trend_labels_nb`.
    PctChange: See `vectorbtpro.labels.nb.pct_trend_labels_nb` with `normalize` set to False.
    PctChangeNorm: See `vectorbtpro.labels.nb.pct_trend_labels_nb` with `normalize` set to True.
"""
