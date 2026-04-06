"""Package providing basic look-ahead indicators and label generators.

All indicators are accessible via `vbt.*`.

Run for the examples:

```pycon
>>> ohlcv = vbt.YFData.pull(
...     "BTC-USD",
...     start="2019-03-01",
...     end="2019-09-01"
... ).get()
```
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.labels.generators.bolb import *
    from vectorbtpro.labels.generators.fixlb import *
    from vectorbtpro.labels.generators.fmax import *
    from vectorbtpro.labels.generators.fmean import *
    from vectorbtpro.labels.generators.fmin import *
    from vectorbtpro.labels.generators.fstd import *
    from vectorbtpro.labels.generators.pivotlb import *
    from vectorbtpro.labels.generators.meanlb import *
    from vectorbtpro.labels.generators.trendlb import *
