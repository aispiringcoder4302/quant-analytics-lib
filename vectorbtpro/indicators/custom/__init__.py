"""Package providing custom indicators built using the indicator factory.

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
    from vectorbtpro.indicators.custom.adx import *
    from vectorbtpro.indicators.custom.atr import *
    from vectorbtpro.indicators.custom.bbands import *
    from vectorbtpro.indicators.custom.hurst import *
    from vectorbtpro.indicators.custom.ma import *
    from vectorbtpro.indicators.custom.macd import *
    from vectorbtpro.indicators.custom.msd import *
    from vectorbtpro.indicators.custom.obv import *
    from vectorbtpro.indicators.custom.ols import *
    from vectorbtpro.indicators.custom.patsim import *
    from vectorbtpro.indicators.custom.pivotinfo import *
    from vectorbtpro.indicators.custom.rsi import *
    from vectorbtpro.indicators.custom.sigdet import *
    from vectorbtpro.indicators.custom.stoch import *
    from vectorbtpro.indicators.custom.supertrend import *
    from vectorbtpro.indicators.custom.vwap import *
