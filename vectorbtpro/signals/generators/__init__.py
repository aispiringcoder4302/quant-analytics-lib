"""Package providing custom signal generators built using the signal factory.

All indicators are accessible via `vbt.*`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.signals.generators.ohlcstcx import *
    from vectorbtpro.signals.generators.ohlcstx import *
    from vectorbtpro.signals.generators.rand import *
    from vectorbtpro.signals.generators.randnx import *
    from vectorbtpro.signals.generators.randx import *
    from vectorbtpro.signals.generators.rprob import *
    from vectorbtpro.signals.generators.rprobcx import *
    from vectorbtpro.signals.generators.rprobnx import *
    from vectorbtpro.signals.generators.rprobx import *
    from vectorbtpro.signals.generators.stcx import *
    from vectorbtpro.signals.generators.stx import *
