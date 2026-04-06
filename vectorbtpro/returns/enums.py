"""Module providing named tuples and enumerated types for representing returns."""

from vectorbtpro import _typing as tp

__pdoc__all__ = __all__ = [
    "RollSharpeAIS",
    "RollSharpeAOS",
]

__pdoc__ = {}


# ############# States ############# #


class RollSharpeAIS(tp.NamedTuple):
    i: int
    ret: float
    pre_window_ret: float
    cumsum: float
    cumsum_sq: float
    nancnt: int
    window: int
    minp: tp.Optional[int]
    ddof: int
    ann_factor: float


__pdoc__[
    "RollSharpeAIS"
] = """Named tuple representing the input state for 
`vectorbtpro.returns.nb.rolling_sharpe_ratio_acc_nb`."""


class RollSharpeAOS(tp.NamedTuple):
    cumsum: float
    cumsum_sq: float
    nancnt: int
    value: float


__pdoc__[
    "RollSharpeAOS"
] = """Named tuple representing the output state for 
`vectorbtpro.returns.nb.rolling_sharpe_ratio_acc_nb`."""
