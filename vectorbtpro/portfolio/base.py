# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module with the base class for simulating a portfolio and measuring its performance."""

import inspect
import string
from functools import partial

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base.indexes import ExceptLevel
from vectorbtpro.base.merging import row_stack_arrays
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.base.reshaping import (
    to_1d_array,
    to_2d_array,
    broadcast_array_to,
    to_pd_array,
    to_2d_shape,
)
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.data.base import OHLCDataMixin
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.generic.drawdowns import Drawdowns
from vectorbtpro.generic.sim_range import SimRangeMixin
from vectorbtpro.portfolio import nb, enums
from vectorbtpro.portfolio.decorators import attach_shortcut_properties, attach_returns_acc_methods
from vectorbtpro.portfolio.logs import Logs
from vectorbtpro.portfolio.orders import Orders, FSOrders
from vectorbtpro.portfolio.pfopt.base import PortfolioOptimizer
from vectorbtpro.portfolio.preparing import (
    PFPrepResult,
    BasePFPreparer,
    FOPreparer,
    FSPreparer,
    FOFPreparer,
    FDOFPreparer,
)
from vectorbtpro.portfolio.trades import Trades, EntryTrades, ExitTrades, Positions
from vectorbtpro.records.base import Records
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.returns.accessors import ReturnsAccessor
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import get_dict_attr
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.colors import adjust_opacity
from vectorbtpro.utils.config import resolve_dict, merge_dicts, Config, ReadonlyConfig, HybridConfig, atomic_dict
from vectorbtpro.utils.decorators import custom_property, cached_property, hybrid_method
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.parsing import get_func_kwargs
from vectorbtpro.utils.template import Rep, RepEval, RepFunc
from vectorbtpro.utils.warnings_ import warn

if tp.TYPE_CHECKING:
    from vectorbtpro.returns.qs_adapter import QSAdapter as QSAdapterT
else:
    QSAdapterT = "vectorbtpro.returns.qs_adapter.QSAdapter"

__all__ = [
    "Portfolio",
    "PF",
]

__pdoc__ = {}


def fix_wrapper_for_records(pf: "Portfolio") -> ArrayWrapper:
    """Adjust wrapper flags for records based on the portfolio's cash sharing setting.

    Args:
        pf (Portfolio): The portfolio instance.

    Returns:
        ArrayWrapper: The adjusted array wrapper with updated flags.
    """
    if pf.cash_sharing:
        return pf.wrapper.replace(allow_enable=True, allow_modify=True)
    return pf.wrapper


def records_indexing_func(
    self: "Portfolio",
    obj: tp.RecordArray,
    wrapper_meta: dict,
    cls: tp.Union[type, str],
    groups_only: bool = False,
    **kwargs,
) -> tp.RecordArray:
    """Apply the indexing function to a record array.

    Args:
        obj (RecordArray): The record array to be indexed.
        wrapper_meta (dict): Metadata for the array wrapper configuration.
        cls (Union[type, str]): The record class or its attribute name.
        groups_only (bool): Whether to apply indexing only for groups.
        **kwargs: Additional keyword arguments.

    Returns:
        RecordArray: The resulting record array after indexing.
    """
    wrapper = fix_wrapper_for_records(self)
    if groups_only:
        wrapper = wrapper.resolve()
        wrapper_meta = dict(wrapper_meta)
        wrapper_meta["col_idxs"] = wrapper_meta["group_idxs"]
    if isinstance(cls, str):
        cls = getattr(self, cls)
    records = cls(wrapper, obj)
    records_meta = records.indexing_func_meta(wrapper_meta=wrapper_meta)
    return records.indexing_func(records_meta=records_meta).values


def records_resample_func(
    self: "Portfolio",
    obj: tp.ArrayLike,
    resampler: tp.Union[Resampler, tp.PandasResampler],
    wrapper: ArrayWrapper,
    cls: tp.Union[type, str],
    **kwargs,
) -> tp.RecordArray:
    """Apply the resampling function to a record array.

    Args:
        obj (ArrayLike): The record data array.
        resampler (Union[Resampler, PandasResampler]): The resampler instance used for resampling.
        wrapper (ArrayWrapper): The array wrapper containing index information.
        cls (Union[type, str]): The record class or its attribute name.
        **kwargs: Additional keyword arguments.

    Returns:
        RecordArray: The resampled record array.
    """
    if isinstance(cls, str):
        cls = getattr(self, cls)
    return cls(wrapper, obj).resample(resampler).values


def returns_resample_func(
    self: "Portfolio",
    obj: tp.ArrayLike,
    resampler: tp.Union[Resampler, tp.PandasResampler],
    wrapper: ArrayWrapper,
    fill_with_zero: bool = True,
    log_returns: bool = False,
    **kwargs,
):
    """Apply the resampling function to returns data.

    Args:
        obj (ArrayLike): The returns data array.
        resampler (Union[Resampler, PandasResampler]): The resampler instance used for resampling.
        wrapper (ArrayWrapper): The array wrapper providing the index.
        fill_with_zero (bool): Whether to fill missing values with zero.
        log_returns (bool): Whether to compute logarithmic returns.
        **kwargs: Additional keyword arguments.

    Returns:
        ArrayLike: The resampled returns array.
    """
    return (
        pd.DataFrame(obj, index=wrapper.index)
        .vbt.returns(log_returns=log_returns)
        .resample(
            resampler,
            fill_with_zero=fill_with_zero,
        )
        .obj.values
    )


returns_acc_config = ReadonlyConfig(
    {
        "daily_returns": dict(source_name="daily"),
        "annual_returns": dict(source_name="annual"),
        "cumulative_returns": dict(source_name="cumulative"),
        "annualized_return": dict(source_name="annualized"),
        "annualized_volatility": dict(),
        "calmar_ratio": dict(),
        "omega_ratio": dict(),
        "sharpe_ratio": dict(),
        "sharpe_ratio_std": dict(),
        "prob_sharpe_ratio": dict(),
        "deflated_sharpe_ratio": dict(),
        "downside_risk": dict(),
        "sortino_ratio": dict(),
        "information_ratio": dict(),
        "beta": dict(),
        "alpha": dict(),
        "tail_ratio": dict(),
        "value_at_risk": dict(),
        "cond_value_at_risk": dict(),
        "capture_ratio": dict(),
        "up_capture_ratio": dict(),
        "down_capture_ratio": dict(),
        "drawdown": dict(),
        "max_drawdown": dict(),
    }
)
"""_"""

__pdoc__[
    "returns_acc_config"
] = f"""Configuration for returns accessor methods attached to `Portfolio`.

```python
{returns_acc_config.prettify_doc()}
```
"""

shortcut_config = ReadonlyConfig(
    {
        "filled_close": dict(group_by_aware=False, decorator=cached_property),
        "filled_bm_close": dict(group_by_aware=False, decorator=cached_property),
        "weights": dict(group_by_aware=False, decorator=cached_property, obj_type="red_array"),
        "long_view": dict(obj_type="portfolio"),
        "short_view": dict(obj_type="portfolio"),
        "orders": dict(
            obj_type="records",
            field_aliases=("order_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.orders_cls.from_records(
                fix_wrapper_for_records(pf),
                obj,
                open=pf.open_flex,
                high=pf.high_flex,
                low=pf.low_flex,
                close=pf.close_flex,
            ),
            indexing_func=partial(records_indexing_func, cls="orders_cls"),
            resample_func=partial(records_resample_func, cls="orders_cls"),
        ),
        "logs": dict(
            obj_type="records",
            field_aliases=("log_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.logs_cls.from_records(
                fix_wrapper_for_records(pf),
                obj,
                open=pf.open_flex,
                high=pf.high_flex,
                low=pf.low_flex,
                close=pf.close_flex,
            ),
            indexing_func=partial(records_indexing_func, cls="logs_cls"),
            resample_func=partial(records_resample_func, cls="logs_cls"),
        ),
        "entry_trades": dict(
            obj_type="records",
            field_aliases=("entry_trade_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.entry_trades_cls.from_records(
                fix_wrapper_for_records(pf),
                obj,
                open=pf.open_flex,
                high=pf.high_flex,
                low=pf.low_flex,
                close=pf.close_flex,
            ),
            indexing_func=partial(records_indexing_func, cls="entry_trades_cls"),
            resample_func=partial(records_resample_func, cls="entry_trades_cls"),
        ),
        "exit_trades": dict(
            obj_type="records",
            field_aliases=("exit_trade_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.exit_trades_cls.from_records(
                fix_wrapper_for_records(pf),
                obj,
                open=pf.open_flex,
                high=pf.high_flex,
                low=pf.low_flex,
                close=pf.close_flex,
            ),
            indexing_func=partial(records_indexing_func, cls="exit_trades_cls"),
            resample_func=partial(records_resample_func, cls="exit_trades_cls"),
        ),
        "trades": dict(
            obj_type="records",
            field_aliases=("trade_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.trades_cls.from_records(
                fix_wrapper_for_records(pf),
                obj,
                open=pf.open_flex,
                high=pf.high_flex,
                low=pf.low_flex,
                close=pf.close_flex,
            ),
            indexing_func=partial(records_indexing_func, cls="trades_cls"),
            resample_func=partial(records_resample_func, cls="trades_cls"),
        ),
        "trade_history": dict(),
        "signals": dict(),
        "positions": dict(
            obj_type="records",
            field_aliases=("position_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.positions_cls.from_records(
                fix_wrapper_for_records(pf),
                obj,
                open=pf.open_flex,
                high=pf.high_flex,
                low=pf.low_flex,
                close=pf.close_flex,
            ),
            indexing_func=partial(records_indexing_func, cls="positions_cls"),
            resample_func=partial(records_resample_func, cls="positions_cls"),
        ),
        "drawdowns": dict(
            obj_type="records",
            field_aliases=("drawdown_records",),
            wrap_func=lambda pf, obj, **kwargs: pf.drawdowns_cls.from_records(
                fix_wrapper_for_records(pf).resolve(),
                obj,
            ),
            indexing_func=partial(records_indexing_func, cls="drawdowns_cls", groups_only=True),
            resample_func=partial(records_resample_func, cls="drawdowns_cls"),
        ),
        "init_position": dict(obj_type="red_array", group_by_aware=False),
        "asset_flow": dict(
            group_by_aware=False,
            resample_func="sum",
            resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0)),
        ),
        "long_asset_flow": dict(
            method_name="get_asset_flow",
            group_by_aware=False,
            method_kwargs=dict(direction="longonly"),
            resample_func="sum",
            resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0)),
        ),
        "short_asset_flow": dict(
            method_name="get_asset_flow",
            group_by_aware=False,
            method_kwargs=dict(direction="shortonly"),
            resample_func="sum",
            resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0)),
        ),
        "assets": dict(group_by_aware=False),
        "long_assets": dict(
            method_name="get_assets",
            group_by_aware=False,
            method_kwargs=dict(direction="longonly"),
        ),
        "short_assets": dict(
            method_name="get_assets",
            group_by_aware=False,
            method_kwargs=dict(direction="shortonly"),
        ),
        "position_mask": dict(),
        "long_position_mask": dict(method_name="get_position_mask", method_kwargs=dict(direction="longonly")),
        "short_position_mask": dict(method_name="get_position_mask", method_kwargs=dict(direction="shortonly")),
        "position_coverage": dict(obj_type="red_array"),
        "long_position_coverage": dict(
            method_name="get_position_coverage",
            obj_type="red_array",
            method_kwargs=dict(direction="longonly"),
        ),
        "short_position_coverage": dict(
            method_name="get_position_coverage",
            obj_type="red_array",
            method_kwargs=dict(direction="shortonly"),
        ),
        "position_entry_price": dict(group_by_aware=False),
        "position_exit_price": dict(group_by_aware=False),
        "init_cash": dict(obj_type="red_array"),
        "cash_deposits": dict(resample_func="sum", resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0))),
        "total_cash_deposits": dict(obj_type="red_array"),
        "cash_earnings": dict(resample_func="sum", resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0))),
        "total_cash_earnings": dict(obj_type="red_array"),
        "cash_flow": dict(resample_func="sum", resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0))),
        "free_cash_flow": dict(
            method_name="get_cash_flow",
            method_kwargs=dict(free=True),
            resample_func="sum",
            resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0)),
        ),
        "cash": dict(),
        "position": dict(method_name="get_assets", group_by_aware=False),
        "debt": dict(method_name=None, group_by_aware=False),
        "locked_cash": dict(method_name=None, group_by_aware=False),
        "free_cash": dict(method_name="get_cash", method_kwargs=dict(free=True)),
        "init_price": dict(obj_type="red_array", group_by_aware=False),
        "init_position_value": dict(obj_type="red_array"),
        "init_value": dict(obj_type="red_array"),
        "input_value": dict(obj_type="red_array"),
        "asset_value": dict(),
        "long_asset_value": dict(method_name="get_asset_value", method_kwargs=dict(direction="longonly")),
        "short_asset_value": dict(method_name="get_asset_value", method_kwargs=dict(direction="shortonly")),
        "gross_exposure": dict(),
        "long_gross_exposure": dict(method_name="get_gross_exposure", method_kwargs=dict(direction="longonly")),
        "short_gross_exposure": dict(method_name="get_gross_exposure", method_kwargs=dict(direction="shortonly")),
        "net_exposure": dict(),
        "value": dict(),
        "allocations": dict(group_by_aware=False),
        "long_allocations": dict(
            method_name="get_allocations",
            method_kwargs=dict(direction="longonly"),
            group_by_aware=False,
        ),
        "short_allocations": dict(
            method_name="get_allocations",
            method_kwargs=dict(direction="shortonly"),
            group_by_aware=False,
        ),
        "total_profit": dict(obj_type="red_array"),
        "final_value": dict(obj_type="red_array"),
        "total_return": dict(obj_type="red_array"),
        "returns": dict(resample_func=returns_resample_func),
        "log_returns": dict(
            method_name="get_returns",
            method_kwargs=dict(log_returns=True),
            resample_func=partial(returns_resample_func, log_returns=True),
        ),
        "daily_log_returns": dict(
            method_name="get_returns",
            method_kwargs=dict(daily_returns=True, log_returns=True),
            resample_func=partial(returns_resample_func, log_returns=True),
        ),
        "asset_pnl": dict(resample_func="sum", resample_kwargs=dict(wrap_kwargs=dict(fillna=0.0))),
        "asset_returns": dict(resample_func=returns_resample_func),
        "market_value": dict(),
        "market_returns": dict(resample_func=returns_resample_func),
        "bm_value": dict(),
        "bm_returns": dict(resample_func=returns_resample_func),
        "total_market_return": dict(obj_type="red_array"),
        "daily_returns": dict(resample_func=returns_resample_func),
        "annual_returns": dict(resample_func=returns_resample_func),
        "cumulative_returns": dict(),
        "annualized_return": dict(obj_type="red_array"),
        "annualized_volatility": dict(obj_type="red_array"),
        "calmar_ratio": dict(obj_type="red_array"),
        "omega_ratio": dict(obj_type="red_array"),
        "sharpe_ratio": dict(obj_type="red_array"),
        "sharpe_ratio_std": dict(obj_type="red_array"),
        "prob_sharpe_ratio": dict(obj_type="red_array"),
        "deflated_sharpe_ratio": dict(obj_type="red_array"),
        "downside_risk": dict(obj_type="red_array"),
        "sortino_ratio": dict(obj_type="red_array"),
        "information_ratio": dict(obj_type="red_array"),
        "beta": dict(obj_type="red_array"),
        "alpha": dict(obj_type="red_array"),
        "tail_ratio": dict(obj_type="red_array"),
        "value_at_risk": dict(obj_type="red_array"),
        "cond_value_at_risk": dict(obj_type="red_array"),
        "capture_ratio": dict(obj_type="red_array"),
        "up_capture_ratio": dict(obj_type="red_array"),
        "down_capture_ratio": dict(obj_type="red_array"),
        "drawdown": dict(),
        "max_drawdown": dict(obj_type="red_array"),
    }
)
"""_"""

__pdoc__[
    "shortcut_config"
] = f"""Configuration for shortcut properties attached to `Portfolio`.

```python
{shortcut_config.prettify_doc()}
```
"""

PortfolioT = tp.TypeVar("PortfolioT", bound="Portfolio")
PortfolioResultT = tp.Union[PortfolioT, BasePFPreparer, PFPrepResult, enums.SimulationOutput]


class MetaPortfolio(type(Analyzable)):
    """Metaclass for `Portfolio`.

    This metaclass defines type-level configuration and behavior for portfolio classes.
    """

    @property
    def in_output_config(cls) -> Config:
        """In-output configuration settings.
        
        Returns:
            Config: The configuration settings for in-outputs.
        """
        return cls._in_output_config


@attach_shortcut_properties(shortcut_config)
@attach_returns_acc_methods(returns_acc_config)
class Portfolio(Analyzable, SimRangeMixin, metaclass=MetaPortfolio):
    """Class for simulating a portfolio and measuring its performance.

    Args:
        wrapper (ArrayWrapper): Wrapper instance.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        order_records (Union[RecordArray, SimulationOutput]): Structured NumPy array of order records, or 
            a `vectorbtpro.portfolio.enums.SimulationOutput` object.
        close (ArrayLike): Last asset price at each time step.

            Provided in a format that supports flexible indexing.
        open (Optional[ArrayLike]): Opening price of each time step.

            Provided in a format that supports flexible indexing.
        high (Optional[ArrayLike]): Highest price of each time step.

            Provided in a format that supports flexible indexing.
        low (Optional[ArrayLike]): Lowest price of each time step.

            Provided in a format that supports flexible indexing.
        log_records (Optional[RecordArray]): Structured NumPy array of log records.
        cash_sharing (bool): Determines whether cash is shared among assets in the same group.
        init_cash (Union[str, ArrayLike]): Initial capital.

            Provided in a format that supports flexible indexing.

            See `vectorbtpro.portfolio.enums.InitCashMode` for more options.
            
            !!! note
                When using `InitCashMode.AutoAlign`, initial cash values are synchronized 
                across columns/groups after initialization.
        init_position (ArrayLike): Initial asset positions.

            Provided in a format that supports flexible indexing.
        init_price (ArrayLike): Initial position prices.

            Provided in a format that supports flexible indexing.
        cash_deposits (ArrayLike): Cash deposited or withdrawn at each timestamp.

            Provided in a format that supports flexible indexing.
        cash_earnings (ArrayLike): Earnings added at each timestamp.

            Provided in a format that supports flexible indexing.
        sim_start (Optional[ArrayLike]): Simulation start per column.
        sim_end (Optional[ArrayLike]): Simulation end per column.
        call_seq (Optional[Array2d]): Sequence of calls per row and column.
        in_outputs (Optional[NamedTuple]): Named tuple containing in-output objects.

            Provide pre-broadcasted and grouped objects to substitute default `Portfolio` attributes.
            See `Portfolio.in_outputs_indexing_func` for indexing details.
        use_in_outputs (Optional[bool]): Indicates whether to return in-output objects when accessing properties.
        bm_close (Optional[ArrayLike]): Last benchmark asset price at each time step.

            Provided in a format that supports flexible indexing.
        fillna_close (Optional[bool]): Whether to forward-backward fill NaN values in `Portfolio.close`.

            Filling is applied post-simulation to avoid NaNs in asset values.
            See `Portfolio.get_filled_close`.
        year_freq (Optional[FrequencyLike]): Year frequency derived from the portfolio's index.
        returns_acc_defaults (KwargsLike): Defaults for `vectorbtpro.returns.accessors.ReturnsAccessor`.
        trades_type (Optional[Union[str, int]]): Default trades type for `Portfolio`.

            See `vectorbtpro.portfolio.enums.TradesType` for details.
        orders_cls (Optional[Type]): Class used for wrapping order records.
        logs_cls (Optional[Type]): Class used for wrapping log records.
        trades_cls (Optional[Type]): Class used for wrapping trade records.
        entry_trades_cls (Optional[Type]): Class used for wrapping entry trade records.
        exit_trades_cls (Optional[Type]): Class used for wrapping exit trade records.
        positions_cls (Optional[Type]): Class used for wrapping position records.
        drawdowns_cls (Optional[Type]): Class used for wrapping drawdown records.
        weights (Union[None, bool, ArrayLike]): Asset weights.

            Applied to initial positions, cash, deposits, earnings, and orders.
        **kwargs: Additional keyword arguments for configuration.

    For defaults, see `vectorbtpro._settings.portfolio`.

    !!! note
        Use class methods with the `from_` prefix to build a portfolio.
        The `__init__` method is reserved for indexing purposes.

    !!! note
        This class is immutable. To change any attribute, use `Portfolio.replace`.
    """
    
    _writeable_attrs: tp.WriteableAttrs = {"_in_output_config"}

    def __init__(
        self,
        wrapper: ArrayWrapper,
        order_records: tp.Union[tp.RecordArray, enums.SimulationOutput],
        *,
        close: tp.ArrayLike,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        log_records: tp.Optional[tp.RecordArray] = None,
        cash_sharing: bool = False,
        init_cash: tp.Union[str, tp.ArrayLike] = "auto",
        init_position: tp.ArrayLike = 0.0,
        init_price: tp.ArrayLike = np.nan,
        cash_deposits: tp.ArrayLike = 0.0,
        cash_deposits_as_input: tp.Optional[bool] = None,
        cash_earnings: tp.ArrayLike = 0.0,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        call_seq: tp.Optional[tp.Array2d] = None,
        in_outputs: tp.Optional[tp.NamedTuple] = None,
        use_in_outputs: tp.Optional[bool] = None,
        bm_close: tp.Optional[tp.ArrayLike] = None,
        fillna_close: tp.Optional[bool] = None,
        year_freq: tp.Optional[tp.FrequencyLike] = None,
        returns_acc_defaults: tp.KwargsLike = None,
        trades_type: tp.Optional[tp.Union[str, int]] = None,
        orders_cls: tp.Optional[type] = None,
        logs_cls: tp.Optional[type] = None,
        trades_cls: tp.Optional[type] = None,
        entry_trades_cls: tp.Optional[type] = None,
        exit_trades_cls: tp.Optional[type] = None,
        positions_cls: tp.Optional[type] = None,
        drawdowns_cls: tp.Optional[type] = None,
        weights: tp.Union[None, bool, tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        from vectorbtpro._settings import settings

        portfolio_cfg = settings["portfolio"]

        if cash_sharing:
            if wrapper.grouper.allow_enable or wrapper.grouper.allow_modify:
                wrapper = wrapper.replace(allow_enable=False, allow_modify=False)
        if isinstance(order_records, enums.SimulationOutput):
            sim_out = order_records
            order_records = sim_out.order_records
            log_records = sim_out.log_records
            cash_deposits = sim_out.cash_deposits
            cash_earnings = sim_out.cash_earnings
            sim_start = sim_out.sim_start
            sim_end = sim_out.sim_end
            call_seq = sim_out.call_seq
            in_outputs = sim_out.in_outputs
        close = to_2d_array(close)
        if open is not None:
            open = to_2d_array(open)
        if high is not None:
            high = to_2d_array(high)
        if low is not None:
            low = to_2d_array(low)
        if isinstance(init_cash, str):
            init_cash = map_enum_fields(init_cash, enums.InitCashMode)
        if not checks.is_int(init_cash) or init_cash not in enums.InitCashMode:
            init_cash = to_1d_array(init_cash)
        init_position = to_1d_array(init_position)
        init_price = to_1d_array(init_price)
        cash_deposits = to_2d_array(cash_deposits)
        cash_earnings = to_2d_array(cash_earnings)
        if cash_deposits_as_input is None:
            cash_deposits_as_input = portfolio_cfg["cash_deposits_as_input"]
        if bm_close is not None and not isinstance(bm_close, bool):
            bm_close = to_2d_array(bm_close)
        if log_records is None:
            log_records = np.array([], dtype=enums.log_dt)
        if use_in_outputs is None:
            use_in_outputs = portfolio_cfg["use_in_outputs"]
        if fillna_close is None:
            fillna_close = portfolio_cfg["fillna_close"]
        if weights is None:
            weights = portfolio_cfg["weights"]
        if trades_type is None:
            trades_type = portfolio_cfg["trades_type"]
        if isinstance(trades_type, str):
            trades_type = map_enum_fields(trades_type, enums.TradesType)

        Analyzable.__init__(
            self,
            wrapper,
            order_records=order_records,
            open=open,
            high=high,
            low=low,
            close=close,
            log_records=log_records,
            cash_sharing=cash_sharing,
            init_cash=init_cash,
            init_position=init_position,
            init_price=init_price,
            cash_deposits=cash_deposits,
            cash_deposits_as_input=cash_deposits_as_input,
            cash_earnings=cash_earnings,
            sim_start=sim_start,
            sim_end=sim_end,
            call_seq=call_seq,
            in_outputs=in_outputs,
            use_in_outputs=use_in_outputs,
            bm_close=bm_close,
            fillna_close=fillna_close,
            year_freq=year_freq,
            returns_acc_defaults=returns_acc_defaults,
            trades_type=trades_type,
            orders_cls=orders_cls,
            logs_cls=logs_cls,
            trades_cls=trades_cls,
            entry_trades_cls=entry_trades_cls,
            exit_trades_cls=exit_trades_cls,
            positions_cls=positions_cls,
            drawdowns_cls=drawdowns_cls,
            weights=weights,
            **kwargs,
        )
        SimRangeMixin.__init__(self, sim_start=sim_start, sim_end=sim_end)

        self._open = open
        self._high = high
        self._low = low
        self._close = close
        self._order_records = order_records
        self._log_records = log_records
        self._cash_sharing = cash_sharing
        self._init_cash = init_cash
        self._init_position = init_position
        self._init_price = init_price
        self._cash_deposits = cash_deposits
        self._cash_deposits_as_input = cash_deposits_as_input
        self._cash_earnings = cash_earnings
        self._call_seq = call_seq
        self._in_outputs = in_outputs
        self._use_in_outputs = use_in_outputs
        self._bm_close = bm_close
        self._fillna_close = fillna_close
        self._year_freq = year_freq
        self._returns_acc_defaults = returns_acc_defaults
        self._trades_type = trades_type
        self._orders_cls = orders_cls
        self._logs_cls = logs_cls
        self._trades_cls = trades_cls
        self._entry_trades_cls = entry_trades_cls
        self._exit_trades_cls = exit_trades_cls
        self._positions_cls = positions_cls
        self._drawdowns_cls = drawdowns_cls
        self._weights = weights

        # Only slices of rows can be selected
        self._range_only_select = True

        # Copy writeable attrs
        self._in_output_config = type(self)._in_output_config.copy()

    @classmethod
    def row_stack_objs(
        cls: tp.Type[PortfolioT],
        objs: tp.Sequence[tp.Any],
        wrappers: tp.Sequence[ArrayWrapper],
        grouping: str = "columns_or_groups",
        obj_name: tp.Optional[str] = None,
        obj_type: tp.Optional[str] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        cash_sharing: bool = False,
        row_stack_func: tp.Optional[tp.Callable] = None,
        **kwargs,
    ) -> tp.Any:
        """Stack two-dimensional objects along rows.

        This method stacks objects row-wise using either a custom stacking function or an internal
        logic based on grouping and object type. If all objects are None, boolean, or empty arrays,
        the first object is returned.

        Args:
            objs (Sequence[Any]): Sequence of objects to be stacked.
            wrappers (Sequence[ArrayWrapper]): Sequence of wrapper instances associated with the objects.
            grouping (str): Strategy for grouping objects. 
            
                Supported options include "columns_or_groups", "columns", "groups", and "cash_sharing".
            obj_name (Optional[str]): Identifier for the object.
            obj_type (Optional[str]): Expected type of the object, which influences the stacking logic.
            wrapper (Optional[ArrayWrapper]): Target array wrapper used for performing the stacking.
            cash_sharing (bool): Flag indicating whether cash sharing is enabled.
            row_stack_func (Optional[Callable]): Custom function for row stacking that must accept the 
                portfolio class and all provided arguments. 
                
                If unused, it should accept additional arguments via `**kwargs`.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The stacked result, or the first object if all inputs are None, boolean, or empty.
        """
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        all_none = True
        for obj in objs:
            if obj is None or isinstance(obj, bool) or (checks.is_np_array(obj) and obj.size == 0):
                if not checks.is_deep_equal(obj, objs[0]):
                    raise ValueError(f"Cannot unify scalar in-outputs with the name '{obj_name}'")
            else:
                all_none = False
                break
        if all_none:
            return objs[0]

        if row_stack_func is not None:
            return row_stack_func(
                cls,
                objs,
                wrappers,
                grouping=grouping,
                obj_name=obj_name,
                obj_type=obj_type,
                wrapper=wrapper,
                **kwargs,
            )

        if grouping == "columns_or_groups":
            obj_group_by = None
        elif grouping == "columns":
            obj_group_by = False
        elif grouping == "groups":
            obj_group_by = None
        elif grouping == "cash_sharing":
            obj_group_by = None if cash_sharing else False
        else:
            raise ValueError(f"Grouping '{grouping}' is not supported")

        if obj_type is None and checks.is_np_array(objs[0]):
            n_cols = wrapper.get_shape_2d(group_by=obj_group_by)[1]
            can_stack = (objs[0].ndim == 1 and n_cols == 1) or (objs[0].ndim == 2 and objs[0].shape[1] == n_cols)
        elif obj_type is not None and obj_type == "array":
            can_stack = True
        else:
            can_stack = False
        if can_stack:
            wrapped_objs = []
            for i, obj in enumerate(objs):
                wrapped_objs.append(wrappers[i].wrap(obj, group_by=obj_group_by))
            return wrapper.row_stack_arrs(*wrapped_objs, group_by=obj_group_by, wrap=False)
        raise ValueError(f"Cannot figure out how to stack in-outputs with the name '{obj_name}' along rows")

    @classmethod
    def row_stack_in_outputs(
        cls: tp.Type[PortfolioT],
        *objs: tp.MaybeTuple[PortfolioT],
        **kwargs,
    ) -> tp.Optional[tp.NamedTuple]:
        """Stack `Portfolio.in_outputs` fields along rows.

        This method stacks the `in_outputs` named tuples from multiple `Portfolio` instances row-wise.
        All `in_outputs` must either be None or contain identical fields. For each field, stacking options 
        are determined by merging options from `Portfolio.parse_field_options` and 
        `Portfolio.in_output_config`. The stacking is then performed on each field using 
        `Portfolio.row_stack_objs`.

        Args:
            *objs (Portfolio): One or more `Portfolio` objects whose `in_outputs` attributes are to be stacked.
            **kwargs: Additional keyword arguments passed to the stacking function.

        Returns:
            Optional[NamedTuple]: A new named tuple with each field stacked row-wise, or 
                None if all `in_outputs` are None.
        """
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        all_none = True
        for obj in objs:
            if obj.in_outputs is not None:
                all_none = False
                break
        if all_none:
            return None
        all_keys = set()
        for obj in objs:
            all_keys |= set(obj.in_outputs._asdict().keys())
        for obj in objs:
            if obj.in_outputs is None or len(all_keys.difference(set(obj.in_outputs._asdict().keys()))) > 0:
                raise ValueError("Objects to be merged must have the same in-output fields")

        cls_dir = set(dir(cls))
        new_in_outputs = {}
        for field in objs[0].in_outputs._asdict().keys():
            field_options = merge_dicts(
                cls.parse_field_options(field),
                cls.in_output_config.get(field, None),
            )
            if field_options.get("field", field) in cls_dir:
                prop = getattr(cls, field_options["field"])
                prop_options = getattr(prop, "options", {})
                obj_type = prop_options.get("obj_type", "array")
                group_by_aware = prop_options.get("group_by_aware", True)
                row_stack_func = prop_options.get("row_stack_func", None)
            else:
                obj_type = None
                group_by_aware = True
                row_stack_func = None
            _kwargs = merge_dicts(
                dict(
                    grouping=field_options.get("grouping", "columns_or_groups" if group_by_aware else "columns"),
                    obj_name=field_options.get("field", field),
                    obj_type=field_options.get("obj_type", obj_type),
                    row_stack_func=field_options.get("row_stack_func", row_stack_func),
                ),
                kwargs,
            )
            new_field_obj = cls.row_stack_objs(
                [getattr(obj.in_outputs, field) for obj in objs],
                [obj.wrapper for obj in objs],
                **_kwargs,
            )
            new_in_outputs[field] = new_field_obj

        return type(objs[0].in_outputs)(**new_in_outputs)

    @hybrid_method
    def row_stack(
        cls_or_self: tp.MaybeType[PortfolioT],
        *objs: tp.MaybeTuple[PortfolioT],
        wrapper_kwargs: tp.KwargsLike = None,
        group_by: tp.GroupByLike = None,
        combine_init_cash: bool = False,
        combine_init_position: bool = False,
        combine_init_price: bool = False,
        **kwargs,
    ) -> PortfolioT:
        """Stack multiple `Portfolio` instances along rows.

        Stacks multiple `Portfolio` instances by merging their underlying data arrays along rows.
        This method leverages `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to combine wrappers 
        and applies specialized stacking functions for two-dimensional arrays such as close prices, 
        benchmark close, cash deposits, cash earnings, and call sequences. Records are merged using
        `vectorbtpro.records.base.Records.row_stack_records_arrs`.

        If an object's initial cash is specified as one of the options in `vectorbtpro.portfolio.enums.InitCashMode`, 
        it is retained in the final portfolio. When an object defines its initial cash as an absolute amount 
        or an array, the first object's value is used by default, and subsequent non-zero values are treated 
        as cash deposits. Set `combine_init_cash` to True to sum all initial cash arrays.

        The initial position is copied from the first object if it is the only one with a positive value;
        otherwise, stacking is disallowed unless `combine_init_position` is True, which will sum the positions.
        For initial price, stacking occurs only if non-NaN values are found across objects; otherwise, 
        the first object's initial price is used. When `combine_init_price` is enabled, a weighted average 
        is computed based on the initial positions.

        !!! note
            When possible, avoid including initial position and price in portfolios to be stacked, 
            as their stacking order might not correctly reflect the simulation chronology and could 
            lead to inaccurate results.

        Args:
            *objs (Portfolio): (Additional) `Portfolio` instances to stack.
            wrapper_kwargs (KwargsLike): Additional keyword arguments for configuring the array wrapper.
            group_by (GroupByLike): Criteria for grouping during stacking.
            combine_init_cash (bool): If True, sums all initial cash arrays instead of using 
                the first object's value.
            combine_init_position (bool): If True, sums initial position arrays when 
                multiple objects have non-zero positions.
            combine_init_price (bool): If True, combines initial price arrays using 
                a weighted average when stacking.
            **kwargs: Additional keyword arguments passed to internal stacking functions and 
                the resulting `Portfolio` constructor.

        Returns:
            Portfolio: A new `Portfolio` instance resulting from stacking the input objects along rows.
        """
        if not isinstance(cls_or_self, type):
            objs = (cls_or_self, *objs)
            cls = type(cls_or_self)
        else:
            cls = cls_or_self
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Portfolio):
                raise TypeError("Each object to be merged must be an instance of Portfolio")
        _objs = list(map(lambda x: x.disable_weights(), objs))
        if "wrapper" not in kwargs:
            wrapper_kwargs = merge_dicts(dict(group_by=group_by), wrapper_kwargs)
            kwargs["wrapper"] = ArrayWrapper.row_stack(*[obj.wrapper for obj in _objs], **wrapper_kwargs)
        for i in range(1, len(_objs)):
            if _objs[i].cash_sharing != _objs[0].cash_sharing:
                raise ValueError("Objects to be merged must have the same 'cash_sharing'")
        kwargs["cash_sharing"] = _objs[0].cash_sharing
        cs_group_by = None if kwargs["cash_sharing"] else False
        cs_n_cols = kwargs["wrapper"].get_shape_2d(group_by=cs_group_by)[1]
        n_cols = kwargs["wrapper"].shape_2d[1]

        if "close" not in kwargs:
            kwargs["close"] = kwargs["wrapper"].row_stack_arrs(
                *[obj.close for obj in _objs],
                group_by=False,
                wrap=False,
            )
        if "open" not in kwargs:
            stack_open_objs = True
            for obj in _objs:
                if obj._open is None:
                    stack_open_objs = False
                    break
            if stack_open_objs:
                kwargs["open"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.open for obj in _objs],
                    group_by=False,
                    wrap=False,
                )
        if "high" not in kwargs:
            stack_high_objs = True
            for obj in _objs:
                if obj._high is None:
                    stack_high_objs = False
                    break
            if stack_high_objs:
                kwargs["high"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.high for obj in _objs],
                    group_by=False,
                    wrap=False,
                )
        if "low" not in kwargs:
            stack_low_objs = True
            for obj in _objs:
                if obj._low is None:
                    stack_low_objs = False
                    break
            if stack_low_objs:
                kwargs["low"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.low for obj in _objs],
                    group_by=False,
                    wrap=False,
                )
        if "order_records" not in kwargs:
            kwargs["order_records"] = Orders.row_stack_records_arrs(*[obj.orders for obj in _objs], **kwargs)
        if "log_records" not in kwargs:
            kwargs["log_records"] = Logs.row_stack_records_arrs(*[obj.logs for obj in _objs], **kwargs)
        if "init_cash" not in kwargs:
            stack_init_cash_objs = False
            for obj in _objs:
                if not checks.is_int(obj._init_cash) or obj._init_cash not in enums.InitCashMode:
                    stack_init_cash_objs = True
                    break
            if stack_init_cash_objs:
                stack_init_cash_objs = False
                init_cash_objs = []
                for i, obj in enumerate(_objs):
                    init_cash_obj = obj.get_init_cash(group_by=cs_group_by)
                    init_cash_obj = to_1d_array(init_cash_obj)
                    init_cash_obj = broadcast_array_to(init_cash_obj, cs_n_cols)
                    if i > 0 and (init_cash_obj != 0).any():
                        stack_init_cash_objs = True
                    init_cash_objs.append(init_cash_obj)
                if stack_init_cash_objs:
                    if not combine_init_cash:
                        cash_deposits_objs = []
                        for i, obj in enumerate(_objs):
                            cash_deposits_obj = obj.get_cash_deposits(group_by=cs_group_by)
                            cash_deposits_obj = to_2d_array(cash_deposits_obj)
                            cash_deposits_obj = broadcast_array_to(
                                cash_deposits_obj,
                                (cash_deposits_obj.shape[0], cs_n_cols),
                            )
                            cash_deposits_obj = cash_deposits_obj.copy()
                            if i > 0:
                                cash_deposits_obj[0] = init_cash_objs[i]
                            cash_deposits_objs.append(cash_deposits_obj)
                        kwargs["cash_deposits"] = row_stack_arrays(cash_deposits_objs)
                        kwargs["init_cash"] = init_cash_objs[0]
                    else:
                        kwargs["init_cash"] = np.asarray(init_cash_objs).sum(axis=0)
                else:
                    kwargs["init_cash"] = init_cash_objs[0]
        if "init_position" not in kwargs:
            stack_init_position_objs = False
            init_position_objs = []
            for i, obj in enumerate(_objs):
                init_position_obj = obj.get_init_position()
                init_position_obj = to_1d_array(init_position_obj)
                init_position_obj = broadcast_array_to(init_position_obj, n_cols)
                if i > 0 and (init_position_obj != 0).any():
                    stack_init_position_objs = True
                init_position_objs.append(init_position_obj)
            if stack_init_position_objs:
                if not combine_init_position:
                    raise ValueError("Initial position cannot be stacked along rows")
                kwargs["init_position"] = np.asarray(init_position_objs).sum(axis=0)
            else:
                kwargs["init_position"] = init_position_objs[0]
        if "init_price" not in kwargs:
            stack_init_price_objs = False
            init_position_objs = []
            init_price_objs = []
            for i, obj in enumerate(_objs):
                init_position_obj = obj.get_init_position()
                init_position_obj = to_1d_array(init_position_obj)
                init_position_obj = broadcast_array_to(init_position_obj, n_cols)
                init_price_obj = obj.get_init_price()
                init_price_obj = to_1d_array(init_price_obj)
                init_price_obj = broadcast_array_to(init_price_obj, n_cols)
                if i > 0 and (init_position_obj != 0).any() and not np.isnan(init_price_obj).all():
                    stack_init_price_objs = True
                init_position_objs.append(init_position_obj)
                init_price_objs.append(init_price_obj)
            if stack_init_price_objs:
                if not combine_init_price:
                    raise ValueError("Initial price cannot be stacked along rows")
                init_position_objs = np.asarray(init_position_objs)
                init_price_objs = np.asarray(init_price_objs)
                mask1 = (init_position_objs != 0).any(axis=1)
                mask2 = (~np.isnan(init_price_objs)).any(axis=1)
                mask = mask1 & mask2
                init_position_objs = init_position_objs[mask]
                init_price_objs = init_price_objs[mask]
                nom = (init_position_objs * init_price_objs).sum(axis=0)
                denum = init_position_objs.sum(axis=0)
                kwargs["init_price"] = nom / denum
            else:
                kwargs["init_price"] = init_price_objs[0]
        if "cash_deposits" not in kwargs:
            stack_cash_deposits_objs = False
            for obj in _objs:
                if obj._cash_deposits.size > 1 or obj._cash_deposits.item() != 0:
                    stack_cash_deposits_objs = True
                    break
            if stack_cash_deposits_objs:
                kwargs["cash_deposits"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.get_cash_deposits(group_by=cs_group_by) for obj in _objs],
                    group_by=cs_group_by,
                    wrap=False,
                )
            else:
                kwargs["cash_deposits"] = np.array([[0.0]])
        if "cash_earnings" not in kwargs:
            stack_cash_earnings_objs = False
            for obj in _objs:
                if obj._cash_earnings.size > 1 or obj._cash_earnings.item() != 0:
                    stack_cash_earnings_objs = True
                    break
            if stack_cash_earnings_objs:
                kwargs["cash_earnings"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.get_cash_earnings(group_by=False) for obj in _objs],
                    group_by=False,
                    wrap=False,
                )
            else:
                kwargs["cash_earnings"] = np.array([[0.0]])
        if "call_seq" not in kwargs:
            stack_call_seq_objs = True
            for obj in _objs:
                if obj.config["call_seq"] is None:
                    stack_call_seq_objs = False
                    break
            if stack_call_seq_objs:
                kwargs["call_seq"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.call_seq for obj in _objs],
                    group_by=False,
                    wrap=False,
                )
        if "bm_close" not in kwargs:
            stack_bm_close_objs = True
            for obj in _objs:
                if obj._bm_close is None or isinstance(obj._bm_close, bool):
                    stack_bm_close_objs = False
                    break
            if stack_bm_close_objs:
                kwargs["bm_close"] = kwargs["wrapper"].row_stack_arrs(
                    *[obj.bm_close for obj in _objs],
                    group_by=False,
                    wrap=False,
                )
        if "in_outputs" not in kwargs:
            kwargs["in_outputs"] = cls.row_stack_in_outputs(*_objs, **kwargs)
        if "sim_start" not in kwargs:
            kwargs["sim_start"] = cls.row_stack_sim_start(kwargs["wrapper"], *_objs)
        if "sim_end" not in kwargs:
            kwargs["sim_end"] = cls.row_stack_sim_end(kwargs["wrapper"], *_objs)

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @classmethod
    def column_stack_objs(
        cls: tp.Type[PortfolioT],
        objs: tp.Sequence[tp.Any],
        wrappers: tp.Sequence[ArrayWrapper],
        grouping: str = "columns_or_groups",
        obj_name: tp.Optional[str] = None,
        obj_type: tp.Optional[str] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        cash_sharing: bool = False,
        column_stack_func: tp.Optional[tp.Callable] = None,
        **kwargs,
    ) -> tp.Any:
        """Stack (one- and two-dimensional) objects along columns.

        `column_stack_func` must take the portfolio class and all arguments passed to this method.
        If not all arguments are needed, define `column_stack_func` to accept them as `**kwargs`.

        If all objects are None, boolean, or empty, returns the first object.

        Args:
            objs (Sequence[Any]): Objects to be stacked.
            wrappers (Sequence[ArrayWrapper]): Wrappers corresponding to the objects.
            grouping (str): Grouping method to use. 
            
                Supported options include "columns_or_groups", "columns", "groups", and "cash_sharing".
            obj_name (Optional[str]): Name of the object, used for error messages.
            obj_type (Optional[str]): Type of the object (e.g., "array" or "red_array").
            wrapper (Optional[ArrayWrapper]): Wrapper to be used for stacking.
            cash_sharing (bool): Flag indicating whether cash sharing mode is enabled.
            column_stack_func (Optional[Callable]): Custom function for stacking columns. 
            
                Must accept the portfolio class and all method arguments.
            **kwargs: Additional keyword arguments passed to `column_stack_func`.

        Returns:
            Any: The result of stacking the objects along columns.
        """
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        all_none = True
        for obj in objs:
            if obj is None or isinstance(obj, bool) or (checks.is_np_array(obj) and obj.size == 0):
                if not checks.is_deep_equal(obj, objs[0]):
                    raise ValueError(f"Cannot unify scalar in-outputs with the name '{obj_name}'")
            else:
                all_none = False
                break
        if all_none:
            return objs[0]

        if column_stack_func is not None:
            return column_stack_func(
                cls,
                objs,
                wrappers,
                grouping=grouping,
                obj_name=obj_name,
                obj_type=obj_type,
                wrapper=wrapper,
                **kwargs,
            )

        if grouping == "columns_or_groups":
            obj_group_by = None
        elif grouping == "columns":
            obj_group_by = False
        elif grouping == "groups":
            obj_group_by = None
        elif grouping == "cash_sharing":
            obj_group_by = None if cash_sharing else False
        else:
            raise ValueError(f"Grouping '{grouping}' is not supported")

        if obj_type is None and checks.is_np_array(obj):
            if to_2d_shape(objs[0].shape) == wrappers[0].get_shape_2d(group_by=obj_group_by):
                can_stack = True
                reduced = False
            elif objs[0].shape == (wrappers[0].get_shape_2d(group_by=obj_group_by)[1],):
                can_stack = True
                reduced = True
            else:
                can_stack = False
        elif obj_type is not None and obj_type == "array":
            can_stack = True
            reduced = False
        elif obj_type is not None and obj_type == "red_array":
            can_stack = True
            reduced = True
        else:
            can_stack = False
        if can_stack:
            if reduced:
                wrapped_objs = []
                for i, obj in enumerate(objs):
                    wrapped_objs.append(wrappers[i].wrap_reduced(obj, group_by=obj_group_by))
                return wrapper.concat_arrs(*wrapped_objs, group_by=obj_group_by).values
            wrapped_objs = []
            for i, obj in enumerate(objs):
                wrapped_objs.append(wrappers[i].wrap(obj, group_by=obj_group_by))
            return wrapper.column_stack_arrs(*wrapped_objs, group_by=obj_group_by, wrap=False)
        raise ValueError(f"Cannot figure out how to stack in-outputs with the name '{obj_name}' along columns")

    @classmethod
    def column_stack_in_outputs(
        cls: tp.Type[PortfolioT],
        *objs: tp.MaybeTuple[PortfolioT],
        **kwargs,
    ) -> tp.Optional[tp.NamedTuple]:
        """Stack `Portfolio.in_outputs` along columns.

        Merge the `in_outputs` from multiple `Portfolio` instances by stacking corresponding fields.
        All in-output tuples must either be None or have identical fields.

        If a field is also defined as an attribute of the `Portfolio` class, its options are used 
        to determine the required type and layout by merging results from `Portfolio.parse_field_options` 
        and `Portfolio.in_output_config`. Stacking for each field is performed using `Portfolio.column_stack_objs`.

        Args:
            *objs (MaybeTuple[PortfolioT]): Portfolio instances whose `in_outputs` will be stacked.
            **kwargs: Additional keyword arguments passed to `Portfolio.column_stack_objs`.

        Returns:
            Optional[NamedTuple]: A new in_outputs named tuple with fields stacked along columns, or 
                None if all `in_outputs` are None.
        """
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        all_none = True
        for obj in objs:
            if obj.in_outputs is not None:
                all_none = False
                break
        if all_none:
            return None
        all_keys = set()
        for obj in objs:
            all_keys |= set(obj.in_outputs._asdict().keys())
        for obj in objs:
            if obj.in_outputs is None or len(all_keys.difference(set(obj.in_outputs._asdict().keys()))) > 0:
                raise ValueError("Objects to be merged must have the same in-output fields")

        cls_dir = set(dir(cls))
        new_in_outputs = {}
        for field in objs[0].in_outputs._asdict().keys():
            field_options = merge_dicts(
                cls.parse_field_options(field),
                cls.in_output_config.get(field, None),
            )
            if field_options.get("field", field) in cls_dir:
                prop = getattr(cls, field_options["field"])
                prop_options = getattr(prop, "options", {})
                obj_type = prop_options.get("obj_type", "array")
                group_by_aware = prop_options.get("group_by_aware", True)
                column_stack_func = prop_options.get("column_stack_func", None)
            else:
                obj_type = None
                group_by_aware = True
                column_stack_func = None
            _kwargs = merge_dicts(
                dict(
                    grouping=field_options.get("grouping", "columns_or_groups" if group_by_aware else "columns"),
                    obj_name=field_options.get("field", field),
                    obj_type=field_options.get("obj_type", obj_type),
                    column_stack_func=field_options.get("column_stack_func", column_stack_func),
                ),
                kwargs,
            )
            new_field_obj = cls.column_stack_objs(
                [getattr(obj.in_outputs, field) for obj in objs],
                [obj.wrapper for obj in objs],
                **_kwargs,
            )
            new_in_outputs[field] = new_field_obj

        return type(objs[0].in_outputs)(**new_in_outputs)

    @hybrid_method
    def column_stack(
        cls_or_self: tp.MaybeType[PortfolioT],
        *objs: tp.MaybeTuple[PortfolioT],
        wrapper_kwargs: tp.KwargsLike = None,
        group_by: tp.GroupByLike = None,
        ffill_close: bool = False,
        fbfill_close: bool = False,
        **kwargs,
    ) -> PortfolioT:
        """Stack multiple `Portfolio` instances along columns.

        This function stacks the provided `Portfolio` objects along their columns using
        `vectorbtpro.base.wrapping.ArrayWrapper.column_stack`. All input portfolios must have
        the same cash sharing configuration.

        Depending on the array dimensions:

        * Two-dimensional arrays are merged via `vectorbtpro.base.wrapping.ArrayWrapper.column_stack_arrs`.
        * One-dimensional arrays are concatenated via `vectorbtpro.base.wrapping.ArrayWrapper.concat_arrs`.

        In-outputs are merged using `Portfolio.column_stack_in_outputs` and records are aggregated
        using `vectorbtpro.records.base.Records.column_stack_records_arrs`.

        Args:
            *objs (Portfolio): (Additional) `Portfolio` objects to stack.
            wrapper_kwargs (KwargsLike): Extra keyword arguments for configuring the wrapper.
            group_by (GroupByLike): Grouping specification for stacking arrays.
            ffill_close (bool): If True, forward-fill missing values in the `close` array.
            fbfill_close (bool): If True, backward-fill missing values in the `close` array.
            **kwargs: Additional keyword arguments to pass to the `Portfolio` constructor.

        Returns:
            Portfolio: A new `Portfolio` instance created by stacking the input objects along columns.
        """
        if not isinstance(cls_or_self, type):
            objs = (cls_or_self, *objs)
            cls = type(cls_or_self)
        else:
            cls = cls_or_self
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Portfolio):
                raise TypeError("Each object to be merged must be an instance of Portfolio")
        _objs = list(map(lambda x: x.disable_weights(), objs))
        if "wrapper" not in kwargs:
            wrapper_kwargs = merge_dicts(dict(group_by=group_by), wrapper_kwargs)
            kwargs["wrapper"] = ArrayWrapper.column_stack(
                *[obj.wrapper for obj in _objs],
                **wrapper_kwargs,
            )
        for i in range(1, len(_objs)):
            if _objs[i].cash_sharing != _objs[0].cash_sharing:
                raise ValueError("Objects to be merged must have the same 'cash_sharing'")
        if "cash_sharing" not in kwargs:
            kwargs["cash_sharing"] = _objs[0].cash_sharing
        cs_group_by = None if kwargs["cash_sharing"] else False

        if "close" not in kwargs:
            new_close = kwargs["wrapper"].column_stack_arrs(
                *[obj.close for obj in _objs],
                group_by=False,
            )
            if fbfill_close:
                new_close = new_close.vbt.fbfill()
            elif ffill_close:
                new_close = new_close.vbt.ffill()
            kwargs["close"] = new_close
        if "open" not in kwargs:
            stack_open_objs = True
            for obj in _objs:
                if obj._open is None:
                    stack_open_objs = False
                    break
            if stack_open_objs:
                kwargs["open"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.open for obj in _objs],
                    group_by=False,
                    wrap=False,
                )
        if "high" not in kwargs:
            stack_high_objs = True
            for obj in _objs:
                if obj._high is None:
                    stack_high_objs = False
                    break
            if stack_high_objs:
                kwargs["high"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.high for obj in _objs],
                    group_by=False,
                    wrap=False,
                )
        if "low" not in kwargs:
            stack_low_objs = True
            for obj in _objs:
                if obj._low is None:
                    stack_low_objs = False
                    break
            if stack_low_objs:
                kwargs["low"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.low for obj in _objs],
                    group_by=False,
                    wrap=False,
                )
        if "order_records" not in kwargs:
            kwargs["order_records"] = Orders.column_stack_records_arrs(*[obj.orders for obj in _objs], **kwargs)
        if "log_records" not in kwargs:
            kwargs["log_records"] = Logs.column_stack_records_arrs(*[obj.logs for obj in _objs], **kwargs)
        if "init_cash" not in kwargs:
            stack_init_cash_objs = False
            for obj in _objs:
                if not checks.is_int(obj._init_cash) or obj._init_cash not in enums.InitCashMode:
                    stack_init_cash_objs = True
                    break
            if stack_init_cash_objs:
                kwargs["init_cash"] = to_1d_array(
                    kwargs["wrapper"].concat_arrs(
                        *[obj.get_init_cash(group_by=cs_group_by) for obj in _objs],
                        group_by=cs_group_by,
                    )
                )
        if "init_position" not in kwargs:
            stack_init_position_objs = False
            for obj in _objs:
                if (to_1d_array(obj.init_position) != 0).any():
                    stack_init_position_objs = True
                    break
            if stack_init_position_objs:
                kwargs["init_position"] = to_1d_array(
                    kwargs["wrapper"].concat_arrs(
                        *[obj.init_position for obj in _objs],
                        group_by=False,
                    ),
                )
            else:
                kwargs["init_position"] = np.array([0.0])
        if "init_price" not in kwargs:
            stack_init_price_objs = False
            for obj in _objs:
                if not np.isnan(to_1d_array(obj.init_price)).all():
                    stack_init_price_objs = True
                    break
            if stack_init_price_objs:
                kwargs["init_price"] = to_1d_array(
                    kwargs["wrapper"].concat_arrs(
                        *[obj.init_price for obj in _objs],
                        group_by=False,
                    ),
                )
            else:
                kwargs["init_price"] = np.array([np.nan])
        if "cash_deposits" not in kwargs:
            stack_cash_deposits_objs = False
            for obj in _objs:
                if obj._cash_deposits.size > 1 or obj._cash_deposits.item() != 0:
                    stack_cash_deposits_objs = True
                    break
            if stack_cash_deposits_objs:
                kwargs["cash_deposits"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.get_cash_deposits(group_by=cs_group_by) for obj in _objs],
                    group_by=cs_group_by,
                    reindex_kwargs=dict(fill_value=0),
                    wrap=False,
                )
            else:
                kwargs["cash_deposits"] = np.array([[0.0]])
        if "cash_earnings" not in kwargs:
            stack_cash_earnings_objs = False
            for obj in _objs:
                if obj._cash_earnings.size > 1 or obj._cash_earnings.item() != 0:
                    stack_cash_earnings_objs = True
                    break
            if stack_cash_earnings_objs:
                kwargs["cash_earnings"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.get_cash_earnings(group_by=False) for obj in _objs],
                    group_by=False,
                    reindex_kwargs=dict(fill_value=0),
                    wrap=False,
                )
            else:
                kwargs["cash_earnings"] = np.array([[0.0]])
        if "call_seq" not in kwargs:
            stack_call_seq_objs = True
            for obj in _objs:
                if obj.config["call_seq"] is None:
                    stack_call_seq_objs = False
                    break
            if stack_call_seq_objs:
                kwargs["call_seq"] = kwargs["wrapper"].column_stack_arrs(
                    *[obj.call_seq for obj in _objs],
                    group_by=False,
                    reindex_kwargs=dict(fill_value=0),
                    wrap=False,
                )
        if "bm_close" not in kwargs:
            stack_bm_close_objs = True
            for obj in _objs:
                if obj._bm_close is None or isinstance(obj._bm_close, bool):
                    stack_bm_close_objs = False
                    break
            if stack_bm_close_objs:
                new_bm_close = kwargs["wrapper"].column_stack_arrs(
                    *[obj.bm_close for obj in _objs],
                    group_by=False,
                    wrap=False,
                )
                if fbfill_close:
                    new_bm_close = new_bm_close.vbt.fbfill()
                elif ffill_close:
                    new_bm_close = new_bm_close.vbt.ffill()
                kwargs["bm_close"] = new_bm_close
        if "in_outputs" not in kwargs:
            kwargs["in_outputs"] = cls.column_stack_in_outputs(*_objs, **kwargs)
        if "sim_start" not in kwargs:
            kwargs["sim_start"] = cls.column_stack_sim_start(kwargs["wrapper"], *_objs)
        if "sim_end" not in kwargs:
            kwargs["sim_end"] = cls.column_stack_sim_end(kwargs["wrapper"], *_objs)

        if "weights" not in kwargs:
            stack_weights_objs = False
            obj_weights = []
            for obj in objs:
                if obj.weights is not None:
                    stack_weights_objs = True
                    obj_weights.append(obj.weights)
                else:
                    obj_weights.append([np.nan] * obj.wrapper.shape_2d[1])
            if stack_weights_objs:
                kwargs["weights"] = to_1d_array(
                    kwargs["wrapper"].concat_arrs(
                        *obj_weights,
                        group_by=False,
                    ),
                )

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    # ############# In-outputs ############# #

    _in_output_config: tp.ClassVar[Config] = HybridConfig(
        dict(
            cash=dict(grouping="cash_sharing"),
            position=dict(grouping="columns"),
            debt=dict(grouping="columns"),
            locked_cash=dict(grouping="columns"),
            free_cash=dict(grouping="cash_sharing"),
            returns=dict(grouping="cash_sharing"),
        )
    )

    @property
    def in_output_config(self) -> Config:
        """In-output configuration of `${cls_name}`.

        ```python
        ${in_output_config}
        ```

        Returns:
            Config: A hybrid-copied in-output configuration from `${cls_name}._in_output_config`. 
                
                Changing this instance's configuration does not affect the class-level configuration.

        !!! note
            To modify in_outputs, change the configuration in-place, override this property, or 
            assign a new value to `${cls_name}._in_output_config`.
        """
        return self._in_output_config

    @classmethod
    def parse_field_options(cls, field: str) -> tp.Kwargs:
        """Parse options from a field name.

        Extracts suffixes from the field name to determine the grouping and object type, 
        and returns a cleaned field name.

        Suffixes for grouping:

        * `_cs`: per group if cash sharing is enabled, otherwise per column.
        * `_pcg`: per group if grouped, otherwise per column.
        * `_pg`: per group.
        * `_pc`: per column.
        * `_records`: records.

        Suffixes for object type:

        * `_2d`: element per timestamp and column or group (time series).
        * `_1d`: element per column or group (reduced time series).

        Args:
            field (str): The original field name containing embedded option suffixes.

        Returns:
            Kwargs: Dictionary with the following keys:

                * `obj_type`: Detected object type if applicable.
                * `grouping`: Detected grouping option if applicable.
                * `field`: The cleaned field name.
        """
        options = dict()
        new_parts = []
        for part in field.split("_"):
            if part == "1d":
                options["obj_type"] = "red_array"
            elif part == "2d":
                options["obj_type"] = "array"
            elif part == "records":
                options["obj_type"] = "records"
            elif part == "pc":
                options["grouping"] = "columns"
            elif part == "pg":
                options["grouping"] = "groups"
            elif part == "pcg":
                options["grouping"] = "columns_or_groups"
            elif part == "cs":
                options["grouping"] = "cash_sharing"
            else:
                new_parts.append(part)
        field = "_".join(new_parts)
        options["field"] = field
        return options

    def matches_field_options(
        self,
        options: tp.Kwargs,
        obj_type: tp.Optional[str] = None,
        group_by_aware: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
    ) -> bool:
        """Determine if a field's options meet the required criteria.

        This method checks whether the field's parsed options match the expected object type 
        and grouping configuration. The evaluation also considers the portfolio's current grouping 
        state and cash sharing setting. If an option is absent from the provided options, it is assumed 
        to satisfy the requirement.

        Args:
            options (Kwargs): Dictionary of parsed field options.
            obj_type (Optional[str]): Expected object type. 
            
                If provided, the field's object type must match.
            group_by_aware (bool): Flag indicating whether to account for grouping awareness.
            wrapper (Optional[ArrayWrapper]): Wrapper used to determine grouping. 
            
                If None, the instance's wrapper is used.
            group_by (GroupByLike): Grouping specification for the evaluation.

        Returns:
            bool: True if the field options satisfy the requirements; otherwise, False.
        """
        field_obj_type = options.get("obj_type", None)
        field_grouping = options.get("grouping", None)
        if field_obj_type is not None and obj_type is not None:
            if field_obj_type != obj_type:
                return False
        if field_grouping is not None:
            if wrapper is None:
                wrapper = self.wrapper
            is_grouped = wrapper.grouper.is_grouped(group_by=group_by)
            if is_grouped:
                if group_by_aware:
                    if field_grouping == "groups":
                        return True
                    if field_grouping == "columns_or_groups":
                        return True
                    if self.cash_sharing:
                        if field_grouping == "cash_sharing":
                            return True
                else:
                    if field_grouping == "columns":
                        return True
                    if not self.cash_sharing:
                        if field_grouping == "cash_sharing":
                            return True
            else:
                if field_grouping == "columns":
                    return True
                if field_grouping == "columns_or_groups":
                    return True
                if field_grouping == "cash_sharing":
                    return True
            return False
        return True

    def wrap_obj(
        self,
        obj: tp.Any,
        obj_name: tp.Optional[str] = None,
        grouping: str = "columns_or_groups",
        obj_type: tp.Optional[str] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_func: tp.Optional[tp.Callable] = None,
        wrap_kwargs: tp.KwargsLike = None,
        force_wrapping: bool = False,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Any:
        """Wrap an object.

        Wraps the provided object according to the specified object type, grouping strategy, 
        and wrapping function. If `wrap_func` is provided, it is called with the portfolio (`Portfolio`), 
        the object, and all other arguments. If the object is None or a boolean, it is returned unchanged.

        Args:
            obj (Any): The object to be wrapped.
            obj_name (Optional[str]): An optional identifier for the object.
            grouping (str): The grouping strategy.

                Supported options include "columns_or_groups", "columns", "groups", and "cash_sharing".
            obj_type (Optional[str]): The type of the object.
            
                Supported options include "records", "array", and "red_array".
            wrapper (Optional[ArrayWrapper]): The wrapper instance to use. 
            
                Defaults to `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Criteria for grouping used by the wrapper.
            wrap_func (Optional[Callable]): A custom wrapping function that takes the portfolio, 
                object, and additional parameters.
            wrap_kwargs (KwargsLike): Additional keyword arguments for the wrapping function.
            force_wrapping (bool): If True, forces wrapping and raises an error when wrapping is not feasible.
            silence_warnings (bool): If True, suppresses warnings when wrapping cannot be determined.
            **kwargs: Additional keyword arguments passed to the wrapping function.

        Returns:
            Any: The wrapped object, or the original object if no wrapping is applied.

        !!! note
            The wrapping process considers the object type, its dimensionality, and 
            the specified grouping or cash sharing settings.
        """
        if obj is None or isinstance(obj, bool):
            return obj
        if wrapper is None:
            wrapper = self.wrapper
        is_grouped = wrapper.grouper.is_grouped(group_by=group_by)
        if wrap_func is not None:
            return wrap_func(
                self,
                obj,
                obj_name=obj_name,
                grouping=grouping,
                obj_type=obj_type,
                wrapper=wrapper,
                group_by=group_by,
                wrap_kwargs=wrap_kwargs,
                force_wrapping=force_wrapping,
                silence_warnings=silence_warnings,
                **kwargs,
            )

        def _wrap_reduced_grouped(obj):
            _wrap_kwargs = merge_dicts(dict(name_or_index=obj_name), wrap_kwargs)
            return wrapper.wrap_reduced(obj, group_by=group_by, **_wrap_kwargs)

        def _wrap_reduced(obj):
            _wrap_kwargs = merge_dicts(dict(name_or_index=obj_name), wrap_kwargs)
            return wrapper.wrap_reduced(obj, group_by=False, **_wrap_kwargs)

        def _wrap_grouped(obj):
            return wrapper.wrap(obj, group_by=group_by, **resolve_dict(wrap_kwargs))

        def _wrap(obj):
            return wrapper.wrap(obj, group_by=False, **resolve_dict(wrap_kwargs))

        if obj_type is not None and obj_type not in {"records"}:
            if grouping == "cash_sharing":
                if obj_type == "array":
                    if is_grouped and self.cash_sharing:
                        return _wrap_grouped(obj)
                    return _wrap(obj)
                if obj_type == "red_array":
                    if is_grouped and self.cash_sharing:
                        return _wrap_reduced_grouped(obj)
                    return _wrap_reduced(obj)
                if obj.ndim == 2:
                    if is_grouped and self.cash_sharing:
                        return _wrap_grouped(obj)
                    return _wrap(obj)
                if obj.ndim == 1:
                    if is_grouped and self.cash_sharing:
                        return _wrap_reduced_grouped(obj)
                    return _wrap_reduced(obj)
            if grouping == "columns_or_groups":
                if obj_type == "array":
                    if is_grouped:
                        return _wrap_grouped(obj)
                    return _wrap(obj)
                if obj_type == "red_array":
                    if is_grouped:
                        return _wrap_reduced_grouped(obj)
                    return _wrap_reduced(obj)
                if obj.ndim == 2:
                    if is_grouped:
                        return _wrap_grouped(obj)
                    return _wrap(obj)
                if obj.ndim == 1:
                    if is_grouped:
                        return _wrap_reduced_grouped(obj)
                    return _wrap_reduced(obj)
            if grouping == "groups":
                if obj_type == "array":
                    return _wrap_grouped(obj)
                if obj_type == "red_array":
                    return _wrap_reduced_grouped(obj)
                if obj.ndim == 2:
                    return _wrap_grouped(obj)
                if obj.ndim == 1:
                    return _wrap_reduced_grouped(obj)
            if grouping == "columns":
                if obj_type == "array":
                    return _wrap(obj)
                if obj_type == "red_array":
                    return _wrap_reduced(obj)
                if obj.ndim == 2:
                    return _wrap(obj)
                if obj.ndim == 1:
                    return _wrap_reduced(obj)
        if obj_type not in {"records"}:
            if checks.is_np_array(obj) and not checks.is_record_array(obj):
                if is_grouped:
                    if obj_type is not None and obj_type == "array":
                        return _wrap_grouped(obj)
                    if obj_type is not None and obj_type == "red_array":
                        return _wrap_reduced_grouped(obj)
                    if to_2d_shape(obj.shape) == wrapper.get_shape_2d():
                        return _wrap_grouped(obj)
                    if obj.shape == (wrapper.get_shape_2d()[1],):
                        return _wrap_reduced_grouped(obj)
                if obj_type is not None and obj_type == "array":
                    return _wrap(obj)
                if obj_type is not None and obj_type == "red_array":
                    return _wrap_reduced(obj)
                if to_2d_shape(obj.shape) == wrapper.shape_2d:
                    return _wrap(obj)
                if obj.shape == (wrapper.shape_2d[1],):
                    return _wrap_reduced(obj)
        if force_wrapping:
            raise NotImplementedError(f"Cannot wrap object '{obj_name}'")
        if not silence_warnings:
            warn(f"Cannot figure out how to wrap object '{obj_name}'")
        return obj

    def get_in_output(
        self,
        field: str,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> tp.Union[None, bool, tp.AnyArray]:
        """Find and wrap an in-output object matching a specified field.

        This method searches for an in-output field that matches the provided name or any of its aliases.
        If the field is found among the attributes of this `Portfolio` instance, its options are used to
        determine the required type and layout. Otherwise, options are resolved from `Portfolio.in_outputs`
        and `Portfolio.in_output_config`, and candidates are filtered using `Portfolio.matches_field_options`.
        The identified in-output object is then wrapped via `Portfolio.wrap_obj`.

        Args:
            field (str): The name of the field to retrieve.
            wrapper (Optional[ArrayWrapper]): An optional wrapper instance.
            group_by (GroupByLike): Grouping specification for the in-output data.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[None, bool, AnyArray]: The wrapped in-output object.
        """
        if self.in_outputs is None:
            raise ValueError("No in-outputs attached")

        if field in self.cls_dir:
            prop = getattr(type(self), field)
            prop_options = getattr(prop, "options", {})
            obj_type = prop_options.get("obj_type", "array")
            group_by_aware = prop_options.get("group_by_aware", True)
            wrap_func = prop_options.get("wrap_func", None)
            wrap_kwargs = prop_options.get("wrap_kwargs", None)
            force_wrapping = prop_options.get("force_wrapping", False)
            silence_warnings = prop_options.get("silence_warnings", False)
            field_aliases = prop_options.get("field_aliases", None)
            if field_aliases is None:
                field_aliases = []
            field_aliases = {field, *field_aliases}
            found_attr = True
        else:
            obj_type = None
            group_by_aware = True
            wrap_func = None
            wrap_kwargs = None
            force_wrapping = False
            silence_warnings = False
            field_aliases = {field}
            found_attr = False

        found_field = None
        found_field_options = None
        for _field in set(self.in_outputs._fields):
            _field_options = merge_dicts(
                self.parse_field_options(_field),
                self.in_output_config.get(_field, None),
            )
            if (not found_attr and field == _field) or (
                (_field in field_aliases or _field_options.get("field", _field) in field_aliases)
                and self.matches_field_options(
                    _field_options,
                    obj_type=obj_type,
                    group_by_aware=group_by_aware,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            ):
                if found_field is not None:
                    raise ValueError(f"Multiple fields for '{field}' found in in_outputs")
                found_field = _field
                found_field_options = _field_options
        if found_field is None:
            raise AttributeError(f"No compatible field for '{field}' found in in_outputs")
        obj = getattr(self.in_outputs, found_field)
        if found_attr and checks.is_np_array(obj) and obj.shape == (0, 0):  # for returns
            return None
        kwargs = merge_dicts(
            dict(
                grouping=found_field_options.get("grouping", "columns_or_groups" if group_by_aware else "columns"),
                obj_type=found_field_options.get("obj_type", obj_type),
                wrap_func=found_field_options.get("wrap_func", wrap_func),
                wrap_kwargs=found_field_options.get("wrap_kwargs", wrap_kwargs),
                force_wrapping=found_field_options.get("force_wrapping", force_wrapping),
                silence_warnings=found_field_options.get("silence_warnings", silence_warnings),
            ),
            kwargs,
        )
        return self.wrap_obj(
            obj,
            found_field_options.get("field", found_field),
            wrapper=wrapper,
            group_by=group_by,
            **kwargs,
        )

    # ############# Indexing ############# #

    def index_obj(
        self,
        obj: tp.Any,
        wrapper_meta: dict,
        obj_name: tp.Optional[str] = None,
        grouping: str = "columns_or_groups",
        obj_type: tp.Optional[str] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        indexing_func: tp.Optional[tp.Callable] = None,
        force_indexing: bool = False,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Any:
        """Perform indexing on an object.

        Indexes an object based on the provided metadata, type, and grouping parameters. 
        If the object is None, a boolean, or an empty array, it is returned as-is.

        Args:
            obj (Any): The object to index.
            wrapper_meta (dict): Metadata dictionary used to compute index positions.
            obj_name (Optional[str]): Name of the object for identification in warnings.
            grouping (str): Specifies the grouping scheme for indexing. 
            
                Possible values include "columns_or_groups", "cash_sharing", "groups", and "columns".
            obj_type (Optional[str]): Specifies the type of the object (e.g., "records", "array", "red_array").
            wrapper (Optional[ArrayWrapper]): The array wrapper instance. 
            
                If None, defaults to `Portfolio.wrapper`.
            group_by (tp.GroupByLike): Determines the grouping structure to check if the object is grouped.
            indexing_func (Optional[Callable]): Custom indexing function that must accept `Portfolio`,
                the object, wrapper_meta, and the remaining arguments, including `**kwargs`. 
                
                If unused, leave as None.
            force_indexing (bool): Indicates if indexing should be forced, 
                raising an error if it cannot be applied.
            silence_warnings (bool): If True, suppresses warnings when 
                indexing decisions cannot be determined.
            **kwargs: Additional keyword arguments for indexing.

        Returns:
            Any: The resulting indexed object.
        """
        if obj is None or isinstance(obj, bool) or (checks.is_np_array(obj) and obj.size == 0):
            return obj
        if wrapper is None:
            wrapper = self.wrapper
        if indexing_func is not None:
            return indexing_func(
                self,
                obj,
                wrapper_meta,
                obj_name=obj_name,
                grouping=grouping,
                obj_type=obj_type,
                wrapper=wrapper,
                group_by=group_by,
                force_indexing=force_indexing,
                silence_warnings=silence_warnings,
                **kwargs,
            )

        def _index_1d_by_group(obj: tp.ArrayLike) -> tp.ArrayLike:
            return to_1d_array(obj)[wrapper_meta["group_idxs"]]

        def _index_1d_by_col(obj: tp.ArrayLike) -> tp.ArrayLike:
            return to_1d_array(obj)[wrapper_meta["col_idxs"]]

        def _index_2d_by_group(obj: tp.ArrayLike) -> tp.ArrayLike:
            return to_2d_array(obj)[wrapper_meta["row_idxs"], :][:, wrapper_meta["group_idxs"]]

        def _index_2d_by_col(obj: tp.ArrayLike) -> tp.ArrayLike:
            return to_2d_array(obj)[wrapper_meta["row_idxs"], :][:, wrapper_meta["col_idxs"]]

        def _index_records(obj: tp.RecordArray) -> tp.RecordArray:
            records = Records(wrapper, obj)
            records_meta = records.indexing_func_meta(wrapper_meta=wrapper_meta)
            return records.indexing_func(records_meta=records_meta).values

        is_grouped = wrapper.grouper.is_grouped(group_by=group_by)
        if obj_type is not None and obj_type == "records":
            return _index_records(obj)
        if grouping == "cash_sharing":
            if obj_type is not None and obj_type == "array":
                if is_grouped and self.cash_sharing:
                    return _index_2d_by_group(obj)
                return _index_2d_by_col(obj)
            if obj_type is not None and obj_type == "red_array":
                if is_grouped and self.cash_sharing:
                    return _index_1d_by_group(obj)
                return _index_1d_by_col(obj)
            if obj.ndim == 2:
                if is_grouped and self.cash_sharing:
                    return _index_2d_by_group(obj)
                return _index_2d_by_col(obj)
            if obj.ndim == 1:
                if is_grouped and self.cash_sharing:
                    return _index_1d_by_group(obj)
                return _index_1d_by_col(obj)
        if grouping == "columns_or_groups":
            if obj_type is not None and obj_type == "array":
                if is_grouped:
                    return _index_2d_by_group(obj)
                return _index_2d_by_col(obj)
            if obj_type is not None and obj_type == "red_array":
                if is_grouped:
                    return _index_1d_by_group(obj)
                return _index_1d_by_col(obj)
            if obj.ndim == 2:
                if is_grouped:
                    return _index_2d_by_group(obj)
                return _index_2d_by_col(obj)
            if obj.ndim == 1:
                if is_grouped:
                    return _index_1d_by_group(obj)
                return _index_1d_by_col(obj)
        if grouping == "groups":
            if obj_type is not None and obj_type == "array":
                return _index_2d_by_group(obj)
            if obj_type is not None and obj_type == "red_array":
                return _index_1d_by_group(obj)
            if obj.ndim == 2:
                return _index_2d_by_group(obj)
            if obj.ndim == 1:
                return _index_1d_by_group(obj)
        if grouping == "columns":
            if obj_type is not None and obj_type == "array":
                return _index_2d_by_col(obj)
            if obj_type is not None and obj_type == "red_array":
                return _index_1d_by_col(obj)
            if obj.ndim == 2:
                return _index_2d_by_col(obj)
            if obj.ndim == 1:
                return _index_1d_by_col(obj)
        if checks.is_np_array(obj):
            if is_grouped:
                if obj_type is not None and obj_type == "array":
                    return _index_2d_by_group(obj)
                if obj_type is not None and obj_type == "red_array":
                    return _index_1d_by_group(obj)
                if to_2d_shape(obj.shape) == wrapper.get_shape_2d():
                    return _index_2d_by_group(obj)
                if obj.shape == (wrapper.get_shape_2d()[1],):
                    return _index_1d_by_group(obj)
            if obj_type is not None and obj_type == "array":
                return _index_2d_by_col(obj)
            if obj_type is not None and obj_type == "red_array":
                return _index_1d_by_col(obj)
            if to_2d_shape(obj.shape) == wrapper.shape_2d:
                return _index_2d_by_col(obj)
            if obj.shape == (wrapper.shape_2d[1],):
                return _index_1d_by_col(obj)
        if force_indexing:
            raise NotImplementedError(f"Cannot index object '{obj_name}'")
        if not silence_warnings:
            warn(f"Cannot figure out how to index object '{obj_name}'")
        return obj

    def in_outputs_indexing_func(self, wrapper_meta: dict, **kwargs) -> tp.Optional[tp.NamedTuple]:
        """Perform indexing on `Portfolio.in_outputs`.

        Processes each field in `Portfolio.in_outputs` by determining the field's configuration 
        from attribute options, parsed field options via `Portfolio.parse_field_options`, and 
        the indexing configuration in `Portfolio.in_output_config`. Each field is then indexed 
        using the `Portfolio.index_obj` method.

        Args:
            wrapper_meta (dict): Metadata information required for indexing the outputs.
            **kwargs: Additional keyword arguments passed to the indexing process.

        Returns:
            Optional[NamedTuple]: A new named tuple with the indexed in-output fields, 
                or None if `Portfolio.in_outputs` is None.
        """
        if self.in_outputs is None:
            return None

        new_in_outputs = {}
        for field, obj in self.in_outputs._asdict().items():
            field_options = merge_dicts(
                self.parse_field_options(field),
                self.in_output_config.get(field, None),
            )
            if field_options.get("field", field) in self.cls_dir:
                prop = getattr(type(self), field_options["field"])
                prop_options = getattr(prop, "options", {})
                obj_type = prop_options.get("obj_type", "array")
                group_by_aware = prop_options.get("group_by_aware", True)
                indexing_func = prop_options.get("indexing_func", None)
                force_indexing = prop_options.get("force_indexing", False)
                silence_warnings = prop_options.get("silence_warnings", False)
            else:
                obj_type = None
                group_by_aware = True
                indexing_func = None
                force_indexing = False
                silence_warnings = False
            _kwargs = merge_dicts(
                dict(
                    grouping=field_options.get("grouping", "columns_or_groups" if group_by_aware else "columns"),
                    obj_name=field_options.get("field", field),
                    obj_type=field_options.get("obj_type", obj_type),
                    indexing_func=field_options.get("indexing_func", indexing_func),
                    force_indexing=field_options.get("force_indexing", force_indexing),
                    silence_warnings=field_options.get("silence_warnings", silence_warnings),
                ),
                kwargs,
            )
            new_obj = self.index_obj(obj, wrapper_meta, **_kwargs)
            new_in_outputs[field] = new_obj
        return type(self.in_outputs)(**new_in_outputs)

    def indexing_func(
        self: PortfolioT,
        *args,
        in_output_kwargs: tp.KwargsLike = None,
        wrapper_meta: tp.DictLike = None,
        **kwargs,
    ) -> PortfolioT:
        """Perform indexing on the Portfolio.

        In-outputs are indexed using `Portfolio.in_outputs_indexing_func`.

        Args:
            *args: Additional positional arguments.
            in_output_kwargs (KwargsLike): Additional keyword arguments for in-output configurations.
            wrapper_meta (DictLike): Metadata dictionary for the wrapper; computed if None.
            **kwargs: Additional keyword arguments.

        Returns:
            Portfolio: Updated Portfolio instance after indexing.
        """
        _self = self.disable_weights()

        if wrapper_meta is None:
            wrapper_meta = _self.wrapper.indexing_func_meta(
                *args,
                column_only_select=_self.column_only_select,
                range_only_select=_self.range_only_select,
                group_select=_self.group_select,
                **kwargs,
            )
        new_wrapper = wrapper_meta["new_wrapper"]
        row_idxs = wrapper_meta["row_idxs"]
        rows_changed = wrapper_meta["rows_changed"]
        col_idxs = wrapper_meta["col_idxs"]
        columns_changed = wrapper_meta["columns_changed"]
        group_idxs = wrapper_meta["group_idxs"]

        new_close = ArrayWrapper.select_from_flex_array(
            _self._close,
            row_idxs=row_idxs,
            col_idxs=col_idxs,
            rows_changed=rows_changed,
            columns_changed=columns_changed,
        )
        if _self._open is not None:
            new_open = ArrayWrapper.select_from_flex_array(
                _self._open,
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
        else:
            new_open = _self._open
        if _self._high is not None:
            new_high = ArrayWrapper.select_from_flex_array(
                _self._high,
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
        else:
            new_high = _self._high
        if _self._low is not None:
            new_low = ArrayWrapper.select_from_flex_array(
                _self._low,
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
        else:
            new_low = _self._low
        new_order_records = _self.orders.indexing_func_meta(wrapper_meta=wrapper_meta)["new_records_arr"]
        new_log_records = _self.logs.indexing_func_meta(wrapper_meta=wrapper_meta)["new_records_arr"]
        new_init_cash = _self._init_cash
        if not checks.is_int(new_init_cash):
            new_init_cash = to_1d_array(new_init_cash)
            if rows_changed and row_idxs.start > 0:
                if _self.wrapper.grouper.is_grouped() and not _self.cash_sharing:
                    cash = _self.get_cash(group_by=False)
                else:
                    cash = _self.cash
                new_init_cash = to_1d_array(cash.iloc[row_idxs.start - 1])
            if columns_changed and new_init_cash.shape[0] > 1:
                if _self.cash_sharing:
                    new_init_cash = new_init_cash[group_idxs]
                else:
                    new_init_cash = new_init_cash[col_idxs]
        new_init_position = to_1d_array(_self._init_position)
        if rows_changed and row_idxs.start > 0:
            new_init_position = to_1d_array(_self.assets.iloc[row_idxs.start - 1])
        if columns_changed and new_init_position.shape[0] > 1:
            new_init_position = new_init_position[col_idxs]
        new_init_price = to_1d_array(_self._init_price)
        if rows_changed and row_idxs.start > 0:
            new_init_price = to_1d_array(_self.close.iloc[: row_idxs.start].ffill().iloc[-1])
        if columns_changed and new_init_price.shape[0] > 1:
            new_init_price = new_init_price[col_idxs]
        new_cash_deposits = ArrayWrapper.select_from_flex_array(
            _self._cash_deposits,
            row_idxs=row_idxs,
            col_idxs=group_idxs if _self.cash_sharing else col_idxs,
            rows_changed=rows_changed,
            columns_changed=columns_changed,
        )
        new_cash_earnings = ArrayWrapper.select_from_flex_array(
            _self._cash_earnings,
            row_idxs=row_idxs,
            col_idxs=col_idxs,
            rows_changed=rows_changed,
            columns_changed=columns_changed,
        )
        if _self._call_seq is not None:
            new_call_seq = ArrayWrapper.select_from_flex_array(
                _self._call_seq,
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
        else:
            new_call_seq = None
        if _self._bm_close is not None and not isinstance(_self._bm_close, bool):
            new_bm_close = ArrayWrapper.select_from_flex_array(
                _self._bm_close,
                row_idxs=row_idxs,
                col_idxs=col_idxs,
                rows_changed=rows_changed,
                columns_changed=columns_changed,
            )
        else:
            new_bm_close = _self._bm_close
        new_in_outputs = _self.in_outputs_indexing_func(wrapper_meta, **resolve_dict(in_output_kwargs))
        new_sim_start = _self.sim_start_indexing_func(wrapper_meta)
        new_sim_end = _self.sim_end_indexing_func(wrapper_meta)

        if self.weights is not None:
            new_weights = to_1d_array(self.weights)
            if columns_changed and new_weights.shape[0] > 1:
                new_weights = new_weights[col_idxs]
        else:
            new_weights = self._weights

        return self.replace(
            wrapper=new_wrapper,
            order_records=new_order_records,
            open=new_open,
            high=new_high,
            low=new_low,
            close=new_close,
            log_records=new_log_records,
            init_cash=new_init_cash,
            init_position=new_init_position,
            init_price=new_init_price,
            cash_deposits=new_cash_deposits,
            cash_earnings=new_cash_earnings,
            call_seq=new_call_seq,
            in_outputs=new_in_outputs,
            bm_close=new_bm_close,
            sim_start=new_sim_start,
            sim_end=new_sim_end,
            weights=new_weights,
        )

    # ############# Resampling ############# #

    def resample_obj(
        self,
        obj: tp.Any,
        resampler: tp.Union[Resampler, tp.PandasResampler],
        obj_name: tp.Optional[str] = None,
        obj_type: tp.Optional[str] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        resample_func: tp.Union[None, str, tp.Callable] = None,
        resample_kwargs: tp.KwargsLike = None,
        force_resampling: bool = False,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Any:
        """Resample an object.

        This function resamples the given object using the specified resampler and resampling function.
        If `resample_func` is a string, it is used as `reduce_func_nb` in
        `vectorbtpro.generic.accessors.GenericAccessor.resample_apply` (default is "last").
        If the object is None, a boolean, or an empty array, it is returned unchanged.

        Args:
            obj (Any): The object to be resampled.
            resampler (Union[Resampler, PandasResampler]): The resampler instance or configuration.
            obj_name (str): Name of the object for reference.
            obj_type (str): Identifier specifying the type of the object.
            wrapper (ArrayWrapper): ArrayWrapper instance; if None, the portfolio's wrapper is used.
            group_by (GroupByLike): Grouping specification.
            resample_func (Union[None, str, Callable]): Function or name used for resampling.
            resample_kwargs (KwargsLike): Additional keyword arguments for the resampling function.
            force_resampling (bool): Flag to force resampling if conditions are met.
            silence_warnings (bool): If True, suppresses warnings during resampling.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The resampled object.
        """
        if obj is None or isinstance(obj, bool) or (checks.is_np_array(obj) and obj.size == 0):
            return obj
        if wrapper is None:
            wrapper = self.wrapper
        if resample_func is None:
            resample_func = "last"
        if not isinstance(resample_func, str):
            return resample_func(
                self,
                obj,
                resampler,
                obj_name=obj_name,
                obj_type=obj_type,
                wrapper=wrapper,
                group_by=group_by,
                resample_kwargs=resample_kwargs,
                force_resampling=force_resampling,
                silence_warnings=silence_warnings,
                **kwargs,
            )

        def _resample(obj: tp.Array) -> tp.SeriesFrame:
            wrapped_obj = ArrayWrapper.from_obj(obj, index=wrapper.index).wrap(obj)
            return wrapped_obj.vbt.resample_apply(resampler, resample_func, **resolve_dict(resample_kwargs)).values

        if obj_type is not None and obj_type == "red_array":
            return obj
        if obj_type is None or obj_type == "array":
            is_grouped = wrapper.grouper.is_grouped(group_by=group_by)
            if checks.is_np_array(obj):
                if is_grouped:
                    if to_2d_shape(obj.shape) == wrapper.get_shape_2d():
                        return _resample(obj)
                    if obj.shape == (wrapper.get_shape_2d()[1],):
                        return obj
                if to_2d_shape(obj.shape) == wrapper.shape_2d:
                    return _resample(obj)
                if obj.shape == (wrapper.shape_2d[1],):
                    return obj
        if force_resampling:
            raise NotImplementedError(f"Cannot resample object '{obj_name}'")
        if not silence_warnings:
            warn(f"Cannot figure out how to resample object '{obj_name}'")
        return obj

    def resample_in_outputs(
        self,
        resampler: tp.Union[Resampler, tp.PandasResampler],
        **kwargs,
    ) -> tp.Optional[tp.NamedTuple]:
        """Resample `Portfolio.in_outputs` using the provided resampler.

        For each field in `Portfolio.in_outputs`, this method merges options obtained from
        `Portfolio.parse_field_options` with those from `Portfolio.in_output_config` and then
        applies the resampling function via `Portfolio.resample_obj`.

        Args:
            resampler (Union[Resampler, PandasResampler]): Resampler instance used for down/up-sampling.
            **kwargs: Additional keyword arguments for resampling.

        Returns:
            Optional[NamedTuple]: A resampled in-outputs object created from the existing outputs,
                or None if `Portfolio.in_outputs` is not set.
        """
        if self.in_outputs is None:
            return None

        new_in_outputs = {}
        for field, obj in self.in_outputs._asdict().items():
            field_options = merge_dicts(
                self.parse_field_options(field),
                self.in_output_config.get(field, None),
            )
            if field_options.get("field", field) in self.cls_dir:
                prop = getattr(type(self), field_options["field"])
                prop_options = getattr(prop, "options", {})
                obj_type = prop_options.get("obj_type", "array")
                resample_func = prop_options.get("resample_func", None)
                resample_kwargs = prop_options.get("resample_kwargs", None)
                force_resampling = prop_options.get("force_resampling", False)
                silence_warnings = prop_options.get("silence_warnings", False)
            else:
                obj_type = None
                resample_func = None
                resample_kwargs = None
                force_resampling = False
                silence_warnings = False
            _kwargs = merge_dicts(
                dict(
                    obj_name=field_options.get("field", field),
                    obj_type=field_options.get("obj_type", obj_type),
                    resample_func=field_options.get("resample_func", resample_func),
                    resample_kwargs=field_options.get("resample_kwargs", resample_kwargs),
                    force_resampling=field_options.get("force_resampling", force_resampling),
                    silence_warnings=field_options.get("silence_warnings", silence_warnings),
                ),
                kwargs,
            )
            new_obj = self.resample_obj(obj, resampler, **_kwargs)
            new_in_outputs[field] = new_obj
        return type(self.in_outputs)(**new_in_outputs)

    def resample(
        self: PortfolioT,
        *args,
        ffill_close: bool = False,
        fbfill_close: bool = False,
        in_output_kwargs: tp.KwargsLike = None,
        wrapper_meta: tp.DictLike = None,
        **kwargs,
    ) -> PortfolioT:
        """Resample the `Portfolio` instance.

        Resamples various portfolio components including price series, order and log records,
        cash deposits, cash earnings, benchmark data, and in-outputs. Downsampling is performed
        using methods provided by the portfolio's wrapper, with optional forward-fill or backward-fill
        applied to the close prices.

        !!! warning
            Downsampling is associated with information loss:

            * Cash deposits and earnings are assumed to be added/removed at the beginning of each time step.
                Imagine depositing $100 and using them up in the same bar, and then depositing another $100
                and using them up. Downsampling both bars into a single bar will aggregate cash deposits
                and earnings, and assign both to the beginning of the new bar, even though the second 
                deposit was made later.
            * Market/benchmark returns are computed by applying the initial close price of the first bar
                and tracking the price change to simulate holding. Moving the close price of the first bar 
                further into the future will affect this computation and likely produce different market 
                values and returns. To mitigate this, ensure that the downsampled index's first bar contains 
                only the first bar from the original timeframe.

        Args:
            *args: Positional arguments passed to 
                `vectorbtpro.base.wrapping.ArrayWrapper.resample_meta`.
            ffill_close (bool): Flag to apply forward-fill on the close prices after resampling.
            fbfill_close (bool): Flag to apply backward-fill on the close prices after resampling.
            in_output_kwargs (KwargsLike): Additional keyword arguments for resampling in-outputs.
            wrapper_meta (DictLike): Metadata for resampling obtained from the portfolio's wrapper.
            **kwargs: Additional keyword arguments passed to 
                `vectorbtpro.base.wrapping.ArrayWrapper.resample_meta`.

        Returns:
            Portfolio: A new resampled `Portfolio` instance.
        """
        _self = self.disable_weights()

        if _self._call_seq is not None:
            raise ValueError("Cannot resample call_seq")
        if wrapper_meta is None:
            wrapper_meta = _self.wrapper.resample_meta(*args, **kwargs)
        resampler = wrapper_meta["resampler"]
        new_wrapper = wrapper_meta["new_wrapper"]

        new_close = _self.close.vbt.resample_apply(resampler, "last")
        if fbfill_close:
            new_close = new_close.vbt.fbfill()
        elif ffill_close:
            new_close = new_close.vbt.ffill()
        new_close = new_close.values
        if _self._open is not None:
            new_open = _self.open.vbt.resample_apply(resampler, "first").values
        else:
            new_open = _self._open
        if _self._high is not None:
            new_high = _self.high.vbt.resample_apply(resampler, "max").values
        else:
            new_high = _self._high
        if _self._low is not None:
            new_low = _self.low.vbt.resample_apply(resampler, "min").values
        else:
            new_low = _self._low
        new_order_records = _self.orders.resample_records_arr(resampler)
        new_log_records = _self.logs.resample_records_arr(resampler)
        if _self._cash_deposits.size > 1 or _self._cash_deposits.item() != 0:
            new_cash_deposits = _self.get_cash_deposits(group_by=None if _self.cash_sharing else False)
            new_cash_deposits = new_cash_deposits.vbt.resample_apply(resampler, generic_nb.sum_reduce_nb)
            new_cash_deposits = new_cash_deposits.fillna(0.0)
            new_cash_deposits = new_cash_deposits.values
        else:
            new_cash_deposits = _self._cash_deposits
        if _self._cash_earnings.size > 1 or _self._cash_earnings.item() != 0:
            new_cash_earnings = _self.get_cash_earnings(group_by=False)
            new_cash_earnings = new_cash_earnings.vbt.resample_apply(resampler, generic_nb.sum_reduce_nb)
            new_cash_earnings = new_cash_earnings.fillna(0.0)
            new_cash_earnings = new_cash_earnings.values
        else:
            new_cash_earnings = _self._cash_earnings
        if _self._bm_close is not None and not isinstance(_self._bm_close, bool):
            new_bm_close = _self.bm_close.vbt.resample_apply(resampler, "last")
            if fbfill_close:
                new_bm_close = new_bm_close.vbt.fbfill()
            elif ffill_close:
                new_bm_close = new_bm_close.vbt.ffill()
            new_bm_close = new_bm_close.values
        else:
            new_bm_close = _self._bm_close
        if _self._in_outputs is not None:
            new_in_outputs = _self.resample_in_outputs(resampler, **resolve_dict(in_output_kwargs))
        else:
            new_in_outputs = None
        new_sim_start = _self.resample_sim_start(new_wrapper)
        new_sim_end = _self.resample_sim_end(new_wrapper)

        return self.replace(
            wrapper=new_wrapper,
            order_records=new_order_records,
            open=new_open,
            high=new_high,
            low=new_low,
            close=new_close,
            log_records=new_log_records,
            cash_deposits=new_cash_deposits,
            cash_earnings=new_cash_earnings,
            in_outputs=new_in_outputs,
            bm_close=new_bm_close,
            sim_start=new_sim_start,
            sim_end=new_sim_end,
        )

    # ############# Class methods ############# #

    @classmethod
    def from_orders(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, OHLCDataMixin, FOPreparer, PFPrepResult],
        size: tp.Optional[tp.ArrayLike] = None,
        size_type: tp.Optional[tp.ArrayLike] = None,
        direction: tp.Optional[tp.ArrayLike] = None,
        price: tp.Optional[tp.ArrayLike] = None,
        fees: tp.Optional[tp.ArrayLike] = None,
        fixed_fees: tp.Optional[tp.ArrayLike] = None,
        slippage: tp.Optional[tp.ArrayLike] = None,
        min_size: tp.Optional[tp.ArrayLike] = None,
        max_size: tp.Optional[tp.ArrayLike] = None,
        size_granularity: tp.Optional[tp.ArrayLike] = None,
        leverage: tp.Optional[tp.ArrayLike] = None,
        leverage_mode: tp.Optional[tp.ArrayLike] = None,
        reject_prob: tp.Optional[tp.ArrayLike] = None,
        price_area_vio_mode: tp.Optional[tp.ArrayLike] = None,
        allow_partial: tp.Optional[tp.ArrayLike] = None,
        raise_reject: tp.Optional[tp.ArrayLike] = None,
        log: tp.Optional[tp.ArrayLike] = None,
        val_price: tp.Optional[tp.ArrayLike] = None,
        from_ago: tp.Optional[tp.ArrayLike] = None,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        init_cash: tp.Optional[tp.ArrayLike] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_earnings: tp.Optional[tp.ArrayLike] = None,
        cash_dividends: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        call_seq: tp.Optional[tp.ArrayLike] = None,
        attach_call_seq: tp.Optional[bool] = None,
        ffill_val_price: tp.Optional[bool] = None,
        update_value: tp.Optional[bool] = None,
        save_state: tp.Optional[bool] = None,
        save_value: tp.Optional[bool] = None,
        save_returns: tp.Optional[bool] = None,
        skip_empty: tp.Optional[bool] = None,
        max_order_records: tp.Optional[int] = None,
        max_log_records: tp.Optional[int] = None,
        seed: tp.Optional[int] = None,
        group_by: tp.GroupByLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        bm_close: tp.Optional[tp.ArrayLike] = None,
        records: tp.Optional[tp.RecordsLike] = None,
        return_preparer: bool = False,
        return_prep_result: bool = False,
        return_sim_out: bool = False,
        **kwargs,
    ) -> PortfolioResultT:
        """Simulate a portfolio from orders using size, price, fees, and other parameters.

        This method simulates portfolio evolution based on provided order inputs.
        Order parameters such as size, price, fees, and others are broadcast as necessary and processed
        to construct the resulting portfolio.

        See:
            `vectorbtpro.portfolio.nb.from_orders.from_orders_nb`

        For defaults, see `vectorbtpro._settings.portfolio`. These defaults are not used to fill NaN values 
        after reindexing; vectorbtpro uses its own defaults (typically NaN for floating arrays and preset 
        flags for integer arrays). Use `vectorbtpro.base.reshaping.BCO` with `fill_value` to override.

        !!! note
            When `call_seq` is not set to `CallSeqType.Auto`, the processing order within a group strictly
            follows the specified `call_seq`. This means the last asset in the sequence is processed only after
            the others, which can affect rebalancing. Use `CallSeqType.Auto` for dynamic execution order.

        !!! hint
            All broadcastable arguments are handled using `vectorbtpro.base.reshaping.broadcast` 
            to preserve their original shapes for flexible indexing and memory efficiency. 
            Each can be provided per frame, series, row, column, or individual element.
    
        Args:
            close (Union[ArrayLike, OHLCDataMixin, FOPreparer, PFPrepResult]): The close prices or 
                OHLC data used for portfolio simulation.
                
                Broadcasts.
            
                * If an instance of `vectorbtpro.data.base.OHLCDataMixin`, extracts open, high, low, and close prices.
                * If an instance of `vectorbtpro.portfolio.preparing.FOPreparer`, it is used as a preparer.
                * If an instance of `vectorbtpro.portfolio.preparing.PFPrepResult`, it is used as a preparer result.
            size (Optional[ArrayLike]): Size to order. 
            
                Broadcasts. See `vectorbtpro.portfolio.enums.Order.size`. 
            size_type (Optional[ArrayLike]): Order size type. 
            
                Broadcasts. See `vectorbtpro.portfolio.enums.SizeType` and 
                `vectorbtpro.portfolio.enums.Order.size_type`. 
            direction (Optional[ArrayLike]): Order direction. 
            
                Broadcasts. See `vectorbtpro.portfolio.enums.Direction` and 
                `vectorbtpro.portfolio.enums.Order.direction`.
            price (Optional[ArrayLike]): Order price.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.price` and 
                `vectorbtpro.portfolio.enums.PriceType`.
            
                Options such as `PriceType.NextOpen` and `PriceType.NextClose` apply per column and 
                require `from_ago` to be None.
            fees (Optional[ArrayLike]): Fees as a percentage of the order value.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.fees`. 
            fixed_fees (Optional[ArrayLike]): Fixed fee amount per order.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.fixed_fees`. 
            slippage (Optional[ArrayLike]): Slippage percentage of the order price.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.slippage`. 
            min_size (Optional[ArrayLike]): Minimum order size.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.min_size`.
            max_size (Optional[ArrayLike]): Maximum order size.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.max_size`.
            size_granularity (Optional[ArrayLike]): Granularity of the order size.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.size_granularity`.
            leverage (Optional[ArrayLike]): Leverage applied in the order.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.leverage`.
            leverage_mode (Optional[ArrayLike]): Leverage mode.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.leverage_mode`.
            reject_prob (Optional[ArrayLike]): Order rejection probability.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.reject_prob`.
            price_area_vio_mode (Optional[ArrayLike]): Price area violation mode.

                Broadcasts. See `vectorbtpro.portfolio.enums.PriceAreaVioMode`.
            allow_partial (Optional[ArrayLike]): Indicates whether partial fills are allowed.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.allow_partial`.
            raise_reject (Optional[ArrayLike]): Indicates whether to raise an exception upon order rejection.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.raise_reject`.
            log (Optional[ArrayLike]): Flag indicating whether to log orders.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.log`.
            val_price (Optional[ArrayLike]): Asset valuation price used in decision making.
            
                Broadcasts. Can also be provided as `vectorbtpro.portfolio.enums.ValPriceType`.
            
                * Any `-np.inf` element is replaced by the latest valuation price 
                    (using `open` or a previously known value if `ffill_val_price` is True).
                * Any `np.inf` element is replaced by the current order price.
            
                !!! note
                    Unlike `Portfolio.from_order_func`, the order price is effectively predetermined,
                    so `val_price` defaults to the current order price when using `np.inf`. 
                    To use the previous close, set it in the settings to `-np.inf`.
            
                !!! note
                    Ensure that the timestamp associated with `val_price` precedes all order 
                    timestamps in a cash-sharing group.
            open (Optional[ArrayLike]): Opening asset price at each time step.

                Broadcasts. Used as a price boundary (see `vectorbtpro.portfolio.enums.PriceArea`).
            high (Optional[ArrayLike]): Highest asset price at each time step.

                Broadcasts. Used as a price boundary (see `vectorbtpro.portfolio.enums.PriceArea`).
            low (Optional[ArrayLike]): Lowest asset price at each time step.

                Broadcasts. Used as a price boundary (see `vectorbtpro.portfolio.enums.PriceArea`).
            init_cash (Optional[ArrayLike]): Initial capital.
            
                Broadcasts to the final number of columns, or to the number of groups if 
                cash sharing is enabled. See `vectorbtpro.portfolio.enums.InitCashMode` for guidance.
            
                !!! note
                    When using `InitCashMode.AutoAlign`, initial cash values are synchronized across 
                    columns/groups after initialization.
            init_position (Optional[ArrayLike]): Initial position. 
            
                Broadcasts to match the final number of columns.
            init_price (Optional[ArrayLike]): Initial position price. 
            
                Broadcasts to match the final number of columns.
            cash_deposits (Optional[ArrayLike]): Cash deposits or withdrawals at the beginning of each timestamp.
            
                Broadcasts to match the shape of `init_cash`.
            cash_earnings (Optional[ArrayLike]): Cash earnings added at the end of each timestamp.
            
                Broadcasts.
            cash_dividends (Optional[ArrayLike]): Cash dividends added at the end of each timestamp.
            
                Broadcasts, are multiplied by the position, and then added to `cash_earnings`.
            cash_sharing (Optional[bool]): Indicates whether cash is shared among assets within the same group.
            
                If `group_by` is None and this is True, all assets are grouped together for cash sharing.
            
                !!! warning
                    Enables cross-asset dependencies by assuming that all orders in a cash-sharing group 
                    execute in the same tick and retain their price.
            from_ago (Optional[ArrayLike]): Number of time steps to look back for order information.
            
                Negative values are converted to positive to avoid look-ahead bias.
            sim_start (Optional[ArrayLike]): Simulation start row or index (inclusive).
            
                Can be "auto" to automatically select the first non-NA size value.
            sim_end (Optional[ArrayLike]): Simulation end row or index (exclusive).
            
                Can be "auto" to automatically select the first non-NA size value.
            call_seq (Optional[ArrayLike]): Sequence dictating the order in which columns are 
                processed per row and group.
            
                Each element specifies the position of a column in the processing order.
                Options include:
            
                * None: Generates a default call sequence.
                * A value from `vectorbtpro.portfolio.enums.CallSeqType`: Creates a full array of the specified type.
                * Custom array: Specifies a user-defined call sequence.
            
                If set to `CallSeqType.Auto`, orders are rearranged dynamically so that sell orders are 
                processed before buy orders.
            
                !!! warning
                    `CallSeqType.Auto` assumes predetermined order prices and flexible execution, 
                    which may not accurately reflect real-time conditions.
                    For stricter control, use `Portfolio.from_order_func`.
            attach_call_seq (Optional[bool]): Indicates whether to attach the computed call sequence 
                to the portfolio instance.
            ffill_val_price (Optional[bool]): If True, tracks the valuation price only when available 
                to prevent propagation of NaN values.
            update_value (Optional[bool]): If True, updates the group value after each filled order.
            save_state (Optional[bool]): If True, saves state arrays 
                (such as `cash`, `position`, `debt`, `locked_cash`, and `free_cash`) in in-outputs.
            save_value (Optional[bool]): If True, saves the portfolio value in in-outputs.
            save_returns (Optional[bool]): If True, saves the returns array in in-outputs.
            skip_empty (Optional[bool]): If True, skips processing rows that do not contain any orders.
            max_order_records (Optional[int]): Maximum number of order records expected per column.
            
                Set to a lower number to conserve memory, or 0 to disable order record collection.
            max_log_records (Optional[int]): Maximum number of log records expected per column.
            
                Set to a lower number to conserve memory, or 0 to disable logging.
            seed (Optional[int]): Seed value for generating the call sequence and initializing the simulation.
            group_by (tp.GroupByLike): Grouping specification for columns.
            
                See `vectorbtpro.base.grouping.base.Grouper` for details.
            broadcast_kwargs (tp.KwargsLike): Additional keyword arguments to pass to 
                `vectorbtpro.base.reshaping.broadcast`.
            jitted (tp.JittedOption): Option to control JIT compilation.
            
                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (tp.ChunkedOption): Option to control chunked processing.
            
                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            bm_close (Optional[ArrayLike]): Benchmark asset price at each time step.
            
                Broadcasts. If not provided, `close` is used; if set to False, benchmarking is disabled.
            records (Optional[tp.RecordsLike]): Records used to construct arrays.
            
                See `vectorbtpro.base.indexing.IdxRecords`.
            return_preparer (bool): If True, returns the preparer instance 
                (`vectorbtpro.portfolio.preparing.FOPreparer`).
            
                !!! note
                    In this case, the seed is not automatically set; 
                    invoke `preparer.set_seed()` explicitly if needed.
            return_prep_result (bool): If True, returns the preparer result 
                (`vectorbtpro.portfolio.preparing.PFPrepResult`).
            return_sim_out (bool): If True, returns the simulation output 
                (`vectorbtpro.portfolio.enums.SimulationOutput`).
            **kwargs: Additional keyword arguments passed to the `Portfolio` constructor.
    
        Returns:
            PortfolioResult: The portfolio result.

        Examples:
            Buy 10 units each tick:

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_orders(close, 10)

            >>> pf.assets
            0    10.0
            1    20.0
            2    30.0
            3    40.0
            4    40.0
            dtype: float64
            >>> pf.cash
            0    90.0
            1    70.0
            2    40.0
            3     0.0
            4     0.0
            dtype: float64
            ```

            Reverse each position by first closing it:

            ```pycon
            >>> size = [1, 0, -1, 0, 1]
            >>> pf = vbt.Portfolio.from_orders(close, size, size_type='targetpercent')

            >>> pf.assets
            0    100.000000
            1      0.000000
            2    -66.666667
            3      0.000000
            4     26.666667
            dtype: float64
            >>> pf.cash
            0      0.000000
            1    200.000000
            2    400.000000
            3    133.333333
            4      0.000000
            dtype: float64
            ```

            Regularly deposit cash at open and invest it within the same bar at close:

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> cash_deposits = pd.Series([10., 0., 10., 0., 10.])
            >>> pf = vbt.Portfolio.from_orders(
            ...     close,
            ...     size=cash_deposits,  # invest the amount deposited
            ...     size_type='value',
            ...     cash_deposits=cash_deposits
            ... )

            >>> pf.cash
            0    100.0
            1    100.0
            2    100.0
            3    100.0
            4    100.0
            dtype: float64

            >>> pf.asset_flow
            0    10.000000
            1     0.000000
            2     3.333333
            3     0.000000
            4     2.000000
            dtype: float64
            ```

            Equal-weighted portfolio as in `vectorbtpro.portfolio.nb.from_order_func.from_order_func_nb` example
            (it's more compact but has less control over execution):

            ```pycon
            >>> np.random.seed(42)
            >>> close = pd.DataFrame(np.random.uniform(1, 10, size=(5, 3)))
            >>> size = pd.Series(np.full(5, 1/3))  # each column 33.3%
            >>> size[1::2] = np.nan  # skip every second tick

            >>> pf = vbt.Portfolio.from_orders(
            ...     close,  # acts both as reference and order price here
            ...     size,
            ...     size_type='targetpercent',
            ...     direction='longonly',
            ...     call_seq='auto',  # first sell then buy
            ...     group_by=True,  # one group
            ...     cash_sharing=True,  # assets share the same cash
            ...     fees=0.001, fixed_fees=1., slippage=0.001  # costs
            ... )

            >>> pf.get_asset_value(group_by=False).vbt.plot().show()
            ```

            ![](/assets/images/api/from_orders.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/from_orders.dark.svg#only-dark){: .iimg loading=lazy }

            Test 10 random weight combinations:

            ```pycon
            >>> np.random.seed(42)
            >>> close = pd.DataFrame(
            ...     np.random.uniform(1, 10, size=(5, 3)),
            ...     columns=pd.Index(['a', 'b', 'c'], name='asset'))

            >>> # Generate random weight combinations
            >>> rand_weights = []
            >>> for i in range(10):
            ...     rand_weights.append(np.random.dirichlet(np.ones(close.shape[1]), size=1)[0])
            >>> rand_weights
            [array([0.15474873, 0.27706078, 0.5681905 ]),
             array([0.30468598, 0.18545189, 0.50986213]),
             array([0.15780486, 0.36292607, 0.47926907]),
             array([0.25697713, 0.64902589, 0.09399698]),
             array([0.43310548, 0.53836359, 0.02853093]),
             array([0.78628605, 0.15716865, 0.0565453 ]),
             array([0.37186671, 0.42150531, 0.20662798]),
             array([0.22441579, 0.06348919, 0.71209502]),
             array([0.41619664, 0.09338007, 0.49042329]),
             array([0.01279537, 0.87770864, 0.10949599])]

            >>> # Bring close and rand_weights to the same shape
            >>> rand_weights = np.concatenate(rand_weights)
            >>> close = close.vbt.tile(10, keys=pd.Index(np.arange(10), name='weights_vector'))
            >>> size = vbt.broadcast_to(weights, close).copy()
            >>> size[1::2] = np.nan
            >>> size
            weights_vector                            0  ...                               9
            asset                  a         b        c  ...           a         b         c
            0               0.154749  0.277061  0.56819  ...    0.012795  0.877709  0.109496
            1                    NaN       NaN      NaN  ...         NaN       NaN       NaN
            2               0.154749  0.277061  0.56819  ...    0.012795  0.877709  0.109496
            3                    NaN       NaN      NaN  ...         NaN       NaN       NaN
            4               0.154749  0.277061  0.56819  ...    0.012795  0.877709  0.109496

            [5 rows x 30 columns]

            >>> pf = vbt.Portfolio.from_orders(
            ...     close,
            ...     size,
            ...     size_type='targetpercent',
            ...     direction='longonly',
            ...     call_seq='auto',
            ...     group_by='weights_vector',  # group by column level
            ...     cash_sharing=True,
            ...     fees=0.001, fixed_fees=1., slippage=0.001
            ... )

            >>> pf.total_return
            weights_vector
            0   -0.294372
            1    0.139207
            2   -0.281739
            3    0.041242
            4    0.467566
            5    0.829925
            6    0.320672
            7   -0.087452
            8    0.376681
            9   -0.702773
            Name: total_return, dtype: float64
            ```
        """
        if isinstance(close, FOPreparer):
            preparer = close
            prep_result = None
        elif isinstance(close, PFPrepResult):
            preparer = None
            prep_result = close
        else:
            local_kwargs = locals()
            local_kwargs = {**local_kwargs, **local_kwargs["kwargs"]}
            del local_kwargs["kwargs"]
            del local_kwargs["cls"]
            del local_kwargs["return_preparer"]
            del local_kwargs["return_prep_result"]
            del local_kwargs["return_sim_out"]
            parsed_data = BasePFPreparer.parse_data(close, all_ohlc=True)
            if parsed_data is not None:
                local_kwargs["data"] = parsed_data
                local_kwargs["close"] = None
            preparer = FOPreparer(**local_kwargs)
            if not return_preparer:
                preparer.set_seed()
            prep_result = None
        if return_preparer:
            return preparer
        if prep_result is None:
            prep_result = preparer.result
        if return_prep_result:
            return prep_result
        sim_out = prep_result.target_func(**prep_result.target_args)
        if return_sim_out:
            return sim_out
        return cls(order_records=sim_out, **prep_result.pf_args)

    @classmethod
    def from_signals(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, OHLCDataMixin, FSPreparer, PFPrepResult],
        entries: tp.Optional[tp.ArrayLike] = None,
        exits: tp.Optional[tp.ArrayLike] = None,
        *,
        direction: tp.Optional[tp.ArrayLike] = None,
        long_entries: tp.Optional[tp.ArrayLike] = None,
        long_exits: tp.Optional[tp.ArrayLike] = None,
        short_entries: tp.Optional[tp.ArrayLike] = None,
        short_exits: tp.Optional[tp.ArrayLike] = None,
        adjust_func_nb: tp.Union[None, tp.PathLike, nb.AdjustFuncT] = None,
        adjust_args: tp.Args = (),
        signal_func_nb: tp.Union[None, tp.PathLike, nb.SignalFuncT] = None,
        signal_args: tp.ArgsLike = (),
        post_segment_func_nb: tp.Union[None, tp.PathLike, nb.PostSegmentFuncT] = None,
        post_segment_args: tp.ArgsLike = (),
        order_mode: bool = False,
        size: tp.Optional[tp.ArrayLike] = None,
        size_type: tp.Optional[tp.ArrayLike] = None,
        price: tp.Optional[tp.ArrayLike] = None,
        fees: tp.Optional[tp.ArrayLike] = None,
        fixed_fees: tp.Optional[tp.ArrayLike] = None,
        slippage: tp.Optional[tp.ArrayLike] = None,
        min_size: tp.Optional[tp.ArrayLike] = None,
        max_size: tp.Optional[tp.ArrayLike] = None,
        size_granularity: tp.Optional[tp.ArrayLike] = None,
        leverage: tp.Optional[tp.ArrayLike] = None,
        leverage_mode: tp.Optional[tp.ArrayLike] = None,
        reject_prob: tp.Optional[tp.ArrayLike] = None,
        price_area_vio_mode: tp.Optional[tp.ArrayLike] = None,
        allow_partial: tp.Optional[tp.ArrayLike] = None,
        raise_reject: tp.Optional[tp.ArrayLike] = None,
        log: tp.Optional[tp.ArrayLike] = None,
        val_price: tp.Optional[tp.ArrayLike] = None,
        accumulate: tp.Optional[tp.ArrayLike] = None,
        upon_long_conflict: tp.Optional[tp.ArrayLike] = None,
        upon_short_conflict: tp.Optional[tp.ArrayLike] = None,
        upon_dir_conflict: tp.Optional[tp.ArrayLike] = None,
        upon_opposite_entry: tp.Optional[tp.ArrayLike] = None,
        order_type: tp.Optional[tp.ArrayLike] = None,
        limit_delta: tp.Optional[tp.ArrayLike] = None,
        limit_tif: tp.Optional[tp.ArrayLike] = None,
        limit_expiry: tp.Optional[tp.ArrayLike] = None,
        limit_reverse: tp.Optional[tp.ArrayLike] = None,
        limit_order_price: tp.Optional[tp.ArrayLike] = None,
        upon_adj_limit_conflict: tp.Optional[tp.ArrayLike] = None,
        upon_opp_limit_conflict: tp.Optional[tp.ArrayLike] = None,
        use_stops: tp.Optional[bool] = None,
        stop_ladder: tp.Optional[bool] = None,
        sl_stop: tp.Optional[tp.ArrayLike] = None,
        tsl_stop: tp.Optional[tp.ArrayLike] = None,
        tsl_th: tp.Optional[tp.ArrayLike] = None,
        tp_stop: tp.Optional[tp.ArrayLike] = None,
        td_stop: tp.Optional[tp.ArrayLike] = None,
        dt_stop: tp.Optional[tp.ArrayLike] = None,
        stop_entry_price: tp.Optional[tp.ArrayLike] = None,
        stop_exit_price: tp.Optional[tp.ArrayLike] = None,
        stop_exit_type: tp.Optional[tp.ArrayLike] = None,
        stop_order_type: tp.Optional[tp.ArrayLike] = None,
        stop_limit_delta: tp.Optional[tp.ArrayLike] = None,
        upon_stop_update: tp.Optional[tp.ArrayLike] = None,
        upon_adj_stop_conflict: tp.Optional[tp.ArrayLike] = None,
        upon_opp_stop_conflict: tp.Optional[tp.ArrayLike] = None,
        delta_format: tp.Optional[tp.ArrayLike] = None,
        time_delta_format: tp.Optional[tp.ArrayLike] = None,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        init_cash: tp.Optional[tp.ArrayLike] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_earnings: tp.Optional[tp.ArrayLike] = None,
        cash_dividends: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = None,
        from_ago: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        call_seq: tp.Optional[tp.ArrayLike] = None,
        attach_call_seq: tp.Optional[bool] = None,
        ffill_val_price: tp.Optional[bool] = None,
        update_value: tp.Optional[bool] = None,
        fill_pos_info: tp.Optional[bool] = None,
        save_state: tp.Optional[bool] = None,
        save_value: tp.Optional[bool] = None,
        save_returns: tp.Optional[bool] = None,
        skip_empty: tp.Optional[bool] = None,
        max_order_records: tp.Optional[int] = None,
        max_log_records: tp.Optional[int] = None,
        in_outputs: tp.Optional[tp.MappingLike] = None,
        seed: tp.Optional[int] = None,
        group_by: tp.GroupByLike = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        staticized: tp.StaticizedOption = None,
        bm_close: tp.Optional[tp.ArrayLike] = None,
        records: tp.Optional[tp.RecordsLike] = None,
        return_preparer: bool = False,
        return_prep_result: bool = False,
        return_sim_out: bool = False,
        **kwargs,
    ) -> PortfolioResultT:
        """Simulate a portfolio based on provided entry and exit signals.

        This method simulates a portfolio by interpreting various signal arrays and custom signal
        functions. It supports several modes:

        **Using `entries` and `exits`:**

        * If an adjustment function is provided (non-cacheable), signals are processed via
            `vectorbtpro.portfolio.nb.from_signals.dir_signal_func_nb`.
        * Otherwise, signals are translated using 
            `vectorbtpro.portfolio.nb.from_signals.dir_to_ls_signals_nb` 
            and simulated statically (cacheable).

        **Using `entries` (for long), `exits` (for long), `short_entries`, and `short_exits`:**

        * If an adjustment function is provided, the function 
            `vectorbtpro.portfolio.nb.from_signals.ls_signal_func_nb`
            is used; otherwise, simulation is executed statically.

        **Order Mode (`order_mode=True`):**

        * Simulates without explicit signals using 
        `vectorbtpro.portfolio.nb.from_signals.order_signal_func_nb` (non-cacheable).

        **Custom Signal Function:**

        * When `signal_func_nb` and `signal_args` are provided, the custom signal 
            function is used (non-cacheable).

        Prepared using `vectorbtpro.portfolio.preparing.FSPreparer`.

        For defaults, see `vectorbtpro._settings.portfolio`. These defaults are not used to fill NaN values 
        after reindexing; vectorbtpro uses its own defaults (typically NaN for floating arrays and preset 
        flags for integer arrays). Use `vectorbtpro.base.reshaping.BCO` with `fill_value` to override.

        !!! note
            When `call_seq` is not set to `CallSeqType.Auto`, the processing order within a group strictly
            follows the specified `call_seq`. This means the last asset in the sequence is processed only after
            the others, which can affect rebalancing. Use `CallSeqType.Auto` for dynamic execution order.

        !!! hint
            All broadcastable arguments are handled using `vectorbtpro.base.reshaping.broadcast` 
            to preserve their original shapes for flexible indexing and memory efficiency. 
            Each can be provided per frame, series, row, column, or individual element.

        Args:
            close (Union[ArrayLike, OHLCDataMixin, FSPreparer, PFPrepResult]): The close prices or 
                OHLC data used for portfolio simulation.

                Broadcasts.
            
                * If an instance of `vectorbtpro.data.base.OHLCDataMixin`, extracts open, high, low, and close prices.
                * If an instance of `vectorbtpro.portfolio.preparing.FOPreparer`, it is used as a preparer.
                * If an instance of `vectorbtpro.portfolio.preparing.PFPrepResult`, it is used as a preparer result.
            entries (Optional[ArrayLike]): Boolean array of entry signals.

                Broadcasts. If all other signal arrays are missing, treated as True; otherwise, as False.
            
                * When `short_entries` and `short_exits` are not provided, acts as a long signal 
                    if `direction` is 'all' or 'longonly', otherwise as a short signal.
                * If `short_entries` or `short_exits` are provided, interpreted as `long_entries`.
            exits (Optional[ArrayLike]): Boolean array of exit signals.

                Broadcasts. If all other signal arrays are missing, treated as False.
            
                * When `short_entries` and `short_exits` are not provided, acts as a short signal 
                    if `direction` is 'all' or 'longonly', otherwise as a long signal.
                * If `short_entries` or `short_exits` are provided, interpreted as `long_exits`.
            direction (Optional[ArrayLike]): Trading direction.

                Broadcasts. See `vectorbtpro.portfolio.enums.Direction` and 
                `vectorbtpro.portfolio.enums.Order.direction`.
                
                Takes effect only if `short_entries` and `short_exits` are not provided.
            long_entries (Optional[ArrayLike]): Boolean array of long entry signals. 
            
                Broadcasts.
            long_exits (Optional[ArrayLike]): Boolean array of long exit signals. 
                
                Broadcasts.
            short_entries (Optional[ArrayLike]): Boolean array of short entry signals. 
            
                Broadcasts.
            short_exits (Optional[ArrayLike]): Boolean array of short exit signals. 
            
                Broadcasts.
            adjust_func_nb (Union[None, PathLike, AdjustFuncT]): User-defined function to adjust 
                the simulation state.
                
                Passed to the corresponding signal function. Can be provided as a module path when staticizing.
            adjust_args (Args): Arguments to pass to `adjust_func_nb`.
            signal_func_nb (Union[None, PathLike, SignalFuncT]): Function to generate signals.

                See `vectorbtpro.portfolio.nb.from_signals.from_signal_func_nb`.
                Can be given as a module path when staticizing.
            signal_args (Args): Arguments to pass to `signal_func_nb`.
            post_segment_func_nb (Union[None, PathLike, PostSegmentFuncT]): Post-segment function.

                See `vectorbtpro.portfolio.nb.from_signals.from_signal_func_nb`.
                Can be provided as a module path when staticizing.
            post_segment_args (Args): Arguments for `post_segment_func_nb`.
            order_mode (bool): If True, simulates in order mode without explicit signals.
            size (Optional[ArrayLike]): Trade size.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.size`. 
            
                !!! note
                    Negative size is not allowed; use signals to express direction.
            size_type (Optional[ArrayLike]): Type of size.

                Broadcasts. See `vectorbtpro.portfolio.enums.SizeType` and 
                `vectorbtpro.portfolio.enums.Order.size_type`. 

                Valid options include `SizeType.Amount`, `SizeType.Value`, `SizeType.Percent(100)`, and
                `SizeType.ValuePercent(100)`. Other types are incompatible with signals.
            
                !!! note
                    `SizeType.Percent(100)` does not support position reversal unless the position 
                    is closed first. See warning in `Portfolio.from_orders`.
            price (Optional[ArrayLike]): Order price.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.price` and 
                `vectorbtpro.portfolio.enums.PriceType`.
            
                Options such as `PriceType.NextOpen` and `PriceType.NextClose` apply per column and 
                require `from_ago` to be None.
            fees (Optional[ArrayLike]): Fees as a percentage of the order value.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.fees`. 
            fixed_fees (Optional[ArrayLike]): Fixed fee amount per order.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.fixed_fees`. 
            slippage (Optional[ArrayLike]): Slippage percentage of the order price.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.slippage`. 
            min_size (Optional[ArrayLike]): Minimum order size.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.min_size`.
            max_size (Optional[ArrayLike]): Maximum trade size.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.max_size`.

                If exceeded, orders may be partially filled. With accumulation enabled, 
                a very low `max_size` might hinder proper position closure.
            size_granularity (Optional[ArrayLike]): Granularity of the order size.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.size_granularity`.
            leverage (Optional[ArrayLike]): Leverage applied in the order.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.leverage`.
            leverage_mode (Optional[ArrayLike]): Leverage mode.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.leverage_mode`.
            reject_prob (Optional[ArrayLike]): Order rejection probability.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.reject_prob`.
            price_area_vio_mode (Optional[ArrayLike]): Price area violation mode.

                Broadcasts. See `vectorbtpro.portfolio.enums.PriceAreaVioMode`.
            allow_partial (Optional[ArrayLike]): Indicates whether partial fills are allowed.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.allow_partial`.
            raise_reject (Optional[ArrayLike]): Indicates whether to raise an exception upon order rejection.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.raise_reject`.
            log (Optional[ArrayLike]): Flag indicating whether to log orders.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.log`.
            val_price (Optional[ArrayLike]): Asset valuation price used in decision making.
            
                Broadcasts. Can also be provided as `vectorbtpro.portfolio.enums.ValPriceType`.
            
                * Any `-np.inf` element is replaced by the latest valuation price 
                    (using `open` or a previously known value if `ffill_val_price` is True).
                * Any `np.inf` element is replaced by the current order price.
            
                !!! note
                    Unlike `Portfolio.from_order_func`, the order price is effectively predetermined,
                    so `val_price` defaults to the current order price when using `np.inf`. 
                    To use the previous close, set it in the settings to `-np.inf`.
            
                !!! note
                    Ensure that the timestamp associated with `val_price` precedes all order 
                    timestamps in a cash-sharing group.
            accumulate (Optional[ArrayLike]): Accumulation mode.

                Broadcasts. If True, becomes 'both'; if False, becomes 'disabled'.
                When enabled, simulation behaves similarly to `Portfolio.from_orders`.
            upon_long_conflict (Optional[ArrayLike]): Conflict resolution mode for long signals.

                Broadcasts. See `vectorbtpro.portfolio.enums.ConflictMode`.
            upon_short_conflict (Optional[ArrayLike]): Conflict resolution mode for short signals.

                Broadcasts. See `vectorbtpro.portfolio.enums.ConflictMode`.
            upon_dir_conflict (Optional[ArrayLike]): Direction conflict mode.

                Broadcasts. See `vectorbtpro.portfolio.enums.DirectionConflictMode`.
            upon_opposite_entry (Optional[ArrayLike]): Mode for handling opposite entry signals.

                Broadcasts. See `vectorbtpro.portfolio.enums.OppositeEntryMode`.
            order_type (Optional[ArrayLike]): Order type.

                Broadcasts. See `vectorbtpro.portfolio.enums.OrderType`.

                Only one active limit order is allowed at a time.
            limit_delta (Optional[ArrayLike]): Delta to compute the limit price from `price`.

                Broadcasts. If NaN, `price` is used directly. The delta adjusts based on trade direction:
                for buying, a positive delta decreases the limit price; for selling, it increases 
                the limit price. Delta may be negative. Set an element to `np.nan` to disable.
                Use `delta_format` to specify the format.
            limit_tif (Optional[ArrayLike]): Time in force for limit orders.

                Broadcasts. Accepts frequency-like objects converted via `vectorbtpro.utils.datetime_.to_timedelta64`. 
                Set an element to -1 to disable. Use `time_delta_format` to specify the format.

                Measured as the duration from the signal bar's open time. If the expiration time happens
                in the middle of the current bar, we pessimistically assume that the order has been expired.
                The check is performed at the beginning of the bar, and the first check is performed at the
                next bar after the signal. For example, if the format is `TimeDeltaFormat.Rows`, 0 or 1 means
                the order must execute at the same bar or not at all; 2 means the order must execute at the
                same or next bar or not at all.
            limit_expiry (Optional[ArrayLike]): Expiration time for limit orders.

                Broadcasts. Accepts frequency-like objects or datetimes, similar to `limit_tif`.
                Set an element to -1 or `pd.Timestamp.max` to disable. 
                Use `time_delta_format` to specify the format.

                Any frequency-like object is used to build a period index, such that each timestamp in the original
                index is pointing to the timestamp where the period ends. For example, providing "d" will
                make any limit order expire on the next day. Any array must either contain timestamps or integers
                (not timedeltas!), and will be cast into integer format after broadcasting. If the object
                provided is of data type `object`, will be converted to datetime and its timezone will
                be removed automatically (as done on the index).
            limit_reverse (Optional[ArrayLike]): Flag to reverse price hit detection.

                Broadcasts. If True, a buy/sell limit is compared against high/low (instead of low/high) 
                and the limit delta is inverted.
            limit_order_price (Optional[ArrayLike]): Price for limit orders.

                Broadcasts. See `vectorbtpro.portfolio.enums.LimitOrderPrice`. Positive values are 
                used directly, while negative values are interpreted as enumerated options.
                If provided on the per-element basis, gets applied upon order creation. 
            upon_adj_limit_conflict (Optional[ArrayLike]): Conflict mode for adjacent limit and signal conflicts.

                Broadcasts. See `vectorbtpro.portfolio.enums.PendingConflictMode`.
            upon_opp_limit_conflict (Optional[ArrayLike]): Conflict mode for opposite limit and signal conflicts.

                Broadcasts. See `vectorbtpro.portfolio.enums.PendingConflictMode`.
            use_stops (bool): Flag indicating whether to enable stop orders.

                Broadcasts. If None, becomes True if any stop parameters are set or a non-default adjustment 
                function is used. Disable to speed up simple simulations.
            stop_ladder (Optional[bool]): Indicates whether to use stop laddering.

                Broadcasts to match the number of columns. See `vectorbtpro.portfolio.enums.StopLadderMode`. 
                If enabled, rows in the provided arrays become ladder steps. For price-based stops, 
                pad with `np.nan`; for time-based stops, pad with -1. Applied to all stop types.
            sl_stop (Optional[ArrayLike]): Stop loss levels.

                Broadcasts. Set an element to `np.nan` to disable. Use `delta_format` for formatting.
            tsl_stop (Optional[ArrayLike]): Trailing stop loss levels.

                Broadcasts. Set an element to `np.nan` to disable. Use `delta_format` for formatting.
            tsl_th (Optional[ArrayLike]): Take profit threshold for trailing stop loss.

                Broadcasts. Set an element to `np.nan` to disable. Use `delta_format` for formatting.
            tp_stop (Optional[ArrayLike]): Take profit levels.

                Broadcasts. Set an element to `np.nan` to disable. Use `delta_format` for formatting.
            td_stop (Optional[ArrayLike]): Timedelta stop values.

                Broadcasts. Set an element to -1 to disable. Use `time_delta_format` for formatting.
            dt_stop (Optional[ArrayLike]): Datetime stop values.

                Broadcasts. Set an element to -1 to disable. Use `time_delta_format` for formatting.
            stop_entry_price (Optional[ArrayLike]): Entry price for stop orders.

                Broadcasts. See `vectorbtpro.portfolio.enums.StopEntryPrice`.
                If provided on the per-element basis, gets applied upon entry. 
            stop_exit_price (Optional[ArrayLike]): Exit price for stop orders.

                Broadcasts. See `vectorbtpro.portfolio.enums.StopExitPrice`.
                If provided on the per-element basis, gets applied upon entry. 
            stop_exit_type (Optional[ArrayLike]): Stop exit type.

                Broadcasts. See `vectorbtpro.portfolio.enums.StopExitType`.
                If provided on the per-element basis, gets applied upon entry. 
            stop_order_type (Optional[ArrayLike]): Order type for stop orders.

                Broadcasts. Similar to `order_type`, but applies to stop orders.
                If provided on the per-element basis, gets applied upon entry. 
            stop_limit_delta (Optional[ArrayLike]): Delta for stop orders, analogous to `limit_delta`.

                Broadcasts. 
            upon_stop_update (Optional[ArrayLike]): Stop update mode.

                Broadcasts. See `vectorbtpro.portfolio.enums.StopUpdateMode`. Effective only if 
                accumulation is enabled. If provided on the per-element basis, gets applied upon repeated entry.
            upon_adj_stop_conflict (Optional[ArrayLike]): Conflict mode for adjacent stop and signal conflicts.

                Broadcasts. See `vectorbtpro.portfolio.enums.PendingConflictMode`.
            upon_opp_stop_conflict (Optional[ArrayLike]): Conflict mode for opposite stop and signal conflicts.

                Broadcasts. See `vectorbtpro.portfolio.enums.PendingConflictMode`.
            delta_format (Optional[ArrayLike]): Format specification for delta values.

                Broadcasts. See `vectorbtpro.portfolio.enums.DeltaFormat`.
            time_delta_format (Optional[ArrayLike]): Format specification for time delta values.

                Broadcasts. See `vectorbtpro.portfolio.enums.TimeDeltaFormat`.
            open (Optional[ArrayLike]): Open prices.

                Broadcasts. Used as a price boundary (see `vectorbtpro.portfolio.enums.PriceArea`).

                For stop signals, `np.nan` is replaced with the corresponding close price.
            high (Optional[ArrayLike]): High prices.

                Broadcasts. Used as a price boundary (see `vectorbtpro.portfolio.enums.PriceArea`).

                For stop signals, `np.nan` is replaced with the maximum of open and close prices.
            low (Optional[ArrayLike]): Low prices.

                Broadcasts. Used as a price boundary (see `vectorbtpro.portfolio.enums.PriceArea`).

                For stop signals, `np.nan` is replaced with the minimum of open and close prices.
            init_cash (Optional[ArrayLike]): Initial capital.
            
                Broadcasts to the final number of columns, or to the number of groups if 
                cash sharing is enabled. See `vectorbtpro.portfolio.enums.InitCashMode` for guidance.
            
                !!! note
                    When using `InitCashMode.AutoAlign`, initial cash values are synchronized across 
                    columns/groups after initialization.
            init_position (Optional[ArrayLike]): Initial position. 
            
                Broadcasts to match the final number of columns.
            init_price (Optional[ArrayLike]): Initial position price. 
            
                Broadcasts to match the final number of columns.
            cash_deposits (Optional[ArrayLike]): Cash deposits or withdrawals at the beginning of each timestamp.
            
                Broadcasts to match the shape of `init_cash`.
            cash_earnings (Optional[ArrayLike]): Cash earnings added at the end of each timestamp.
            
                Broadcasts.
            cash_dividends (Optional[ArrayLike]): Cash dividends added at the end of each timestamp.
            
                Broadcasts, are multiplied by the position, and then added to `cash_earnings`.
            cash_sharing (Optional[bool]): Indicates whether cash is shared among assets within the same group.
            
                If `group_by` is None and this is True, all assets are grouped together for cash sharing.
            
                !!! warning
                    Enables cross-asset dependencies by assuming that all orders in a cash-sharing group 
                    execute in the same tick and retain their price.
            from_ago (Optional[ArrayLike]): Number of time steps to look back for order information.
            
                Negative values are converted to positive to avoid look-ahead bias.
            sim_start (Optional[ArrayLike]): Simulation start (inclusive).

                Can be "auto", which will be substituted by the index of the first signal across
                long and short entries and long and short exits.
            sim_end (Optional[ArrayLike]): Simulation end (exclusive).

                Can be "auto", which will be substituted by the index of the last signal across
                long and short entries and long and short exits.
            call_seq (Optional[ArrayLike]): Sequence dictating the order in which columns are 
                processed per row and group.
            
                Each element specifies the position of a column in the processing order.
                Options include:
            
                * None: Generates a default call sequence.
                * A value from `vectorbtpro.portfolio.enums.CallSeqType`: Creates a full array of the specified type.
                * Custom array: Specifies a user-defined call sequence.
            
                If set to `CallSeqType.Auto`, orders are rearranged dynamically so that sell orders are 
                processed before buy orders.
            
                !!! warning
                    `CallSeqType.Auto` assumes predetermined order prices and flexible execution, 
                    which may not accurately reflect real-time conditions.
                    For stricter control, use `Portfolio.from_order_func`.
            attach_call_seq (Optional[bool]): Indicates whether to attach the computed call sequence 
                to the portfolio instance.
            ffill_val_price (Optional[bool]): If True, tracks the valuation price only when available 
                to prevent propagation of NaN values.
            update_value (Optional[bool]): If True, updates the group value after each filled order.
            fill_pos_info (Optional[bool]): Whether to fill position records.

                Disabling this may speed up simulation for simple cases.
            save_state (Optional[bool]): If True, saves state arrays 
                (such as `cash`, `position`, `debt`, `locked_cash`, and `free_cash`) in in-outputs.
            save_value (Optional[bool]): If True, saves the portfolio value in in-outputs.
            save_returns (Optional[bool]): If True, saves the returns array in in-outputs.
            skip_empty (Optional[bool]): If True, skips processing rows that do not contain any orders.
            max_order_records (Optional[int]): Maximum number of order records expected per column.
            
                Set to a lower number to conserve memory, or 0 to disable order record collection.
            max_log_records (Optional[int]): Maximum number of log records expected per column.
            
                Set to a lower number to conserve memory, or 0 to disable logging.
            in_outputs (Optional[tp.MappingLike]): Mapping of in-output objects for flexible mode.

                These objects become available via `Portfolio.in_outputs` as a named tuple.
            
                * To substitute `Portfolio` attributes, provide broadcasted and grouped objects 
                    (e.g., via `broadcast_named_args` and templates).
                * When chunking, supply the chunk specification and merging function.
                
                See `vectorbtpro.portfolio.chunking.merge_sim_outs`.
            
                !!! note
                    When using Numba below 0.54, `in_outputs` must be a globally defined 
                    named tuple rather than a mapping.
            seed (Optional[int]): Seed value for generating the call sequence and initializing the simulation.
            group_by (tp.GroupByLike): Grouping specification for columns.
            
                See `vectorbtpro.base.grouping.base.Grouper` for details.
            broadcast_named_args (tp.KwargsLike): Dictionary of named arguments for broadcasting.

                Allows substitution of argument names wrapped with `vectorbtpro.utils.template.Rep`.
            broadcast_kwargs (tp.KwargsLike): Additional keyword arguments to pass to 
                `vectorbtpro.base.reshaping.broadcast`.
            template_context (tp.KwargsLike): Context for template substitution in arguments.
            jitted (tp.JittedOption): Option to control JIT compilation.
            
                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (tp.ChunkedOption): Option to control chunked processing.
            
                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            staticized (tp.StaticizedOption): Keyword arguments or a task id for staticizing.
            
                If True or a dict, these are passed to `vectorbtpro.utils.cutting.cut_and_save_func` 
                to cache the simulator. If a hashable or callable, it is used as a task id for an already 
                registered simulator. The dict may include options like `override` and `reload`.
            bm_close (Optional[ArrayLike]): Benchmark asset price at each time step.
            
                Broadcasts. If not provided, `close` is used; if set to False, benchmarking is disabled.
            records (Optional[tp.RecordsLike]): Records used to construct arrays.
            
                See `vectorbtpro.base.indexing.IdxRecords`.
            return_preparer (bool): If True, returns the preparer instance 
                (`vectorbtpro.portfolio.preparing.FSPreparer`).
            
                !!! note
                    In this case, the seed is not automatically set; 
                    invoke `preparer.set_seed()` explicitly if needed.
            return_prep_result (bool): If True, returns the preparer result 
                (`vectorbtpro.portfolio.preparing.PFPrepResult`).
            return_sim_out (bool): If True, returns the simulation output 
                (`vectorbtpro.portfolio.enums.SimulationOutput`).
            **kwargs: Additional keyword arguments passed to the `Portfolio` constructor.

        Returns:
            PortfolioResult: The portfolio result.

        Examples:
            By default, if all signal arrays are None, `entries` is treated as True,
            opening a position at the first tick:

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_signals(close, size=1)
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3    0.0
            4    0.0
            dtype: float64
            ```

            Entry opens long, exit closes long:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1,
            ...     direction='longonly'
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3   -1.0
            4    0.0
            dtype: float64

            >>> # Using direction-aware arrays instead of `direction`
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),  # long_entries
            ...     exits=pd.Series([False, False, True, True, True]),  # long_exits
            ...     short_entries=False,
            ...     short_exits=False,
            ...     size=1
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3   -1.0
            4    0.0
            dtype: float64
            ```

            Notice how both `short_entries` and `short_exits` are provided as constants - as any other
            broadcastable argument, they are treated as arrays where each element is False.

            Entry opens short, exit closes short:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1,
            ...     direction='shortonly'
            ... )
            >>> pf.asset_flow
            0   -1.0
            1    0.0
            2    0.0
            3    1.0
            4    0.0
            dtype: float64

            >>> # Using direction-aware arrays instead of `direction`
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=False,  # long_entries
            ...     exits=False,  # long_exits
            ...     short_entries=pd.Series([True, True, True, False, False]),
            ...     short_exits=pd.Series([False, False, True, True, True]),
            ...     size=1
            ... )
            >>> pf.asset_flow
            0   -1.0
            1    0.0
            2    0.0
            3    1.0
            4    0.0
            dtype: float64
            ```

            Entry opens long and closes short, exit closes long and opens short:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1,
            ...     direction='both'
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3   -2.0
            4    0.0
            dtype: float64

            >>> # Using direction-aware arrays instead of `direction`
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),  # long_entries
            ...     exits=False,  # long_exits
            ...     short_entries=pd.Series([False, False, True, True, True]),
            ...     short_exits=False,
            ...     size=1
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3   -2.0
            4    0.0
            dtype: float64
            ```

            More complex signal combinations using direction-aware arrays.
            For example, ignore opposite signals as long as the current position is open:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries      =pd.Series([True, False, False, False, False]),  # long_entries
            ...     exits        =pd.Series([False, False, True, False, False]),  # long_exits
            ...     short_entries=pd.Series([False, True, False, True, False]),
            ...     short_exits  =pd.Series([False, False, False, False, True]),
            ...     size=1,
            ...     upon_opposite_entry='ignore'
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2   -1.0
            3   -1.0
            4    1.0
            dtype: float64
            ```

            First opposite signal closes the position, second one opens a new position:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1,
            ...     direction='both',
            ...     upon_opposite_entry='close'
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2    0.0
            3   -1.0
            4   -1.0
            dtype: float64
            ```

            If both long entry and exit signals are True (a signal conflict), choose exit:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1.,
            ...     direction='longonly',
            ...     upon_long_conflict='exit')
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2   -1.0
            3    0.0
            4    0.0
            dtype: float64
            ```

            If both long entry and short entry signal are True (a direction conflict), choose short:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1.,
            ...     direction='both',
            ...     upon_dir_conflict='short')
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2   -2.0
            3    0.0
            4    0.0
            dtype: float64
            ```

            !!! note
                Remember that when direction is set to 'both', entries become `long_entries` and exits become
                `short_entries`, so this becomes a conflict of directions rather than signals.

            If both signal and direction conflicts occur:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=True,  # long_entries
            ...     exits=True,  # long_exits
            ...     short_entries=True,
            ...     short_exits=True,
            ...     size=1,
            ...     upon_long_conflict='entry',
            ...     upon_short_conflict='entry',
            ...     upon_dir_conflict='short'
            ... )
            >>> pf.asset_flow
            0   -1.0
            1    0.0
            2    0.0
            3    0.0
            4    0.0
            dtype: float64
            ```

            Turn on accumulation (entry implies long order, exit implies short order):

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1.,
            ...     direction='both',
            ...     accumulate=True)
            >>> pf.asset_flow
            0    1.0
            1    1.0
            2    0.0
            3   -1.0
            4   -1.0
            dtype: float64
            ```

            Allow increasing position size without decreasing the existing position:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     size=1.,
            ...     direction='both',
            ...     accumulate='addonly')
            >>> pf.asset_flow
            0    1.0  << open a long position
            1    1.0  << add to the position
            2    0.0
            3   -3.0  << close and open a short position
            4   -1.0  << add to the position
            dtype: float64
            ```

            Test multiple parameters via regular broadcasting:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     direction=[list(Direction)],
            ...     broadcast_kwargs=dict(columns_from=pd.Index(vbt.pf_enums.Direction._fields, name='direction')))
            >>> pf.asset_flow
            direction  LongOnly  ShortOnly   Both
            0             100.0     -100.0  100.0
            1               0.0        0.0    0.0
            2               0.0        0.0    0.0
            3            -100.0       50.0 -200.0
            4               0.0        0.0    0.0
            ```

            Test multiple parameters via `vectorbtpro.base.reshaping.BCO`:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close,
            ...     entries=pd.Series([True, True, True, False, False]),
            ...     exits=pd.Series([False, False, True, True, True]),
            ...     direction=vbt.Param(Direction))
            >>> pf.asset_flow
            direction  LongOnly  ShortOnly   Both
            0             100.0     -100.0  100.0
            1               0.0        0.0    0.0
            2               0.0        0.0    0.0
            3            -100.0       50.0 -200.0
            4               0.0        0.0    0.0
            ```

            Set risk/reward ratio using trailing stop loss and take profit thresholds:

            ```pycon
            >>> close = pd.Series([10, 11, 12, 11, 10, 9])
            >>> entries = pd.Series([True, False, False, False, False, False])
            >>> exits = pd.Series([False, False, False, False, False, True])
            >>> pf = vbt.Portfolio.from_signals(
            ...     close, entries, exits,
            ...     tsl_stop=0.1, tp_stop=0.2)  # take profit hit
            >>> pf.asset_flow
            0    10.0
            1     0.0
            2   -10.0
            3     0.0
            4     0.0
            5     0.0
            dtype: float64

            >>> pf = vbt.Portfolio.from_signals(
            ...     close, entries, exits,
            ...     tsl_stop=0.1, tp_stop=0.3)  # trailing stop loss hit
            >>> pf.asset_flow
            0    10.0
            1     0.0
            2     0.0
            3     0.0
            4   -10.0
            5     0.0
            dtype: float64

            >>> pf = vbt.Portfolio.from_signals(
            ...     close, entries, exits,
            ...     tsl_stop=np.inf, tp_stop=np.inf)  # nothing hit, exit as usual
            >>> pf.asset_flow
            0    10.0
            1     0.0
            2     0.0
            3     0.0
            4     0.0
            5   -10.0
            dtype: float64
            ```

            Test different stop combinations:

            ```pycon
            >>> pf = vbt.Portfolio.from_signals(
            ...     close, entries, exits,
            ...     tsl_stop=vbt.Param([0.1, 0.2]),
            ...     tp_stop=vbt.Param([0.2, 0.3])
            ... )
            >>> pf.asset_flow
            tsl_stop   0.1         0.2
            tp_stop    0.2   0.3   0.2   0.3
            0         10.0  10.0  10.0  10.0
            1          0.0   0.0   0.0   0.0
            2        -10.0   0.0 -10.0   0.0
            3          0.0   0.0   0.0   0.0
            4          0.0 -10.0   0.0   0.0
            5          0.0   0.0   0.0 -10.0
            ```

            This works because `pd.Index` automatically translates into `vectorbtpro.base.reshaping.BCO`
            with `product` set to True.

            We can implement our own stop loss or take profit, or adjust the existing one at each time step.
            Let's implement [stepped stop-loss](https://www.freqtrade.io/en/stable/strategy-advanced/#stepped-stoploss):

            ```pycon
            >>> @njit
            ... def adjust_func_nb(c):
            ...     val_price_now = c.last_val_price[c.col]
            ...     tsl_init_price = c.last_tsl_info["init_price"][c.col]
            ...     current_profit = (val_price_now - tsl_init_price) / tsl_init_price
            ...     if current_profit >= 0.40:
            ...         c.last_tsl_info["stop"][c.col] = 0.25
            ...     elif current_profit >= 0.25:
            ...         c.last_tsl_info["stop"][c.col] = 0.15
            ...     elif current_profit >= 0.20:
            ...         c.last_tsl_info["stop"][c.col] = 0.07

            >>> close = pd.Series([10, 11, 12, 11, 10])
            >>> pf = vbt.Portfolio.from_signals(close, adjust_func_nb=adjust_func_nb)
            >>> pf.asset_flow
            0    10.0
            1     0.0
            2     0.0
            3   -10.0  # 7% from 12 hit
            4    11.16
            dtype: float64
            ```

            Sometimes there is a need to provide or transform signals dynamically. For this, we can implement
            a custom signal function `signal_func_nb`. For example, let's implement a signal function that
            takes two numerical arrays - long and short one - and transforms them into 4 direction-aware boolean
            arrays that vectorbtpro understands:

            ```pycon
            >>> @njit
            ... def signal_func_nb(c, long_num_arr, short_num_arr):
            ...     long_num = vbt.pf_nb.select_nb(c, long_num_arr)
            ...     short_num = vbt.pf_nb.select_nb(c, short_num_arr)
            ...     is_long_entry = long_num > 0
            ...     is_long_exit = long_num < 0
            ...     is_short_entry = short_num > 0
            ...     is_short_exit = short_num < 0
            ...     return is_long_entry, is_long_exit, is_short_entry, is_short_exit

            >>> pf = vbt.Portfolio.from_signals(
            ...     pd.Series([1, 2, 3, 4, 5]),
            ...     signal_func_nb=signal_func_nb,
            ...     signal_args=(vbt.Rep('long_num_arr'), vbt.Rep('short_num_arr')),
            ...     broadcast_named_args=dict(
            ...         long_num_arr=pd.Series([1, 0, -1, 0, 0]),
            ...         short_num_arr=pd.Series([0, 1, 0, 1, -1])
            ...     ),
            ...     size=1,
            ...     upon_opposite_entry='ignore'
            ... )
            >>> pf.asset_flow
            0    1.0
            1    0.0
            2   -1.0
            3   -1.0
            4    1.0
            dtype: float64
            ```

            Passing both arrays as `broadcast_named_args` broadcasts them internally as any other array,
            so we don't have to worry about their dimensions every time we change our data.
        """
        if isinstance(close, FSPreparer):
            preparer = close
            prep_result = None
        elif isinstance(close, PFPrepResult):
            preparer = None
            prep_result = close
        else:
            local_kwargs = locals()
            local_kwargs = {**local_kwargs, **local_kwargs["kwargs"]}
            del local_kwargs["kwargs"]
            del local_kwargs["cls"]
            del local_kwargs["return_preparer"]
            del local_kwargs["return_prep_result"]
            del local_kwargs["return_sim_out"]
            parsed_data = BasePFPreparer.parse_data(close, all_ohlc=True)
            if parsed_data is not None:
                local_kwargs["data"] = parsed_data
                local_kwargs["close"] = None
            preparer = FSPreparer(**local_kwargs)
            if not return_preparer:
                preparer.set_seed()
            prep_result = None
        if return_preparer:
            return preparer
        if prep_result is None:
            prep_result = preparer.result
        if return_prep_result:
            return prep_result
        sim_out = prep_result.target_func(**prep_result.target_args)
        if return_sim_out:
            return sim_out
        return cls(order_records=sim_out, **prep_result.pf_args)

    @classmethod
    def from_holding(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, OHLCDataMixin],
        direction: tp.Optional[int] = None,
        at_first_valid_in: tp.Optional[str] = "close",
        close_at_end: tp.Optional[bool] = None,
        dynamic_mode: bool = False,
        **kwargs,
    ) -> PortfolioResultT:
        """Simulate portfolio from plain holding using signals.

        If `close_at_end` is True, an opposite signal is placed at the very end.

        Args:
            close (Union[ArrayLike, OHLCDataMixin]): The close prices or OHLC data used for portfolio simulation.
            direction (Optional[int]): Holding direction. 
            
                If None, the default hold direction from portfolio settings is used.
            at_first_valid_in (Optional[str]): Column name for determining the first valid entry signal.
            close_at_end (Optional[bool]): Flag indicating whether to place an exit signal at the end. 
            
                If None, the default is used.
            dynamic_mode (bool): Specifies whether to use dynamic mode for signal generation.
            **kwargs: Additional keyword arguments forwarded to `Portfolio.from_signals`.

        Returns:
            PortfolioResult: The portfolio result.
        """
        from vectorbtpro._settings import settings

        portfolio_cfg = settings["portfolio"]

        if direction is None:
            direction = portfolio_cfg["hold_direction"]
        direction = map_enum_fields(direction, enums.Direction)
        if not checks.is_int(direction):
            raise TypeError("Direction must be a scalar")
        if close_at_end is None:
            close_at_end = portfolio_cfg["close_at_end"]

        if dynamic_mode:

            def _substitute_signal_args(preparer):
                return (
                    direction,
                    close_at_end,
                    *((preparer.adjust_func_nb,) if preparer.staticized is None else ()),
                    preparer.adjust_args,
                )

            return cls.from_signals(
                close,
                signal_func_nb=nb.holding_enex_signal_func_nb,
                signal_args=RepFunc(_substitute_signal_args),
                accumulate=False,
                **kwargs,
            )

        def _entries(wrapper, new_objs):
            if at_first_valid_in is None:
                entries = np.full((wrapper.shape_2d[0], 1), False)
                entries[0] = True
                return entries
            ts = new_objs[at_first_valid_in]
            valid_index = generic_nb.first_valid_index_nb(ts)
            if (valid_index == -1).all():
                return np.array([[False]])
            if (valid_index == 0).all():
                entries = np.full((wrapper.shape_2d[0], 1), False)
                entries[0] = True
                return entries
            entries = np.full(wrapper.shape_2d, False)
            entries[valid_index, np.arange(wrapper.shape_2d[1])] = True
            return entries

        def _exits(wrapper):
            if close_at_end:
                exits = np.full((wrapper.shape_2d[0], 1), False)
                exits[-1] = True
            else:
                exits = np.array([[False]])
            return exits

        return cls.from_signals(
            close,
            entries=RepFunc(_entries),
            exits=RepFunc(_exits),
            direction=direction,
            accumulate=False,
            **kwargs,
        )

    @classmethod
    def from_random_signals(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, OHLCDataMixin],
        n: tp.Optional[tp.ArrayLike] = None,
        prob: tp.Optional[tp.ArrayLike] = None,
        entry_prob: tp.Optional[tp.ArrayLike] = None,
        exit_prob: tp.Optional[tp.ArrayLike] = None,
        param_product: bool = False,
        seed: tp.Optional[int] = None,
        run_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> PortfolioResultT:
        """Simulate portfolio from random entry and exit signals.

        Generates random entry and exit signals based on either the number of signals or 
        the probability of encountering a signal.

        * If `n` is provided, see `vectorbtpro.signals.generators.randnx.RANDNX`.
        * If `prob` is provided, see `vectorbtpro.signals.generators.rprobnx.RPROBNX`.

        Based on `Portfolio.from_signals`.

        !!! note
            To generate random signals, the shape of `close` is used. Broadcasting with other arrays 
            occurs after signal generation.

        Args:
            close (Union[ArrayLike, OHLCDataMixin]): The close prices or OHLC data used for portfolio simulation.
            n (Optional[ArrayLike]): Number of signals to generate. 
            
                Mutually exclusive with `entry_prob` and `exit_prob`.
            prob (Optional[ArrayLike]): Probability of encountering a signal, used for both entries and exits 
                if specific probabilities are not provided.
            entry_prob (Optional[ArrayLike]): Probability for generating an entry signal. 
            
                Defaults to `prob` if not specified.
            exit_prob (Optional[ArrayLike]): Probability for generating an exit signal. 
            
                Defaults to `prob` if not specified.
            param_product (bool): Flag to generate the Cartesian product of signal parameters.
            seed (Optional[int]): Random seed for signal generation. 
            
                If None, the seed from portfolio settings is used.
            run_kwargs (KwargsLike): Additional keyword arguments passed to the signal generator's run function.
            **kwargs: Additional keyword arguments forwarded to `Portfolio.from_signals`.

        Returns:
            PortfolioResult: The portfolio result.

        Examples:
            Test multiple combinations of random entries and exits:

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_random_signals(close, n=[2, 1, 0], seed=42)
            >>> pf.orders.count()
            randnx_n
            2    4
            1    2
            0    0
            Name: count, dtype: int64
            ```

            Test the Cartesian product of entry and exit encounter probabilities:

            ```pycon
            >>> pf = vbt.Portfolio.from_random_signals(
            ...     close,
            ...     entry_prob=[0, 0.5, 1],
            ...     exit_prob=[0, 0.5, 1],
            ...     param_product=True,
            ...     seed=42)
            >>> pf.orders.count()
            rprobnx_entry_prob  rprobnx_exit_prob
            0.0                 0.0                  0
                                0.5                  0
                                1.0                  0
            0.5                 0.0                  1
                                0.5                  4
                                1.0                  3
            1.0                 0.0                  1
                                0.5                  4
                                1.0                  5
            Name: count, dtype: int64
            ```
        """
        from vectorbtpro._settings import settings

        portfolio_cfg = settings["portfolio"]

        parsed_data = BasePFPreparer.parse_data(close, all_ohlc=True)
        if parsed_data is not None:
            data = parsed_data
            close = data.close
            if close is None:
                raise ValueError("Column for close couldn't be found in data")
            close_wrapper = data.symbol_wrapper
        else:
            close = to_pd_array(close)
            close_wrapper = ArrayWrapper.from_obj(close)
            data = close
        if entry_prob is None:
            entry_prob = prob
        if exit_prob is None:
            exit_prob = prob
        if seed is None:
            seed = portfolio_cfg["seed"]
        if run_kwargs is None:
            run_kwargs = {}

        if n is not None and (entry_prob is not None or exit_prob is not None):
            raise ValueError("Must provide either n or entry_prob and exit_prob")
        if n is not None:
            from vectorbtpro.signals.generators.randnx import RANDNX

            rand = RANDNX.run(
                n=n,
                input_shape=close_wrapper.shape,
                input_index=close_wrapper.index,
                input_columns=close_wrapper.columns,
                seed=seed,
                **run_kwargs,
            )
            entries = rand.entries
            exits = rand.exits
        elif entry_prob is not None and exit_prob is not None:
            from vectorbtpro.signals.generators.rprobnx import RPROBNX

            rprobnx = RPROBNX.run(
                entry_prob=entry_prob,
                exit_prob=exit_prob,
                param_product=param_product,
                input_shape=close_wrapper.shape,
                input_index=close_wrapper.index,
                input_columns=close_wrapper.columns,
                seed=seed,
                **run_kwargs,
            )
            entries = rprobnx.entries
            exits = rprobnx.exits
        else:
            raise ValueError("Must provide at least n or entry_prob and exit_prob")

        return cls.from_signals(data, entries, exits, seed=seed, **kwargs)

    @classmethod
    def from_optimizer(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, OHLCDataMixin],
        optimizer: PortfolioOptimizer,
        pf_method: str = "from_orders",
        squeeze_groups: bool = True,
        dropna: tp.Optional[str] = None,
        fill_value: tp.Scalar = np.nan,
        size_type: tp.ArrayLike = "targetpercent",
        direction: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = True,
        call_seq: tp.Optional[tp.ArrayLike] = "auto",
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> PortfolioResultT:
        """Build a portfolio from a `vectorbtpro.portfolio.pfopt.base.PortfolioOptimizer` instance.

        If the `direction` parameter is not provided, it is inferred from the allocation values:

        * If both positive and negative values exist, set to "both".
        * If only positive values exist, set to "longonly".
        * If only negative values exist, set to "shortonly" (allocations are converted to absolute values).

        The simulation method is selected based on `pf_method`:

        * "from_orders": Uses the orders-based simulation (`Portfolio.from_orders`).
        * "from_signals": Uses the signals-based simulation (`Portfolio.from_signals`).

        By default, `size_type` is "targetpercent", `cash_sharing` is True, `call_seq` is "auto", and
        if `group_by` is None, it defaults to the optimizer's grouper.

        Args:
            close (Union[ArrayLike, OHLCDataMixin]): The close prices or OHLC data used for portfolio simulation.
            optimizer (PortfolioOptimizer): Optimizer instance that provides allocation weights.
            pf_method (str): Portfolio simulation method, either "from_orders" or "from_signals".
            squeeze_groups (bool): Whether to squeeze groups in the optimizer's allocation output.
            dropna (Optional[str]): Strategy for handling missing allocation values during filling.
            fill_value (Scalar): Value used to fill missing allocation entries.
            size_type (ArrayLike): Order size type. 
            direction (Optional[ArrayLike]): Order direction; if None, determined automatically from allocations.
            cash_sharing (Optional[bool]): Flag indicating if cash is shared across assets.
            call_seq (Optional[ArrayLike]): Call sequence configuration, defaults to "auto".
            group_by (GroupByLike): Grouping specification for simulation. 
            
                If None, derived from the optimizer wrapper.
            **kwargs: Additional keyword arguments forwarded to the underlying simulation function.

        Returns:
            PortfolioResult: The portfolio result.

        Examples:
            ```pycon
            >>> close = pd.DataFrame({
            ...     "MSFT": [1, 2, 3, 4, 5],
            ...     "GOOG": [5, 4, 3, 2, 1],
            ...     "AAPL": [1, 2, 3, 2, 1]
            ... }, index=pd.date_range(start="2020-01-01", periods=5))

            >>> pfo = vbt.PortfolioOptimizer.from_random(
            ...     close.vbt.wrapper,
            ...     every="2D",
            ...     seed=42
            ... )
            >>> pfo.fill_allocations()
                             MSFT      GOOG      AAPL
            2020-01-01   0.182059  0.462129  0.355812
            2020-01-02        NaN       NaN       NaN
            2020-01-03   0.657381  0.171323  0.171296
            2020-01-04        NaN       NaN       NaN
            2020-01-05   0.038078  0.567845  0.394077

            >>> pf = vbt.Portfolio.from_optimizer(close, pfo)
            >>> pf.get_asset_value(group_by=False).vbt / pf.value
            alloc_group                         group
                             MSFT      GOOG      AAPL
            2020-01-01   0.182059  0.462129  0.355812  << rebalanced
            2020-01-02   0.251907  0.255771  0.492322
            2020-01-03   0.657381  0.171323  0.171296  << rebalanced
            2020-01-04   0.793277  0.103369  0.103353
            2020-01-05   0.038078  0.567845  0.394077  << rebalanced
            ```
        """
        size = optimizer.fill_allocations(squeeze_groups=squeeze_groups, dropna=dropna, fill_value=fill_value)
        if direction is None:
            pos_size_any = (size.values > 0).any()
            neg_size_any = (size.values < 0).any()
            if pos_size_any and neg_size_any:
                direction = "both"
            elif pos_size_any:
                direction = "longonly"
            else:
                direction = "shortonly"
                size = size.abs()

        if group_by is None:

            def _substitute_group_by(index):
                columns = optimizer.wrapper.columns
                if squeeze_groups and optimizer.wrapper.grouped_ndim == 1:
                    columns = columns.droplevel(level=0)
                if not index.equals(columns):
                    if "symbol" in index.names:
                        return ExceptLevel("symbol")
                    raise ValueError("Column hierarchy has changed. Disable squeeze_groups and provide group_by.")
                return optimizer.wrapper.grouper.group_by

            group_by = RepFunc(_substitute_group_by)

        if pf_method.lower() == "from_orders":
            return cls.from_orders(
                close,
                size=size,
                size_type=size_type,
                direction=direction,
                cash_sharing=cash_sharing,
                call_seq=call_seq,
                group_by=group_by,
                **kwargs,
            )
        elif pf_method.lower() == "from_signals":
            return cls.from_signals(
                close,
                order_mode=True,
                size=size,
                size_type=size_type,
                direction=direction,
                accumulate=True,
                cash_sharing=cash_sharing,
                call_seq=call_seq,
                group_by=group_by,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid pf_method: '{pf_method}'")

    @classmethod
    def from_order_func(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, OHLCDataMixin, FOFPreparer, PFPrepResult],
        *,
        init_cash: tp.Optional[tp.ArrayLike] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_earnings: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        call_seq: tp.Optional[tp.ArrayLike] = None,
        attach_call_seq: tp.Optional[bool] = None,
        segment_mask: tp.Optional[tp.ArrayLike] = None,
        call_pre_segment: tp.Optional[bool] = None,
        call_post_segment: tp.Optional[bool] = None,
        pre_sim_func_nb: tp.Optional[nb.PreSimFuncT] = None,
        pre_sim_args: tp.Args = (),
        post_sim_func_nb: tp.Optional[nb.PostSimFuncT] = None,
        post_sim_args: tp.Args = (),
        pre_group_func_nb: tp.Optional[nb.PreGroupFuncT] = None,
        pre_group_args: tp.Args = (),
        post_group_func_nb: tp.Optional[nb.PostGroupFuncT] = None,
        post_group_args: tp.Args = (),
        pre_row_func_nb: tp.Optional[nb.PreRowFuncT] = None,
        pre_row_args: tp.Args = (),
        post_row_func_nb: tp.Optional[nb.PostRowFuncT] = None,
        post_row_args: tp.Args = (),
        pre_segment_func_nb: tp.Optional[nb.PreSegmentFuncT] = None,
        pre_segment_args: tp.Args = (),
        post_segment_func_nb: tp.Optional[nb.PostSegmentFuncT] = None,
        post_segment_args: tp.Args = (),
        order_func_nb: tp.Optional[nb.OrderFuncT] = None,
        order_args: tp.Args = (),
        flex_order_func_nb: tp.Optional[nb.FlexOrderFuncT] = None,
        flex_order_args: tp.Args = (),
        post_order_func_nb: tp.Optional[nb.PostOrderFuncT] = None,
        post_order_args: tp.Args = (),
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        ffill_val_price: tp.Optional[bool] = None,
        update_value: tp.Optional[bool] = None,
        fill_pos_info: tp.Optional[bool] = None,
        track_value: tp.Optional[bool] = None,
        row_wise: tp.Optional[bool] = None,
        max_order_records: tp.Optional[int] = None,
        max_log_records: tp.Optional[int] = None,
        in_outputs: tp.Optional[tp.MappingLike] = None,
        seed: tp.Optional[int] = None,
        group_by: tp.GroupByLike = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        keep_inout_flex: tp.Optional[bool] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        staticized: tp.StaticizedOption = None,
        bm_close: tp.Optional[tp.ArrayLike] = None,
        records: tp.Optional[tp.RecordsLike] = None,
        return_preparer: bool = False,
        return_prep_result: bool = False,
        return_sim_out: bool = False,
        **kwargs,
    ) -> PortfolioResultT:
        """Build a portfolio using a custom order function.

        !!! hint
            See `vectorbtpro.portfolio.nb.from_order_func.from_order_func_nb` for 
            illustrations and argument definitions.

        For more details on individual simulation functions:

        * `order_func_nb`: See `vectorbtpro.portfolio.nb.from_order_func.from_order_func_nb`
        * `order_func_nb` and `row_wise`: See `vectorbtpro.portfolio.nb.from_order_func.from_order_func_rw_nb`
        * `flex_order_func_nb`: See `vectorbtpro.portfolio.nb.from_order_func.from_flex_order_func_nb`
        * `flex_order_func_nb` and `row_wise`: See `vectorbtpro.portfolio.nb.from_order_func.from_flex_order_func_rw_nb`

        Prepared by `vectorbtpro.portfolio.preparing.FOFPreparer`.

        For defaults, see `vectorbtpro._settings.portfolio`. These defaults are not used to fill NaN values 
        after reindexing; vectorbtpro uses its own defaults (typically NaN for floating arrays and preset 
        flags for integer arrays). Use `vectorbtpro.base.reshaping.BCO` with `fill_value` to override.

        !!! hint
            All broadcastable arguments are handled using `vectorbtpro.base.reshaping.broadcast` 
            to preserve their original shapes for flexible indexing and memory efficiency. 
            Each can be provided per frame, series, row, column, or individual element.

        !!! note
            All provided functions must be Numba-compiled if Numba is enabled. 
            Also see the notes in `Portfolio.from_orders`.

        !!! note
            Unlike other methods, the valuation price is taken from the previous `close` rather than 
            the order price, since an order's price is unknown until execution. You can override 
            the valuation price in `pre_segment_func_nb`.

        Args:
            close (Union[ArrayLike, OHLCDataMixin, FOFPreparer, PFPrepResult]): The close prices or 
                OHLC data used for portfolio simulation.

                Broadcasts.
            
                * If an instance of `vectorbtpro.data.base.OHLCDataMixin`, extracts open, high, low, and close prices.
                * If an instance of `vectorbtpro.portfolio.preparing.FOPreparer`, it is used as a preparer.
                * If an instance of `vectorbtpro.portfolio.preparing.PFPrepResult`, it is used as a preparer result.
            init_cash (Optional[ArrayLike]): Initial capital.
            
                Broadcasts to the final number of columns, or to the number of groups if 
                cash sharing is enabled. See `vectorbtpro.portfolio.enums.InitCashMode` for guidance.
            
                !!! note
                    When using `InitCashMode.AutoAlign`, initial cash values are synchronized across 
                    columns/groups after initialization.
            init_position (Optional[ArrayLike]): Initial position. 
            
                Broadcasts to match the final number of columns.
            init_price (Optional[ArrayLike]): Initial position price. 
            
                Broadcasts to match the final number of columns.
            cash_deposits (Optional[ArrayLike]): Cash deposits or withdrawals at the beginning of each timestamp.
            
                Broadcasts to match the shape of `init_cash`.
            cash_earnings (Optional[ArrayLike]): Cash earnings added at the end of each timestamp.
            
                Broadcasts.
            cash_dividends (Optional[ArrayLike]): Cash dividends added at the end of each timestamp.
            
                Broadcasts, are multiplied by the position, and then added to `cash_earnings`.
            cash_sharing (Optional[bool]): Indicates whether cash is shared among assets within the same group.
            
                If `group_by` is None and this is True, all assets are grouped together for cash sharing.
            
                !!! warning
                    Enables cross-asset dependencies by assuming that all orders in a cash-sharing group 
                    execute in the same tick and retain their price.
            sim_start (Optional[ArrayLike]): Simulation start index (inclusive).
            sim_end (Optional[ArrayLike]): Simulation end index (exclusive).
            call_seq (Optional[ArrayLike]): Sequence dictating the order in which columns are 
                processed per row and group.
            
                Each element specifies the position of a column in the processing order.
                Options include:
            
                * None: Generates a default call sequence.
                * A value from `vectorbtpro.portfolio.enums.CallSeqType`: Creates a full array of the specified type.
                * Custom array: Specifies a user-defined call sequence.
            
                !!! note
                    `CallSeqType.Auto` must be implemented manually. Use 
                    `vectorbtpro.portfolio.nb.from_order_func.sort_call_seq_1d_nb` or 
                    `vectorbtpro.portfolio.nb.from_order_func.sort_call_seq_out_1d_nb` in `pre_segment_func_nb`.
            attach_call_seq (Optional[bool]): Indicates whether to attach the computed call sequence 
                to the portfolio instance.
            segment_mask (Optional[ArrayLike]): Mask that indicates whether a segment should be executed.
            
                An integer activates every n-th row, while a boolean or an array of booleans broadcasts 
                to the number of rows and groups. It does not broadcast together with `close` and 
                `broadcast_named_args`, only against the final shape.
            call_pre_segment (Optional[bool]): Whether to call `pre_segment_func_nb` regardless of `segment_mask`.
            call_post_segment (Optional[bool]): Whether to call `post_segment_func_nb` regardless of `segment_mask`.
            pre_sim_func_nb (Optional[PreSimFuncT]): Function called before simulation.
            
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_pre_func_nb`.
            pre_sim_args (Args): Arguments passed to `pre_sim_func_nb`.
            post_sim_func_nb (Optional[PostSimFuncT]): Function called after simulation.
            
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb`.
            post_sim_args (Args): Arguments passed to `post_sim_func_nb`.
            pre_group_func_nb (Optional[PreGroupFuncT]): Function called before each group.
            
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_pre_func_nb` and 
                is used only when `row_wise` is False.
            pre_group_args (Args): Arguments passed to `pre_group_func_nb`.
            post_group_func_nb (Optional[PostGroupFuncT]): Function called after each group.
            
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb` and 
                is used only when `row_wise` is False.
            post_group_args (Args): Arguments passed to `post_group_func_nb`.
            pre_row_func_nb (Optional[PreRowFuncT]): Function called before each row.
            
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_pre_func_nb` and 
                is used only when `row_wise` is True.
            pre_row_args (Args): Arguments passed to `pre_row_func_nb`.
            post_row_func_nb (Optional[PostRowFuncT]): Function called after each row.
            
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb` and 
                is used only when `row_wise` is True.
            post_row_args (Args): Arguments passed to `post_row_func_nb`.
            pre_segment_func_nb (Optional[PreSegmentFuncT]): Function called before each segment.
            
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_pre_func_nb`.
            pre_segment_args (Args): Arguments passed to `pre_segment_func_nb`.
            post_segment_func_nb (Optional[PostSegmentFuncT]): Function called after each segment.
            
                Defaults to `vectorbtpro.portfolio.nb.from_order_func.no_post_func_nb`.
            post_segment_args (Args): Arguments passed to `post_segment_func_nb`.
            order_func_nb (Optional[OrderFuncT]): Order generation function.
            order_args (Args): Arguments passed to `order_func_nb`.
            flex_order_func_nb (Optional[FlexOrderFuncT]): Flexible order generation function.
            flex_order_args (Args): Arguments passed to `flex_order_func_nb`.
            post_order_func_nb (Optional[PostOrderFuncT]): Callback invoked after processing an order.
            post_order_args (Args): Arguments passed to `post_order_func_nb`.
            open (Optional[ArrayLike]): Open prices.

                Broadcasts. Used as a price boundary (see `vectorbtpro.portfolio.enums.PriceArea`).

                For stop signals, `np.nan` is replaced with the corresponding close price.
            high (Optional[ArrayLike]): High prices.

                Broadcasts. Used as a price boundary (see `vectorbtpro.portfolio.enums.PriceArea`).

                For stop signals, `np.nan` is replaced with the maximum of open and close prices.
            low (Optional[ArrayLike]): Low prices.

                Broadcasts. Used as a price boundary (see `vectorbtpro.portfolio.enums.PriceArea`).

                For stop signals, `np.nan` is replaced with the minimum of open and close prices.
            ffill_val_price (Optional[bool]): If True, valuation price is tracked only when known; 
                otherwise, an unknown `close` may lead to NaN.
            update_value (Optional[bool]): Whether to update group value after each filled order.
            fill_pos_info (Optional[bool]): Whether to fill the position record.
            
                Disable to improve simulation speed in simple cases.
            track_value (Optional[bool]): Whether to track metrics such as current valuation price, value, and return.
            
                Disable to improve simulation speed in simple cases.
            row_wise (Optional[bool]): Whether to iterate over rows instead of by columns/groups.
            max_order_records (Optional[int]): Maximum number of order records expected per column.
            
                Defaults to the number of rows in the broadcasted shape. Set to 0 to disable, 
                lower to reduce memory usage, or higher if multiple orders per timestamp are expected.
            max_log_records (Optional[int]): Maximum number of log records expected per column.
            
                Defaults to the number of rows in the broadcasted shape. Set to 0 to disable, lower to 
                reduce memory usage, or higher if multiple logs per timestamp are expected.
            in_outputs (Optional[tp.MappingLike]): Mapping of in-output objects available via 
                `Portfolio.in_outputs` as a named tuple.
            
                To override `Portfolio` attributes, provide objects that are already broadcasted and grouped 
                (e.g. using `broadcast_named_args` and templates). Also see `Portfolio.in_outputs_indexing_func` 
                for indexing details. When chunking, supply the chunk specification and merging function as per 
                `vectorbtpro.portfolio.chunking.merge_sim_outs`.
            
                !!! note
                    For Numba versions below 0.54, `in_outputs` must be a globally defined named tuple 
                    rather than a mapping.
            seed (Optional[int]): Seed value for generating the call sequence and initializing the simulation.
            group_by (tp.GroupByLike): Grouping specification for columns.
            
                See `vectorbtpro.base.grouping.base.Grouper` for details.
            broadcast_named_args (tp.KwargsLike): Dictionary of named arguments for broadcasting.

                Allows substitution of argument names wrapped with `vectorbtpro.utils.template.Rep`.
            broadcast_kwargs (tp.KwargsLike): Additional keyword arguments to pass to 
                `vectorbtpro.base.reshaping.broadcast`.
            template_context (tp.KwargsLike): Context for template substitution in arguments.
            keep_inout_flex (Optional[bool]): Whether to preserve raw, editable arrays during 
                broadcasting for in-outputs.
            
                Disable to allow editing of `segment_mask`, `cash_deposits`, and `cash_earnings` during simulation.
            jitted (tp.JittedOption): Option to control JIT compilation.
            
                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            
                !!! note
                    Disabling jitting affects only the main simulation function. 
                    To disable compilation entirely, ensure that all functions are uncompiled 
                    (e.g. via the `py_func` attribute or by setting `os.environ['NUMBA_DISABLE_JIT'] = '1'` 
                    before importing vectorbtpro).
            
                !!! warning
                    Parallelization assumes that groups are independent with no data flow between them.
            chunked (tp.ChunkedOption): Option to control chunked processing.
            
                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            staticized (tp.StaticizedOption): Keyword arguments or a task id for staticizing.
            
                If True or a dict, these are passed to `vectorbtpro.utils.cutting.cut_and_save_func` 
                to cache the simulator. If a hashable or callable, it is used as a task id for an already 
                registered simulator. The dict may include options like `override` and `reload`.
            bm_close (Optional[ArrayLike]): Benchmark asset price at each time step.
            
                Broadcasts. If not provided, `close` is used; if set to False, benchmarking is disabled.
            records (Optional[tp.RecordsLike]): Records used to construct arrays.
            
                See `vectorbtpro.base.indexing.IdxRecords`.
            return_preparer (bool): If True, returns the preparer instance 
                (`vectorbtpro.portfolio.preparing.FOFPreparer`).
            
                !!! note
                    In this case, the seed is not automatically set; 
                    invoke `preparer.set_seed()` explicitly if needed.
            return_prep_result (bool): If True, returns the preparer result 
                (`vectorbtpro.portfolio.preparing.PFPrepResult`).
            return_sim_out (bool): If True, returns the simulation output 
                (`vectorbtpro.portfolio.enums.SimulationOutput`).
            **kwargs: Additional keyword arguments passed to the `Portfolio` constructor.

        Returns:
            PortfolioResult: The portfolio result.

        Examples:
            Buy 10 units each tick using closing price:

            ```pycon
            >>> @njit
            ... def order_func_nb(c, size):
            ...     return vbt.pf_nb.order_nb(size=size)

            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_order_func(
            ...     close,
            ...     order_func_nb=order_func_nb,
            ...     order_args=(10,)
            ... )

            >>> pf.assets
            0    10.0
            1    20.0
            2    30.0
            3    40.0
            4    40.0
            dtype: float64
            >>> pf.cash
            0    90.0
            1    70.0
            2    40.0
            3     0.0
            4     0.0
            dtype: float64
            ```

            Reverse each position by first closing it. Use a stateful function to alternate positions:

            ```pycon
            >>> @njit
            ... def pre_group_func_nb(c):
            ...     last_pos_state = np.array([-1])
            ...     return (last_pos_state,)

            >>> @njit
            ... def order_func_nb(c, last_pos_state):
            ...     if c.position_now != 0:
            ...         return vbt.pf_nb.close_position_nb()
            ...
            ...     if last_pos_state[0] == 1:
            ...         size = -np.inf  # open short
            ...         last_pos_state[0] = -1
            ...     else:
            ...         size = np.inf  # open long
            ...         last_pos_state[0] = 1
            ...     return vbt.pf_nb.order_nb(size=size)

            >>> pf = vbt.Portfolio.from_order_func(
            ...     close,
            ...     order_func_nb=order_func_nb,
            ...     pre_group_func_nb=pre_group_func_nb
            ... )

            >>> pf.assets
            0    100.000000
            1      0.000000
            2    -66.666667
            3      0.000000
            4     26.666667
            dtype: float64
            >>> pf.cash
            0      0.000000
            1    200.000000
            2    400.000000
            3    133.333333
            4      0.000000
            dtype: float64
            ```

            Equal-weighted portfolio as in the example under 
            `vectorbtpro.portfolio.nb.from_order_func.from_order_func_nb`:

            ```pycon
            >>> @njit
            ... def pre_group_func_nb(c):
            ...     order_value_out = np.empty(c.group_len, dtype=float_)
            ...     return (order_value_out,)

            >>> @njit
            ... def pre_segment_func_nb(c, order_value_out, size, price, size_type, direction):
            ...     for col in range(c.from_col, c.to_col):
            ...         c.last_val_price[col] = vbt.pf_nb.select_from_col_nb(c, col, price)
            ...     vbt.pf_nb.sort_call_seq_nb(c, size, size_type, direction, order_value_out)
            ...     return ()

            >>> @njit
            ... def order_func_nb(c, size, price, size_type, direction, fees, fixed_fees, slippage):
            ...     return vbt.pf_nb.order_nb(
            ...         size=vbt.pf_nb.select_nb(c, size),
            ...         price=vbt.pf_nb.select_nb(c, price),
            ...         size_type=vbt.pf_nb.select_nb(c, size_type),
            ...         direction=vbt.pf_nb.select_nb(c, direction),
            ...         fees=vbt.pf_nb.select_nb(c, fees),
            ...         fixed_fees=vbt.pf_nb.select_nb(c, fixed_fees),
            ...         slippage=vbt.pf_nb.select_nb(c, slippage)
            ...     )

            >>> np.random.seed(42)
            >>> close = np.random.uniform(1, 10, size=(5, 3))
            >>> size_template = vbt.RepEval('np.array([[1 / group_lens[0]]])')

            >>> pf = vbt.Portfolio.from_order_func(
            ...     close,
            ...     order_func_nb=order_func_nb,
            ...     order_args=(
            ...         size_template,
            ...         vbt.Rep('price'),
            ...         vbt.Rep('size_type'),
            ...         vbt.Rep('direction'),
            ...         vbt.Rep('fees'),
            ...         vbt.Rep('fixed_fees'),
            ...         vbt.Rep('slippage'),
            ...     ),
            ...     segment_mask=2,  # rebalance every second tick
            ...     pre_group_func_nb=pre_group_func_nb,
            ...     pre_segment_func_nb=pre_segment_func_nb,
            ...     pre_segment_args=(
            ...         size_template,
            ...         vbt.Rep('price'),
            ...         vbt.Rep('size_type'),
            ...         vbt.Rep('direction')
            ...     ),
            ...     broadcast_named_args=dict(  # broadcast against each other
            ...         price=close,
            ...         size_type=vbt.pf_enums.SizeType.TargetPercent,
            ...         direction=vbt.pf_enums.Direction.LongOnly,
            ...         fees=0.001,
            ...         fixed_fees=1.,
            ...         slippage=0.001
            ...     ),
            ...     template_context=dict(np=np),  # required by size_template
            ...     cash_sharing=True, group_by=True,  # one group with cash sharing
            ... )

            >>> pf.get_asset_value(group_by=False).vbt.plot().show()
            ```

            ![](/assets/images/api/from_order_func.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/from_order_func.dark.svg#only-dark){: .iimg loading=lazy }

            Templates are a powerful tool to prepare custom arguments after broadcasting and before passing 
            them to simulation functions. In the above example, `broadcast_named_args` broadcasts arguments 
            and templates pass them to callbacks. An evaluation template is used to compute position size 
            based on the number of assets in each group.

            You may ask: why should we bother using broadcasting and templates if we could just pass `size=1/3`?
            Because of flexibility those features provide: we can now pass whatever parameter combinations we want
            and it will work flawlessly. For example, to create two groups of equally-allocated positions,
            we need to change only two parameters:

            ```pycon
            >>> close = np.random.uniform(1, 10, size=(5, 6))  # 6 columns instead of 3
            >>> group_by = ['g1', 'g1', 'g1', 'g2', 'g2', 'g2']  # 2 groups instead of 1
            >>> # Replace close and group_by in the example above

            >>> pf['g1'].get_asset_value(group_by=False).vbt.plot().show()
            ```

            ![](/assets/images/api/from_order_func_g1.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/from_order_func_g1.dark.svg#only-dark){: .iimg loading=lazy }

            ```pycon
            >>> pf['g2'].get_asset_value(group_by=False).vbt.plot().show()
            ```

            ![](/assets/images/api/from_order_func_g2.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/from_order_func_g2.dark.svg#only-dark){: .iimg loading=lazy }

            Combine multiple exit conditions. Exit early if the price exceeds a threshold before an actual exit:

            ```pycon
            >>> @njit
            ... def pre_sim_func_nb(c):
            ...     # We need to define stop price per column once
            ...     stop_price = np.full(c.target_shape[1], np.nan, dtype=float_)
            ...     return (stop_price,)

            >>> @njit
            ... def order_func_nb(c, stop_price, entries, exits, size):
            ...     # Select info related to this order
            ...     entry_now = vbt.pf_nb.select_nb(c, entries)
            ...     exit_now = vbt.pf_nb.select_nb(c, exits)
            ...     size_now = vbt.pf_nb.select_nb(c, size)
            ...     price_now = vbt.pf_nb.select_nb(c, c.close)
            ...     stop_price_now = stop_price[c.col]
            ...
            ...     # Our logic
            ...     if entry_now:
            ...         if c.position_now == 0:
            ...             return vbt.pf_nb.order_nb(
            ...                 size=size_now,
            ...                 price=price_now,
            ...                 direction=vbt.pf_enums.Direction.LongOnly)
            ...     elif exit_now or price_now >= stop_price_now:
            ...         if c.position_now > 0:
            ...             return vbt.pf_nb.order_nb(
            ...                 size=-size_now,
            ...                 price=price_now,
            ...                 direction=vbt.pf_enums.Direction.LongOnly)
            ...     return vbt.pf_enums.NoOrder

            >>> @njit
            ... def post_order_func_nb(c, stop_price, stop):
            ...     # Same broadcasting as for size
            ...     stop_now = vbt.pf_nb.select_nb(c, stop)
            ...
            ...     if c.order_result.status == vbt.pf_enums.OrderStatus.Filled:
            ...         if c.order_result.side == vbt.pf_enums.OrderSide.Buy:
            ...             # Position entered: Set stop condition
            ...             stop_price[c.col] = (1 + stop_now) * c.order_result.price
            ...         else:
            ...             # Position exited: Remove stop condition
            ...             stop_price[c.col] = np.nan

            >>> def simulate(close, entries, exits, size, stop):
            ...     return vbt.Portfolio.from_order_func(
            ...         close,
            ...         order_func_nb=order_func_nb,
            ...         order_args=(vbt.Rep('entries'), vbt.Rep('exits'), vbt.Rep('size')),
            ...         pre_sim_func_nb=pre_sim_func_nb,
            ...         post_order_func_nb=post_order_func_nb,
            ...         post_order_args=(vbt.Rep('stop'),),
            ...         broadcast_named_args=dict(  # broadcast against each other
            ...             entries=entries,
            ...             exits=exits,
            ...             size=size,
            ...             stop=stop
            ...         )
            ...     )

            >>> close = pd.Series([10, 11, 12, 13, 14])
            >>> entries = pd.Series([True, True, False, False, False])
            >>> exits = pd.Series([False, False, False, True, True])
            >>> simulate(close, entries, exits, np.inf, 0.1).asset_flow
            0    10.0
            1     0.0
            2   -10.0
            3     0.0
            4     0.0
            dtype: float64

            >>> simulate(close, entries, exits, np.inf, 0.2).asset_flow
            0    10.0
            1     0.0
            2   -10.0
            3     0.0
            4     0.0
            dtype: float64

            >>> simulate(close, entries, exits, np.inf, np.nan).asset_flow
            0    10.0
            1     0.0
            2     0.0
            3   -10.0
            4     0.0
            dtype: float64
            ```
            The stop of 10% does not trigger an order at the second time step because it occurs simultaneously 
            with entry; it must wait until no entry is present. To exit regardless of an entry, replace "elif" 
            with "if" (similar to using `ConflictMode.Opposite` in `Portfolio.from_signals`).

            Multiple parameter combinations can be tested at once via broadcasting using
            `vectorbtpro.base.reshaping.broadcast`:

            ```pycon
            >>> stop = pd.DataFrame([[0.1, 0.2, np.nan]])
            >>> simulate(close, entries, exits, np.inf, stop).asset_flow
                  0     1     2
            0  10.0  10.0  10.0
            1   0.0   0.0   0.0
            2 -10.0 -10.0   0.0
            3   0.0   0.0 -10.0
            4   0.0   0.0   0.0
            ```

            Or use a Cartesian product:

            ```pycon
            >>> stop = vbt.Param([0.1, 0.2, np.nan])
            >>> simulate(close, entries, exits, np.inf, stop).asset_flow
            threshold   0.1   0.2   NaN
            0          10.0  10.0  10.0
            1           0.0   0.0   0.0
            2         -10.0 -10.0   0.0
            3           0.0   0.0 -10.0
            4           0.0   0.0   0.0
            ```

            This works because `pd.Index` automatically translates into `vectorbtpro.base.reshaping.BCO`
            with `product` set to True.

            Let's illustrate how to generate multiple orders per symbol and bar.
            For each bar, buy at open and sell at close:

            ```pycon
            >>> @njit
            ... def flex_order_func_nb(c, size):
            ...     if c.call_idx == 0:
            ...         return c.from_col, vbt.pf_nb.order_nb(size=size, price=c.open[c.i, c.from_col])
            ...     if c.call_idx == 1:
            ...         return c.from_col, vbt.pf_nb.close_position_nb(price=c.close[c.i, c.from_col])
            ...     return -1, vbt.pf_enums.NoOrder

            >>> open = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> close = pd.DataFrame({'a': [2, 3, 4], 'b': [3, 4, 5]})
            >>> size = 1
            >>> pf = vbt.Portfolio.from_order_func(
            ...     close,
            ...     flex_order_func_nb=flex_order_func_nb,
            ...     flex_order_args=(size,),
            ...     open=open,
            ...     max_order_records=close.shape[0] * 2
            ... )

            >>> pf.orders.readable
                Order Id Column  Timestamp  Size  Price  Fees  Side
            0          0      a          0   1.0    1.0   0.0   Buy
            1          1      a          0   1.0    2.0   0.0  Sell
            2          2      a          1   1.0    2.0   0.0   Buy
            3          3      a          1   1.0    3.0   0.0  Sell
            4          4      a          2   1.0    3.0   0.0   Buy
            5          5      a          2   1.0    4.0   0.0  Sell
            6          0      b          0   1.0    4.0   0.0   Buy
            7          1      b          0   1.0    3.0   0.0  Sell
            8          2      b          1   1.0    5.0   0.0   Buy
            9          3      b          1   1.0    4.0   0.0  Sell
            10         4      b          2   1.0    6.0   0.0   Buy
            11         5      b          2   1.0    5.0   0.0  Sell
            ```

            !!! warning
                Each bar is treated as a black box—price movements within a bar are unknown. 
                Since order processing must mirror real-world conditions, only the opening and 
                closing prices remain reliably ordered.
        """
        if isinstance(close, FOFPreparer):
            preparer = close
            prep_result = None
        elif isinstance(close, PFPrepResult):
            preparer = None
            prep_result = close
        else:
            local_kwargs = locals()
            local_kwargs = {**local_kwargs, **local_kwargs["kwargs"]}
            del local_kwargs["kwargs"]
            del local_kwargs["cls"]
            del local_kwargs["return_preparer"]
            del local_kwargs["return_prep_result"]
            del local_kwargs["return_sim_out"]
            parsed_data = BasePFPreparer.parse_data(close, all_ohlc=True)
            if parsed_data is not None:
                local_kwargs["data"] = parsed_data
                local_kwargs["close"] = None
            preparer = FOFPreparer(**local_kwargs)
            if not return_preparer:
                preparer.set_seed()
            prep_result = None
        if return_preparer:
            return preparer
        if prep_result is None:
            prep_result = preparer.result
        if return_prep_result:
            return prep_result
        sim_out = prep_result.target_func(**prep_result.target_args)
        if return_sim_out:
            return sim_out
        return cls(order_records=sim_out, **prep_result.pf_args)

    @classmethod
    def from_def_order_func(
        cls: tp.Type[PortfolioT],
        close: tp.Union[tp.ArrayLike, OHLCDataMixin, FDOFPreparer, PFPrepResult],
        size: tp.Optional[tp.ArrayLike] = None,
        size_type: tp.Optional[tp.ArrayLike] = None,
        direction: tp.Optional[tp.ArrayLike] = None,
        price: tp.Optional[tp.ArrayLike] = None,
        fees: tp.Optional[tp.ArrayLike] = None,
        fixed_fees: tp.Optional[tp.ArrayLike] = None,
        slippage: tp.Optional[tp.ArrayLike] = None,
        min_size: tp.Optional[tp.ArrayLike] = None,
        max_size: tp.Optional[tp.ArrayLike] = None,
        size_granularity: tp.Optional[tp.ArrayLike] = None,
        leverage: tp.Optional[tp.ArrayLike] = None,
        leverage_mode: tp.Optional[tp.ArrayLike] = None,
        reject_prob: tp.Optional[tp.ArrayLike] = None,
        price_area_vio_mode: tp.Optional[tp.ArrayLike] = None,
        allow_partial: tp.Optional[tp.ArrayLike] = None,
        raise_reject: tp.Optional[tp.ArrayLike] = None,
        log: tp.Optional[tp.ArrayLike] = None,
        pre_segment_func_nb: tp.Optional[nb.PreSegmentFuncT] = None,
        order_func_nb: tp.Optional[nb.OrderFuncT] = None,
        flex_order_func_nb: tp.Optional[nb.FlexOrderFuncT] = None,
        val_price: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        call_seq: tp.Optional[tp.ArrayLike] = None,
        flexible: bool = False,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        chunked: tp.ChunkedOption = None,
        return_preparer: bool = False,
        return_prep_result: bool = False,
        return_sim_out: bool = False,
        **kwargs,
    ) -> PortfolioResultT:
        """Build portfolio using the default order function.

        This method constructs a portfolio by applying the default order function, which processes
        trading orders based on parameters such as size, price, fees, and other available data. It also
        employs a segment preprocessing function that adjusts valuation prices and sorts the call sequence,
        behaving similarly to `Portfolio.from_orders` but with enhanced control through pre- and postprocessing.
        The method supports argument chunking for efficient processing, although `Portfolio.from_orders` may be
        up to 5x faster.

        If `flexible` is True:

        * `pre_segment_func_nb` is set to `vectorbtpro.portfolio.nb.from_order_func.def_flex_pre_segment_func_nb`
        * `flex_order_func_nb` is set to `vectorbtpro.portfolio.nb.from_order_func.def_flex_order_func_nb`

        If `flexible` is False:

        * `pre_segment_func_nb` is set to `vectorbtpro.portfolio.nb.from_order_func.def_pre_segment_func_nb`
        * `order_func_nb` is set to `vectorbtpro.portfolio.nb.from_order_func.def_order_func_nb`

        Prepared by `vectorbtpro.portfolio.preparing.FDOFPreparer`.

        For defaults, see `vectorbtpro._settings.portfolio`. These defaults are not used to fill NaN values 
        after reindexing; vectorbtpro uses its own defaults (typically NaN for floating arrays and preset 
        flags for integer arrays). Use `vectorbtpro.base.reshaping.BCO` with `fill_value` to override.

        !!! hint
            All broadcastable arguments are handled using `vectorbtpro.base.reshaping.broadcast` 
            to preserve their original shapes for flexible indexing and memory efficiency. 
            Each can be provided per frame, series, row, column, or individual element.

        !!! note
            All provided functions must be Numba-compiled if Numba is enabled. 
            Also see the notes in `Portfolio.from_orders`.

        Args:
            close (Union[ArrayLike, OHLCDataMixin, FDOFPreparer, PFPrepResult]): The close prices or 
                OHLC data used for portfolio simulation.

                Broadcasts.
            
                * If an instance of `vectorbtpro.data.base.OHLCDataMixin`, extracts open, high, low, and close prices.
                * If an instance of `vectorbtpro.portfolio.preparing.FOPreparer`, it is used as a preparer.
                * If an instance of `vectorbtpro.portfolio.preparing.PFPrepResult`, it is used as a preparer result.
            size (Optional[ArrayLike]): Size to order. 
            
                Broadcasts. See `vectorbtpro.portfolio.enums.Order.size`. 
            size_type (Optional[ArrayLike]): Order size type. 
            
                Broadcasts. See `vectorbtpro.portfolio.enums.SizeType` and 
                `vectorbtpro.portfolio.enums.Order.size_type`. 
            direction (Optional[ArrayLike]): Order direction. 
            
                Broadcasts. See `vectorbtpro.portfolio.enums.Direction` and 
                `vectorbtpro.portfolio.enums.Order.direction`.
            price (Optional[ArrayLike]): Order price.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.price` and 
                `vectorbtpro.portfolio.enums.PriceType`.
            
                Options such as `PriceType.NextOpen` and `PriceType.NextClose` apply per column and 
                require `from_ago` to be None.
            fees (Optional[ArrayLike]): Fees as a percentage of the order value.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.fees`. 
            fixed_fees (Optional[ArrayLike]): Fixed fee amount per order.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.fixed_fees`. 
            slippage (Optional[ArrayLike]): Slippage percentage of the order price.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.slippage`. 
            min_size (Optional[ArrayLike]): Minimum order size.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.min_size`.
            max_size (Optional[ArrayLike]): Maximum order size.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.max_size`.
            size_granularity (Optional[ArrayLike]): Granularity of the order size.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.size_granularity`.
            leverage (Optional[ArrayLike]): Leverage applied in the order.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.leverage`.
            leverage_mode (Optional[ArrayLike]): Leverage mode.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.leverage_mode`.
            reject_prob (Optional[ArrayLike]): Order rejection probability.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.reject_prob`.
            price_area_vio_mode (Optional[ArrayLike]): Price area violation mode.

                Broadcasts. See `vectorbtpro.portfolio.enums.PriceAreaVioMode`.
            allow_partial (Optional[ArrayLike]): Indicates whether partial fills are allowed.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.allow_partial`.
            raise_reject (Optional[ArrayLike]): Indicates whether to raise an exception upon order rejection.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.raise_reject`.
            log (Optional[ArrayLike]): Flag indicating whether to log orders.

                Broadcasts. See `vectorbtpro.portfolio.enums.Order.log`.
            pre_segment_func_nb (Optional[PreSegmentFuncT]): Function called before each segment.
            order_func_nb (Optional[OrderFuncT]): Order generation function.
            flex_order_func_nb (Optional[FlexOrderFuncT]): Flexible order generation function.
            val_price (Optional[ArrayLike]): Asset valuation price used in decision making.
            
                Broadcasts. Can also be provided as `vectorbtpro.portfolio.enums.ValPriceType`.
            
                * Any `-np.inf` element is replaced by the latest valuation price 
                    (using `open` or a previously known value if `ffill_val_price` is True).
                * Any `np.inf` element is replaced by the current order price.
            
                !!! note
                    Unlike `Portfolio.from_order_func`, the order price is effectively predetermined,
                    so `val_price` defaults to the current order price when using `np.inf`. 
                    To use the previous close, set it in the settings to `-np.inf`.
            
                !!! note
                    Ensure that the timestamp associated with `val_price` precedes all order 
                    timestamps in a cash-sharing group.
            sim_start (Optional[ArrayLike]): Simulation start row or index (inclusive).
            
                Can be "auto" to automatically select the first non-NA size value.
            sim_end (Optional[ArrayLike]): Simulation end row or index (exclusive).
            
                Can be "auto" to automatically select the first non-NA size value.
            call_seq (Optional[ArrayLike]): Sequence dictating the order in which columns are 
                processed per row and group.
            
                Each element specifies the position of a column in the processing order.
                Options include:
            
                * None: Generates a default call sequence.
                * A value from `vectorbtpro.portfolio.enums.CallSeqType`: Creates a full array of the specified type.
                * Custom array: Specifies a user-defined call sequence.
            
                If set to `CallSeqType.Auto`, orders are rearranged dynamically so that sell orders are 
                processed before buy orders.
            
                !!! warning
                    `CallSeqType.Auto` assumes predetermined order prices and flexible execution, 
                    which may not accurately reflect real-time conditions.
                    For stricter control, use `Portfolio.from_order_func`.
            flexible (bool): Whether to apply flexible processing.
            broadcast_named_args (tp.KwargsLike): Dictionary of named arguments for broadcasting.

                Allows substitution of argument names wrapped with `vectorbtpro.utils.template.Rep`.
            broadcast_kwargs (tp.KwargsLike): Additional keyword arguments to pass to 
                `vectorbtpro.base.reshaping.broadcast`.
            chunked (tp.ChunkedOption): Option to control chunked processing.
            
                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            return_preparer (bool): If True, returns the preparer instance 
                (`vectorbtpro.portfolio.preparing.FPreparer`).
            
                !!! note
                    In this case, the seed is not automatically set; 
                    invoke `preparer.set_seed()` explicitly if needed.
            return_prep_result (bool): If True, returns the preparer result 
                (`vectorbtpro.portfolio.preparing.PFPrepResult`).
            return_sim_out (bool): If True, returns the simulation output 
                (`vectorbtpro.portfolio.enums.SimulationOutput`).
            **kwargs: Additional keyword arguments for `Portfolio.from_order_func`.

        Returns:
            PortfolioResult: The portfolio result.

        Examples:
            Working with `Portfolio.from_def_order_func` is similar to using `Portfolio.from_orders`:

            ```pycon
            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> pf = vbt.Portfolio.from_def_order_func(close, 10)

            >>> pf.assets
            0    10.0
            1    20.0
            2    30.0
            3    40.0
            4    40.0
            dtype: float64
            >>> pf.cash
            0    90.0
            1    70.0
            2    40.0
            3     0.0
            4     0.0
            dtype: float64
            ```

            Equal-weighted portfolio as in the example under `Portfolio.from_order_func` but 
            much less verbose and with pre-computed asset values during simulation for enhanced performance:

            ```pycon
            >>> np.random.seed(42)
            >>> close = np.random.uniform(1, 10, size=(5, 3))

            >>> @njit
            ... def post_segment_func_nb(c):
            ...     for col in range(c.from_col, c.to_col):
            ...         c.in_outputs.asset_value_pc[c.i, col] = c.last_position[col] * c.last_val_price[col]

            >>> pf = vbt.Portfolio.from_def_order_func(
            ...     close,
            ...     size=1/3,
            ...     size_type='targetpercent',
            ...     direction='longonly',
            ...     fees=0.001,
            ...     fixed_fees=1.,
            ...     slippage=0.001,
            ...     segment_mask=2,
            ...     cash_sharing=True,
            ...     group_by=True,
            ...     call_seq='auto',
            ...     post_segment_func_nb=post_segment_func_nb,
            ...     call_post_segment=True,
            ...     in_outputs=dict(asset_value_pc=vbt.RepEval('np.empty_like(close)'))
            ... )

            >>> asset_value = pf.wrapper.wrap(pf.in_outputs.asset_value_pc, group_by=False)
            >>> asset_value.vbt.plot().show()
            ```

            ![](/assets/images/api/from_def_order_func.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/from_def_order_func.dark.svg#only-dark){: .iimg loading=lazy }
        """
        if isinstance(close, FDOFPreparer):
            preparer = close
            prep_result = None
        elif isinstance(close, PFPrepResult):
            preparer = None
            prep_result = close
        else:
            local_kwargs = locals()
            local_kwargs = {**local_kwargs, **local_kwargs["kwargs"]}
            del local_kwargs["kwargs"]
            del local_kwargs["cls"]
            del local_kwargs["return_preparer"]
            del local_kwargs["return_prep_result"]
            del local_kwargs["return_sim_out"]
            parsed_data = BasePFPreparer.parse_data(close, all_ohlc=True)
            if parsed_data is not None:
                local_kwargs["data"] = parsed_data
                local_kwargs["close"] = None
            preparer = FDOFPreparer(**local_kwargs)
            if not return_preparer:
                preparer.set_seed()
            prep_result = None
        if return_preparer:
            return preparer
        if prep_result is None:
            prep_result = preparer.result
        if return_prep_result:
            return prep_result
        sim_out = prep_result.target_func(**prep_result.target_args)
        if return_sim_out:
            return sim_out
        return cls(order_records=sim_out, **prep_result.pf_args)

    # ############# Grouping ############# #

    def regroup(self: PortfolioT, group_by: tp.GroupByLike, **kwargs) -> PortfolioT:
        """Regroup the portfolio based on a new grouping specification.

        This method reassigns the grouping of the portfolio. See
        `vectorbtpro.base.wrapping.Wrapping.regroup` for further details.

        !!! note
            All cached objects will be lost.

        Args:
            group_by (GroupByLike): Grouping key or specification.
            **kwargs: Additional keyword arguments for grouping.

        Returns:
            Portfolio: A new portfolio instance with the updated grouping.
        """
        if self.cash_sharing:
            if self.wrapper.grouper.is_grouping_modified(group_by=group_by):
                raise ValueError("Cannot modify grouping globally when cash_sharing=True")
        return Wrapping.regroup(self, group_by, **kwargs)

    # ############# Properties ############# #

    @property
    def cash_sharing(self) -> bool:
        """Whether to share cash within the same group.

        Returns:
            bool: True if portfolio cash is shared among assets within the same group, False otherwise.
        """
        return self._cash_sharing

    @property
    def in_outputs(self) -> tp.Optional[tp.NamedTuple]:
        """Named tuple with in-output objects.

        Returns:
            Optional[NamedTuple]: A named tuple containing additional in-output objects, or None if not set.
        """
        return self._in_outputs

    @property
    def use_in_outputs(self) -> bool:
        """Whether to return in-output objects when calling properties.

        Returns:
            bool: True if in-output objects should be used for property values; otherwise, False.
        """
        return self._use_in_outputs

    @property
    def fillna_close(self) -> bool:
        """Whether to forward-backward fill NaN values in `Portfolio.close`.

        Returns:
            bool: True if NaN values in the close price series should be filled both forward 
                and backward, False otherwise.
        """
        return self._fillna_close

    @property
    def year_freq(self) -> tp.Optional[tp.PandasFrequency]:
        """Year frequency derived from the portfolio's index.

        Returns:
            Optional[PandasFrequency]: The inferred yearly frequency based on the portfolio's index, 
                or None if it cannot be determined.
        """
        return ReturnsAccessor.get_year_freq(
            year_freq=self._year_freq,
            index=self.wrapper.index,
            freq=self.wrapper.freq,
        )

    @property
    def returns_acc_defaults(self) -> tp.KwargsLike:
        """Defaults for `vectorbtpro.returns.accessors.ReturnsAccessor`.

        Returns:
            KwargsLike: A dictionary of default parameters to configure the returns accessor.
        """
        return self._returns_acc_defaults

    @property
    def trades_type(self) -> int:
        """Default `vectorbtpro.portfolio.trades.Trades` type to use across `Portfolio`.

        Returns:
            int: An integer representing the default trades type.
        """
        return self._trades_type

    @property
    def orders_cls(self) -> tp.Type:
        """Order records wrapper class.

        Returns:
            Type: The class used to wrap order records. 
            
                Defaults to `vectorbtpro.portfolio.orders.Orders` if not explicitly set.
        """
        if self._orders_cls is None:
            return Orders
        return self._orders_cls

    @property
    def logs_cls(self) -> tp.Type:
        """Log records wrapper class.

        Returns:
            Type: The class used to wrap log records. 
            
                Defaults to `vectorbtpro.portfolio.logs.Logs` if not explicitly specified.
        """
        if self._logs_cls is None:
            return Logs
        return self._logs_cls

    @property
    def trades_cls(self) -> tp.Type:
        """Trade records wrapper class.

        Returns:
            Type: The class used to wrap trade records. 
            
                Defaults to `vectorbtpro.portfolio.trades.Trades` if not explicitly configured.
        """
        if self._trades_cls is None:
            return Trades
        return self._trades_cls

    @property
    def entry_trades_cls(self) -> tp.Type:
        """Entry trade records wrapper class.

        Returns:
            Type: The class used to wrap entry trade records.

                Defaults to `vectorbtpro.portfolio.trades.EntryTrades` if not explicitly configured.
        """
        if self._entry_trades_cls is None:
            return EntryTrades
        return self._entry_trades_cls

    @property
    def exit_trades_cls(self) -> tp.Type:
        """Exit trade records wrapper class.

        Returns:
            Type: The class used for wrapping exit trade records.

                Defaults to `vectorbtpro.portfolio.trades.ExitTrades` if not explicitly configured.
        """
        if self._exit_trades_cls is None:
            return ExitTrades
        return self._exit_trades_cls

    @property
    def positions_cls(self) -> tp.Type:
        """Position records wrapper class.

        Returns:
            Type: The class used to represent position records.

                Defaults to `vectorbtpro.portfolio.trades.Positions` if not explicitly configured.
        """
        if self._positions_cls is None:
            return Positions
        return self._positions_cls

    @property
    def drawdowns_cls(self) -> tp.Type:
        """Drawdown records wrapper class.

        Returns:
            Type: The class used to wrap drawdown records.

                Defaults to `vectorbtpro.generic.drawdowns.Drawdowns` if not explicitly configured.
        """
        if self._drawdowns_cls is None:
            return Drawdowns
        return self._drawdowns_cls

    @custom_property(group_by_aware=False)
    def call_seq(self) -> tp.Optional[tp.SeriesFrame]:
        """Sequence of call identifiers per row and group.

        Returns:
            Optional[SeriesFrame]: A pandas Series or DataFrame representing the sequence of call 
                identifiers, or None if no call sequence exists.
        """
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "call_seq"):
            call_seq = self.in_outputs.call_seq
        else:
            call_seq = self._call_seq
        if call_seq is None:
            return None

        return self.wrapper.wrap(call_seq, group_by=False)

    @property
    def cash_deposits_as_input(self) -> bool:
        """Flag indicating whether cash deposits are added to the input value when calculating returns,
        as opposed to subtracting them from the output value.

        Returns:
            bool: True if cash deposits are included in the input value for return calculations; False otherwise.
        """
        return self._cash_deposits_as_input

    # ############# Price ############# #

    @property
    def open_flex(self) -> tp.Optional[tp.ArrayLike]:
        """`Portfolio.open` returned in a format that supports flexible indexing.

        Returns:
            Optional[ArrayLike]: The open price data in a flexible format, or None if not provided.
        """
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "open"):
            open = self.in_outputs.open
        else:
            open = self._open
        return open

    @property
    def high_flex(self) -> tp.Optional[tp.ArrayLike]:
        """`Portfolio.high` returned in a format that supports flexible indexing.

        Returns:
            Optional[ArrayLike]: The high price data in a flexible format, or None if not specified.
        """
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "high"):
            high = self.in_outputs.high
        else:
            high = self._high
        return high

    @property
    def low_flex(self) -> tp.Optional[tp.ArrayLike]:
        """`Portfolio.low` returned in a format that supports flexible indexing.

        Returns:
            Optional[ArrayLike]: The low price data in a flexible format, or None if not available.
        """
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "low"):
            low = self.in_outputs.low
        else:
            low = self._low
        return low

    @property
    def close_flex(self) -> tp.ArrayLike:
        """`Portfolio.close` returned in a format that supports flexible indexing.

        Returns:
            ArrayLike: The close price data in a flexible format.
        """
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "close"):
            close = self.in_outputs.close
        else:
            close = self._close
        return close

    @custom_property(group_by_aware=False, resample_func="first")
    def open(self) -> tp.Optional[tp.SeriesFrame]:
        """Open price of each time step.

        Returns:
            Optional[SeriesFrame]: A SeriesFrame containing the open prices for each time step, 
                or None if open price data is not provided.
        """
        if self.open_flex is None:
            return None
        return self.wrapper.wrap(self.open_flex, group_by=False)

    @custom_property(group_by_aware=False, resample_func="max")
    def high(self) -> tp.Optional[tp.SeriesFrame]:
        """High price of each time step.

        Returns:
            Optional[SeriesFrame]: A SeriesFrame containing the high prices for each time step, 
                or None if high price data is missing.
        """
        if self.high_flex is None:
            return None
        return self.wrapper.wrap(self.high_flex, group_by=False)

    @custom_property(group_by_aware=False, resample_func="min")
    def low(self) -> tp.Optional[tp.SeriesFrame]:
        """Low price of each time step.

        Returns:
            Optional[SeriesFrame]: A SeriesFrame of the low prices for each time step, 
                or None if low price data is not provided.
        """
        if self.low_flex is None:
            return None
        return self.wrapper.wrap(self.low_flex, group_by=False)

    @custom_property(group_by_aware=False, resample_func="last")
    def close(self) -> tp.SeriesFrame:
        """Last asset price at each time step.

        Returns:
            SeriesFrame: A SeriesFrame containing the close prices for each time step.
        """
        return self.wrapper.wrap(self.close_flex, group_by=False)

    @hybrid_method
    def get_filled_close(
        cls_or_self,
        close: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get forward and backward filled closing price.
    
        See:
            `vectorbtpro.generic.nb.base.fbfill_nb`
    
        Args:
            close (Optional[SeriesFrame]): Price data to fill.
        
                Uses `Portfolio.close` if not provided.
            jitted (JittedOption): Option controlling JIT compilation.
            chunked (ChunkedOption): Option controlling chunked processing.
            wrapper (Optional[ArrayWrapper]): Wrapper instance.

                Uses `Portfolio.wrapper` if not provided.
            wrap_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.
    
        Returns:
            SeriesFrame: The forward and backward filled closing price data.
        """
        if not isinstance(cls_or_self, type):
            if close is None:
                close = cls_or_self.close
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(close, arg_name="close")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        func = jit_reg.resolve_option(generic_nb.fbfill_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        filled_close = func(to_2d_array(close))
        return wrapper.wrap(filled_close, group_by=False, **resolve_dict(wrap_kwargs))

    @custom_property(group_by_aware=False, resample_func="last")
    def bm_close(self) -> tp.Union[None, bool, tp.SeriesFrame]:
        """Benchmark price per unit series.
    
        Returns:
            Union[None, bool, SeriesFrame]
        """
        if self.use_in_outputs and self.in_outputs is not None and hasattr(self.in_outputs, "bm_close"):
            bm_close = self.in_outputs.bm_close
        else:
            bm_close = self._bm_close

        if bm_close is None or isinstance(bm_close, bool):
            return bm_close
        return self.wrapper.wrap(bm_close, group_by=False)

    @hybrid_method
    def get_filled_bm_close(
        cls_or_self,
        bm_close: tp.Optional[tp.SeriesFrame] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Union[None, bool, tp.SeriesFrame]:
        """Get forward and backward filled benchmark closing price.
    
        See:
            `vectorbtpro.generic.nb.base.fbfill_nb`
    
        Args:
            bm_close (Optional[SeriesFrame]): Benchmark price data to fill.
        
                Uses `Portfolio.bm_close` if not provided.
            jitted (JittedOption): Option controlling JIT compilation.
            chunked (ChunkedOption): Option controlling chunked processing.
            wrapper (Optional[ArrayWrapper]): Wrapper instance.

                Uses `Portfolio.wrapper` if not provided.
            wrap_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.
    
        Returns:
            Union[None, bool, SeriesFrame]: The forward and backward filled benchmark closing price data, 
                or the original boolean/None value.
        """
        if not isinstance(cls_or_self, type):
            if bm_close is None:
                bm_close = cls_or_self.bm_close
                if bm_close is None or isinstance(bm_close, bool):
                    return bm_close
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(bm_close, arg_name="bm_close")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        func = jit_reg.resolve_option(generic_nb.fbfill_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        filled_bm_close = func(to_2d_array(bm_close))
        return wrapper.wrap(filled_bm_close, group_by=False, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_weights(
        cls_or_self,
        weights: tp.Union[None, bool, tp.ArrayLike] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
    ) -> tp.Union[None, tp.ArrayLike, tp.Series]:
        """Get asset weights.
    
        Args:
            weights (Union[None, bool, ArrayLike]): The asset weights data.

                Uses `Portfolio._weights` if not provided.
        
                If None or False, the function returns None.
            wrapper (Optional[ArrayWrapper]): Wrapper instance.

                Uses `Portfolio.wrapper` if not provided.
    
        Returns:
            Union[None, ArrayLike, Series]: The wrapped asset weights, or None if weights are not provided.
        """
        if not isinstance(cls_or_self, type):
            if weights is None:
                weights = cls_or_self._weights
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        if weights is None or weights is False:
            return None
        return wrapper.wrap_reduced(weights, group_by=False)

    # ############# Views ############# #

    def apply_weights(
        self: PortfolioT,
        weights: tp.Union[None, bool, tp.ArrayLike] = None,
        rescale: bool = False,
        group_by: tp.GroupByLike = None,
        apply_group_by: bool = False,
        **kwargs,
    ) -> PortfolioT:
        """Get view of portfolio with asset weights applied and optionally rescaled.

        If `rescale` is True, weights are rescaled relative to other weights in the same group.
        For example, weights 0.5 and 0.5 are rescaled to 1.0 and 1.0 respectively, while
        weights 0.7 and 0.3 are rescaled to 1.4 and 0.6, respectively.

        Args:
            weights (Union[None, bool, ArrayLike]): Asset weights to apply. 

                Uses `Portfolio.get_weights` if not provided.
            
                Passing None or False disables weights.
            rescale (bool): If True, rescale weights relative to other weights in the same group.
            group_by (GroupByLike): Specification for grouping assets.
            apply_group_by (bool): Flag indicating whether to apply grouping based on `group_by`.
            **kwargs: Additional keyword arguments passed to the replace method.

        Returns:
            Portfolio: Portfolio view with asset weights applied.
        """
        if weights is not None and weights is not False:
            weights = to_1d_array(self.get_weights(weights=weights))
            if rescale:
                if self.wrapper.grouper.is_grouped(group_by=group_by):
                    new_weights = np.empty(len(weights), dtype=float_)
                    for group_idxs in self.wrapper.grouper.iter_group_idxs(group_by=group_by):
                        group_weights = weights[group_idxs]
                        new_weights[group_idxs] = group_weights * len(group_weights) / group_weights.sum()
                    weights = new_weights
                else:
                    weights = weights * len(weights) / weights.sum()
        if group_by is not None and apply_group_by:
            _self = self.regroup(group_by=group_by)
        else:
            _self = self
        return _self.replace(weights=weights, **kwargs)

    def disable_weights(self: PortfolioT, **kwargs) -> PortfolioT:
        """Get view of portfolio with asset weights disabled.

        Args:
            **kwargs: Additional keyword arguments passed to the replace method.

        Returns:
            Portfolio: Portfolio view with asset weights disabled.
        """
        return self.replace(weights=False, **kwargs)

    def get_long_view(
        self: PortfolioT,
        orders: tp.Optional[Orders] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> PortfolioT:
        """Get view of portfolio with long positions only.

        Args:
            orders (Optional[Orders]): Orders to generate the long view. 
            
                Uses `Portfolio.get_orders` if not provided.
            init_position (Optional[ArrayLike]): Initial positions.

                Uses `Portfolio._init_position` if not provided.
            init_price (Optional[ArrayLike]): Initial prices.

                Uses `Portfolio._init_price` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start time.
            sim_end (Optional[ArrayLike]): Simulation end time.
            rec_sim_range (bool): Flag indicating whether to record the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to enable chunked processing.
            **kwargs: Additional keyword arguments passed to the replace method.

        Returns:
            Portfolio: Portfolio view containing only long positions, with non-long positions 
                set to zero and price values updated accordingly.
        """
        if orders is None:
            orders = self.resolve_shortcut_attr(
                "orders",
                sim_start=sim_start,
                sim_end=sim_end,
                rec_sim_range=rec_sim_range,
                weights=False,
            )
        if init_position is None:
            init_position = self._init_position
        if init_price is None:
            init_price = self._init_price
        new_order_records = orders.get_long_view(
            init_position=init_position,
            init_price=init_price,
            jitted=jitted,
            chunked=chunked,
        ).values
        init_position = broadcast_array_to(init_position, self.wrapper.shape_2d[1])
        init_price = broadcast_array_to(init_price, self.wrapper.shape_2d[1])
        new_init_position = np.where(init_position > 0, init_position, 0)
        new_init_price = np.where(init_position > 0, init_price, np.nan)
        return self.replace(
            order_records=new_order_records,
            init_position=new_init_position,
            init_price=new_init_price,
            **kwargs,
        )

    def get_short_view(
        self: PortfolioT,
        orders: tp.Optional[Orders] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> PortfolioT:
        """Get view of portfolio with short positions only.

        Args:
            orders (Optional[Orders]): Orders to generate the short view. 
            
                Uses `Portfolio.get_orders` if not provided.
            init_position (Optional[ArrayLike]): Initial positions.

                Uses `Portfolio._init_position` if not provided.
            init_price (Optional[ArrayLike]): Initial prices.

                Uses `Portfolio._init_price` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start time.
            sim_end (Optional[ArrayLike]): Simulation end time.
            rec_sim_range (bool): Flag indicating whether to record the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to enable chunked processing.
            **kwargs: Additional keyword arguments passed to the replace method.

        Returns:
            Portfolio: Portfolio view containing only short positions, with non-short positions 
                set to zero and price values updated accordingly.
        """
        if orders is None:
            orders = self.resolve_shortcut_attr(
                "orders",
                sim_start=sim_start,
                sim_end=sim_end,
                rec_sim_range=rec_sim_range,
                weights=False,
            )
        if init_position is None:
            init_position = self._init_position
        if init_price is None:
            init_price = self._init_price
        new_order_records = orders.get_short_view(
            init_position=init_position,
            init_price=init_price,
            jitted=jitted,
            chunked=chunked,
        ).values
        init_position = broadcast_array_to(init_position, self.wrapper.shape_2d[1])
        init_price = broadcast_array_to(init_price, self.wrapper.shape_2d[1])
        new_init_position = np.where(init_position < 0, init_position, 0)
        new_init_price = np.where(init_position < 0, init_price, np.nan)
        return self.replace(
            order_records=new_order_records,
            init_position=new_init_position,
            init_price=new_init_price,
            **kwargs,
        )

    # ############# Records ############# #

    @property
    def order_records(self) -> tp.RecordArray:
        """Structured NumPy array of order records.

        Returns:
            RecordArray: A structured array containing order records.
        """
        return self._order_records

    @hybrid_method
    def get_orders(
        cls_or_self,
        order_records: tp.Optional[tp.RecordArray] = None,
        open: tp.Optional[tp.SeriesFrame] = None,
        high: tp.Optional[tp.SeriesFrame] = None,
        low: tp.Optional[tp.SeriesFrame] = None,
        close: tp.Optional[tp.SeriesFrame] = None,
        orders_cls: tp.Optional[type] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        weights: tp.Union[None, bool, tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> Orders:
        """Get order records.

        Args:
            order_records (Optional[RecordArray]): Structured array of order records.

                Uses `Portfolio.order_records` if not provided.
            open (Optional[SeriesFrame]): Array of open prices.

                Uses `Portfolio.open_flex` if not provided.
            high (Optional[SeriesFrame]): Array of high prices.

                Uses `Portfolio.high_flex` if not provided.
            low (Optional[SeriesFrame]): Array of low prices.

                Uses `Portfolio.low_flex` if not provided.
            close (Optional[SeriesFrame]): Array of close prices.

                Uses `Portfolio.close_flex` if not provided.
            orders_cls (Optional[type]): Class to instantiate orders. 
            
                Uses `Portfolio.orders_cls` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start time.
            sim_end (Optional[ArrayLike]): Simulation end time.
            rec_sim_range (bool): Flag indicating whether to filter order records by simulation range.
            weights (Union[None, bool, ArrayLike]): Asset weights; if False, weights are ignored.

                Uses `Portfolio.get_weights` if not provided.
            jitted (JittedOption): Option to control JIT compilation.
            wrapper (Optional[ArrayWrapper]): Wrapper instance for processing records.
            group_by (GroupByLike): Grouping specification.
            **kwargs: Additional keyword arguments for orders instantiation.

        Returns:
            Orders: An instance of `vectorbtpro.portfolio.orders.Orders` with order records.
        """
        if not isinstance(cls_or_self, type):
            if order_records is None:
                order_records = cls_or_self.order_records
            if open is None:
                open = cls_or_self.open_flex
            if high is None:
                high = cls_or_self.high_flex
            if low is None:
                low = cls_or_self.low_flex
            if close is None:
                close = cls_or_self.close_flex
            if orders_cls is None:
                orders_cls = cls_or_self.orders_cls
            if weights is None:
                weights = cls_or_self.resolve_shortcut_attr("weights", wrapper=wrapper)
            elif weights is False:
                weights = None
            if wrapper is None:
                wrapper = fix_wrapper_for_records(cls_or_self)
        else:
            checks.assert_not_none(order_records, arg_name="order_records")
            if orders_cls is None:
                orders_cls = Orders
            checks.assert_not_none(wrapper, arg_name="wrapper")
            weights = cls_or_self.get_weights(weights=weights, wrapper=wrapper)
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

        if sim_start is not None or sim_end is not None:
            func = jit_reg.resolve_option(nb.records_within_sim_range_nb, jitted)
            order_records = func(
                wrapper.shape_2d,
                order_records,
                order_records["col"],
                order_records["idx"],
                sim_start=sim_start,
                sim_end=sim_end,
            )
        if weights is not None:
            func = jit_reg.resolve_option(nb.apply_weights_to_orders_nb, jitted)
            order_records = func(
                order_records,
                order_records["col"],
                to_1d_array(weights),
            )
        return orders_cls(
            wrapper,
            order_records,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        ).regroup(group_by)

    @property
    def log_records(self) -> tp.RecordArray:
        """Structured NumPy array of log records.

        Returns:
            RecordArray: A structured array containing log records.
        """
        return self._log_records

    @hybrid_method
    def get_logs(
        cls_or_self,
        log_records: tp.Optional[tp.RecordArray] = None,
        open: tp.Optional[tp.SeriesFrame] = None,
        high: tp.Optional[tp.SeriesFrame] = None,
        low: tp.Optional[tp.SeriesFrame] = None,
        close: tp.Optional[tp.SeriesFrame] = None,
        logs_cls: tp.Optional[type] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> Logs:
        """Return log records.

        Args:
            log_records (Optional[RecordArray]): Log records data. 
            
                Uses `Portfolio.log_records` if not provided.
            open (Optional[SeriesFrame]): Open price data. 
            
                Uses `Portfolio.open_flex` if not provided.
            high (Optional[SeriesFrame]): High price data.

                Uses `Portfolio.high_flex` if not provided.
            low (Optional[SeriesFrame]): Low price data.

                Uses `Portfolio.low_flex` if not provided.
            close (Optional[SeriesFrame]): Close price data.

                Uses `Portfolio.close_flex` if not provided.
            logs_cls (Optional[type]): Class to construct logs.

                Uses `Portfolio.logs_cls` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start time.
            sim_end (Optional[ArrayLike]): Simulation end time.
            rec_sim_range (bool): Flag to indicate whether to record simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            wrapper (Optional[ArrayWrapper]): Array wrapper for records.
            group_by (GroupByLike): Grouping specification.
            **kwargs: Additional keyword arguments for the logs constructor.

        Returns:
            Logs: An instance of `vectorbtpro.portfolio.logs.Logs` with log records.
        """
        if not isinstance(cls_or_self, type):
            if log_records is None:
                log_records = cls_or_self.log_records
            if open is None:
                open = cls_or_self.open_flex
            if high is None:
                high = cls_or_self.high_flex
            if low is None:
                low = cls_or_self.low_flex
            if close is None:
                close = cls_or_self.close_flex
            if logs_cls is None:
                logs_cls = cls_or_self.logs_cls
            if wrapper is None:
                wrapper = fix_wrapper_for_records(cls_or_self)
        else:
            checks.assert_not_none(log_records, arg_name="log_records")
            if logs_cls is None:
                logs_cls = Logs
            checks.assert_not_none(wrapper, arg_name="wrapper")
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

        if sim_start is not None or sim_end is not None:
            func = jit_reg.resolve_option(nb.records_within_sim_range_nb, jitted)
            log_records = func(
                wrapper.shape_2d,
                log_records,
                log_records["col"],
                log_records["idx"],
                sim_start=sim_start,
                sim_end=sim_end,
            )
        return logs_cls(
            wrapper,
            log_records,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        ).regroup(group_by)

    @hybrid_method
    def get_entry_trades(
        cls_or_self,
        orders: tp.Optional[Orders] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        entry_trades_cls: tp.Optional[type] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> EntryTrades:
        """Return entry trade records.

        Args:
            orders (Optional[Orders]): Order records.

                Uses `Portfolio.get_orders` if not provided.
            init_position (Optional[ArrayLike]): Initial position data.

                Uses `Portfolio.get_init_position` with `keep_flex=True` or 0 if not provided.
            init_price (Optional[ArrayLike]): Initial price data.

                Uses `Portfolio.get_init_price` with `keep_flex=True` or NaN if not provided.
            entry_trades_cls (Optional[type]): Class used to generate entry trades. 
            
                Uses `Portfolio.entry_trades_cls` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start time.
            sim_end (Optional[ArrayLike]): Simulation end time.
            rec_sim_range (bool): Flag indicating whether simulation range should be recorded.
            wrapper (Optional[ArrayWrapper]): Array wrapper for formatting.
            group_by (GroupByLike): Grouping specification.
            **kwargs: Additional keyword arguments for entry trades initialization.

        Returns:
            EntryTrades: An instance of `vectorbtpro.portfolio.trades.EntryTrades` with entry trade records.
        """
        if not isinstance(cls_or_self, type):
            if orders is None:
                orders = cls_or_self.resolve_shortcut_attr(
                    "orders",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if init_position is None:
                init_position = cls_or_self.resolve_shortcut_attr(
                    "init_position",
                    wrapper=wrapper,
                    keep_flex=True,
                )
            if init_price is None:
                init_price = cls_or_self.resolve_shortcut_attr(
                    "init_price",
                    wrapper=wrapper,
                    keep_flex=True,
                )
            if entry_trades_cls is None:
                entry_trades_cls = cls_or_self.entry_trades_cls
        else:
            checks.assert_not_none(orders, arg_name="orders")
            if init_position is None:
                init_position = 0.0
            if init_price is None:
                init_price = np.nan
            if entry_trades_cls is None:
                entry_trades_cls = EntryTrades
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, group_by=False)

        return entry_trades_cls.from_orders(
            orders,
            init_position=init_position,
            init_price=init_price,
            sim_start=sim_start,
            sim_end=sim_end,
            **kwargs,
        )

    @hybrid_method
    def get_exit_trades(
        cls_or_self,
        orders: tp.Optional[Orders] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        exit_trades_cls: tp.Optional[type] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> ExitTrades:
        """Return exit trade records.

        Args:
            orders (Optional[Orders]): Order records.

                Uses `Portfolio.get_orders` if not provided.
            init_position (Optional[ArrayLike]): Initial position data.

                Uses `Portfolio.get_init_position` with `keep_flex=True` or 0 if not provided.
            init_price (Optional[ArrayLike]): Initial price data.

                Uses `Portfolio.get_init_price` with `keep_flex=True` or NaN if not provided.
            exit_trades_cls (Optional[type]): Class used to generate exit trades. 
            
                Uses `Portfolio.exit_trades_cls` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start time.
            sim_end (Optional[ArrayLike]): Simulation end time.
            rec_sim_range (bool): Flag indicating whether simulation range should be recorded.
            wrapper (Optional[ArrayWrapper]): Array wrapper for formatting.
            group_by (GroupByLike): Grouping specification.
            **kwargs: Additional keyword arguments for exit trades initialization.

        Returns:
            ExitTrades: An instance of `vectorbtpro.portfolio.trades.ExitTrades` with exit trade records.
        """
        if not isinstance(cls_or_self, type):
            if orders is None:
                orders = cls_or_self.resolve_shortcut_attr(
                    "orders",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if init_position is None:
                init_position = cls_or_self.resolve_shortcut_attr(
                    "init_position",
                    wrapper=wrapper,
                    keep_flex=True,
                )
            if init_price is None:
                init_price = cls_or_self.resolve_shortcut_attr(
                    "init_price",
                    wrapper=wrapper,
                    keep_flex=True,
                )
            if exit_trades_cls is None:
                exit_trades_cls = cls_or_self.exit_trades_cls
        else:
            checks.assert_not_none(orders, arg_name="orders")
            if init_position is None:
                init_position = 0.0
            if init_price is None:
                init_price = np.nan
            if exit_trades_cls is None:
                exit_trades_cls = ExitTrades
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, group_by=False)

        return exit_trades_cls.from_orders(
            orders,
            init_position=init_position,
            init_price=init_price,
            sim_start=sim_start,
            sim_end=sim_end,
            **kwargs,
        )

    @hybrid_method
    def get_positions(
        cls_or_self,
        trades: tp.Optional[Trades] = None,
        positions_cls: tp.Optional[type] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> Positions:
        """Return position records.

        Args:
            trades (Optional[Trades]): Trade records used for constructing positions. 
            
                Uses `Portfolio.get_exit_trades` if not provided.
            positions_cls (Optional[type]): Class used to represent positions. 
            
                Uses `Portfolio.positions_cls` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start time.
            sim_end (Optional[ArrayLike]): Simulation end time.
            rec_sim_range (bool): Flag indicating whether simulation range should be recorded.
            wrapper (Optional[ArrayWrapper]): Array wrapper for formatting.
            group_by (GroupByLike): Grouping specification.
            **kwargs: Additional keyword arguments for constructing positions.

        Returns:
            Positions: An instance of `vectorbtpro.portfolio.trades.Positions` with position records.
        """
        if not isinstance(cls_or_self, type):
            if trades is None:
                trades = cls_or_self.resolve_shortcut_attr(
                    "exit_trades",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if positions_cls is None:
                positions_cls = cls_or_self.positions_cls
        else:
            checks.assert_not_none(trades, arg_name="trades")
            if positions_cls is None:
                positions_cls = Positions

        return positions_cls.from_trades(trades, **kwargs)

    def get_trades(
        self,
        trades_type: tp.Optional[tp.Union[str, int]] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> Trades:
        """Return trade or position records based on the portfolio's trades type.

        Args:
            trades_type (Optional[Union[str, int]]): The type of trades to retrieve.

                Uses `Portfolio.trades_type` if not provided.
            sim_start (Optional[ArrayLike]): The simulation start index.
            sim_end (Optional[ArrayLike]): The simulation end index.
            rec_sim_range (bool): Flag indicating whether to record the simulation range.
            wrapper (Optional[ArrayWrapper]): Optional wrapper instance for record transformation.
            group_by (GroupByLike): Grouping specification.
            **kwargs: Additional keyword arguments passed to the resolve shortcut attribute.

        Returns:
            Trades: Trade or position records corresponding to the specified trades type.
        """
        if trades_type is None:
            trades_type = self.trades_type
        else:
            if isinstance(trades_type, str):
                trades_type = map_enum_fields(trades_type, enums.TradesType)
        if trades_type == enums.TradesType.EntryTrades:
            return self.resolve_shortcut_attr(
                "entry_trades",
                sim_start=sim_start,
                sim_end=sim_end,
                rec_sim_range=rec_sim_range,
                wrapper=wrapper,
                group_by=group_by,
                **kwargs,
            )
        if trades_type == enums.TradesType.ExitTrades:
            return self.resolve_shortcut_attr(
                "exit_trades",
                sim_start=sim_start,
                sim_end=sim_end,
                rec_sim_range=rec_sim_range,
                wrapper=wrapper,
                group_by=group_by,
                **kwargs,
            )
        if trades_type == enums.TradesType.Positions:
            return self.resolve_shortcut_attr(
                "positions",
                sim_start=sim_start,
                sim_end=sim_end,
                rec_sim_range=rec_sim_range,
                wrapper=wrapper,
                group_by=group_by,
                **kwargs,
            )
        raise NotImplementedError

    @hybrid_method
    def get_trade_history(
        cls_or_self,
        orders: tp.Optional[Orders] = None,
        entry_trades: tp.Optional[EntryTrades] = None,
        exit_trades: tp.Optional[ExitTrades] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
    ) -> tp.Frame:
        """Return a readable DataFrame merging order history with entry and exit trade records.

        Args:
            orders (Optional[Orders]): Orders instance. 
            
                Uses `Portfolio.get_orders` if not provided.
            entry_trades (Optional[EntryTrades]): Entry trade records. 
            
                Uses `Portfolio.get_entry_trades` if not provided.
            exit_trades (Optional[ExitTrades]): Exit trade records. 
            
                Uses `Portfolio.get_exit_trades` if not provided.
            sim_start (Optional[ArrayLike]): The simulation start index.
            sim_end (Optional[ArrayLike]): The simulation end index.
            rec_sim_range (bool): Flag indicating whether to record the simulation range.
            wrapper (Optional[ArrayWrapper]): Optional wrapper instance for record transformation.
            group_by (GroupByLike): Grouping specification.

        Returns:
            DataFrame: A readable DataFrame merging order history with entry and exit trade records.

        !!! note
            The P&L and return aggregated across the DataFrame may not match the actual total 
            P&L and return, as the DataFrame annotates entry and exit orders with performance 
            relative to their respective trade types. For accurate total statistics, aggregate 
            only statistics of a single trade type. Additionally, entry orders include open 
            statistics, whereas exit orders do not.
        """
        if not isinstance(cls_or_self, type):
            if orders is None:
                orders = cls_or_self.resolve_shortcut_attr(
                    "orders",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if entry_trades is None:
                entry_trades = cls_or_self.resolve_shortcut_attr(
                    "entry_trades",
                    orders=orders,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if exit_trades is None:
                exit_trades = cls_or_self.resolve_shortcut_attr(
                    "exit_trades",
                    orders=orders,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
        else:
            checks.assert_not_none(orders, arg_name="orders")
            checks.assert_not_none(entry_trades, arg_name="entry_trades")
            checks.assert_not_none(exit_trades, arg_name="exit_trades")

        order_history = orders.records_readable
        del order_history["Size"]
        del order_history["Price"]
        del order_history["Fees"]

        entry_trade_history = entry_trades.records_readable
        del entry_trade_history["Entry Index"]
        del entry_trade_history["Exit Order Id"]
        del entry_trade_history["Exit Index"]
        del entry_trade_history["Avg Exit Price"]
        del entry_trade_history["Exit Fees"]
        entry_trade_history.rename(columns={"Entry Order Id": "Order Id"}, inplace=True)
        entry_trade_history.rename(columns={"Avg Entry Price": "Price"}, inplace=True)
        entry_trade_history.rename(columns={"Entry Fees": "Fees"}, inplace=True)

        exit_trade_history = exit_trades.records_readable
        del exit_trade_history["Exit Index"]
        del exit_trade_history["Entry Order Id"]
        del exit_trade_history["Entry Index"]
        del exit_trade_history["Avg Entry Price"]
        del exit_trade_history["Entry Fees"]
        exit_trade_history.rename(columns={"Exit Order Id": "Order Id"}, inplace=True)
        exit_trade_history.rename(columns={"Avg Exit Price": "Price"}, inplace=True)
        exit_trade_history.rename(columns={"Exit Fees": "Fees"}, inplace=True)

        trade_history = pd.concat((entry_trade_history, exit_trade_history), axis=0)
        trade_history = pd.merge(order_history, trade_history, on=["Column", "Order Id"])
        trade_history = trade_history.sort_values(by=["Column", "Order Id", "Position Id"])
        trade_history = trade_history.reset_index(drop=True)
        trade_history["Entry Trade Id"] = trade_history["Entry Trade Id"].fillna(-1).astype(int)
        trade_history["Exit Trade Id"] = trade_history["Exit Trade Id"].fillna(-1).astype(int)
        trade_history["Entry Trade Id"] = trade_history.pop("Entry Trade Id")
        trade_history["Exit Trade Id"] = trade_history.pop("Exit Trade Id")
        trade_history["Position Id"] = trade_history.pop("Position Id")
        return trade_history

    @hybrid_method
    def get_signals(
        cls_or_self,
        orders: tp.Optional[Orders] = None,
        entry_trades: tp.Optional[EntryTrades] = None,
        exit_trades: tp.Optional[ExitTrades] = None,
        idx_arr: tp.Union[None, str, tp.Array1d] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
    ) -> tp.Tuple[tp.SeriesFrame, tp.SeriesFrame, tp.SeriesFrame, tp.SeriesFrame]:
        """Return signals for long and short entries and exits.

        Compute boolean signals indicating the occurrence of long entry orders, long exit orders, 
        short entry orders, and short exit orders. Signals are computed per group if grouping is enabled. 
        Pass `group_by=False` to disable grouping.

        Args:
            orders (Optional[Orders]): Order records used to determine signals. 

                Uses `Portfolio.get_orders` if not provided.
            entry_trades (Optional[EntryTrades]): Entry trade records used to determine signals. 

                Uses `Portfolio.get_entry_trades` if not provided.
            exit_trades (Optional[ExitTrades]): Exit trade records used to determine signals. 

                Uses `Portfolio.get_exit_trades` if not provided.
            idx_arr (Union[None, str, Array1d]): Key or index array used to map order indices. 

                If a string, it is used with `vectorbtpro.records.base.Records.map_field`; 
                otherwise, with `vectorbtpro.records.base.Records.map_array`.
            sim_start (Optional[ArrayLike]): Simulation start dates passed to shortcut resolution.
            sim_end (Optional[ArrayLike]): Simulation end dates passed to shortcut resolution.
            rec_sim_range (bool): Whether to recalculate the simulation range.
            wrapper (Optional[ArrayWrapper]): Array wrapper for formatting output.
            group_by (GroupByLike): Grouping parameter; set to False to disable grouping.

        Returns:
            Tuple[SeriesFrame, SeriesFrame, SeriesFrame, SeriesFrame]: 
                Signals for long entries, long exits, short entries, and short exits.
        """
        if not isinstance(cls_or_self, type):
            if orders is None:
                orders = cls_or_self.resolve_shortcut_attr(
                    "orders",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if entry_trades is None:
                entry_trades = cls_or_self.resolve_shortcut_attr(
                    "entry_trades",
                    orders=orders,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if exit_trades is None:
                exit_trades = cls_or_self.resolve_shortcut_attr(
                    "exit_trades",
                    orders=orders,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
        else:
            checks.assert_not_none(orders, arg_name="orders")
            checks.assert_not_none(entry_trades, arg_name="entry_trades")
            checks.assert_not_none(exit_trades, arg_name="exit_trades")

        if isinstance(orders, FSOrders) and idx_arr is None:
            idx_arr = "signal_idx"
        if idx_arr is not None:
            if isinstance(idx_arr, str):
                idx_ma = orders.map_field(idx_arr, idx_arr=idx_arr)
            else:
                idx_ma = orders.map_array(idx_arr, idx_arr=idx_arr)
        else:
            idx_ma = orders.idx
        order_index = pd.MultiIndex.from_arrays(
            (orders.col_mapper.get_col_arr(group_by=group_by), orders.id_arr), names=["col", "id"]
        )
        order_idx_sr = pd.Series(idx_ma.values, index=order_index, name="idx")

        def _get_type_signals(type_order_ids):
            type_order_ids = type_order_ids.apply_mask(type_order_ids.values != -1)
            type_order_index = pd.MultiIndex.from_arrays(
                (type_order_ids.col_mapper.get_col_arr(group_by=group_by), type_order_ids.values), names=["col", "id"]
            )
            type_idx_df = order_idx_sr.loc[type_order_index].reset_index()
            type_signals = orders.wrapper.fill(False, group_by=group_by)
            if isinstance(type_signals, pd.Series):
                type_signals.values[type_idx_df["idx"].values] = True
            else:
                type_signals.values[type_idx_df["idx"].values, type_idx_df["col"].values] = True
            return type_signals

        return (
            _get_type_signals(entry_trades.long_view.entry_order_id),
            _get_type_signals(exit_trades.long_view.exit_order_id),
            _get_type_signals(entry_trades.short_view.entry_order_id),
            _get_type_signals(exit_trades.short_view.exit_order_id),
        )

    @hybrid_method
    def get_drawdowns(
        cls_or_self,
        value: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        drawdowns_cls: tp.Optional[type] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> Drawdowns:
        """Return drawdown records computed from portfolio value.

        The drawdown records are calculated using the provided price series through the 
        `vectorbtpro.generic.drawdowns.Drawdowns.from_price` method of the designated drawdowns class.

        Args:
            value (Optional[SeriesFrame]): Price series used to compute drawdowns. 

                Uses `Portfolio.get_value` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start dates passed to shortcut resolution.
            sim_end (Optional[ArrayLike]): Simulation end dates passed to shortcut resolution.
            rec_sim_range (bool): Whether to recalculate the simulation range.
            drawdowns_cls (Optional[type]): Class for creating drawdown records. 

                Uses `Portfolio.drawdowns_cls` if not provided.
            wrapper (Optional[ArrayWrapper]): Array wrapper for formatting output. 
            group_by (GroupByLike): Grouping specification for the output.
            **kwargs: Additional keyword arguments passed to the 
                `vectorbtpro.generic.drawdowns.Drawdowns.from_price` method.

        Returns:
            Drawdowns: Drawdown records computed from the provided price series.
        """
        if not isinstance(cls_or_self, type):
            if value is None:
                value = cls_or_self.resolve_shortcut_attr(
                    "value",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if drawdowns_cls is None:
                drawdowns_cls = cls_or_self.drawdowns_cls
            if wrapper is None:
                wrapper = fix_wrapper_for_records(cls_or_self)
        else:
            checks.assert_not_none(value, arg_name="value")
            if drawdowns_cls is None:
                drawdowns_cls = Drawdowns
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, group_by=False)

        if wrapper is not None:
            wrapper = wrapper.resolve(group_by=group_by)
        return drawdowns_cls.from_price(
            value,
            sim_start=sim_start,
            sim_end=sim_end,
            wrapper=wrapper,
            **kwargs,
        )

    # ############# Assets ############# #

    @hybrid_method
    def get_init_position(
        cls_or_self,
        init_position_raw: tp.Optional[tp.ArrayLike] = None,
        weights: tp.Union[None, bool, tp.ArrayLike] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
        keep_flex: bool = False,
    ) -> tp.Union[tp.ArrayLike, tp.MaybeSeries]:
        """Return initial position per column.

        Compute and return the initial position for each column. The initial position is 
        optionally weighted and broadcasted to a 2D shape based on the provided array wrapper.

        Args:
            init_position_raw (Optional[ArrayLike]): Raw initial position values. 

                Uses `Portfolio._init_position` if not provided.
            weights (Union[None, bool, ArrayLike]): Weights applied to the initial positions. 

                Uses `Portfolio.get_weights` if not provided.
            
                If False, weights are ignored.
            wrapper (Optional[ArrayWrapper]): Array wrapper used to shape and format the output. 

                Uses `Portfolio.wrapper` if not provided.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping the result.
            keep_flex (bool): If True, returns the broadcasted array without wrapping when 
                weights are not applied.

        Returns:
            Union[ArrayLike, MaybeSeries]: The computed initial position per column, potentially wrapped.
        """
        if not isinstance(cls_or_self, type):
            if init_position_raw is None:
                init_position_raw = cls_or_self._init_position
            if weights is None:
                weights = cls_or_self.resolve_shortcut_attr("weights", wrapper=wrapper)
            elif weights is False:
                weights = None
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_position_raw, arg_name="init_position_raw")
            checks.assert_not_none(wrapper, arg_name="wrapper")
            weights = cls_or_self.get_weights(weights=weights, wrapper=wrapper)

        if keep_flex and weights is None:
            return init_position_raw
        init_position = broadcast_array_to(init_position_raw, wrapper.shape_2d[1])
        if weights is not None:
            weights = to_1d_array(weights)
            init_position = np.where(np.isnan(weights), init_position, weights * init_position)
        if keep_flex:
            return init_position
        wrap_kwargs = merge_dicts(dict(name_or_index="init_position"), wrap_kwargs)
        return wrapper.wrap_reduced(init_position, group_by=False, **wrap_kwargs)

    @hybrid_method
    def get_asset_flow(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        orders: tp.Optional[Orders] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get asset flow series per column.

        Returns the total transacted amount of assets at each time step.

        Args:
            direction (Union[str, int]): Direction for filtering asset flows.

                See `vectorbtpro.portfolio.enums.Direction` for options.
            orders (Optional[Orders]): Orders instance containing order data. 
            
                Uses `Portfolio.get_orders` if not provided.
            init_position (Optional[ArrayLike]): Initial position values used at the start of the simulation.

                Uses `Portfolio.get_init_position` with `keep_flex=True` or 0 if not provided.
            sim_start (Optional[ArrayLike]): Start time of the simulation.
            sim_end (Optional[ArrayLike]): End time of the simulation.
            rec_sim_range (bool): Flag indicating whether to record and use the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Wrapper instance used for formatting the output array.

                Uses `Portfolio.wrapper` if not provided.
            wrap_kwargs (KwargsLike): Additional keyword arguments for the wrapper.

        Returns:
            SeriesFrame: Asset flow series representing the total transacted amount of assets.
        """
        if not isinstance(cls_or_self, type):
            if orders is None:
                orders = cls_or_self.resolve_shortcut_attr(
                    "orders",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=None,
                )
            if init_position is None:
                init_position = cls_or_self.resolve_shortcut_attr(
                    "init_position",
                    wrapper=wrapper,
                    keep_flex=True,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(orders, arg_name="orders")
            if init_position is None:
                init_position = 0.0
            if wrapper is None:
                wrapper = orders.wrapper
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

        direction = map_enum_fields(direction, enums.Direction)
        func = jit_reg.resolve_option(nb.asset_flow_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        asset_flow = func(
            wrapper.shape_2d,
            orders.values,
            orders.col_mapper.col_map,
            direction=direction,
            init_position=to_1d_array(init_position),
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return wrapper.wrap(asset_flow, group_by=False, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_assets(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        asset_flow: tp.Optional[tp.SeriesFrame] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get asset series per column.

        Returns the position (asset series) at each time step.

        Args:
            direction (Union[str, int]): Direction for filtering asset data.
        
                See `vectorbtpro.portfolio.enums.Direction` for options.
            asset_flow (Optional[SeriesFrame]): Asset flow series.

                Uses `Portfolio.get_asset_flow` with `direction="both"` if not provided.
            init_position (Optional[ArrayLike]): Initial position values used at the start of the simulation.

                Uses `Portfolio.get_init_position` with `keep_flex=True` or 0 if not provided.
            sim_start (Optional[ArrayLike]): Start time of the simulation.
            sim_end (Optional[ArrayLike]): End time of the simulation.
            rec_sim_range (bool): Flag indicating whether to record and use the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Wrapper instance used for formatting the output array.

                Uses `Portfolio.wrapper` if not provided.
            wrap_kwargs (KwargsLike): Additional keyword arguments for the wrapper.

        Returns:
            SeriesFrame: Asset series representing the position at each time step.
        """
        if not isinstance(cls_or_self, type):
            if asset_flow is None:
                asset_flow = cls_or_self.resolve_shortcut_attr(
                    "asset_flow",
                    direction="both",
                    init_position=init_position,
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                )
            if init_position is None:
                init_position = cls_or_self.resolve_shortcut_attr(
                    "init_position",
                    wrapper=wrapper,
                    keep_flex=True,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(asset_flow, arg_name="asset_flow")
            if init_position is None:
                init_position = 0.0
            checks.assert_not_none(wrapper, arg_name="wrapper")
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

        direction = map_enum_fields(direction, enums.Direction)
        func = jit_reg.resolve_option(nb.assets_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        assets = func(
            to_2d_array(asset_flow),
            direction=direction,
            init_position=to_1d_array(init_position),
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return wrapper.wrap(assets, group_by=False, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_position_mask(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        assets: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get position mask per column or group.

        Creates a boolean mask where each element is True if a position exists at the corresponding time step.

        Args:
            direction (Union[str, int]): Direction for filtering asset positions.
        
                See `vectorbtpro.portfolio.enums.Direction` for options.
            assets (Optional[SeriesFrame]): Asset series data to generate the mask. 
            
                Uses `Portfolio.get_assets` if not provided.
            sim_start (Optional[ArrayLike]): Start time of the simulation.
            sim_end (Optional[ArrayLike]): End time of the simulation.
            rec_sim_range (bool): Flag indicating whether to record and use the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Wrapper instance used for formatting the output array.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Grouping specification to apply when creating the mask.
            wrap_kwargs (KwargsLike): Additional keyword arguments for the wrapper.

        Returns:
            SeriesFrame: Boolean mask indicating the presence of a position at each time step.
        """
        if not isinstance(cls_or_self, type):
            if assets is None:
                assets = cls_or_self.resolve_shortcut_attr(
                    "assets",
                    direction=direction,
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(assets, arg_name="assets")
            checks.assert_not_none(wrapper, arg_name="wrapper")
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.position_mask_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            position_mask = func(
                to_2d_array(assets),
                group_lens=group_lens,
                sim_start=sim_start,
                sim_end=sim_end,
            )
        else:
            func = jit_reg.resolve_option(nb.position_mask_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            position_mask = func(
                to_2d_array(assets),
                sim_start=sim_start,
                sim_end=sim_end,
            )
        return wrapper.wrap(position_mask, group_by=group_by, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_position_coverage(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        assets: tp.Optional[tp.SeriesFrame] = None,
        granular_groups: bool = False,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get position coverage per column or group.

        Calculates the proportion of time steps with a held position relative to the total number
        of time steps.

        Args:
            direction (Union[str, int]): Direction identifier for the position. 
            
                Use "both" to analyze both long and short.

                See `vectorbtpro.portfolio.enums.Direction` for options.
            assets (Optional[SeriesFrame]): Asset data used to determine positions.

                Uses `Portfolio.get_assets` if not provided.
            granular_groups (bool): Flag to determine if coverage is computed per individual column within a group.
            sim_start (Optional[ArrayLike]): Simulation start index.
            sim_end (Optional[ArrayLike]): Simulation end index.
            rec_sim_range (bool): Flag to record the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked execution.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance used for post-processing.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Specification for grouping data.
            wrap_kwargs (KwargsLike): Additional keyword arguments for the wrapping process.

        Returns:
            MaybeSeries: Series representing the computed position coverage.
        """
        if not isinstance(cls_or_self, type):
            if assets is None:
                assets = cls_or_self.resolve_shortcut_attr(
                    "assets",
                    direction=direction,
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(assets, arg_name="assets")
            checks.assert_not_none(wrapper, arg_name="wrapper")
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.position_coverage_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            position_coverage = func(
                to_2d_array(assets),
                group_lens=group_lens,
                granular_groups=granular_groups,
                sim_start=sim_start,
                sim_end=sim_end,
            )
        else:
            func = jit_reg.resolve_option(nb.position_coverage_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            position_coverage = func(
                to_2d_array(assets),
                sim_start=sim_start,
                sim_end=sim_end,
            )
        wrap_kwargs = merge_dicts(dict(name_or_index="position_coverage"), wrap_kwargs)
        return wrapper.wrap_reduced(position_coverage, group_by=group_by, **wrap_kwargs)

    @hybrid_method
    def get_position_entry_price(
        cls_or_self,
        orders: tp.Optional[Orders] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        fill_closed_position: bool = False,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get the position's entry price at each time step.

        Calculates the entry price applied to each time step based on provided orders and initial values.

        Args:
            orders (Optional[Orders]): Order data used for computing entry prices.

                Uses `Portfolio.get_orders` if not provided.
            init_position (Optional[ArrayLike]): Initial position prior to processing orders.

                Uses `Portfolio.get_init_position` with `keep_flex=True` or 0 if not provided.
            init_price (Optional[ArrayLike]): Initial price prior to processing orders.

                Uses `Portfolio.get_init_price` with `keep_flex=True` or NaN if not provided.
            fill_closed_position (bool): If True, forward-fill missing values using
                prices from a previously closed position.
            sim_start (Optional[ArrayLike]): Simulation start index.
            sim_end (Optional[ArrayLike]): Simulation end index.
            rec_sim_range (bool): Flag to record the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked execution.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance used for processing.

                Uses `Portfolio.wrapper` if not provided.
            wrap_kwargs (KwargsLike): Additional keyword arguments for the wrapping procedure.

        Returns:
            SeriesFrame: A frame containing the computed entry prices per time step.
        """
        if not isinstance(cls_or_self, type):
            if orders is None:
                if orders is None:
                    orders = cls_or_self.resolve_shortcut_attr(
                        "orders",
                        sim_start=sim_start if rec_sim_range else None,
                        sim_end=sim_end if rec_sim_range else None,
                        rec_sim_range=rec_sim_range,
                        wrapper=wrapper,
                        group_by=None,
                    )
            if init_position is None:
                init_position = cls_or_self.resolve_shortcut_attr(
                    "init_position",
                    wrapper=wrapper,
                    keep_flex=True,
                )
            if init_price is None:
                init_price = cls_or_self.resolve_shortcut_attr(
                    "init_price",
                    wrapper=wrapper,
                    keep_flex=True,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(orders, arg_name="orders")
            if init_position is None:
                init_position = 0.0
            if init_price is None:
                init_price = np.nan
            if wrapper is None:
                wrapper = orders.wrapper
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

        func = jit_reg.resolve_option(nb.get_position_feature_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        entry_price = func(
            orders.values,
            to_2d_array(orders.close),
            orders.col_mapper.col_map,
            feature=enums.PositionFeature.EntryPrice,
            init_position=to_1d_array(init_position),
            init_price=to_1d_array(init_price),
            fill_closed_position=fill_closed_position,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return wrapper.wrap(entry_price, group_by=False, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_position_exit_price(
        cls_or_self,
        orders: tp.Optional[Orders] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        fill_closed_position: bool = False,
        fill_exit_price: bool = True,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get the position's exit price at each time step.

        Determines the exit price for each time step based on order data and initial conditions.

        Args:
            orders (Optional[Orders]): Order data used for computing entry prices.

                Uses `Portfolio.get_orders` if not provided.
            init_position (Optional[ArrayLike]): Initial position prior to processing orders.

                Uses `Portfolio.get_init_position` with `keep_flex=True` or 0 if not provided.
            init_price (Optional[ArrayLike]): Initial price prior to processing orders.

                Uses `Portfolio.get_init_price` with `keep_flex=True` or NaN if not provided.
            fill_closed_position (bool): If True, forward-fill missing values using
                prices from a previously closed position.
            fill_exit_price (bool): If True, fill exit prices for open positions using
                the current close price.
            sim_start (Optional[ArrayLike]): Simulation start index.
            sim_end (Optional[ArrayLike]): Simulation end index.
            rec_sim_range (bool): Flag to record the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked execution.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance used for processing.

                Uses `Portfolio.wrapper` if not provided.
            wrap_kwargs (KwargsLike): Additional keyword arguments for the wrapping process.

        Returns:
            SeriesFrame: A frame containing the computed exit prices per time step.
        """
        if not isinstance(cls_or_self, type):
            if orders is None:
                if orders is None:
                    orders = cls_or_self.resolve_shortcut_attr(
                        "orders",
                        sim_start=sim_start if rec_sim_range else None,
                        sim_end=sim_end if rec_sim_range else None,
                        rec_sim_range=rec_sim_range,
                        wrapper=wrapper,
                        group_by=None,
                    )
            if init_position is None:
                init_position = cls_or_self.resolve_shortcut_attr(
                    "init_position",
                    wrapper=wrapper,
                    keep_flex=True,
                )
            if init_price is None:
                init_price = cls_or_self.resolve_shortcut_attr(
                    "init_price",
                    wrapper=wrapper,
                    keep_flex=True,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(orders, arg_name="orders")
            if init_position is None:
                init_position = 0.0
            if init_price is None:
                init_price = np.nan
            if wrapper is None:
                wrapper = orders.wrapper
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

        func = jit_reg.resolve_option(nb.get_position_feature_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        exit_price = func(
            orders.values,
            to_2d_array(orders.close),
            orders.col_mapper.col_map,
            feature=enums.PositionFeature.ExitPrice,
            init_position=to_1d_array(init_position),
            init_price=to_1d_array(init_price),
            fill_closed_position=fill_closed_position,
            fill_exit_price=fill_exit_price,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return wrapper.wrap(exit_price, group_by=False, **resolve_dict(wrap_kwargs))

    # ############# Cash ############# #

    @hybrid_method
    def get_cash_deposits(
        cls_or_self,
        cash_deposits_raw: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = None,
        split_shared: bool = False,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        weights: tp.Union[None, bool, tp.ArrayLike] = None,
        keep_flex: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Union[tp.ArrayLike, tp.MaybeSeries]:
        """Get cash deposit series per column or group.

        Calculates cash deposits per column or group, applying grouping and simulation period adjustments.
        Set `keep_flex` to True to preserve a format suitable for flexible indexing, which consumes less memory.

        Args:
            cash_deposits_raw (Optional[ArrayLike]): Raw cash deposit values.

                Uses `Portfolio._cash_deposits` or 0 if not provided.
            cash_sharing (Optional[bool]): Flag indicating whether cash sharing is enabled.

                Uses `Portfolio.cash_sharing` if not provided.
            split_shared (bool): If cash is shared, determines whether to split the deposits evenly across columns.
            sim_start (Optional[ArrayLike]): Start index or time for the simulation.
            sim_end (Optional[ArrayLike]): End index or time for the simulation.
            rec_sim_range (bool): Flag indicating whether to record the simulation range.
            weights (Union[None, bool, ArrayLike]): Weights for scaling cash deposits.

                Uses `Portfolio.get_weights` if not provided.
            
                If False, weights are ignored.
            keep_flex (bool): If True, returns the output in a flexible format.
            jitted (JittedOption): Option to control just-in-time compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Wrapper instance for formatting the output.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Specification for grouping.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.

        Returns:
            Union[ArrayLike, MaybeSeries]: Cash deposit series, either as a raw array or a wrapped series.
        """
        if not isinstance(cls_or_self, type):
            if cash_deposits_raw is None:
                cash_deposits_raw = cls_or_self._cash_deposits
            if cash_sharing is None:
                cash_sharing = cls_or_self.cash_sharing
            if weights is None:
                weights = cls_or_self.resolve_shortcut_attr("weights", wrapper=wrapper)
            elif weights is False:
                weights = None
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            if cash_deposits_raw is None:
                cash_deposits_raw = 0.0
            checks.assert_not_none(cash_sharing, arg_name="cash_sharing")
            checks.assert_not_none(wrapper, arg_name="wrapper")
            weights = cls_or_self.get_weights(weights=weights, wrapper=wrapper)
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

        cash_deposits_arr = to_2d_array(cash_deposits_raw)
        if keep_flex and not cash_deposits_arr.any():
            return cash_deposits_raw

        if wrapper.grouper.is_grouped(group_by=group_by):
            if keep_flex and cash_sharing and weights is None and sim_start is None and sim_end is None:
                return cash_deposits_raw
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.cash_deposits_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash_deposits = func(
                wrapper.shape_2d,
                group_lens,
                cash_sharing,
                cash_deposits_raw=cash_deposits_arr,
                weights=to_1d_array(weights) if weights is not None else None,
                sim_start=sim_start,
                sim_end=sim_end,
            )
        else:
            if keep_flex and not cash_sharing and weights is None and sim_start is None and sim_end is None:
                return cash_deposits_raw
            group_lens = wrapper.grouper.get_group_lens()
            func = jit_reg.resolve_option(nb.cash_deposits_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash_deposits = func(
                wrapper.shape_2d,
                group_lens,
                cash_sharing,
                cash_deposits_raw=cash_deposits_arr,
                split_shared=split_shared,
                weights=to_1d_array(weights) if weights is not None else None,
                sim_start=sim_start,
                sim_end=sim_end,
            )
        if keep_flex:
            return cash_deposits
        return wrapper.wrap(cash_deposits, group_by=group_by, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_total_cash_deposits(
        cls_or_self,
        cash_deposits_raw: tp.Optional[tp.ArrayLike] = None,
        cash_sharing: tp.Optional[bool] = None,
        split_shared: bool = False,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        weights: tp.Union[None, bool, tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.ArrayLike:
        """Get total cash deposit series per column or group.

        Aggregates cash deposit data across columns or groups by summing the individual cash deposits.

        Args:
            cash_deposits_raw (Optional[ArrayLike]): Raw cash deposit values.

                Uses `Portfolio._cash_deposits` if not provided.
            cash_sharing (Optional[bool]): Flag indicating whether cash sharing is enabled.

                Uses `Portfolio.cash_sharing` if not provided.
            split_shared (bool): If cash is shared, determines whether to split the deposits evenly across columns.
            sim_start (Optional[ArrayLike]): Start index or time for the simulation.
            sim_end (Optional[ArrayLike]): End index or time for the simulation.
            rec_sim_range (bool): Flag indicating whether to record the simulation range.
            weights (Union[None, bool, ArrayLike]): Weights for scaling cash deposits.

                Uses `Portfolio.get_weights` if not provided.
            
                If False, weights are ignored.
            jitted (JittedOption): Option to control just-in-time compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Wrapper instance for formatting the output.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Specification for grouping.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.

        Returns:
            ArrayLike: Total cash deposit series aggregated over columns.
        """
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")
        cash_deposits = cls_or_self.get_cash_deposits(
            cash_deposits_raw=cash_deposits_raw,
            cash_sharing=cash_sharing,
            split_shared=split_shared,
            sim_start=sim_start,
            sim_end=sim_end,
            rec_sim_range=rec_sim_range,
            weights=weights,
            keep_flex=True,
            jitted=jitted,
            chunked=chunked,
            wrapper=wrapper,
            group_by=group_by,
            wrap_kwargs=wrap_kwargs,
        )
        total_cash_deposits = np.nansum(cash_deposits, axis=0)
        return wrapper.wrap_reduced(total_cash_deposits, group_by=group_by, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_cash_earnings(
        cls_or_self,
        cash_earnings_raw: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        weights: tp.Union[None, bool, tp.ArrayLike] = None,
        keep_flex: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Union[tp.ArrayLike, tp.MaybeSeries]:
        """Get cash earnings series per column or group.

        Calculates cash earnings based on provided raw earnings data, with grouping and simulation period adjustments.
        Set `keep_flex` to True to preserve a format suitable for flexible indexing, which consumes less memory.

        Args:
            cash_earnings_raw (Optional[ArrayLike]): Raw cash earnings data.

                Uses `Portfolio._cash_earnings` or 0 if not provided.
            sim_start (Optional[ArrayLike]): Start index or time for the simulation.
            sim_end (Optional[ArrayLike]): End index or time for the simulation.
            rec_sim_range (bool): Flag indicating whether to record the simulation range.
            weights (Union[None, bool, ArrayLike]): Weights for scaling cash earnings.

                Uses `Portfolio.get_weights` if not provided.
            
                If False, weights are ignored.
            keep_flex (bool): If True, returns the output in a flexible format.
            jitted (JittedOption): Option to control just-in-time compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Wrapper instance for formatting the output.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Specification for grouping.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.

        Returns:
            Union[ArrayLike, MaybeSeries]: Cash earnings series, either as a raw array or a wrapped series.
        """
        if not isinstance(cls_or_self, type):
            if cash_earnings_raw is None:
                cash_earnings_raw = cls_or_self._cash_earnings
            if weights is None:
                weights = cls_or_self.resolve_shortcut_attr("weights", wrapper=wrapper)
            elif weights is False:
                weights = None
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            if cash_earnings_raw is None:
                cash_earnings_raw = 0.0
            checks.assert_not_none(wrapper, arg_name="wrapper")
            weights = cls_or_self.get_weights(weights=weights, wrapper=wrapper)
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

        cash_earnings_arr = to_2d_array(cash_earnings_raw)
        if keep_flex and not cash_earnings_arr.any():
            return cash_earnings_raw

        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.cash_earnings_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash_earnings = func(
                wrapper.shape_2d,
                group_lens,
                cash_earnings_raw=cash_earnings_arr,
                weights=to_1d_array(weights) if weights is not None else None,
                sim_start=sim_start,
                sim_end=sim_end,
            )
        else:
            if keep_flex and weights is None and sim_start is None and sim_end is None:
                return cash_earnings_raw
            func = jit_reg.resolve_option(nb.cash_earnings_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash_earnings = func(
                wrapper.shape_2d,
                cash_earnings_raw=cash_earnings_arr,
                weights=to_1d_array(weights) if weights is not None else None,
                sim_start=sim_start,
                sim_end=sim_end,
            )
        if keep_flex:
            return cash_earnings
        return wrapper.wrap(cash_earnings, group_by=group_by, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_total_cash_earnings(
        cls_or_self,
        cash_earnings_raw: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        weights: tp.Union[None, bool, tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.ArrayLike:
        """Get total cash earnings series aggregated per column or group.
    
        Args:
            cash_earnings_raw (Optional[ArrayLike]): Raw cash earnings data.

                Uses `Portfolio._cash_earnings` if not provided.
            sim_start (Optional[ArrayLike]): Start index or time for the simulation.
            sim_end (Optional[ArrayLike]): End index or time for the simulation.
            rec_sim_range (bool): Flag indicating whether to record the simulation range.
            weights (Union[None, bool, ArrayLike]): Weights for scaling cash earnings.

                Uses `Portfolio.get_weights` if not provided.
            
                If False, weights are ignored.
            jitted (JittedOption): Option to control just-in-time compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Wrapper instance for formatting the output.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Specification for grouping.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.
        
        Returns:
            ArrayLike: Wrapped total cash earnings aggregated per column or group.
        """
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")
        cash_earnings = cls_or_self.get_cash_earnings(
            cash_earnings_raw=cash_earnings_raw,
            sim_start=sim_start,
            sim_end=sim_end,
            rec_sim_range=rec_sim_range,
            weights=weights,
            keep_flex=True,
            jitted=jitted,
            chunked=chunked,
            wrapper=wrapper,
            group_by=group_by,
            wrap_kwargs=wrap_kwargs,
        )
        total_cash_earnings = np.nansum(cash_earnings, axis=0)
        return wrapper.wrap_reduced(total_cash_earnings, group_by=group_by, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_cash_flow(
        cls_or_self,
        free: bool = False,
        orders: tp.Optional[Orders] = None,
        cash_earnings: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        weights: tp.Union[None, bool, tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get cash flow series per column or group.
    
        Use `free` to return the flow of free cash that does not exceed the initial level 
        since any operation incurs a cost.
    
        !!! note
            Does not include cash deposits, but includes earnings.
        
            Using `free` yields the same result as during simulation only when `leverage=1`. 
            For other cases, prefill the state instead of reconstructing it.
    
        Args:
            free (bool): Flag indicating whether to calculate free cash flow.
            orders (Optional[Orders]): Orders instance used for computing cash flow.

                Uses `Portfolio.get_orders` if not provided.
            cash_earnings (Optional[ArrayLike]): Cash earnings array.

                Uses `Portfolio.get_cash_earnings` with `keep_flex=True` and `group_by=False` or 0 if not provided.
            sim_start (Optional[ArrayLike]): Simulation start boundary.
            sim_end (Optional[ArrayLike]): Simulation end boundary.
            rec_sim_range (bool): Flag to use reconstructed simulation range.
            weights (Union[None, bool, ArrayLike]): Weights used for calculations.

                Uses `Portfolio.get_weights` if not provided.
            
                If False, weights are ignored.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to enable chunked processing.
            wrapper (Optional[ArrayWrapper]): Instance used to wrap the resulting array.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Grouping specification for the result.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.
    
        Returns:
            SeriesFrame: Wrapped cash flow series per column or group.
        """
        if not isinstance(cls_or_self, type):
            if orders is None:
                if orders is None:
                    orders = cls_or_self.resolve_shortcut_attr(
                        "orders",
                        sim_start=sim_start if rec_sim_range else None,
                        sim_end=sim_end if rec_sim_range else None,
                        rec_sim_range=rec_sim_range,
                        weights=weights,
                        wrapper=wrapper,
                        group_by=None,
                    )
            if cash_earnings is None:
                cash_earnings = cls_or_self.resolve_shortcut_attr(
                    "cash_earnings",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    weights=weights,
                    keep_flex=True,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=False,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(orders, arg_name="orders")
            if cash_earnings is None:
                cash_earnings = 0.0
            if wrapper is None:
                wrapper = orders.wrapper
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

        func = jit_reg.resolve_option(nb.cash_flow_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        cash_flow = func(
            wrapper.shape_2d,
            orders.values,
            orders.col_mapper.col_map,
            free=free,
            cash_earnings=to_2d_array(cash_earnings),
            sim_start=sim_start,
            sim_end=sim_end,
        )
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.cash_flow_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            cash_flow = func(
                cash_flow,
                group_lens,
                sim_start=sim_start,
                sim_end=sim_end,
            )
        return wrapper.wrap(cash_flow, group_by=group_by, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_init_cash(
        cls_or_self,
        init_cash_raw: tp.Optional[tp.ArrayLike] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        free_cash_flow: tp.Optional[tp.SeriesFrame] = None,
        cash_sharing: tp.Optional[bool] = None,
        split_shared: bool = False,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        weights: tp.Union[None, bool, tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get initial amount of cash per column or group.

        Args:
            init_cash_raw (Optional[ArrayLike]): Initial cash amount or mode identifier.
            
                Uses `Portfolio._init_cash` if not provided.
            cash_deposits (Optional[ArrayLike]): Cash deposits to apply.

                Uses `Portfolio.get_cash_deposits` with `keep_flex=True` or 0 if not provided.
            free_cash_flow (Optional[SeriesFrame]): Cash flow data representing available free cash.

                Uses `Portfolio.get_cash_flow` with `free=True` if not provided.
            cash_sharing (Optional[bool]): Flag indicating whether cash is shared among columns or groups.

                Uses `Portfolio.cash_sharing` if not provided.
            split_shared (bool): If True, split shared cash equally among columns in a group.
            sim_start (Optional[ArrayLike]): Simulation start date or index.
            sim_end (Optional[ArrayLike]): Simulation end date or index.
            rec_sim_range (bool): Indicates if simulation range details are recorded.
            weights (Union[None, bool, ArrayLike]): Weighting factor; if set to False, weights are ignored.

                Uses `Portfolio.get_weights` if not provided.
            
                If False, weights are ignored.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Grouping specification.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.

        Returns:
            MaybeSeries: Wrapped series representing the initial cash.
        """
        if not isinstance(cls_or_self, type):
            if init_cash_raw is None:
                init_cash_raw = cls_or_self._init_cash
            if checks.is_int(init_cash_raw) and init_cash_raw in enums.InitCashMode:
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        "cash_deposits",
                        sim_start=sim_start if rec_sim_range else None,
                        sim_end=sim_end if rec_sim_range else None,
                        rec_sim_range=rec_sim_range,
                        weights=weights,
                        keep_flex=True,
                        jitted=jitted,
                        chunked=chunked,
                        wrapper=wrapper,
                        group_by=group_by,
                    )
                if free_cash_flow is None:
                    free_cash_flow = cls_or_self.resolve_shortcut_attr(
                        "cash_flow",
                        sim_start=sim_start if rec_sim_range else None,
                        sim_end=sim_end if rec_sim_range else None,
                        rec_sim_range=rec_sim_range,
                        weights=weights,
                        free=True,
                        jitted=jitted,
                        chunked=chunked,
                        wrapper=wrapper,
                        group_by=group_by,
                    )
            if cash_sharing is None:
                cash_sharing = cls_or_self.cash_sharing
            if weights is None:
                weights = cls_or_self.resolve_shortcut_attr("weights", wrapper=wrapper)
            elif weights is False:
                weights = None
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_cash_raw, arg_name="init_cash_raw")
            if checks.is_int(init_cash_raw) and init_cash_raw in enums.InitCashMode:
                checks.assert_not_none(free_cash_flow, arg_name="free_cash_flow")
                if cash_deposits is None:
                    cash_deposits = 0.0
            checks.assert_not_none(cash_sharing, arg_name="cash_sharing")
            checks.assert_not_none(wrapper, arg_name="wrapper")
            weights = cls_or_self.get_weights(weights=weights, wrapper=wrapper)
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=group_by)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=group_by)

        if checks.is_int(init_cash_raw) and init_cash_raw in enums.InitCashMode:
            func = jit_reg.resolve_option(nb.align_init_cash_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            init_cash = func(
                init_cash_raw,
                to_2d_array(free_cash_flow),
                cash_deposits=to_2d_array(cash_deposits),
                sim_start=sim_start,
                sim_end=sim_end,
            )
        else:
            init_cash_raw = to_1d_array(init_cash_raw)
            if wrapper.grouper.is_grouped(group_by=group_by):
                group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
                func = jit_reg.resolve_option(nb.init_cash_grouped_nb, jitted)
                init_cash = func(
                    init_cash_raw,
                    group_lens,
                    cash_sharing,
                    weights=to_1d_array(weights) if weights is not None else None,
                )
            else:
                group_lens = wrapper.grouper.get_group_lens()
                func = jit_reg.resolve_option(nb.init_cash_nb, jitted)
                init_cash = func(
                    init_cash_raw,
                    group_lens,
                    cash_sharing,
                    split_shared=split_shared,
                    weights=to_1d_array(weights) if weights is not None else None,
                )
        wrap_kwargs = merge_dicts(dict(name_or_index="init_cash"), wrap_kwargs)
        return wrapper.wrap_reduced(init_cash, group_by=group_by, **wrap_kwargs)

    @hybrid_method
    def get_cash(
        cls_or_self,
        free: bool = False,
        init_cash: tp.Optional[tp.ArrayLike] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_flow: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get cash balance series per column or group.

        Args:
            free (bool): Flag indicating whether free cash flow is used.
            
                For details, see `Portfolio.get_cash_flow`.
            init_cash (Optional[ArrayLike]): Initial cash amount.

                Uses `Portfolio.get_init_cash` if not provided.
            cash_deposits (Optional[ArrayLike]): Cash deposits applied.

                Uses `Portfolio.get_cash_deposits` with `keep_flex=True` or 0 if not provided.
            cash_flow (Optional[SeriesFrame]): Cash flow data.

                Uses `Portfolio.get_cash_flow` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start date or index.
            sim_end (Optional[ArrayLike]): Simulation end date or index.
            rec_sim_range (bool): Indicates if the simulation range should be recorded.
            jitted (JittedOption): Option for JIT compilation.
            chunked (ChunkedOption): Option for chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Grouping specification.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.

        Returns:
            SeriesFrame: Wrapped series with the cash balance.
        """
        if not isinstance(cls_or_self, type):
            if init_cash is None:
                init_cash = cls_or_self.resolve_shortcut_attr(
                    "init_cash",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if cash_deposits is None:
                cash_deposits = cls_or_self.resolve_shortcut_attr(
                    "cash_deposits",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    keep_flex=True,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if cash_flow is None:
                cash_flow = cls_or_self.resolve_shortcut_attr(
                    "cash_flow",
                    free=free,
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_cash, arg_name="init_cash")
            if cash_deposits is None:
                cash_deposits = 0.0
            checks.assert_not_none(cash_flow, arg_name="cash_flow")
            checks.assert_not_none(wrapper, arg_name="wrapper")
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=group_by)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=group_by)

        func = jit_reg.resolve_option(nb.cash_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        cash = func(
            to_2d_array(cash_flow),
            to_1d_array(init_cash),
            cash_deposits=to_2d_array(cash_deposits),
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return wrapper.wrap(cash, group_by=group_by, **resolve_dict(wrap_kwargs))

    # ############# Value ############# #

    @hybrid_method
    def get_init_price(
        cls_or_self,
        init_price_raw: tp.Optional[tp.ArrayLike] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
        keep_flex: bool = False,
    ) -> tp.Union[tp.ArrayLike, tp.MaybeSeries]:
        """Get initial price per column.

        Args:
            init_price_raw (Optional[ArrayLike]): Raw initial price.

                Uses `Portfolio._init_price` if not provided.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.

                Uses `Portfolio.wrapper` if not provided.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.
            keep_flex (bool): If True, returns the input as flexible data without broadcasting.

        Returns:
            Union[ArrayLike, MaybeSeries]: The initial price, possibly wrapped.
        """
        if not isinstance(cls_or_self, type):
            if init_price_raw is None:
                init_price_raw = cls_or_self._init_price
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_price_raw, arg_name="init_price_raw")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        if keep_flex:
            return init_price_raw
        init_price = broadcast_array_to(init_price_raw, wrapper.shape_2d[1])
        if keep_flex:
            return init_price
        wrap_kwargs = merge_dicts(dict(name_or_index="init_price"), wrap_kwargs)
        return wrapper.wrap_reduced(init_price, group_by=False, **wrap_kwargs)

    @hybrid_method
    def get_init_position_value(
        cls_or_self,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        jitted: tp.JittedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return the initial position value per column.

        Args:
            init_position (Optional[ArrayLike]): Initial positions per column.

                Uses `Portfolio.get_init_position` with `keep_flex=True` or 0 if not provided.
            init_price (Optional[ArrayLike]): Prices corresponding to the initial positions.

                Uses `Portfolio.get_init_price` with `keep_flex=True` or NaN if not provided.
            jitted (JittedOption): Option to control JIT compilation.
            wrapper (Optional[ArrayWrapper]): Wrapper instance for output formatting.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Specification for grouping columns.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.

        Returns:
            MaybeSeries: The computed initial position value for each column.
        """
        if not isinstance(cls_or_self, type):
            if init_position is None:
                init_position = cls_or_self.resolve_shortcut_attr(
                    "init_position",
                    wrapper=wrapper,
                    keep_flex=True,
                )
            if init_price is None:
                init_price = cls_or_self.resolve_shortcut_attr(
                    "init_price",
                    wrapper=wrapper,
                    keep_flex=True,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            if init_position is None:
                init_position = 0.0
            if init_price is None:
                init_price = np.nan
            checks.assert_not_none(wrapper, arg_name="wrapper")

        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.init_position_value_grouped_nb, jitted)
            init_position_value = func(
                group_lens,
                init_position=to_1d_array(init_position),
                init_price=to_1d_array(init_price),
            )
        else:
            func = jit_reg.resolve_option(nb.init_position_value_nb, jitted)
            init_position_value = func(
                wrapper.shape_2d[1],
                init_position=to_1d_array(init_position),
                init_price=to_1d_array(init_price),
            )
        wrap_kwargs = merge_dicts(dict(name_or_index="init_position_value"), wrap_kwargs)
        return wrapper.wrap_reduced(init_position_value, group_by=group_by, **wrap_kwargs)

    @hybrid_method
    def get_init_value(
        cls_or_self,
        init_position_value: tp.Optional[tp.MaybeSeries] = None,
        init_cash: tp.Optional[tp.MaybeSeries] = None,
        split_shared: bool = False,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return the initial value per column or group.

        Combines initial cash with the value of the initial position.

        Args:
            init_position_value (Optional[MaybeSeries]): The initial position value per column. 

                Uses `Portfolio.get_init_position` if not provided.
            init_cash (Optional[MaybeSeries]): The initial cash per column.
            
                Uses `Portfolio.get_init_cash` if not provided.
            split_shared (bool): Indicates whether to split shared cash among columns.
            sim_start (Optional[ArrayLike]): Simulation start time.
            sim_end (Optional[ArrayLike]): Simulation end time.
            rec_sim_range (bool): Flag indicating whether to record the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Wrapper instance for output formatting.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Specification for grouping columns.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.

        Returns:
            MaybeSeries: The computed initial value per column or group.
        """
        if not isinstance(cls_or_self, type):
            if init_position_value is None:
                init_position_value = cls_or_self.resolve_shortcut_attr(
                    "init_position_value",
                    jitted=jitted,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if init_cash is None:
                init_cash = cls_or_self.resolve_shortcut_attr(
                    "init_cash",
                    split_shared=split_shared,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_position_value, arg_name="init_position_value")
            checks.assert_not_none(init_cash, arg_name="init_cash")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        func = jit_reg.resolve_option(nb.init_value_nb, jitted)
        init_value = func(to_1d_array(init_position_value), to_1d_array(init_cash))
        wrap_kwargs = merge_dicts(dict(name_or_index="init_value"), wrap_kwargs)
        return wrapper.wrap_reduced(init_value, group_by=group_by, **wrap_kwargs)

    @hybrid_method
    def get_input_value(
        cls_or_self,
        total_cash_deposits: tp.Optional[tp.ArrayLike] = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return the total input value per column or group.

        Aggregates the initial value with cash deposits made over time.

        Args:
            total_cash_deposits (Optional[ArrayLike]): Cash deposits over time.

                Uses `Portfolio.get_total_cash_deposits` or 0 if not provided.
            init_value (Optional[MaybeSeries]): The initial portfolio value.

                Uses `Portfolio.get_init_value` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start time for resolving deposits and value.
            sim_end (Optional[ArrayLike]): Simulation end time for resolving deposits and value.
            rec_sim_range (bool): Flag indicating whether to record the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Wrapper instance for output formatting.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Specification for grouping columns.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.

        Returns:
            MaybeSeries: The aggregated total input value per column or group.
        """
        if not isinstance(cls_or_self, type):
            if total_cash_deposits is None:
                total_cash_deposits = cls_or_self.resolve_shortcut_attr(
                    "total_cash_deposits",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if init_value is None:
                init_value = cls_or_self.resolve_shortcut_attr(
                    "init_value",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            if total_cash_deposits is None:
                total_cash_deposits = 0.0
            checks.assert_not_none(init_value, arg_name="init_value")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        input_value = to_1d_array(total_cash_deposits) + to_1d_array(init_value)
        wrap_kwargs = merge_dicts(dict(name_or_index="input_value"), wrap_kwargs)
        return wrapper.wrap_reduced(input_value, group_by=group_by, **wrap_kwargs)

    @hybrid_method
    def get_asset_value(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        close: tp.Optional[tp.SeriesFrame] = None,
        assets: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Return asset value series per column or group.

        Computes asset values based on close prices and asset quantities. 
        If grouping is active, grouped calculations are applied.

        Args:
            direction (Union[str, int]): Specifies the asset direction. 
            
                Typically "both" is used to consider all directions.

                See `vectorbtpro.portfolio.enums.Direction` for options.
            close (Optional[SeriesFrame]): Series or DataFrame of close prices. 
            
                If not provided, uses `Portfolio.filled_close` if available; otherwise, uses `Portfolio.close`.
            assets (Optional[SeriesFrame]): Series or DataFrame of asset amounts. 

                Uses `Portfolio.get_assets` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start time.
            sim_end (Optional[ArrayLike]): Simulation end time.
            rec_sim_range (bool): Flag indicating whether to record the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Wrapper instance for output formatting.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Specification for grouping columns.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.

        Returns:
            SeriesFrame: The asset value series per column or group.
        """
        if not isinstance(cls_or_self, type):
            if close is None:
                if cls_or_self.fillna_close:
                    close = cls_or_self.filled_close
                else:
                    close = cls_or_self.close
            if assets is None:
                assets = cls_or_self.resolve_shortcut_attr(
                    "assets",
                    direction=direction,
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(close, arg_name="close")
            checks.assert_not_none(assets, arg_name="assets")
            checks.assert_not_none(wrapper, arg_name="wrapper")
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

        func = jit_reg.resolve_option(nb.asset_value_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        asset_value = func(
            to_2d_array(close),
            to_2d_array(assets),
            sim_start=sim_start,
            sim_end=sim_end,
        )
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.asset_value_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            asset_value = func(
                asset_value,
                group_lens,
                sim_start=sim_start,
                sim_end=sim_end,
            )
        return wrapper.wrap(asset_value, group_by=group_by, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_value(
        cls_or_self,
        cash: tp.Optional[tp.SeriesFrame] = None,
        asset_value: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get portfolio value series per column or group.

        Calculates the portfolio value series for each asset or group based on the provided 
        cash flows and asset values. By default, each asset's portfolio value is computed 
        independently with the initial cash balance and positions representing the entire group. 
        This functionality is useful for generating returns and comparing assets within the same group.

        Args:
            cash (Optional[SeriesFrame]): Cash flow series.

                Uses `Portfolio.get_cash` if not provided.
            asset_value (Optional[SeriesFrame]): Asset value series for computation.

                Uses `Portfolio.get_asset_value` if not provided.
            sim_start (Optional[ArrayLike]): Start indices for simulation.
            sim_end (Optional[ArrayLike]): End indices for simulation.
            rec_sim_range (bool): Flag indicating whether to recalculate the simulation range.
            jitted (JittedOption): Option controlling JIT compilation.
            chunked (ChunkedOption): Option controlling chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Grouping specification.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.

        Returns:
            SeriesFrame: The computed portfolio value series.
        """
        if not isinstance(cls_or_self, type):
            if cash is None:
                cash = cls_or_self.resolve_shortcut_attr(
                    "cash",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if asset_value is None:
                asset_value = cls_or_self.resolve_shortcut_attr(
                    "asset_value",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(cash, arg_name="cash")
            checks.assert_not_none(asset_value, arg_name="asset_value")
            checks.assert_not_none(wrapper, arg_name="wrapper")
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=group_by)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=group_by)

        func = jit_reg.resolve_option(nb.value_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        value = func(
            to_2d_array(cash),
            to_2d_array(asset_value),
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return wrapper.wrap(value, group_by=group_by, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_gross_exposure(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        asset_value: tp.Optional[tp.SeriesFrame] = None,
        value: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get gross exposure.

        Calculates the gross exposure for the portfolio by combining asset values and 
        portfolio value. When both directions are considered, ensure that `asset_value` 
        represents the sum of absolute long-only and short-only asset values.

        Args:
            direction (Union[str, int]): Direction indicating the exposure type. 
            
                See `vectorbtpro.portfolio.enums.Direction` for options.
            asset_value (Optional[SeriesFrame]): Asset value series used in the exposure calculation.

                Uses `Portfolio.get_asset_value` if not provided.
            value (Optional[SeriesFrame]): Portfolio value series.

                Uses `Portfolio.get_value` if not provided.
            sim_start (Optional[ArrayLike]): Start indices for simulation.
            sim_end (Optional[ArrayLike]): End indices for simulation.
            rec_sim_range (bool): Flag indicating whether to recalculate the simulation range.
            jitted (JittedOption): Option controlling JIT compilation.
            chunked (ChunkedOption): Option controlling chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Grouping specification.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.

        Returns:
            SeriesFrame: The computed gross exposure series.

        !!! note
            When both directions, `asset_value` must include the addition of the absolute long-only and 
            short-only asset values.
        """
        direction = map_enum_fields(direction, enums.Direction)

        if not isinstance(cls_or_self, type):
            if asset_value is None:
                if direction == enums.Direction.Both and cls_or_self.wrapper.grouper.is_grouped(group_by=group_by):
                    long_asset_value = cls_or_self.resolve_shortcut_attr(
                        "asset_value",
                        direction="longonly",
                        sim_start=sim_start if rec_sim_range else None,
                        sim_end=sim_end if rec_sim_range else None,
                        rec_sim_range=rec_sim_range,
                        jitted=jitted,
                        chunked=chunked,
                        wrapper=wrapper,
                        group_by=group_by,
                    )
                    short_asset_value = cls_or_self.resolve_shortcut_attr(
                        "asset_value",
                        direction="shortonly",
                        sim_start=sim_start if rec_sim_range else None,
                        sim_end=sim_end if rec_sim_range else None,
                        rec_sim_range=rec_sim_range,
                        jitted=jitted,
                        chunked=chunked,
                        wrapper=wrapper,
                        group_by=group_by,
                    )
                    asset_value = long_asset_value + short_asset_value
                else:
                    asset_value = cls_or_self.resolve_shortcut_attr(
                        "asset_value",
                        direction=direction,
                        sim_start=sim_start if rec_sim_range else None,
                        sim_end=sim_end if rec_sim_range else None,
                        rec_sim_range=rec_sim_range,
                        jitted=jitted,
                        chunked=chunked,
                        wrapper=wrapper,
                        group_by=group_by,
                    )
            if value is None:
                value = cls_or_self.resolve_shortcut_attr(
                    "value",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(asset_value, arg_name="asset_value")
            checks.assert_not_none(value, arg_name="value")
            checks.assert_not_none(wrapper, arg_name="wrapper")
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=group_by)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=group_by)

        func = jit_reg.resolve_option(nb.gross_exposure_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        gross_exposure = func(
            to_2d_array(asset_value),
            to_2d_array(value),
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return wrapper.wrap(gross_exposure, group_by=group_by, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_net_exposure(
        cls_or_self,
        long_exposure: tp.Optional[tp.SeriesFrame] = None,
        short_exposure: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get net exposure.

        Calculates the net exposure of the portfolio by combining long and short exposure values.

        Args:
            long_exposure (Optional[SeriesFrame]): Exposure series for long positions.

                Uses `Portfolio.get_gross_exposure` with `direction="longonly"` if not provided.
            short_exposure (Optional[SeriesFrame]): Exposure series for short positions.

                Uses `Portfolio.get_gross_exposure` with `direction="shortonly"` if not provided.
            sim_start (Optional[ArrayLike]): Start indices for simulation.
            sim_end (Optional[ArrayLike]): End indices for simulation.
            rec_sim_range (bool): Flag indicating whether to recalculate the simulation range.
            jitted (JittedOption): Option controlling JIT compilation.
            chunked (ChunkedOption): Option controlling chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Grouping specification.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.

        Returns:
            SeriesFrame: The computed net exposure series.
        """
        if not isinstance(cls_or_self, type):
            if long_exposure is None:
                long_exposure = cls_or_self.resolve_shortcut_attr(
                    "gross_exposure",
                    direction="longonly",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if short_exposure is None:
                short_exposure = cls_or_self.resolve_shortcut_attr(
                    "gross_exposure",
                    direction="shortonly",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(long_exposure, arg_name="long_exposure")
            checks.assert_not_none(short_exposure, arg_name="short_exposure")
            checks.assert_not_none(wrapper, arg_name="wrapper")
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=group_by)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=group_by)

        func = jit_reg.resolve_option(nb.net_exposure_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        net_exposure = func(
            to_2d_array(long_exposure),
            to_2d_array(short_exposure),
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return wrapper.wrap(net_exposure, group_by=group_by, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_allocations(
        cls_or_self,
        direction: tp.Union[str, int] = "both",
        asset_value: tp.Optional[tp.SeriesFrame] = None,
        value: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Return portfolio allocation series per column.
    
        Calculate the portfolio allocation series using the provided asset and 
        portfolio value data. If called on an instance and asset_value or value 
        is not provided, they are derived from shortcut attributes.
    
        Args:
            direction (Union[str, int]): Direction for allocation calculation.

                See `vectorbtpro.portfolio.enums.Direction` for options.
            asset_value (Optional[SeriesFrame]): Asset value series used for allocation calculation.

                Uses `Portfolio.get_asset_value` with `group_by=False` if not provided.
            value (Optional[SeriesFrame]): Portfolio value series used for allocation calculation.

                Uses `Portfolio.get_value` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start index.
            sim_end (Optional[ArrayLike]): Simulation end index.
            rec_sim_range (bool): Flag indicating whether the simulation range should be recovered.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance for formatting the output.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Specification for grouping the results.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping the result.
    
        Returns:
            SeriesFrame: Portfolio allocation series per column.
        """
        if not isinstance(cls_or_self, type):
            if asset_value is None:
                asset_value = cls_or_self.resolve_shortcut_attr(
                    "asset_value",
                    direction=direction,
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=False,
                )
            if value is None:
                value = cls_or_self.resolve_shortcut_attr(
                    "value",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(asset_value, arg_name="asset_value")
            checks.assert_not_none(value, arg_name="value")
            checks.assert_not_none(wrapper, arg_name="wrapper")
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

        group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
        func = jit_reg.resolve_option(nb.allocations_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        allocations = func(
            to_2d_array(asset_value),
            to_2d_array(value),
            group_lens,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return wrapper.wrap(allocations, group_by=False, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_total_profit(
        cls_or_self,
        close: tp.Optional[tp.SeriesFrame] = None,
        orders: tp.Optional[Orders] = None,
        init_position: tp.Optional[tp.ArrayLike] = None,
        init_price: tp.Optional[tp.ArrayLike] = None,
        cash_earnings: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return total profit per column or group.
    
        Compute the total profit from order records using an efficient calculation pathway.
        For instance-level calls, if various inputs are not provided, they are derived using shortcut attributes.
    
        Args:
            close (Optional[SeriesFrame]): Price series (close prices) used for profit calculation.

                If not provided, uses `Portfolio.filled_close` if available; otherwise, uses `Portfolio.close`.
            orders (Optional[Orders]): Orders object containing order records.

                Uses `Portfolio.get_orders` if not provided.
            init_position (Optional[ArrayLike]): Initial portfolio position.

                Uses `Portfolio.get_init_position` with `keep_flex=True` or 0 if not provided.
            init_price (Optional[ArrayLike]): Initial price used for profit calculation.

                Uses `Portfolio.get_init_price` with `keep_flex=True` or NaN if not provided.
            cash_earnings (Optional[ArrayLike]): Cash earnings component included in the profit computation.

                Uses `Portfolio.get_cash_earnings` with `keep_flex=True` and `group_by=False` or 0 if not provided.
            sim_start (Optional[ArrayLike]): Simulation start index.
            sim_end (Optional[ArrayLike]): Simulation end index.
            rec_sim_range (bool): Flag indicating whether to recover the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance for formatting the output.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Specification for grouping profit aggregation.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping the result.
    
        Returns:
            MaybeSeries: Total profit calculated from order records.
        """
        if not isinstance(cls_or_self, type):
            if close is None:
                if cls_or_self.fillna_close:
                    close = cls_or_self.filled_close
                else:
                    close = cls_or_self.close
            if orders is None:
                if orders is None:
                    orders = cls_or_self.resolve_shortcut_attr(
                        "orders",
                        sim_start=sim_start if rec_sim_range else None,
                        sim_end=sim_end if rec_sim_range else None,
                        rec_sim_range=rec_sim_range,
                        wrapper=wrapper,
                        group_by=None,
                    )
            if init_position is None:
                init_position = cls_or_self.resolve_shortcut_attr(
                    "init_position",
                    wrapper=wrapper,
                    keep_flex=True,
                )
            if init_price is None:
                init_price = cls_or_self.resolve_shortcut_attr(
                    "init_price",
                    wrapper=wrapper,
                    keep_flex=True,
                )
            if cash_earnings is None:
                cash_earnings = cls_or_self.resolve_shortcut_attr(
                    "cash_earnings",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    keep_flex=True,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=False,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(orders, arg_name="orders")
            if close is None:
                close = orders.close
            checks.assert_not_none(close, arg_name="close")
            checks.assert_not_none(init_price, arg_name="init_price")
            if init_position is None:
                init_position = 0.0
            if init_price is None:
                init_price = np.nan
            if cash_earnings is None:
                cash_earnings = 0.0
            if wrapper is None:
                wrapper = orders.wrapper
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

        func = jit_reg.resolve_option(nb.total_profit_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        total_profit = func(
            wrapper.shape_2d,
            to_2d_array(close),
            orders.values,
            orders.col_mapper.col_map,
            init_position=to_1d_array(init_position),
            init_price=to_1d_array(init_price),
            cash_earnings=to_2d_array(cash_earnings),
            sim_start=sim_start,
            sim_end=sim_end,
        )
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.total_profit_grouped_nb, jitted)
            total_profit = func(total_profit, group_lens)
        wrap_kwargs = merge_dicts(dict(name_or_index="total_profit"), wrap_kwargs)
        return wrapper.wrap_reduced(total_profit, group_by=group_by, **wrap_kwargs)

    @hybrid_method
    def get_final_value(
        cls_or_self,
        input_value: tp.Optional[tp.MaybeSeries] = None,
        total_profit: tp.Optional[tp.MaybeSeries] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return final portfolio value per column or group.
    
        Calculate the final portfolio value by summing the input value and the total profit.
    
        Args:
            input_value (Optional[MaybeSeries]): Initial portfolio or input value used for calculation.

                Uses `Portfolio.get_input_value` if not provided.
            total_profit (Optional[MaybeSeries]): Total profit to be added to the input value.

                Uses `Portfolio.get_total_profit` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start index.
            sim_end (Optional[ArrayLike]): Simulation end index.
            rec_sim_range (bool): Flag indicating whether to recover the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance for formatting the output.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Specification for grouping the result.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping the result.
    
        Returns:
            MaybeSeries: Final portfolio value after adding the total profit to the input value.
        """
        if not isinstance(cls_or_self, type):
            if input_value is None:
                input_value = cls_or_self.resolve_shortcut_attr(
                    "input_value",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if total_profit is None:
                total_profit = cls_or_self.resolve_shortcut_attr(
                    "total_profit",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(input_value, arg_name="input_value")
            checks.assert_not_none(total_profit, arg_name="total_profit")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        final_value = to_1d_array(input_value) + to_1d_array(total_profit)
        wrap_kwargs = merge_dicts(dict(name_or_index="final_value"), wrap_kwargs)
        return wrapper.wrap_reduced(final_value, group_by=group_by, **wrap_kwargs)

    @hybrid_method
    def get_total_return(
        cls_or_self,
        input_value: tp.Optional[tp.MaybeSeries] = None,
        total_profit: tp.Optional[tp.MaybeSeries] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return total return per column or group.

        Args:
            input_value (Optional[MaybeSeries]): Input values for computing returns.

                Uses `Portfolio.get_input_value` if not provided.
            total_profit (Optional[MaybeSeries]): Total profit values per column or group.

                Uses `Portfolio.get_total_profit` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start time.
            sim_end (Optional[ArrayLike]): Simulation end time.
            rec_sim_range (bool): Flag indicating whether to recalculate the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance used to wrap the result.

                Uses `Portfolio.wrapper` if not provided.
            group_by (tp.GroupByLike): Grouping specification for aggregating data.
            wrap_kwargs (tp.KwargsLike): Additional keyword arguments for the wrapping process.

        Returns:
            MaybeSeries: Wrapped total return computed as the total profit divided by the input value.
        """
        if not isinstance(cls_or_self, type):
            if input_value is None:
                input_value = cls_or_self.resolve_shortcut_attr(
                    "input_value",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if total_profit is None:
                total_profit = cls_or_self.resolve_shortcut_attr(
                    "total_profit",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(input_value, arg_name="input_value")
            checks.assert_not_none(total_profit, arg_name="total_profit")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        total_return = to_1d_array(total_profit) / to_1d_array(input_value)
        wrap_kwargs = merge_dicts(dict(name_or_index="total_return"), wrap_kwargs)
        return wrapper.wrap_reduced(total_return, group_by=group_by, **wrap_kwargs)

    @hybrid_method
    def get_returns(
        cls_or_self,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_deposits_as_input: tp.Optional[bool] = None,
        value: tp.Optional[tp.SeriesFrame] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Return return series calculated from portfolio value.

        Args:
            init_value (Optional[MaybeSeries]): Initial portfolio value.

                Uses `Portfolio.get_input_value` if not provided.
            cash_deposits (Optional[ArrayLike]): Cash deposit amounts for adjusting returns.

                Uses `Portfolio.get_cash_deposits` with `keep_flex=True` or 0 if not provided.
            cash_deposits_as_input (Optional[bool]): Whether to add cash deposits to the input value.

                Uses `Portfolio.cash_deposits_as_input` or False if not provided.
            value (Optional[SeriesFrame]): Portfolio value series.

                Uses `Portfolio.get_value` if not provided.
            log_returns (bool): Flag to compute logarithmic returns.
            daily_returns (bool): Flag to convert computed returns to daily returns.
            sim_start (Optional[ArrayLike]): Simulation start time.
            sim_end (Optional[ArrayLike]): Simulation end time.
            rec_sim_range (bool): Flag indicating whether to recalculate the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance used to wrap the output.

                Uses `Portfolio.wrapper` if not provided.
            group_by (tp.GroupByLike): Grouping specification for aggregating data.
            wrap_kwargs (tp.KwargsLike): Additional keyword arguments for the wrapping process.

        Returns:
            SeriesFrame: Wrapped return series based on portfolio value and cash adjustments.
        """
        if not isinstance(cls_or_self, type):
            if init_value is None:
                init_value = cls_or_self.resolve_shortcut_attr(
                    "init_value",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if cash_deposits is None:
                cash_deposits = cls_or_self.resolve_shortcut_attr(
                    "cash_deposits",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    keep_flex=True,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if cash_deposits_as_input is None:
                cash_deposits_as_input = cls_or_self.cash_deposits_as_input
            if value is None:
                value = cls_or_self.resolve_shortcut_attr(
                    "value",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_value, arg_name="init_value")
            if cash_deposits is None:
                cash_deposits = 0.0
            if cash_deposits_as_input is None:
                cash_deposits_as_input = False
            checks.assert_not_none(value, arg_name="value")
            checks.assert_not_none(wrapper, arg_name="wrapper")
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=group_by)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=group_by)

        func = jit_reg.resolve_option(nb.returns_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        returns = func(
            to_2d_array(value),
            to_1d_array(init_value),
            cash_deposits=to_2d_array(cash_deposits),
            cash_deposits_as_input=cash_deposits_as_input,
            log_returns=log_returns,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        returns = wrapper.wrap(returns, group_by=group_by, **resolve_dict(wrap_kwargs))
        if daily_returns:
            returns = returns.vbt.returns(log_returns=log_returns).daily(jitted=jitted)
        return returns

    @hybrid_method
    def get_asset_pnl(
        cls_or_self,
        init_position_value: tp.Optional[tp.MaybeSeries] = None,
        asset_value: tp.Optional[tp.SeriesFrame] = None,
        cash_flow: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Return asset PnL series combining realized and unrealized profit and loss per column or group.

        Args:
            init_position_value (Optional[MaybeSeries]): Initial asset position values.

                Uses `Portfolio.get_init_position_value` if not provided.
            asset_value (Optional[SeriesFrame]): Asset value series.

                Uses `Portfolio.get_asset_value` if not provided.
            cash_flow (Optional[SeriesFrame]): Cash flow series related to asset transactions.

                Uses `Portfolio.get_cash_flow` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start time.
            sim_end (Optional[ArrayLike]): Simulation end time.
            rec_sim_range (bool): Flag indicating whether to recalculate the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance used to wrap the result.

                Uses `Portfolio.wrapper` if not provided.
            group_by (tp.GroupByLike): Grouping specification for aggregating data.
            wrap_kwargs (tp.KwargsLike): Additional keyword arguments for the wrapping process.

        Returns:
            SeriesFrame: Wrapped asset PnL series combining both realized and unrealized profit and loss.
        """
        if not isinstance(cls_or_self, type):
            if init_position_value is None:
                init_position_value = cls_or_self.resolve_shortcut_attr(
                    "init_position_value",
                    jitted=jitted,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if asset_value is None:
                asset_value = cls_or_self.resolve_shortcut_attr(
                    "asset_value",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if cash_flow is None:
                cash_flow = cls_or_self.resolve_shortcut_attr(
                    "cash_flow",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_position_value, arg_name="init_position_value")
            checks.assert_not_none(asset_value, arg_name="asset_value")
            checks.assert_not_none(cash_flow, arg_name="cash_flow")
            checks.assert_not_none(wrapper, arg_name="wrapper")
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=group_by)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=group_by)

        func = jit_reg.resolve_option(nb.asset_pnl_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        asset_pnl = func(
            to_2d_array(asset_value),
            to_2d_array(cash_flow),
            init_position_value=to_1d_array(init_position_value),
            sim_start=sim_start,
            sim_end=sim_end,
        )
        return wrapper.wrap(asset_pnl, group_by=group_by, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_asset_returns(
        cls_or_self,
        init_position_value: tp.Optional[tp.MaybeSeries] = None,
        asset_value: tp.Optional[tp.SeriesFrame] = None,
        cash_flow: tp.Optional[tp.SeriesFrame] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get asset return series per column or group.

        Computes returns based solely on cash flows and asset value, ignoring passive cash.
        The computed returns remain unaffected by the amount of cash available, even when it is `np.inf`,
        and are comparable to an all-in investment with zero cash held.

        Args:
            init_position_value (Optional[MaybeSeries]): Initial position value series.

                Uses `Portfolio.get_init_position_value` if not provided.
            asset_value (Optional[SeriesFrame]): Asset value series.

                Uses `Portfolio.get_asset_value` if not provided.
            cash_flow (Optional[SeriesFrame]): Cash flow series.

                Uses `Portfolio.get_cash_flow` if not provided.
            log_returns (bool): Compute logarithmic returns if True.
            daily_returns (bool): Convert the computed returns to daily return rates if True.
            sim_start (Optional[ArrayLike]): Simulation start date or index.
            sim_end (Optional[ArrayLike]): Simulation end date or index.
            rec_sim_range (bool): Flag to resolve the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance for processing.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Specification for grouping.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.
        
        Returns:
            SeriesFrame: Computed asset return series.
        """
        if not isinstance(cls_or_self, type):
            if init_position_value is None:
                init_position_value = cls_or_self.resolve_shortcut_attr(
                    "init_position_value",
                    jitted=jitted,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if asset_value is None:
                asset_value = cls_or_self.resolve_shortcut_attr(
                    "asset_value",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if cash_flow is None:
                cash_flow = cls_or_self.resolve_shortcut_attr(
                    "cash_flow",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_position_value, arg_name="init_position_value")
            checks.assert_not_none(asset_value, arg_name="asset_value")
            checks.assert_not_none(cash_flow, arg_name="cash_flow")
            checks.assert_not_none(wrapper, arg_name="wrapper")
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=group_by)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=group_by)

        func = jit_reg.resolve_option(nb.asset_returns_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        asset_returns = func(
            to_2d_array(asset_value),
            to_2d_array(cash_flow),
            init_position_value=to_1d_array(init_position_value),
            log_returns=log_returns,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        asset_returns = wrapper.wrap(asset_returns, group_by=group_by, **resolve_dict(wrap_kwargs))
        if daily_returns:
            asset_returns = asset_returns.vbt.returns(log_returns=log_returns).daily(jitted=jitted)
        return asset_returns

    @hybrid_method
    def get_market_value(
        cls_or_self,
        close: tp.Optional[tp.SeriesFrame] = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get market value series per column or group.

        Computes the market value based on close prices, initial portfolio value, and cash deposits.
        When grouping is applied, the initial cash is evenly distributed among the assets in the group.

        !!! note
            Does not account for fees and slippage. For accurate results, create a separate portfolio.

        Args:
            close (Optional[SeriesFrame]): Price data used for market value computation.

                If not provided, uses `Portfolio.filled_close` if available; otherwise, uses `Portfolio.close`.
            init_value (Optional[MaybeSeries]): Initial portfolio value.

                Uses `Portfolio.get_init_value` with `group_by=False` if not provided.
            cash_deposits (Optional[ArrayLike]): Cash deposit values.

                Uses `Portfolio.get_cash_deposits` with `split_shared=True`, `keep_flex=True`, 
                and `group_by=False` or 0 if not provided.
            sim_start (Optional[ArrayLike]): Simulation start date or index.
            sim_end (Optional[ArrayLike]): Simulation end date or index.
            rec_sim_range (bool): Flag to resolve the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance for processing.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Specification for grouping.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.
        
        Returns:
            SeriesFrame: Computed market value series.
        """
        if not isinstance(cls_or_self, type):
            if close is None:
                if cls_or_self.fillna_close:
                    close = cls_or_self.filled_close
                else:
                    close = cls_or_self.close
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(close, arg_name="close")
            checks.assert_not_none(init_value, arg_name="init_value")
            if cash_deposits is None:
                cash_deposits = 0.0
            checks.assert_not_none(wrapper, arg_name="wrapper")

        if wrapper.grouper.is_grouped(group_by=group_by):
            if not isinstance(cls_or_self, type):
                if init_value is None:
                    init_value = cls_or_self.resolve_shortcut_attr(
                        "init_value",
                        split_shared=True,
                        sim_start=sim_start if rec_sim_range else None,
                        sim_end=sim_end if rec_sim_range else None,
                        rec_sim_range=rec_sim_range,
                        jitted=jitted,
                        chunked=chunked,
                        wrapper=wrapper,
                        group_by=False,
                    )
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        "cash_deposits",
                        split_shared=True,
                        sim_start=sim_start if rec_sim_range else None,
                        sim_end=sim_end if rec_sim_range else None,
                        rec_sim_range=rec_sim_range,
                        keep_flex=True,
                        jitted=jitted,
                        chunked=chunked,
                        wrapper=wrapper,
                        group_by=False,
                    )
            sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
            sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_reg.resolve_option(nb.market_value_grouped_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            market_value = func(
                to_2d_array(close),
                group_lens,
                to_1d_array(init_value),
                cash_deposits=to_2d_array(cash_deposits),
                sim_start=sim_start,
                sim_end=sim_end,
            )
        else:
            if not isinstance(cls_or_self, type):
                if init_value is None:
                    init_value = cls_or_self.resolve_shortcut_attr(
                        "init_value",
                        sim_start=sim_start if rec_sim_range else None,
                        sim_end=sim_end if rec_sim_range else None,
                        rec_sim_range=rec_sim_range,
                        jitted=jitted,
                        chunked=chunked,
                        wrapper=wrapper,
                        group_by=False,
                    )
                if cash_deposits is None:
                    cash_deposits = cls_or_self.resolve_shortcut_attr(
                        "cash_deposits",
                        sim_start=sim_start if rec_sim_range else None,
                        sim_end=sim_end if rec_sim_range else None,
                        rec_sim_range=rec_sim_range,
                        keep_flex=True,
                        jitted=jitted,
                        chunked=chunked,
                        wrapper=wrapper,
                        group_by=False,
                    )
            sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=False)
            sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=False)

            func = jit_reg.resolve_option(nb.market_value_nb, jitted)
            func = ch_reg.resolve_option(func, chunked)
            market_value = func(
                to_2d_array(close),
                to_1d_array(init_value),
                cash_deposits=to_2d_array(cash_deposits),
                sim_start=sim_start,
                sim_end=sim_end,
            )
        return wrapper.wrap(market_value, group_by=group_by, **resolve_dict(wrap_kwargs))

    @hybrid_method
    def get_market_returns(
        cls_or_self,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_deposits_as_input: tp.Optional[bool] = None,
        market_value: tp.Optional[tp.SeriesFrame] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Return market return series per column or group.

        Args:
            init_value (Optional[MaybeSeries]): Initial market value used as a baseline for computations.

                Uses `Portfolio.get_init_value` if not provided.
            cash_deposits (Optional[ArrayLike]): Cash deposit amounts for the computation.

                Uses `Portfolio.get_cash_deposits` with `keep_flex=True` or 0 if not provided.
            cash_deposits_as_input (Optional[bool]): Flag indicating whether cash deposits are provided as input.

                Uses `Portfolio.cash_deposits_as_input` or False if not provided.
            market_value (Optional[SeriesFrame]): Market value series required for the return calculation.

                Uses `Portfolio.get_market_value` if not provided.
            log_returns (bool): Compute logarithmic returns if True.
            daily_returns (bool): Convert the computed returns to daily frequency if True.
            sim_start (Optional[ArrayLike]): Simulation start date or index.
            sim_end (Optional[ArrayLike]): Simulation end date or index.
            rec_sim_range (bool): Flag to recalculate the simulation range based on provided parameters.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Wrapper instance used to format the result.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Grouping specification for processing.
            wrap_kwargs (KwargsLike): Additional keyword arguments for the wrapper.

        Returns:
            SeriesFrame: The computed market return series.
        """
        if not isinstance(cls_or_self, type):
            if init_value is None:
                init_value = cls_or_self.resolve_shortcut_attr(
                    "init_value",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if cash_deposits is None:
                cash_deposits = cls_or_self.resolve_shortcut_attr(
                    "cash_deposits",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    keep_flex=True,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if cash_deposits_as_input is None:
                cash_deposits_as_input = cls_or_self.cash_deposits_as_input
            if market_value is None:
                market_value = cls_or_self.resolve_shortcut_attr(
                    "market_value",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_value, arg_name="init_value")
            if cash_deposits is None:
                cash_deposits = 0.0
            if cash_deposits_as_input is None:
                cash_deposits_as_input = False
            checks.assert_not_none(market_value, arg_name="market_value")
            checks.assert_not_none(wrapper, arg_name="wrapper")
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=group_by)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=group_by)

        func = jit_reg.resolve_option(nb.returns_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        market_returns = func(
            to_2d_array(market_value),
            to_1d_array(init_value),
            cash_deposits=to_2d_array(cash_deposits),
            cash_deposits_as_input=cash_deposits_as_input,
            log_returns=log_returns,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        market_returns = wrapper.wrap(market_returns, group_by=group_by, **resolve_dict(wrap_kwargs))
        if daily_returns:
            market_returns = market_returns.vbt.returns(log_returns=log_returns).daily(jitted=jitted)
        return market_returns

    @hybrid_method
    def get_total_market_return(
        cls_or_self,
        input_value: tp.Optional[tp.MaybeSeries] = None,
        market_value: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Return total market return.

        Args:
            init_value (Optional[MaybeSeries]): Initial market value used as a baseline for computations.

                Uses `Portfolio.get_init_value` if not provided.
            market_value (Optional[SeriesFrame]): Market value series used for computing returns.

                Uses `Portfolio.get_market_value` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start date or index.
            sim_end (Optional[ArrayLike]): Simulation end date or index.
            rec_sim_range (bool): Flag to recalculate the simulation range based on provided parameters.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Wrapper instance for formatting the result.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Grouping specification for processing.
            wrap_kwargs (KwargsLike): Additional keyword arguments for the wrapper.

        Returns:
            MaybeSeries: The total market return as a reduced value.
        """
        if not isinstance(cls_or_self, type):
            if input_value is None:
                input_value = cls_or_self.resolve_shortcut_attr(
                    "input_value",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if market_value is None:
                market_value = cls_or_self.resolve_shortcut_attr(
                    "market_value",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(input_value, arg_name="input_value")
            checks.assert_not_none(market_value, arg_name="market_value")
            checks.assert_not_none(wrapper, arg_name="wrapper")
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=group_by)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=group_by)

        func = jit_reg.resolve_option(nb.total_market_return_nb, jitted)
        total_market_return = func(
            to_2d_array(market_value),
            to_1d_array(input_value),
            sim_start=sim_start,
            sim_end=sim_end,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="total_market_return"), wrap_kwargs)
        return wrapper.wrap_reduced(total_market_return, group_by=group_by, **wrap_kwargs)

    @hybrid_method
    def get_bm_value(
        cls_or_self,
        bm_close: tp.Optional[tp.ArrayLike] = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Return benchmark value series per column or group.

        Based on `Portfolio.bm_close` and `Portfolio.get_market_value`.

        Args:
            bm_close (Optional[ArrayLike]): Benchmark closing price data.

                If not provided, uses `Portfolio.filled_bm_close` if available; otherwise, uses `Portfolio.bm_close`.
            init_value (Optional[MaybeSeries]): Initial portfolio value.
            cash_deposits (Optional[ArrayLike]): Cash deposit values.
            sim_start (Optional[ArrayLike]): Simulation start date or index.
            sim_end (Optional[ArrayLike]): Simulation end date or index.
            rec_sim_range (bool): Flag to resolve the simulation range.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance for processing.
            group_by (GroupByLike): Specification for grouping.
            wrap_kwargs (KwargsLike): Additional keyword arguments for wrapping.

        Returns:
            Optional[SeriesFrame]: A wrapped series of benchmark values per column or group, or 
                None if benchmarking is disabled.
        """
        if not isinstance(cls_or_self, type):
            if bm_close is None:
                bm_close = cls_or_self.bm_close
                if isinstance(bm_close, bool):
                    if not bm_close:
                        return None
                    bm_close = None
                if bm_close is not None:
                    if cls_or_self.fillna_close:
                        bm_close = cls_or_self.filled_bm_close
        return cls_or_self.get_market_value(
            close=bm_close,
            init_value=init_value,
            cash_deposits=cash_deposits,
            sim_start=sim_start,
            sim_end=sim_end,
            rec_sim_range=rec_sim_range,
            jitted=jitted,
            chunked=chunked,
            wrapper=wrapper,
            group_by=group_by,
            wrap_kwargs=wrap_kwargs,
        )

    @hybrid_method
    def get_bm_returns(
        cls_or_self,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        cash_deposits: tp.Optional[tp.ArrayLike] = None,
        cash_deposits_as_input: tp.Optional[bool] = None,
        bm_value: tp.Optional[tp.SeriesFrame] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Get benchmark return series per column or group.

        Based on `Portfolio.get_bm_value` and `Portfolio.get_market_returns`.

        Args:
            init_value (Optional[MaybeSeries]): Initial portfolio value.
            cash_deposits (Optional[ArrayLike]): Cash deposit values.
            cash_deposits_as_input (Optional[bool]): Flag indicating whether cash deposits are provided as input.
            bm_value (Optional[SeriesFrame]): Benchmark value series or frame.

                Uses `Portfolio.get_bm_value` if not provided.
            log_returns (bool): Compute logarithmic returns if True.
            daily_returns (bool): Convert the computed returns to daily frequency if True.
            sim_start (Optional[ArrayLike]): Simulation start date or index.
            sim_end (Optional[ArrayLike]): Simulation end date or index.
            rec_sim_range (bool): Flag to recalculate the simulation range based on provided parameters.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Wrapper instance used to format the result.
            group_by (GroupByLike): Grouping specification for processing.
            wrap_kwargs (KwargsLike): Additional keyword arguments for the wrapper.

        Returns:
            Optional[SeriesFrame]: Benchmark return series per column or group, or 
                None if benchmarking is disabled.
        """
        if not isinstance(cls_or_self, type):
            bm_value = cls_or_self.resolve_shortcut_attr(
                "bm_value",
                sim_start=sim_start if rec_sim_range else None,
                sim_end=sim_end if rec_sim_range else None,
                rec_sim_range=rec_sim_range,
                jitted=jitted,
                chunked=chunked,
                wrapper=wrapper,
                group_by=group_by,
            )
            if bm_value is None:
                return None
        return cls_or_self.get_market_returns(
            init_value=init_value,
            cash_deposits=cash_deposits,
            cash_deposits_as_input=cash_deposits_as_input,
            market_value=bm_value,
            log_returns=log_returns,
            daily_returns=daily_returns,
            sim_start=sim_start,
            sim_end=sim_end,
            rec_sim_range=rec_sim_range,
            jitted=jitted,
            chunked=chunked,
            wrapper=wrapper,
            group_by=group_by,
            wrap_kwargs=wrap_kwargs,
        )

    @hybrid_method
    def get_returns_acc(
        cls_or_self,
        returns: tp.Optional[tp.SeriesFrame] = None,
        use_asset_returns: bool = False,
        bm_returns: tp.Union[None, bool, tp.ArrayLike] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        freq: tp.Optional[tp.FrequencyLike] = None,
        year_freq: tp.Optional[tp.FrequencyLike] = None,
        defaults: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> ReturnsAccessor:
        """Get returns accessor of type `vectorbtpro.returns.accessors.ReturnsAccessor`.

        Retrieve the returns accessor for the portfolio. If the return series is not provided, 
        it is resolved based on the `use_asset_returns` flag and other parameters.

        Args:
            returns (Optional[SeriesFrame]): Return series or frame.

                Uses `Portfolio.get_returns` or `Portfolio.get_asset_returns` if not provided.
            use_asset_returns (bool): Indicates whether to use asset returns for calculation.
            bm_returns (Union[None, bool, ArrayLike]): Benchmark returns or a flag to resolve benchmark returns.

                Uses `Portfolio.get_bm_returns` if not provided.
            log_returns (bool): Flag to compute logarithmic returns.
            daily_returns (bool): Flag to compute daily returns.
            sim_start (Optional[ArrayLike]): Simulation start time.
            sim_end (Optional[ArrayLike]): Simulation end time.
            rec_sim_range (bool): Flag indicating whether to record simulation range.
            freq (Optional[FrequencyLike]): Frequency for returns calculation.

                Uses `vectorbtpro.base.wrapping.ArrayWrapper.freq` if not provided.
            year_freq (Optional[FrequencyLike]): Yearly frequency for returns calculation.

                Uses `Portfolio.year_freq` if not provided.
            defaults (KwargsLike): Dictionary of default parameters.

                Merges with `Portfolio.returns_acc_defaults`.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Grouping specification.
            **kwargs: Additional keyword arguments for the returns accessor.

        Returns:
            ReturnsAccessor: Returns accessor instance.
        """
        if not isinstance(cls_or_self, type):
            if returns is None:
                if use_asset_returns:
                    returns = cls_or_self.resolve_shortcut_attr(
                        "asset_returns",
                        log_returns=log_returns,
                        daily_returns=daily_returns,
                        sim_start=sim_start if rec_sim_range else None,
                        sim_end=sim_end if rec_sim_range else None,
                        rec_sim_range=rec_sim_range,
                        jitted=jitted,
                        chunked=chunked,
                        wrapper=wrapper,
                        group_by=group_by,
                    )
                else:
                    returns = cls_or_self.resolve_shortcut_attr(
                        "returns",
                        log_returns=log_returns,
                        daily_returns=daily_returns,
                        sim_start=sim_start if rec_sim_range else None,
                        sim_end=sim_end if rec_sim_range else None,
                        rec_sim_range=rec_sim_range,
                        jitted=jitted,
                        chunked=chunked,
                        wrapper=wrapper,
                        group_by=group_by,
                    )
            if bm_returns is None or (isinstance(bm_returns, bool) and bm_returns):
                bm_returns = cls_or_self.resolve_shortcut_attr(
                    "bm_returns",
                    log_returns=log_returns,
                    daily_returns=daily_returns,
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    jitted=jitted,
                    chunked=chunked,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            elif isinstance(bm_returns, bool) and not bm_returns:
                bm_returns = None
            if freq is None:
                freq = cls_or_self.wrapper.freq
            if year_freq is None:
                year_freq = cls_or_self.year_freq
            defaults = merge_dicts(cls_or_self.returns_acc_defaults, defaults)
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(returns, arg_name="returns")
        sim_start = cls_or_self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=group_by)
        sim_end = cls_or_self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=group_by)

        if daily_returns:
            freq = "D"
        if wrapper is not None:
            wrapper = wrapper.resolve(group_by=group_by)
        return returns.vbt.returns(
            wrapper=wrapper,
            bm_returns=bm_returns,
            log_returns=log_returns,
            freq=freq,
            year_freq=year_freq,
            defaults=defaults,
            sim_start=sim_start,
            sim_end=sim_end,
            **kwargs,
        )

    @property
    def returns_acc(self) -> ReturnsAccessor:
        """Return the returns accessor computed by `Portfolio.get_returns_acc` with default arguments.

        Returns:
            ReturnsAccessor: An instance of `vectorbtpro.returns.accessors.ReturnsAccessor`.
        """
        return self.get_returns_acc()

    @hybrid_method
    def get_qs(
        cls_or_self,
        returns: tp.Optional[tp.SeriesFrame] = None,
        use_asset_returns: bool = False,
        bm_returns: tp.Union[None, bool, tp.ArrayLike] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        freq: tp.Optional[tp.FrequencyLike] = None,
        year_freq: tp.Optional[tp.FrequencyLike] = None,
        defaults: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> QSAdapterT:
        """Get quantstats adapter of type `vectorbtpro.returns.qs_adapter.QSAdapter`.

        Compute and return a quantstats adapter based on the portfolio's returns accessor.

        Based on `Portfolio.get_returns_acc`.

        Args:
            returns (Optional[SeriesFrame]): Return series or frame.
            use_asset_returns (bool): Indicates whether to use asset returns for calculation.
            bm_returns (Union[None, bool, ArrayLike]): Benchmark returns or a flag to resolve benchmark returns.
            log_returns (bool): Flag to compute logarithmic returns.
            daily_returns (bool): Flag to compute daily returns.
            sim_start (Optional[ArrayLike]): Simulation start time.
            sim_end (Optional[ArrayLike]): Simulation end time.
            rec_sim_range (bool): Flag indicating whether to record simulation range.
            freq (Optional[FrequencyLike]): Frequency for returns calculation.
            year_freq (Optional[FrequencyLike]): Yearly frequency for returns calculation.
            defaults (KwargsLike): Dictionary of default parameters.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            group_by (GroupByLike): Grouping specification.
            **kwargs: Additional keyword arguments for the quantstats adapter.

        Returns:
            QSAdapter: Quantstats adapter instance.
        """
        from vectorbtpro.returns.qs_adapter import QSAdapter

        returns_acc = cls_or_self.get_returns_acc(
            returns=returns,
            use_asset_returns=use_asset_returns,
            bm_returns=bm_returns,
            log_returns=log_returns,
            daily_returns=daily_returns,
            sim_start=sim_start,
            sim_end=sim_end,
            rec_sim_range=rec_sim_range,
            freq=freq,
            year_freq=year_freq,
            defaults=defaults,
            jitted=jitted,
            chunked=chunked,
            wrapper=wrapper,
            group_by=group_by,
        )
        return QSAdapter(returns_acc, **kwargs)

    @property
    def qs(self) -> QSAdapterT:
        """Return the quantstats adapter computed by `Portfolio.get_qs` with default arguments.

        Returns:
            QSAdapter: An instance of `vectorbtpro.returns.qs.QSAdapter`.
        """
        return self.get_qs()

    # ############# Resolution ############# #

    @property
    def self_aliases(self) -> tp.Set[str]:
        """Names to associate with this object.

        Returns:
            Set[str]: A set of names that can be used to refer to this object.
        """
        return {"self", "portfolio", "pf"}

    def pre_resolve_attr(self, attr: str, final_kwargs: tp.KwargsLike = None) -> str:
        """Pre-process an attribute before resolution.

        Apply modifications based on the provided keyword arguments. Specifically, if `use_asset_returns` 
        is set in `final_kwargs` and the attribute is "returns", it is changed to "asset_returns". 
        Similarly, if `trades_type` is provided, the attribute may be modified to "entry_trades", 
        "exit_trades", or "positions" based on the trade type.

        Args:
            attr (str): The attribute name to be processed.
            final_kwargs (KwargsLike): Additional keyword arguments that may influence attribute resolution.

        Returns:
            str: The pre-processed attribute name.
        """
        if "use_asset_returns" in final_kwargs:
            if attr == "returns" and final_kwargs["use_asset_returns"]:
                attr = "asset_returns"
        if "trades_type" in final_kwargs:
            trades_type = final_kwargs["trades_type"]
            if isinstance(final_kwargs["trades_type"], str):
                trades_type = map_enum_fields(trades_type, enums.TradesType)
            if attr == "trades" and trades_type != self.trades_type:
                if trades_type == enums.TradesType.EntryTrades:
                    attr = "entry_trades"
                elif trades_type == enums.TradesType.ExitTrades:
                    attr = "exit_trades"
                else:
                    attr = "positions"
        return attr

    def post_resolve_attr(self, attr: str, out: tp.Any, final_kwargs: tp.KwargsLike = None) -> str:
        """Post-process a resolved attribute value.

        If `final_kwargs` contains the key `incl_open` set to False and `out` is an instance of
        `vectorbtpro.portfolio.trades.Trades`, only closed trades are returned.

        Args:
            attr (str): The attribute name being processed.
            out (Any): The resolved attribute value.
            final_kwargs (KwargsLike, optional): Additional keyword arguments that may include `incl_open`
                to control the inclusion of open trades.

        Returns:
            str: The post-processed attribute value.
        """
        if "incl_open" in final_kwargs:
            if isinstance(out, Trades) and not final_kwargs["incl_open"]:
                out = out.status_closed
        return out

    def resolve_shortcut_attr(self, attr_name: str, *args, **kwargs) -> tp.Any:
        """Resolve an attribute, utilizing shortcut properties if available.

        Checks if the given attribute name or its shortcut variant (prefixed with `get_`) can be accessed
        directly as a property or needs to be called as a function. When no positional arguments are provided,
        additional keyword arguments such as `free`, `direction`, and `group_by` may influence the resolution
        by selecting an alternative cached property.

        Args:
            attr_name (str): The name of the attribute to resolve.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments for attribute resolution.

        Returns:
            Any: The resolved attribute value.
        """
        if not attr_name.startswith("get_"):
            if "get_" + attr_name not in self.cls_dir or (len(args) == 0 and len(kwargs) == 0):
                if isinstance(getattr(type(self), attr_name), property):
                    return getattr(self, attr_name)
                return getattr(self, attr_name)(*args, **kwargs)
            attr_name = "get_" + attr_name

        if len(args) == 0:
            naked_attr_name = attr_name[4:]
            prop_name = naked_attr_name
            _kwargs = dict(kwargs)

            if "free" in _kwargs:
                if _kwargs.pop("free"):
                    prop_name = "free_" + naked_attr_name
            if "direction" in _kwargs:
                direction = map_enum_fields(_kwargs.pop("direction"), enums.Direction)
                if direction == enums.Direction.LongOnly:
                    prop_name = "long_" + naked_attr_name
                elif direction == enums.Direction.ShortOnly:
                    prop_name = "short_" + naked_attr_name

            if prop_name in self.cls_dir:
                prop = getattr(type(self), prop_name)
                options = getattr(prop, "options", {})

                can_call_prop = True
                if "group_by" in _kwargs:
                    group_by = _kwargs.pop("group_by")
                    group_aware = options.get("group_aware", True)
                    if group_aware:
                        if self.wrapper.grouper.is_grouping_modified(group_by=group_by):
                            can_call_prop = False
                    else:
                        group_by = _kwargs.pop("group_by")
                        if self.wrapper.grouper.is_grouping_enabled(group_by=group_by):
                            can_call_prop = False
                if can_call_prop:
                    _kwargs.pop("jitted", None)
                    _kwargs.pop("chunked", None)
                    for k, v in get_func_kwargs(getattr(type(self), attr_name)).items():
                        if k in _kwargs and v is not _kwargs.pop(k):
                            can_call_prop = False
                            break
                    if can_call_prop:
                        if len(_kwargs) > 0:
                            can_call_prop = False
                        if can_call_prop:
                            return getattr(self, prop_name)

        return getattr(self, attr_name)(*args, **kwargs)

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Default configuration for portfolio statistics.

        Merges defaults from `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults`
        with the portfolio statistics settings from `vectorbtpro._settings.portfolio`.

        Returns:
            Kwargs: A dictionary containing the default configuration for portfolio statistics.
        """
        from vectorbtpro._settings import settings

        portfolio_stats_cfg = settings["portfolio"]["stats"]

        return merge_dicts(
            Analyzable.stats_defaults.__get__(self),
            dict(settings=dict(trades_type=self.trades_type)),
            portfolio_stats_cfg,
        )

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start_index=dict(
                title="Start Index",
                calc_func="sim_start_index",
                tags="wrapper",
            ),
            end_index=dict(
                title="End Index",
                calc_func="sim_end_index",
                tags="wrapper",
            ),
            total_duration=dict(
                title="Total Duration",
                calc_func="sim_duration",
                apply_to_timedelta=True,
                tags="wrapper",
            ),
            start_value=dict(
                title="Start Value",
                calc_func="init_value",
                tags="portfolio",
            ),
            min_value=dict(
                title="Min Value",
                calc_func="value.vbt.min",
                tags="portfolio",
            ),
            max_value=dict(
                title="Max Value",
                calc_func="value.vbt.max",
                tags="portfolio",
            ),
            end_value=dict(
                title="End Value",
                calc_func="final_value",
                tags="portfolio",
            ),
            cash_deposits=dict(
                title="Total Cash Deposits",
                calc_func="total_cash_deposits",
                check_has_cash_deposits=True,
                tags="portfolio",
            ),
            cash_earnings=dict(
                title="Total Cash Earnings",
                calc_func="total_cash_earnings",
                check_has_cash_earnings=True,
                tags="portfolio",
            ),
            total_return=dict(
                title="Total Return [%]",
                calc_func="total_return",
                post_calc_func=lambda self, out, settings: out * 100,
                tags="portfolio",
            ),
            bm_return=dict(
                title="Benchmark Return [%]",
                calc_func="bm_returns.vbt.returns.total",
                post_calc_func=lambda self, out, settings: out * 100,
                check_has_bm_returns=True,
                tags="portfolio",
            ),
            total_time_exposure=dict(
                title="Position Coverage [%]",
                calc_func="position_coverage",
                post_calc_func=lambda self, out, settings: out * 100,
                tags="portfolio",
            ),
            max_gross_exposure=dict(
                title="Max Gross Exposure [%]",
                calc_func="gross_exposure.vbt.max",
                post_calc_func=lambda self, out, settings: out * 100,
                tags="portfolio",
            ),
            max_dd=dict(
                title="Max Drawdown [%]",
                calc_func="drawdowns.max_drawdown",
                post_calc_func=lambda self, out, settings: -out * 100,
                tags=["portfolio", "drawdowns"],
            ),
            max_dd_duration=dict(
                title="Max Drawdown Duration",
                calc_func="drawdowns.max_duration",
                fill_wrap_kwargs=True,
                tags=["portfolio", "drawdowns", "duration"],
            ),
            total_orders=dict(
                title="Total Orders",
                calc_func="orders.count",
                tags=["portfolio", "orders"],
            ),
            total_fees_paid=dict(
                title="Total Fees Paid",
                calc_func="orders.fees.sum",
                tags=["portfolio", "orders"],
            ),
            total_trades=dict(
                title="Total Trades",
                calc_func="trades.count",
                incl_open=True,
                tags=["portfolio", "trades"],
            ),
            win_rate=dict(
                title="Win Rate [%]",
                calc_func="trades.win_rate",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]"),
            ),
            best_trade=dict(
                title="Best Trade [%]",
                calc_func="trades.returns.max",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]"),
            ),
            worst_trade=dict(
                title="Worst Trade [%]",
                calc_func="trades.returns.min",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]"),
            ),
            avg_winning_trade=dict(
                title="Avg Winning Trade [%]",
                calc_func="trades.winning.returns.mean",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'winning']"),
            ),
            avg_losing_trade=dict(
                title="Avg Losing Trade [%]",
                calc_func="trades.losing.returns.mean",
                post_calc_func=lambda self, out, settings: out * 100,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'losing']"),
            ),
            avg_winning_trade_duration=dict(
                title="Avg Winning Trade Duration",
                calc_func="trades.winning.duration.mean",
                apply_to_timedelta=True,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'winning', 'duration']"),
            ),
            avg_losing_trade_duration=dict(
                title="Avg Losing Trade Duration",
                calc_func="trades.losing.duration.mean",
                apply_to_timedelta=True,
                tags=RepEval("['portfolio', 'trades', *incl_open_tags, 'losing', 'duration']"),
            ),
            profit_factor=dict(
                title="Profit Factor",
                calc_func="trades.profit_factor",
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]"),
            ),
            expectancy=dict(
                title="Expectancy",
                calc_func="trades.expectancy",
                tags=RepEval("['portfolio', 'trades', *incl_open_tags]"),
            ),
            sharpe_ratio=dict(
                title="Sharpe Ratio",
                calc_func="returns_acc.sharpe_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags=["portfolio", "returns"],
            ),
            calmar_ratio=dict(
                title="Calmar Ratio",
                calc_func="returns_acc.calmar_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags=["portfolio", "returns"],
            ),
            omega_ratio=dict(
                title="Omega Ratio",
                calc_func="returns_acc.omega_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags=["portfolio", "returns"],
            ),
            sortino_ratio=dict(
                title="Sortino Ratio",
                calc_func="returns_acc.sortino_ratio",
                check_has_freq=True,
                check_has_year_freq=True,
                tags=["portfolio", "returns"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    def returns_stats(
        self,
        returns: tp.Optional[tp.SeriesFrame] = None,
        use_asset_returns: bool = False,
        bm_returns: tp.Union[None, bool, tp.ArrayLike] = None,
        log_returns: bool = False,
        daily_returns: bool = False,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        year_freq: tp.Optional[tp.FrequencyLike] = None,
        defaults: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Compute various statistics on portfolio returns.

        This method computes a range of return statistics from the portfolio's returns, with additional
        keyword arguments forwarded to the corresponding stats method of the returns accessor.

        Based on `Portfolio.get_returns_acc`.

        Args:
            returns (Optional[SeriesFrame]): Return series or frame.
            use_asset_returns (bool): Indicates whether to use asset returns for calculation.
            bm_returns (Union[None, bool, ArrayLike]): Benchmark returns or a flag to resolve benchmark returns.
            log_returns (bool): Flag to compute logarithmic returns.
            daily_returns (bool): Flag to compute daily returns.
            sim_start (Optional[ArrayLike]): Simulation start time.

                Passed inside `settings`.
            sim_end (Optional[ArrayLike]): Simulation end time.

                Passed inside `settings`.
            freq (Optional[FrequencyLike]): Frequency for returns calculation.
            year_freq (Optional[FrequencyLike]): Yearly frequency for returns calculation.
            defaults (KwargsLike): Dictionary of default parameters.
            jitted (JittedOption): Option to control JIT compilation.
            chunked (ChunkedOption): Option to control chunked processing.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance.
            group_by (GroupByLike): Grouping specification.
            **kwargs: Additional keyword arguments passed to `vectorbtpro.returns.accessors.ReturnsAccessor.stats`.

        Returns:
            SeriesFrame: A DataFrame or Series containing the computed return statistics.
        """
        returns_acc = self.get_returns_acc(
            returns=returns,
            use_asset_returns=use_asset_returns,
            bm_returns=bm_returns,
            log_returns=log_returns,
            daily_returns=daily_returns,
            sim_start=False,
            sim_end=False,
            freq=freq,
            year_freq=year_freq,
            defaults=defaults,
            jitted=jitted,
            chunked=chunked,
            wrapper=wrapper,
            group_by=group_by,
        )
        settings = dict(kwargs.pop("settings", {}))
        settings["sim_start"] = self.resolve_sim_start(sim_start=sim_start, wrapper=wrapper, group_by=group_by)
        settings["sim_end"] = self.resolve_sim_end(sim_end=sim_end, wrapper=wrapper, group_by=group_by)
        return getattr(returns_acc, "stats")(settings=settings, **kwargs)

    # ############# Plotting ############# #

    @hybrid_method
    def plot_orders(
        cls_or_self,
        column: tp.Optional[tp.Label] = None,
        orders: tp.Optional[Drawdowns] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        fit_sim_range: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        xref: tp.Optional[str] = None,
        yref: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column of orders.

        Args:
            column (Optional[Label]): Column name to plot orders.
            orders (Optional[Drawdowns]): Orders instance for plotting.

                Uses `Portfolio.get_orders` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start index.
            sim_end (Optional[ArrayLike]): Simulation end index.
            rec_sim_range (bool): Flag indicating whether to record the simulation range.
            fit_sim_range (bool): Flag specifying whether to adjust the figure to the simulation range.
            wrapper (Optional[ArrayWrapper]): Array wrapper used for data transformations.
            xref (Optional[str]): X-axis reference identifier; if None, it is derived from the figure.
            yref (Optional[str]): Y-axis reference identifier; if None, it is derived from the figure.
            **kwargs: Additional keyword arguments passed to `vectorbtpro.portfolio.orders.Orders.plot`.

        Returns:
            BaseFigure: Plotly figure with the orders plot.
        """
        if not isinstance(cls_or_self, type):
            if orders is None:
                orders = cls_or_self.resolve_shortcut_attr(
                    "orders",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                )
        else:
            checks.assert_not_none(orders, arg_name="orders")

        fig = orders.plot(column=column, **kwargs)
        if xref is None:
            xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        if yref is None:
            yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        if fit_sim_range:
            fig = cls_or_self.fit_fig_to_sim_range(
                fig,
                column=column,
                sim_start=sim_start,
                sim_end=sim_end,
                wrapper=wrapper,
                group_by=False,
                xref=xref,
            )
        return fig

    @hybrid_method
    def plot_trades(
        cls_or_self,
        column: tp.Optional[tp.Label] = None,
        trades: tp.Optional[Drawdowns] = None,
        trades_type: tp.Optional[tp.Union[str, int]] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        fit_sim_range: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        xref: str = "x",
        yref: str = "y",
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column of trades.

        Args:
            column (Optional[Label]): Column name to plot trades.
            trades (Optional[Drawdowns]): Trades instance for plotting.

                Uses `Portfolio.get_trades` if not provided.
            trades_type (Optional[Union[str, int]]): Identifier for the type of trades.
            sim_start (Optional[ArrayLike]): Simulation start index.
            sim_end (Optional[ArrayLike]): Simulation end index.
            rec_sim_range (bool): Flag indicating whether to record the simulation range.
            fit_sim_range (bool): Flag specifying whether to adjust the figure to the simulation range.
            wrapper (Optional[ArrayWrapper]): Array wrapper used for data transformations.
            xref (str): X-axis reference identifier.
            yref (str): Y-axis reference identifier.
            **kwargs: Additional keyword arguments passed to `vectorbtpro.portfolio.trades.Trades.plot`.

        Returns:
            BaseFigure: Plotly figure with the trades plot.
        """
        if not isinstance(cls_or_self, type):
            if trades is None:
                trades = cls_or_self.resolve_shortcut_attr(
                    "trades",
                    trades_type=trades_type,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                )
        else:
            checks.assert_not_none(trades, arg_name="trades")

        fig = trades.plot(column=column, xref=xref, yref=yref, **kwargs)
        if fit_sim_range:
            fig = cls_or_self.fit_fig_to_sim_range(
                fig,
                column=column,
                sim_start=sim_start,
                sim_end=sim_end,
                wrapper=wrapper,
                group_by=False,
                xref=xref,
            )
        return fig

    @hybrid_method
    def plot_trade_pnl(
        cls_or_self,
        column: tp.Optional[tp.Label] = None,
        trades: tp.Optional[Drawdowns] = None,
        trades_type: tp.Optional[tp.Union[str, int]] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        fit_sim_range: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        pct_scale: bool = False,
        xref: str = "x",
        yref: str = "y",
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column of trade P&L.

        Args:
            column (Optional[Label]): Column name to plot trade P&L.
            trades (Optional[Drawdowns]): Trades instance used for plotting trade P&L.

                Uses `Portfolio.get_trades` if not provided.
            trades_type (Optional[Union[str, int]]): Identifier for the type of trades.
            sim_start (Optional[ArrayLike]): Simulation start index.
            sim_end (Optional[ArrayLike]): Simulation end index.
            rec_sim_range (bool): Flag indicating whether to record the simulation range.
            fit_sim_range (bool): Flag specifying whether to adjust the figure to the simulation range.
            wrapper (Optional[ArrayWrapper]): Array wrapper used for data transformations.
            pct_scale (bool): Flag to display trade P&L on a percentage scale.
            xref (str): X-axis reference identifier.
            yref (str): Y-axis reference identifier.
            **kwargs: Additional keyword arguments passed to `vectorbtpro.portfolio.trades.Trades.plot_pnl`.

        Returns:
            BaseFigure: Plotly figure with the trade P&L plot.
        """
        if not isinstance(cls_or_self, type):
            if trades is None:
                trades = cls_or_self.resolve_shortcut_attr(
                    "trades",
                    trades_type=trades_type,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                )
        else:
            checks.assert_not_none(trades, arg_name="trades")

        fig = trades.plot_pnl(column=column, pct_scale=pct_scale, xref=xref, yref=yref, **kwargs)
        if fit_sim_range:
            fig = cls_or_self.fit_fig_to_sim_range(
                fig,
                column=column,
                sim_start=sim_start,
                sim_end=sim_end,
                wrapper=wrapper,
                group_by=False,
                xref=xref,
            )
        return fig

    @hybrid_method
    def plot_trade_signals(
        cls_or_self,
        column: tp.Optional[tp.Label] = None,
        entry_trades: tp.Optional[EntryTrades] = None,
        exit_trades: tp.Optional[ExitTrades] = None,
        positions: tp.Optional[Positions] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        fit_sim_range: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        plot_positions: tp.Union[bool, str] = "zones",
        long_entry_trace_kwargs: tp.KwargsLike = None,
        short_entry_trace_kwargs: tp.KwargsLike = None,
        long_exit_trace_kwargs: tp.KwargsLike = None,
        short_exit_trace_kwargs: tp.KwargsLike = None,
        long_shape_kwargs: tp.KwargsLike = None,
        short_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        xref: tp.Optional[str] = None,
        yref: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column or group of trade signals.

        Markers and shapes are colored by trade direction (green = long, red = short).

        Args:
            column (Optional[Label]): Column name or index to filter the trade signals.
            entry_trades (Optional[EntryTrades]): Entry trades data used for plotting.

                Uses `Portfolio.get_entry_trades` if not provided.
            exit_trades (Optional[ExitTrades]): Exit trades data used for plotting.

                Uses `Portfolio.get_exit_trades` if not provided.
            positions (Optional[Positions]): Positions data for plotting position shapes.
            
                Uses `Portfolio.get_positions` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start index.
            sim_end (Optional[ArrayLike]): Simulation end index.
            rec_sim_range (bool): Flag to record the simulation range.
            fit_sim_range (bool): If True, adjusts the figure to the simulation range.
            wrapper (Optional[ArrayWrapper]): Array wrapper for processing data.

                Uses `Portfolio.wrapper` if not provided.
            plot_positions (Union[bool, str]): Determines how to plot positions. Valid options:

                * "zones" or True
                * "lines"
            long_entry_trace_kwargs (KwargsLike): Additional keyword arguments for plotting long entry trade traces.
            short_entry_trace_kwargs (KwargsLike): Additional keyword arguments for plotting short entry trade traces.
            long_exit_trace_kwargs (KwargsLike): Additional keyword arguments for plotting long exit trade traces.
            short_exit_trace_kwargs (KwargsLike): Additional keyword arguments for plotting short exit trade traces.
            long_shape_kwargs (KwargsLike): Additional keyword arguments for customizing long position shapes.
            short_shape_kwargs (KwargsLike): Additional keyword arguments for customizing short position shapes.
            add_trace_kwargs (KwargsLike): Additional keyword arguments for adding traces.
            fig (Optional[BaseFigure]): Figure object to update; if None, a new figure is created.
            xref (Optional[str]): Reference for the x-axis to configure the plot layout.
            yref (Optional[str]): Reference for the y-axis to configure the plot layout.
            **kwargs: Additional keyword arguments passed to inner plotting functions.

        Returns:
            BaseFigure: Figure with plotted trade signals.
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if not isinstance(cls_or_self, type):
            if entry_trades is None:
                entry_trades = cls_or_self.resolve_shortcut_attr(
                    "entry_trades",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=False,
                )
            if exit_trades is None:
                exit_trades = cls_or_self.resolve_shortcut_attr(
                    "exit_trades",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=False,
                )
            if positions is None:
                positions = cls_or_self.resolve_shortcut_attr(
                    "positions",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=False,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(entry_trades, arg_name="entry_trades")
            checks.assert_not_none(exit_trades, arg_name="exit_trades")
            checks.assert_not_none(positions, arg_name="positions")
            if wrapper is None:
                wrapper = entry_trades.wrapper

        fig = entry_trades.plot_signals(
            column=column,
            long_entry_trace_kwargs=long_entry_trace_kwargs,
            short_entry_trace_kwargs=short_entry_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **kwargs,
        )
        fig = exit_trades.plot_signals(
            column=column,
            plot_ohlc=False,
            plot_close=False,
            long_exit_trace_kwargs=long_exit_trace_kwargs,
            short_exit_trace_kwargs=short_exit_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        if xref is None:
            xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        if yref is None:
            yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        if isinstance(plot_positions, bool):
            if plot_positions:
                plot_positions = "zones"
            else:
                plot_positions = None
        if plot_positions is not None:
            if plot_positions.lower() == "zones":
                long_shape_kwargs = merge_dicts(
                    dict(fillcolor=plotting_cfg["contrast_color_schema"]["green"]),
                    long_shape_kwargs,
                )
                short_shape_kwargs = merge_dicts(
                    dict(fillcolor=plotting_cfg["contrast_color_schema"]["red"]),
                    short_shape_kwargs,
                )
            elif plot_positions.lower() == "lines":
                base_shape_kwargs = dict(
                    type="line",
                    line=dict(dash="dot"),
                    xref=Rep("xref"),
                    yref=Rep("yref"),
                    x0=Rep("start_index"),
                    x1=Rep("end_index"),
                    y0=RepFunc(lambda record: record["entry_price"]),
                    y1=RepFunc(lambda record: record["exit_price"]),
                    opacity=0.75,
                )
                long_shape_kwargs = atomic_dict(
                    merge_dicts(
                        base_shape_kwargs,
                        dict(line=dict(color=plotting_cfg["contrast_color_schema"]["green"])),
                        long_shape_kwargs,
                    )
                )
                short_shape_kwargs = atomic_dict(
                    merge_dicts(
                        base_shape_kwargs,
                        dict(line=dict(color=plotting_cfg["contrast_color_schema"]["red"])),
                        short_shape_kwargs,
                    )
                )
            else:
                raise ValueError(f"Invalid plot_positions: '{plot_positions}'")
            fig = positions.direction_long.plot_shapes(
                column=column,
                plot_ohlc=False,
                plot_close=False,
                shape_kwargs=long_shape_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                xref=xref,
                yref=yref,
                fig=fig,
            )
            fig = positions.direction_short.plot_shapes(
                column=column,
                plot_ohlc=False,
                plot_close=False,
                shape_kwargs=short_shape_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                xref=xref,
                yref=yref,
                fig=fig,
            )
        if fit_sim_range:
            fig = cls_or_self.fit_fig_to_sim_range(
                fig,
                column=column,
                sim_start=sim_start,
                sim_end=sim_end,
                wrapper=wrapper,
                group_by=False,
                xref=xref,
            )
        return fig

    @hybrid_method
    def plot_cash_flow(
        cls_or_self,
        column: tp.Optional[tp.Label] = None,
        free: bool = False,
        cash_flow: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        fit_sim_range: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        line_shape: str = "hv",
        xref: tp.Optional[str] = None,
        yref: tp.Optional[str] = None,
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column or group of cash flow.

        Additional keyword arguments are passed to `vectorbtpro.generic.accessors.GenericAccessor.plot`.

        Args:
            column (Optional[Label]): Column label to select from the cash flow data.
            free (bool): Flag indicating whether to use free cash flow.
            cash_flow (Optional[SeriesFrame]): Cash flow data used for plotting.

                Uses `Portfolio.get_cash_flow` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start index.
            sim_end (Optional[ArrayLike]): Simulation end index.
            rec_sim_range (bool): Flag to record the simulation range.
            fit_sim_range (bool): If True, adjusts the figure to the simulation range.
            wrapper (Optional[ArrayWrapper]): Array wrapper for processing the cash flow data.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Grouping specification for selecting columns.
            line_shape (str): Shape of the plotted line (e.g., "hv" for horizontal-vertical lines).
            xref (Optional[str]): X-axis reference for layout; if not provided, it is inferred from the figure.
            yref (Optional[str]): Y-axis reference for layout; if not provided, it is inferred from the figure.
            hline_shape_kwargs (KwargsLike): Additional keyword arguments for customizing the horizontal line shape.
            **kwargs: Additional keyword arguments passed to `GenericAccessor.plot`.

        Returns:
            BaseFigure: Figure with the plotted cash flow.
        """
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if not isinstance(cls_or_self, type):
            if cash_flow is None:
                cash_flow = cls_or_self.resolve_shortcut_attr(
                    "cash_flow",
                    free=free,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(cash_flow, arg_name="cash_flow")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        cash_flow = wrapper.select_col_from_obj(cash_flow, column=column, group_by=group_by)
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(
                    line=dict(
                        color=plotting_cfg["color_schema"]["green"],
                        shape=line_shape,
                    ),
                    name="Cash",
                )
            ),
            kwargs,
        )
        fig = cash_flow.vbt.lineplot(**kwargs)
        if xref is None:
            xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        if yref is None:
            yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0.0,
                    x1=x_domain[1],
                    y1=0.0,
                ),
                hline_shape_kwargs,
            )
        )
        if fit_sim_range:
            fig = cls_or_self.fit_fig_to_sim_range(
                fig,
                column=column,
                sim_start=sim_start,
                sim_end=sim_end,
                wrapper=wrapper,
                group_by=group_by,
                xref=xref,
            )
        return fig

    @hybrid_method
    def plot_cash(
        cls_or_self,
        column: tp.Optional[tp.Label] = None,
        free: bool = False,
        init_cash: tp.Optional[tp.MaybeSeries] = None,
        cash: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        fit_sim_range: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        line_shape: str = "hv",
        xref: tp.Optional[str] = None,
        yref: tp.Optional[str] = None,
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column or group of cash balance.

        Args:
            column (Optional[Label]): Column label to select cash balance.
            free (bool): Whether to use free cash balance.
            init_cash (Optional[MaybeSeries]): Initial cash balance.

                Uses `Portfolio.get_init_cash` if not provided.
            cash (Optional[SeriesFrame]): Cash balance data.

                Uses `Portfolio.get_cash` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start time or index.
            sim_end (Optional[ArrayLike]): Simulation end time or index.
            rec_sim_range (bool): Record simulation range when resolving shortcuts.
            fit_sim_range (bool): Adjust the figure to simulation range if True.
            wrapper (Optional[ArrayWrapper]): Wrapper instance for data handling.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Grouping specification for column selection.
            line_shape (str): Line shape style for the plot.
            xref (Optional[str]): Reference for the x-axis.
            yref (Optional[str]): Reference for the y-axis.
            hline_shape_kwargs (KwargsLike): Additional keyword arguments for horizontal line shape.
            **kwargs (KwargsLike): Additional keyword arguments passed to 
                `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`.

        Returns:
            BaseFigure: Figure object with the depicted cash balance.
        """
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if not isinstance(cls_or_self, type):
            if init_cash is None:
                init_cash = cls_or_self.resolve_shortcut_attr(
                    "init_cash",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if cash is None:
                cash = cls_or_self.resolve_shortcut_attr(
                    "cash",
                    free=free,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_cash, arg_name="init_cash")
            checks.assert_not_none(cash, arg_name="cash")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        init_cash = wrapper.select_col_from_obj(init_cash, column=column, group_by=group_by)
        cash = wrapper.select_col_from_obj(cash, column=column, group_by=group_by)
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(
                    line=dict(
                        color=plotting_cfg["color_schema"]["green"],
                        shape=line_shape,
                    ),
                    name="Cash",
                ),
                pos_trace_kwargs=dict(
                    fillcolor=adjust_opacity(plotting_cfg["color_schema"]["green"], 0.3),
                    line=dict(shape=line_shape),
                ),
                neg_trace_kwargs=dict(
                    fillcolor=adjust_opacity(plotting_cfg["color_schema"]["orange"], 0.3),
                    line=dict(shape=line_shape),
                ),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = cash.vbt.plot_against(init_cash, **kwargs)
        if xref is None:
            xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        if yref is None:
            yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=init_cash,
                    x1=x_domain[1],
                    y1=init_cash,
                ),
                hline_shape_kwargs,
            )
        )
        if fit_sim_range:
            fig = cls_or_self.fit_fig_to_sim_range(
                fig,
                column=column,
                sim_start=sim_start,
                sim_end=sim_end,
                wrapper=wrapper,
                group_by=group_by,
                xref=xref,
            )
        return fig

    @hybrid_method
    def plot_asset_flow(
        cls_or_self,
        column: tp.Optional[tp.Label] = None,
        direction: tp.Union[str, int] = "both",
        asset_flow: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        fit_sim_range: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        line_shape: str = "hv",
        xref: tp.Optional[str] = None,
        yref: tp.Optional[str] = None,
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column of asset flow.

        Args:
            column (Optional[Label]): Column label to select asset flow.
            direction (Union[str, int]): Direction filter for asset flow.

                See `vectorbtpro.portfolio.enums.Direction` for options.
            asset_flow (Optional[SeriesFrame]): Asset flow data.

                Uses `Portfolio.get_asset_flow` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start time or index.
            sim_end (Optional[ArrayLike]): Simulation end time or index.
            rec_sim_range (bool): Record simulation range when resolving shortcuts.
            fit_sim_range (bool): Adjust the figure to simulation range if True.
            wrapper (Optional[ArrayWrapper]): Wrapper instance for data handling.

                Uses `Portfolio.wrapper` if not provided.
            line_shape (str): Line shape style for the plot.
            xref (Optional[str]): Reference for the x-axis.
            yref (Optional[str]): Reference for the y-axis.
            hline_shape_kwargs (KwargsLike): Additional keyword arguments for horizontal line shape.
            **kwargs (KwargsLike): Additional keyword arguments passed to 
                `vectorbtpro.generic.accessors.GenericAccessor.plot`.

        Returns:
            BaseFigure: Figure object with the depicted asset flow.
        """
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if not isinstance(cls_or_self, type):
            if asset_flow is None:
                asset_flow = cls_or_self.resolve_shortcut_attr(
                    "asset_flow",
                    direction=direction,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(asset_flow, arg_name="asset_flow")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        asset_flow = wrapper.select_col_from_obj(asset_flow, column=column, group_by=False)
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(
                    line=dict(
                        color=plotting_cfg["color_schema"]["blue"],
                        shape=line_shape,
                    ),
                    name="Assets",
                )
            ),
            kwargs,
        )
        fig = asset_flow.vbt.lineplot(**kwargs)
        if xref is None:
            xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        if yref is None:
            yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0,
                    x1=x_domain[1],
                    y1=0,
                ),
                hline_shape_kwargs,
            )
        )
        if fit_sim_range:
            fig = cls_or_self.fit_fig_to_sim_range(
                fig,
                column=column,
                sim_start=sim_start,
                sim_end=sim_end,
                wrapper=wrapper,
                group_by=False,
                xref=xref,
            )
        return fig

    @hybrid_method
    def plot_assets(
        cls_or_self,
        column: tp.Optional[tp.Label] = None,
        direction: tp.Union[str, int] = "both",
        assets: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        fit_sim_range: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        line_shape: str = "hv",
        xref: tp.Optional[str] = None,
        yref: tp.Optional[str] = None,
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column of asset data.

        Args:
            column (Optional[Label]): Column label to select from asset data.
            direction (Union[str, int]): Asset direction indicator (e.g. "both").

                See `vectorbtpro.portfolio.enums.Direction` for options.
            assets (Optional[SeriesFrame]): Asset data to plot.

                Uses `Portfolio.get_assets` if not provided.
            sim_start (Optional[ArrayLike]): Start of the simulation range.
            sim_end (Optional[ArrayLike]): End of the simulation range.
            rec_sim_range (bool): Flag indicating whether to use the recorded simulation range.
            fit_sim_range (bool): Flag indicating whether to adjust the plot to the simulation range.
            wrapper (Optional[ArrayWrapper]): Wrapper instance for processing asset data.

                Uses `Portfolio.wrapper` if not provided.
            line_shape (str): Shape of the plot line (e.g. "hv").
            xref (Optional[str]): Reference for the x-axis. If None, inferred from the figure.
            yref (Optional[str]): Reference for the y-axis. If None, inferred from the figure.
            hline_shape_kwargs (KwargsLike): Additional keyword arguments for 
                customizing the horizontal line shape.
            **kwargs: Additional keyword arguments for 
                `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`.

        Returns:
            BaseFigure: Figure object with the plotted asset data.
        """
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if not isinstance(cls_or_self, type):
            if assets is None:
                assets = cls_or_self.resolve_shortcut_attr(
                    "assets",
                    direction=direction,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(assets, arg_name="assets")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        assets = wrapper.select_col_from_obj(assets, column=column, group_by=False)
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(
                    line=dict(
                        color=plotting_cfg["color_schema"]["blue"],
                        shape=line_shape,
                    ),
                    name="Assets",
                ),
                pos_trace_kwargs=dict(
                    fillcolor=adjust_opacity(plotting_cfg["color_schema"]["blue"], 0.3),
                    line=dict(shape=line_shape),
                ),
                neg_trace_kwargs=dict(
                    fillcolor=adjust_opacity(plotting_cfg["color_schema"]["orange"], 0.3),
                    line=dict(shape=line_shape),
                ),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = assets.vbt.plot_against(0, **kwargs)
        if xref is None:
            xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        if yref is None:
            yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0.0,
                    x1=x_domain[1],
                    y1=0.0,
                ),
                hline_shape_kwargs,
            )
        )
        if fit_sim_range:
            fig = cls_or_self.fit_fig_to_sim_range(
                fig,
                column=column,
                sim_start=sim_start,
                sim_end=sim_end,
                wrapper=wrapper,
                group_by=False,
                xref=xref,
            )
        return fig

    @hybrid_method
    def plot_asset_value(
        cls_or_self,
        column: tp.Optional[tp.Label] = None,
        direction: tp.Union[str, int] = "both",
        asset_value: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        fit_sim_range: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        line_shape: str = "hv",
        xref: tp.Optional[str] = None,
        yref: tp.Optional[str] = None,
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot asset value data for a single column or a group of columns.

        Args:
            column (Optional[Label]): Column label to select from asset value data.
            direction (Union[str, int]): Direction indicator for asset value data (e.g. "both").

                See `vectorbtpro.portfolio.enums.Direction` for options.
            asset_value (Optional[SeriesFrame]): Asset value data to plot.

                Uses `Portfolio.get_asset_value` if not provided.
            sim_start (Optional[ArrayLike]): Start of the simulation range.
            sim_end (Optional[ArrayLike]): End of the simulation range.
            rec_sim_range (bool): Flag indicating whether to use the recorded simulation range.
            fit_sim_range (bool): Flag indicating whether to adjust the plot to the simulation range.
            wrapper (Optional[ArrayWrapper]): Wrapper instance for processing asset value data.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Grouping specification for column selection.
            line_shape (str): Shape of the plot line (e.g. "hv").
            xref (Optional[str]): Reference for the x-axis. If None, inferred from the figure.
            yref (Optional[str]): Reference for the y-axis. If None, inferred from the figure.
            hline_shape_kwargs (KwargsLike): Additional keyword arguments for 
                customizing the horizontal line shape.
            **kwargs: Additional keyword arguments for 
                `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`.

        Returns:
            BaseFigure: Figure object with the plotted asset value data.
        """
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if not isinstance(cls_or_self, type):
            if asset_value is None:
                asset_value = cls_or_self.resolve_shortcut_attr(
                    "asset_value",
                    direction=direction,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(asset_value, arg_name="asset_value")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        asset_value = wrapper.select_col_from_obj(asset_value, column=column, group_by=group_by)
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(
                    line=dict(
                        color=plotting_cfg["color_schema"]["purple"],
                        shape=line_shape,
                    ),
                    name="Value",
                ),
                pos_trace_kwargs=dict(
                    fillcolor=adjust_opacity(plotting_cfg["color_schema"]["purple"], 0.3),
                    line=dict(shape=line_shape),
                ),
                neg_trace_kwargs=dict(
                    fillcolor=adjust_opacity(plotting_cfg["color_schema"]["orange"], 0.3),
                    line=dict(shape=line_shape),
                ),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = asset_value.vbt.plot_against(0, **kwargs)
        if xref is None:
            xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        if yref is None:
            yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0.0,
                    x1=x_domain[1],
                    y1=0.0,
                ),
                hline_shape_kwargs,
            )
        )
        if fit_sim_range:
            fig = cls_or_self.fit_fig_to_sim_range(
                fig,
                column=column,
                sim_start=sim_start,
                sim_end=sim_end,
                wrapper=wrapper,
                group_by=group_by,
                xref=xref,
            )
        return fig

    @hybrid_method
    def plot_value(
        cls_or_self,
        column: tp.Optional[tp.Label] = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        value: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        fit_sim_range: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        xref: tp.Optional[str] = None,
        yref: tp.Optional[str] = None,
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column or group of value.

        Args:
            column (Optional[Label]): Column label for selecting the data.
            init_value (Optional[MaybeSeries]): Initial value data.

                Uses `Portfolio.get_init_value` if not provided.
            value (Optional[SeriesFrame]): Value data to plot.

                Uses `Portfolio.get_value` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start index or date.
            sim_end (Optional[ArrayLike]): Simulation end index or date.
            rec_sim_range (bool): If True, pass simulation range parameters when resolving the initial value.
            fit_sim_range (bool): If True, adjust the figure range to match the simulation period.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance for column selection.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Grouping specification to aggregate data.
            xref (Optional[str]): x-axis reference identifier. 
            
                If None, it is inferred from the figure.
            yref (Optional[str]): y-axis reference identifier. 
            
                If None, it is inferred from the figure.
            hline_shape_kwargs (KwargsLike): Additional keyword arguments for 
                the horizontal line shape.
            **kwargs: Additional keyword arguments passed to 
                `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`.

        Returns:
            BaseFigure: The figure displaying the plotted value.
        """
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if not isinstance(cls_or_self, type):
            if init_value is None:
                init_value = cls_or_self.resolve_shortcut_attr(
                    "init_value",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if value is None:
                value = cls_or_self.resolve_shortcut_attr(
                    "value",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_value, arg_name="init_value")
            checks.assert_not_none(value, arg_name="value")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        init_value = wrapper.select_col_from_obj(init_value, column=column, group_by=group_by)
        value = wrapper.select_col_from_obj(value, column=column, group_by=group_by)
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(
                    line=dict(
                        color=plotting_cfg["color_schema"]["purple"],
                    ),
                    name="Value",
                ),
                pos_trace_kwargs=dict(
                    fillcolor=adjust_opacity(plotting_cfg["color_schema"]["purple"], 0.3),
                ),
                neg_trace_kwargs=dict(
                    fillcolor=adjust_opacity(plotting_cfg["color_schema"]["red"], 0.3),
                ),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = value.vbt.plot_against(init_value, **kwargs)
        if xref is None:
            xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        if yref is None:
            yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=init_value,
                    x1=x_domain[1],
                    y1=init_value,
                ),
                hline_shape_kwargs,
            )
        )
        if fit_sim_range:
            fig = cls_or_self.fit_fig_to_sim_range(
                fig,
                column=column,
                sim_start=sim_start,
                sim_end=sim_end,
                wrapper=wrapper,
                group_by=group_by,
                xref=xref,
            )
        return fig

    @hybrid_method
    def plot_cumulative_returns(
        cls_or_self,
        column: tp.Optional[tp.Label] = None,
        returns_acc: tp.Optional[ReturnsAccessor] = None,
        use_asset_returns: bool = False,
        bm_returns: tp.Union[None, bool, tp.ArrayLike] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        fit_sim_range: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        pct_scale: bool = False,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column or group of cumulative returns.

        Args:
            column (Optional[Label]): Column label for selecting cumulative returns data.
            returns_acc (Optional[ReturnsAccessor]): Returns accessor instance.

                Uses `Portfolio.get_returns_acc` if not provided.
            use_asset_returns (bool): Flag indicating whether to use asset returns instead of portfolio returns.
            bm_returns (Union[None, bool, ArrayLike]): Benchmark returns specification.
            sim_start (Optional[ArrayLike]): Simulation start index or date.
            sim_end (Optional[ArrayLike]): Simulation end index or date.
            rec_sim_range (bool): If True, pass simulation range parameters when resolving returns.
            fit_sim_range (bool): If True, adjust the figure range to match the simulation period.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance used for data extraction.
            group_by (GroupByLike): Grouping specification to aggregate data.
            pct_scale (bool): Flag indicating whether to scale the returns as percentages.
            **kwargs: Additional keyword arguments passed to 
                `vectorbtpro.returns.accessors.ReturnsSRAccessor.plot_cumulative`.

        Returns:
            BaseFigure: The figure displaying cumulative returns.
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if not isinstance(cls_or_self, type):
            if returns_acc is None:
                returns_acc = cls_or_self.resolve_shortcut_attr(
                    "returns_acc",
                    use_asset_returns=use_asset_returns,
                    bm_returns=bm_returns,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
        else:
            checks.assert_not_none(returns_acc, arg_name="returns_acc")
        kwargs = merge_dicts(
            dict(
                main_kwargs=dict(
                    trace_kwargs=dict(name="Value"),
                    pos_trace_kwargs=dict(
                        fillcolor=adjust_opacity(plotting_cfg["color_schema"]["purple"], 0.3),
                    ),
                    neg_trace_kwargs=dict(
                        fillcolor=adjust_opacity(plotting_cfg["color_schema"]["red"], 0.3),
                    ),
                ),
                hline_shape_kwargs=dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                ),
            ),
            kwargs,
        )
        fig = returns_acc.plot_cumulative(
            column=column,
            fit_sim_range=fit_sim_range,
            pct_scale=pct_scale,
            **kwargs,
        )
        return fig

    @hybrid_method
    def plot_drawdowns(
        cls_or_self,
        column: tp.Optional[tp.Label] = None,
        drawdowns: tp.Optional[Drawdowns] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        fit_sim_range: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        xref: str = "x",
        yref: str = "y",
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot one column or group of drawdowns.

        Args:
            column (Optional[Label]): Column label for selecting drawdown data.
            drawdowns (Optional[Drawdowns]): Drawdowns instance.
            
                Uses `Portfolio.get_drawdowns` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start index or date.
            sim_end (Optional[ArrayLike]): Simulation end index or date.
            rec_sim_range (bool): If True, pass simulation range parameters when resolving drawdowns.
            fit_sim_range (bool): If True, adjust the figure range to match the simulation period.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance used for data processing.
            group_by (GroupByLike): Grouping specification to aggregate data.
            xref (str): x-axis reference identifier.
            yref (str): y-axis reference identifier.
            **kwargs: Additional keyword arguments passed to 
                `vectorbtpro.generic.drawdowns.Drawdowns.plot`.

        Returns:
            BaseFigure: The figure displaying the plotted drawdowns.
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if not isinstance(cls_or_self, type):
            if drawdowns is None:
                drawdowns = cls_or_self.resolve_shortcut_attr(
                    "drawdowns",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
        else:
            checks.assert_not_none(drawdowns, arg_name="drawdowns")

        kwargs = merge_dicts(
            dict(
                close_trace_kwargs=dict(
                    line=dict(
                        color=plotting_cfg["color_schema"]["purple"],
                    ),
                    name="Value",
                ),
            ),
            kwargs,
        )
        fig = drawdowns.plot(column=column, xref=xref, yref=yref, **kwargs)
        if fit_sim_range:
            fig = cls_or_self.fit_fig_to_sim_range(
                fig,
                column=column,
                sim_start=sim_start,
                sim_end=sim_end,
                wrapper=wrapper,
                group_by=group_by,
                xref=xref,
            )
        return fig

    @hybrid_method
    def plot_underwater(
        cls_or_self,
        column: tp.Optional[tp.Label] = None,
        init_value: tp.Optional[tp.MaybeSeries] = None,
        returns_acc: tp.Optional[ReturnsAccessor] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        fit_sim_range: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        pct_scale: bool = True,
        xref: str = "x",
        yref: str = "y",
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot underwater for a specified column or group.

        Args:
            column (Optional[Label]): Column label for selecting a specific sub-data.
            init_value (Optional[MaybeSeries]): Initial portfolio value used for 
                scaling when `pct_scale` is False.

                Uses `Portfolio.get_init_value` if not provided.
            returns_acc (Optional[ReturnsAccessor]): Returns accessor instance to compute drawdowns.

                Uses `Portfolio.get_returns_acc` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start date or index.
            sim_end (Optional[ArrayLike]): Simulation end date or index.
            rec_sim_range (bool): Flag to record the simulation range when resolving attributes.
            fit_sim_range (bool): Flag to adjust the figure to fit the simulation range.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance for data manipulation.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Specification for grouping data.
            pct_scale (bool): If True, format the y-axis ticks as percentages.
            xref (str): Reference identifier for the x-axis.
            yref (str): Reference identifier for the y-axis.
            hline_shape_kwargs (KwargsLike): Additional keyword arguments for 
                customizing the horizontal line shape.
            **kwargs: Additional keyword arguments passed to 
            `vectorbtpro.generic.accessors.GenericAccessor.plot`.

        Returns:
            BaseFigure: A figure object containing the underwater plot.
        """
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if not isinstance(cls_or_self, type):
            if init_value is None:
                init_value = cls_or_self.resolve_shortcut_attr(
                    "init_value",
                    sim_start=sim_start if rec_sim_range else None,
                    sim_end=sim_end if rec_sim_range else None,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if returns_acc is None:
                returns_acc = cls_or_self.resolve_shortcut_attr(
                    "returns_acc",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(init_value, arg_name="init_value")
            checks.assert_not_none(returns_acc, arg_name="returns_acc")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        drawdown = returns_acc.drawdown()
        drawdown = wrapper.select_col_from_obj(drawdown, column=column, group_by=group_by)
        if not pct_scale:
            cumulative_returns = returns_acc.cumulative()
            cumret = wrapper.select_col_from_obj(cumulative_returns, column=column, group_by=group_by)
            init_value = wrapper.select_col_from_obj(init_value, column=column, group_by=group_by)
            drawdown = cumret * init_value * drawdown / (1 + drawdown)
        default_kwargs = dict(
            trace_kwargs=dict(
                line=dict(color=plotting_cfg["color_schema"]["red"]),
                fillcolor=adjust_opacity(plotting_cfg["color_schema"]["red"], 0.3),
                fill="tozeroy",
                name="Drawdown",
            )
        )
        if pct_scale:
            yaxis = "yaxis" + yref[1:]
            default_kwargs[yaxis] = dict(tickformat=".2%")
        kwargs = merge_dicts(default_kwargs, kwargs)
        fig = drawdown.vbt.lineplot(**kwargs)
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=0,
                    x1=x_domain[1],
                    y1=0,
                ),
                hline_shape_kwargs,
            )
        )
        if fit_sim_range:
            fig = cls_or_self.fit_fig_to_sim_range(
                fig,
                column=column,
                sim_start=sim_start,
                sim_end=sim_end,
                wrapper=wrapper,
                group_by=group_by,
                xref=xref,
            )
        return fig

    @hybrid_method
    def plot_gross_exposure(
        cls_or_self,
        column: tp.Optional[tp.Label] = None,
        direction: tp.Union[str, int] = "both",
        gross_exposure: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        fit_sim_range: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        line_shape: str = "hv",
        xref: tp.Optional[str] = None,
        yref: tp.Optional[str] = None,
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot gross exposure for a specified column or group.

        Args:
            column (Optional[Label]): Column label for selecting a specific sub-data.
            direction (Union[str, int]): Indicator for the exposure direction 
                (e.g., "both" for combined exposure).

                See `vectorbtpro.portfolio.enums.Direction` for options.
            gross_exposure (Optional[SeriesFrame]): Gross exposure data series.

                Uses `Portfolio.get_gross_exposure` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start date or index.
            sim_end (Optional[ArrayLike]): Simulation end date or index.
            rec_sim_range (bool): Flag to record the simulation range when resolving attributes.
            fit_sim_range (bool): Flag to adjust the figure to fit the simulation range.
            wrapper (Optional[ArrayWrapper]): Array wrapper instance for data manipulation.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Specification for grouping data.
            line_shape (str): Plot line shape style, such as "hv".
            xref (Optional[str]): Reference identifier for the x-axis.
            yref (Optional[str]): Reference identifier for the y-axis.
            hline_shape_kwargs (KwargsLike): Additional keyword arguments for 
                customizing the horizontal line shape.
            **kwargs: Additional keyword arguments passed to 
                `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`.

        Returns:
            BaseFigure: A figure object containing the gross exposure plot.
        """
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if not isinstance(cls_or_self, type):
            if gross_exposure is None:
                gross_exposure = cls_or_self.resolve_shortcut_attr(
                    "gross_exposure",
                    direction=direction,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(gross_exposure, arg_name="gross_exposure")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        gross_exposure = wrapper.select_col_from_obj(gross_exposure, column=column, group_by=group_by)
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(
                    line=dict(
                        color=plotting_cfg["color_schema"]["pink"],
                        shape=line_shape,
                    ),
                    name="Exposure",
                ),
                pos_trace_kwargs=dict(
                    fillcolor=adjust_opacity(plotting_cfg["color_schema"]["orange"], 0.3),
                    line=dict(shape=line_shape),
                ),
                neg_trace_kwargs=dict(
                    fillcolor=adjust_opacity(plotting_cfg["color_schema"]["pink"], 0.3),
                    line=dict(shape=line_shape),
                ),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = gross_exposure.vbt.plot_against(1, **kwargs)
        if xref is None:
            xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        if yref is None:
            yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=1,
                    x1=x_domain[1],
                    y1=1,
                ),
                hline_shape_kwargs,
            )
        )
        if fit_sim_range:
            fig = cls_or_self.fit_fig_to_sim_range(
                fig,
                column=column,
                sim_start=sim_start,
                sim_end=sim_end,
                wrapper=wrapper,
                group_by=group_by,
                xref=xref,
            )
        return fig

    @hybrid_method
    def plot_net_exposure(
        cls_or_self,
        column: tp.Optional[tp.Label] = None,
        net_exposure: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        fit_sim_range: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        line_shape: str = "hv",
        xref: tp.Optional[str] = None,
        yref: tp.Optional[str] = None,
        hline_shape_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot net exposure for a specified column or group.

        Args:
            column (Optional[Label]): Column label to select net exposure data.
            net_exposure (Optional[SeriesFrame]): Net exposure data.

                Uses `Portfolio.get_net_exposure` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start time.
            sim_end (Optional[ArrayLike]): Simulation end time.
            rec_sim_range (bool): Flag to recalculate simulation range.
            fit_sim_range (bool): Flag to fit the figure to the simulation range.
            wrapper (Optional[ArrayWrapper]): Data wrapper for managing exposures.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Grouping specification for the data.
            line_shape (str): Shape of the plot line (e.g., "hv").
            xref (Optional[str]): Reference for the x-axis; determined from the plot if not provided.
            yref (Optional[str]): Reference for the y-axis; determined from the plot if not provided.
            hline_shape_kwargs (KwargsLike): Additional keyword arguments for 
                configuring the horizontal line shape.
            **kwargs (KwargsLike): Additional keyword arguments passed to 
                `vectorbtpro.generic.accessors.GenericSRAccessor.plot_against`.

        Returns:
            BaseFigure: Figure instance representing the net exposure plot.
        """
        from vectorbtpro.utils.figure import get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if not isinstance(cls_or_self, type):
            if net_exposure is None:
                net_exposure = cls_or_self.resolve_shortcut_attr(
                    "net_exposure",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(net_exposure, arg_name="net_exposure")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        net_exposure = wrapper.select_col_from_obj(net_exposure, column=column, group_by=group_by)
        kwargs = merge_dicts(
            dict(
                trace_kwargs=dict(
                    line=dict(
                        color=plotting_cfg["color_schema"]["pink"],
                        shape=line_shape,
                    ),
                    name="Exposure",
                ),
                pos_trace_kwargs=dict(
                    fillcolor=adjust_opacity(plotting_cfg["color_schema"]["orange"], 0.3),
                    line=dict(shape=line_shape),
                ),
                neg_trace_kwargs=dict(
                    fillcolor=adjust_opacity(plotting_cfg["color_schema"]["pink"], 0.3),
                    line=dict(shape=line_shape),
                ),
                other_trace_kwargs="hidden",
            ),
            kwargs,
        )
        fig = net_exposure.vbt.plot_against(1, **kwargs)
        if xref is None:
            xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        if yref is None:
            yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        x_domain = get_domain(xref, fig)
        fig.add_shape(
            **merge_dicts(
                dict(
                    type="line",
                    line=dict(
                        color="gray",
                        dash="dash",
                    ),
                    xref="paper",
                    yref=yref,
                    x0=x_domain[0],
                    y0=1,
                    x1=x_domain[1],
                    y1=1,
                ),
                hline_shape_kwargs,
            )
        )
        if fit_sim_range:
            fig = cls_or_self.fit_fig_to_sim_range(
                fig,
                column=column,
                sim_start=sim_start,
                sim_end=sim_end,
                wrapper=wrapper,
                group_by=group_by,
                xref=xref,
            )
        return fig

    @hybrid_method
    def plot_allocations(
        cls_or_self,
        column: tp.Optional[tp.Label] = None,
        allocations: tp.Optional[tp.SeriesFrame] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        rec_sim_range: bool = False,
        fit_sim_range: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        line_shape: str = "hv",
        line_visible: bool = True,
        colorway: tp.Union[None, str, tp.Sequence[str]] = "Vivid",
        xref: tp.Optional[str] = None,
        yref: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.BaseFigure:
        """Plot allocations for a specified group.

        Args:
            column (Optional[Label]): Column label to select allocation data.
            allocations (Optional[SeriesFrame]): Allocation data.

                Uses `Portfolio.get_allocations` if not provided.
            sim_start (Optional[ArrayLike]): Simulation start time for the plot.
            sim_end (Optional[ArrayLike]): Simulation end time for the plot.
            rec_sim_range (bool): Flag to recalculate simulation range.
            fit_sim_range (bool): Flag to adjust the figure to the simulation range.
            wrapper (Optional[ArrayWrapper]): Data wrapper for allocations.

                Uses `Portfolio.wrapper` if not provided.
            group_by (GroupByLike): Grouping specification for the data.
            line_shape (str): Shape of the plot lines (e.g., "hv").
            line_visible (bool): Determines if plot lines are visible.
            colorway (Union[None, str, Sequence[str]]): Color scheme used for the plot.
            xref (Optional[str]): Reference for the x-axis; determined from the plot if not provided.
            yref (Optional[str]): Reference for the y-axis; determined from the plot if not provided.
            **kwargs (KwargsLike): Additional keyword arguments passed to 
                `vectorbtpro.generic.accessors.GenericAccessor.areaplot`.

        Returns:
            BaseFigure: Figure instance representing the allocations plot.
        """
        if not isinstance(cls_or_self, type):
            if allocations is None:
                allocations = cls_or_self.resolve_shortcut_attr(
                    "allocations",
                    sim_start=sim_start,
                    sim_end=sim_end,
                    rec_sim_range=rec_sim_range,
                    wrapper=wrapper,
                    group_by=group_by,
                )
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(allocations, arg_name="allocations")
            checks.assert_not_none(wrapper, arg_name="wrapper")

        allocations = wrapper.select_col_from_obj(allocations, column=column, obj_ungrouped=True, group_by=group_by)
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_names = wrapper.grouper.get_index(group_by=group_by).names
            allocations = allocations.vbt.drop_levels(group_names, strict=False)
        if isinstance(allocations, pd.Series) and allocations.name is None:
            allocations.name = "Allocation"
        fig = allocations.vbt.areaplot(line_shape=line_shape, line_visible=line_visible, colorway=colorway, **kwargs)
        if xref is None:
            xref = fig.data[-1]["xaxis"] if fig.data[-1]["xaxis"] is not None else "x"
        if yref is None:
            yref = fig.data[-1]["yaxis"] if fig.data[-1]["yaxis"] is not None else "y"
        if fit_sim_range:
            fig = cls_or_self.fit_fig_to_sim_range(
                fig,
                column=column,
                sim_start=sim_start,
                sim_end=sim_end,
                wrapper=wrapper,
                group_by=group_by,
                xref=xref,
            )
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Portfolio.plot`.

        Merges `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults` and
        `plots` from `vectorbtpro._settings.portfolio`.

        Returns:
            Kwargs: Merged configuration dictionary for plotting.
        """
        from vectorbtpro._settings import settings

        portfolio_plots_cfg = settings["portfolio"]["plots"]

        return merge_dicts(
            Analyzable.plots_defaults.__get__(self),
            dict(settings=dict(trades_type=self.trades_type)),
            portfolio_plots_cfg,
        )

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            orders=dict(
                title="Orders",
                yaxis_kwargs=dict(title="Price"),
                check_is_not_grouped=True,
                plot_func="plot_orders",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "orders"],
            ),
            trades=dict(
                title="Trades",
                yaxis_kwargs=dict(title="Price"),
                check_is_not_grouped=True,
                plot_func="plot_trades",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "trades"],
            ),
            trade_pnl=dict(
                title="Trade PnL",
                yaxis_kwargs=dict(title="PnL"),
                check_is_not_grouped=True,
                plot_func="plot_trade_pnl",
                pct_scale=True,
                pass_add_trace_kwargs=True,
                tags=["portfolio", "trades"],
            ),
            trade_signals=dict(
                title="Trade Signals",
                yaxis_kwargs=dict(title="Price"),
                check_is_not_grouped=True,
                plot_func="plot_trade_signals",
                tags=["portfolio", "trades"],
            ),
            cash_flow=dict(
                title="Cash Flow",
                yaxis_kwargs=dict(title="Amount"),
                plot_func="plot_cash_flow",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "cash"],
            ),
            cash=dict(
                title="Cash",
                yaxis_kwargs=dict(title="Amount"),
                plot_func="plot_cash",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "cash"],
            ),
            asset_flow=dict(
                title="Asset Flow",
                yaxis_kwargs=dict(title="Amount"),
                check_is_not_grouped=True,
                plot_func="plot_asset_flow",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "assets"],
            ),
            assets=dict(
                title="Assets",
                yaxis_kwargs=dict(title="Amount"),
                check_is_not_grouped=True,
                plot_func="plot_assets",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "assets"],
            ),
            asset_value=dict(
                title="Asset Value",
                yaxis_kwargs=dict(title="Value"),
                plot_func="plot_asset_value",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "assets", "value"],
            ),
            value=dict(
                title="Value",
                yaxis_kwargs=dict(title="Value"),
                plot_func="plot_value",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "value"],
            ),
            cumulative_returns=dict(
                title="Cumulative Returns",
                yaxis_kwargs=dict(title="Cumulative return"),
                plot_func="plot_cumulative_returns",
                pass_hline_shape_kwargs=True,
                pass_add_trace_kwargs=True,
                pass_xref=True,
                pass_yref=True,
                tags=["portfolio", "returns"],
            ),
            drawdowns=dict(
                title="Drawdowns",
                yaxis_kwargs=dict(title="Value"),
                plot_func="plot_drawdowns",
                pass_add_trace_kwargs=True,
                pass_xref=True,
                pass_yref=True,
                tags=["portfolio", "value", "drawdowns"],
            ),
            underwater=dict(
                title="Underwater",
                yaxis_kwargs=dict(title="Drawdown"),
                plot_func="plot_underwater",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "value", "drawdowns"],
            ),
            gross_exposure=dict(
                title="Gross Exposure",
                yaxis_kwargs=dict(title="Exposure"),
                plot_func="plot_gross_exposure",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "exposure"],
            ),
            net_exposure=dict(
                title="Net Exposure",
                yaxis_kwargs=dict(title="Exposure"),
                plot_func="plot_net_exposure",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "exposure"],
            ),
            allocations=dict(
                title="Allocations",
                yaxis_kwargs=dict(title="Allocation"),
                plot_func="plot_allocations",
                pass_add_trace_kwargs=True,
                tags=["portfolio", "allocations"],
            ),
        )
    )

    plot = Analyzable.plots

    @property
    def subplots(self) -> Config:
        return self._subplots

    # ############# Docs ############# #

    @classmethod
    def build_in_output_config_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """Build in-output configuration documentation.

        Args:
            source_cls (Optional[type]): Source class to obtain the original `in_output_config` documentation.

                If None, `Portfolio` is used.

        Returns:
            str: The generated in-output configuration documentation.
        """
        if source_cls is None:
            source_cls = Portfolio
        return string.Template(inspect.cleandoc(get_dict_attr(source_cls, "in_output_config").__doc__)).substitute(
            {"in_output_config": cls.in_output_config.prettify_doc(), "cls_name": cls.__name__},
        )

    @classmethod
    def override_in_output_config_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """Override the in-output configuration documentation in the __pdoc__ dictionary.

        Args:
            __pdoc__ (dict): Dictionary for storing module-level documentation.
            source_cls (Optional[type]): Source class providing the original `in_output_config` documentation.

                If None, `Portfolio` is used.

        Returns:
            None
        """
        __pdoc__[cls.__name__ + ".in_output_config"] = cls.build_in_output_config_doc(source_cls=source_cls)


Portfolio.override_in_output_config_doc(__pdoc__)
Portfolio.override_metrics_doc(__pdoc__)
Portfolio.override_subplots_doc(__pdoc__)

__pdoc__["Portfolio.plot"] = """Refer to `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` 
for additional details."""

PF = Portfolio
"""Alias for the `Portfolio` class."""

__pdoc__["PF"] = False
