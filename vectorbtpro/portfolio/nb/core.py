# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing core Numba-compiled functions for portfolio simulation."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb, flex_select_nb
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.portfolio.enums import *
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils.math_ import is_close_nb, is_close_or_less_nb, is_close_or_greater_nb, is_less_nb, add_nb


@register_jitted(cache=True)
def order_not_filled_nb(status: int, status_info: int) -> OrderResult:
    """Return an `vectorbtpro.portfolio.enums.OrderResult` for an order that has not been filled.

    Args:
        status (int): The order status code.
        status_info (int): Additional status information code.

    Returns:
        OrderResult: An order result with NaN size, price, and fees, a side of -1,
            and the provided status and status_info.
    """
    return OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=status, status_info=status_info)


@register_jitted(cache=True)
def check_adj_price_nb(
    adj_price: float,
    price_area: PriceArea,
    is_closing_price: bool,
    price_area_vio_mode: int,
) -> float:
    """Adjust the given price to ensure it falls within the specified price boundaries.

    Args:
        adj_price (float): The initial adjusted price.
        price_area (PriceArea): An object defining price boundaries.
        is_closing_price (bool): If True, enforce the price to match the closing price.

            If the adjusted price differs from `PriceArea.close`, it may trigger an error or cap.
        price_area_vio_mode (int): Mode for handling price violations.

            Expected values are defined in `PriceAreaVioMode`.

    Returns:
        float: The validated and, if necessary, adjusted price.
    """
    if price_area_vio_mode == PriceAreaVioMode.Ignore:
        return adj_price
    if adj_price > price_area.high:
        if price_area_vio_mode == PriceAreaVioMode.Error:
            raise ValueError("Adjusted order price is above the highest price")
        elif price_area_vio_mode == PriceAreaVioMode.Cap:
            adj_price = price_area.high
    if adj_price < price_area.low:
        if price_area_vio_mode == PriceAreaVioMode.Error:
            raise ValueError("Adjusted order price is below the lowest price")
        elif price_area_vio_mode == PriceAreaVioMode.Cap:
            adj_price = price_area.low
    if is_closing_price and adj_price != price_area.close:
        if price_area_vio_mode == PriceAreaVioMode.Error:
            raise ValueError("Adjusted order price is beyond the closing price")
        elif price_area_vio_mode == PriceAreaVioMode.Cap:
            adj_price = price_area.close
    return adj_price


@register_jitted(cache=True)
def approx_long_buy_value_nb(val_price: float, size: float) -> float:
    """Approximate the value of a long-buy operation.

    Calculates the order value as the product of the absolute order size and the valuation price,
    and returns it as a negative value to represent spending.

    Args:
        val_price (float): The valuation price.
        size (float): The order size.

    Returns:
        float: A negative value representing spending, or 0.0 if the order size is zero.
    """
    if size == 0:
        return 0.0
    order_value = abs(size) * val_price
    add_free_cash = -order_value
    return -add_free_cash


@register_jitted(cache=True)
def should_apply_size_granularity_nb(size: float, size_granularity: float) -> bool:
    """Determine whether size granularity should be applied to the given size.

    Args:
        size (float): The original size value.
        size_granularity (float): The granularity factor used to round the size.

    Returns:
        bool: True if size granularity should be applied; otherwise, False.
    """
    if np.isnan(size_granularity):
        return False
    if size_granularity % 1 == 0:
        return True
    adj_size = size // size_granularity * size_granularity
    return not is_close_nb(size, adj_size) and not is_close_nb(size, adj_size + size_granularity)


@register_jitted(cache=True)
def apply_size_granularity_nb(size: float, size_granularity: float) -> float:
    """Apply size granularity to the given size by rounding it down to the nearest multiple.

    Args:
        size (float): The original size value.
        size_granularity (float): The granularity factor used to adjust the size.

    Returns:
        float: The size rounded down to a multiple of size_granularity.
    """
    return size // size_granularity * size_granularity


@register_jitted(cache=True)
def cast_account_state_nb(account_state: AccountState) -> AccountState:
    """Convert all numeric fields of the given `AccountState` to floats.

    Args:
        account_state (AccountState): The original account state instance with numeric attributes.

    Returns:
        AccountState: A new account state instance with all numerical attributes cast to float.
    """
    return AccountState(
        cash=float(account_state.cash),
        position=float(account_state.position),
        debt=float(account_state.debt),
        locked_cash=float(account_state.locked_cash),
        free_cash=float(account_state.free_cash),
    )


@register_jitted(cache=True)
def long_buy_nb(
    account_state: AccountState,
    size: float,
    price: float,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    leverage_mode: int = LeverageMode.Lazy,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Open or increase a long position.

    Args:
        account_state (AccountState): Current account state.
        size (float): Requested order size.
        price (float): Order price.
        fees (float): Proportional fees applied to the order price.
        fixed_fees (float): Fixed fees for the order.
        slippage (float): Fractional slippage applied to the order price.
        min_size (float): Minimum acceptable order size.
        max_size (float): Maximum acceptable order size.
        size_granularity (float): Granularity used to adjust order size.
        leverage (float): Leverage factor.
        leverage_mode (int): Mode for leverage calculation (e.g., lazy or eager).

            See `vectorbtpro.portfolio.enums.LeverageMode` for options.
        price_area_vio_mode (int): Mode for handling price area violations.

            See `vectorbtpro.portfolio.enums.PriceAreaVioMode` for options.
        allow_partial (bool): Determines whether partial order execution is permitted.
        percent (float): Fraction of free cash to consider for the order.
        price_area (PriceArea): Price area used to validate the adjusted price; see
            `vectorbtpro.portfolio.enums.PriceArea`.
        is_closing_price (bool): Indicates if the provided price is a closing price.

    Returns:
        Tuple[OrderResult, AccountState]: A tuple containing the order result and the updated account state.
    """
    _account_state = cast_account_state_nb(account_state)

    # Get cash limit
    cash_limit = _account_state.free_cash
    if not np.isnan(percent):
        cash_limit = cash_limit * percent
    if cash_limit <= 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoCash), _account_state
    cash_limit = cash_limit * leverage

    # Adjust for granularity
    if should_apply_size_granularity_nb(size, size_granularity):
        size = apply_size_granularity_nb(size, size_granularity)

    # Adjust for max size
    if not np.isnan(max_size) and size > max_size:
        if not allow_partial:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MaxSizeExceeded), _account_state

        size = max_size
    if np.isinf(size) and np.isinf(cash_limit):
        raise ValueError("Attempt to go in long direction infinitely")

    # Get price adjusted with slippage
    adj_price = price * (1 + slippage)
    adj_price = check_adj_price_nb(adj_price, price_area, is_closing_price, price_area_vio_mode)

    # Get cash required to complete this order
    if np.isinf(size):
        req_cash = np.inf
        req_fees = np.inf
    else:
        order_value = size * adj_price
        req_fees = order_value * fees + fixed_fees
        req_cash = order_value + req_fees

    if is_close_or_less_nb(req_cash, cash_limit):
        # Sufficient amount of cash
        final_size = size
        fees_paid = req_fees
    else:
        # Insufficient amount of cash, size will be less than requested

        # For fees of 10% and 1$ per transaction, you can buy for 90$ (new_req_cash)
        # to spend 100$ (cash_limit) in total
        max_req_cash = add_nb(cash_limit, -fixed_fees) / (1 + fees)
        if max_req_cash <= 0:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), _account_state

        max_acq_size = max_req_cash / adj_price

        # Adjust for granularity
        if should_apply_size_granularity_nb(max_acq_size, size_granularity):
            final_size = apply_size_granularity_nb(max_acq_size, size_granularity)
            new_order_value = final_size * adj_price
            fees_paid = new_order_value * fees + fixed_fees
            req_cash = new_order_value + fees_paid
        else:
            final_size = max_acq_size
            fees_paid = cash_limit - max_req_cash
            req_cash = cash_limit

    # Check against size of zero
    if is_close_nb(final_size, 0):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero), _account_state

    # Check against minimum size
    if not np.isnan(min_size) and is_less_nb(final_size, min_size):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached), _account_state

    # Check against partial fill (np.inf doesn't count)
    if np.isfinite(size) and is_less_nb(final_size, size) and not allow_partial:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.PartialFill), _account_state

    # Create a filled order
    order_result = OrderResult(
        float(final_size),
        float(adj_price),
        float(fees_paid),
        OrderSide.Buy,
        OrderStatus.Filled,
        -1,
    )

    # Update the current account state
    new_cash = add_nb(_account_state.cash, -req_cash)
    new_position = add_nb(_account_state.position, final_size)
    if leverage_mode == LeverageMode.Lazy:
        debt_diff = max(add_nb(req_cash, -_account_state.free_cash), 0.0)
        if debt_diff > 0:
            new_debt = _account_state.debt + debt_diff
            new_locked_cash = _account_state.locked_cash + _account_state.free_cash
            new_free_cash = 0.0
        else:
            new_debt = _account_state.debt
            new_locked_cash = _account_state.locked_cash
            new_free_cash = add_nb(_account_state.free_cash, -req_cash)
    else:
        if leverage > 1:
            if np.isinf(leverage):
                raise ValueError("Leverage must be finite for LeverageMode.Eager")
            order_value = final_size * adj_price
            new_debt = _account_state.debt + order_value * (leverage - 1) / leverage
            new_locked_cash = _account_state.locked_cash + order_value / leverage
            new_free_cash = add_nb(_account_state.free_cash, -order_value / leverage - fees_paid)
        else:
            new_debt = _account_state.debt
            new_locked_cash = _account_state.locked_cash
            new_free_cash = add_nb(_account_state.free_cash, -req_cash)
    new_account_state = AccountState(
        cash=float(new_cash),
        position=float(new_position),
        debt=float(new_debt),
        locked_cash=float(new_locked_cash),
        free_cash=float(new_free_cash),
    )
    return order_result, new_account_state


@register_jitted(cache=True)
def approx_long_sell_value_nb(position: float, debt: float, val_price: float, size: float) -> float:
    """Approximate the value of a long-sell operation.

    The computed value represents the spending amount for sorting purposes,
    where a positive value indicates spending.

    Args:
        position (float): Current position size.
        debt (float): Current account debt.
        val_price (float): Valuation price for the asset.
        size (float): Requested order size for the long-sell operation.

    Returns:
        float: The approximate value of the long-sell operation.
    """
    if size == 0 or position == 0:
        return 0.0
    size_limit = min(abs(size), position)
    order_value = size_limit * val_price
    size_fraction = size_limit / position
    released_debt = size_fraction * debt
    add_free_cash = order_value - released_debt
    return -add_free_cash


@register_jitted(cache=True)
def long_sell_nb(
    account_state: AccountState,
    size: float,
    price: float,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Decrease or close a long position.

    Args:
        account_state (AccountState): Current account state with cash, position, debt,
            and locked/free cash details.
        size (float): Desired size to sell. The size is capped by the current open long position.
        price (float): Price at which to execute the sell order.
        fees (float): Fee rate applied proportionally to the acquired cash.
        fixed_fees (float): Fixed fee amount applied to the transaction.
        slippage (float): Price adjustment factor to compensate for slippage.
        min_size (float): Minimum allowable order size.
        max_size (float): Maximum allowable order size.
        size_granularity (float): Granularity factor for rounding the order size.
        price_area_vio_mode (int): Mode to handle price area violations.

            See `vectorbtpro.portfolio.enums.PriceAreaVioMode` for options.
        allow_partial (bool): Indicator whether a partial fill of the order is permitted.
        percent (float): Percentage factor to adjust the order size relative to the current position.
        price_area (PriceArea): Permissible price area; see
            `vectorbtpro.portfolio.enums.PriceArea`.
        is_closing_price (bool): Flag indicating if the provided price is derived from a closing value.

    Returns:
        Tuple[OrderResult, AccountState]: A tuple containing the created order result and
            the updated account state.
    """
    _account_state = cast_account_state_nb(account_state)

    # Check for open position
    if _account_state.position == 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoOpenPosition), _account_state

    # Get size limit
    size_limit = min(size, _account_state.position)
    if not np.isnan(percent):
        size_limit = size_limit * percent

    # Adjust for granularity
    if should_apply_size_granularity_nb(size_limit, size_granularity):
        size = apply_size_granularity_nb(size, size_granularity)
        size_limit = apply_size_granularity_nb(size_limit, size_granularity)

    # Adjust for max size
    if not np.isnan(max_size) and size_limit > max_size:
        if not allow_partial:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MaxSizeExceeded), _account_state

        size_limit = max_size

    # Check against size of zero
    if is_close_nb(size_limit, 0):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero), _account_state

    # Check against minimum size
    if not np.isnan(min_size) and is_less_nb(size_limit, min_size):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached), _account_state

    # Check against partial fill
    if np.isfinite(size) and is_less_nb(size_limit, size) and not allow_partial:  # np.inf doesn't count
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.PartialFill), _account_state

    # Get price adjusted with slippage
    adj_price = price * (1 - slippage)
    adj_price = check_adj_price_nb(adj_price, price_area, is_closing_price, price_area_vio_mode)

    # Get acquired cash
    acq_cash = size_limit * adj_price

    # Update fees
    fees_paid = acq_cash * fees + fixed_fees

    # Get final cash by subtracting costs
    final_acq_cash = add_nb(acq_cash, -fees_paid)
    if final_acq_cash < 0 and is_less_nb(_account_state.free_cash, -final_acq_cash):
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), _account_state

    # Create a filled order
    order_result = OrderResult(
        float(size_limit),
        float(adj_price),
        float(fees_paid),
        OrderSide.Sell,
        OrderStatus.Filled,
        -1,
    )

    # Update the current account state
    new_cash = _account_state.cash + final_acq_cash
    new_position = add_nb(_account_state.position, -size_limit)
    new_pos_fraction = abs(new_position) / abs(_account_state.position)
    new_debt = new_pos_fraction * _account_state.debt
    new_locked_cash = new_pos_fraction * _account_state.locked_cash
    size_fraction = size_limit / _account_state.position
    released_debt = size_fraction * _account_state.debt
    new_free_cash = add_nb(_account_state.free_cash, final_acq_cash - released_debt)
    new_account_state = AccountState(
        cash=float(new_cash),
        position=float(new_position),
        debt=float(new_debt),
        locked_cash=float(new_locked_cash),
        free_cash=float(new_free_cash),
    )
    return order_result, new_account_state


@register_jitted(cache=True)
def approx_short_sell_value_nb(val_price: float, size: float) -> float:
    """Approximate the value of a short-sell operation.

    Calculates the transaction value based on the provided price and position size.
    A positive result indicates expenditure (for sorting purposes).

    Args:
        val_price (float): Reference price for evaluating the short sale.
        size (float): Size of the short position.

    Returns:
        float: The approximated value of the short-sell operation.
    """
    if size == 0:
        return 0.0
    order_value = abs(size) * val_price
    add_free_cash = -order_value
    return -add_free_cash


@register_jitted(cache=True)
def short_sell_nb(
    account_state: AccountState,
    size: float,
    price: float,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Open or increase a short position by placing a sell order.

    Args:
        account_state (AccountState): The current account state.
        size (float): Intended order size to initiate or increase a short position.
        price (float): Market price used to compute the adjusted order price.
        fees (float): Proportional fee rate applied to the order.
        fixed_fees (float): Fixed fee amount charged per order.
        slippage (float): Fraction representing potential price slippage.
        min_size (float): Minimum allowed order size.
        max_size (float): Maximum allowed order size.
        size_granularity (float): Order size step size for rounding.
        leverage (float): Leverage multiplier to adjust the cash limit.
        price_area_vio_mode (int): Mode for handling price area violations.

            See `vectorbtpro.portfolio.enums.PriceAreaVioMode` for options.
        allow_partial (bool): Indicates whether a partial fill is permitted.
        percent (float): Fraction of free cash to allocate for the order.
        price_area (PriceArea): Price area specification; see
            `vectorbtpro.portfolio.enums.PriceArea`.
        is_closing_price (bool): Indicates if the provided price is a closing price.

    Returns:
        Tuple[OrderResult, AccountState]: A tuple containing the filled order result and
            the updated account state.
    """
    _account_state = cast_account_state_nb(account_state)

    # Get cash limit
    cash_limit = _account_state.free_cash
    if not np.isnan(percent):
        cash_limit = cash_limit * percent
    if cash_limit <= 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoCash), _account_state
    cash_limit = cash_limit * leverage

    # Get price adjusted with slippage
    adj_price = price * (1 - slippage)
    adj_price = check_adj_price_nb(adj_price, price_area, is_closing_price, price_area_vio_mode)

    # Get size limit
    fees_adj_price = adj_price * (1 + fees)
    if fees_adj_price == 0:
        max_size_limit = np.inf
    else:
        max_size_limit = add_nb(cash_limit, -fixed_fees) / (adj_price * (1 + fees))
    size_limit = min(size, max_size_limit)
    if size_limit <= 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), _account_state

    # Adjust for granularity
    if should_apply_size_granularity_nb(size_limit, size_granularity):
        size = apply_size_granularity_nb(size, size_granularity)
        size_limit = apply_size_granularity_nb(size_limit, size_granularity)

    # Adjust for max size
    if not np.isnan(max_size) and size_limit > max_size:
        if not allow_partial:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MaxSizeExceeded), _account_state

        size_limit = max_size
    if np.isinf(size_limit):
        raise ValueError("Attempt to go in short direction infinitely")

    # Check against size of zero
    if is_close_nb(size_limit, 0):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero), _account_state

    # Check against minimum size
    if not np.isnan(min_size) and is_less_nb(size_limit, min_size):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached), _account_state

    # Check against partial fill
    if np.isfinite(size) and is_less_nb(size_limit, size) and not allow_partial:  # np.inf doesn't count
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.PartialFill), _account_state

    # Get acquired cash
    order_value = size_limit * adj_price

    # Update fees
    fees_paid = order_value * fees + fixed_fees

    # Get final cash by subtracting costs
    final_acq_cash = add_nb(order_value, -fees_paid)
    if final_acq_cash < 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), _account_state

    # Create a filled order
    order_result = OrderResult(
        float(size_limit),
        float(adj_price),
        float(fees_paid),
        OrderSide.Sell,
        OrderStatus.Filled,
        -1,
    )

    # Update the current account state
    new_cash = _account_state.cash + final_acq_cash
    new_position = _account_state.position - size_limit
    new_debt = _account_state.debt + order_value
    if np.isinf(leverage):
        if np.isinf(_account_state.free_cash):
            raise ValueError("Leverage must be finite when _account_state.free_cash is infinite")
        if is_close_or_less_nb(_account_state.free_cash, fees_paid):
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), _account_state
        leverage_ = order_value / (_account_state.free_cash - fees_paid)
    else:
        leverage_ = float(leverage)
    new_locked_cash = _account_state.locked_cash + order_value / leverage_
    new_free_cash = add_nb(_account_state.free_cash, -order_value / leverage_ - fees_paid)
    new_account_state = AccountState(
        cash=float(new_cash),
        position=float(new_position),
        debt=float(new_debt),
        locked_cash=float(new_locked_cash),
        free_cash=float(new_free_cash),
    )
    return order_result, new_account_state


@register_jitted(cache=True)
def approx_short_buy_value_nb(position: float, debt: float, locked_cash: float, val_price: float, size: float) -> float:
    """Approximate the cash value adjustment for a short-buy operation.

    Computes the additional cash required or freed by comparing the released debt and locked cash
    against the order value. A positive return value indicates a spending requirement.

    Args:
        position (float): The current short position size.
        debt (float): Total debt associated with the short position.
        locked_cash (float): Cash currently locked as collateral.
        val_price (float): Valuation price used to compute the order's value.
        size (float): Desired order size for the short-buy operation.

    Returns:
        float: The approximate cash value adjustment, where a positive value signifies spending.
    """
    if size == 0 or position == 0:
        return 0.0
    size_limit = min(abs(size), abs(position))
    order_value = size_limit * val_price
    size_fraction = size_limit / abs(position)
    released_debt = size_fraction * debt
    released_cash = size_fraction * locked_cash
    add_free_cash = released_cash + released_debt - order_value
    return -add_free_cash


@register_jitted(cache=True)
def short_buy_nb(
    account_state: AccountState,
    size: float,
    price: float,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Decrease or close a short position.

    This function processes an order to reduce or cover a short position. It verifies that an open
    position exists, checks for sufficient cash, and adjusts the order size based on constraints
    such as maximum size, granularity, and percentage limits. The function calculates the required
    fees and cash, and returns the order result along with an updated account state.

    Args:
        account_state (AccountState): Current state of the trading account.
        size (float): Requested order size for covering the short position.
        price (float): Execution price for the order.
        fees (float): Fraction of the order value charged as fee.
        fixed_fees (float): Fixed fee amount applied to the order.
        slippage (float): Slippage percentage to adjust the execution price.
        min_size (float): Minimum acceptable order size.
        max_size (float): Maximum allowed order size.
        size_granularity (float): Step size used for adjusting order size.
        price_area_vio_mode (int): Price area violation mode as defined by
            `vectorbtpro.portfolio.enums.PriceAreaVioMode`.
        allow_partial (bool): Whether to allow a partial fill if the full size cannot be executed.
        percent (float): Scaling factor to restrict the allowed order size.
        price_area (PriceArea): Price area constraint for order execution; see
            `vectorbtpro.portfolio.enums.PriceArea`.
        is_closing_price (bool): Indicates if the provided price is to be treated as a closing price.

    Returns:
        Tuple[OrderResult, AccountState]: A tuple containing the executed order result and
            the updated account state.
    """
    _account_state = cast_account_state_nb(account_state)

    # Check for open position
    if _account_state.position == 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoOpenPosition), _account_state

    # Get cash limit
    cash_limit = _account_state.free_cash + _account_state.debt + _account_state.locked_cash
    if cash_limit <= 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoCash), _account_state

    # Get size limit
    size_limit = min(size, abs(_account_state.position))
    if not np.isnan(percent):
        size_limit = size_limit * percent

    # Adjust for granularity
    if should_apply_size_granularity_nb(size_limit, size_granularity):
        size_limit = apply_size_granularity_nb(size_limit, size_granularity)

    # Adjust for max size
    if not np.isnan(max_size) and size_limit > max_size:
        if not allow_partial:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MaxSizeExceeded), _account_state

        size_limit = max_size

    # Get price adjusted with slippage
    adj_price = price * (1 + slippage)
    adj_price = check_adj_price_nb(adj_price, price_area, is_closing_price, price_area_vio_mode)

    # Get cash required to complete this order
    if np.isinf(size_limit):
        req_cash = np.inf
        req_fees = np.inf
    else:
        order_value = size_limit * adj_price
        req_fees = order_value * fees + fixed_fees
        req_cash = order_value + req_fees

    if is_close_or_less_nb(req_cash, cash_limit):
        # Sufficient amount of cash
        final_size = size_limit
        fees_paid = req_fees
    else:
        # Insufficient amount of cash, size will be less than requested

        # For fees of 10% and 1$ per transaction, you can buy for 90$ (new_req_cash)
        # to spend 100$ (cash_limit) in total
        max_req_cash = add_nb(cash_limit, -fixed_fees) / (1 + fees)
        if max_req_cash <= 0:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), _account_state

        max_acq_size = max_req_cash / adj_price

        # Adjust for granularity
        if should_apply_size_granularity_nb(max_acq_size, size_granularity):
            final_size = apply_size_granularity_nb(max_acq_size, size_granularity)
            new_order_value = final_size * adj_price
            fees_paid = new_order_value * fees + fixed_fees
            req_cash = new_order_value + fees_paid
        else:
            final_size = max_acq_size
            fees_paid = cash_limit - max_req_cash
            req_cash = cash_limit

    # Check size of zero
    if is_close_nb(final_size, 0):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero), _account_state

    # Check against minimum size
    if not np.isnan(min_size) and is_less_nb(final_size, min_size):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached), _account_state

    # Check against partial fill (np.inf doesn't count)
    if np.isfinite(size_limit) and is_less_nb(final_size, size_limit) and not allow_partial:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.PartialFill), _account_state

    # Create a filled order
    order_result = OrderResult(
        float(final_size),
        float(adj_price),
        float(fees_paid),
        OrderSide.Buy,
        OrderStatus.Filled,
        -1,
    )

    # Update the current account state
    new_cash = add_nb(_account_state.cash, -req_cash)
    new_position = add_nb(_account_state.position, final_size)
    new_pos_fraction = abs(new_position) / abs(_account_state.position)
    new_debt = new_pos_fraction * _account_state.debt
    new_locked_cash = new_pos_fraction * _account_state.locked_cash
    size_fraction = final_size / abs(_account_state.position)
    released_debt = size_fraction * _account_state.debt
    released_cash = size_fraction * _account_state.locked_cash
    new_free_cash = add_nb(_account_state.free_cash, released_cash + released_debt - req_cash)
    new_account_state = AccountState(
        cash=float(new_cash),
        position=float(new_position),
        debt=float(new_debt),
        locked_cash=float(new_locked_cash),
        free_cash=float(new_free_cash),
    )
    return order_result, new_account_state


@register_jitted(cache=True)
def approx_buy_value_nb(
    position: float,
    debt: float,
    locked_cash: float,
    val_price: float,
    size: float,
    direction: int,
) -> float:
    """Approximate value of a buy operation.

    Calculates an estimated order value based on the current position, debt, locked cash,
    valuation price, and desired order size. Depending on the position and trade direction,
    it uses a short or long buy approximation or a combination of both. A positive return
    value indicates spending, which is useful for sorting.

    Args:
        position (float): Current position amount.
        debt (float): Current debt amount.
        locked_cash (float): Amount of cash locked in orders.
        val_price (float): Valuation price of the asset.
        size (float): Requested order size.
        direction (int): Order direction.

            See `vectorbtpro.portfolio.enums.Direction`.

    Returns:
        float: Approximate order value representing the spending amount.
    """
    if position <= 0 and direction == Direction.ShortOnly:
        return approx_short_buy_value_nb(position, debt, locked_cash, val_price, size)
    if position >= 0:
        return approx_long_buy_value_nb(val_price, size)
    value1 = approx_short_buy_value_nb(position, debt, locked_cash, val_price, size)
    new_size = add_nb(size, -abs(position))
    if new_size <= 0:
        return value1
    value2 = approx_long_buy_value_nb(val_price, new_size)
    return value1 + value2


@register_jitted(cache=True)
def buy_nb(
    account_state: AccountState,
    size: float,
    price: float,
    direction: int = Direction.Both,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    leverage_mode: int = LeverageMode.Lazy,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Perform a buy operation on an account by determining the appropriate long or short strategy
    based on the current state and provided parameters.

    Args:
        account_state (AccountState): Current state of the account.
        size (float): Order size.
        price (float): Execution price for the order.
        direction (int): Order direction.

            See `vectorbtpro.portfolio.enums.Direction`.
        fees (float): Trading fees.
        fixed_fees (float): Fixed fee charges.
        slippage (float): Slippage applied to the order.
        min_size (float): Minimum allowed order size, may be NaN.
        max_size (float): Maximum allowed order size, may be NaN.
        size_granularity (float): Granularity factor for the order size, may be NaN.
        leverage (float): Leverage factor.
        leverage_mode (int): Leverage mode; see `vectorbtpro.portfolio.enums.LeverageMode`.
        price_area_vio_mode (int): Price area violation mode; see `vectorbtpro.portfolio.enums.PriceAreaVioMode`.
        allow_partial (bool): Flag permitting partial order fills.
        percent (float): Percentage parameter used in order calculations, may be NaN.
        price_area (PriceArea): Constraint for the price area; see `vectorbtpro.portfolio.enums.PriceArea`.
        is_closing_price (bool): Indicates whether the provided price is for closing positions.

    Returns:
        Tuple[OrderResult, AccountState]: A tuple containing the order execution result and
            the updated account state.
    """
    _account_state = cast_account_state_nb(account_state)

    if _account_state.position <= 0 and direction == Direction.ShortOnly:
        return short_buy_nb(
            account_state=_account_state,
            size=size,
            price=price,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
            min_size=min_size,
            max_size=max_size,
            size_granularity=size_granularity,
            price_area_vio_mode=price_area_vio_mode,
            allow_partial=allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )
    if _account_state.position >= 0:
        return long_buy_nb(
            account_state=_account_state,
            size=size,
            price=price,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
            min_size=min_size,
            max_size=max_size,
            size_granularity=size_granularity,
            leverage=leverage,
            leverage_mode=leverage_mode,
            price_area_vio_mode=price_area_vio_mode,
            allow_partial=allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )
    short_size = min(size, abs(_account_state.position))
    if not np.isnan(min_size):
        min_size1 = min(min_size, abs(_account_state.position))
    else:
        min_size1 = np.nan
    if not np.isnan(max_size):
        max_size1 = min(max_size, abs(_account_state.position))
    else:
        max_size1 = np.nan
    new_order_result1, new_account_state1 = short_buy_nb(
        account_state=_account_state,
        size=short_size,
        price=price,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        min_size=min_size1,
        max_size=max_size1,
        size_granularity=size_granularity,
        price_area_vio_mode=price_area_vio_mode,
        allow_partial=allow_partial,
        percent=np.nan,
        price_area=price_area,
        is_closing_price=is_closing_price,
    )
    if new_order_result1.status != OrderStatus.Filled:
        return new_order_result1, _account_state
    if new_account_state1.position != 0:
        return new_order_result1, new_account_state1
    new_size = add_nb(size, -abs(_account_state.position))
    if new_size <= 0:
        return new_order_result1, new_account_state1
    if not np.isnan(min_size):
        min_size2 = max(min_size - abs(_account_state.position), 0.0)
    else:
        min_size2 = np.nan
    if not np.isnan(max_size):
        max_size2 = max(max_size - abs(_account_state.position), 0.0)
    else:
        max_size2 = np.nan
    new_order_result2, new_account_state2 = long_buy_nb(
        account_state=new_account_state1,
        size=new_size,
        price=price,
        fees=fees,
        fixed_fees=0.0,
        slippage=slippage,
        min_size=min_size2,
        max_size=max_size2,
        size_granularity=size_granularity,
        leverage=leverage,
        leverage_mode=leverage_mode,
        price_area_vio_mode=price_area_vio_mode,
        allow_partial=allow_partial,
        percent=percent,
        price_area=price_area,
        is_closing_price=is_closing_price,
    )
    if new_order_result2.status != OrderStatus.Filled:
        if allow_partial or np.isinf(new_size):
            if new_order_result2.status_info == OrderStatusInfo.SizeZero:
                return new_order_result1, new_account_state1
            if new_order_result2.status_info == OrderStatusInfo.NoCash:
                return new_order_result1, new_account_state1
        return new_order_result2, _account_state
    new_order_result = OrderResult(
        new_order_result1.size + new_order_result2.size,
        new_order_result2.price,
        new_order_result1.fees + new_order_result2.fees,
        new_order_result2.side,
        new_order_result2.status,
        new_order_result2.status_info,
    )
    return new_order_result, new_account_state2


@register_jitted(cache=True)
def approx_sell_value_nb(
    position: float,
    debt: float,
    val_price: float,
    size: float,
    direction: int,
) -> float:
    """Approximate the sell operation value based on asset position, debt, and valuation price.

    Calculates an estimated sell value given the current position, associated debt, valuation price,
    and order size. A positive result represents spending, which is useful for sorting operations.

    Args:
        position (float): Current asset position.
        debt (float): Debt associated with the asset.
        val_price (float): Valuation price of the asset.
        size (float): Size of the sell order.
        direction (int): Order direction.

            See `vectorbtpro.portfolio.enums.Direction`.

    Returns:
        float: The approximate value of the sell operation.
    """
    if position >= 0 and direction == Direction.LongOnly:
        return approx_long_sell_value_nb(position, debt, val_price, size)
    if position <= 0:
        return approx_short_sell_value_nb(val_price, size)
    value1 = approx_long_sell_value_nb(position, debt, val_price, size)
    new_size = add_nb(size, -abs(position))
    if new_size <= 0:
        return value1
    value2 = approx_short_sell_value_nb(val_price, new_size)
    return value1 + value2


@register_jitted(cache=True)
def sell_nb(
    account_state: AccountState,
    size: float,
    price: float,
    direction: int = Direction.Both,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Execute a sell order based on the current account state and specified parameters.

    Depending on the account state and trade direction, this function delegates the sell
    order to either `long_sell_nb` or `short_sell_nb`. For positions spanning both long and
    short, it processes the long portion first and, if fully filled, proceeds with a short sell
    for the remaining size.

    Args:
        account_state (AccountState): Current account state.
        size (float): Order size.
        price (float): Order price.
        direction (int): Order direction.

            See `vectorbtpro.portfolio.enums.Direction`.
        fees (float): Fee associated with the order.
        fixed_fees (float): Fixed fee amount.
        slippage (float): Allowed price slippage.
        min_size (float): Minimum order size constraint.
        max_size (float): Maximum order size constraint.
        size_granularity (float): Size granularity factor.
        leverage (float): Leverage factor.
        price_area_vio_mode (int): Mode for handling price area violation; see
            `vectorbtpro.portfolio.enums.PriceAreaVioMode`.
        allow_partial (bool): Whether partial fills are permitted.
        percent (float): Order size as a percentage.
        price_area (PriceArea): Price area constraint; see `vectorbtpro.portfolio.enums.PriceArea`.
        is_closing_price (bool): Flag indicating if the price is a closing price.

    Returns:
        Tuple[OrderResult, AccountState]: A tuple containing the order result and the updated account state.
    """
    _account_state = cast_account_state_nb(account_state)

    if _account_state.position >= 0 and direction == Direction.LongOnly:
        return long_sell_nb(
            account_state=_account_state,
            size=size,
            price=price,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
            min_size=min_size,
            max_size=max_size,
            size_granularity=size_granularity,
            price_area_vio_mode=price_area_vio_mode,
            allow_partial=allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )
    if _account_state.position <= 0:
        return short_sell_nb(
            account_state=_account_state,
            size=size,
            price=price,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
            min_size=min_size,
            max_size=max_size,
            size_granularity=size_granularity,
            leverage=leverage,
            price_area_vio_mode=price_area_vio_mode,
            allow_partial=allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )
    long_size = min(size, _account_state.position)
    if not np.isnan(min_size):
        min_size1 = min(min_size, _account_state.position)
    else:
        min_size1 = np.nan
    if not np.isnan(max_size):
        max_size1 = min(max_size, _account_state.position)
    else:
        max_size1 = np.nan
    new_order_result1, new_account_state1 = long_sell_nb(
        account_state=_account_state,
        size=long_size,
        price=price,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        min_size=min_size1,
        max_size=max_size1,
        size_granularity=size_granularity,
        price_area_vio_mode=price_area_vio_mode,
        allow_partial=allow_partial,
        percent=np.nan,
        price_area=price_area,
        is_closing_price=is_closing_price,
    )
    if new_order_result1.status != OrderStatus.Filled:
        return new_order_result1, _account_state
    if new_account_state1.position != 0:
        return new_order_result1, new_account_state1
    new_size = add_nb(size, -abs(_account_state.position))
    if new_size <= 0:
        return new_order_result1, new_account_state1
    if not np.isnan(min_size):
        min_size2 = max(min_size - _account_state.position, 0.0)
    else:
        min_size2 = np.nan
    if not np.isnan(max_size):
        max_size2 = max(max_size - _account_state.position, 0.0)
    else:
        max_size2 = np.nan
    new_order_result2, new_account_state2 = short_sell_nb(
        account_state=new_account_state1,
        size=new_size,
        price=price,
        fees=fees,
        fixed_fees=0.0,
        slippage=slippage,
        min_size=min_size2,
        max_size=max_size2,
        size_granularity=size_granularity,
        leverage=leverage,
        price_area_vio_mode=price_area_vio_mode,
        allow_partial=allow_partial,
        percent=percent,
        price_area=price_area,
        is_closing_price=is_closing_price,
    )
    if new_order_result2.status != OrderStatus.Filled:
        if allow_partial or np.isinf(new_size):
            if new_order_result2.status_info == OrderStatusInfo.SizeZero:
                return new_order_result1, new_account_state1
            if new_order_result2.status_info == OrderStatusInfo.NoCash:
                return new_order_result1, new_account_state1
        return new_order_result2, _account_state
    new_order_result = OrderResult(
        new_order_result1.size + new_order_result2.size,
        new_order_result2.price,
        new_order_result1.fees + new_order_result2.fees,
        new_order_result2.side,
        new_order_result2.status,
        new_order_result2.status_info,
    )
    return new_order_result, new_account_state2


@register_jitted(cache=True)
def update_value_nb(
    cash_before: float,
    cash_now: float,
    position_before: float,
    position_now: float,
    val_price_before: float,
    val_price_now: float,
    value_before: float,
) -> float:
    """Calculate updated portfolio value based on changes in cash and asset positions.

    Args:
        cash_before (float): Cash amount before the update.
        cash_now (float): Current cash amount.
        position_before (float): Asset position before the update.
        position_now (float): Current asset position.
        val_price_before (float): Valuation price before the update.
        val_price_now (float): Current valuation price.
        value_before (float): Portfolio value before the update.

    Returns:
        float: Updated portfolio value.
    """
    cash_flow = cash_now - cash_before
    if position_before != 0:
        asset_value_before = position_before * val_price_before
    else:
        asset_value_before = 0.0
    if position_now != 0:
        asset_value_now = position_now * val_price_now
    else:
        asset_value_now = 0.0
    asset_value_diff = asset_value_now - asset_value_before
    value_now = value_before + cash_flow + asset_value_diff
    return value_now


@register_jitted(cache=True)
def get_diraware_size_nb(size: float, direction: int) -> float:
    """Adjust size based on trade direction.

    Args:
        size (float): Base size value.
        direction (int): Order direction; if equal to `Direction.ShortOnly`, the size is negated.

            See `vectorbtpro.portfolio.enums.Direction` for options.

    Returns:
        float: Adjusted size reflecting trade direction.
    """
    if direction == Direction.ShortOnly:
        return size * -1
    return size


@register_jitted(cache=True)
def resolve_size_nb(
    size: float,
    size_type: int,
    position: float,
    val_price: float,
    value: float,
    target_size_type: int = SizeType.Amount,
    as_requirement: bool = False,
) -> tp.Tuple[float, float]:
    """Convert size specification into an absolute asset amount and its corresponding percentage.

    The percentage is set only when the `SizeType.Percent(100)` option is used.

    Args:
        size (float): Input size value.
        size_type (int): Indicator of the size specification type.
        position (float): Current asset position.
        val_price (float): Valuation price for converting value into asset amount.
        value (float): Total portfolio value.
        target_size_type (int): Desired size type for the output.

            See `vectorbtpro.portfolio.enums.SizeType` for options.
        as_requirement (bool): Whether to treat the size as a requirement adjustment.

    Returns:
        Tuple[float, float]: A tuple containing the absolute asset amount and the percentage representation.
    """
    percent = np.nan
    if size_type == target_size_type:
        return float(size), percent

    if size_type == SizeType.ValuePercent100:
        if size_type == target_size_type:
            return float(size), percent

        size /= 100
        size_type = SizeType.ValuePercent

    if size_type == SizeType.TargetPercent100:
        if size_type == target_size_type:
            return float(size), percent

        size /= 100
        size_type = SizeType.TargetPercent

    if size_type == SizeType.ValuePercent or size_type == SizeType.TargetPercent:
        if size_type == target_size_type:
            return float(size), percent

        size *= value
        if size_type == SizeType.ValuePercent:
            size_type = SizeType.Value
        else:
            size_type = SizeType.TargetValue

    if size_type == SizeType.Value or size_type == SizeType.TargetValue:
        if size_type == target_size_type:
            return float(size), percent

        size /= val_price
        if size_type == SizeType.Value:
            size_type = SizeType.Amount
        else:
            size_type = SizeType.TargetAmount

    if size_type == SizeType.TargetAmount:
        if size_type == target_size_type:
            return float(size), percent

        if not as_requirement:
            size -= position
        size_type = SizeType.Amount

    if size_type == SizeType.Percent100:
        if size_type == target_size_type:
            return float(size), percent

        size /= 100
        size_type = SizeType.Percent

    if size_type == SizeType.Percent:
        if size_type == target_size_type:
            return float(size), percent

        percent = abs(size)
        if as_requirement:
            size = np.nan
        else:
            size = np.sign(size) * np.inf
        size_type = SizeType.Amount

    if size_type != target_size_type:
        raise ValueError("Cannot convert size to target size type")
    if as_requirement:
        size = abs(size)
    return float(size), percent


@register_jitted(cache=True)
def approx_order_value_nb(
    exec_state: ExecState,
    size: float,
    size_type: int = SizeType.Amount,
    direction: int = Direction.Both,
) -> float:
    """Approximate the value of an order.

    Assumes infinite cash.

    Positive value indicates spending (used for sorting purposes).

    Args:
        exec_state (ExecState): Execution state containing portfolio details such as position,
            value, debt, locked cash, and valuation price.
        size (float): Order size.
        size_type (int): Indicator specifying the type of the order size.

            See `vectorbtpro.portfolio.enums.SizeType`.
        direction (int): Order direction.

            See `vectorbtpro.portfolio.enums.Direction`.

    Returns:
        float: The approximate order value.
    """
    size = get_diraware_size_nb(float(size), direction)
    amount_size, _ = resolve_size_nb(
        size=size,
        size_type=size_type,
        position=exec_state.position,
        val_price=exec_state.val_price,
        value=exec_state.value,
    )
    if amount_size >= 0:
        order_value = approx_buy_value_nb(
            position=exec_state.position,
            debt=exec_state.debt,
            locked_cash=exec_state.locked_cash,
            val_price=exec_state.val_price,
            size=abs(amount_size),
            direction=direction,
        )
    else:
        order_value = approx_sell_value_nb(
            position=exec_state.position,
            debt=exec_state.debt,
            val_price=exec_state.val_price,
            size=abs(amount_size),
            direction=direction,
        )
    return order_value


@register_jitted(cache=True)
def execute_order_nb(
    exec_state: ExecState,
    order: Order,
    price_area: PriceArea = NoPriceArea,
    update_value: bool = False,
) -> tp.Tuple[OrderResult, ExecState]:
    """Execute an order given the current state.

    Error is raised if any input value is invalid.

    The order is ignored if its execution does not affect the current balance.
    The order is rejected if any input violates a limit or restriction.

    Args:
        exec_state (ExecState): The current execution state.

            See `vectorbtpro.portfolio.enums.ExecState`.
        order (Order): The order to execute.

            See `vectorbtpro.portfolio.enums.Order`.
        price_area (PriceArea): Price area information.

            See `vectorbtpro.portfolio.enums.PriceArea`.
        update_value (bool): Update the account value after executing the order.

    Returns:
        Tuple[OrderResult, ExecState]: A tuple containing the order execution result and
            the updated execution state.
    """
    # numerical stability
    cash = float(exec_state.cash)
    if is_close_nb(cash, 0):
        cash = 0.0
    position = float(exec_state.position)
    if is_close_nb(position, 0):
        position = 0.0
    debt = float(exec_state.debt)
    if is_close_nb(debt, 0):
        debt = 0.0
    locked_cash = float(exec_state.locked_cash)
    if is_close_nb(locked_cash, 0):
        locked_cash = 0.0
    free_cash = float(exec_state.free_cash)
    if is_close_nb(free_cash, 0):
        free_cash = 0.0
    val_price = float(exec_state.val_price)
    if is_close_nb(val_price, 0):
        val_price = 0.0
    value = float(exec_state.value)
    if is_close_nb(value, 0):
        value = 0.0

    # Pre-fill account state
    account_state = AccountState(
        cash=cash,
        position=position,
        debt=debt,
        locked_cash=locked_cash,
        free_cash=free_cash,
    )

    # Check price area
    if np.isinf(price_area.open) or price_area.open < 0:
        raise ValueError("price_area.open must be either NaN, or finite and 0 or greater")
    if np.isinf(price_area.high) or price_area.high < 0:
        raise ValueError("price_area.high must be either NaN, or finite and 0 or greater")
    if np.isinf(price_area.low) or price_area.low < 0:
        raise ValueError("price_area.low must be either NaN, or finite and 0 or greater")
    if np.isinf(price_area.close) or price_area.close < 0:
        raise ValueError("price_area.close must be either NaN, or finite and 0 or greater")

    # Resolve price
    order_price = order.price
    is_closing_price = False
    if np.isinf(order_price):
        if order_price > 0:
            order_price = price_area.close
            is_closing_price = True
        else:
            order_price = price_area.open
    elif order_price == PriceType.NextOpen:
        raise ValueError("Next open must be handled higher in the stack")
    elif order_price == PriceType.NextClose:
        raise ValueError("Next close must be handled higher in the stack")

    # Ignore order if size or price is nan
    if np.isnan(order.size):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeNaN), exec_state
    if np.isnan(order_price):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.PriceNaN), exec_state

    # Check account state
    if np.isnan(cash):
        raise ValueError("exec_state.cash cannot be NaN")
    if not np.isfinite(position):
        raise ValueError("exec_state.position must be finite")
    if not np.isfinite(debt) or debt < 0:
        raise ValueError("exec_state.debt must be finite and 0 or greater")
    if not np.isfinite(locked_cash) or locked_cash < 0:
        raise ValueError("exec_state.locked_cash must be finite and 0 or greater")
    if np.isnan(free_cash):
        raise ValueError("exec_state.free_cash cannot be NaN")

    # Check order
    if not np.isfinite(order_price) or order_price < 0:
        raise ValueError("order.price must be finite and 0 or greater")
    if order.size_type < 0 or order.size_type >= len(SizeType):
        raise ValueError("order.size_type is invalid")
    if order.direction < 0 or order.direction >= len(Direction):
        raise ValueError("order.direction is invalid")
    if not np.isfinite(order.fees):
        raise ValueError("order.fees must be finite")
    if not np.isfinite(order.fixed_fees):
        raise ValueError("order.fixed_fees must be finite")
    if not np.isfinite(order.slippage) or order.slippage < 0:
        raise ValueError("order.slippage must be finite and 0 or greater")
    if np.isinf(order.min_size) or order.min_size < 0:
        raise ValueError("order.min_size must be either NaN, 0, or greater")
    if order.max_size <= 0:
        raise ValueError("order.max_size must be either NaN or greater than 0")
    if np.isinf(order.size_granularity) or order.size_granularity <= 0:
        raise ValueError("order.size_granularity must be either NaN, or finite and greater than 0")
    if np.isnan(order.leverage) or order.leverage <= 0:
        raise ValueError("order.leverage must be greater than 0")
    if order.leverage_mode < 0 or order.leverage_mode >= len(LeverageMode):
        raise ValueError("order.leverage_mode is invalid")
    if not np.isfinite(order.reject_prob) or order.reject_prob < 0 or order.reject_prob > 1:
        raise ValueError("order.reject_prob must be between 0 and 1")

    # Positive/negative size in short direction should be treated as negative/positive
    order_size = get_diraware_size_nb(order.size, order.direction)
    min_order_size = order.min_size
    max_order_size = order.max_size
    order_size_type = order.size_type

    if (
        order_size_type == SizeType.ValuePercent100
        or order_size_type == SizeType.ValuePercent
        or order_size_type == SizeType.TargetPercent100
        or order_size_type == SizeType.TargetPercent
        or order_size_type == SizeType.Value
        or order_size_type == SizeType.TargetValue
    ):
        if np.isinf(val_price) or val_price <= 0:
            raise ValueError("val_price_now must be finite and greater than 0")
        if np.isnan(val_price):
            return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.ValPriceNaN), exec_state
        if (
            order_size_type == SizeType.ValuePercent100
            or order_size_type == SizeType.ValuePercent
            or order_size_type == SizeType.TargetPercent100
            or order_size_type == SizeType.TargetPercent
        ):
            if np.isnan(value):
                return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.ValueNaN), exec_state
            if value <= 0:
                return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.ValueZeroNeg), exec_state

    order_size, percent = resolve_size_nb(
        size=order_size,
        size_type=order_size_type,
        position=position,
        val_price=val_price,
        value=value,
    )
    if not np.isnan(min_order_size):
        min_order_size, min_percent = resolve_size_nb(
            size=min_order_size,
            size_type=order_size_type,
            position=position,
            val_price=val_price,
            value=value,
            as_requirement=True,
        )
        if not np.isnan(percent) and not np.isnan(min_percent) and is_less_nb(percent, min_percent):
            return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached), exec_state
    if not np.isnan(max_order_size):
        max_order_size, max_percent = resolve_size_nb(
            size=max_order_size,
            size_type=order_size_type,
            position=position,
            val_price=val_price,
            value=value,
            as_requirement=True,
        )
        if not np.isnan(percent) and not np.isnan(max_percent) and is_less_nb(max_percent, percent):
            percent = max_percent

    if order_size >= 0:
        order_result, new_account_state = buy_nb(
            account_state=account_state,
            size=order_size,
            price=order_price,
            direction=order.direction,
            fees=order.fees,
            fixed_fees=order.fixed_fees,
            slippage=order.slippage,
            min_size=min_order_size,
            max_size=max_order_size,
            size_granularity=order.size_granularity,
            leverage=order.leverage,
            leverage_mode=order.leverage_mode,
            price_area_vio_mode=order.price_area_vio_mode,
            allow_partial=order.allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )
    else:
        order_result, new_account_state = sell_nb(
            account_state=account_state,
            size=-order_size,
            price=order_price,
            direction=order.direction,
            fees=order.fees,
            fixed_fees=order.fixed_fees,
            slippage=order.slippage,
            min_size=min_order_size,
            max_size=max_order_size,
            size_granularity=order.size_granularity,
            leverage=order.leverage,
            price_area_vio_mode=order.price_area_vio_mode,
            allow_partial=order.allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )

    if order.reject_prob > 0:
        if np.random.uniform(0, 1) < order.reject_prob:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.RandomEvent), exec_state

    if order_result.status == OrderStatus.Rejected and order.raise_reject:
        raise_rejected_order_nb(order_result)

    is_filled = order_result.status == OrderStatus.Filled
    if is_filled and update_value:
        new_val_price = order_result.price
        new_value = update_value_nb(
            cash,
            new_account_state.cash,
            position,
            new_account_state.position,
            val_price,
            order_result.price,
            value,
        )
    else:
        new_val_price = val_price
        new_value = value

    new_exec_state = ExecState(
        cash=new_account_state.cash,
        position=new_account_state.position,
        debt=new_account_state.debt,
        locked_cash=new_account_state.locked_cash,
        free_cash=new_account_state.free_cash,
        val_price=new_val_price,
        value=new_value,
    )

    return order_result, new_exec_state


@register_jitted(cache=True)
def fill_log_record_nb(
    records: tp.RecordArray2d,
    r: int,
    group: int,
    col: int,
    i: int,
    price_area: PriceArea,
    exec_state: ExecState,
    order: Order,
    order_result: OrderResult,
    new_exec_state: ExecState,
    order_id: int,
) -> None:
    """Fill a log record with order and execution state details.

    Args:
        records (RecordArray2d): The array where log records are stored.
        r (int): The index of the log record in the column.
        group (int): The current group index.
        col (int): The current column index.
        i (int): The current row index.
        price_area (PriceArea): Price area data with attributes `open`, `high`, `low`, and `close`.
        exec_state (ExecState): The execution state before executing the order.
        order (Order): The order details including size, price, fees, and other parameters.
        order_result (OrderResult): The result of the order execution.
        new_exec_state (ExecState): The execution state after executing the order.
        order_id (int): The unique identifier of the order.

    Returns:
        None: This function modifies `records` in place.
    """

    records["id"][r, col] = r
    records["group"][r, col] = group
    records["col"][r, col] = col
    records["idx"][r, col] = i
    records["price_area_open"][r, col] = price_area.open
    records["price_area_high"][r, col] = price_area.high
    records["price_area_low"][r, col] = price_area.low
    records["price_area_close"][r, col] = price_area.close
    records["st0_cash"][r, col] = exec_state.cash
    records["st0_position"][r, col] = exec_state.position
    records["st0_debt"][r, col] = exec_state.debt
    records["st0_locked_cash"][r, col] = exec_state.locked_cash
    records["st0_free_cash"][r, col] = exec_state.free_cash
    records["st0_val_price"][r, col] = exec_state.val_price
    records["st0_value"][r, col] = exec_state.value
    records["req_size"][r, col] = order.size
    records["req_price"][r, col] = order.price
    records["req_size_type"][r, col] = order.size_type
    records["req_direction"][r, col] = order.direction
    records["req_fees"][r, col] = order.fees
    records["req_fixed_fees"][r, col] = order.fixed_fees
    records["req_slippage"][r, col] = order.slippage
    records["req_min_size"][r, col] = order.min_size
    records["req_max_size"][r, col] = order.max_size
    records["req_size_granularity"][r, col] = order.size_granularity
    records["req_leverage"][r, col] = order.leverage
    records["req_leverage_mode"][r, col] = order.leverage_mode
    records["req_reject_prob"][r, col] = order.reject_prob
    records["req_price_area_vio_mode"][r, col] = order.price_area_vio_mode
    records["req_allow_partial"][r, col] = order.allow_partial
    records["req_raise_reject"][r, col] = order.raise_reject
    records["req_log"][r, col] = order.log
    records["res_size"][r, col] = order_result.size
    records["res_price"][r, col] = order_result.price
    records["res_fees"][r, col] = order_result.fees
    records["res_side"][r, col] = order_result.side
    records["res_status"][r, col] = order_result.status
    records["res_status_info"][r, col] = order_result.status_info
    records["st1_cash"][r, col] = new_exec_state.cash
    records["st1_position"][r, col] = new_exec_state.position
    records["st1_debt"][r, col] = new_exec_state.debt
    records["st1_locked_cash"][r, col] = new_exec_state.locked_cash
    records["st1_free_cash"][r, col] = new_exec_state.free_cash
    records["st1_val_price"][r, col] = new_exec_state.val_price
    records["st1_value"][r, col] = new_exec_state.value
    records["order_id"][r, col] = order_id


@register_jitted(cache=True)
def fill_order_record_nb(records: tp.RecordArray2d, r: int, col: int, i: int, order_result: OrderResult) -> None:
    """Fill an order record with executed order details.

    Args:
        records (RecordArray2d): The array where order records are stored.
        r (int): The index of the order record in the column.
        col (int): The current column index.
        i (int): The current row index.
        order_result (OrderResult): The result of the order execution containing size, price, fees, and side.

    Returns:
        None: This function modifies `records` in place.
    """

    records["id"][r, col] = r
    records["col"][r, col] = col
    records["idx"][r, col] = i
    records["size"][r, col] = order_result.size
    records["price"][r, col] = order_result.price
    records["fees"][r, col] = order_result.fees
    records["side"][r, col] = order_result.side


@register_jitted(cache=True)
def raise_rejected_order_nb(order_result: OrderResult) -> None:
    """Raise a `vectorbtpro.portfolio.enums.RejectedOrderError` based on the order result's status.

    Args:
        order_result (OrderResult): The order result containing status information used to
            determine the rejection reason.

    Returns:
        None
    """

    if order_result.status_info == OrderStatusInfo.SizeNaN:
        raise RejectedOrderError("Size is NaN")
    if order_result.status_info == OrderStatusInfo.PriceNaN:
        raise RejectedOrderError("Price is NaN")
    if order_result.status_info == OrderStatusInfo.ValPriceNaN:
        raise RejectedOrderError("Asset valuation price is NaN")
    if order_result.status_info == OrderStatusInfo.ValueNaN:
        raise RejectedOrderError("Asset/group value is NaN")
    if order_result.status_info == OrderStatusInfo.ValueZeroNeg:
        raise RejectedOrderError("Asset/group value is zero or negative")
    if order_result.status_info == OrderStatusInfo.SizeZero:
        raise RejectedOrderError("Size is zero")
    if order_result.status_info == OrderStatusInfo.NoCash:
        raise RejectedOrderError("Not enough cash")
    if order_result.status_info == OrderStatusInfo.NoOpenPosition:
        raise RejectedOrderError("No open position to reduce/close")
    if order_result.status_info == OrderStatusInfo.MaxSizeExceeded:
        raise RejectedOrderError("Size is greater than maximum allowed")
    if order_result.status_info == OrderStatusInfo.RandomEvent:
        raise RejectedOrderError("Random event happened")
    if order_result.status_info == OrderStatusInfo.CantCoverFees:
        raise RejectedOrderError("Not enough cash to cover fees")
    if order_result.status_info == OrderStatusInfo.MinSizeNotReached:
        raise RejectedOrderError("Final size is less than minimum allowed")
    if order_result.status_info == OrderStatusInfo.PartialFill:
        raise RejectedOrderError("Final size is less than requested")
    raise RejectedOrderError


@register_jitted(cache=True)
def process_order_nb(
    group: int,
    col: int,
    i: int,
    exec_state: ExecState,
    order: Order,
    price_area: PriceArea = NoPriceArea,
    update_value: bool = False,
    order_records: tp.Optional[tp.RecordArray2d] = None,
    order_counts: tp.Optional[tp.Array1d] = None,
    log_records: tp.Optional[tp.RecordArray2d] = None,
    log_counts: tp.Optional[tp.Array1d] = None,
) -> tp.Tuple[OrderResult, ExecState]:
    """Process an order by executing it, logging details, and returning the result with updated state.

    Args:
        group (int): The current group index.
        col (int): The current column index.
        i (int): The current row index.
        exec_state (ExecState): The current execution state before processing the order.
        order (Order): The order details to be processed.
        price_area (PriceArea): The price area data used in order execution.
        update_value (bool): Flag indicating whether to update the portfolio value.
        order_records (Optional[RecordArray2d]): The array to store order record details.
        order_counts (Optional[Array1d]): Counters for order records per column.
        log_records (Optional[RecordArray2d]): The array to store detailed log records.
        log_counts (Optional[Array1d]): Counters for log records per column.

    Returns:
        Tuple[OrderResult, ExecState]: A tuple containing the result of the order execution and
            the updated execution state.
    """
    # Execute the order
    order_result, new_exec_state = execute_order_nb(
        exec_state=exec_state,
        order=order,
        price_area=price_area,
        update_value=update_value,
    )

    is_filled = order_result.status == OrderStatus.Filled
    if order_records is not None and order_counts is not None:
        if is_filled and order_records.shape[0] > 0:
            # Fill order record
            if order_counts[col] >= order_records.shape[0]:
                raise IndexError("order_records index out of range. Set a higher max_order_records.")
            fill_order_record_nb(order_records, order_counts[col], col, i, order_result)
            order_counts[col] += 1

    if log_records is not None and log_counts is not None:
        if order.log and log_records.shape[0] > 0:
            # Fill log record
            if log_counts[col] >= log_records.shape[0]:
                raise IndexError("log_records index out of range. Set a higher max_log_records.")
            fill_log_record_nb(
                log_records,
                log_counts[col],
                group,
                col,
                i,
                price_area,
                exec_state,
                order,
                order_result,
                new_exec_state,
                order_counts[col] - 1 if order_counts is not None and is_filled else -1,
            )
            log_counts[col] += 1

    return order_result, new_exec_state


@register_jitted(cache=True)
def order_nb(
    size: float = np.inf,
    price: float = np.inf,
    size_type: int = SizeType.Amount,
    direction: int = Direction.Both,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    leverage_mode: int = LeverageMode.Lazy,
    reject_prob: float = 0.0,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    raise_reject: bool = False,
    log: bool = False,
) -> Order:
    """Create an order.

    Args:
        size (float): Order size.
        price (float): Order price.
        size_type (int): Type of order size.

            See `vectorbtpro.portfolio.enums.SizeType` for details.
        direction (int): Order direction.

            See `vectorbtpro.portfolio.enums.Direction` for details.
        fees (float): Fee rate or amount.
        fixed_fees (float): Fixed fees.
        slippage (float): Expected slippage.
        min_size (float): Minimum order size.
        max_size (float): Maximum order size.
        size_granularity (float): Granularity for order size.
        leverage (float): Leverage multiplier.
        leverage_mode (int): Leverage mode.

            See `vectorbtpro.portfolio.enums.LeverageMode` for details.
        reject_prob (float): Rejection probability threshold.
        price_area_vio_mode (int): Price area violation mode.

            See `vectorbtpro.portfolio.enums.PriceAreaVioMode` for details.
        allow_partial (bool): Allow partial order fulfillment.
        raise_reject (bool): Raise an exception if the order is rejected.
        log (bool): Log the order execution.

    Returns:
        Order: Created order object.
    """
    return Order(
        size=float(size),
        price=float(price),
        size_type=int(size_type),
        direction=int(direction),
        fees=float(fees),
        fixed_fees=float(fixed_fees),
        slippage=float(slippage),
        min_size=float(min_size),
        max_size=float(max_size),
        size_granularity=float(size_granularity),
        leverage=float(leverage),
        leverage_mode=int(leverage_mode),
        reject_prob=float(reject_prob),
        price_area_vio_mode=int(price_area_vio_mode),
        allow_partial=bool(allow_partial),
        raise_reject=bool(raise_reject),
        log=bool(log),
    )


@register_jitted(cache=True)
def close_position_nb(
    price: float = np.inf,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    leverage_mode: int = LeverageMode.Lazy,
    reject_prob: float = 0.0,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    raise_reject: bool = False,
    log: bool = False,
) -> Order:
    """Close the current position.

    Args:
        price (float): Order price.
        fees (float): Fee rate or amount.
        fixed_fees (float): Fixed fees.
        slippage (float): Expected slippage.
        min_size (float): Minimum order size.
        max_size (float): Maximum order size.
        size_granularity (float): Granularity for order size.
        leverage (float): Leverage multiplier.
        leverage_mode (int): Leverage mode.

            See `vectorbtpro.portfolio.enums.LeverageMode` for details.
        reject_prob (float): Rejection probability threshold.
        price_area_vio_mode (int): Price area violation mode.

            See `vectorbtpro.portfolio.enums.PriceAreaVioMode` for details.
        allow_partial (bool): Allow partial order fulfillment.
        raise_reject (bool): Raise an exception if the order is rejected.
        log (bool): Log the order execution.

    Returns:
        Order: Order object representing the closed position.
    """
    return order_nb(
        size=0.0,
        price=price,
        size_type=SizeType.TargetAmount,
        direction=Direction.Both,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        min_size=min_size,
        max_size=max_size,
        size_granularity=size_granularity,
        leverage=leverage,
        leverage_mode=leverage_mode,
        reject_prob=reject_prob,
        price_area_vio_mode=price_area_vio_mode,
        allow_partial=allow_partial,
        raise_reject=raise_reject,
        log=log,
    )


@register_jitted(cache=True)
def order_nothing_nb() -> Order:
    """Create an order representing no operation.

    Returns:
        Order: An order that effectively does nothing.
    """
    return NoOrder


@register_jitted(cache=True)
def check_group_lens_nb(group_lens: tp.GroupLens, n_cols: int) -> None:
    """Ensure the group lengths sum to the total number of columns.

    Args:
        group_lens (GroupLens): An array specifying the length of each group.
        n_cols (int): Total expected number of columns.

    Returns:
        None
    """
    if np.sum(group_lens) != n_cols:
        raise ValueError("group_lens has incorrect total number of columns")


@register_jitted(cache=True)
def is_grouped_nb(group_lens: tp.GroupLens) -> bool:
    """Determine if columns are grouped.

    Args:
        group_lens (GroupLens): Array of group lengths.

    Returns:
        bool: True if at least one group contains more than one column, otherwise False.
    """
    return np.any(group_lens > 1)


@register_jitted(cache=True)
def prepare_records_nb(
    target_shape: tp.Shape,
    max_order_records: tp.Optional[int] = None,
    max_log_records: tp.Optional[int] = 0,
) -> tp.Tuple[tp.RecordArray2d, tp.RecordArray2d]:
    """Prepare order and log records arrays.

    Args:
        target_shape (Shape): Target shape to determine array dimensions.
        max_order_records (Optional[int]): Maximum number of order records.

            If None, `target_shape[0]` is used.
        max_log_records (Optional[int]): Maximum number of log records.

            If None, `target_shape[0]` is used.

    Returns:
        Tuple[RecordArray2d, RecordArray2d]: Tuple containing the order records array and
            the log records array.
    """
    if max_order_records is None:
        order_records = np.empty((target_shape[0], target_shape[1]), dtype=order_dt)
    else:
        order_records = np.empty((max_order_records, target_shape[1]), dtype=order_dt)
    if max_log_records is None:
        log_records = np.empty((target_shape[0], target_shape[1]), dtype=log_dt)
    else:
        log_records = np.empty((max_log_records, target_shape[1]), dtype=log_dt)
    return order_records, log_records


@register_jitted(cache=True)
def prepare_last_cash_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    cash_sharing: bool,
    init_cash: tp.FlexArray1d,
) -> tp.Array1d:
    """Prepare the last cash array based on cash sharing.

    Args:
        target_shape (Shape): Target shape used to determine dimensions.
        group_lens (GroupLens): Array of group lengths.
        cash_sharing (bool): Indicates whether cash is shared across columns within each group.
        init_cash (FlexArray1d): Array of initial cash values.

    Returns:
        Array1d: Array containing the last cash values per group or column.
    """
    if cash_sharing:
        last_cash = np.empty(len(group_lens), dtype=float_)
        for group in range(len(group_lens)):
            last_cash[group] = float(flex_select_1d_pc_nb(init_cash, group))
    else:
        last_cash = np.empty(target_shape[1], dtype=float_)
        for col in range(target_shape[1]):
            last_cash[col] = float(flex_select_1d_pc_nb(init_cash, col))
    return last_cash


@register_jitted(cache=True)
def prepare_last_position_nb(target_shape: tp.Shape, init_position: tp.FlexArray1d) -> tp.Array1d:
    """Prepare the last position array.

    Args:
        target_shape (Shape): Target shape for determining the number of columns.
        init_position (FlexArray1d): Array of initial position values.

    Returns:
        Array1d: Array containing the last positions for each column.
    """
    last_position = np.empty(target_shape[1], dtype=float_)
    for col in range(target_shape[1]):
        last_position[col] = float(flex_select_1d_pc_nb(init_position, col))
    return last_position


@register_jitted(cache=True)
def prepare_last_value_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    cash_sharing: bool,
    init_cash: tp.FlexArray1d,
    init_position: tp.FlexArray1d,
    init_price: tp.FlexArray1d,
) -> tp.Array1d:
    """Prepare the last value array by combining cash and position values.

    Args:
        target_shape (Shape): Target shape used to determine the output dimensions.
        group_lens (GroupLens): Array of group lengths.
        cash_sharing (bool): Indicates if cash is shared among columns within each group.
        init_cash (FlexArray1d): Array of initial cash values.
        init_position (FlexArray1d): Array of initial position values.
        init_price (FlexArray1d): Array of initial prices corresponding to positions.

    Returns:
        Array1d: Array with computed last values calculated as cash plus position times price.
    """
    if cash_sharing:
        last_value = np.empty(len(group_lens), dtype=float_)
        from_col = 0
        for group in range(len(group_lens)):
            to_col = from_col + group_lens[group]
            _init_cash = float(flex_select_1d_pc_nb(init_cash, group))
            last_value[group] = _init_cash
            for col in range(from_col, to_col):
                _init_position = float(flex_select_1d_pc_nb(init_position, col))
                _init_price = float(flex_select_1d_pc_nb(init_price, col))
                if _init_position != 0:
                    last_value[group] += _init_position * _init_price
            from_col = to_col
    else:
        last_value = np.empty(target_shape[1], dtype=float_)
        for col in range(target_shape[1]):
            _init_cash = float(flex_select_1d_pc_nb(init_cash, col))
            _init_position = float(flex_select_1d_pc_nb(init_position, col))
            _init_price = float(flex_select_1d_pc_nb(init_price, col))
            if _init_position == 0:
                last_value[col] = _init_cash
            else:
                last_value[col] = _init_cash + _init_position * _init_price
    return last_value


@register_jitted(cache=True)
def prepare_last_pos_info_nb(
    target_shape: tp.Shape,
    init_position: tp.FlexArray1d,
    init_price: tp.FlexArray1d,
    fill_pos_info: bool = True,
) -> tp.RecordArray:
    """Prepare last position information array.

    Args:
        target_shape (Shape): Target shape for the position info array.
        init_position (FlexArray1d): Array containing initial positions.
        init_price (FlexArray1d): Array containing initial prices.
        fill_pos_info (bool): If True, fill the position info with initial data.

    Returns:
        RecordArray: Array of last position information records.
    """
    if fill_pos_info:
        last_pos_info = np.empty(target_shape[1], dtype=trade_dt)
        last_pos_info["id"][:] = -1
        last_pos_info["col"][:] = -1
        last_pos_info["size"][:] = np.nan
        last_pos_info["entry_order_id"][:] = -1
        last_pos_info["entry_idx"][:] = -1
        last_pos_info["entry_price"][:] = np.nan
        last_pos_info["entry_fees"][:] = np.nan
        last_pos_info["exit_order_id"][:] = -1
        last_pos_info["exit_idx"][:] = -1
        last_pos_info["exit_price"][:] = np.nan
        last_pos_info["exit_fees"][:] = np.nan
        last_pos_info["pnl"][:] = np.nan
        last_pos_info["return"][:] = np.nan
        last_pos_info["direction"][:] = -1
        last_pos_info["status"][:] = -1
        last_pos_info["parent_id"][:] = -1

        for col in range(target_shape[1]):
            _init_position = float(flex_select_1d_pc_nb(init_position, col))
            _init_price = float(flex_select_1d_pc_nb(init_price, col))
            if _init_position != 0:
                fill_init_pos_info_nb(last_pos_info[col], col, _init_position, _init_price)
    else:
        last_pos_info = np.empty(0, dtype=trade_dt)
    return last_pos_info


@register_jitted
def prepare_sim_out_nb(
    order_records: tp.RecordArray2d,
    order_counts: tp.Array1d,
    log_records: tp.RecordArray2d,
    log_counts: tp.Array1d,
    cash_deposits: tp.Array2d,
    cash_earnings: tp.Array2d,
    call_seq: tp.Optional[tp.Array2d] = None,
    in_outputs: tp.Optional[tp.NamedTuple] = None,
    sim_start: tp.Optional[tp.Array1d] = None,
    sim_end: tp.Optional[tp.Array1d] = None,
) -> SimulationOutput:
    """Prepare simulation output.

    Args:
        order_records (RecordArray2d): Array of order records.
        order_counts (Array1d): Array containing order counts.
        log_records (RecordArray2d): Array of log records.
        log_counts (Array1d): Array containing log counts.
        cash_deposits (Array2d): Array of cash deposit entries.
        cash_earnings (Array2d): Array of cash earning entries.
        call_seq (Optional[Array2d]): Optional array for call sequence.
        in_outputs (Optional[NamedTuple]): Optional additional outputs.
        sim_start (Optional[Array1d]): Optional simulation start times.
        sim_end (Optional[Array1d]): Optional simulation end times.

    Returns:
        SimulationOutput: Simulation output record with repartitioned order and
            log records and cash entries.
    """
    order_records_flat = generic_nb.repartition_nb(order_records, order_counts)
    log_records_flat = generic_nb.repartition_nb(log_records, log_counts)
    return SimulationOutput(
        order_records=order_records_flat,
        log_records=log_records_flat,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        call_seq=call_seq,
        in_outputs=in_outputs,
        sim_start=sim_start,
        sim_end=sim_end,
    )


@register_jitted(cache=True)
def get_trade_stats_nb(
    size: float,
    entry_price: float,
    entry_fees: float,
    exit_price: float,
    exit_fees: float,
    direction: int,
) -> tp.Tuple[float, float]:
    """Get trade statistics.

    Args:
        size (float): Trade size.
        entry_price (float): Entry price of the trade.
        entry_fees (float): Fees incurred at entry.
        exit_price (float): Exit price of the trade.
        exit_fees (float): Fees incurred at exit.
        direction (int): Trade direction.

            See `vectorbtpro.portfolio.enums.TradeDirection` for details.

    Returns:
        Tuple[float, float]: A tuple containing the profit and loss (pnl) and the return ratio.
    """
    entry_val = size * entry_price
    exit_val = size * exit_price
    val_diff = add_nb(exit_val, -entry_val)
    if val_diff != 0 and direction == TradeDirection.Short:
        val_diff *= -1
    pnl = val_diff - entry_fees - exit_fees
    if is_close_nb(entry_val, 0):
        ret = np.nan
    else:
        ret = pnl / entry_val
    return pnl, ret


@register_jitted(cache=True)
def update_open_pos_info_stats_nb(record: tp.Record, position_now: float, price: float) -> None:
    """Update statistics of an open position record using a custom price.

    Args:
        record (Record): Position record to update.
        position_now (float): Current position size.
        price (float): Price used for updating the statistics.

    Returns:
        None: This function modifies `record` in place.
    """
    if record["id"] >= 0 and record["status"] == TradeStatus.Open:
        if np.isnan(record["exit_price"]):
            exit_price = price
        else:
            exit_size_sum = record["size"] - abs(position_now)
            exit_gross_sum = exit_size_sum * record["exit_price"]
            exit_gross_sum += abs(position_now) * price
            exit_price = exit_gross_sum / record["size"]
        pnl, ret = get_trade_stats_nb(
            record["size"],
            record["entry_price"],
            record["entry_fees"],
            exit_price,
            record["exit_fees"],
            record["direction"],
        )
        record["pnl"] = pnl
        record["return"] = ret


@register_jitted(cache=True)
def fill_init_pos_info_nb(record: tp.Record, col: int, position_now: float, price: float) -> None:
    """Fill a position record for an initial position.

    Args:
        record (Record): The position record to populate.
        col (int): Current column index.
        position_now (float): The initial position size.
        price (float): The initial price.

    Returns:
        None: This function modifies `record` in place.
    """
    record["id"] = 0
    record["col"] = col
    record["size"] = abs(position_now)
    record["entry_order_id"] = -1
    record["entry_idx"] = -1
    record["entry_price"] = price
    record["entry_fees"] = 0.0
    record["exit_order_id"] = -1
    record["exit_idx"] = -1
    record["exit_price"] = np.nan
    record["exit_fees"] = 0.0
    if position_now >= 0:
        record["direction"] = TradeDirection.Long
    else:
        record["direction"] = TradeDirection.Short
    record["status"] = TradeStatus.Open
    record["parent_id"] = record["id"]

    # Update open position stats
    update_open_pos_info_stats_nb(record, position_now, np.nan)


@register_jitted(cache=True)
def update_pos_info_nb(
    record: tp.Record,
    i: int,
    col: int,
    position_before: float,
    position_now: float,
    order_result: OrderResult,
    order_id: int,
) -> None:
    """Update the position record after an order is filled.

    Args:
        record (Record): Record structure containing trade details.
        i (int): Current row index.
        col (int): Current column index.
        position_before (float): Position size before the order execution.
        position_now (float): Position size after the order execution.
        order_result (OrderResult): Object containing order execution details.
        order_id (int): Identifier of the executed order.

    Returns:
        None: This function modifies `record` in place.
    """
    if order_result.status == OrderStatus.Filled:
        if position_before == 0 and position_now != 0:
            # New position opened
            record["id"] += 1
            record["col"] = col
            record["size"] = order_result.size
            record["entry_order_id"] = order_id
            record["entry_idx"] = i
            record["entry_price"] = order_result.price
            record["entry_fees"] = order_result.fees
            record["exit_order_id"] = -1
            record["exit_idx"] = -1
            record["exit_price"] = np.nan
            record["exit_fees"] = 0.0
            if order_result.side == OrderSide.Buy:
                record["direction"] = TradeDirection.Long
            else:
                record["direction"] = TradeDirection.Short
            record["status"] = TradeStatus.Open
            record["parent_id"] = record["id"]
        elif position_before != 0 and position_now == 0:
            # Position closed
            record["exit_order_id"] = order_id
            record["exit_idx"] = i
            if np.isnan(record["exit_price"]):
                exit_price = order_result.price
            else:
                exit_size_sum = record["size"] - abs(position_before)
                exit_gross_sum = exit_size_sum * record["exit_price"]
                exit_gross_sum += abs(position_before) * order_result.price
                exit_price = exit_gross_sum / record["size"]
            record["exit_price"] = exit_price
            record["exit_fees"] += order_result.fees
            pnl, ret = get_trade_stats_nb(
                record["size"],
                record["entry_price"],
                record["entry_fees"],
                record["exit_price"],
                record["exit_fees"],
                record["direction"],
            )
            record["pnl"] = pnl
            record["return"] = ret
            record["status"] = TradeStatus.Closed
        elif np.sign(position_before) != np.sign(position_now):
            # Position reversed
            record["id"] += 1
            record["size"] = abs(position_now)
            record["entry_order_id"] = order_id
            record["entry_idx"] = i
            record["entry_price"] = order_result.price
            new_pos_fraction = abs(position_now) / abs(position_now - position_before)
            record["entry_fees"] = new_pos_fraction * order_result.fees
            record["exit_order_id"] = -1
            record["exit_idx"] = -1
            record["exit_price"] = np.nan
            record["exit_fees"] = 0.0
            if order_result.side == OrderSide.Buy:
                record["direction"] = TradeDirection.Long
            else:
                record["direction"] = TradeDirection.Short
            record["status"] = TradeStatus.Open
            record["parent_id"] = record["id"]
        else:
            # Position changed
            if abs(position_before) <= abs(position_now):
                # Position increased
                entry_gross_sum = record["size"] * record["entry_price"]
                entry_gross_sum += order_result.size * order_result.price
                entry_price = entry_gross_sum / (record["size"] + order_result.size)
                record["entry_price"] = entry_price
                record["entry_fees"] += order_result.fees
                record["size"] += order_result.size
            else:
                # Position decreased
                record["exit_order_id"] = order_id
                if np.isnan(record["exit_price"]):
                    exit_price = order_result.price
                else:
                    exit_size_sum = record["size"] - abs(position_before)
                    exit_gross_sum = exit_size_sum * record["exit_price"]
                    exit_gross_sum += order_result.size * order_result.price
                    exit_price = exit_gross_sum / (exit_size_sum + order_result.size)
                record["exit_price"] = exit_price
                record["exit_fees"] += order_result.fees

        # Update open position stats
        update_open_pos_info_stats_nb(record, position_now, order_result.price)


@register_jitted(cache=True)
def resolve_hl_nb(open: float, high: float, low: float, close: float) -> tp.Tuple[float, float]:
    """Resolve the current high and low prices based on OHLC data.

    Args:
        open (float): The opening price.
        high (float): The high price.
        low (float): The low price.
        close (float): The closing price.

    Returns:
        Tuple[float, float]: The resolved high and low prices.
    """
    if np.isnan(high):
        if np.isnan(open):
            high = close
        elif np.isnan(close):
            high = open
        else:
            high = max(open, close)
    if np.isnan(low):
        if np.isnan(open):
            low = close
        elif np.isnan(close):
            low = open
        else:
            low = min(open, close)
    return high, low


@register_jitted(cache=True)
def check_price_hit_nb(
    open: float,
    high: float,
    low: float,
    close: float,
    price: float,
    hit_below: bool = True,
    can_use_ohlc: bool = True,
    check_open: bool = True,
    hard_price: bool = False,
) -> tp.Tuple[float, bool, bool]:
    """Determine whether the target price is hit and compute the effective price and hit flags.

    Args:
        open (float): The opening price.
        high (float): The high price.
        low (float): The low price.
        close (float): The closing price.
        price (float): The target price to evaluate.
        hit_below (bool): If True, check whether the target price is hit from above.
        can_use_ohlc (bool): Indicates if OHLC data should be used for the evaluation.
        check_open (bool): Determines whether the open price should be considered.
        hard_price (bool): If True, enforces the target price when hit by the open price.

    Returns:
        Tuple[float, bool, bool]: A tuple containing:

            * The effective stop price,
            * A flag indicating if the open price was used,
            * A flag indicating if the price was hit during the bar.
    """
    high, low = resolve_hl_nb(
        open=open,
        high=high,
        low=low,
        close=close,
    )
    if hit_below:
        if can_use_ohlc and check_open and is_close_or_less_nb(open, price):
            if hard_price:
                return price, True, True
            return open, True, True
        if is_close_or_less_nb(close, price) or (can_use_ohlc and is_close_or_less_nb(low, price)):
            return price, False, True
        return price, False, False
    if can_use_ohlc and check_open and is_close_or_greater_nb(open, price):
        if hard_price:
            return price, True, True
        return open, True, True
    if is_close_or_greater_nb(close, price) or (can_use_ohlc and is_close_or_greater_nb(high, price)):
        return price, False, True
    return price, False, False


@register_jitted(cache=True)
def resolve_stop_exit_price_nb(
    stop_price: float,
    close: float,
    stop_exit_price: float,
) -> float:
    """Resolve the exit price for a stop order based on the specified stop exit option.

    Args:
        stop_price (float): The computed stop price.
        close (float): The closing price.
        stop_exit_price (float): Option indicating how to determine the exit price.

            If equal to `StopExitPrice.Stop` or `StopExitPrice.HardStop`, the stop price is used;
            if equal to `StopExitPrice.Close`, the closing price is used; otherwise, the provided value is used.

    Returns:
        float: The resolved exit price.
    """
    if stop_exit_price == StopExitPrice.Stop or stop_exit_price == StopExitPrice.HardStop:
        return float(stop_price)
    elif stop_exit_price == StopExitPrice.Close:
        return float(close)
    elif stop_exit_price < 0:
        raise ValueError("Invalid StopExitPrice option")
    return float(stop_exit_price)


@register_jitted(cache=True)
def is_limit_active_nb(init_idx: int, init_price: float) -> bool:
    """Check whether a limit order is active.

    Args:
        init_idx (int): The initial row index for the limit order; -1 indicates inactivity.
        init_price (float): The initial price for the limit order; NaN implies inactivity.

    Returns:
        bool: True if the limit order is active, otherwise False.
    """
    return init_idx != -1 and not np.isnan(init_price)


@register_jitted(cache=True)
def is_stop_active_nb(init_idx: int, stop: float) -> bool:
    """Check whether a stop order is active.

    Args:
        init_idx (int): The initial row index for the stop order; -1 indicates inactivity.
        stop (float): The stop value; NaN indicates that the stop is inactive.

    Returns:
        bool: True if the stop order is active, otherwise False.
    """
    return init_idx != -1 and not np.isnan(stop)


@register_jitted(cache=True)
def is_time_stop_active_nb(init_idx: int, stop: int) -> bool:
    """Check whether a time stop order is active.

    Args:
        init_idx (int): The initial row index for the time stop order; -1 indicates inactivity.
        stop (int): The time stop value; -1 signifies that the stop is inactive.

    Returns:
        bool: True if the time stop order is active, otherwise False.
    """
    return init_idx != -1 and stop != -1


@register_jitted(cache=True)
def should_update_stop_nb(new_stop: float, upon_stop_update: int) -> bool:
    """Determine whether the stop value should be updated based on the update mode.

    Args:
        new_stop (float): The new candidate for the stop value.
        upon_stop_update (int): The mode for updating the stop value, where `StopUpdateMode.Keep`
            means no update, and `StopUpdateMode.Override` or `StopUpdateMode.OverrideNaN`
            may trigger an update.

    Returns:
        bool: True if the stop value should be updated, otherwise False.
    """
    if upon_stop_update == StopUpdateMode.Keep:
        return False
    if upon_stop_update == StopUpdateMode.Override or upon_stop_update == StopUpdateMode.OverrideNaN:
        if not np.isnan(new_stop) or upon_stop_update == StopUpdateMode.OverrideNaN:
            return True
        return False
    raise ValueError("Invalid StopUpdateMode option")


@register_jitted(cache=True)
def should_update_time_stop_nb(new_stop: int, upon_stop_update: int) -> bool:
    """Determine whether the time stop value should be updated based on the update mode.

    Args:
        new_stop (int): The new candidate for the time stop value.
        upon_stop_update (int): The mode for updating the time stop value.

            Mode `StopUpdateMode.Keep` indicates no update, while `StopUpdateMode.Override` or
            `StopUpdateMode.OverrideNaN` may trigger an update.

    Returns:
        bool: True if the time stop value should be updated, otherwise False.
    """
    if upon_stop_update == StopUpdateMode.Keep:
        return False
    if upon_stop_update == StopUpdateMode.Override or upon_stop_update == StopUpdateMode.OverrideNaN:
        if new_stop != -1 or upon_stop_update == StopUpdateMode.OverrideNaN:
            return True
        return False
    raise ValueError("Invalid StopUpdateMode option")


@register_jitted(cache=True)
def check_limit_expired_nb(
    creation_idx: int,
    i: int,
    tif: int = -1,
    expiry: int = -1,
    time_delta_format: int = TimeDeltaFormat.Index,
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
) -> tp.Tuple[bool, bool]:
    """Check whether the limit order is expired based on the current and creation indices.

    Args:
        creation_idx (int): The row index at which the limit order was created.
        i (int): The current row index.
        tif (int): The time-in-force duration; use -1 if undefined.
        expiry (int): The explicit expiry time; use -1 if undefined.
        time_delta_format (int): The format for calculating time differences.

            See `vectorbtpro.portfolio.enums.TimeDeltaFormat` for options.
        index (Optional[Array1d]): Array of time indices, required when using index-based time deltas.

            Must be in nanosecond format.
        freq (Optional[int]): Frequency value for time calculations when using index format.

            Must be in nanosecond format.

    Returns:
        Tuple[bool, bool]: A tuple where the first element indicates if the limit expires on open,
            and the second element indicates if it expires during the current bar.
    """
    if tif == -1 and expiry == -1:
        return False, False
    if time_delta_format == TimeDeltaFormat.Rows:
        is_expired_on_open = False
        is_expired = False
        if tif != -1:
            if creation_idx + tif <= i:
                is_expired_on_open = True
                is_expired = True
            elif i < creation_idx + tif < i + 1:
                is_expired = True
        if expiry != -1:
            if expiry <= i:
                is_expired_on_open = True
                is_expired = True
            elif i < expiry < i + 1:
                is_expired = True
        return is_expired_on_open, is_expired
    elif time_delta_format == TimeDeltaFormat.Index:
        if index is None:
            raise ValueError("Must provide index for TimeDeltaFormat.Index")
        if freq is None:
            raise ValueError("Must provide frequency for TimeDeltaFormat.Index")
        is_expired_on_open = False
        is_expired = False
        if tif != -1:
            if index[creation_idx] + tif <= index[i]:
                is_expired_on_open = True
                is_expired = True
            elif index[i] < index[creation_idx] + tif < index[i] + freq:
                is_expired = True
        if expiry != -1:
            if expiry <= index[i]:
                is_expired_on_open = True
                is_expired = True
            elif index[i] < expiry < index[i] + freq:
                is_expired = True
        return is_expired_on_open, is_expired
    else:
        raise ValueError("Invalid TimeDeltaFormat option")


@register_jitted(cache=True)
def resolve_limit_price_nb(
    init_price: float,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    hit_below: bool = True,
) -> float:
    """Resolve the limit price based on the initial price and adjustment delta.

    Args:
        init_price (float): The initial reference price.
        limit_delta (float): The delta value used to adjust the limit price.
        delta_format (int): The format of the delta.

            See `vectorbtpro.portfolio.enums.DeltaFormat` for options.
        hit_below (bool): Indicates whether the order should hit below the reference price.

    Returns:
        float: The computed limit price.
    """
    if delta_format == DeltaFormat.Percent100:
        limit_delta /= 100
        delta_format = DeltaFormat.Percent
    if not np.isnan(limit_delta):
        if hit_below:
            if np.isinf(limit_delta) and delta_format != DeltaFormat.Target:
                if limit_delta > 0:
                    limit_price = -np.inf
                else:
                    limit_price = np.inf
            else:
                if delta_format == DeltaFormat.Absolute:
                    limit_price = init_price - limit_delta
                elif delta_format == DeltaFormat.Percent:
                    limit_price = init_price * (1 - limit_delta)
                elif delta_format == DeltaFormat.Target:
                    limit_price = limit_delta
                else:
                    raise ValueError("Invalid DeltaFormat option")
        else:
            if np.isinf(limit_delta) and delta_format != DeltaFormat.Target:
                if limit_delta < 0:
                    limit_price = -np.inf
                else:
                    limit_price = np.inf
            else:
                if delta_format == DeltaFormat.Absolute:
                    limit_price = init_price + limit_delta
                elif delta_format == DeltaFormat.Percent:
                    limit_price = init_price * (1 + limit_delta)
                elif delta_format == DeltaFormat.Target:
                    limit_price = limit_delta
                else:
                    raise ValueError("Invalid DeltaFormat option")
    else:
        limit_price = init_price
    return limit_price


@register_jitted(cache=True)
def check_limit_hit_nb(
    open: float,
    high: float,
    low: float,
    close: float,
    price: float,
    size: float,
    direction: int = Direction.Both,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    limit_reverse: bool = False,
    can_use_ohlc: bool = True,
    check_open: bool = True,
    hard_limit: bool = False,
) -> tp.Tuple[float, bool, bool]:
    """Resolve the limit price using `resolve_limit_price_nb` and determine if the limit was hit.

    Args:
        open (float): The opening price.
        high (float): The highest price of the period.
        low (float): The lowest price of the period.
        close (float): The closing price.
        price (float): The reference price used for calculating the limit.
        size (float): The order size (must be non-zero).
        direction (int): The order direction.

            See `vectorbtpro.portfolio.enums.Direction` for options.
        limit_delta (float): The delta adjustment for computing the limit price.
        delta_format (int): The format of the delta.

            See `vectorbtpro.portfolio.enums.DeltaFormat` for options.
        limit_reverse (bool): Flag to reverse the limit condition.
        can_use_ohlc (bool): Indicates whether OHLC data can be used for detecting a limit hit.
        check_open (bool): Flag indicating whether to check if the limit is hit at the open price.
        hard_limit (bool): Enforces a hard limit without fallback to the open price if True.

    Returns:
        Tuple[float, bool, bool]: A tuple containing:

            * The computed limit price (which may be adjusted to the open price if conditions are met).
            * True if the limit was hit before open; otherwise, False.
            * True if the limit was hit during the current bar; otherwise, False.
    """
    if size == 0:
        raise ValueError("Limit order size cannot be zero")
    _size = get_diraware_size_nb(size, direction)
    hit_below = (_size > 0 and not limit_reverse) or (_size < 0 and limit_reverse)
    limit_price = resolve_limit_price_nb(
        init_price=price,
        limit_delta=limit_delta,
        delta_format=delta_format,
        hit_below=hit_below,
    )
    hit_on_open = False

    if can_use_ohlc:
        high, low = resolve_hl_nb(
            open=open,
            high=high,
            low=low,
            close=close,
        )
        if hit_below:
            if check_open and is_close_or_less_nb(open, limit_price):
                hit_on_open = True
                hit = True
                if not hard_limit:
                    limit_price = open
            else:
                hit = is_close_or_less_nb(low, limit_price)
                if hit and np.isinf(limit_price):
                    limit_price = low
        else:
            if check_open and is_close_or_greater_nb(open, limit_price):
                hit_on_open = True
                hit = True
                if not hard_limit:
                    limit_price = open
            else:
                hit = is_close_or_greater_nb(high, limit_price)
                if hit and np.isinf(limit_price):
                    limit_price = high
    else:
        if hit_below:
            hit = is_close_or_less_nb(close, limit_price)
        else:
            hit = is_close_or_greater_nb(close, limit_price)
        if hit and np.isinf(limit_price):
            limit_price = close
    return limit_price, hit_on_open, hit


@register_jitted(cache=True)
def resolve_limit_order_price_nb(
    limit_price: float,
    close: float,
    limit_order_price: float,
) -> float:
    """Determine the appropriate limit order price based on the specified option.

    Args:
        limit_price (float): The computed limit price.
        close (float): The closing price.
        limit_order_price (float): An option to select the price from
            `vectorbtpro.portfolio.enums.LimitOrderPrice`:

            * If equal to `LimitOrderPrice.Limit` or `LimitOrderPrice.HardLimit`, returns `limit_price`.
            * If equal to `LimitOrderPrice.Close`, returns `close`.
            * Otherwise, returns the provided value.

    Returns:
        float: The resolved limit order price.

    Raises:
        ValueError: If `limit_order_price` is less than 0.
    """
    if limit_order_price == LimitOrderPrice.Limit or limit_order_price == LimitOrderPrice.HardLimit:
        return float(limit_price)
    elif limit_order_price == LimitOrderPrice.Close:
        return float(close)
    elif limit_order_price < 0:
        raise ValueError("Invalid LimitOrderPrice option")
    return float(limit_order_price)


@register_jitted(cache=True)
def resolve_stop_price_nb(
    init_price: float,
    stop: float,
    delta_format: int = DeltaFormat.Percent,
    hit_below: bool = True,
) -> float:
    """Resolve the stop price based on the initial price and adjustment value.

    Args:
        init_price (float): The initial reference price.
        stop (float): The stop adjustment value.
        delta_format (int): The format of the stop adjustment.

            See `vectorbtpro.portfolio.enums.DeltaFormat` for options.
        hit_below (bool): Indicates whether the stop should be applied below the initial price.

    Returns:
        float: The computed stop price.
    """
    if delta_format == DeltaFormat.Percent100:
        stop /= 100
        delta_format = DeltaFormat.Percent
    if hit_below:
        if delta_format == DeltaFormat.Absolute:
            stop_price = init_price - abs(stop)
        elif delta_format == DeltaFormat.Percent:
            stop_price = init_price * (1 - abs(stop))
        elif delta_format == DeltaFormat.Target:
            stop_price = stop
        else:
            raise ValueError("Invalid DeltaFormat option")
    else:
        if delta_format == DeltaFormat.Absolute:
            stop_price = init_price + abs(stop)
        elif delta_format == DeltaFormat.Percent:
            stop_price = init_price * (1 + abs(stop))
        elif delta_format == DeltaFormat.Target:
            stop_price = stop
        else:
            raise ValueError("Invalid DeltaFormat option")
    return stop_price


@register_jitted(cache=True)
def check_stop_hit_nb(
    open: float,
    high: float,
    low: float,
    close: float,
    is_position_long: bool,
    init_price: float,
    stop: float,
    delta_format: int = DeltaFormat.Percent,
    hit_below: bool = True,
    can_use_ohlc: bool = True,
    check_open: bool = True,
    hard_stop: bool = False,
) -> tp.Tuple[float, bool, bool]:
    """Resolve the stop price using `resolve_stop_price_nb` and check if it was hit.

    Args:
        open (float): Opening price.
        high (float): High price.
        low (float): Low price.
        close (float): Close price.
        is_position_long (bool): Indicates whether the position is long.
        init_price (float): Initial price used for stop price calculation.
        stop (float): Stop parameter value.
        delta_format (int): Format for the stop delta.

            See `vectorbtpro.portfolio.enums.DeltaFormat` for options.
        hit_below (bool): Direction flag for stop hit determination.
        can_use_ohlc (bool): Whether to use OHLC data when checking the stop hit.
        check_open (bool): Whether to include the open price in the hit check.
        hard_stop (bool): Whether the stop is considered a hard stop.

    Returns:
        Tuple[float, bool, bool]: A tuple containing:

            * The resolved stop price.
            * True if the stop was hit on open.
            * True if the stop was hit during the bar.
    """
    hit_below = (is_position_long and hit_below) or (not is_position_long and not hit_below)
    stop_price = resolve_stop_price_nb(
        init_price=init_price,
        stop=stop,
        delta_format=delta_format,
        hit_below=hit_below,
    )
    return check_price_hit_nb(
        open=open,
        high=high,
        low=low,
        close=close,
        price=stop_price,
        hit_below=hit_below,
        can_use_ohlc=can_use_ohlc,
        check_open=check_open,
        hard_price=hard_stop,
    )


@register_jitted(cache=True)
def check_td_stop_hit_nb(
    init_idx: int,
    i: int,
    stop: int = -1,
    time_delta_format: int = TimeDeltaFormat.Index,
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
) -> tp.Tuple[bool, bool]:
    """Check whether a TD stop was hit by comparing the initial index with the current index.

    Args:
        init_idx (int): Row index of the initial event.
        i (int): Current row index.
        stop (int): Stop offset; -1 indicates no stop.
        time_delta_format (int): Format for time delta comparisons (e.g., rows or index).

            See `vectorbtpro.portfolio.enums.TimeDeltaFormat` for options.
        index (Optional[Array1d]): Array of index values, required when using index-based time delta.

            Must be in nanosecond format.
        freq (Optional[int]): Frequency for time delta comparisons when using index-based format.

            Must be in nanosecond format.

    Returns:
        Tuple[bool, bool]: A tuple containing:

            * True if the stop was hit on open.
            * True if the stop was hit during the current bar.
    """
    if stop == -1:
        return False, False
    if time_delta_format == TimeDeltaFormat.Rows:
        is_hit_on_open = False
        is_hit = False
        if stop != -1:
            if init_idx + stop <= i:
                is_hit_on_open = True
                is_hit = True
            elif i < init_idx + stop < i + 1:
                is_hit = True
        return is_hit_on_open, is_hit
    elif time_delta_format == TimeDeltaFormat.Index:
        if index is None:
            raise ValueError("Must provide index for TimeDeltaFormat.Index")
        if freq is None:
            raise ValueError("Must provide frequency for TimeDeltaFormat.Index")
        is_hit_on_open = False
        is_hit = False
        if stop != -1:
            if index[init_idx] + stop <= index[i]:
                is_hit_on_open = True
                is_hit = True
            elif index[i] < index[init_idx] + stop < index[i] + freq:
                is_hit = True
        return is_hit_on_open, is_hit
    else:
        raise ValueError("Invalid TimeDeltaFormat option")


@register_jitted(cache=True)
def check_dt_stop_hit_nb(
    i: int,
    stop: int = -1,
    time_delta_format: int = TimeDeltaFormat.Index,
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
) -> tp.Tuple[bool, bool]:
    """Check whether a DT stop was hit by comparing the current index against the stop threshold.

    Args:
        i (int): Current row index.
        stop (int): Stop threshold offset; -1 indicates no stop.
        time_delta_format (int): Format for time delta calculations (rows or index).

            See `vectorbtpro.portfolio.enums.TimeDeltaFormat` for options.
        index (Optional[Array1d]): Array of index values, required for index-based time delta.

            Must be in nanosecond format.
        freq (Optional[int]): Frequency for index-based time delta comparisons.

            Must be in nanosecond format.

    Returns:
        Tuple[bool, bool]: A tuple containing:

            * True if the stop was hit on open.
            * True if the stop was hit during the current bar.
    """
    if stop == -1:
        return False, False
    if time_delta_format == TimeDeltaFormat.Rows:
        is_hit_on_open = False
        is_hit = False
        if stop != -1:
            if stop <= i:
                is_hit_on_open = True
                is_hit = True
            elif i < stop < i + 1:
                is_hit = True
        return is_hit_on_open, is_hit
    elif time_delta_format == TimeDeltaFormat.Index:
        if index is None:
            raise ValueError("Must provide index for TimeDeltaFormat.Index")
        if freq is None:
            raise ValueError("Must provide frequency for TimeDeltaFormat.Index")
        is_hit_on_open = False
        is_hit = False
        if stop != -1:
            if stop <= index[i]:
                is_hit_on_open = True
                is_hit = True
            elif index[i] < stop < index[i] + freq:
                is_hit = True
        return is_hit_on_open, is_hit
    else:
        raise ValueError("Invalid TimeDeltaFormat option")


@register_jitted(cache=True)
def check_tsl_th_hit_nb(
    is_position_long: bool,
    init_price: float,
    peak_price: float,
    threshold: float,
    delta_format: int = DeltaFormat.Percent,
) -> bool:
    """Resolve the TSL threshold price using `resolve_stop_price_nb` and check if it was hit.

    Args:
        is_position_long (bool): True if the position is long, False if short.
        init_price (float): Initial price used for threshold computation.
        peak_price (float): Peak price reached (maximum for long, minimum for short).
        threshold (float): TSL threshold value.
        delta_format (int): Format for computing the trailing stop threshold.

            See `vectorbtpro.portfolio.enums.DeltaFormat` for options.

    Returns:
        bool: True if the TSL threshold was hit, False otherwise.
    """
    hit_below = not is_position_long
    tsl_th_price = resolve_stop_price_nb(
        init_price=init_price,
        stop=threshold,
        delta_format=delta_format,
        hit_below=hit_below,
    )
    if hit_below:
        return is_close_or_less_nb(peak_price, tsl_th_price)
    return is_close_or_greater_nb(peak_price, tsl_th_price)


@register_jitted(cache=True)
def resolve_dyn_limit_price_nb(val_price: float, price: float, limit_price: float) -> float:
    """Resolve the dynamic limit price.

    Determines a bounded limit price using the valuation price as the lower bound and
    the order price as the upper bound.

    Args:
        val_price (float): Valuation price used as the lower bound.
        price (float): Order price used as the upper bound.
        limit_price (float): Proposed limit price value; if infinite, selects a bound based on its sign.

    Returns:
        float: The resolved limit price.
    """
    if np.isinf(limit_price):
        if limit_price < 0:
            return float(val_price)
        return float(price)
    return float(limit_price)


@register_jitted(cache=True)
def resolve_dyn_stop_entry_price_nb(val_price: float, price: float, stop_entry_price: float) -> float:
    """Resolve the dynamic stop entry price.

    Determines the stop entry price using the valuation/open price as the lower bound and
    the order price as the upper bound.

    Args:
        val_price (float): Valuation price used as the lower bound.
        price (float): Order price used as the upper bound.
        stop_entry_price (float): Proposed stop entry price value.

            A negative value selects between valuation and order price.

    Returns:
        float: The resolved stop entry price.
    """
    if np.isinf(stop_entry_price):
        if stop_entry_price < 0:
            return float(val_price)
        return float(price)
    if stop_entry_price < 0:
        if stop_entry_price == StopEntryPrice.ValPrice:
            return float(val_price)
        if stop_entry_price == StopEntryPrice.Price:
            return float(price)
        raise ValueError("Only valuation and order price are supported when setting stop entry price dynamically")
    return float(stop_entry_price)


@register_jitted(cache=True)
def get_stop_ladder_exit_size_nb(
    stop_: tp.FlexArray2d,
    step: int,
    col: int,
    init_price: float,
    init_position: float,
    position_now: float,
    ladder: int = StopLadderMode.Disabled,
    delta_format: int = DeltaFormat.Percent,
    hit_below: bool = True,
) -> float:
    """Calculate the exit size for the current ladder step in a stop ladder strategy.

    Args:
        stop_ (FlexArray2d): 2D array of stop values.
        step (int): Current ladder step index.
        col (int): Current column index.
        init_price (float): Initial price used to resolve stop levels.
        init_position (float): Initial position size.
        position_now (float): Current position size.
        ladder (int): Stop ladder mode; must be static.

            See `vectorbtpro.portfolio.enums.StopLadderMode` for options.
        delta_format (int): Format for computing stop deltas.

            See `vectorbtpro.portfolio.enums.DeltaFormat` for options.
        hit_below (bool): Determines the direction for stop price resolution.

    Returns:
        float: The exit size corresponding to the current ladder step, or NaN if no valid stop is found.
    """
    if ladder == StopLadderMode.Disabled:
        raise ValueError("Stop ladder must be enabled to select exit size")
    if ladder == StopLadderMode.Dynamic:
        raise ValueError("Stop ladder must be static to select exit size")
    stop = flex_select_nb(stop_, step, col)
    if np.isnan(stop):
        return np.nan
    last_step = -1
    for i in range(step, stop_.shape[0]):
        if not np.isnan(flex_select_nb(stop_, i, col)):
            last_step = i
        else:
            break
    if last_step == -1:
        return np.nan
    if step == last_step:
        return abs(position_now)

    if ladder == StopLadderMode.Uniform:
        exit_fraction = 1 / (last_step + 1)
        return exit_fraction * abs(init_position)
    if ladder == StopLadderMode.AdaptUniform:
        exit_fraction = 1 / (last_step + 1 - step)
        return exit_fraction * abs(position_now)
    hit_below = (init_position >= 0 and hit_below) or (init_position < 0 and not hit_below)
    price = resolve_stop_price_nb(
        init_price=init_price,
        stop=stop,
        delta_format=delta_format,
        hit_below=hit_below,
    )
    last_stop = flex_select_nb(stop_, last_step, col)
    last_price = resolve_stop_price_nb(
        init_price=init_price,
        stop=last_stop,
        delta_format=delta_format,
        hit_below=hit_below,
    )
    if step == 0:
        prev_price = init_price
    else:
        prev_stop = flex_select_nb(stop_, step - 1, col)
        prev_price = resolve_stop_price_nb(
            init_price=init_price,
            stop=prev_stop,
            delta_format=delta_format,
            hit_below=hit_below,
        )
    if ladder == StopLadderMode.Weighted:
        exit_fraction = (price - prev_price) / (last_price - init_price)
        return exit_fraction * abs(init_position)
    if ladder == StopLadderMode.AdaptWeighted:
        exit_fraction = (price - prev_price) / (last_price - prev_price)
        return exit_fraction * abs(position_now)
    raise ValueError("Invalid StopLadderMode option")


@register_jitted(cache=True)
def get_time_stop_ladder_exit_size_nb(
    stop_: tp.FlexArray2d,
    step: int,
    col: int,
    init_idx: int,
    init_position: float,
    position_now: float,
    ladder: int = StopLadderMode.Disabled,
    time_delta_format: int = TimeDeltaFormat.Index,
    index: tp.Optional[tp.Array1d] = None,
) -> float:
    """Get the exit size corresponding to the current step in the ladder.

    Args:
        stop_ (FlexArray2d): A 2D array representing stop values.
        step (int): Current step index in the ladder.
        col (int): Current column index.
        init_idx (int): Initial row index in the ladder.
        init_position (float): Initial position size.
        position_now (float): Current position size.
        ladder (int): Stop ladder mode.

            See `vectorbtpro.portfolio.enums.StopLadderMode` for available options.
        time_delta_format (int): Time delta format.

            See `vectorbtpro.portfolio.enums.TimeDeltaFormat` for details.
        index (Optional[Array1d]): Array of index values when using index-based time delta.

            Must be in nanosecond format.

    Returns:
        float: The calculated exit size.
    """
    if ladder == StopLadderMode.Disabled:
        raise ValueError("Stop ladder must be enabled to select exit size")
    if ladder == StopLadderMode.Dynamic:
        raise ValueError("Stop ladder must be static to select exit size")
    if init_idx == -1:
        raise ValueError("Initial index of the ladder must be known")
    if time_delta_format == TimeDeltaFormat.Index:
        if index is None:
            raise ValueError("Must provide index for TimeDeltaFormat.Index")
        init_idx = index[init_idx]
    idx = flex_select_nb(stop_, step, col)
    if idx == -1:
        return np.nan
    last_step = -1
    for i in range(step, stop_.shape[0]):
        if flex_select_nb(stop_, i, col) != -1:
            last_step = i
        else:
            break
    if last_step == -1:
        return np.nan
    if step == last_step:
        return abs(position_now)

    if ladder == StopLadderMode.Uniform:
        exit_fraction = 1 / (last_step + 1)
        return exit_fraction * abs(init_position)
    if ladder == StopLadderMode.AdaptUniform:
        exit_fraction = 1 / (last_step + 1 - step)
        return exit_fraction * abs(position_now)
    last_idx = flex_select_nb(stop_, last_step, col)
    if step == 0:
        prev_idx = init_idx
    else:
        prev_idx = flex_select_nb(stop_, step - 1, col)
    if ladder == StopLadderMode.Weighted:
        exit_fraction = (idx - prev_idx) / (last_idx - init_idx)
        return exit_fraction * abs(init_position)
    if ladder == StopLadderMode.AdaptWeighted:
        exit_fraction = (idx - prev_idx) / (last_idx - prev_idx)
        return exit_fraction * abs(position_now)
    raise ValueError("Invalid StopLadderMode option")


@register_jitted(cache=True)
def is_limit_info_active_nb(limit_info: tp.Record) -> bool:
    """Check whether the information record for a limit order is active.

    Args:
        limit_info (Record): Record containing limit order information.

    Returns:
        bool: True if the limit order record is active, otherwise False.
    """
    return is_limit_active_nb(limit_info["init_idx"], limit_info["init_price"])


@register_jitted(cache=True)
def is_stop_info_active_nb(stop_info: tp.Record) -> bool:
    """Check whether the information record for a stop order is active.

    Args:
        stop_info (Record): Record containing stop order information.

    Returns:
        bool: True if the stop order record is active, otherwise False.
    """
    return is_stop_active_nb(stop_info["init_idx"], stop_info["stop"])


@register_jitted(cache=True)
def is_time_stop_info_active_nb(time_stop_info: tp.Record) -> bool:
    """Check whether the information record for a time stop order is active.

    Args:
        time_stop_info (Record): Record containing time stop order information.

    Returns:
        bool: True if the time stop order record is active, otherwise False.
    """
    return is_time_stop_active_nb(time_stop_info["init_idx"], time_stop_info["stop"])


@register_jitted(cache=True)
def is_stop_info_ladder_active_nb(info: tp.Record) -> bool:
    """Check whether the information record for a stop ladder is active.

    Args:
        info (Record): Record containing stop ladder information.

    Returns:
        bool: True if the stop ladder record is active, otherwise False.
    """
    return info["ladder"] != -1 and info["ladder"] > 0 and info["step"] != -1


@register_jitted(cache=True)
def set_limit_info_nb(
    limit_info: tp.Record,
    signal_idx: int,
    creation_idx: tp.Optional[int] = None,
    init_idx: tp.Optional[int] = None,
    init_price: float = -np.inf,
    init_size: float = np.inf,
    init_size_type: int = SizeType.Amount,
    init_direction: int = Direction.Both,
    init_stop_type: int = -1,
    delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    tif: int = -1,
    expiry: int = -1,
    time_delta_format: int = TimeDeltaFormat.Index,
    reverse: bool = False,
    order_price: int = LimitOrderPrice.Limit,
) -> None:
    """Set limit order information in the provided record.

    Args:
        limit_info (Record): Record for limit order information.
        signal_idx (int): Row index of the signal triggering the limit order.
        creation_idx (Optional[int]): Creation row index.
        init_idx (Optional[int]): Initial row index for the order.
        init_price (float): Initial price.
        init_size (float): Initial size for the order.
        init_size_type (int): Indicator for size type.

            See `vectorbtpro.portfolio.enums.SizeType`.
        init_direction (int): Order direction indicator.

            See `vectorbtpro.portfolio.enums.Direction`.
        init_stop_type (int): Initial stop type.

            See `vectorbtpro.portfolio.enums.StopType`.
        delta (float): Delta value associated with the order.
        delta_format (int): Format for the delta value.

            See `vectorbtpro.portfolio.enums.DeltaFormat`.
        tif (int): Time in force parameter.
        expiry (int): Expiry index.
        time_delta_format (int): Time delta format.

            See `vectorbtpro.portfolio.enums.TimeDeltaFormat`.
        reverse (bool): Flag indicating if the order is reversed.
        order_price (int): Order price type.

            See `vectorbtpro.portfolio.enums.LimitOrderPrice`.

    Returns:
        None: The function modifies the `limit_info` record in place.

    See:
        `vectorbtpro.portfolio.enums.limit_info_dt`
    """
    limit_info["signal_idx"] = signal_idx
    limit_info["creation_idx"] = creation_idx if creation_idx is not None else signal_idx
    limit_info["init_idx"] = init_idx if init_idx is not None else signal_idx
    limit_info["init_price"] = init_price
    limit_info["init_size"] = init_size
    limit_info["init_size_type"] = init_size_type
    limit_info["init_direction"] = init_direction
    limit_info["init_stop_type"] = init_stop_type
    limit_info["delta"] = delta
    limit_info["delta_format"] = delta_format
    limit_info["tif"] = tif
    limit_info["expiry"] = expiry
    limit_info["time_delta_format"] = time_delta_format
    limit_info["reverse"] = reverse
    limit_info["order_price"] = order_price


@register_jitted(cache=True)
def clear_limit_info_nb(limit_info: tp.Record) -> None:
    """Clear limit order information in the provided record.

    Args:
        limit_info (Record): Record for limit order information to be cleared.

    Returns:
        None: The function modifies the `limit_info` record in place.
    """
    limit_info["signal_idx"] = -1
    limit_info["creation_idx"] = -1
    limit_info["init_idx"] = -1
    limit_info["init_price"] = np.nan
    limit_info["init_size"] = np.nan
    limit_info["init_size_type"] = -1
    limit_info["init_direction"] = -1
    limit_info["init_stop_type"] = -1
    limit_info["delta"] = np.nan
    limit_info["delta_format"] = -1
    limit_info["tif"] = -1
    limit_info["expiry"] = -1
    limit_info["time_delta_format"] = -1
    limit_info["reverse"] = False
    limit_info["order_price"] = np.nan


@register_jitted(cache=True)
def set_sl_info_nb(
    sl_info: tp.Record,
    init_idx: int,
    init_price: float = -np.inf,
    init_position: float = np.nan,
    stop: float = np.nan,
    exit_price: float = StopExitPrice.Stop,
    exit_size: float = np.nan,
    exit_size_type: int = -1,
    exit_type: int = StopExitType.Close,
    order_type: int = OrderType.Market,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    ladder: int = StopLadderMode.Disabled,
    step: int = -1,
    step_idx: int = -1,
) -> None:
    """Set SL (stop loss) order information in the provided record.

    Args:
        sl_info (Record): Record for SL order information.
        init_idx (int): Initial row index for the SL order.
        init_price (float): Initial price for the SL order.
        init_position (float): Initial position size.
        stop (float): Stop price.
        exit_price (float): Indicator for the exit price.

            See `vectorbtpro.portfolio.enums.StopExitPrice`.
        exit_size (float): Exit position size.
        exit_size_type (int): Type indicator for the exit size.

            See `vectorbtpro.portfolio.enums.SizeType`.
        exit_type (int): Exit type indicator.

            See `vectorbtpro.portfolio.enums.StopExitType`.
        order_type (int): Order type for the SL order.

            See `vectorbtpro.portfolio.enums.OrderType`.
        limit_delta (float): Delta value used for limit calculations.
        delta_format (int): Format for the delta value.

            See `vectorbtpro.portfolio.enums.DeltaFormat`.
        ladder (int): Stop ladder mode.

            See `vectorbtpro.portfolio.enums.StopLadderMode`.
        step (int): Ladder step.
        step_idx (int): Ladder step index.

    Returns:
        None: The function modifies the `sl_info` record in place.

    See:
        `vectorbtpro.portfolio.enums.sl_info_dt`
    """
    sl_info["init_idx"] = init_idx
    sl_info["init_price"] = init_price
    sl_info["init_position"] = init_position
    sl_info["stop"] = stop
    sl_info["exit_price"] = exit_price
    sl_info["exit_size"] = exit_size
    sl_info["exit_size_type"] = exit_size_type
    sl_info["exit_type"] = exit_type
    sl_info["order_type"] = order_type
    sl_info["limit_delta"] = limit_delta
    sl_info["delta_format"] = delta_format
    sl_info["ladder"] = ladder
    sl_info["step"] = step
    sl_info["step_idx"] = step_idx


@register_jitted(cache=True)
def clear_sl_info_nb(sl_info: tp.Record) -> None:
    """Clear SL (stop loss) order information in the provided record.

    Args:
        sl_info (Record): Record for SL order information to be cleared.

    Returns:
        None: The function modifies the `sl_info` record in place.
    """
    sl_info["init_idx"] = -1
    sl_info["init_price"] = np.nan
    sl_info["init_position"] = np.nan
    sl_info["stop"] = np.nan
    sl_info["exit_price"] = -1
    sl_info["exit_size"] = np.nan
    sl_info["exit_size_type"] = -1
    sl_info["exit_type"] = -1
    sl_info["order_type"] = -1
    sl_info["limit_delta"] = np.nan
    sl_info["delta_format"] = -1
    sl_info["ladder"] = -1
    sl_info["step"] = -1
    sl_info["step_idx"] = -1


@register_jitted(cache=True)
def set_tsl_info_nb(
    tsl_info: tp.Record,
    init_idx: int,
    init_price: float = -np.inf,
    init_position: float = np.nan,
    peak_idx: tp.Optional[int] = None,
    peak_price: tp.Optional[float] = None,
    stop: float = np.nan,
    th: float = np.nan,
    exit_price: float = StopExitPrice.Stop,
    exit_size: float = np.nan,
    exit_size_type: int = -1,
    exit_type: int = StopExitType.Close,
    order_type: int = OrderType.Market,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    ladder: int = StopLadderMode.Disabled,
    step: int = -1,
    step_idx: int = -1,
) -> None:
    """Set TSL/TTP order information.

    Args:
        tsl_info (Record): Record to store trailing stop (TSL) or trailing take profit (TTP) order data.
        init_idx (int): Initialization row index.
        init_price (float): Initial price.
        init_position (float): Initial position.
        peak_idx (Optional[int]): Peak row index.

            Defaults to `init_idx` if not provided.
        peak_price (Optional[float]): Peak price.

            Defaults to `init_price` if not provided.
        stop (float): Stop loss value.
        th (float): Trailing threshold value.
        exit_price (float): Exit price mode.

            See `vectorbtpro.portfolio.enums.StopExitPrice`.
        exit_size (float): Exit size.
        exit_size_type (int): Exit size type.

            See `vectorbtpro.portfolio.enums.SizeType`.
        exit_type (int): Exit order type.

            See `vectorbtpro.portfolio.enums.OrderType`.
        order_type (int): Order execution type.

            See `vectorbtpro.portfolio.enums.OrderType`.
        limit_delta (float): Limit delta value.
        delta_format (int): Delta format.

            See `vectorbtpro.portfolio.enums.DeltaFormat`.
        ladder (int): Ladder mode indicator.

            See `vectorbtpro.portfolio.enums.StopLadderMode`.
        step (int): Ladder step value.
        step_idx (int): Ladder step index.

    Returns:
        None: The function modifies the `tsl_info` record in place.

    See:
        `vectorbtpro.portfolio.enums.tsl_info_dt`
    """
    tsl_info["init_idx"] = init_idx
    tsl_info["init_price"] = init_price
    tsl_info["init_position"] = init_position
    tsl_info["peak_idx"] = peak_idx if peak_idx is not None else init_idx
    tsl_info["peak_price"] = peak_price if peak_price is not None else init_price
    tsl_info["stop"] = stop
    tsl_info["th"] = th
    tsl_info["exit_price"] = exit_price
    tsl_info["exit_size"] = exit_size
    tsl_info["exit_size_type"] = exit_size_type
    tsl_info["exit_type"] = exit_type
    tsl_info["order_type"] = order_type
    tsl_info["limit_delta"] = limit_delta
    tsl_info["delta_format"] = delta_format
    tsl_info["ladder"] = ladder
    tsl_info["step"] = step
    tsl_info["step_idx"] = step_idx


@register_jitted(cache=True)
def clear_tsl_info_nb(tsl_info: tp.Record) -> None:
    """Clear TSL/TTP order information.

    Args:
        tsl_info (Record): Record containing trailing stop order information to clear.

    Returns:
        None: The function modifies the `tsl_info` record in place.
    """
    tsl_info["init_idx"] = -1
    tsl_info["init_price"] = np.nan
    tsl_info["init_position"] = np.nan
    tsl_info["peak_idx"] = -1
    tsl_info["peak_price"] = np.nan
    tsl_info["stop"] = np.nan
    tsl_info["th"] = np.nan
    tsl_info["exit_price"] = -1
    tsl_info["exit_size"] = np.nan
    tsl_info["exit_size_type"] = -1
    tsl_info["exit_type"] = -1
    tsl_info["order_type"] = -1
    tsl_info["limit_delta"] = np.nan
    tsl_info["delta_format"] = -1
    tsl_info["ladder"] = -1
    tsl_info["step"] = -1
    tsl_info["step_idx"] = -1


@register_jitted(cache=True)
def set_tp_info_nb(
    tp_info: tp.Record,
    init_idx: int,
    init_price: float = -np.inf,
    init_position: float = np.nan,
    stop: float = np.nan,
    exit_price: float = StopExitPrice.Stop,
    exit_size: float = np.nan,
    exit_size_type: int = -1,
    exit_type: int = StopExitType.Close,
    order_type: int = OrderType.Market,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    ladder: int = StopLadderMode.Disabled,
    step: int = -1,
    step_idx: int = -1,
) -> None:
    """Set TP order information.

    Args:
        tp_info (Record): Record to store take profit (TP) order data.
        init_idx (int): Initialization row index.
        init_price (float): Initial price.
        init_position (float): Initial position.
        stop (float): Stop loss value.
        exit_price (float): Exit price mode.

            See `vectorbtpro.portfolio.enums.StopExitPrice`.
        exit_size (float): Exit size.
        exit_size_type (int): Exit size type.

            See `vectorbtpro.portfolio.enums.SizeType`.
        exit_type (int): Exit order type.

            See `vectorbtpro.portfolio.enums.StopExitType`.
        order_type (int): Order execution type.

            See `vectorbtpro.portfolio.enums.OrderType`.
        limit_delta (float): Limit delta value.
        delta_format (int): Delta format.

            See `vectorbtpro.portfolio.enums.DeltaFormat`.
        ladder (int): Ladder mode indicator.

            See `vectorbtpro.portfolio.enums.StopLadderMode`.
        step (int): Ladder step value.
        step_idx (int): Ladder step index.

    Returns:
        None: The function modifies the `tp_info` record in place.

    See:
        `vectorbtpro.portfolio.enums.tp_info_dt`
    """
    tp_info["init_idx"] = init_idx
    tp_info["init_price"] = init_price
    tp_info["init_position"] = init_position
    tp_info["stop"] = stop
    tp_info["exit_price"] = exit_price
    tp_info["exit_size"] = exit_size
    tp_info["exit_size_type"] = exit_size_type
    tp_info["exit_type"] = exit_type
    tp_info["order_type"] = order_type
    tp_info["limit_delta"] = limit_delta
    tp_info["delta_format"] = delta_format
    tp_info["ladder"] = ladder
    tp_info["step"] = step
    tp_info["step_idx"] = step_idx


@register_jitted(cache=True)
def clear_tp_info_nb(tp_info: tp.Record) -> None:
    """Clear TP order information.

    Args:
        tp_info (Record): Record containing take profit order information to clear.

    Returns:
        None: The function modifies the `tp_info` record in place.
    """
    tp_info["init_idx"] = -1
    tp_info["init_price"] = np.nan
    tp_info["init_position"] = np.nan
    tp_info["stop"] = np.nan
    tp_info["exit_price"] = -1
    tp_info["exit_size"] = np.nan
    tp_info["exit_size_type"] = -1
    tp_info["exit_type"] = -1
    tp_info["order_type"] = -1
    tp_info["limit_delta"] = np.nan
    tp_info["delta_format"] = -1
    tp_info["ladder"] = -1
    tp_info["step"] = -1
    tp_info["step_idx"] = -1


@register_jitted(cache=True)
def set_time_info_nb(
    time_info: tp.Record,
    init_idx: int,
    init_position: float = np.nan,
    stop: int = -1,
    exit_price: float = StopExitPrice.Stop,
    exit_size: float = np.nan,
    exit_size_type: int = -1,
    exit_type: int = StopExitType.Close,
    order_type: int = OrderType.Market,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    time_delta_format: int = TimeDeltaFormat.Index,
    ladder: int = StopLadderMode.Disabled,
    step: int = -1,
    step_idx: int = -1,
) -> None:
    """Set time order information.

    Args:
        time_info (Record): Record to store time-based order information.
        init_idx (int): Initialization row index.
        init_position (float): Initial position.
        stop (int): Stop indicator or time.
        exit_price (float): Exit price mode.

            See `vectorbtpro.portfolio.enums.StopExitPrice`.
        exit_size (float): Exit size.
        exit_size_type (int): Exit size type.

            See `vectorbtpro.portfolio.enums.SizeType`.
        exit_type (int): Exit order type.

            See `vectorbtpro.portfolio.enums.StopExitType`.
        order_type (int): Order execution type.

            See `vectorbtpro.portfolio.enums.OrderType`.
        limit_delta (float): Limit delta value.
        delta_format (int): Delta format.

            See `vectorbtpro.portfolio.enums.DeltaFormat`.
        time_delta_format (int): Time delta format.

            See `vectorbtpro.portfolio.enums.TimeDeltaFormat`.
        ladder (int): Ladder mode indicator.

            See `vectorbtpro.portfolio.enums.StopLadderMode`.
        step (int): Ladder step value.
        step_idx (int): Ladder step index.

    Returns:
        None: The function modifies the `time_info` record in place.

    See:
        `vectorbtpro.portfolio.enums.time_info_dt`
    """
    time_info["init_idx"] = init_idx
    time_info["init_position"] = init_position
    time_info["stop"] = stop
    time_info["exit_price"] = exit_price
    time_info["exit_size"] = exit_size
    time_info["exit_size_type"] = exit_size_type
    time_info["exit_type"] = exit_type
    time_info["order_type"] = order_type
    time_info["limit_delta"] = limit_delta
    time_info["delta_format"] = delta_format
    time_info["time_delta_format"] = time_delta_format
    time_info["ladder"] = ladder
    time_info["step"] = step
    time_info["step_idx"] = step_idx


@register_jitted(cache=True)
def clear_time_info_nb(time_info: tp.Record) -> None:
    """Clear time order information.

    Args:
        time_info (Record): Record containing time-based order information to clear.

    Returns:
        None: The function modifies the `time_info` record in place.
    """
    time_info["init_idx"] = -1
    time_info["init_position"] = np.nan
    time_info["stop"] = -1
    time_info["exit_price"] = -1
    time_info["exit_size"] = np.nan
    time_info["exit_size_type"] = -1
    time_info["exit_type"] = -1
    time_info["order_type"] = -1
    time_info["limit_delta"] = np.nan
    time_info["delta_format"] = -1
    time_info["time_delta_format"] = -1
    time_info["ladder"] = -1
    time_info["step"] = -1
    time_info["step_idx"] = -1


@register_jitted(cache=True)
def get_limit_info_target_price_nb(limit_info: tp.Record) -> float:
    """Get target price from limit order information.

    Args:
        limit_info (Record): Record with limit order details.

    Returns:
        float: The target price if the limit order information is active; otherwise, NaN.
    """
    if not is_limit_info_active_nb(limit_info):
        return np.nan
    if limit_info["init_size"] == 0:
        raise ValueError("Limit order size cannot be zero")
    size = get_diraware_size_nb(limit_info["init_size"], limit_info["init_direction"])
    hit_below = (size > 0 and not limit_info["reverse"]) or (size < 0 and limit_info["reverse"])
    return resolve_limit_price_nb(
        init_price=limit_info["init_price"],
        limit_delta=limit_info["delta"],
        delta_format=limit_info["delta_format"],
        hit_below=hit_below,
    )


@register_jitted
def get_sl_info_target_price_nb(sl_info: tp.Record, position_now: float) -> float:
    """Get target price from SL order information.

    Args:
        sl_info (Record): Record with stop loss (SL) order details.
        position_now (float): Current position used to determine the hit direction.

    Returns:
        float: The target stop loss price if the SL order is active; otherwise, NaN.
    """
    if not is_stop_info_active_nb(sl_info):
        return np.nan
    hit_below = position_now > 0
    return resolve_stop_price_nb(
        init_price=sl_info["init_price"],
        stop=sl_info["stop"],
        delta_format=sl_info["delta_format"],
        hit_below=hit_below,
    )


@register_jitted
def get_tsl_info_target_price_nb(tsl_info: tp.Record, position_now: float) -> float:
    """Get target price from trailing stop loss (TSL) order information.

    Args:
        tsl_info (Record): Trailing stop loss order information.
        position_now (float): Current position size, used to determine whether
            the stop should be applied below the peak.

    Returns:
        float: The computed target price based on TSL information.

            Returns NaN if the stop information is inactive.
    """
    if not is_stop_info_active_nb(tsl_info):
        return np.nan
    hit_below = position_now > 0
    return resolve_stop_price_nb(
        init_price=tsl_info["peak_price"],
        stop=tsl_info["stop"],
        delta_format=tsl_info["delta_format"],
        hit_below=hit_below,
    )


@register_jitted
def get_tp_info_target_price_nb(tp_info: tp.Record, position_now: float) -> float:
    """Get target price from take profit (TP) order information.

    Args:
        tp_info (Record): Take profit order information.
        position_now (float): Current position size, used to determine whether
            the stop should be applied below the initial price.

    Returns:
        float: The computed target price based on TP information.

            Returns NaN if the stop information is inactive.
    """
    if not is_stop_info_active_nb(tp_info):
        return np.nan
    hit_below = position_now < 0
    return resolve_stop_price_nb(
        init_price=tp_info["init_price"],
        stop=tp_info["stop"],
        delta_format=tp_info["delta_format"],
        hit_below=hit_below,
    )
