# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing generic Numba-compiled functions for simulation ranges.

!!! warning
    Resolution is more flexible and may return None while preparation always returns NumPy arrays.
    Thus, use preparation, not resolution, in Numba-parallel workflows."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro._dtypes import *
from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb
from vectorbtpro.base.reshaping import to_1d_array_nb
from vectorbtpro.registries.jit_registry import register_jitted


@register_jitted(cache=True)
def resolve_sim_start_nb(
    sim_shape: tp.Shape,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Optional[tp.Array1d]:
    """Resolve simulation start positions for simulation.

    Args:
        sim_shape (Shape): A tuple representing the dimensions of the simulation.
        sim_start (Optional[FlexArray1dLike]): Simulation start positions.
        allow_none (bool): Flag to allow returning None if all start positions are default.
        check_bounds (bool): Flag to validate that start positions are within simulation bounds.

    Returns:
        Optional[Array1d]: An array of simulation start positions or None.
    """
    if sim_start is None:
        if allow_none:
            return None
        return np.full(sim_shape[1], 0, dtype=int_)

    sim_start_ = to_1d_array_nb(np.asarray(sim_start).astype(int_))
    if not check_bounds and len(sim_start_) == sim_shape[1]:
        return sim_start_

    sim_start_out = np.empty(sim_shape[1], dtype=int_)
    can_be_none = True

    for i in range(sim_shape[1]):
        _sim_start = flex_select_1d_pc_nb(sim_start_, i)
        if _sim_start < 0:
            _sim_start = sim_shape[0] + _sim_start
        elif _sim_start > sim_shape[0]:
            _sim_start = sim_shape[0]
        sim_start_out[i] = _sim_start
        if _sim_start != 0:
            can_be_none = False

    if allow_none and can_be_none:
        return None
    return sim_start_out


@register_jitted(cache=True)
def resolve_sim_end_nb(
    sim_shape: tp.Shape,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Optional[tp.Array1d]:
    """Resolve simulation end positions for simulation.

    Args:
        sim_shape (Shape): A tuple representing the dimensions of the simulation.
        sim_end (Optional[FlexArray1dLike]): Simulation end positions.
        allow_none (bool): Flag to allow returning None if all end positions are default.
        check_bounds (bool): Flag to validate that end positions are within simulation bounds.

    Returns:
        Optional[Array1d]: An array of simulation end positions or None.
    """
    if sim_end is None:
        if allow_none:
            return None
        return np.full(sim_shape[1], sim_shape[0], dtype=int_)

    sim_end_ = to_1d_array_nb(np.asarray(sim_end).astype(int_))
    if not check_bounds and len(sim_end_) == sim_shape[1]:
        return sim_end_

    new_sim_end = np.empty(sim_shape[1], dtype=int_)
    can_be_none = True

    for i in range(sim_shape[1]):
        _sim_end = flex_select_1d_pc_nb(sim_end_, i)
        if _sim_end < 0:
            _sim_end = sim_shape[0] + _sim_end
        elif _sim_end > sim_shape[0]:
            _sim_end = sim_shape[0]
        new_sim_end[i] = _sim_end
        if _sim_end != sim_shape[0]:
            can_be_none = False

    if allow_none and can_be_none:
        return None
    return new_sim_end


@register_jitted(cache=True)
def resolve_sim_range_nb(
    sim_shape: tp.Shape,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Tuple[tp.Optional[tp.Array1d], tp.Optional[tp.Array1d]]:
    """Resolve simulation start and end positions for simulation.

    Args:
        sim_shape (Shape): A tuple representing the dimensions of the simulation.
        sim_start (Optional[FlexArray1dLike]): Simulation start positions.
        sim_end (Optional[FlexArray1dLike]): Simulation end positions.
        allow_none (bool): Flag to allow returning None if default positions are used.
        check_bounds (bool): Flag to validate that positions are within simulation bounds.

    Returns:
        Tuple[Optional[Array1d], Optional[Array1d]]: A tuple containing the simulation start and end positions.
    """
    new_sim_start = resolve_sim_start_nb(
        sim_shape=sim_shape,
        sim_start=sim_start,
        allow_none=allow_none,
        check_bounds=check_bounds,
    )
    new_sim_end = resolve_sim_end_nb(
        sim_shape=sim_shape,
        sim_end=sim_end,
        allow_none=allow_none,
        check_bounds=check_bounds,
    )
    return new_sim_start, new_sim_end


@register_jitted(cache=True)
def resolve_grouped_sim_start_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Optional[tp.Array1d]:
    """Resolve grouped simulation start positions for simulation groups.

    Args:
        target_shape (Shape): A tuple representing the dimensions of the target simulation.
        group_lens (GroupLens): A sequence indicating the lengths of each simulation group.
        sim_start (Optional[FlexArray1dLike]): Simulation start positions for individual simulations.
        allow_none (bool): Flag to allow returning None if all group start positions are default.
        check_bounds (bool): Flag to validate that start positions are within simulation bounds.

    Returns:
        Optional[Array1d]: An array of simulation start positions for each group or None.
    """
    if sim_start is None:
        if allow_none:
            return None
        return np.full(len(group_lens), 0, dtype=int_)

    sim_start_ = to_1d_array_nb(np.asarray(sim_start).astype(int_))
    if len(sim_start_) == len(group_lens):
        if not check_bounds:
            return sim_start_
        return resolve_sim_start_nb(
            (target_shape[0], len(group_lens)),
            sim_start=sim_start_,
            allow_none=allow_none,
            check_bounds=check_bounds,
        )

    new_sim_start = np.empty(len(group_lens), dtype=int_)
    can_be_none = True

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        min_sim_start = target_shape[0]
        for col in range(from_col, to_col):
            _sim_start = flex_select_1d_pc_nb(sim_start_, col)
            if _sim_start < 0:
                _sim_start = target_shape[0] + _sim_start
            elif _sim_start > target_shape[0]:
                _sim_start = target_shape[0]
            if _sim_start < min_sim_start:
                min_sim_start = _sim_start
        new_sim_start[group] = min_sim_start
        if min_sim_start != 0:
            can_be_none = False

    if allow_none and can_be_none:
        return None
    return new_sim_start


@register_jitted(cache=True)
def resolve_grouped_sim_end_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Optional[tp.Array1d]:
    """Resolve grouped simulation end positions for simulation groups.

    Args:
        target_shape (Shape): A tuple representing the dimensions of the target simulation.
        group_lens (GroupLens): A sequence indicating the lengths of each simulation group.
        sim_end (Optional[FlexArray1dLike]): Simulation end positions for individual simulations.
        allow_none (bool): Flag to allow returning None if all group end positions are default.
        check_bounds (bool): Flag to validate that end positions are within simulation bounds.

    Returns:
        Optional[Array1d]: An array of simulation end positions for each group or None.
    """
    if sim_end is None:
        if allow_none:
            return None
        return np.full(len(group_lens), target_shape[0], dtype=int_)

    sim_end_ = to_1d_array_nb(np.asarray(sim_end).astype(int_))
    if len(sim_end_) == len(group_lens):
        if not check_bounds:
            return sim_end_
        return resolve_sim_end_nb(
            (target_shape[0], len(group_lens)),
            sim_end=sim_end_,
            allow_none=allow_none,
            check_bounds=check_bounds,
        )

    new_sim_end = np.empty(len(group_lens), dtype=int_)
    can_be_none = True

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        max_sim_end = 0
        for col in range(from_col, to_col):
            _sim_end = flex_select_1d_pc_nb(sim_end_, col)
            if _sim_end < 0:
                _sim_end = target_shape[0] + _sim_end
            elif _sim_end > target_shape[0]:
                _sim_end = target_shape[0]
            if _sim_end > max_sim_end:
                max_sim_end = _sim_end
        new_sim_end[group] = max_sim_end
        if max_sim_end != target_shape[0]:
            can_be_none = False

    if allow_none and can_be_none:
        return None
    return new_sim_end


@register_jitted(cache=True)
def resolve_grouped_sim_range_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Tuple[tp.Optional[tp.Array1d], tp.Optional[tp.Array1d]]:
    """Resolve simulation start and end for grouped data.

    Args:
        target_shape (Shape): The target simulation array shape.
        group_lens (GroupLens): The lengths of each group.
        sim_start (Optional[FlexArray1dLike]): The initial simulation start positions.
        sim_end (Optional[FlexArray1dLike]): The initial simulation end positions.
        allow_none (bool): Allow simulation positions to be None when applicable.
        check_bounds (bool): Enforce bounds checking for simulation positions.

    Returns:
        Tuple[Optional[Array1d], Optional[Array1d]]: A tuple containing the resolved
            simulation start and end positions.
    """
    new_sim_start = resolve_grouped_sim_start_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_start=sim_start,
        allow_none=allow_none,
        check_bounds=check_bounds,
    )
    new_sim_end = resolve_grouped_sim_end_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_end=sim_end,
        allow_none=allow_none,
        check_bounds=check_bounds,
    )
    return new_sim_start, new_sim_end


@register_jitted(cache=True)
def resolve_ungrouped_sim_start_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Optional[tp.Array1d]:
    """Resolve simulation start for ungrouped data.

    Args:
        target_shape (Shape): The target simulation array shape.
        group_lens (GroupLens): The lengths of each group.
        sim_start (Optional[FlexArray1dLike]): The input simulation start positions.
        allow_none (bool): Allow simulation start to be None when applicable.
        check_bounds (bool): Enforce simulation start bounds if True.

    Returns:
        Optional[Array1d]: The resolved simulation start array or None.
    """
    if sim_start is None:
        if allow_none:
            return None
        return np.full(target_shape[1], 0, dtype=int_)

    sim_start_ = to_1d_array_nb(np.asarray(sim_start).astype(int_))
    if len(sim_start_) == target_shape[1]:
        if not check_bounds:
            return sim_start_
        return resolve_sim_start_nb(
            target_shape,
            sim_start=sim_start_,
            allow_none=allow_none,
            check_bounds=check_bounds,
        )

    new_sim_start = np.empty(target_shape[1], dtype=int_)
    can_be_none = True

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        _sim_start = flex_select_1d_pc_nb(sim_start_, group)
        if _sim_start < 0:
            _sim_start = target_shape[0] + _sim_start
        elif _sim_start > target_shape[0]:
            _sim_start = target_shape[0]
        for col in range(from_col, to_col):
            new_sim_start[col] = _sim_start
        if _sim_start != 0:
            can_be_none = False

    if allow_none and can_be_none:
        return None
    return new_sim_start


@register_jitted(cache=True)
def resolve_ungrouped_sim_end_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Optional[tp.Array1d]:
    """Resolve simulation end for ungrouped data.

    Args:
        target_shape (Shape): The target simulation array shape.
        group_lens (GroupLens): The lengths of each group.
        sim_end (Optional[FlexArray1dLike]): The input simulation end positions.
        allow_none (bool): Allow simulation end to be None when applicable.
        check_bounds (bool): Enforce simulation end bounds if True.

    Returns:
        Optional[Array1d]: The resolved simulation end array or None.
    """
    if sim_end is None:
        if allow_none:
            return None
        return np.full(target_shape[1], target_shape[0], dtype=int_)

    sim_end_ = to_1d_array_nb(np.asarray(sim_end).astype(int_))
    if len(sim_end_) == target_shape[1]:
        if not check_bounds:
            return sim_end_
        return resolve_sim_end_nb(
            target_shape,
            sim_end=sim_end_,
            allow_none=allow_none,
            check_bounds=check_bounds,
        )

    new_sim_end = np.empty(target_shape[1], dtype=int_)
    can_be_none = True

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        _sim_end = flex_select_1d_pc_nb(sim_end_, group)
        if _sim_end < 0:
            _sim_end = target_shape[0] + _sim_end
        elif _sim_end > target_shape[0]:
            _sim_end = target_shape[0]
        for col in range(from_col, to_col):
            new_sim_end[col] = _sim_end
        if _sim_end != target_shape[0]:
            can_be_none = False

    if allow_none and can_be_none:
        return None
    return new_sim_end


@register_jitted(cache=True)
def resolve_ungrouped_sim_range_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    allow_none: bool = False,
    check_bounds: bool = True,
) -> tp.Tuple[tp.Optional[tp.Array1d], tp.Optional[tp.Array1d]]:
    """Resolve simulation start and end for ungrouped data.

    Args:
        target_shape (Shape): The target simulation array shape.
        group_lens (GroupLens): The lengths of each group.
        sim_start (Optional[FlexArray1dLike]): The input simulation start positions.
        sim_end (Optional[FlexArray1dLike]): The input simulation end positions.
        allow_none (bool): Allow simulation positions to be None when applicable.
        check_bounds (bool): Enforce bounds checking for simulation positions.

    Returns:
        Tuple[Optional[Array1d], Optional[Array1d]]: A tuple containing the resolved
            simulation start and end positions.
    """
    new_sim_start = resolve_ungrouped_sim_start_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_start=sim_start,
        allow_none=allow_none,
        check_bounds=check_bounds,
    )
    new_sim_end = resolve_ungrouped_sim_end_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_end=sim_end,
        allow_none=allow_none,
        check_bounds=check_bounds,
    )
    return new_sim_start, new_sim_end


@register_jitted(cache=True)
def prepare_sim_start_nb(
    sim_shape: tp.Shape,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Array1d:
    """Prepare the simulation start array based on provided parameters.

    Args:
        sim_shape (Shape): The shape of the simulation array.
        sim_start (Optional[FlexArray1dLike]): The input simulation start positions.
        check_bounds (bool): Enforce bounds checking for the simulation start positions if True.

    Returns:
        Array1d: The prepared simulation start array.
    """
    if sim_start is None:
        return np.full(sim_shape[1], 0, dtype=int_)

    sim_start_ = to_1d_array_nb(np.asarray(sim_start).astype(int_))
    if not check_bounds and len(sim_start_) == sim_shape[1]:
        return sim_start_

    sim_start_out = np.empty(sim_shape[1], dtype=int_)

    for i in range(sim_shape[1]):
        _sim_start = flex_select_1d_pc_nb(sim_start_, i)
        if _sim_start < 0:
            _sim_start = sim_shape[0] + _sim_start
        elif _sim_start > sim_shape[0]:
            _sim_start = sim_shape[0]
        sim_start_out[i] = _sim_start

    return sim_start_out


@register_jitted(cache=True)
def prepare_sim_end_nb(
    sim_shape: tp.Shape,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Array1d:
    """Prepare the simulation end array based on provided parameters.

    Args:
        sim_shape (Shape): The shape of the simulation array.
        sim_end (Optional[FlexArray1dLike]): The input simulation end positions.
        check_bounds (bool): Enforce bounds checking for the simulation end positions if True.

    Returns:
        Array1d: The prepared simulation end array.
    """
    if sim_end is None:
        return np.full(sim_shape[1], sim_shape[0], dtype=int_)

    sim_end_ = to_1d_array_nb(np.asarray(sim_end).astype(int_))
    if not check_bounds and len(sim_end_) == sim_shape[1]:
        return sim_end_

    new_sim_end = np.empty(sim_shape[1], dtype=int_)

    for i in range(sim_shape[1]):
        _sim_end = flex_select_1d_pc_nb(sim_end_, i)
        if _sim_end < 0:
            _sim_end = sim_shape[0] + _sim_end
        elif _sim_end > sim_shape[0]:
            _sim_end = sim_shape[0]
        new_sim_end[i] = _sim_end

    return new_sim_end


@register_jitted(cache=True)
def prepare_sim_range_nb(
    sim_shape: tp.Shape,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Prepare both simulation start and end positions based on provided parameters.

    Args:
        sim_shape (Shape): The shape of the simulation array.
        sim_start (Optional[FlexArray1dLike]): The input simulation start positions.
        sim_end (Optional[FlexArray1dLike]): The input simulation end positions.
        check_bounds (bool): Enforce bounds checking for simulation positions if True.

    Returns:
        Tuple[Array1d, Array1d]: A tuple containing the prepared simulation start and end positions.
    """
    new_sim_start = prepare_sim_start_nb(
        sim_shape=sim_shape,
        sim_start=sim_start,
        check_bounds=check_bounds,
    )
    new_sim_end = prepare_sim_end_nb(
        sim_shape=sim_shape,
        sim_end=sim_end,
        check_bounds=check_bounds,
    )
    return new_sim_start, new_sim_end


@register_jitted(cache=True)
def prepare_grouped_sim_start_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Array1d:
    """Prepare grouped simulation start positions for grouped data.

    Args:
        target_shape (Shape): A tuple where the first element indicates the simulation step count.
        group_lens (GroupLens): An array-like sequence representing the number of columns per group.
        sim_start (Optional[FlexArray1dLike]): An array-like object containing simulation start positions.
            If None, all groups default to a start index of zero.
        check_bounds (bool): Flag indicating whether to validate and adjust start positions within bounds.

    Returns:
        Array1d: An array containing the prepared simulation start positions for each group.
    """
    if sim_start is None:
        return np.full(len(group_lens), 0, dtype=int_)

    sim_start_ = to_1d_array_nb(np.asarray(sim_start).astype(int_))
    if len(sim_start_) == len(group_lens):
        if not check_bounds:
            return sim_start_
        return prepare_sim_start_nb(
            (target_shape[0], len(group_lens)),
            sim_start=sim_start_,
            check_bounds=check_bounds,
        )

    new_sim_start = np.empty(len(group_lens), dtype=int_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        min_sim_start = target_shape[0]
        for col in range(from_col, to_col):
            _sim_start = flex_select_1d_pc_nb(sim_start_, col)
            if _sim_start < 0:
                _sim_start = target_shape[0] + _sim_start
            elif _sim_start > target_shape[0]:
                _sim_start = target_shape[0]
            if _sim_start < min_sim_start:
                min_sim_start = _sim_start
        new_sim_start[group] = min_sim_start

    return new_sim_start


@register_jitted(cache=True)
def prepare_grouped_sim_end_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Array1d:
    """Prepare grouped simulation end positions for grouped data.

    Args:
        target_shape (Shape): A tuple where the first element indicates the simulation step count.
        group_lens (GroupLens): An array-like sequence representing the number of columns per group.
        sim_end (Optional[FlexArray1dLike]): An array-like object containing simulation end positions.

            If None, all groups default to an end index equal to the simulation step count.
        check_bounds (bool): Flag indicating whether to validate and adjust end positions within bounds.

    Returns:
        Array1d: An array containing the prepared simulation end positions for each group.
    """
    if sim_end is None:
        return np.full(len(group_lens), target_shape[0], dtype=int_)

    sim_end_ = to_1d_array_nb(np.asarray(sim_end).astype(int_))
    if len(sim_end_) == len(group_lens):
        if not check_bounds:
            return sim_end_
        return prepare_sim_end_nb(
            (target_shape[0], len(group_lens)),
            sim_end=sim_end_,
            check_bounds=check_bounds,
        )

    new_sim_end = np.empty(len(group_lens), dtype=int_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        max_sim_end = 0
        for col in range(from_col, to_col):
            _sim_end = flex_select_1d_pc_nb(sim_end_, col)
            if _sim_end < 0:
                _sim_end = target_shape[0] + _sim_end
            elif _sim_end > target_shape[0]:
                _sim_end = target_shape[0]
            if _sim_end > max_sim_end:
                max_sim_end = _sim_end
        new_sim_end[group] = max_sim_end

    return new_sim_end


@register_jitted(cache=True)
def prepare_grouped_sim_range_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Prepare grouped simulation start and end positions for grouped data.

    Args:
        target_shape (Shape): A tuple where the first element indicates the simulation step count.
        group_lens (GroupLens): An array-like sequence representing the number of columns per group.
        sim_start (Optional[FlexArray1dLike]): An array-like object containing simulation start positions.

            If None, all groups default to a start index of zero.
        sim_end (Optional[FlexArray1dLike]): An array-like object containing simulation end positions.

            If None, all groups default to an end index equal to the simulation step count.
        check_bounds (bool): Flag indicating whether to validate and adjust positions within bounds.

    Returns:
        Tuple[Array1d, Array1d]: A tuple containing the prepared simulation start and end positions.
    """
    new_sim_start = prepare_grouped_sim_start_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_start=sim_start,
        check_bounds=check_bounds,
    )
    new_sim_end = prepare_grouped_sim_end_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_end=sim_end,
        check_bounds=check_bounds,
    )
    return new_sim_start, new_sim_end


@register_jitted(cache=True)
def prepare_ungrouped_sim_start_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Array1d:
    """Prepare ungrouped simulation start positions for ungrouped data.

    Args:
        target_shape (Shape): A tuple where the first element indicates the simulation step count
            and the second indicates the number of columns.
        group_lens (GroupLens): An array-like sequence representing group lengths.
        sim_start (Optional[FlexArray1dLike]): An array-like object containing simulation start positions.

            If None, all columns default to a start index of zero.
        check_bounds (bool): Flag indicating whether to validate and adjust start positions within bounds.

    Returns:
        Array1d: An array containing the prepared simulation start positions for ungrouped data.
    """
    if sim_start is None:
        return np.full(target_shape[1], 0, dtype=int_)

    sim_start_ = to_1d_array_nb(np.asarray(sim_start).astype(int_))
    if len(sim_start_) == target_shape[1]:
        if not check_bounds:
            return sim_start_
        return prepare_sim_start_nb(
            target_shape,
            sim_start=sim_start_,
            check_bounds=check_bounds,
        )

    new_sim_start = np.empty(target_shape[1], dtype=int_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        _sim_start = flex_select_1d_pc_nb(sim_start_, group)
        if _sim_start < 0:
            _sim_start = target_shape[0] + _sim_start
        elif _sim_start > target_shape[0]:
            _sim_start = target_shape[0]
        for col in range(from_col, to_col):
            new_sim_start[col] = _sim_start

    return new_sim_start


@register_jitted(cache=True)
def prepare_ungrouped_sim_end_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Array1d:
    """Prepare ungrouped simulation end positions for ungrouped data.

    Args:
        target_shape (Shape): A tuple where the first element indicates the simulation step
            count and the second indicates the number of columns.
        group_lens (GroupLens): An array-like sequence representing group lengths.
        sim_end (Optional[FlexArray1dLike]): An array-like object containing simulation end positions.

            If None, all columns default to an end index equal to the simulation step count.
        check_bounds (bool): Flag indicating whether to validate and adjust end positions within bounds.

    Returns:
        Array1d: An array containing the prepared simulation end positions for ungrouped data.
    """
    if sim_end is None:
        return np.full(target_shape[1], target_shape[0], dtype=int_)

    sim_end_ = to_1d_array_nb(np.asarray(sim_end).astype(int_))
    if len(sim_end_) == target_shape[1]:
        if not check_bounds:
            return sim_end_
        return prepare_sim_end_nb(
            target_shape,
            sim_end=sim_end_,
            check_bounds=check_bounds,
        )

    new_sim_end = np.empty(target_shape[1], dtype=int_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        _sim_end = flex_select_1d_pc_nb(sim_end_, group)
        if _sim_end < 0:
            _sim_end = target_shape[0] + _sim_end
        elif _sim_end > target_shape[0]:
            _sim_end = target_shape[0]
        for col in range(from_col, to_col):
            new_sim_end[col] = _sim_end

    return new_sim_end


@register_jitted(cache=True)
def prepare_ungrouped_sim_range_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    check_bounds: bool = True,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Prepare ungrouped simulation start and end positions for ungrouped data.

    Args:
        target_shape (Shape): A tuple where the first element indicates the simulation step count
            and the second indicates the number of columns.
        group_lens (GroupLens): An array-like sequence representing group lengths.
        sim_start (Optional[FlexArray1dLike]): An array-like object containing simulation start positions.

            If None, all columns default to a start index of zero.
        sim_end (Optional[FlexArray1dLike]): An array-like object containing simulation end positions.

            If None, all columns default to an end index equal to the simulation step count.
        check_bounds (bool): Flag indicating whether to validate and adjust positions within bounds.

    Returns:
        Tuple[Array1d, Array1d]: A tuple containing the prepared simulation start and
            end positions for ungrouped data.
    """
    new_sim_start = prepare_ungrouped_sim_start_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_start=sim_start,
        check_bounds=check_bounds,
    )
    new_sim_end = prepare_ungrouped_sim_end_nb(
        target_shape=target_shape,
        group_lens=group_lens,
        sim_end=sim_end,
        check_bounds=check_bounds,
    )
    return new_sim_start, new_sim_end
