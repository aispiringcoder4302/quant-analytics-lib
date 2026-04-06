"""Module containing configurations for custom indicators."""

from vectorbtpro.utils.config import ReadonlyConfig

__all__ = [
    "flex_col_param_config",
    "flex_elem_param_config",
]

flex_elem_param_config = ReadonlyConfig(
    dict(
        is_array_like=True,
        bc_to_input=True,
        broadcast_kwargs=dict(keep_flex=True, min_ndim=2),
    )
)
"""Configuration for flexible element-wise parameters."""

flex_row_param_config = ReadonlyConfig(
    dict(
        is_array_like=True,
        bc_to_input=0,
        broadcast_kwargs=dict(keep_flex=True, min_ndim=1),
    )
)
"""Configuration for flexible row-wise parameters."""

flex_col_param_config = ReadonlyConfig(
    dict(
        is_array_like=True,
        bc_to_input=1,
        per_column=True,
        broadcast_kwargs=dict(keep_flex=True, min_ndim=1),
    )
)
"""Configuration for flexible column-wise parameters."""
