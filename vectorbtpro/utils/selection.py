# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Utilities for selecting."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import DefineMixin, define

__all__ = [
    "PosSel",
    "LabelSel",
]


@define
class PosSel(DefineMixin):
    """Class that represents a selection by position."""

    value: tp.MaybeIterable[tp.Hashable] = define.field()
    """Selection of one or more positions."""


@define
class LabelSel(DefineMixin):
    """Class that represents a selection by label."""

    value: tp.MaybeIterable[tp.Hashable] = define.field()
    """Selection of one or more labels."""
