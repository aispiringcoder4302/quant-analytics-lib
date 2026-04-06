"""Module providing utilities for selection."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import DefineMixin, define

__all__ = [
    "PosSel",
    "LabelSel",
]


@define
class PosSel(DefineMixin):
    """Class representing a positional selection."""

    value: tp.MaybeIterable[tp.Hashable] = define.field()
    """Selection of one or more positions."""


@define
class LabelSel(DefineMixin):
    """Class representing a label-based selection."""

    value: tp.MaybeIterable[tp.Hashable] = define.field()
    """Selection of one or more labels."""
