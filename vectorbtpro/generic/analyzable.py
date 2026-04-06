"""Module providing the `Analyzable` class for analyzing data."""

from vectorbtpro import _typing as tp
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.generic.plots_builder import PlotsBuilderMixin
from vectorbtpro.generic.stats_builder import StatsBuilderMixin

__all__ = [
    "Analyzable",
]


class MetaAnalyzable(type(Wrapping), type(StatsBuilderMixin), type(PlotsBuilderMixin)):
    """Metaclass for the `Analyzable` class."""

    pass


AnalyzableT = tp.TypeVar("AnalyzableT", bound="Analyzable")


class Analyzable(Wrapping, StatsBuilderMixin, PlotsBuilderMixin, metaclass=MetaAnalyzable):
    """Class that can be analyzed by computing and plotting various attributes.

    Args:
        wrapper (ArrayWrapper): Array wrapper instance.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.Wrapping`.
    """

    def __init__(self, wrapper: ArrayWrapper, **kwargs) -> None:
        Wrapping.__init__(self, wrapper, **kwargs)
        StatsBuilderMixin.__init__(self)
        PlotsBuilderMixin.__init__(self)
