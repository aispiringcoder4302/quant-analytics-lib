"""Package for plotting with Plotly Express."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.px.accessors import *
    from vectorbtpro.px.decorators import *

__import_if_installed__ = dict()
__import_if_installed__["accessors"] = "plotly"
__import_if_installed__["decorators"] = "plotly"
