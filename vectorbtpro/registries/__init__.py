"""Package for registering objects across vectorbtpro."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.registries.ca_registry import *
    from vectorbtpro.registries.ch_registry import *
    from vectorbtpro.registries.jit_registry import *
    from vectorbtpro.registries.pbar_registry import *
