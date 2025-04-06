# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for hashing."""

from functools import cached_property as cachedproperty

from vectorbtpro import _typing as tp
from vectorbtpro.utils.base import Base

__all__ = []


class Hashable(Base):
    """Class for representing hashable objects using a custom hash key."""

    @staticmethod
    def get_hash(*args, **kwargs) -> int:
        """Compute a hash value based on the provided arguments.

        Args:
            *args: Positional arguments passed for hash computation.
            **kwargs: Keyword arguments passed for hash computation.
        """
        raise NotImplementedError

    @property
    def hash_key(self) -> tuple:
        """Unique key used for computing the instance's hash."""
        raise NotImplementedError

    @cachedproperty
    def hash(self) -> int:
        """Computed hash value of the instance based on its hash key."""
        return hash(self.hash_key)

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, other: tp.Any) -> bool:
        if isinstance(other, type(self)):
            return self.hash_key == other.hash_key
        raise NotImplementedError
