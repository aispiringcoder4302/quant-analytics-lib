# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Utilities for hashing."""

from functools import cached_property as cachedproperty

from vectorbtpro import _typing as tp
from vectorbtpro.utils.base import Base

__all__ = []


class Hashable(Base):
    """Hashable class."""

    @staticmethod
    def get_hash(*args, **kwargs) -> int:
        """Static method to get the hash of the instance based on its arguments."""
        raise NotImplementedError

    @property
    def hash_key(self) -> tuple:
        """Key that can be used for hashing the instance."""
        raise NotImplementedError

    @cachedproperty
    def hash(self) -> int:
        """Hash of the instance."""
        return hash(self.hash_key)

    def __hash__(self) -> int:
        return self.hash

    def __eq__(self, other: tp.Any) -> bool:
        if isinstance(other, type(self)):
            return self.hash_key == other.hash_key
        raise NotImplementedError
