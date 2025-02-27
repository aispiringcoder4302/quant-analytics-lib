# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Utilities for warnings."""

import io
import warnings
from contextlib import contextmanager, redirect_stdout

from vectorbtpro import _typing as tp
from vectorbtpro.utils.base import Base

__all__ = [
    "warn",
]


@contextmanager
def use_formatwarning(formatwarning: tp.Any) -> tp.Generator[None, None, None]:
    """Context manager to temporarily set a custom warning formatter."""
    old_formatter = warnings.formatwarning
    warnings.formatwarning = formatwarning
    try:
        yield
    finally:
        warnings.formatwarning = old_formatter


def custom_formatwarning(
    message: tp.Any,
    category: tp.Type[Warning],
    filename: str,
    lineno: int,
    line: tp.Optional[str] = None,
) -> str:
    """Custom warning formatter."""
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"


class VBTWarning(Warning):
    """Base class for warnings raised by VBT."""


def warn(message: tp.Any, category: type = VBTWarning, stacklevel: int = 2) -> None:
    """Emit a warning with a custom formatter."""
    with use_formatwarning(custom_formatwarning):
        warnings.warn(message, category, stacklevel=stacklevel)


def warn_stdout(func: tp.Callable) -> tp.Callable:
    """Supress and convert to a warning output from a function."""

    def wrapper(*a, **ka):
        with redirect_stdout(io.StringIO()) as f:
            out = func(*a, **ka)
        s = f.getvalue()
        if len(s) > 0:
            warn(s)
        return out

    return wrapper


class WarningsFiltered(warnings.catch_warnings, Base):
    """Context manager to ignore warnings."""

    def __init__(self, entries: tp.Optional[tp.MaybeSequence[tp.Union[str, tp.Kwargs]]] = "ignore", **kwargs) -> None:
        warnings.catch_warnings.__init__(self, **kwargs)
        self._entries = entries

    @property
    def entries(self) -> tp.Optional[tp.MaybeSequence[tp.Union[str, tp.Kwargs]]]:
        """One or more simple entries to add into the list of warnings filters."""
        return self._entries

    def __enter__(self) -> tp.Self:
        warnings.catch_warnings.__enter__(self)
        if self.entries is not None:
            if isinstance(self.entries, (str, dict)):
                entry = self.entries
                if isinstance(entry, str):
                    entry = dict(action=entry)
                warnings.simplefilter(**entry)
            else:
                for entry in self.entries:
                    if isinstance(entry, str):
                        entry = dict(action=entry)
                    warnings.simplefilter(**entry)
        return self
