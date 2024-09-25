# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for working with knowledge."""

from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import Configured
from vectorbtpro.utils.path_ import file_exists


JSONKnowledgeT = tp.TypeVar("JSONKnowledgeT", bound="JSONKnowledge")


class JSONKnowledge(Configured):
    """Class for working with knowledge in JSON format."""

    @classmethod
    def from_file(cls, file_path: tp.PathLike):
        file_path = Path(file_path)
        if not file_exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found")

    _expected_keys: tp.ExpectedKeys = (Configured._expected_keys or set()) | {
        "json_array"
    }

    def __init__(self, json_array: tp.JsonArray, **kwargs) -> None:
        Configured.__init__(
            self,
            json_array=json_array,
            **kwargs,
        )

        self._json_array = json_array

    @property
    def json_array(self) -> tp.JsonArray:
        """JSON array."""
        return self._json_array
