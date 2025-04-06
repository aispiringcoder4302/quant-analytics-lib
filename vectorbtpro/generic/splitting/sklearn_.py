# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing a Scikit-learn compatible cross-validator for data splitting."""

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import indexable

from vectorbtpro import _typing as tp
from vectorbtpro.generic.splitting.base import Splitter
from vectorbtpro.utils.base import Base

__all__ = [
    "SplitterCV",
]


class SplitterCV(BaseCrossValidator, Base):
    """Scikit-learn compatible cross-validator based on `vectorbtpro.generic.splitting.base.Splitter`.

    Args:
        splitter (Union[None, str, Splitter, Callable]): The splitter instance, factory name,
            or factory function.

            If None, it is determined automatically using the guess method of the splitter class.
        splitter_cls (Optional[Type[Splitter]]): The splitter class used as the factory.

            Defaults to `vectorbtpro.generic.splitting.base.Splitter`.
        split_group_by (tp.AnyGroupByLike): Group labels for splitting.
        set_group_by (tp.AnyGroupByLike): Group labels for setting.
        template_context (tp.KwargsLike): Mapping for template substitution in ranges,
            passed to the splitter factory.
        **splitter_kwargs: Keyword arguments for the splitter factory method.

    Usage:
        * Replicate `TimeSeriesSplit` from scikit-learn:

        ```pycon
        >>> from vectorbtpro import *

        >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = np.array([1, 2, 3, 4])

        >>> cv = vbt.SplitterCV(
        ...     "from_expanding",
        ...     min_length=2,
        ...     offset=1,
        ...     split=-1
        ... )
        >>> for i, (train_indices, test_indices) in enumerate(cv.split(X)):
        ...     print("Split %d:" % i)
        ...     X_train, X_test = X[train_indices], X[test_indices]
        ...     print("  X:", X_train.tolist(), X_test.tolist())
        ...     y_train, y_test = y[train_indices], y[test_indices]
        ...     print("  y:", y_train.tolist(), y_test.tolist())
        Split 0:
          X: [[1, 2]] [[3, 4]]
          y: [1] [2]
        Split 1:
          X: [[1, 2], [3, 4]] [[5, 6]]
          y: [1, 2] [3]
        Split 2:
          X: [[1, 2], [3, 4], [5, 6]] [[7, 8]]
          y: [1, 2, 3] [4]
        ```
    """

    def __init__(
        self,
        splitter: tp.Union[None, str, Splitter, tp.Callable] = None,
        *,
        splitter_cls: tp.Optional[tp.Type[Splitter]] = None,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        template_context: tp.KwargsLike = None,
        **splitter_kwargs,
    ) -> None:
        if splitter_cls is None:
            splitter_cls = Splitter
        if splitter is None:
            splitter = splitter_cls.guess_method(**splitter_kwargs)

        self._splitter = splitter
        self._splitter_kwargs = splitter_kwargs
        self._splitter_cls = splitter_cls
        self._split_group_by = split_group_by
        self._set_group_by = set_group_by
        self._template_context = template_context

    @property
    def splitter(self) -> tp.Union[str, Splitter, tp.Callable]:
        """Splitter instance, factory name, or factory function used for splitting.

        If None, it is determined automatically based on `SplitterCV.splitter_kwargs`.
        """
        return self._splitter

    @property
    def splitter_cls(self) -> tp.Type[Splitter]:
        """Splitter class used as the factory for creating splitter instances.

        Defaults to `vectorbtpro.generic.splitting.base.Splitter`.
        """
        return self._splitter_cls

    @property
    def splitter_kwargs(self) -> tp.KwargsLike:
        """Keyword arguments for the splitter factory method."""
        return self._splitter_kwargs

    @property
    def split_group_by(self) -> tp.AnyGroupByLike:
        """Group labels for splitting. See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.

        Not passed to the factory method.
        """
        return self._split_group_by

    @property
    def set_group_by(self) -> tp.AnyGroupByLike:
        """Group labels for setting. See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.

        Not passed to the factory method.
        """
        return self._set_group_by

    @property
    def template_context(self) -> tp.KwargsLike:
        """Mapping used for template substitution in ranges, passed to the splitter factory."""
        return self._template_context

    def get_splitter(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> Splitter:
        """Return a splitter instance of type `vectorbtpro.generic.splitting.base.Splitter`.

        Args:
            X (Any): Input data for splitting.
            y (Any): Target values corresponding to `X`.
            groups (Any): Group labels for splitting.

        Returns:
            Splitter: A splitter object configured with the provided data and splitter parameters.

        !!! note
            If the splitter is provided as a string, it is resolved as
                an attribute of the splitter class.
        """
        X, y, groups = indexable(X, y, groups)
        try:
            index = self.splitter_cls.get_obj_index(X)
        except ValueError as e:
            index = pd.RangeIndex(stop=len(X))
        if isinstance(self.splitter, str):
            splitter = getattr(self.splitter_cls, self.splitter)
        else:
            splitter = self.splitter
        splitter = splitter(
            index,
            template_context=self.template_context,
            **self.splitter_kwargs,
        )
        if splitter.get_n_sets(set_group_by=self.set_group_by) != 2:
            raise ValueError("Number of sets in the splitter must be 2: train and test")
        return splitter

    def _iter_masks(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> tp.Iterator[tp.Tuple[tp.Array1d, tp.Array1d]]:
        """Generate boolean masks corresponding to train and test splits.

        Args:
            X (Any): Input data.
            y (Any): Target values.
            groups (Any): Group labels.

        Returns:
            Iterator[Tuple[Array1d, Array1d]]: An iterator over tuples of
                boolean arrays for train and test masks.
        """
        splitter = self.get_splitter(X=X, y=y, groups=groups)
        for mask_arr in splitter.get_iter_split_mask_arrs(
            split_group_by=self.split_group_by,
            set_group_by=self.set_group_by,
            template_context=self.template_context,
        ):
            yield mask_arr[0], mask_arr[1]

    def _iter_train_masks(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> tp.Iterator[tp.Array1d]:
        """Generate boolean masks corresponding to train splits.

        Args:
            X (Any): Input data.
            y (Any): Target values.
            groups (Any): Group labels.

        Returns:
            Iterator[Array1d]: An iterator over boolean arrays for train masks.
        """
        for train_mask_arr, _ in self._iter_masks(X=X, y=y, groups=groups):
            yield train_mask_arr

    def _iter_test_masks(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> tp.Iterator[tp.Array1d]:
        """Generate boolean masks corresponding to test splits.

        Args:
            X (Any): Input data.
            y (Any): Target values.
            groups (Any): Group labels.

        Returns:
            Iterator[Array1d]: An iterator over boolean arrays for test masks.
        """
        for _, test_mask_arr in self._iter_masks(X=X, y=y, groups=groups):
            yield test_mask_arr

    def _iter_indices(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> tp.Iterator[tp.Tuple[tp.Array1d, tp.Array1d]]:
        """Generate integer indices corresponding to train and test splits.

        Args:
            X (Any): Input data.
            y (Any): Target values.
            groups (Any): Group labels.

        Returns:
            Iterator[Tuple[Array1d, Array1d]]: An iterator over tuples of integer indices
                for train and test splits.
        """
        for train_mask_arr, test_mask_arr in self._iter_masks(X=X, y=y, groups=groups):
            yield np.flatnonzero(train_mask_arr), np.flatnonzero(test_mask_arr)

    def _iter_train_indices(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> tp.Iterator[tp.Array1d]:
        """Generate integer indices corresponding to train splits.

        Args:
            X (Any): Input data.
            y (Any): Target values.
            groups (Any): Group labels.

        Returns:
            Iterator[Array1d]: An iterator over integer indices for train splits.
        """
        for train_indices, _ in self._iter_indices(X=X, y=y, groups=groups):
            yield train_indices

    def _iter_test_indices(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> tp.Iterator[tp.Array1d]:
        """Generate integer indices corresponding to test splits.

        Args:
            X (Any): Input data.
            y (Any): Target values.
            groups (Any): Group labels.

        Returns:
            Iterator[Array1d]: An iterator over integer indices for test splits.
        """
        for _, test_indices in self._iter_indices(X=X, y=y, groups=groups):
            yield test_indices

    def get_n_splits(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> int:
        """Return the number of splitting iterations in the cross-validator.

        Args:
            X (Any): Input data.
            y (Any): Target values.
            groups (Any): Group labels.

        Returns:
            int: The number of splits provided by the splitter.
        """
        splitter = self.get_splitter(X=X, y=y, groups=groups)
        return splitter.get_n_splits(split_group_by=self.split_group_by)

    def split(
        self,
        X: tp.Any = None,
        y: tp.Any = None,
        groups: tp.Any = None,
    ) -> tp.Iterator[tp.Tuple[tp.Array1d, tp.Array1d]]:
        """Generate indices to split data into training and test sets.

        Args:
            X (Any): Input data.
            y (Any): Target values.
            groups (Any): Group labels.

        Returns:
            Iterator[Tuple[Array1d, Array1d]]: An iterator yielding tuples of train and test indices.
        """
        return self._iter_indices(X=X, y=y, groups=groups)
