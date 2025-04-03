# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing classes for scheduling data saves."""

import logging

from vectorbtpro import _typing as tp
from vectorbtpro.data.base import Data
from vectorbtpro.data.updater import DataUpdater
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "DataSaver",
    "CSVDataSaver",
    "HDFDataSaver",
    "SQLDataSaver",
    "DuckDBDataSaver",
]

logger = logging.getLogger(__name__)


class DataSaver(DataUpdater):
    """Class for scheduling data saves.

    Subclasses `vectorbtpro.data.updater.DataUpdater`.

    Args:
        data (Data): Data instance.
        save_kwargs (dict): Default keyword arguments for `DataSaver.init_save_data`
            and `DataSaver.save_data`.
        init_save_kwargs (dict): Keyword arguments overriding `save_kwargs` for initial data
            saving via `DataSaver.init_save_data`.
        **kwargs: Additional keyword arguments passed to the constructor of `DataUpdater`.
    """

    def __init__(
        self,
        data: Data,
        save_kwargs: tp.KwargsLike = None,
        init_save_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        DataUpdater.__init__(
            self,
            data=data,
            save_kwargs=save_kwargs,
            init_save_kwargs=init_save_kwargs,
            **kwargs,
        )
        self._save_kwargs = save_kwargs
        self._init_save_kwargs = init_save_kwargs

    @property
    def save_kwargs(self) -> tp.KwargsLike:
        """Property providing keyword arguments for data saving via `DataSaver.save_data`."""
        return self._save_kwargs

    @property
    def init_save_kwargs(self) -> tp.KwargsLike:
        """Property providing keyword arguments for initial data saving via `DataSaver.init_save_data`."""
        return self._init_save_kwargs

    def init_save_data(self, **kwargs) -> None:
        """Perform an initial data save.

        This method must be overridden in subclasses with custom logic.
        """
        raise NotImplementedError

    def save_data(self, **kwargs) -> None:
        """Perform a data save.

        This method must be overridden in subclasses to implement custom saving logic.
        """
        raise NotImplementedError

    def update(self, save_kwargs: tp.KwargsLike = None, **kwargs) -> None:
        """Update data and save changes using `DataSaver.save_data`.

        This method merges update parameters, updates the data, and then calls `DataSaver.save_data`.
        Override to add pre- and postprocessing behavior.

        To cancel subsequent updates, raise `vectorbtpro.utils.schedule_.CancelledError`.
        """
        # In case the method was called by the user
        kwargs = merge_dicts(
            dict(save_kwargs=self.save_kwargs),
            self.update_kwargs,
            {"save_kwargs": save_kwargs, **kwargs},
        )
        save_kwargs = kwargs.pop("save_kwargs")

        self._data = self.data.update(concat=False, **kwargs)
        self.update_config(data=self.data)
        if save_kwargs is None:
            save_kwargs = {}
        self.save_data(**save_kwargs)

    def update_every(
        self,
        *args,
        save_kwargs: tp.KwargsLike = None,
        init_save: bool = False,
        init_save_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        """Perform an update cycle with an optional initial data save.

        If `init_save` is True, an initial data save is performed via `DataSaver.init_save_data`.
        Overrides `vectorbtpro.data.updater.DataUpdater.update_every` to incorporate this step.
        """
        if init_save:
            init_save_kwargs = merge_dicts(
                self.save_kwargs,
                save_kwargs,
                self.init_save_kwargs,
                init_save_kwargs,
            )
            self.init_save_data(**init_save_kwargs)
        DataUpdater.update_every(self, *args, save_kwargs=save_kwargs, **kwargs)


class CSVDataSaver(DataSaver):
    """Class for saving data to CSV using `vectorbtpro.data.base.Data.to_csv`."""

    def init_save_data(self, **to_csv_kwargs) -> None:
        """Perform an initial data save to a CSV file."""
        # In case the method was called by the user
        to_csv_kwargs = merge_dicts(
            self.save_kwargs,
            self.init_save_kwargs,
            to_csv_kwargs,
        )

        self.data.to_csv(**to_csv_kwargs)
        new_index = self.data.wrapper.index
        logger.info(f"Saved initial {len(new_index)} rows from {new_index[0]} to {new_index[-1]}")

    def save_data(self, **to_csv_kwargs) -> None:
        """Append new data to a CSV file, omitting the header by default."""
        # In case the method was called by the user
        to_csv_kwargs = merge_dicts(
            dict(mode="a", header=False),
            self.save_kwargs,
            to_csv_kwargs,
        )

        self.data.to_csv(**to_csv_kwargs)
        new_index = self.data.wrapper.index
        logger.info(f"Saved {len(new_index)} rows from {new_index[0]} to {new_index[-1]}")


class HDFDataSaver(DataSaver):
    """Class for saving data to HDF format using `vectorbtpro.data.base.Data.to_hdf`."""

    def init_save_data(self, **to_hdf_kwargs) -> None:
        """Perform an initial data save to an HDF file."""
        # In case the method was called by the user
        to_hdf_kwargs = merge_dicts(
            self.save_kwargs,
            self.init_save_kwargs,
            to_hdf_kwargs,
        )

        self.data.to_hdf(**to_hdf_kwargs)
        new_index = self.data.wrapper.index
        logger.info(f"Saved initial {len(new_index)} rows from {new_index[0]} to {new_index[-1]}")

    def save_data(self, **to_hdf_kwargs) -> None:
        """Append new data to an HDF file in table format."""
        # In case the method was called by the user
        to_hdf_kwargs = merge_dicts(
            dict(mode="a", append=True),
            self.save_kwargs,
            to_hdf_kwargs,
        )

        self.data.to_hdf(**to_hdf_kwargs)
        new_index = self.data.wrapper.index
        logger.info(f"Saved {len(new_index)} rows from {new_index[0]} to {new_index[-1]}")


class SQLDataSaver(DataSaver):
    """Class for saving data to a SQL database using `vectorbtpro.data.base.Data.to_sql`."""

    def init_save_data(self, **to_sql_kwargs) -> None:
        """Perform an initial data save to a SQL database."""
        # In case the method was called by the user
        to_sql_kwargs = merge_dicts(
            self.save_kwargs,
            self.init_save_kwargs,
            to_sql_kwargs,
        )

        self.data.to_sql(**to_sql_kwargs)
        new_index = self.data.wrapper.index
        logger.info(f"Saved initial {len(new_index)} rows from {new_index[0]} to {new_index[-1]}")

    def save_data(self, **to_sql_kwargs) -> None:
        """Append new data to a SQL database table, omitting the header by default."""
        # In case the method was called by the user
        to_sql_kwargs = merge_dicts(
            dict(if_exists="append"),
            self.save_kwargs,
            to_sql_kwargs,
        )

        self.data.to_sql(**to_sql_kwargs)
        new_index = self.data.wrapper.index
        logger.info(f"Saved {len(new_index)} rows from {new_index[0]} to {new_index[-1]}")


class DuckDBDataSaver(DataSaver):
    """Class for saving data to a DuckDB database using `vectorbtpro.data.base.Data.to_duckdb`."""

    def init_save_data(self, **to_duckdb_kwargs) -> None:
        """Perform an initial data save to a DuckDB database."""
        # In case the method was called by the user
        to_duckdb_kwargs = merge_dicts(
            self.save_kwargs,
            self.init_save_kwargs,
            to_duckdb_kwargs,
        )

        self.data.to_duckdb(**to_duckdb_kwargs)
        new_index = self.data.wrapper.index
        logger.info(f"Saved initial {len(new_index)} rows from {new_index[0]} to {new_index[-1]}")

    def save_data(self, **to_duckdb_kwargs) -> None:
        """Append new data to a DuckDB database, omitting the header by default."""
        # In case the method was called by the user
        to_duckdb_kwargs = merge_dicts(
            dict(if_exists="append"),
            self.save_kwargs,
            to_duckdb_kwargs,
        )

        self.data.to_duckdb(**to_duckdb_kwargs)
        new_index = self.data.wrapper.index
        logger.info(f"Saved {len(new_index)} rows from {new_index[0]} to {new_index[-1]}")
