# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `FinPyData` class."""

from itertools import product

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.remote import RemoteData
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts

if tp.TYPE_CHECKING:
    from findatapy.market import Market as MarketT
    from findatapy.util import ConfigManager as ConfigManagerT
else:
    MarketT = "findatapy.market.Market"
    ConfigManagerT = "findatapy.util.ConfigManager"

__all__ = [
    "FinPyData",
]

FinPyDataT = tp.TypeVar("FinPyDataT", bound="FinPyData")


class FinPyData(RemoteData):
    """Class for fetching financial data using the findatapy API.

    See https://github.com/cuemacro/findatapy for API documentation.
    See `FinPyData.fetch_symbol` for argument details.

    Examples:
        Pull data (keyword argument format):

        ```pycon
        >>> data = vbt.FinPyData.pull(
        ...     "EURUSD",
        ...     start="14 June 2016",
        ...     end="15 June 2016",
        ...     timeframe="tick",
        ...     category="fx",
        ...     fields=["bid", "ask"],
        ...     data_source="dukascopy"
        ... )
        ```

        Pull data (string format):

        ```pycon
        >>> data = vbt.FinPyData.pull(
        ...     "fx.dukascopy.tick.NYC.EURUSD.bid,ask",
        ...     start="14 June 2016",
        ...     end="15 June 2016",
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.finpy")

    @classmethod
    def resolve_market(
        cls,
        market: tp.Optional[MarketT] = None,
        **market_config,
    ) -> MarketT:
        """Resolve and return a Market instance.

        Args:
            market (Optional[Market]): An optional market instance.
            **market_config: Additional configuration options for the market.

        Returns:
            Market: The resolved market instance.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("findatapy")
        from findatapy.market import Market, MarketDataGenerator

        market = cls.resolve_custom_setting(market, "market")
        if market_config is None:
            market_config = {}
        has_market_config = len(market_config) > 0
        market_config = cls.resolve_custom_setting(market_config, "market_config", merge=True)
        if "market_data_generator" not in market_config:
            market_config["market_data_generator"] = MarketDataGenerator()
        if market is None:
            market = Market(**market_config)
        elif has_market_config:
            raise ValueError("Cannot apply market_config to already initialized market")
        return market

    @classmethod
    def resolve_config_manager(
        cls,
        config_manager: tp.Optional[ConfigManagerT] = None,
        **config_manager_config,
    ) -> MarketT:
        """Resolve and return a ConfigManager instance.

        Args:
            config_manager (Optional[ConfigManager]): An optional configuration manager instance.
            **config_manager_config: Additional configuration options for the configuration manager.

        Returns:
            ConfigManager: The resolved configuration manager instance.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("findatapy")
        from findatapy.util import ConfigManager

        config_manager = cls.resolve_custom_setting(config_manager, "config_manager")
        if config_manager_config is None:
            config_manager_config = {}
        has_config_manager_config = len(config_manager_config) > 0
        config_manager_config = cls.resolve_custom_setting(config_manager_config, "config_manager_config", merge=True)
        if config_manager is None:
            config_manager = ConfigManager().get_instance(**config_manager_config)
        elif has_config_manager_config:
            raise ValueError("Cannot apply config_manager_config to already initialized config_manager")
        return config_manager

    @classmethod
    def list_symbols(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        config_manager: tp.Optional[ConfigManagerT] = None,
        config_manager_config: tp.KwargsLike = None,
        category: tp.Optional[tp.MaybeList[str]] = None,
        data_source: tp.Optional[tp.MaybeList[str]] = None,
        freq: tp.Optional[tp.MaybeList[str]] = None,
        cut: tp.Optional[tp.MaybeList[str]] = None,
        tickers: tp.Optional[tp.MaybeList[str]] = None,
        dict_filter: tp.DictLike = None,
        smart_group: bool = False,
        return_fields: tp.Optional[tp.MaybeList[str]] = None,
        combine_parts: bool = True,
    ) -> tp.List[str]:
        """List symbols matching the specified filters.

        This method passes most parameters to `findatapy.util.ConfigManager.free_form_tickers_regex_query`
        and uses `vectorbtpro.data.custom.custom.CustomData.key_match` to filter symbols based on `pattern`.

        Args:
            pattern (Optional[str]): A pattern to filter symbols.
            use_regex (bool): Whether the pattern should be interpreted as a regular expression.
            sort (bool): Whether to return the symbols in sorted order.
            config_manager (Optional[ConfigManager]): An optional configuration manager instance.
            config_manager_config (KwargsLike): Keyword arguments for configuring the configuration manager.
            category (Optional[MaybeList[str]]): A filter for the symbol category.
            data_source (Optional[MaybeList[str]]): A filter for the data source.
            freq (Optional[MaybeList[str]]): A filter for the frequency.
            cut (Optional[MaybeList[str]]): A filter for cut information.
            tickers (Optional[MaybeList[str]]): A filter for tickers.
            dict_filter (DictLike): A dictionary of additional filters.
            smart_group (bool): Whether to apply smart grouping to symbols.
            return_fields (Optional[MaybeList[str]]): Fields to include in the query output.
            combine_parts (bool): Whether to combine parts of symbol components into a single string.

        Returns:
            List[str]: A list of symbols that match the specified filters.
        """
        if config_manager_config is None:
            config_manager_config = {}
        config_manager = cls.resolve_config_manager(config_manager=config_manager, **config_manager_config)
        if dict_filter is None:
            dict_filter = {}
        def_ret_fields = ["category", "data_source", "freq", "cut", "tickers"]
        if return_fields is None:
            ret_fields = def_ret_fields
        elif isinstance(return_fields, str):
            if return_fields.lower() == "all":
                ret_fields = def_ret_fields + ["fields"]
            else:
                ret_fields = [return_fields]
        else:
            ret_fields = return_fields

        df = config_manager.free_form_tickers_regex_query(
            category=category,
            data_source=data_source,
            freq=freq,
            cut=cut,
            tickers=tickers,
            dict_filter=dict_filter,
            smart_group=smart_group,
            ret_fields=ret_fields,
        )
        all_symbols = []
        for _, row in df.iterrows():
            parts = []
            if "category" in row.index:
                parts.append(row.loc["category"])
            if "data_source" in row.index:
                parts.append(row.loc["data_source"])
            if "freq" in row.index:
                parts.append(row.loc["freq"])
            if "cut" in row.index:
                parts.append(row.loc["cut"])
            if "tickers" in row.index:
                parts.append(row.loc["tickers"])
            if "fields" in row.index:
                parts.append(row.loc["fields"])
            if combine_parts:
                split_parts = [part.split(",") for part in parts]
                combinations = list(product(*split_parts))
            else:
                combinations = [parts]
            for symbol in [".".join(combination) for combination in combinations]:
                if pattern is not None:
                    if not cls.key_match(symbol, pattern, use_regex=use_regex):
                        continue
                all_symbols.append(symbol)

        if sort:
            return sorted(dict.fromkeys(all_symbols))
        return list(dict.fromkeys(all_symbols))

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        market: tp.Optional[MarketT] = None,
        market_config: tp.KwargsLike = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.TimezoneLike = None,
        **request_kwargs,
    ) -> tp.SymbolData:
        """Fetch symbol data using findatapy.

        Overrides `vectorbtpro.data.base.Data.fetch_symbol` to retrieve and process symbol data
        from findatapy.

        Args:
            symbol (str): The symbol identifier.

                Accepts formats such as "fx.bloomberg.daily.NYC.EURUSD.close".

                The fields `freq`, `cut`, `tickers`, and `fields` are optional.
            market (Market): The market instance.

                See `FinPyData.resolve_market`.
            market_config (KwargsLike): The client configuration.

                See `FinPyData.resolve_market`.
            start (DatetimeLike): The start datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (DatetimeLike): The end datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            timeframe (str): The timeframe string.

                Accepts human-readable formats such as "15 minutes".
            tz (TimezoneLike): The timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            **request_kwargs: Keyword arguments passed to
                `findatapy.market.marketdatarequest.MarketDataRequest`.

        Returns:
            SymbolData: The fetched data and a metadata dictionary.

        For defaults, see `custom.finpy` in `vectorbtpro._settings.data`.
        Global settings can be provided per exchange id using the `exchanges` dictionary.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("findatapy")
        from findatapy.market import MarketDataRequest

        if market_config is None:
            market_config = {}
        market = cls.resolve_market(market=market, **market_config)

        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        timeframe = cls.resolve_custom_setting(timeframe, "timeframe")
        tz = cls.resolve_custom_setting(tz, "tz")
        request_kwargs = cls.resolve_custom_setting(request_kwargs, "request_kwargs", merge=True)

        split = dt.split_freq_str(timeframe)
        if split is None:
            raise ValueError(f"Invalid timeframe: '{timeframe}'")
        multiplier, unit = split

        if unit == "s":
            unit = "second"
            freq = timeframe
        elif unit == "m":
            unit = "minute"
            freq = timeframe
        elif unit == "h":
            unit = "hourly"
            freq = timeframe
        elif unit == "D":
            unit = "daily"
            freq = timeframe
        elif unit == "W":
            unit = "weekly"
            freq = timeframe
        elif unit == "M":
            unit = "monthly"
            freq = timeframe
        elif unit == "Q":
            unit = "quarterly"
            freq = timeframe
        elif unit == "Y":
            unit = "annually"
            freq = timeframe
        else:
            freq = None
        if "resample" in request_kwargs:
            freq = request_kwargs["resample"]

        if start is not None:
            start = dt.to_naive_datetime(dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc"))
        if end is not None:
            end = dt.to_naive_datetime(dt.to_tzaware_datetime(end, naive_tz=tz, tz="utc"))

        if "md_request" in request_kwargs:
            md_request = request_kwargs["md_request"]
        elif "md_request_df" in request_kwargs:
            md_request = market.create_md_request_from_dataframe(
                md_request_df=request_kwargs["md_request_df"],
                start_date=start,
                finish_date=end,
                freq_mult=multiplier,
                freq=unit,
                **request_kwargs,
            )
        elif "md_request_str" in request_kwargs:
            md_request = market.create_md_request_from_str(
                md_request_str=request_kwargs["md_request_str"],
                start_date=start,
                finish_date=end,
                freq_mult=multiplier,
                freq=unit,
                **request_kwargs,
            )
        elif "md_request_dict" in request_kwargs:
            md_request = market.create_md_request_from_dict(
                md_request_dict=request_kwargs["md_request_dict"],
                start_date=start,
                finish_date=end,
                freq_mult=multiplier,
                freq=unit,
                **request_kwargs,
            )
        elif symbol.count(".") >= 2:
            md_request = market.create_md_request_from_str(
                md_request_str=symbol,
                start_date=start,
                finish_date=end,
                freq_mult=multiplier,
                freq=unit,
                **request_kwargs,
            )
        else:
            md_request = MarketDataRequest(
                tickers=symbol,
                start_date=start,
                finish_date=end,
                freq_mult=multiplier,
                freq=unit,
                **request_kwargs,
            )

        df = market.fetch_market(md_request=md_request)
        if df is None:
            return None
        if isinstance(md_request.tickers, str):
            ticker = md_request.tickers
        elif len(md_request.tickers) == 1:
            ticker = md_request.tickers[0]
        else:
            ticker = None
        if ticker is not None:
            df.columns = df.columns.map(lambda x: x.replace(ticker + ".", ""))
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df = df.tz_localize("utc")
        return df, dict(tz=tz, freq=freq)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        fetch_kwargs["start"] = self.select_last_index(symbol)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)
