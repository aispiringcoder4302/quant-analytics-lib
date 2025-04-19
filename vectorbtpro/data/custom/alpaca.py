# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `AlpacaData` class for fetching data from Alpaca."""

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.remote import RemoteData
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.parsing import get_func_arg_names

if tp.TYPE_CHECKING:
    from alpaca.common.rest import RESTClient as RESTClientT
else:
    RESTClientT = "alpaca.common.rest.RESTClient"

__all__ = [
    "AlpacaData",
]

AlpacaDataT = tp.TypeVar("AlpacaDataT", bound="AlpacaData")


class AlpacaData(RemoteData):
    """Data class for fetching data from Alpaca.

    See:
        * https://github.com/alpacahq/alpaca-py for Alpaca API.
        * `AlpacaData.fetch_symbol` for argument details.

    !!! info
        For default settings, see `custom.alpaca` in `vectorbtpro._settings.data`.

        Global settings can be provided per exchange id using the `exchanges` dictionary.

    Examples:
        Set up the API key globally (optional for crypto):

        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.AlpacaData.set_custom_settings(
        ...     client_config=dict(
        ...         api_key="YOUR_KEY",
        ...         secret_key="YOUR_SECRET"
        ...     )
        ... )
        ```

        Pull stock data:

        ```pycon
        >>> data = vbt.AlpacaData.pull(
        ...     "AAPL",
        ...     start="2021-01-01",
        ...     end="2022-01-01",
        ...     timeframe="1 day"
        ... )
        ```

        Pull crypto data:

        ```pycon
        >>> data = vbt.AlpacaData.pull(
        ...     "BTC/USD",
        ...     start="2021-01-01",
        ...     end="2022-01-01",
        ...     timeframe="1 day"
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.alpaca")

    @classmethod
    def list_symbols(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        status: tp.Optional[str] = None,
        asset_class: tp.Optional[str] = None,
        exchange: tp.Optional[str] = None,
        trading_client: tp.Optional[RESTClientT] = None,
        client_config: tp.KwargsLike = None,
    ) -> tp.List[str]:
        """Return a list of symbols that match the specified criteria.

        This method filters symbols using `vectorbtpro.data.custom.custom.CustomData.key_match`
        based on the provided pattern.

        Args:
            pattern (Optional[str]): Pattern to filter symbols.
            use_regex (bool): Determines whether to treat `pattern` as a regular expression.
            sort (bool): Specifies whether to return the symbols in sorted order.
            status (Optional[str]): Filter for asset status.

                For possible values, refer to `alpaca.trading.enums`.
            asset_class (Optional[str]): Filter for asset class.

                For possible values, refer to `alpaca.trading.enums`.
            exchange (Optional[str]): Filter for the exchange.

                For possible values, refer to `alpaca.trading.enums`.
            trading_client (Optional[RESTClient]): An existing trading client instance.
            client_config (KwargsLike): Configuration parameters for initializing the trading client.

        Returns:
            List[str]: A list of symbol strings.

        !!! note
            If encountering an authorization error, verify that the `paper` flag in `client_config` is set
            appropriately based on the account credentials used (paper trading or live).
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("alpaca")
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetStatus, AssetClass, AssetExchange

        if client_config is None:
            client_config = {}
        has_client_config = len(client_config) > 0
        client_config = cls.resolve_custom_setting(client_config, "client_config", merge=True)
        if trading_client is None:
            arg_names = get_func_arg_names(TradingClient.__init__)
            client_config = {k: v for k, v in client_config.items() if k in arg_names}
            trading_client = TradingClient(**client_config)
        elif has_client_config:
            raise ValueError("Cannot apply client_config to already initialized client")

        if status is not None:
            if isinstance(status, str):
                status = getattr(AssetStatus, status.upper())
        if asset_class is not None:
            if isinstance(asset_class, str):
                asset_class = getattr(AssetClass, asset_class.upper())
        if exchange is not None:
            if isinstance(exchange, str):
                exchange = getattr(AssetExchange, exchange.upper())
        search_params = GetAssetsRequest(status=status, asset_class=asset_class, exchange=exchange)
        assets = trading_client.get_all_assets(search_params)
        all_symbols = []
        for asset in assets:
            symbol = asset.symbol
            if pattern is not None:
                if not cls.key_match(symbol, pattern, use_regex=use_regex):
                    continue
            all_symbols.append(symbol)

        if sort:
            return sorted(dict.fromkeys(all_symbols))
        return list(dict.fromkeys(all_symbols))

    @classmethod
    def resolve_client(
        cls,
        client: tp.Optional[RESTClientT] = None,
        client_type: tp.Optional[str] = None,
        **client_config,
    ) -> RESTClientT:
        """Resolve and return a trading client instance based on the provided parameters.

        If a client is provided, it must be of the type corresponding to `client_type`:
        `alpaca.data.historical.CryptoHistoricalDataClient` for `client_type="crypto"` and
        `alpaca.data.historical.StockHistoricalDataClient` for `client_type="stocks"`.
        If no client is provided, a new instance is created using the supplied `client_config`.

        Args:
            client (Optional[RESTClient]): An existing client instance to use.
            client_type (Optional[str]): Specifies the type of client to create;
                expected values are "crypto" or "stocks".
            **client_config: Keyword arguments for client initialization.

        Returns:
            RESTClient: An instance of the trading client.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("alpaca")
        from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

        client = cls.resolve_custom_setting(client, "client")
        client_type = cls.resolve_custom_setting(client_type, "client_type")
        if client_config is None:
            client_config = {}
        has_client_config = len(client_config) > 0
        client_config = cls.resolve_custom_setting(client_config, "client_config", merge=True)
        if client is None:
            if client_type == "crypto":
                arg_names = get_func_arg_names(CryptoHistoricalDataClient.__init__)
                client_config = {k: v for k, v in client_config.items() if k in arg_names}
                client = CryptoHistoricalDataClient(**client_config)
            elif client_type == "stocks":
                arg_names = get_func_arg_names(StockHistoricalDataClient.__init__)
                client_config = {k: v for k, v in client_config.items() if k in arg_names}
                client = StockHistoricalDataClient(**client_config)
            else:
                raise ValueError(f"Invalid client type: '{client_type}'")
        elif has_client_config:
            raise ValueError("Cannot apply client_config to already initialized client")
        return client

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        client: tp.Optional[RESTClientT] = None,
        client_type: tp.Optional[str] = None,
        client_config: tp.KwargsLike = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.TimezoneLike = None,
        adjustment: tp.Optional[str] = None,
        feed: tp.Optional[str] = None,
        limit: tp.Optional[int] = None,
    ) -> tp.SymbolData:
        """Fetch a symbol from Alpaca via overriding `vectorbtpro.data.base.Data.fetch_symbol`.

        Args:
            symbol (str): The symbol to fetch.
            client (Optional[RESTClient]): Alpaca REST client instance.

                See `AlpacaData.resolve_client`.
            client_type (Optional[str]): Type of Alpaca client.

                Automatically determined based on the symbol, as crypto symbols contain "/".
                Also see `AlpacaData.resolve_client`.
            client_config (KwargsLike): Configuration for the Alpaca client.

                See `AlpacaData.resolve_client`.
            start (Optional[DatetimeLike]): Start datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (Optional[DatetimeLike]): End datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            timeframe (Optional[str]): Data timeframe.

                Accepts human-readable strings such as "15 minutes".
            tz (TimezoneLike): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            adjustment (Optional[str]): Corporate action adjustment for returned bars.

                Options: "raw", "split", "dividend", or "all".
            feed (Optional[str]): Market data feed.

                Options: "iex", "otc", or "sip". Feeds "sip" and "otc" require a subscription.
            limit (Optional[int]): Maximum number of items to return.

        Returns:
            SymbolData: The fetched data and a metadata dictionary.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("alpaca")
        from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        if client_type is None:
            client_type = "crypto" if "/" in symbol else "stocks"

        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, client_type=client_type, **client_config)

        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        timeframe = cls.resolve_custom_setting(timeframe, "timeframe")
        tz = cls.resolve_custom_setting(tz, "tz")
        adjustment = cls.resolve_custom_setting(adjustment, "adjustment")
        feed = cls.resolve_custom_setting(feed, "feed")
        limit = cls.resolve_custom_setting(limit, "limit")

        freq = timeframe
        split = dt.split_freq_str(timeframe)
        if split is None:
            raise ValueError(f"Invalid timeframe: '{timeframe}'")
        multiplier, unit = split
        if unit == "m":
            unit = TimeFrameUnit.Minute
        elif unit == "h":
            unit = TimeFrameUnit.Hour
        elif unit == "D":
            unit = TimeFrameUnit.Day
        elif unit == "W":
            unit = TimeFrameUnit.Week
        elif unit == "M":
            unit = TimeFrameUnit.Month
        else:
            raise ValueError(f"Invalid timeframe: '{timeframe}'")
        timeframe = TimeFrame(multiplier, unit)

        if start is not None:
            start = dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc")
            start_str = start.replace(tzinfo=None).isoformat("T")
        else:
            start_str = None
        if end is not None:
            end = dt.to_tzaware_datetime(end, naive_tz=tz, tz="utc")
            end_str = end.replace(tzinfo=None).isoformat("T")
        else:
            end_str = None

        if isinstance(client, CryptoHistoricalDataClient):
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_str,
                end=end_str,
                limit=limit,
            )
            df = client.get_crypto_bars(request).df
        elif isinstance(client, StockHistoricalDataClient):
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_str,
                end=end_str,
                limit=limit,
                adjustment=adjustment,
                feed=feed,
            )
            df = client.get_stock_bars(request).df
        else:
            raise TypeError(f"Invalid client of type {type(client)}")

        df = df.droplevel("symbol", axis=0)
        df.index = df.index.rename("Open time")
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "trade_count": "Trade count",
                "vwap": "VWAP",
            },
            inplace=True,
        )
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df = df.tz_localize("utc")

        if "Open" in df.columns:
            df["Open"] = df["Open"].astype(float)
        if "High" in df.columns:
            df["High"] = df["High"].astype(float)
        if "Low" in df.columns:
            df["Low"] = df["Low"].astype(float)
        if "Close" in df.columns:
            df["Close"] = df["Close"].astype(float)
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(float)
        if "Trade count" in df.columns:
            df["Trade count"] = df["Trade count"].astype(int, errors="ignore")
        if "VWAP" in df.columns:
            df["VWAP"] = df["VWAP"].astype(float)

        if not df.empty:
            if start is not None:
                start = dt.to_timestamp(start, tz=df.index.tz)
                if df.index[0] < start:
                    df = df[df.index >= start]
            if end is not None:
                end = dt.to_timestamp(end, tz=df.index.tz)
                if df.index[-1] >= end:
                    df = df[df.index < end]
        return df, dict(tz=tz, freq=freq)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        fetch_kwargs["start"] = self.select_last_index(symbol)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)
