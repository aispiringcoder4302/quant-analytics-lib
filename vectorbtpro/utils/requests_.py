# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Utilities for requests."""

from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from vectorbtpro import _typing as tp

__all__ = []


def requests_retry_session(
    retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: tp.Tuple[int, ...] = (500, 502, 504),
    session: tp.Optional[requests.Session] = None,
) -> requests.Session:
    """Retry `retries` times if unsuccessful."""
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def text_to_giphy_url(text: str, api_key: tp.Optional[str] = None, weirdness: tp.Optional[int] = None) -> str:
    """Translate text to GIF.

    See https://engineering.giphy.com/contextually-aware-search-giphy-gets-work-specific/"""
    from vectorbtpro._settings import settings

    giphy_cfg = settings["telegram"]["giphy"]

    if api_key is None:
        api_key = giphy_cfg["api_key"]
    if weirdness is None:
        weirdness = giphy_cfg["weirdness"]

    params = {"api_key": api_key, "s": text, "weirdness": weirdness}
    url = "http://api.giphy.com/v1/gifs/translate?" + urlencode(params)
    response = requests_retry_session().get(url)
    return response.json()["data"]["images"]["fixed_height"]["url"]
