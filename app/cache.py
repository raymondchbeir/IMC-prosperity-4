from __future__ import annotations

import time
import uuid
from collections import OrderedDict

import pandas as pd


# Small in-process cache. Good for local Dash usage.
# If you run multiple gunicorn workers later, move this to disk/Redis.
_SESSION_CACHE: OrderedDict[str, dict] = OrderedDict()
_MAX_SESSIONS = 4


def store_session(prices_df: pd.DataFrame, trades_df: pd.DataFrame) -> str:
    session_id = str(uuid.uuid4())

    _SESSION_CACHE[session_id] = {
        "created_at": time.time(),
        "last_access": time.time(),
        "prices": prices_df,
        "trades": trades_df,
    }
    _SESSION_CACHE.move_to_end(session_id)

    while len(_SESSION_CACHE) > _MAX_SESSIONS:
        _SESSION_CACHE.popitem(last=False)

    return session_id


def get_session_frames(session_id: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not session_id or session_id not in _SESSION_CACHE:
        return pd.DataFrame(), pd.DataFrame()

    item = _SESSION_CACHE[session_id]
    item["last_access"] = time.time()
    _SESSION_CACHE.move_to_end(session_id)

    return item["prices"], item["trades"]


def cache_info() -> dict:
    return {
        "sessions": len(_SESSION_CACHE),
        "ids": list(_SESSION_CACHE.keys()),
    }
