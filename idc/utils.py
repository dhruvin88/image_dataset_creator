from __future__ import annotations

import asyncio
import time
from typing import List, Tuple, TypeVar

import httpx

T = TypeVar("T")


# ------------------------------------------------------------------ #
# HTTP retry
# ------------------------------------------------------------------ #


def retry_request(
    client: httpx.Client,
    method: str,
    url: str,
    max_retries: int = 3,
    backoff: float = 1.0,
    **kwargs,
) -> httpx.Response:
    """
    Make an HTTP request, retrying on 429 (rate-limit) and 5xx (server error).

    On 429 the Retry-After header is respected (capped at 300 s).
    On 5xx and transient network errors, exponential back-off is used.
    """
    for attempt in range(max_retries + 1):
        try:
            resp = client.request(method, url, **kwargs)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            if attempt < max_retries:
                time.sleep(backoff * (2**attempt))
                continue
            raise

        if resp.status_code == 429 and attempt < max_retries:
            wait = int(resp.headers.get("Retry-After", backoff * 60 * (2**attempt)))
            time.sleep(min(wait, 300))
            continue

        if resp.status_code >= 500 and attempt < max_retries:
            time.sleep(backoff * (2**attempt))
            continue

        resp.raise_for_status()
        return resp

    # Should not be reached, but satisfies type checker
    resp.raise_for_status()  # type: ignore[possibly-undefined]
    return resp  # type: ignore[possibly-undefined]


async def async_retry_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    max_retries: int = 3,
    backoff: float = 1.0,
    **kwargs,
) -> httpx.Response:
    """
    Async version of retry_request for use with httpx.AsyncClient.

    Retries on 429 (rate-limit) and 5xx (server error) with exponential back-off.
    On 429 the Retry-After header is respected (capped at 300 s).
    """
    for attempt in range(max_retries + 1):
        try:
            resp = await client.request(method, url, **kwargs)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            if attempt < max_retries:
                await asyncio.sleep(backoff * (2**attempt))
                continue
            raise

        if resp.status_code == 429 and attempt < max_retries:
            wait = int(resp.headers.get("Retry-After", backoff * 60 * (2**attempt)))
            await asyncio.sleep(min(wait, 300))
            continue

        if resp.status_code >= 500 and attempt < max_retries:
            await asyncio.sleep(backoff * (2**attempt))
            continue

        resp.raise_for_status()
        return resp

    resp.raise_for_status()  # type: ignore[possibly-undefined]
    return resp  # type: ignore[possibly-undefined]


# ------------------------------------------------------------------ #
# Split utility
# ------------------------------------------------------------------ #


def split_records(
    records: list,
    val_split: float = 0.1,
    test_split: float = 0.0,
) -> Tuple[list, list, list]:
    """
    Partition records into (train, val, test) lists.

    Sizes are computed as fractions of total; train gets the remainder.
    At least 1 record goes into val/test if splits are > 0 and records exist.
    """
    n = len(records)
    if n == 0:
        return [], [], []

    n_test = min(max(int(n * test_split), 1 if test_split > 0 else 0), n)
    n_val = min(max(int(n * val_split), 1 if val_split > 0 else 0), n - n_test)
    n_train = n - n_val - n_test

    train = records[:n_train]
    val = records[n_train : n_train + n_val]
    test = records[n_train + n_val :]
    return train, val, test
