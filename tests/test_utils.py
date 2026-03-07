"""Tests for retry_request and split_records utilities."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import time

import httpx
import pytest

from idc.utils import retry_request, split_records


# ------------------------------------------------------------------ #
# split_records
# ------------------------------------------------------------------ #


class TestSplitRecords:
    def test_empty_input(self):
        train, val, test = split_records([], 0.1, 0.1)
        assert train == val == test == []

    def test_no_splits(self):
        data = list(range(10))
        train, val, test = split_records(data, 0.0, 0.0)
        assert train == data
        assert val == []
        assert test == []

    def test_val_only(self):
        data = list(range(10))
        train, val, test = split_records(data, 0.2, 0.0)
        assert len(val) == 2
        assert len(test) == 0
        assert len(train) + len(val) == 10

    def test_val_and_test(self):
        data = list(range(10))
        train, val, test = split_records(data, 0.2, 0.1)
        assert len(val) == 2
        assert len(test) == 1
        assert len(train) + len(val) + len(test) == 10

    def test_no_overlap(self):
        data = list(range(20))
        train, val, test = split_records(data, 0.2, 0.1)
        all_items = train + val + test
        assert sorted(all_items) == sorted(data)

    def test_at_least_one_in_val_when_split_positive(self):
        data = list(range(3))
        _, val, _ = split_records(data, 0.1, 0.0)  # 0.1 * 3 = 0.3 → rounds to 0 normally
        assert len(val) >= 1

    def test_large_dataset(self):
        data = list(range(1000))
        train, val, test = split_records(data, 0.1, 0.05)
        assert len(val) == 100
        assert len(test) == 50
        assert len(train) == 850


# ------------------------------------------------------------------ #
# retry_request
# ------------------------------------------------------------------ #


class TestRetryRequest:
    def _make_client(self, responses):
        """Build a mock httpx.Client that returns responses in sequence."""
        client = MagicMock()
        client.request.side_effect = responses
        return client

    def _ok(self, status=200):
        resp = MagicMock()
        resp.status_code = status
        resp.raise_for_status = MagicMock()
        resp.headers = {}
        return resp

    def _err(self, status):
        resp = MagicMock()
        resp.status_code = status
        resp.headers = {}
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "", request=MagicMock(), response=resp
        )
        return resp

    def test_success_on_first_try(self):
        client = self._make_client([self._ok()])
        resp = retry_request(client, "GET", "https://example.com")
        assert resp.status_code == 200
        assert client.request.call_count == 1

    def test_retries_on_500(self):
        with patch("time.sleep"):
            client = self._make_client([self._err(500), self._err(500), self._ok()])
            resp = retry_request(client, "GET", "https://example.com", max_retries=3)
        assert resp.status_code == 200
        assert client.request.call_count == 3

    def test_retries_on_429_with_retry_after(self):
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "1"}

        slept = []
        with patch("time.sleep", side_effect=lambda s: slept.append(s)):
            client = self._make_client([rate_limited, self._ok()])
            retry_request(client, "GET", "https://example.com", max_retries=3)

        assert len(slept) == 1
        assert slept[0] == 1  # Retry-After value

    def test_raises_after_max_retries_exceeded(self):
        with patch("time.sleep"):
            client = self._make_client([self._err(500)] * 4)
            with pytest.raises(httpx.HTTPStatusError):
                retry_request(client, "GET", "https://example.com", max_retries=3)

    def test_retries_on_timeout(self):
        with patch("time.sleep"):
            client = self._make_client(
                [httpx.TimeoutException("timeout"), self._ok()]
            )
            resp = retry_request(client, "GET", "https://example.com", max_retries=3)
        assert resp.status_code == 200

    def test_raises_timeout_after_max_retries(self):
        with patch("time.sleep"):
            client = self._make_client(
                [httpx.TimeoutException("timeout")] * 4
            )
            with pytest.raises(httpx.TimeoutException):
                retry_request(client, "GET", "https://example.com", max_retries=3)

    def test_passes_kwargs_to_client(self):
        client = self._make_client([self._ok()])
        retry_request(client, "GET", "https://example.com", params={"q": "test"})
        client.request.assert_called_once_with(
            "GET", "https://example.com", params={"q": "test"}
        )
