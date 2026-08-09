from __future__ import annotations

from unittest.mock import MagicMock, patch

from data_access.openalgo_source import OpenAlgoSource, _split_ticker


def _response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


class TestSplitTicker:
    def test_ns_suffix_maps_to_nse(self):
        assert _split_ticker("RELIANCE.NS") == ("RELIANCE", "NSE")

    def test_bo_suffix_maps_to_bse(self):
        assert _split_ticker("TCS.BO") == ("TCS", "BSE")

    def test_us_ticker_returns_none(self):
        assert _split_ticker("AAPL") is None

    def test_unknown_suffix_returns_none(self):
        assert _split_ticker("FOO.LSE") is None

    def test_none_ticker_returns_none_instead_of_raising(self):
        """Regression test: reproduced live as an unhandled TypeError
        ("argument of type 'NoneType' is not iterable") — a None ticker
        must degrade gracefully like every other bad input, not crash."""
        assert _split_ticker(None) is None

    def test_empty_string_ticker_returns_none(self):
        assert _split_ticker("") is None


class TestOpenAlgoSource:
    def test_unconfigured_source_returns_empty_without_network_call(self):
        source = OpenAlgoSource(base_url="", api_key="")
        with patch("data_access.openalgo_source.requests.post") as mock_post:
            df = source.get_history("RELIANCE.NS", "2024-01-01", "2024-01-31")
            mock_post.assert_not_called()
        assert df.empty

    def test_get_history_with_none_ticker_does_not_raise(self):
        """Regression test at the public-method level, matching exactly
        how this was first reproduced: PredictionService.analyze(None, ...)
        propagating straight down to get_history via the composite chain."""
        source = OpenAlgoSource(base_url="https://example.ngrok-free.dev", api_key="key123")
        assert source.get_history(None, "2024-01-01", "2024-01-31").empty

    def test_get_quote_with_none_ticker_does_not_raise(self):
        source = OpenAlgoSource(base_url="https://example.ngrok-free.dev", api_key="key123")
        assert source.get_quote(None) is None

    def test_get_history_returns_empty_for_non_indian_ticker(self):
        source = OpenAlgoSource(base_url="https://example.ngrok-free.dev", api_key="key123")
        with patch("data_access.openalgo_source.requests.post") as mock_post:
            df = source.get_history("AAPL", "2024-01-01", "2024-01-31")
            mock_post.assert_not_called()
        assert df.empty

    def test_get_history_returns_candles_on_success(self):
        source = OpenAlgoSource(base_url="https://example.ngrok-free.dev/", api_key="key123")
        candle_resp = _response(
            {
                "status": "success",
                "data": [
                    {"timestamp": "2024-01-01 09:15:00+05:30", "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1000},
                    {"timestamp": "2024-01-02 09:15:00+05:30", "open": 104.0, "high": 108.0, "low": 103.0, "close": 107.0, "volume": 1200},
                ],
            }
        )
        with patch("data_access.openalgo_source.requests.post", return_value=candle_resp) as mock_post:
            df = source.get_history("RELIANCE.NS", "2024-01-01", "2024-01-02")

        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(df) == 2
        assert df["Close"].iloc[-1] == 107.0
        # Base URL trailing slash is stripped, and the request carries a
        # single flat apikey field — no separate auth header, no login.
        called_url, called_kwargs = mock_post.call_args[0][0], mock_post.call_args[1]
        assert called_url == "https://example.ngrok-free.dev/api/v1/history"
        assert called_kwargs["json"] == {
            "apikey": "key123", "symbol": "RELIANCE", "exchange": "NSE",
            "interval": "D", "start_date": "2024-01-01", "end_date": "2024-01-02",
        }

    def test_get_history_returns_empty_on_error_status(self):
        source = OpenAlgoSource(base_url="https://example.ngrok-free.dev", api_key="key123")
        with patch(
            "data_access.openalgo_source.requests.post",
            return_value=_response({"status": "error", "message": "Invalid API key"}),
        ):
            df = source.get_history("RELIANCE.NS", "2024-01-01", "2024-01-02")
        assert df.empty

    def test_get_history_returns_empty_on_network_exception(self):
        source = OpenAlgoSource(base_url="https://example.ngrok-free.dev", api_key="key123")
        with patch("data_access.openalgo_source.requests.post", side_effect=Exception("connection refused")):
            df = source.get_history("RELIANCE.NS", "2024-01-01", "2024-01-02")
        assert df.empty

    def test_get_quote_returns_ltp_on_success(self):
        source = OpenAlgoSource(base_url="https://example.ngrok-free.dev", api_key="key123")
        quote_resp = _response(
            {"status": "success", "data": {"open": 1172.0, "high": 1196.6, "low": 1163.3, "ltp": 1187.75, "volume": 14414545}}
        )
        with patch("data_access.openalgo_source.requests.post", return_value=quote_resp) as mock_post:
            price = source.get_quote("RELIANCE.NS")
        assert price == 1187.75
        assert mock_post.call_args[0][0] == "https://example.ngrok-free.dev/api/v1/quotes"

    def test_get_quote_returns_none_for_non_indian_ticker(self):
        source = OpenAlgoSource(base_url="https://example.ngrok-free.dev", api_key="key123")
        assert source.get_quote("TSLA") is None

    def test_get_quote_returns_none_when_unconfigured(self):
        source = OpenAlgoSource(base_url="", api_key="")
        assert source.get_quote("RELIANCE.NS") is None
