from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from data_access.openalgo_calendar import IST, OpenAlgoMarketCalendar


def _response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


class TestOpenAlgoMarketCalendar:
    def test_unconfigured_returns_none_without_network_call(self):
        cal = OpenAlgoMarketCalendar(base_url="", api_key="")
        with patch("data_access.openalgo_calendar.requests.post") as mock_post:
            session = cal.get_session("NSE")
            mock_post.assert_not_called()
        assert session is None

    def test_is_open_now_when_current_time_is_within_session(self):
        cal = OpenAlgoMarketCalendar(base_url="https://example.ngrok-free.dev", api_key="key123")
        now = datetime.now(IST)
        start_ms = int((now - timedelta(hours=1)).timestamp() * 1000)
        end_ms = int((now + timedelta(hours=1)).timestamp() * 1000)
        resp = _response({"status": "success", "data": [{"exchange": "NSE", "start_time": start_ms, "end_time": end_ms}]})
        with patch("data_access.openalgo_calendar.requests.post", return_value=resp):
            session = cal.get_session("NSE")
        assert session is not None
        assert session.is_open_now is True
        assert session.exchange == "NSE"

    def test_is_closed_when_current_time_is_outside_session(self):
        cal = OpenAlgoMarketCalendar(base_url="https://example.ngrok-free.dev", api_key="key123")
        now = datetime.now(IST)
        start_ms = int((now + timedelta(hours=2)).timestamp() * 1000)
        end_ms = int((now + timedelta(hours=8)).timestamp() * 1000)
        resp = _response({"status": "success", "data": [{"exchange": "NSE", "start_time": start_ms, "end_time": end_ms}]})
        with patch("data_access.openalgo_calendar.requests.post", return_value=resp):
            session = cal.get_session("NSE")
        assert session is not None
        assert session.is_open_now is False

    def test_returns_none_on_weekend_or_holiday_empty_response(self):
        cal = OpenAlgoMarketCalendar(base_url="https://example.ngrok-free.dev", api_key="key123")
        resp = _response({"status": "success", "data": []})
        with patch("data_access.openalgo_calendar.requests.post", return_value=resp):
            session = cal.get_session("NSE")
        assert session is None

    def test_returns_none_on_network_exception(self):
        cal = OpenAlgoMarketCalendar(base_url="https://example.ngrok-free.dev", api_key="key123")
        with patch("data_access.openalgo_calendar.requests.post", side_effect=Exception("timeout")):
            session = cal.get_session("NSE")
        assert session is None

    def test_returns_none_on_error_status(self):
        cal = OpenAlgoMarketCalendar(base_url="https://example.ngrok-free.dev", api_key="key123")
        resp = _response({"status": "error", "message": "bad key"})
        with patch("data_access.openalgo_calendar.requests.post", return_value=resp):
            session = cal.get_session("NSE")
        assert session is None
