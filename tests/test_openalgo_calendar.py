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


class TestGetNextHoliday:
    def test_unconfigured_returns_none_without_network_call(self):
        cal = OpenAlgoMarketCalendar(base_url="", api_key="")
        with patch("data_access.openalgo_calendar.requests.post") as mock_post:
            holiday = cal.get_next_holiday("NSE")
            mock_post.assert_not_called()
        assert holiday is None

    def test_returns_nearest_upcoming_holiday_for_exchange(self):
        cal = OpenAlgoMarketCalendar(base_url="https://example.ngrok-free.dev", api_key="key123")
        today = datetime.now(IST).date()
        past = (today - timedelta(days=5)).isoformat()
        near = (today + timedelta(days=10)).isoformat()
        far = (today + timedelta(days=40)).isoformat()
        resp = _response(
            {
                "status": "success",
                "data": [
                    {"date": past, "description": "Already passed", "closed_exchanges": ["NSE"]},
                    {"date": far, "description": "Far off holiday", "closed_exchanges": ["NSE"]},
                    {"date": near, "description": "Independence Day", "closed_exchanges": ["NSE", "BSE"]},
                ],
            }
        )
        with patch("data_access.openalgo_calendar.requests.post", return_value=resp) as mock_post:
            holiday = cal.get_next_holiday("NSE")

        assert holiday is not None
        assert holiday.date == near
        assert holiday.description == "Independence Day"
        called_url, called_kwargs = mock_post.call_args[0][0], mock_post.call_args[1]
        assert called_url == "https://example.ngrok-free.dev/api/v1/market/holidays"
        assert called_kwargs["json"] == {"apikey": "key123", "year": today.year}

    def test_ignores_holidays_for_other_exchanges(self):
        cal = OpenAlgoMarketCalendar(base_url="https://example.ngrok-free.dev", api_key="key123")
        today = datetime.now(IST).date()
        near = (today + timedelta(days=3)).isoformat()
        resp = _response(
            {"status": "success", "data": [{"date": near, "description": "BSE only", "closed_exchanges": ["BSE"]}]}
        )
        with patch("data_access.openalgo_calendar.requests.post", return_value=resp):
            holiday = cal.get_next_holiday("NSE")
        assert holiday is None

    def test_returns_none_when_no_upcoming_holidays_this_year(self):
        cal = OpenAlgoMarketCalendar(base_url="https://example.ngrok-free.dev", api_key="key123")
        resp = _response({"status": "success", "data": []})
        with patch("data_access.openalgo_calendar.requests.post", return_value=resp):
            holiday = cal.get_next_holiday("NSE")
        assert holiday is None

    def test_returns_none_on_error_status(self):
        cal = OpenAlgoMarketCalendar(base_url="https://example.ngrok-free.dev", api_key="key123")
        resp = _response({"status": "error", "message": "bad key"})
        with patch("data_access.openalgo_calendar.requests.post", return_value=resp):
            holiday = cal.get_next_holiday("NSE")
        assert holiday is None

    def test_returns_none_on_network_exception(self):
        cal = OpenAlgoMarketCalendar(base_url="https://example.ngrok-free.dev", api_key="key123")
        with patch("data_access.openalgo_calendar.requests.post", side_effect=Exception("timeout")):
            holiday = cal.get_next_holiday("NSE")
        assert holiday is None
