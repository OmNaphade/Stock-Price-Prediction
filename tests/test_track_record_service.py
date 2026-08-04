from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from services.track_record_service import TrackRecordService
from track_record.repository import SqlitePredictionRecordRepository


def _fake_report(ticker="AAPL", model_name="Ridge (linear)", target_date=None, last_close=200.0, predicted_close=202.0):
    return SimpleNamespace(
        ticker=ticker,
        model_name=model_name,
        target_date=target_date or pd.Timestamp(date.today() + timedelta(days=1)),
        last_close=last_close,
        predicted_next_close=predicted_close,
        predicted_log_return=0.01,
    )


class _FakeDataSource:
    def __init__(self, history_by_ticker: dict[str, pd.DataFrame]):
        self._history = history_by_ticker

    def get_history(self, ticker, start, end):
        df = self._history.get(ticker, pd.DataFrame())
        if df.empty:
            return df
        return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]

    def get_quote(self, ticker):
        return None


def _service(tmp_path, data_source=None) -> TrackRecordService:
    repo = SqlitePredictionRecordRepository(str(tmp_path / "test_track.db"))
    return TrackRecordService(repo, data_source or _FakeDataSource({}))


def test_record_prediction_then_appears_as_pending(tmp_path):
    service = _service(tmp_path)
    service.record_prediction(_fake_report())

    summary = service.get_track_record(ticker="AAPL")
    assert summary.pending_count == 1
    assert summary.resolved_count == 0


def test_resolve_pending_fills_in_actual_close_and_computes_accuracy(tmp_path):
    yesterday = date.today() - timedelta(days=1)
    history = pd.DataFrame(
        {"Close": [210.0]}, index=[pd.Timestamp(yesterday)]
    )
    service = _service(tmp_path, _FakeDataSource({"AAPL": history}))
    service.record_prediction(_fake_report(
        target_date=pd.Timestamp(yesterday), last_close=200.0, predicted_close=202.0,
    ))

    resolved_count = service.resolve_pending()
    assert resolved_count == 1

    summary = service.get_track_record(ticker="AAPL")
    assert summary.resolved_count == 1
    assert summary.pending_count == 0
    # predicted up (202 > 200), actual up (210 > 200) -> direction correct
    assert summary.directional_accuracy == 1.0
    record = summary.records[0]
    assert record.actual_close == 210.0


def test_resolve_pending_does_not_touch_future_target_dates(tmp_path):
    tomorrow = date.today() + timedelta(days=1)
    service = _service(tmp_path)
    service.record_prediction(_fake_report(target_date=pd.Timestamp(tomorrow)))

    resolved_count = service.resolve_pending()
    assert resolved_count == 0
    assert service.get_track_record(ticker="AAPL").pending_count == 1


def test_resolve_pending_leaves_record_pending_when_no_data_available(tmp_path):
    yesterday = date.today() - timedelta(days=1)
    service = _service(tmp_path, _FakeDataSource({}))  # no history at all -> can't resolve
    service.record_prediction(_fake_report(target_date=pd.Timestamp(yesterday)))

    resolved_count = service.resolve_pending()
    assert resolved_count == 0
    assert service.get_track_record(ticker="AAPL").pending_count == 1


def test_wrong_direction_prediction_is_scored_incorrect(tmp_path):
    yesterday = date.today() - timedelta(days=1)
    history = pd.DataFrame({"Close": [190.0]}, index=[pd.Timestamp(yesterday)])  # went DOWN
    service = _service(tmp_path, _FakeDataSource({"AAPL": history}))
    service.record_prediction(_fake_report(
        target_date=pd.Timestamp(yesterday), last_close=200.0, predicted_close=205.0,  # predicted UP
    ))
    service.resolve_pending()

    summary = service.get_track_record(ticker="AAPL")
    assert summary.directional_accuracy == 0.0
    assert summary.mean_abs_pct_error == pytest.approx(abs(205.0 - 190.0) / 190.0 * 100)
