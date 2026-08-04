from __future__ import annotations

from track_record.models import PredictionRecord
from track_record.repository import SqlitePredictionRecordRepository


def _repo(tmp_path) -> SqlitePredictionRecordRepository:
    return SqlitePredictionRecordRepository(str(tmp_path / "test_track_record.db"))


def _record(**overrides) -> PredictionRecord:
    defaults = dict(
        username="alice",
        ticker="AAPL",
        model_name="Ridge (linear)",
        made_at="2026-08-03T12:00:00+00:00",
        target_date="2026-08-04",
        last_close=200.0,
        predicted_close=201.0,
        predicted_log_return=0.005,
    )
    defaults.update(overrides)
    return PredictionRecord(**defaults)


def test_save_then_get_history_roundtrips(tmp_path):
    repo = _repo(tmp_path)
    repo.save(_record())

    history = repo.get_history("alice", ticker="AAPL")
    assert len(history) == 1
    assert history[0].ticker == "AAPL"
    assert history[0].predicted_close == 201.0
    assert not history[0].is_resolved


def test_save_upserts_on_same_user_ticker_model_target_date(tmp_path):
    """Re-analyzing the same ticker/model before its target date resolves
    must update that user's pending prediction, not create a duplicate row."""
    repo = _repo(tmp_path)
    repo.save(_record(predicted_close=201.0))
    repo.save(_record(predicted_close=205.0))  # re-predicted later that day

    history = repo.get_history("alice", ticker="AAPL")
    assert len(history) == 1
    assert history[0].predicted_close == 205.0


def test_different_users_get_separate_rows_not_a_shared_one(tmp_path):
    """The whole point of scoping by username: two users predicting the
    same ticker/model/date must not collide into one shared record."""
    repo = _repo(tmp_path)
    repo.save(_record(username="alice", predicted_close=201.0))
    repo.save(_record(username="bob", predicted_close=999.0))

    alice_history = repo.get_history("alice", ticker="AAPL")
    bob_history = repo.get_history("bob", ticker="AAPL")
    assert len(alice_history) == 1 and alice_history[0].predicted_close == 201.0
    assert len(bob_history) == 1 and bob_history[0].predicted_close == 999.0


def test_resolve_sets_actual_close(tmp_path):
    repo = _repo(tmp_path)
    repo.save(_record())
    repo.resolve("alice", "AAPL", "Ridge (linear)", "2026-08-04", actual_close=203.0)

    record = repo.get_history("alice", ticker="AAPL")[0]
    assert record.is_resolved
    assert record.actual_close == 203.0
    assert record.resolved_at is not None


def test_resolve_only_affects_the_matching_user(tmp_path):
    repo = _repo(tmp_path)
    repo.save(_record(username="alice"))
    repo.save(_record(username="bob"))

    repo.resolve("alice", "AAPL", "Ridge (linear)", "2026-08-04", actual_close=203.0)

    assert repo.get_history("alice", ticker="AAPL")[0].is_resolved
    assert not repo.get_history("bob", ticker="AAPL")[0].is_resolved


def test_get_unresolved_before_only_returns_past_due_unresolved(tmp_path):
    repo = _repo(tmp_path)
    repo.save(_record(target_date="2026-08-01"))  # due
    repo.save(_record(ticker="MSFT", target_date="2026-08-10"))  # not due yet
    repo.resolve("alice", "AAPL", "Ridge (linear)", "2026-08-01", actual_close=199.0)  # resolved

    still_pending = repo.get_unresolved_before("2026-08-05")
    assert len(still_pending) == 0  # the only due one is already resolved

    repo.save(_record(ticker="GOOGL", target_date="2026-08-02"))
    due_now = repo.get_unresolved_before("2026-08-05")
    assert [r.ticker for r in due_now] == ["GOOGL"]


def test_get_unresolved_before_spans_all_users(tmp_path):
    """This one query is deliberately NOT scoped to a single user —
    resolving against real market data is the same fact-check for
    everyone, done in one pass."""
    repo = _repo(tmp_path)
    repo.save(_record(username="alice", target_date="2026-08-01"))
    repo.save(_record(username="bob", target_date="2026-08-01"))

    due = repo.get_unresolved_before("2026-08-05")
    assert {r.username for r in due} == {"alice", "bob"}


def test_resolve_is_idempotent_and_cannot_overwrite_an_already_resolved_record(tmp_path):
    """resolve() must be a true no-op on a record that's already resolved —
    not just harmless because callers happen to pre-filter to unresolved
    records (resolve_pending() does, but the repository shouldn't rely on
    that discipline to protect a real recorded outcome from being silently
    overwritten by a second call)."""
    repo = _repo(tmp_path)
    repo.save(_record())
    repo.resolve("alice", "AAPL", "Ridge (linear)", "2026-08-04", actual_close=203.0)

    repo.resolve("alice", "AAPL", "Ridge (linear)", "2026-08-04", actual_close=999.0)  # ignored

    record = repo.get_history("alice", ticker="AAPL")[0]
    assert record.actual_close == 203.0  # unchanged by the second call


def test_get_history_filters_by_ticker_and_model(tmp_path):
    repo = _repo(tmp_path)
    repo.save(_record(ticker="AAPL", model_name="Ridge (linear)"))
    repo.save(_record(ticker="AAPL", model_name="Gradient Boosting", target_date="2026-08-05"))
    repo.save(_record(ticker="MSFT", model_name="Ridge (linear)", target_date="2026-08-06"))

    assert len(repo.get_history("alice", ticker="AAPL")) == 2
    assert len(repo.get_history("alice", model_name="Ridge (linear)")) == 2
    assert len(repo.get_history("alice", ticker="AAPL", model_name="Gradient Boosting")) == 1
    assert len(repo.get_history("alice")) == 3
