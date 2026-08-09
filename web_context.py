"""The app's composition root for Streamlit. All pages import from here
instead of each constructing their own auth/data-source/model stack —
this is what used to be two separate copies of the login logic and two
separate sqlite3 connections opened ad hoc."""

from __future__ import annotations

import streamlit as st

from auth import AuthService, SqliteOtpRepository, SqliteUserRepository, build_default_email_sender
from config import settings
from data_access import build_default_source
from data_access.macro import FredMacroSource, NullMacroSource
from data_access.news import build_default_news_source
from data_access.sources import MarketDataSource
from features.pipeline import FeaturePipeline
from features.sentiment import VaderSentimentScorer
from monitoring.experiment_tracking import build_experiment_tracker
from monitoring.sqlite_tracker import SqliteExperimentTracker
from services import ModelMonitoringService, PredictionService, SentimentService, TrackRecordService
from track_record.repository import SqlitePredictionRecordRepository


@st.cache_resource
def get_auth_service() -> AuthService:
    return AuthService(
        SqliteUserRepository(settings.db_path),
        SqliteOtpRepository(settings.db_path),
        build_default_email_sender(),
    )


@st.cache_resource
def get_market_data_source() -> MarketDataSource:
    # Shared by PredictionService and TrackRecordService so both reuse the
    # same provider chain and HTTP session, rather than each building its
    # own independent one.
    return build_default_source()


@st.cache_resource
def get_prediction_service() -> PredictionService:
    macro_source = FredMacroSource(settings.fred_api_key) if settings.fred_api_key else NullMacroSource()
    feature_pipeline = FeaturePipeline(macro_source=macro_source)
    return PredictionService(get_market_data_source(), feature_pipeline=feature_pipeline)


@st.cache_resource
def get_sentiment_service() -> SentimentService:
    return SentimentService(build_default_news_source(settings.news_api_key), VaderSentimentScorer())


@st.cache_resource
def get_track_record_service() -> TrackRecordService:
    repository = SqlitePredictionRecordRepository(settings.track_record_db_path)
    return TrackRecordService(repository, get_market_data_source())


@st.cache_resource
def get_monitoring_service() -> ModelMonitoringService:
    # The writer is build_experiment_tracker()'s composite (always-on
    # SQLite + optional MLflow); the reader is a plain SqliteExperimentTracker
    # of its own — the Monitoring page only ever reads from the always-on
    # store (MLflow isn't browsable in-app anyway). Two narrow interfaces,
    # injected separately, even though the writer's SQLite half and the
    # reader point at the same underlying file.
    tracker = build_experiment_tracker()
    reader = SqliteExperimentTracker(settings.monitoring_db_path)
    return ModelMonitoringService(tracker, reader)
