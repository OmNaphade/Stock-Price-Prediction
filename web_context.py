"""The app's composition root for Streamlit. All pages import from here
instead of each constructing their own auth/data-source/model stack —
this is what used to be two separate copies of the login logic and two
separate sqlite3 connections opened ad hoc."""

from __future__ import annotations

import streamlit as st

from auth import AuthService, SqliteUserRepository
from config import settings
from data_access import build_default_source
from data_access.macro import FredMacroSource, NullMacroSource
from data_access.news import build_default_news_source
from features.pipeline import FeaturePipeline
from features.sentiment import VaderSentimentScorer
from services import PredictionService, SentimentService


@st.cache_resource
def get_auth_service() -> AuthService:
    return AuthService(SqliteUserRepository(settings.db_path))


@st.cache_resource
def get_prediction_service() -> PredictionService:
    macro_source = FredMacroSource(settings.fred_api_key) if settings.fred_api_key else NullMacroSource()
    feature_pipeline = FeaturePipeline(macro_source=macro_source)
    return PredictionService(build_default_source(), feature_pipeline=feature_pipeline)


@st.cache_resource
def get_sentiment_service() -> SentimentService:
    return SentimentService(build_default_news_source(settings.news_api_key), VaderSentimentScorer())
