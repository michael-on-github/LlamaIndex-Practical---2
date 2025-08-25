from typing import List

from pydantic import BaseModel, Field


class CandidateRoster(BaseModel):
    """Data model for a candidate."""

    job_title: str
    profession_or_field: str
    years_of_commercial_experience: int


class CandidateSummary(BaseModel):
    """Data model for a candidate's summary."""

    summary: str
    details: str
