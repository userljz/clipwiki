"""Standalone schemas used by ClipWiki."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskType(str, Enum):
    MULTIPLE_CHOICE = "multiple-choice"
    OPEN_QA = "open-qa"


class SessionTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str
    has_answer: bool | None = None


class HistoryClip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str
    conversation_id: str
    session_id: str
    speaker: str
    timestamp: datetime
    text: str
    turn_id: str | None = None
    source_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChoiceOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choice_id: str
    label: str
    text: str


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    dataset_name: str
    task_type: TaskType
    question: str
    choices: list[ChoiceOption] = Field(default_factory=list)
    history_clips: list[HistoryClip] = Field(default_factory=list)
    correct_choice_id: str | None = None
    correct_choice_index: int | None = None
    question_id: str | None = None
    question_type: str | None = None
    answer: str | None = None
    haystack_sessions: list[list[SessionTurn]] = Field(default_factory=list)
    haystack_session_ids: list[str] = Field(default_factory=list)
    haystack_session_summaries: list[str] = Field(default_factory=list)
    haystack_session_datetimes: list[datetime] = Field(default_factory=list)
    gold_evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def sync_multiple_choice_fields(self) -> "EvalCase":
        if self.question_id is None:
            self.question_id = self.example_id
        if self.question_type is None:
            self.question_type = str(self.metadata.get("question_type", "unknown"))
        if self.task_type == TaskType.OPEN_QA:
            if not self.answer:
                raise ValueError("answer must be provided for open-qa tasks")
            return self
        if not self.choices:
            raise ValueError("choices must not be empty")
        choice_ids = [choice.choice_id for choice in self.choices]
        if self.correct_choice_index is None:
            if self.correct_choice_id is None:
                raise ValueError("either correct_choice_index or correct_choice_id must be provided")
            try:
                self.correct_choice_index = choice_ids.index(self.correct_choice_id)
            except ValueError as error:
                raise ValueError("correct_choice_id must match one of the available choices") from error
        if self.correct_choice_index < 0 or self.correct_choice_index >= len(self.choices):
            raise ValueError("correct_choice_index is out of range")
        if self.correct_choice_id is None:
            self.correct_choice_id = self.choices[self.correct_choice_index].choice_id
        if self.answer is None:
            self.answer = self.choices[self.correct_choice_index].text
        return self


PreparedExample = EvalCase


class RetrievedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str
    rank: int
    score: float
    text: str
    retrieved_tokens: int = 0


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @model_validator(mode="after")
    def sync_total(self) -> "TokenUsage":
        self.total_tokens = self.input_tokens + self.output_tokens
        return self
