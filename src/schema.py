import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from dataclasses import dataclass


class TicketDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal["account", "billing", "technical", "other"]
    priority: int = Field(ge=1, le=5, strict=True)
    summary: str = Field(min_length=3, max_length=120)
    evidence: str | None = Field(min_length=1, max_length=160)


@dataclass
class DecisionResult:
    json_ok: bool
    schema_ok: bool
    ok: bool
    value: TicketDecision | None = None
    error_category: str | None = None
    public_message: str | None = None

PUBLIC_MESSAGES = {
    "invalid_json": "Model zwrócił niepoprawny format.",
    "schema_validation": "Odpowiedź nie spełnia kontraktu danych.",
    "invalid_evidence": "Nie można potwierdzić cytatu ze zgłoszenia.",
    "provider_error": "Nie udało się uzyskać odpowiedzi modelu.",
    "timeout": "Upłynął czas oczekiwania na model.",
}

def reject(code: str, *, json_ok: bool = False,
           schema_ok: bool = False) -> DecisionResult:
    return DecisionResult(
        json_ok=json_ok, schema_ok=schema_ok, ok=False,
        error_category=code, public_message=PUBLIC_MESSAGES,
    )

def validate_decision(raw_text: str, ticket: str) -> DecisionResult:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return reject("invalid_json")

    try:
        value = TicketDecision.model_validate(parsed)
    except ValidationError:
        return reject("schema_validation", json_ok=True)

    if value.category == "other":
        if value.evidence is not None:
            return reject("invalid_evidence", json_ok=True, schema_ok=True)
    elif value.evidence is None or value.evidence not in ticket:
        return reject("invalid_evidence", json_ok=True, schema_ok=True)

    return DecisionResult(json_ok=True, schema_ok=True, ok=True, value=value)