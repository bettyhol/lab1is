from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ValueSource(Enum):
    """Джерело факту в робочій пам'яті."""

    USER = "USER"
    RULE = "RULE"
    SCENARIO = "SCENARIO"


class ConditionResult(Enum):
    """Трьохзначний результат перевірки умови для протоколу."""

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Condition:
    variable: str
    expected_value: str


@dataclass(frozen=True)
class Conclusion:
    variable: str
    value: str


@dataclass(frozen=True)
class Rule:
    """Продукційне правило без залежностей від інтерфейсу."""

    id: str
    conditions: tuple[Condition, ...]
    conclusion: Conclusion
    description: str


@dataclass
class Fact:
    """Стан однієї змінної в робочій пам'яті."""

    variable: str
    initialized: bool = False
    value: Optional[str] = None
    source: Optional[ValueSource] = None


@dataclass(frozen=True)
class ConditionCheck:
    """Один запис детального протоколу перевірки умови."""

    evaluation_number: int
    trigger_variable: str
    rule_id: str
    condition_number: int
    variable: str
    expected_value: str
    actual_value: Optional[str]
    result: bool
    status: ConditionResult
    rule_fired: bool


@dataclass(frozen=True)
class QueueEvent:
    """Фактична операція FIFO або спрацювання правила між операціями."""

    action: str
    variable: str
    queue: tuple[str, ...]
    rule_id: Optional[str] = None


@dataclass(frozen=True)
class VariableMetadata:
    """Опис змінної, її допустимих значень та ролі."""

    name: str
    label: str
    values: tuple[tuple[str, str], ...]
    is_input: bool

    @property
    def allowed_values(self) -> tuple[str, ...]:
        return tuple(value for value, _ in self.values)

    def value_label(self, value: str) -> str:
        return dict(self.values).get(value, value)
