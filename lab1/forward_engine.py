from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

from models import (
    ConditionCheck,
    ConditionResult,
    Conclusion,
    QueueEvent,
    Rule,
    ValueSource,
    VariableMetadata,
)
from working_memory import WorkingMemory


FactProvider = Callable[[str], Optional[str]]


@dataclass
class InferenceResult:
    """Повний результат одного незалежного запуску виведення."""

    working_memory: WorkingMemory
    fired_rules: tuple[Rule, ...]
    protocol: tuple[ConditionCheck, ...]
    queue_history: tuple[str, ...]
    queue_events: tuple[QueueEvent, ...]
    processed_variables: tuple[str, ...]
    asked_variables: tuple[str, ...]
    conflicts: tuple[str, ...]

    @property
    def logical_chain(self) -> tuple[Rule, ...]:
        return self.fired_rules

    @property
    def last_conclusion(self) -> Optional[Conclusion]:
        if not self.fired_rules:
            return None
        return self.fired_rules[-1].conclusion


class ForwardInferenceEngine:
    """Універсальний рушій прямого виведення з BFS/FIFO-активацією."""

    def __init__(
        self,
        rules: Sequence[Rule],
        variables: Mapping[str, VariableMetadata],
    ) -> None:
        self.rules = tuple(rules)
        self.variables = dict(variables)
        self.rules_by_condition_variable = self._build_rule_index()
        self._validate_knowledge_base()

    def _build_rule_index(self) -> dict[str, tuple[Rule, ...]]:
        index: defaultdict[str, list[Rule]] = defaultdict(list)
        for rule in self.rules:
            for condition in rule.conditions:
                index[condition.variable].append(rule)
        return {variable: tuple(rules) for variable, rules in index.items()}

    def _validate_knowledge_base(self) -> None:
        rule_ids: set[str] = set()
        for rule in self.rules:
            if rule.id in rule_ids:
                raise ValueError(f"Повторний ID правила: {rule.id}")
            rule_ids.add(rule.id)
            if not rule.conditions:
                raise ValueError(f"Правило {rule.id} не має умов")
            for condition in rule.conditions:
                self._validate_value(condition.variable, condition.expected_value)
            self._validate_value(rule.conclusion.variable, rule.conclusion.value)

    def _validate_value(self, variable: str, value: str) -> None:
        metadata = self.variables.get(variable)
        if metadata is None:
            raise ValueError(f"Невідома змінна: {variable}")
        if value not in metadata.allowed_values:
            raise ValueError(f"Недопустиме значення {value!r} для {variable}")

    def infer(
        self,
        initial_facts: Mapping[str, str],
        *,
        source: ValueSource = ValueSource.USER,
        fact_provider: Optional[FactProvider] = None,
    ) -> InferenceResult:
        """Виконує новий сеанс прямого виведення без стану минулих запусків."""

        memory = WorkingMemory(self.variables)
        fifo: deque[str] = deque()
        queued: set[str] = set()
        processed: set[str] = set()
        queue_history: list[str] = []
        queue_events: list[QueueEvent] = []
        processed_history: list[str] = []

        def enqueue(variable: str) -> None:
            if variable not in queued and variable not in processed:
                fifo.append(variable)
                queued.add(variable)
                queue_history.append(variable)
                queue_events.append(QueueEvent("PUSH", variable, tuple(fifo)))

        for variable, value in initial_facts.items():
            self._validate_value(variable, value)
            memory.set_fact(variable, value, source)
            enqueue(variable)

        fired_ids: set[str] = set()
        fired_rules: list[Rule] = []
        protocol: list[ConditionCheck] = []
        requested: set[str] = set()
        asked_history: list[str] = []
        conflicts: list[str] = []
        evaluation_number = 0

        while fifo:
            trigger_variable = fifo.popleft()
            queued.remove(trigger_variable)
            processed.add(trigger_variable)
            processed_history.append(trigger_variable)
            queue_events.append(QueueEvent("POP", trigger_variable, tuple(fifo)))

            for rule in self.rules_by_condition_variable.get(trigger_variable, ()):
                if rule.id in fired_ids:
                    continue
                evaluation_number += 1

                # Якщо вже відома умова FALSE, додаткові дані не потрібні.
                known_false = any(
                    memory.is_initialized(condition.variable)
                    and memory.get_value(condition.variable) != condition.expected_value
                    for condition in rule.conditions
                )

                if not known_false and fact_provider is not None:
                    for condition in rule.conditions:
                        variable = condition.variable
                        metadata = self.variables[variable]
                        if (
                            not memory.is_initialized(variable)
                            and metadata.is_input
                            and variable not in requested
                        ):
                            requested.add(variable)
                            asked_history.append(variable)
                            supplied_value = fact_provider(variable)
                            if supplied_value is not None:
                                self._validate_value(variable, supplied_value)
                                if memory.set_fact(variable, supplied_value, ValueSource.USER):
                                    enqueue(variable)

                condition_statuses = [
                    ConditionResult.UNKNOWN
                    if not memory.is_initialized(condition.variable)
                    else (
                        ConditionResult.TRUE
                        if memory.get_value(condition.variable)
                        == condition.expected_value
                        else ConditionResult.FALSE
                    )
                    for condition in rule.conditions
                ]
                condition_results = [
                    status is ConditionResult.TRUE for status in condition_statuses
                ]
                all_true = all(condition_results)
                rule_fired = all_true

                if all_true:
                    conclusion = rule.conclusion
                    try:
                        is_new = memory.set_fact(
                            conclusion.variable, conclusion.value, ValueSource.RULE
                        )
                    except ValueError as error:
                        # Однозначна робоча пам'ять не перезаписує суперечливий діагноз.
                        rule_fired = False
                        conflicts.append(str(error))
                    else:
                        fired_ids.add(rule.id)
                        fired_rules.append(rule)
                        queue_events.append(
                            QueueEvent(
                                "RULE_FIRED",
                                conclusion.variable,
                                tuple(fifo),
                                rule.id,
                            )
                        )
                        if is_new:
                            enqueue(conclusion.variable)

                for number, (condition, result, status) in enumerate(
                    zip(rule.conditions, condition_results, condition_statuses), start=1
                ):
                    protocol.append(
                        ConditionCheck(
                            evaluation_number=evaluation_number,
                            trigger_variable=trigger_variable,
                            rule_id=rule.id,
                            condition_number=number,
                            variable=condition.variable,
                            expected_value=condition.expected_value,
                            actual_value=memory.get_value(condition.variable),
                            result=result,
                            status=status,
                            rule_fired=rule_fired,
                        )
                    )

        return InferenceResult(
            working_memory=memory,
            fired_rules=tuple(fired_rules),
            protocol=tuple(protocol),
            queue_history=tuple(queue_history),
            queue_events=tuple(queue_events),
            processed_variables=tuple(processed_history),
            asked_variables=tuple(asked_history),
            conflicts=tuple(conflicts),
        )
