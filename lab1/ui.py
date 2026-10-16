from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

from forward_engine import ForwardInferenceEngine, InferenceResult
from models import Rule, ValueSource, VariableMetadata
from scenarios import SCENARIOS
from variables import INPUT_VARIABLES, VARIABLES


LINE = "=" * 58


class ConsoleUI:
    """Україномовний CLI, відділений від механізму виведення."""

    def __init__(self, engine: ForwardInferenceEngine) -> None:
        self.engine = engine
        used_condition_variables = {
            condition.variable for rule in engine.rules for condition in rule.conditions
        }
        self.manual_input_variables = tuple(
            variable
            for variable in INPUT_VARIABLES
            if variable in used_condition_variables
        )

    def run(self) -> None:
        self._show_header()
        while True:
            choice = self._main_menu()
            if choice == 6:
                print("\nБортову експертну систему завершено.")
                return

            if choice == 1:
                initial_facts = self._manual_facts()
                source = ValueSource.USER
                scenario_name = "Ручна діагностика"
            else:
                scenario_keys = (
                    "hull_breach",
                    "power_failure",
                    "communication_failure",
                    "engine_overheating",
                )
                scenario = SCENARIOS[scenario_keys[choice - 2]]
                initial_facts = scenario.facts
                source = ValueSource.SCENARIO
                scenario_name = scenario.name

            print(f"\n[ ЗАПУСК ] {scenario_name}")
            self._show_initial_data(initial_facts)
            result = self.engine.infer(
                initial_facts,
                source=source,
                fact_provider=self._provide_fact,
            )
            self._show_inference_summary(result)
            self._show_result(result)
            if not self._result_menu(result):
                return

    def _show_header(self) -> None:
        print(LINE)
        print("SPACECRAFT EMERGENCY DIAGNOSTIC SYSTEM")
        print(LINE)
        print("\nБортова експертна система активована.\n")
        print("[ SYSTEM STATUS ]")
        print("Expert system ........ ONLINE")
        print(f"Knowledge base ....... {len(self.engine.rules)} rules")
        print("Inference mode ....... FORWARD CHAINING")
        print("Search strategy ...... BREADTH-FIRST")

    def _main_menu(self) -> int:
        print("\nОберіть режим:\n")
        print("1. Ручна діагностика")
        print("2. Демонстраційний сценарій: розгерметизація")
        print("3. Демонстраційний сценарій: аварія енергосистеми")
        print("4. Демонстраційний сценарій: проблема зв'язку")
        print("5. Демонстраційний сценарій: перегрів двигуна")
        print("6. Вихід")
        return self._read_choice(1, 6)

    def _manual_facts(self) -> dict[str, str]:
        facts: dict[str, str] = {}
        while True:
            print("\n[ РУЧНА ДІАГНОСТИКА ]")
            print("Задайте лише відомі показники. Решту система запитає за потреби.\n")
            for number, variable in enumerate(self.manual_input_variables, start=1):
                metadata = VARIABLES[variable]
                current = facts.get(variable)
                suffix = f" [{metadata.value_label(current)}]" if current else ""
                print(f"{number}. {metadata.label}{suffix}")
            print(f"{len(self.manual_input_variables) + 1}. Почати діагностику")

            choice = self._read_choice(1, len(self.manual_input_variables) + 1)
            if choice == len(self.manual_input_variables) + 1:
                return facts
            variable = self.manual_input_variables[choice - 1]
            value = self._ask_value(VARIABLES[variable], allow_unknown=True)
            if value is None:
                facts.pop(variable, None)
            else:
                facts[variable] = value

    def _provide_fact(self, variable: str) -> Optional[str]:
        print("\n[ ПОТРІБНІ ДОДАТКОВІ ДАНІ ]")
        print("Для перевірки правила необхідна додаткова інформація.")
        return self._ask_value(VARIABLES[variable], allow_unknown=True)

    def _ask_value(
        self, metadata: VariableMetadata, *, allow_unknown: bool
    ) -> Optional[str]:
        print(f"\n{metadata.label}:")
        for number, (_, label) in enumerate(metadata.values, start=1):
            print(f"{number}. {label}")
        if allow_unknown:
            print(f"{len(metadata.values) + 1}. Невідомо")
        maximum = len(metadata.values) + int(allow_unknown)
        choice = self._read_choice(1, maximum)
        if allow_unknown and choice == maximum:
            return None
        return metadata.values[choice - 1][0]

    @staticmethod
    def _read_choice(minimum: int, maximum: int) -> int:
        while True:
            raw = input("Ваш вибір: ").strip()
            try:
                choice = int(raw)
            except ValueError:
                choice = minimum - 1
            if minimum <= choice <= maximum:
                return choice
            choices = ", ".join(str(number) for number in range(minimum, maximum + 1))
            print(f"Оберіть {choices}.")

    @staticmethod
    def _show_initial_data(facts: Mapping[str, str]) -> None:
        print("\n[ DATA ]")
        if not facts:
            print("Початкові факти не задано.")
            return
        for variable, value in facts.items():
            metadata = VARIABLES[variable]
            print(f"{metadata.label}: {metadata.value_label(value).upper()}")

    @staticmethod
    def _show_inference_summary(result: InferenceResult) -> None:
        for rule in result.fired_rules:
            print(f"\n[ INFERENCE ] {rule.id} спрацювало: {rule.description}")
            for condition in rule.conditions:
                print(f"  {condition.variable} = {condition.expected_value} ... TRUE")
            conclusion = rule.conclusion
            print(f"[ NEW FACT ] {conclusion.variable} = {conclusion.value}")

    @staticmethod
    def _show_result(result: InferenceResult) -> None:
        print(f"\n{LINE}")
        print("РЕЗУЛЬТАТ ДІАГНОСТИКИ")
        print(LINE)
        incident = result.working_memory.get_value("incident")
        action = result.working_memory.get_value("recommended_action")
        if incident:
            print("\nАварійна ситуація:")
            print(VARIABLES["incident"].value_label(incident).upper())
        else:
            print("\nАварійну ситуацію не встановлено за наявними даними.")
        if action:
            print("\nРекомендована дія:")
            print(VARIABLES["recommended_action"].value_label(action).upper())
        print(f"\nСпрацювало правил: {len(result.fired_rules)}")
        chain = " -> ".join(rule.id for rule in result.fired_rules)
        print(chain or "Логічний ланцюг відсутній")
        if result.last_conclusion:
            conclusion = result.last_conclusion
            print(f"\nОстанній висновок: {conclusion.variable} = {conclusion.value}")
        if result.conflicts:
            print("\n[ УВАГА ] Виявлено суперечливі можливі діагнози.")
        print(LINE)

    def _result_menu(self, result: InferenceResult) -> bool:
        while True:
            print("\n1. Показати логічний ланцюг")
            print("2. Показати робочу пам'ять")
            print("3. Показати історію FIFO-черги")
            print("4. Показати повний протокол перевірки правил")
            print("5. Нова діагностика")
            print("6. Вихід")
            choice = self._read_choice(1, 6)
            if choice == 1:
                self._show_logical_chain(result)
            elif choice == 2:
                self._show_working_memory(result)
            elif choice == 3:
                self._show_queue_history(result)
            elif choice == 4:
                self._show_protocol(result)
            elif choice == 5:
                return True
            else:
                print("\nБортову експертну систему завершено.")
                return False

    @staticmethod
    def _show_logical_chain(result: InferenceResult) -> None:
        print("\nЛОГІЧНИЙ ЛАНЦЮГ")
        if not result.logical_chain:
            print("Жодне правило не спрацювало.")
            return
        for number, rule in enumerate(result.logical_chain, start=1):
            print(f"\n{number}. {rule.id}")
            for condition_number, condition in enumerate(rule.conditions):
                prefix = "   " if condition_number == 0 else "   AND "
                print(f"{prefix}{condition.variable} = {condition.expected_value}")
            print("   ->")
            print(f"   {rule.conclusion.variable} = {rule.conclusion.value}")
        conclusion = result.last_conclusion
        if conclusion:
            print(f"\nВисновок:\n{conclusion.variable} = {conclusion.value}")

    @staticmethod
    def _show_working_memory(result: InferenceResult) -> None:
        print("\nРОБОЧА ПАМ'ЯТЬ")
        for fact in result.working_memory.all_facts():
            value = fact.value if fact.initialized else "None"
            source = fact.source.value if fact.source else "—"
            print(f"{fact.variable:<27} = {value:<31} [{source}]")

    @staticmethod
    def _show_queue_history(result: InferenceResult) -> None:
        print(f"\n{LINE}")
        print("ІСТОРІЯ FIFO-ЧЕРГИ")
        print(LINE)
        if not result.queue_events:
            print("Операцій з чергою не було.")
            return
        for event in result.queue_events:
            if event.action == "RULE_FIRED":
                print(f"\n[RULE {event.rule_id} FIRED]")
            else:
                print(f"\n[{event.action}]\n{event.variable}")
            queue = " -> ".join(event.queue) or "[порожня]"
            print(f"QUEUE:\n{queue}")

    @staticmethod
    def _show_protocol(result: InferenceResult) -> None:
        print("\nПОВНИЙ ПРОТОКОЛ ПЕРЕВІРКИ")
        if not result.protocol:
            print("Правила не перевірялися.")
            return
        for check in result.protocol:
            actual = check.actual_value if check.actual_value is not None else "UNKNOWN"
            status = check.status.value
            fired = "; RULE FIRED" if check.rule_fired else ""
            print(
                f"#{check.evaluation_number} {check.rule_id}, умова {check.condition_number}: "
                f"{check.variable} = {check.expected_value}; фактично {actual} "
                f"-> {status}{fired}"
            )


def rule_to_text(rule: Rule) -> str:
    conditions = " AND ".join(
        f"{condition.variable} = {condition.expected_value}" for condition in rule.conditions
    )
    return f"{rule.id}: {conditions} -> {rule.conclusion.variable} = {rule.conclusion.value}"
