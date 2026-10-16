from __future__ import annotations

import unittest
from unittest.mock import patch

from forward_engine import ForwardInferenceEngine
from knowledge_base import KNOWLEDGE_BASE
from models import ConditionResult, ValueSource
from scenarios import SCENARIOS
from ui import ConsoleUI
from variables import VARIABLES
from working_memory import WorkingMemory


class ForwardInferenceEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ForwardInferenceEngine(KNOWLEDGE_BASE, VARIABLES)

    def infer_scenario(self, name: str):
        return self.engine.infer(
            SCENARIOS[name].facts,
            source=ValueSource.SCENARIO,
            fact_provider=lambda _variable: None,
        )

    def test_hull_breach_scenario(self) -> None:
        result = self.infer_scenario("hull_breach")

        self.assertEqual("hull_breach", result.working_memory.get_value("incident"))
        self.assertEqual(
            "isolate_damaged_compartment",
            result.working_memory.get_value("recommended_action"),
        )
        self.assertEqual(["R1", "R2", "R3"], [rule.id for rule in result.fired_rules])

    def test_power_failure_scenario(self) -> None:
        result = self.infer_scenario("power_failure")

        self.assertEqual("critical", result.working_memory.get_value("power_problem"))
        self.assertEqual(
            "power_system_failure", result.working_memory.get_value("incident")
        )
        self.assertEqual(["R4", "R5"], [rule.id for rule in result.fired_rules])

    def test_communication_failure_scenario(self) -> None:
        result = self.infer_scenario("communication_failure")

        self.assertEqual(
            "detected", result.working_memory.get_value("communication_problem")
        )
        self.assertEqual(
            "communication_system_failure",
            result.working_memory.get_value("incident"),
        )
        self.assertEqual(["R6", "R7"], [rule.id for rule in result.fired_rules])

    def test_engine_overheating_scenario(self) -> None:
        result = self.infer_scenario("engine_overheating")

        self.assertEqual(
            "engine_cooling_failure", result.working_memory.get_value("incident")
        )
        self.assertEqual(
            "emergency_engine_shutdown",
            result.working_memory.get_value("recommended_action"),
        )
        self.assertEqual(["R8", "R9", "R10"], [rule.id for rule in result.fired_rules])

    def test_rule_does_not_fire_when_condition_is_false(self) -> None:
        provider_calls: list[str] = []
        result = self.engine.infer(
            {"oxygen_level": "normal"},
            fact_provider=lambda variable: provider_calls.append(variable) or "falling",
        )

        self.assertNotIn("R1", [rule.id for rule in result.fired_rules])
        self.assertIsNone(result.working_memory.get_value("atmosphere_leak"))
        self.assertEqual([], provider_calls, "Не слід питати дані, якщо вже є FALSE")

    def test_rule_never_fires_twice(self) -> None:
        result = self.engine.infer(
            {"oxygen_level": "low", "pressure": "falling"},
            fact_provider=lambda _variable: None,
        )

        fired_ids = [rule.id for rule in result.fired_rules]
        self.assertEqual(1, fired_ids.count("R1"))
        self.assertEqual(len(fired_ids), len(set(fired_ids)))

    def test_conclusion_is_added_to_working_memory_with_rule_source(self) -> None:
        result = self.engine.infer(
            {"generator": "failed", "battery_level": "low"},
            fact_provider=lambda _variable: None,
        )

        fact = result.working_memory.get_fact("power_problem")
        self.assertTrue(fact.initialized)
        self.assertEqual("critical", fact.value)
        self.assertEqual(ValueSource.RULE, fact.source)

    def test_new_conclusion_enters_fifo_and_activates_next_rule(self) -> None:
        result = self.engine.infer(
            {"generator": "failed", "battery_level": "low"},
            fact_provider=lambda _variable: None,
        )

        self.assertIn("power_problem", result.queue_history)
        self.assertIn("power_problem", result.processed_variables)
        self.assertLess(
            result.processed_variables.index("battery_level"),
            result.processed_variables.index("power_problem"),
        )
        self.assertEqual(["R4", "R5"], [rule.id for rule in result.logical_chain])

    def test_logical_chain_has_correct_breadth_first_order(self) -> None:
        result = self.infer_scenario("hull_breach")

        self.assertEqual(["R1", "R2", "R3"], [rule.id for rule in result.logical_chain])
        self.assertEqual(
            "recommended_action", result.last_conclusion.variable  # type: ignore[union-attr]
        )

    def test_repeated_inference_uses_fresh_working_memory(self) -> None:
        first = self.infer_scenario("hull_breach")
        second = self.infer_scenario("power_failure")

        self.assertIsNot(first.working_memory, second.working_memory)
        self.assertIsNone(second.working_memory.get_value("oxygen_level"))
        self.assertIsNone(second.working_memory.get_value("atmosphere_leak"))
        self.assertEqual(["R4", "R5"], [rule.id for rule in second.fired_rules])

    def test_unknown_input_can_be_resolved_by_callback(self) -> None:
        answers = {"pressure": "falling", "hull_sensor": "damaged"}
        result = self.engine.infer(
            {"oxygen_level": "low"}, fact_provider=answers.get
        )

        self.assertEqual(("pressure", "hull_sensor"), result.asked_variables)
        self.assertEqual(["R1", "R2", "R3"], [rule.id for rule in result.fired_rules])

    def test_intermediate_variables_are_never_requested(self) -> None:
        requested: list[str] = []
        self.engine.infer(
            {"hull_sensor": "damaged"},
            fact_provider=lambda variable: requested.append(variable) or None,
        )

        self.assertNotIn("atmosphere_leak", requested)
        self.assertTrue(all(VARIABLES[name].is_input for name in requested))

    def test_engine_does_not_use_console_input(self) -> None:
        with patch("builtins.input", side_effect=AssertionError("input() викликано")):
            result = self.infer_scenario("engine_overheating")

        self.assertEqual(["R8", "R9", "R10"], [rule.id for rule in result.fired_rules])

    def test_knowledge_base_and_condition_index_contain_ten_rules(self) -> None:
        indexed_ids = {
            rule.id
            for rules in self.engine.rules_by_condition_variable.values()
            for rule in rules
        }

        self.assertEqual(10, len(KNOWLEDGE_BASE))
        self.assertEqual({f"R{number}" for number in range(1, 11)}, indexed_ids)

    def test_empty_initial_facts_finish_without_conclusion(self) -> None:
        result = self.engine.infer({}, fact_provider=lambda _variable: None)

        self.assertEqual((), result.fired_rules)
        self.assertIsNone(result.last_conclusion)
        self.assertEqual((), result.queue_events)

    def test_identical_derived_fact_does_not_break_inference(self) -> None:
        result = self.engine.infer(
            {
                "oxygen_level": "low",
                "pressure": "falling",
                "atmosphere_leak": "detected",
            },
            fact_provider=lambda _variable: None,
        )

        self.assertIn("R1", [rule.id for rule in result.fired_rules])
        self.assertEqual("detected", result.working_memory.get_value("atmosphere_leak"))
        self.assertEqual((), result.conflicts)

    def test_working_memory_rejects_conflicting_fact(self) -> None:
        memory = WorkingMemory(VARIABLES)
        memory.set_fact("pressure", "falling", ValueSource.USER)

        with self.assertRaises(ValueError):
            memory.set_fact("pressure", "stable", ValueSource.USER)

        self.assertEqual("falling", memory.get_value("pressure"))

    def test_missing_fact_provider_answer_finishes_predictably(self) -> None:
        result = self.engine.infer(
            {"oxygen_level": "low"}, fact_provider=lambda _variable: None
        )

        self.assertEqual(("pressure",), result.asked_variables)
        self.assertEqual((), result.fired_rules)
        self.assertIsNone(result.last_conclusion)

    def test_protocol_distinguishes_unknown_from_false(self) -> None:
        unknown = self.engine.infer(
            {"oxygen_level": "low"}, fact_provider=lambda _variable: None
        )
        false = self.engine.infer(
            {"oxygen_level": "normal"}, fact_provider=lambda _variable: None
        )

        pressure_check = next(
            check
            for check in unknown.protocol
            if check.rule_id == "R1" and check.variable == "pressure"
        )
        oxygen_check = next(
            check
            for check in false.protocol
            if check.rule_id == "R1" and check.variable == "oxygen_level"
        )
        self.assertEqual(ConditionResult.UNKNOWN, pressure_check.status)
        self.assertEqual(ConditionResult.FALSE, oxygen_check.status)

    def test_fifo_events_contain_actual_push_and_pop_snapshots(self) -> None:
        result = self.infer_scenario("hull_breach")
        events = result.queue_events

        self.assertEqual(("PUSH", "oxygen_level", ("oxygen_level",)), (
            events[0].action, events[0].variable, events[0].queue
        ))
        first_pop = next(event for event in events if event.action == "POP")
        self.assertEqual("oxygen_level", first_pop.variable)
        self.assertEqual(("pressure", "hull_sensor"), first_pop.queue)
        r1_event = next(event for event in events if event.rule_id == "R1")
        atmosphere_push = next(
            event
            for event in events
            if event.action == "PUSH" and event.variable == "atmosphere_leak"
        )
        self.assertLess(events.index(r1_event), events.index(atmosphere_push))
        self.assertEqual(
            ("pressure", "hull_sensor", "atmosphere_leak"),
            atmosphere_push.queue,
        )

    def test_competing_incidents_preserve_first_and_record_conflict(self) -> None:
        result = self.engine.infer(
            {
                "oxygen_level": "low",
                "pressure": "falling",
                "hull_sensor": "damaged",
                "engine_temperature": "high",
                "cooling_system": "failed",
            },
            fact_provider=lambda _variable: None,
        )

        self.assertEqual("hull_breach", result.working_memory.get_value("incident"))
        self.assertEqual(1, len(result.conflicts))
        self.assertIn("engine_cooling_failure", result.conflicts[0])

    def test_unused_navigation_system_is_hidden_from_manual_input(self) -> None:
        ui = ConsoleUI(self.engine)

        self.assertNotIn("navigation_system", ui.manual_input_variables)
        self.assertIn("oxygen_level", ui.manual_input_variables)


if __name__ == "__main__":
    unittest.main()
