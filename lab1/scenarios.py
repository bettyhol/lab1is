from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    facts: dict[str, str]


SCENARIOS: dict[str, Scenario] = {
    "hull_breach": Scenario(
        "Розгерметизація",
        {"oxygen_level": "low", "pressure": "falling", "hull_sensor": "damaged"},
    ),
    "power_failure": Scenario(
        "Аварія енергосистеми",
        {"generator": "failed", "battery_level": "low"},
    ),
    "communication_failure": Scenario(
        "Проблема зв'язку",
        {"communication_module": "failed", "antenna": "damaged"},
    ),
    "engine_overheating": Scenario(
        "Перегрів двигуна",
        {"engine_temperature": "high", "cooling_system": "failed"},
    ),
}
