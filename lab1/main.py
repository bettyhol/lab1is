from __future__ import annotations

from forward_engine import ForwardInferenceEngine
from knowledge_base import KNOWLEDGE_BASE
from ui import ConsoleUI
from variables import VARIABLES


def main() -> None:
    engine = ForwardInferenceEngine(KNOWLEDGE_BASE, VARIABLES)
    ConsoleUI(engine).run()


if __name__ == "__main__":
    main()
