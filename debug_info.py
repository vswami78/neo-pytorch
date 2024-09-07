from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class DebugInfo:
    original_node_counts: Dict[str, int] = field(default_factory=dict)
    filtered_node_counts: Dict[str, int] = field(default_factory=dict)
    filter_conditions: Dict[str, List[str]] = field(default_factory=dict)
    combine_method: str = ""
    messages: List[str] = field(default_factory=list)

    def add_message(self, message: str):
        self.messages.append(message)

    def __str__(self):
        return (
            f"Original node counts: {self.original_node_counts}\n"
            f"Filtered node counts: {self.filtered_node_counts}\n"
            f"Filter conditions: {self.filter_conditions}\n"
            f"Combine method: {self.combine_method}\n"
            f"Debug messages:\n" + "\n".join(self.messages)
        )