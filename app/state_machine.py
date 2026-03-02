"""Conversation state machine for the voice agent.

States:
  GREETING  → initial welcome
  LISTENING → waiting for user input
  THINKING  → processing / tool calls
  SPEAKING  → agent is responding
  ESCALATED → handed off to human
  ENDED     → session complete
"""

import enum
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("voice-agent.state")


class State(str, enum.Enum):
    GREETING = "greeting"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ESCALATED = "escalated"
    ENDED = "ended"


# Valid transitions
_TRANSITIONS: dict[State, set[State]] = {
    State.GREETING: {State.LISTENING, State.SPEAKING},
    State.LISTENING: {State.THINKING, State.ESCALATED, State.ENDED},
    State.THINKING: {State.SPEAKING, State.ESCALATED},
    State.SPEAKING: {State.LISTENING, State.ENDED},
    State.ESCALATED: {State.ENDED},
    State.ENDED: set(),
}


@dataclass
class ConversationContext:
    """Tracks conversation state + short-term memory."""

    state: State = State.GREETING
    turn_count: int = 0
    escalation_reason: str | None = None
    history: list[dict] = field(default_factory=list)
    _state_log: list[dict] = field(default_factory=list)

    # Escalation thresholds
    MAX_TURNS_BEFORE_ESCALATION_HINT: int = 10
    ESCALATION_KEYWORDS: tuple = (
        "speak to a human", "talk to someone", "real person",
        "manager", "supervisor", "escalate", "transfer",
        "representative", "agent", "operator",
    )

    def transition(self, new_state: State) -> None:
        if new_state not in _TRANSITIONS[self.state]:
            logger.warning("Invalid transition %s → %s", self.state, new_state)
            return
        old = self.state
        self.state = new_state
        self._state_log.append({
            "from": old.value, "to": new_state.value, "ts": time.time()
        })
        logger.debug("State: %s → %s", old.value, new_state.value)

    def add_turn(self, role: str, text: str) -> None:
        self.history.append({"role": role, "text": text, "ts": time.time()})
        if role == "user":
            self.turn_count += 1

    def should_escalate(self, user_text: str) -> bool:
        """Check if user is requesting human escalation."""
        lower = user_text.lower()
        return any(kw in lower for kw in self.ESCALATION_KEYWORDS)

    def needs_escalation_hint(self) -> bool:
        """After many turns, hint that escalation is available."""
        return self.turn_count >= self.MAX_TURNS_BEFORE_ESCALATION_HINT

    def get_memory_summary(self, last_n: int = 6) -> str:
        """Return recent conversation as context string for the LLM."""
        recent = self.history[-last_n:]
        return "\n".join(f"{m['role']}: {m['text']}" for m in recent)
