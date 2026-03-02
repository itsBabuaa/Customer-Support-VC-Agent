"""NovaLaptops — Voice Customer Support Agent.

Autonomous voice AI with tool use, RAG, memory, human escalation,
interrupt handling, and full observability.
"""

import json
import logging
import uuid
from datetime import datetime, timezone

from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.agents.metrics import (
    LLMMetrics, TTSMetrics, STTMetrics, EOUMetrics, VADMetrics, UsageCollector,
)
from livekit.plugins import groq, silero, deepgram, elevenlabs
from livekit.plugins.noise_cancellation import BVC

from app.config import (
    LOGS_DIR, TRANSCRIPTS_DIR, COST_RATES,
    LLM_MODEL, STT_MODEL, STT_LANGUAGE,
    TTS_VOICE_ID, TTS_MODEL,
    VAD_MIN_SPEECH, VAD_MIN_SILENCE, VAD_THRESHOLD,
    MIN_INTERRUPTION_DURATION, MIN_ENDPOINTING_DELAY,
    MAX_ENDPOINTING_DELAY, FALSE_INTERRUPTION_TIMEOUT,
)
from app.tools import ALL_TOOLS
from app.state_machine import ConversationContext, State

logger = logging.getLogger("voice-agent")
logger.setLevel(logging.DEBUG)

SYSTEM_PROMPT = """\
You are a phone support agent at NovaLaptops, an online laptop retailer.

ABSOLUTE RULE — BREVITY (THIS IS THE #1 PRIORITY):
- MAXIMUM 2 sentences per reply. This is a PHONE CALL — less is more.
- For product questions: model name + one key fact. Ask if they want more detail.
- For order status: status + expected date. Nothing else.
- NEVER list all products in one reply. Mention one, then ask.
- NEVER use filler words or repeat what the user said.

RULES:
1. LANGUAGE: Reply in the same language the customer uses.
2. TOOL USAGE:
   - Greetings, thanks, goodbyes, chitchat → answer directly. No tool call.
   - Specific product question → call search_knowledge or get_laptop_specs.
   - "What laptops do you have?" → call list_all_laptops.
   - Order ID (like NLT-10001) → call check_order.
   - Phone number → call check_order_by_phone.
   - Email → call check_order_by_email.
   - Call each tool EXACTLY ONCE per turn. Never call the same tool twice.
3. ESCALATION:
   - Customer asks for a human → call escalate_to_human.
   - Hardware defect or refund dispute → call escalate_to_human.
   - After 3 failed attempts to resolve → offer escalation.
4. RETURNS: 30-day return policy. Must be in original packaging. Refund within 5-7 business days.
5. WARRANTY: Standard 1-2 year warranty depending on model. Covers manufacturing defects, not accidental damage.
6. If you don't know → say so, offer to transfer to a specialist.
7. Never invent specs or prices.
8. NEVER mention tools, databases, or knowledge bases. Answer naturally.

KEY PRODUCT LINEUP (answer general questions without tools):
- NovaPro 16: $1,899 — i9/RTX 4070/32GB — professional workstation
- NovaAir 14: $1,299 — Ultra 7/OLED/1.2kg — ultrabook
- NovaBook 15: $699 — Ryzen 5/16GB — student/budget
- NovaForce 17: $2,499 — Ryzen 9/RTX 4080/240Hz — gaming
- Free shipping on orders over $999
- Payment: Credit card, debit card, PayPal, financing (0% APR 12 months)
"""


async def entrypoint(ctx):
    """LiveKit agent entrypoint — one instance per call session."""
    session_id = str(uuid.uuid4())[:12]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = f"{ts}_{session_id}.log"
    transcript_file = f"{ts}_{session_id}.json"

    # Per-session file logger
    slog = logging.getLogger(f"session-{session_id}")
    slog.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOGS_DIR / log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    slog.addHandler(fh)

    slog.info("Session started | id=%s | room=%s", session_id, ctx.room.name)

    conv = ConversationContext()
    transcript = {"session_id": session_id, "utterances": []}
    usage = UsageCollector()

    turn_metrics = {
        "eou_delay": 0.0, "transcription_delay": 0.0,
        "llm_ttft": 0.0, "llm_duration": 0.0,
        "tts_ttfb": 0.0, "tts_duration": 0.0,
        "turn_count": 0,
    }

    def save_transcript():
        with open(TRANSCRIPTS_DIR / transcript_file, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)

    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=VAD_MIN_SPEECH,
            min_silence_duration=VAD_MIN_SILENCE,
            activation_threshold=VAD_THRESHOLD,
        ),
        stt=deepgram.STT(model=STT_MODEL, language=STT_LANGUAGE, smart_format=True),
        llm=groq.LLM(model=LLM_MODEL),
        tts=elevenlabs.TTS(voice_id=TTS_VOICE_ID, model=TTS_MODEL),
        turn_detection="vad",
        min_interruption_duration=MIN_INTERRUPTION_DURATION,
        min_endpointing_delay=MIN_ENDPOINTING_DELAY,
        max_endpointing_delay=MAX_ENDPOINTING_DELAY,
        false_interruption_timeout=FALSE_INTERRUPTION_TIMEOUT,
        resume_false_interruption=True,
    )

    # ── Event handlers ─────────────────────────────────────

    @session.on("user_input_transcribed")
    def _on_user(ev):
        if not ev.is_final:
            return
        text = ev.transcript
        slog.info("[USER] %s", text)
        conv.transition(State.LISTENING)

        if conv.should_escalate(text):
            conv.transition(State.ESCALATED)
            conv.escalation_reason = "Customer requested human agent"
            slog.info("[ESCALATION] triggered by user request")

    @session.on("conversation_item_added")
    def _on_item(ev):
        role = ev.item.role
        text = ev.item.text_content or ""
        if not text:
            return
        speaker = "agent" if role == "assistant" else "user"
        slog.info("[%s] %s", speaker.upper(), text)
        conv.add_turn(speaker, text)
        transcript["utterances"].append({
            "speaker": speaker, "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        save_transcript()

    @session.on("agent_speech_stopped")
    def _on_speech_stop(ev):
        conv.transition(State.LISTENING)
        if ev.interrupted:
            slog.info("[AGENT] speech interrupted by user")

    @session.on("metrics_collected")
    def _on_metrics(ev):
        m = ev.metrics
        usage.collect(m)

        if isinstance(m, LLMMetrics):
            conv.transition(State.THINKING)
            turn_metrics["llm_ttft"] = m.ttft
            turn_metrics["llm_duration"] = m.duration
            ttft_ms = m.ttft * 1000
            tag = "OK" if ttft_ms < 500 else ("SLOW" if ttft_ms < 1000 else "CRITICAL")
            slog.info(
                "[LLM] ttft=%.0fms (%s) duration=%.0fms tokens=%d tps=%.1f",
                ttft_ms, tag, m.duration * 1000, m.total_tokens, m.tokens_per_second,
            )

        elif isinstance(m, TTSMetrics):
            conv.transition(State.SPEAKING)
            turn_metrics["tts_ttfb"] = m.ttfb
            turn_metrics["tts_duration"] = m.duration
            ttfb_ms = m.ttfb * 1000
            tag = "OK" if ttfb_ms < 300 else ("SLOW" if ttfb_ms < 500 else "CRITICAL")
            slog.info(
                "[TTS] ttfb=%.0fms (%s) duration=%.0fms chars=%d",
                ttfb_ms, tag, m.duration * 1000, m.characters_count,
            )
            # Per-turn latency summary
            turn_metrics["turn_count"] += 1
            eou = turn_metrics["eou_delay"] * 1000
            stt = turn_metrics["transcription_delay"] * 1000
            llm = turn_metrics["llm_ttft"] * 1000
            tts = ttfb_ms
            ttfa = eou + llm + tts
            tag2 = "OK" if ttfa < 1000 else ("SLOW" if ttfa < 2000 else "CRITICAL")
            slog.info("── TURN %d ──", turn_metrics["turn_count"])
            slog.info("  VAD: %.0fms | STT: %.0fms | LLM: %.0fms | TTS: %.0fms", eou, stt, llm, tts)
            slog.info("  TTFA: %.0fms (%s) | Total: %.0fms", ttfa, tag2,
                       eou + stt + turn_metrics["llm_duration"] * 1000 + turn_metrics["tts_duration"] * 1000)

        elif isinstance(m, EOUMetrics):
            turn_metrics["eou_delay"] = m.end_of_utterance_delay
            turn_metrics["transcription_delay"] = m.transcription_delay

        elif isinstance(m, STTMetrics):
            slog.info("[STT] audio=%.2fs streamed=%s", m.audio_duration, m.streamed)

        elif isinstance(m, VADMetrics):
            per_inf = (m.inference_duration_total / max(m.inference_count, 1)) * 1000
            slog.info("[VAD] idle=%.2fs per_inf=%.1fms count=%d", m.idle_time, per_inf, m.inference_count)

    @session.on("close")
    def _on_close(*args):
        conv.transition(State.ENDED)
        s = usage.get_summary()
        llm_cost = (s.llm_prompt_tokens / 1e6) * COST_RATES["llm_input_per_1m"] + \
                   (s.llm_completion_tokens / 1e6) * COST_RATES["llm_output_per_1m"]
        stt_cost = (s.stt_audio_duration / 60) * COST_RATES["stt_per_min"]
        tts_cost = s.tts_characters_count * COST_RATES["tts_per_char"]
        total = llm_cost + stt_cost + tts_cost

        slog.info("═══ SESSION SUMMARY ═══")
        slog.info("ID: %s | Turns: %d | State: %s", session_id, turn_metrics["turn_count"], conv.state.value)
        slog.info("Tokens: %d prompt + %d completion", s.llm_prompt_tokens, s.llm_completion_tokens)
        slog.info("Audio: STT %.2fs | TTS %.2fs (%d chars)", s.stt_audio_duration, s.tts_audio_duration, s.tts_characters_count)
        slog.info("Cost: LLM $%.6f | STT $%.6f | TTS $%.6f | TOTAL $%.6f", llm_cost, stt_cost, tts_cost, total)
        if conv.escalation_reason:
            slog.info("Escalated: %s", conv.escalation_reason)
        slog.info("═══════════════════════")

        save_transcript()
        slog.removeHandler(fh)
        fh.close()

    # ── Start session ──────────────────────────────────────
    await session.start(
        room=ctx.room,
        agent=Agent(instructions=SYSTEM_PROMPT, tools=ALL_TOOLS),
        room_input_options=RoomInputOptions(noise_cancellation=BVC()),
    )

    conv.transition(State.SPEAKING)
    await session.say(
        "Hey, thanks for calling NovaLaptops! How can I help you today?",
        allow_interruptions=True,
    )
    conv.transition(State.LISTENING)


def cli():
    from livekit.agents import cli as lk_cli, WorkerOptions
    lk_cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    cli()
