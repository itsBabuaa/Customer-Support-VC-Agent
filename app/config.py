"""Centralized configuration."""


from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).parent.parent

load_dotenv(dotenv_path=ROOT_DIR / ".env")


LOGS_DIR = ROOT_DIR / "logs"

TRANSCRIPTS_DIR = ROOT_DIR / "transcripts"

RAG_SOURCE = ROOT_DIR / "knowledge_base.txt"


LOGS_DIR.mkdir(exist_ok=True)

TRANSCRIPTS_DIR.mkdir(exist_ok=True)


# LLM

LLM_MODEL = "llama-3.1-8b-instant"


# STT

STT_MODEL = "nova-3"

STT_LANGUAGE = "multi"


# TTS

TTS_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"

TTS_MODEL = "eleven_turbo_v2_5"


# VAD

VAD_MIN_SPEECH = 0.3

VAD_MIN_SILENCE = 0.5

VAD_THRESHOLD = 0.6


# Turn detection

MIN_INTERRUPTION_DURATION = 1.0

MIN_ENDPOINTING_DELAY = 0.8

MAX_ENDPOINTING_DELAY = 5.0

FALSE_INTERRUPTION_TIMEOUT = 3.0


# Cost rates (USD)

COST_RATES = {

    "llm_input_per_1m": 0.05,

    "llm_output_per_1m": 0.08,

    "stt_per_min": 0.0043,

    "tts_per_char": 0.00003,

}

