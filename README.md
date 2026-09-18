# Z Cloud AI Telecaller

Z Cloud AI Telecaller is a real-time conversational AI telecalling platform built on top of [LiveKit Agents](https://docs.livekit.io/agents/).

## Phase 1 Architecture & Voice Pipeline

The foundation establishes a WebRTC audio streaming pipeline connecting browser clients with LiveKit and autonomous AI agent workers:

```
Browser / WebRTC  <--->  LiveKit Server  <--->  AI Agent Worker
                                                   |
                     +-----------------------------+-----------------------------+
                     |                             |                             |
                     v                             v                             v
           Silero VAD / Whisper STT           GPT-4o mini LLM               OpenAI TTS (Alloy)
          (Speech-to-Text & VAD)        (Language Intelligence)            (Text-to-Speech)
```

## Project Structure

```text
zcloud-ai-telecaller/
├── .env.example                 # Environment variable templates
├── .gitignore                   # Git ignore configurations (protects secrets, venv, caches)
├── .dockerignore                # Docker ignore rules
├── Dockerfile                   # Production-ready Docker container specification
├── pyproject.toml               # Project metadata and dependencies (managed via uv)
├── uv.lock                      # Deterministic dependency lockfile
├── README.md                    # Project documentation
├── src/
│   └── zcloud_ai_telecaller/
│       ├── __init__.py          # Package root exports
│       ├── agent.py             # LiveKit Agent entry point and voice pipeline configuration
│       ├── config.py            # Pydantic Settings configuration management
│       └── logging_config.py    # Structured logging configuration
└── tests/
    ├── __init__.py
    ├── test_agent.py            # Agent and AgentSession pipeline tests
    ├── test_config.py           # Configuration and environment tests
    └── test_logging.py          # Logging setup tests
```

## Getting Started

### Prerequisites

- Python `>= 3.12`
- [uv](https://docs.astral.sh/uv/) package manager

### Setup Environment

1. Clone the repository and install dependencies with `uv`:
   ```bash
   uv sync
   ```

2. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your LiveKit credentials and OpenAI API key:
   - `LIVEKIT_URL`
   - `LIVEKIT_API_KEY`
   - `LIVEKIT_API_SECRET`
   - `OPENAI_API_KEY`

### Running the Agent Worker

Run the agent worker in development mode:

```bash
# Connect and register as a LiveKit agent worker
uv run python -m zcloud_ai_telecaller.agent dev
```

Or run via the project script:

```bash
uv run zcloud-ai-telecaller
```

### Running Tests

Run the test suite with `pytest`:

```bash
uv run pytest
```

### Docker

Build and run using Docker:

```bash
docker build -t zcloud-ai-telecaller .
docker run --env-file .env zcloud-ai-telecaller
```
