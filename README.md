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
- LiveKit Cloud account (or self-hosted LiveKit Server)
- OpenAI API Key

### 1. Install Dependencies

Install all project dependencies into the virtual environment using `uv`:

```bash
uv sync
```

### 2. Create and Configure `.env`

Copy the example environment file and populate your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:

```ini
# LiveKit Server Configuration (from LiveKit Cloud Dashboard -> Settings -> Keys)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=APIxxxxxxxxxxxx
LIVEKIT_API_SECRET=secxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OpenAI API Key (for Whisper STT, GPT-4o-mini LLM, and TTS Alloy)
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Agent Worker Configuration
AGENT_NAME=zcloud-ai-telecaller-agent
AGENT_LOG_LEVEL=INFO
```

### 3. Start the Agent Worker

Start the agent worker locally in development mode:

```bash
# Run in dev mode (hot reload enabled)
uv run python -m zcloud_ai_telecaller.agent dev
```

Or run via the installed project script:

```bash
uv run zcloud-ai-telecaller
```

Upon startup, the worker validates environment variables, pre-warms the Silero VAD model, connects to your LiveKit server, and logs:
```text
INFO | zcloud_ai_telecaller - Worker startup: Initializing Z Cloud AI Telecaller agent worker (agent=zcloud-ai-telecaller-agent)
INFO | zcloud_ai_telecaller - Environment configuration validated successfully
INFO | livekit.agents - registering worker...
INFO | livekit.agents - worker registered and listening for jobs
```

### 4. Connect a Browser Participant to LiveKit

Once the agent worker is registered and listening for rooms:

1. Open the [LiveKit Agents Playground](https://agents-playground.livekit.io/) in your browser (Google Chrome or Microsoft Edge recommended).
2. Click **Settings** (gear icon) or **Connect** in the playground.
3. Paste your:
   - **LiveKit URL**: `wss://your-project.livekit.cloud`
   - **Token**: Generate an access token via your LiveKit Cloud Dashboard (under **Sandbox** or **Token Generator**) or using the LiveKit CLI:
     ```bash
     # Generate token for room "telecaller-demo" with identity "browser-user"
     lk token create --api-key <LIVEKIT_API_KEY> --api-secret <LIVEKIT_API_SECRET> --join --room telecaller-demo --identity browser-user
     ```
   *(Alternatively, if using LiveKit Cloud, navigate to **Sandbox** -> **Agents** and connect directly from the dashboard).*
4. Allow browser microphone access when prompted.
5. Click **Connect**.

### 5. Test Voice Conversation

When the browser participant connects:
1. **Worker logs connection**:
   ```text
   INFO | zcloud_ai_telecaller - Connecting to LiveKit room: telecaller-demo
   INFO | zcloud_ai_telecaller - Participant joined: identity=browser-user in room: telecaller-demo
   INFO | zcloud_ai_telecaller - Voice session started for room: telecaller-demo
   ```
2. **Initial Greeting**: The agent speaks back over WebRTC:
   > *"Hello! I am your Z Cloud AI Voice Assistant. How can I help you today?"*
3. **Speak into microphone**: Say a test phrase, for example:
   > *"Hi, can you introduce yourself?"*
4. **Agent Response**: The agent will transcribe your speech via OpenAI Whisper, process it via GPT-4o-mini adhering to the Phase 1 instructions, synthesize speech with OpenAI TTS, and respond over audio:
   > *"I am Z Cloud AI Voice Assistant. I am friendly, concise and professional. This is only a Phase 1 technical demo."*

### Running Tests

Execute the automated test suite using `pytest`:

```bash
uv run pytest -v
```

All 8 tests verify configuration defaults, environment variable loading, strict validation of required keys, agent instructions, Silero VAD prewarming, and logging levels.

### Docker (Optional for Production)

Docker is **not** required for local development. For production deployments or containerized runs:

```bash
# Build the Docker container (pre-caches Silero VAD model assets)
docker build -t zcloud-ai-telecaller .

# Run container with environment configuration
docker run --rm --env-file .env zcloud-ai-telecaller
```

