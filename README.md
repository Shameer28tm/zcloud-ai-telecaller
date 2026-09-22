# Z Cloud AI Telecaller — Production Multi-Tenant AI Voice SaaS

Z Cloud AI Telecaller is a production-grade conversational AI voice telecalling SaaS built on [LiveKit Agents](https://docs.livekit.io/agents/), featuring dynamic multi-tenant agent management, multi-provider LLM abstraction (local Ollama, Google Gemini, and Hybrid fallback), deterministic clinic business tools, conversational intelligence, authoritative usage metering, and strict tenant isolation.

---

## 🏗️ High-Level System Architecture

```text
       ┌─────────────────────────────────────────────────────────┐
       │             PUBLIC WEB CLIENT / BROWSER                 │
       └──────────────────────────┬──────────────────────────────┘
                                  │ WebRTC Audio
                                  ▼
       ┌─────────────────────────────────────────────────────────┐
       │                      LIVEKIT SERVER                     │
       └──────────────────────────┬──────────────────────────────┘
                                  │ Room Dispatch
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          PYTHON VOICE WORKER                            │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      VOICE PIPELINE ENGINE                      │   │
│   │                                                                 │   │
│   │  Silero VAD  ──>  Faster-Whisper STT  ──>  Rule Engine & State  │   │
│   │                                                    │            │   │
│   │                                                    ▼            │   │
│   │  Local Piper TTS  <──  Streaming Thinking Filter  <──  LLM      │   │
│   └───────────────────────────────────┬─────────────────────────────┘   │
│                                       │                                 │
│                                       ▼                                 │
│                ┌───────────────────────────────────────┐                │
│                │        LLM PROVIDER ABSTRACTION       │                │
│                │                                       │                │
│                │  ├── LOCAL_AI:  Ollama (qwen3:4b)     │                │
│                │  ├── CLOUD_AI:  Google Gemini Flash   │                │
│                │  └── HYBRID:    Ollama ──> Gemini     │                │
│                └──────────────────────┬────────────────┘                │
│                                       │ Validated Tool Calls            │
│                                       ▼                                 │
│                ┌───────────────────────────────────────┐                │
│                │     CLINIC TOOLS & BUSINESS RULES     │                │
│                │                                       │                │
│                │  - Check Availability                 │                │
│                │  - Book Appointment (Atomic Lock)     │                │
│                │  - Cancel / Reschedule                │                │
│                │  - Emergency 911 Escalation           │                │
│                │  - Medical Advice Prohibition         │                │
│                └──────────────────────┬────────────────┘                │
│                                       │ M2M Authenticated JSON          │
└───────────────────────────────────────┼─────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   CONTROL PLANE / SAAS (NEXT.JS)                        │
│                                                                         │
│   GET  /api/internal/agent-runtime   ──> Dynamic agent config & prompt  │
│   POST /api/internal/clinic/*        ──> Atomic DB transactions         │
│   POST /api/internal/call-session    ──> Authoritative usage & metering │
│                                                                         │
│   Database: SQLite / PostgreSQL (Prisma ORM)                            │
│   Multi-Tenant Isolation: Tenant-scoped queries, 0 client-side trust    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🌟 Operating Modes (LOCAL_AI, CLOUD_AI, HYBRID)

The platform supports 3 primary operating modes configurable per tenant/agent or globally via environment variables:

| Mode | STT | LLM | TTS | API Cost | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`LOCAL_AI`** | `faster-whisper` (int8) | **Ollama** (`qwen3:4b`) | `Piper` (neural) | **₹0.00** | 100% local, self-hosted, CPU-compatible voice telecaller without any third-party API dependencies. |
| **`CLOUD_AI`** | `faster-whisper` | **Gemini 2.5 Flash** | `Piper` | Low Cloud | Cloud LLM inference via Google Gemini for complex reasoning and large context windows. |
| **`HYBRID`** | `faster-whisper` | **Ollama ➔ Gemini** | `Piper` | Minimal | Primary requests run locally via Ollama at ₹0; automatically falls back to Gemini if Ollama times out or is unreachable. |

### Thinking Tag Filter (`<think>...</think>`)
Reasoning models like Qwen 3 emit internal thought blocks wrapped in `<think>...</think>`. The worker implements an asynchronous streaming `OllamaThinkingFilter` that intercepts and discards thinking tokens in real-time, preventing internal reasoning from ever leaking to Piper TTS or call transcripts.

---

## 🩺 Phase 3 Clinic AI Capabilities

The clinic AI capability provides deterministic, safety-enforced receptionist services:
1. **Clinic Information**: Operating hours, address, services, and policies configured per tenant.
2. **Deterministic Availability**: Real-time lookup of available appointment slots directly from the database.
3. **Double-Booking Prevention**: Database transactions and concurrency locks ensure conflicting bookings are rejected (`409 Conflict`).
4. **Rescheduling & Cancellations**: Securely frees previously reserved time slots.
5. **Human Escalation**: Clean boundary for customer service or staff transfers.

---

## 🛡️ Phase 4 Conversational Intelligence & Clinical Safety

A deterministic 7-priority rule engine evaluates user turns before routing to the LLM:
1. **Emergency Detection (Priority 1)**: Severe medical symptoms (chest pain, shortness of breath, heavy bleeding) immediately trigger emergency escalation to call 911 or visit the nearest emergency room.
2. **Prompt Injection Defense (Priority 2)**: Detects and blocks jailbreak phrases and attempts to bypass system instructions.
3. **Medical Advice Prohibition (Priority 3)**: Strictly refuses requests for diagnosis, medication prescriptions, or treatment advice:
   > *"As an administrative voice assistant, I cannot diagnose medical conditions or prescribe medications. I would be glad to schedule an in-person consultation..."*
4. **Booking Confirmation Gate (Priority 4)**: Explicit user confirmation is required before booking; asking about availability does not trigger booking.
5. **Response Sanitization**: Trims long responses and redacts API keys, bearer tokens, or sensitive credentials.

---

## 📊 Authoritative Call Persistence & Metering (Phase 2B)

- Every call session records room metadata, start/end timestamps, duration, and full conversation transcripts.
- **Server-Authoritative Metering**: Voice minutes are calculated server-side as `ceil(durationSeconds / 60)`.
- **Strict Idempotency**: Duplicate session reports from network retries safely return existing records without double-charging usage.

---

## 🚀 Getting Started

### 1. Prerequisites

- Python `>= 3.12`
- Node.js `>= 18.0.0`
- [Ollama](https://ollama.com/) (for `LOCAL_AI` or `HYBRID` mode)
- LiveKit Cloud account or local LiveKit Server

### 2. Setup Ollama (Local AI)

1. Install Ollama from [ollama.com](https://ollama.com/).
2. Pull the recommended local model:
   ```bash
   ollama run qwen3:4b
   ```
3. Verify Ollama is running:
   ```bash
   curl http://127.0.0.1:11434/api/tags
   ```

### 3. Setup Next.js Control Plane

```bash
cd LIVEKIT-AGENT-
npm install
npx prisma generate
npx prisma db push
npm run build
npm run dev
```

The Control Plane dashboard will be available at `http://localhost:3000`.

### 4. Setup Python Voice Worker

```bash
cd ZCloud-AI-Telecaller
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -e .
```

Create a `.env` file in `ZCloud-AI-Telecaller`:

```ini
# LiveKit Server
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

# AI Mode & Provider Configuration
AI_MODE=LOCAL_AI
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
OLLAMA_TIMEOUT=60.0

# Cloud Fallback (required for CLOUD_AI or HYBRID)
GEMINI_API_KEY=your_gemini_api_key

# Control Plane M2M Integration
CONTROL_PLANE_URL=http://localhost:3000
WORKER_API_SECRET=your_shared_worker_secret

# Local STT & TTS (Zero Cost)
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
PIPER_VOICE=en_US-lessac-medium
```

Run the worker:
```bash
python -m zcloud_ai_telecaller.agent dev
```

---

## 🧪 Verification & Test Suite

The platform includes comprehensive test suites across both repositories:

### Python Worker Tests (76 Tests)
```bash
cd ZCloud-AI-Telecaller
pytest -v
```
- `test_agent.py`: Agent lifecycle, VAD prewarming, local STT & TTS.
- `test_config.py`: Environment validation, AI modes, zero-OpenAI guarantee.
- `test_llm_provider.py`: Ollama streaming, thinking filter, tool calling, hybrid fallback.
- `test_clinic_ai.py`: Appointment tools, date/time normalization, double-booking prevention.
- `test_phase4_intelligence.py`: Emergency rules, prompt injection defense, medical refusal.
- `test_runtime_client.py`: Control Plane integration, participant metadata extraction.
- `test_e2e_voice_saas.py`: End-to-end multi-tenant isolation, billing metering, provider switching.

### Control Plane Tests (45 Tests)
```bash
cd LIVEKIT-AGENT-
npm run test:isolation      # 6/6 Multi-tenant security & IDOR tests
npm run test:runtime        # 8/8 M2M agent runtime API tests
npm run test:call-session    # 8/8 Atomic call persistence & idempotency tests
npm run test:clinic-tools   # 14/14 Clinic availability & transactional booking tests
npm run test:phase4         # 9/9 High-concurrency race condition & rule engine tests
```

**Total Automated Tests: 121 / 121 Passing (100% Success)**

---

## 🔒 Multi-Tenant Security Model

- **Zero Browser Trust**: All tenant IDs, agent IDs, and ownership checks are validated server-side.
- **M2M Secret Auth**: Internal APIs require `Authorization: Bearer <WORKER_API_SECRET>`.
- **Zero Exposed Secrets**: Control Plane runtime endpoints strip all API keys, database credentials, and secrets before responding to workers.
- **Cross-Tenant Prevention**: Direct IDOR attacks return `404 Not Found` without revealing resource existence.
