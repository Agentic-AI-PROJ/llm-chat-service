# LLM Chat Service

The **LLM Chat Service** is a FastAPI-based microservice that handles direct interactions with Large Language Models (LLMs). It abstracts model providers (Gemini, OpenAI, etc.), handles token counting, rate limiting, and provides both streaming and non-streaming chat interfaces.

## 🚀 Features

- **Unified Chat Interface**: Standardized API for interacting with different LLM providers through `litellm`.
- **Streaming Support**: Real-time token streaming using Server-Sent Events (SSE).
- **Model Management**: Dynamic configuration of AI models (API keys, costs, limits) via API.
- **Observability**: Comprehensive request logging, token usage tracking, and cost calculation.
- **Failover Logic**: Automatic model fallback strategies for reliability.

## 🛠️ Technology Stack

- **Runtime**: Python 3.10+
- **Framework**: FastAPI, Uvicorn
- **LLM Abstraction**: LiteLLM
- **Database**: MongoDB (Motor async driver)

## 📦 Installation & Setup

1.  **Create Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables**:
    Create a `.env` file in the root of the service:
    ```env
    PORT=3005
    MONGO_URI=mongodb://localhost:27017/ai-agents
    ```

4.  **Run Development Server**:
    ```bash
    python src/main.py
    # or
    uvicorn src.main:app --reload --port 3005
    ```
    The server will start at `http://localhost:3005`.

## 🔌 API Reference & Curl Examples

### 1. Chat Completion (Stream)

Stream a chat response token-by-token.

- **Endpoint**: `POST /stream`
- **Body**: `{ "model_id": "optional_id", "messages": [...] }`

**Curl Example:**

```bash
curl -N -X POST http://localhost:3005/stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain quantum computing briefly."}]
  }'
```

### 2. Chat Completion (Non-Stream)

Get a complete chat response in a single request.

- **Endpoint**: `POST /non-stream`

**Curl Example:**

```bash
curl -X POST http://localhost:3005/non-stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello world"}]
  }'
```

### 3. Manage Models

#### List Active Models
```bash
curl http://localhost:3005/models/active
```

#### Create New Model Config
```bash
curl -X POST http://localhost:3005/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Gemini Pro",
    "model_id": "gemini/gemini-pro",
    "provider": "google",
    "isActive": true,
    "api_key": "your_api_key"
  }'
```

### 4. View Logs & Stats

#### Get Request Logs
```bash
curl "http://localhost:3005/logs?limit=5&success=true"
```

#### Get Usage Stats
```bash
curl "http://localhost:3005/logs/stats?sort_by=total_tokens&sort_order=desc"
```

## 💓 Health Checks

- **Server Health**: `GET /health`
- **Database Health**: `GET /db-health`
