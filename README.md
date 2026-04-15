# Multilingual Tirumala AI Assistant

[![CI-CD Pipeline](https://github.com/Mahesh1827/Multilingual/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Mahesh1827/Multilingual/actions/workflows/ci-cd.yml)
[![Docker Hub](https://img.shields.io/docker/v/maheshmeesala/tirumala-ai?label=Docker%20Hub&logo=docker)](https://hub.docker.com/r/maheshmeesala/tirumala-ai)

A multi-agent AI assistant designed to help pilgrims with queries related to Tirumala and TTD services.
The system supports multiple languages including **English, Telugu, Hindi, Tamil, and Kannada** via text and voice interfaces.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) (running locally for LLM support)
- Docker (optional, for containerised run)

### Installation
```bash
# Clone the repository
git clone https://github.com/Mahesh1827/Multilingual.git
cd Multilingual

# Install dependencies
pip install -r requirements.txt
```

### Run the API server
```bash
flask --app server run --port 8000
# OR with gunicorn (production)
gunicorn -w 1 -b 0.0.0.0:8000 server:app
```

### Run with Docker
```bash
docker pull maheshmeesala/tirumala-ai:latest
docker run -p 8000:8000 --env-file .env maheshmeesala/tirumala-ai:latest
```

---

## 🛠 CI/CD Pipeline

The project uses **GitHub Actions** with a 2-stage automated pipeline:

| Stage | Trigger | What it does |
|-------|---------|-------------|
| **1. Lint & Test** | Every push / PR | Runs flake8 linting + pytest with coverage |
| **2. Docker Build & Push** | Push to `main` only | Builds Docker image → pushes to Docker Hub |
| **3. Deploy** _(disabled)_ | Push to `main` | SSH deploy to production server _(enable when server is ready)_ |

### Enable Production Deploy
When you have a production server, add these secrets to **GitHub → Settings → Secrets → Actions**:

| Secret | Value |
|--------|-------|
| `SSH_HOST` | Your server IP / hostname |
| `SSH_USER` | SSH username (e.g. `ubuntu`) |
| `SSH_PRIVATE_KEY` | Your private SSH key |
| `SSH_PORT` | SSH port (default: `22`) |

Then uncomment the `deploy` job in `.github/workflows/ci-cd.yml`.

---

## ✨ Features
- **Multilingual Support**: Auto-detection of native scripts (Telugu, Hindi, Tamil, Kannada, English)
- **Voice Interface**: Speak queries directly in your native language
- **RAG Pipeline**: Accurate answers grounded in Tirumala-specific knowledge bases
- **Multi-Agent**: LangGraph-based agent routing for FAQ, RAG, Web Search, and Scripture queries
- **Dockerized**: Production-ready multi-stage Docker image with Flask + Gunicorn
- **CI/CD Integrated**: Automated lint, test, and Docker publish on every push
