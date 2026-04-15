# Multilingual Tirumala AI Assistant

A multi-agent AI assistant designed to help pilgrims with queries related to Tirumala and TTD services. 
The system supports multiple languages including **English, Telugu, Hindi, Tamil, and Kannada** via text and voice interfaces.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)
- [Ollama](https://ollama.com/) (running locally for LLM support)

### Installation
```bash
# Clone the repository
git clone https://github.com/Mahesh1827/Multilingual_tirumala.git
cd Multilingual_tirumala

# Install dependencies using uv
uv sync
```

### Usage
Run the main interactive CLI:
```bash
uv run python query_cli.py
```

### Features
- **Multilingual Support**: Auto-detection of native scripts and romanized Indic text.
- **Voice Interface**: Speak your queries directly in your native language.
- **RAG Pipeline**: Accurate answers grounded in Tirumala-specific knowledge bases.
- **CI/CD Integrated**: Automated initialization health checks on every push.

## 🛠 CI/CD
The project uses GitHub Actions to verify the pipeline health non-interactively:
```bash
uv run python query_cli.py --check
```
