# UdaPlay - AI-Powered Game Research Agent

An intelligent research agent for the video game industry built with RAG (Retrieval-Augmented Generation), semantic search, and agentic workflows. Features multi-source information retrieval, automatic quality assessment, and both standard and advanced implementations with long-term memory capabilities.

## 🎯 Overview

UdaPlay is a production-ready AI agent that intelligently answers questions about video games by combining:
- **Semantic search** over a local vector database of game information
- **Quality evaluation** of retrieved results using LLM-based assessment
- **Automatic fallback** to web search when local knowledge is insufficient
- **Stateful conversations** with session management
- **Explicit state machine architecture** for transparent, debuggable workflows

## ✨ Key Features

### Core Capabilities
- **Multi-Source RAG Pipeline**: Queries vector database, evaluates results, and falls back to web search automatically
- **Intelligent Retrieval Evaluation**: LLM-powered assessment determines if retrieved documents are sufficient
- **Stateful Agent Architecture**: Maintains conversation context across multiple queries per session
- **Tool Integration**: Modular design with pluggable tools for retrieval, evaluation, and web search
- **Structured Outputs**: Citations, reasoning transparency, and source attribution

### Advanced Implementation
- **Long-Term Memory**: Persistent storage of learned facts using vector embeddings
- **Explicit State Machine**: Pre-defined tool nodes for deterministic, observable workflows
- **Conditional Branching**: Dynamic path selection based on retrieval quality
- **Enhanced State Tracking**: Full visibility into intermediate results at each workflow step
- **Fact Extraction & Storage**: Automatic identification and persistence of important information

## 🏗️ Architecture

### Standard Agent Workflow
```
User Query → Retrieve from Vector DB → Evaluate Quality
                                            ↓
                          Sufficient? → Generate Answer
                                 ↓
                          Insufficient? → Web Search → Generate Answer
```

### Advanced Agent Workflow
```
User Query → Long-Term Memory Search → Vector DB Search → Quality Evaluation
                                                               ↓
                                          Sufficient? → Generate Answer → Extract & Store Facts
                                                 ↓
                                          Insufficient? → Web Search → Generate Answer → Extract & Store Facts
```

## 🛠️ Tech Stack

**Core Technologies:**
- **Python 3.11+** - Primary language
- **ChromaDB** - Vector database for semantic search
- **OpenAI API** - LLM for generation and embeddings (text-embedding-3-small)
- **Tavily API** - Web search integration
- **Pydantic** - Data validation and structured outputs

**Custom Frameworks:**
- **State Machine Engine** - Custom TypedDict-based workflow orchestration
- **Memory Management** - Short-term (session-based) and long-term (vector-based) memory
- **Tool System** - Decorator-based tool registration and execution
- **LLM Abstraction Layer** - Unified interface for OpenAI models

## 📁 Project Structure

```
gaming-research-agent-ai/
├── games/                                    # Game data corpus (JSON)
├── lib/                                      # Core library implementations
│   ├── agents.py                            # Agent orchestration & state machine
│   ├── llm.py                               # LLM interface abstractions
│   ├── memory.py                            # Short-term & long-term memory
│   ├── messages.py                          # Message type definitions
│   ├── state_machine.py                     # Workflow engine
│   ├── tooling.py                           # Tool decorator & execution
│   ├── vector_db.py                         # Vector store management
│   └── parsers.py                           # Output parsing utilities
├── chromadb/                                # Persistent vector database
├── Udaplay_01_solution_project.ipynb        # RAG pipeline implementation
├── Udaplay_02_solution_project.ipynb        # Standard agent implementation
├── Udaplay_03_solution_advanced_project.ipynb # Advanced agent with LTM
└── requirements.txt                         # Python dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd gaming-research-agent-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY="your_openai_api_key"
TAVILY_API_KEY="your_tavily_api_key"
```

### Usage

#### 1. Build Vector Database (Notebook 01)
```python
# Initialize ChromaDB and process game data
chroma_client = chromadb.PersistentClient(path="./chromadb")
collection = chroma_client.create_collection("udaplay_games")

# Load and index game documents
# Documents include: Name, Platform, Genre, Publisher, Description, Year
```

#### 2. Run Standard Agent (Notebook 02)
```python
from lib.agents import Agent

agent = Agent(
    model_name="gpt-4o-mini",
    instructions="...",
    tools=[retrieve_game, evaluate_retrieval, game_web_search]
)

run = agent.invoke("When was Pokémon Gold and Silver released?", session_id="demo")
```

#### 3. Run Advanced Agent (Notebook 03)
```python
from lib.memory import LongTermMemory

advanced_agent = AdvancedAgent(model_name="gpt-4o-mini")
run = advanced_agent.invoke(
    query="When was Super Mario 64 released?",
    session_id="session_1",
    owner="user_1"
)
```

## 🔍 Example Queries

The agent handles various query types:

**Factual Queries:**
- "When was Pokémon Gold and Silver released?" → Uses vector DB
- "What platform was Gran Turismo originally on?" → Uses vector DB

**Complex Queries:**
- "Which was the first 3D platformer Mario game?" → Semantic search + reasoning
- "Was Mortal Kombat X released for PlayStation 5?" → Multi-source verification

**Recent/Missing Data:**
- "When is GTA VI expected to release?" → Automatic web search fallback

## 📊 Implementation Highlights

### Three-Tool RAG Architecture
1. **`retrieve_game`**: Semantic search over game vector database
2. **`evaluate_retrieval`**: LLM-based quality assessment of retrieved documents
3. **`game_web_search`**: Tavily-powered web search as fallback mechanism

### State Machine Design
- **TypedDict-based state schemas** for type safety
- **Generic step functions** with automatic state merging
- **Conditional transitions** based on runtime evaluation
- **Complete execution history** via snapshots

### Memory Systems
- **Short-term**: Session-scoped conversation history
- **Long-term**: Persistent vector-based fact storage with semantic retrieval

## 🎓 Technical Achievements

- Built custom state machine engine with TypedDict support and generic types
- Implemented RAG evaluation pattern with LLM-as-judge architecture
- Designed modular tool system with decorator-based registration
- Created dual-memory architecture (ephemeral + persistent)
- Implemented explicit state machine with pre-defined tool nodes for production observability

## 📝 Documentation

Each notebook contains:
- Detailed implementation explanations
- Code documentation and comments
- Example outputs with reasoning traces
- Performance metrics (token usage)
- Comparison tables (Standard vs. Advanced)

## 🔐 Security Notes

- API keys managed via environment variables (`.env`)
- `.gitignore` configured to exclude sensitive files
- No hardcoded credentials in codebase

## 📄 License

This project is available for portfolio and educational purposes.

---

**Built with:** Python • ChromaDB • OpenAI • LangChain Patterns • RAG • Agentic AI
