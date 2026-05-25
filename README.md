# 🤖 Advanced Data Scientist System
### Multi-Agent AI · LangGraph · Llama 3 (Groq) · CAMEL AI · DuckDB · DuckDuckGo

A production-ready **multi-agent data science assistant** that lets anyone query databases, generate insights, and explore data using plain English — no SQL or Python knowledge required.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Natural Language to SQL** | Ask questions in plain English; the SQL Agent generates and executes DuckDB queries |
| **CAMEL AI Role Playing** | Each agent adopts a specialised expert persona for more reliable, focused outputs |
| **LangGraph Orchestration** | Deterministic agent routing with conditional edges and retry logic |
| **Llama 3 via Groq** | Ultra-fast inference on `llama3-70b-8192` (free tier available) |
| **Statistical Analysis** | Automated insights: trends, anomalies, distributions, correlations |
| **Visualisation Code** | Matplotlib/Seaborn chart code generated on demand |
| **Web Search** | DuckDuckGo integration for external context and research |
| **DuckDB Backend** | In-memory or file-based SQL database; CSV import supported |
| **Rich CLI** | Beautiful terminal UI with tables, syntax-highlighted SQL, and Markdown |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Orchestrator   │  ← Routes query to the right agent
│     Agent       │    (LLM-based task classification)
└────────┬────────┘
         │
    ┌────┴────────────────────────────┐
    │                                 │
    ▼                                 ▼
┌──────────┐  ┌────────────┐  ┌─────────────┐  ┌──────────────┐
│SQL Agent │  │ Analysis   │  │Visualization│  │ Search Agent │
│(DuckDB)  │  │  Agent     │  │   Agent     │  │(DuckDuckGo)  │
└────┬─────┘  └─────┬──────┘  └──────┬──────┘  └──────┬───────┘
     │              │                │                  │
     └──────────────┴────────────────┴──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Response Agent   │  ← Synthesises all outputs
                    └──────────────────┘
```

### CAMEL AI Role-Playing Principle
Each agent is given an **explicit role persona** (inspired by the CAMEL AI paper) that dramatically improves reliability with open-source models:

- 🎯 **Orchestrator** — "Master coordinator" that classifies tasks and routes them
- 🗄️ **SQL Agent** — "Precise SQL expert for DuckDB" that generates only valid SQL
- 📊 **Analysis Agent** — "Senior data scientist" that interprets results statistically
- 📈 **Visualisation Agent** — "Data visualisation specialist" that designs charts
- 🔍 **Search Agent** — "Research agent" that summarises web findings
- ✨ **Response Agent** — "Synthesis expert" that composes the final answer

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/data-scientist-system.git
cd data-scientist-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```
Get a **free** Groq API key at [console.groq.com](https://console.groq.com)

### 4. Run
```bash
python main.py
```

---

## 💬 Example Queries

```
You> Show total revenue by product category

You> Who are the top 5 salespeople by revenue this year?

You> Analyse customer segmentation and spending patterns

You> Visualise monthly sales trend with a line chart

You> What is the average order value by region?

You> Search for best practices in customer churn prediction

You> Show customers who spent more than $1000
```

### Built-in Commands

| Command | Description |
|---|---|
| `schema` | Display all tables and their columns |
| `tables` | Same as `schema` |
| `load <path.csv> <table_name>` | Import a CSV file as a new table |
| `exit` / `quit` | Exit the application |

---

## 📁 Project Structure

```
data-scientist-system/
├── main.py              # Complete system (single-file architecture)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

---

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `MODEL_NAME` | `llama3-70b-8192` | Groq model to use |
| `DB_PATH` | `:memory:` | DuckDB path (`:memory:` or file path) |

**Available Groq models:**
- `llama3-70b-8192` — Best quality (recommended)
- `llama3-8b-8192` — Faster, lower latency
- `mixtral-8x7b-32768` — Longer context window

---

## 🧪 Sample Data

The system auto-generates two sample tables on startup:

**`sales`** (500 rows) — Product sales with date, product, category, quantity, revenue, region, salesperson

**`customers`** (200 rows) — Customer profiles with age, country, total spent, segment

Load your own data:
```
You> load /path/to/my_data.csv my_table
You> Show the first 10 rows of my_table
```

---

## 🔬 How CAMEL AI Improves Reliability

Open-source models (like Llama 3) can be less reliable at complex agentic tasks than GPT-4. The **CAMEL AI role-playing technique** addresses this by:

1. **Role assignment** — Each agent is told exactly WHO they are (e.g., "You are the SQL Agent — a precise SQL expert")
2. **Constrained output** — Each role has explicit output format rules (e.g., "Return ONLY the SQL query")
3. **Task isolation** — No agent tries to do everything; each has one clear job
4. **Error recovery** — SQL Agent retries with error context on failures

This mirrors the findings of the [CAMEL paper](https://arxiv.org/abs/2303.17760) where role-playing significantly improves LLM task performance.

---

## 📊 Agent Flow (LangGraph)

```
orchestrator
     │
     ├─[sql]──────────► sql_agent ──► analysis_agent ──► response_agent
     │
     ├─[analysis]─────► analysis_agent ──────────────────► response_agent
     │
     ├─[visualization]► visualization_agent ───────────── ► response_agent
     │
     ├─[search]───────► search_agent ─────────────────── ► response_agent
     │
     └─[general]──────────────────────────────────────── ► response_agent
```

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [CAMEL AI](https://arxiv.org/abs/2303.17760) — Role-playing for LLMs
- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent orchestration
- [Groq](https://groq.com) — Ultra-fast Llama 3 inference
- [DuckDB](https://duckdb.org) — Embedded analytical database
