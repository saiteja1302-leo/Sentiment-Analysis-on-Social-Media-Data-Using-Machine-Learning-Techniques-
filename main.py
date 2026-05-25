"""
Advanced Data Scientist System with Llama3 and CAMEL AI
Multi-Agent Architecture using LangGraph, Groq, and CAMEL-inspired Role Playing
"""

import os
import sys
import json
import re
import warnings
from typing import TypedDict, Annotated, Sequence, Any, Optional
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Dependencies ────────────────────────────────────────────────────────────
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_community.tools import DuckDuckGoSearchRun
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    import duckdb
    import pandas as pd
    import numpy as np
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich import print as rprint
    from dotenv import load_dotenv
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

load_dotenv()
console = Console()

# ─── Configuration ────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama3-70b-8192")
DB_PATH      = os.getenv("DB_PATH", ":memory:")          # or path to .duckdb file
MAX_RETRIES  = 3
TEMPERATURE  = 0.1

if not GROQ_API_KEY:
    console.print("[bold red]❌  GROQ_API_KEY not set. Add it to your .env file.[/bold red]")
    sys.exit(1)

# ─── Shared LLM ───────────────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=MODEL_NAME,
    temperature=TEMPERATURE,
    max_retries=MAX_RETRIES,
)

# ─── Database Manager ─────────────────────────────────────────────────────────
class DatabaseManager:
    """DuckDB-backed database manager with schema introspection."""

    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.conn = duckdb.connect(path)
        self._seed_sample_data()

    def _seed_sample_data(self):
        """Load sample datasets so the system works out of the box."""
        # Sales data
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                id          INTEGER PRIMARY KEY,
                date        DATE,
                product     VARCHAR,
                category    VARCHAR,
                quantity    INTEGER,
                unit_price  DECIMAL(10,2),
                revenue     DECIMAL(10,2),
                region      VARCHAR,
                salesperson VARCHAR
            )
        """)
        if self.conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 0:
            np.random.seed(42)
            n = 500
            products   = ["Laptop","Phone","Tablet","Monitor","Keyboard","Mouse","Headphones","Webcam"]
            categories = {"Laptop":"Electronics","Phone":"Electronics","Tablet":"Electronics",
                          "Monitor":"Peripherals","Keyboard":"Peripherals","Mouse":"Peripherals",
                          "Headphones":"Audio","Webcam":"Peripherals"}
            regions    = ["North","South","East","West","Central"]
            salespeople= ["Alice","Bob","Charlie","Diana","Eve","Frank"]
            dates = pd.date_range("2023-01-01","2023-12-31", periods=n)
            rows  = []
            for i, d in enumerate(dates):
                prod  = np.random.choice(products)
                qty   = np.random.randint(1,20)
                price = round(np.random.uniform(10,1500),2)
                rows.append((i+1, d.date(), prod, categories[prod], qty, price,
                              round(qty*price,2), np.random.choice(regions),
                              np.random.choice(salespeople)))
            self.conn.executemany(
                "INSERT INTO sales VALUES (?,?,?,?,?,?,?,?,?)", rows)

        # Customer data
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id  INTEGER PRIMARY KEY,
                name         VARCHAR,
                age          INTEGER,
                email        VARCHAR,
                country      VARCHAR,
                signup_date  DATE,
                total_spent  DECIMAL(12,2),
                segment      VARCHAR
            )
        """)
        if self.conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 0:
            np.random.seed(99)
            n = 200
            countries = ["USA","UK","Canada","Germany","France","Japan","Australia","India"]
            segments  = ["Premium","Standard","Basic","Enterprise"]
            rows = []
            for i in range(n):
                spent = round(np.random.exponential(500),2)
                seg   = "Enterprise" if spent>2000 else ("Premium" if spent>1000
                        else ("Standard" if spent>300 else "Basic"))
                rows.append((i+1, f"Customer_{i+1}", np.random.randint(18,70),
                              f"customer{i+1}@example.com", np.random.choice(countries),
                              pd.Timestamp("2020-01-01")+pd.Timedelta(days=int(np.random.uniform(0,1460))),
                              spent, seg))
            self.conn.executemany(
                "INSERT INTO customers VALUES (?,?,?,?,?,?,?,?)", rows)

    def get_schema(self) -> str:
        tables = self.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
        schema_parts = []
        for (tname,) in tables:
            cols = self.conn.execute(
                f"SELECT column_name, data_type FROM information_schema.columns "
                f"WHERE table_name='{tname}'").fetchall()
            col_defs = ", ".join(f"{c} ({t})" for c, t in cols)
            cnt = self.conn.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
            schema_parts.append(f"Table: {tname} ({cnt} rows)\n  Columns: {col_defs}")
        return "\n\n".join(schema_parts)

    def execute_query(self, sql: str) -> tuple[pd.DataFrame | None, str | None]:
        try:
            sql = sql.strip().rstrip(";")
            result = self.conn.execute(sql).df()
            return result, None
        except Exception as e:
            return None, str(e)

    def load_csv(self, path: str, table_name: str) -> str:
        try:
            df = pd.read_csv(path)
            self.conn.register(table_name, df)
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM {table_name}")
            return f"✅  Loaded '{path}' as table '{table_name}' ({len(df)} rows)"
        except Exception as e:
            return f"❌  Failed: {e}"

db = DatabaseManager(DB_PATH)

# ─── DuckDuckGo Search Tool ───────────────────────────────────────────────────
search_tool = DuckDuckGoSearchRun()

def web_search(query: str) -> str:
    try:
        return search_tool.run(query)
    except Exception as e:
        return f"Search failed: {e}"

# ─── Agent State ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages:          Annotated[Sequence[BaseMessage], add_messages]
    user_query:        str
    task_type:         str          # sql | analysis | visualization | search | general
    sql_query:         str
    query_result:      Optional[Any]
    analysis_result:   str
    search_result:     str
    final_response:    str
    error:             str
    iteration:         int
    schema:            str

# ─── CAMEL-Inspired Role Prompts ──────────────────────────────────────────────
ROLE_PROMPTS = {
    "orchestrator": """You are the Orchestrator Agent — the master coordinator of a multi-agent data science system.
Your role is to analyse the user's query and route it to the right specialist agent.

Available agents:
  • sql_agent        – Translates natural language into SQL and queries the database
  • analysis_agent   – Performs statistical analysis, finds patterns, generates insights
  • visualization_agent – Suggests charts, describes plots, creates Matplotlib code
  • search_agent     – Searches the web for context, documentation, or external data
  • response_agent   – Synthesises all results into a clear, actionable final answer

Database schema:
{schema}

Respond ONLY with valid JSON:
{{
  "task_type": "<sql|analysis|visualization|search|general>",
  "reasoning": "<one sentence why>",
  "sub_tasks": ["<task1>", "<task2>"]
}}""",

    "sql_agent": """You are the SQL Agent — a precise SQL expert for DuckDB.
Your role (inspired by CAMEL AI role-play) is to act as a SQL assistant helping a data analyst.

Database schema:
{schema}

Rules:
- Generate ONLY valid DuckDB SQL
- Use standard SQL functions (DATE_TRUNC, STRFTIME, etc.)
- Return ONLY the SQL query, no markdown fences
- Limit results to 50 rows unless asked otherwise
- Prefer readable column aliases

User request: {query}
SQL:""",

    "analyst": """You are the Data Analyst Agent — a senior data scientist with expertise in statistical analysis.
You receive query results and produce insightful analysis.

Role (CAMEL-inspired): You play the role of a senior data scientist reviewing results for a business stakeholder.

Data summary:
{data_summary}

User question: {query}

Provide:
1. Key findings (bullet points)
2. Statistical insights (mean, trends, anomalies)
3. Business implications
4. Recommended next steps

Be specific, concise, and data-driven.""",

    "visualization": """You are the Visualization Agent — a data visualisation specialist.

Data summary:
{data_summary}

User request: {query}

Provide:
1. Recommended chart type(s) and why
2. Complete Matplotlib/Seaborn Python code to generate the chart
3. Key design choices

Return the Python code inside a ```python``` block.""",

    "search": """You are the Research Agent. Using the following web search results, answer the user's question.

Search results:
{search_results}

User question: {query}

Provide a concise, accurate answer citing the sources.""",

    "response": """You are the Response Synthesis Agent — you craft the final answer for the user.

You have access to:
- Original question: {query}
- Task type: {task_type}
- SQL query used: {sql_query}
- Data result summary: {data_summary}
- Analysis: {analysis}
- Search context: {search_context}

Compose a clear, comprehensive, well-structured answer in Markdown.
Include tables, bullet points, and code blocks where appropriate.
Be friendly, professional, and insightful.""",
}

# ─── Helper: extract SQL from LLM output ──────────────────────────────────────
def _clean_sql(text: str) -> str:
    # strip markdown fences
    text = re.sub(r"```sql\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    # keep only the first statement
    stmts = [s.strip() for s in text.strip().split(";") if s.strip()]
    return stmts[0] if stmts else text.strip()

# ─── Node: Orchestrator ───────────────────────────────────────────────────────
def orchestrator_node(state: AgentState) -> AgentState:
    console.print("\n[bold cyan]🎯  Orchestrator[/bold cyan] routing query…")
    prompt = ROLE_PROMPTS["orchestrator"].format(schema=state["schema"])
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=state["user_query"]),
    ])
    try:
        text = response.content.strip()
        # extract JSON even if wrapped in markdown
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group()) if m else {}
        task_type = data.get("task_type", "general")
        console.print(f"   → Task type: [bold yellow]{task_type}[/bold yellow]  |  {data.get('reasoning','')}")
    except Exception:
        task_type = "sql"
    return {**state, "task_type": task_type, "iteration": state.get("iteration", 0) + 1}

# ─── Node: SQL Agent ──────────────────────────────────────────────────────────
def sql_agent_node(state: AgentState) -> AgentState:
    console.print("\n[bold green]🗄️   SQL Agent[/bold green] generating query…")
    prompt = ROLE_PROMPTS["sql_agent"].format(schema=state["schema"], query=state["user_query"])
    response = llm.invoke([SystemMessage(content=prompt)])
    sql = _clean_sql(response.content)
    console.print(f"   [dim]SQL:[/dim] {sql[:120]}{'…' if len(sql)>120 else ''}")

    df, error = db.execute_query(sql)
    if error:
        console.print(f"   [red]Query error:[/red] {error}")
        # retry with error context
        retry_prompt = f"Fix this SQL error:\nSQL: {sql}\nError: {error}\nReturn ONLY the corrected SQL."
        response2 = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=retry_prompt)])
        sql = _clean_sql(response2.content)
        df, error2 = db.execute_query(sql)
        if error2:
            return {**state, "error": error2, "sql_query": sql}

    return {**state, "sql_query": sql, "query_result": df, "error": ""}

# ─── Node: Analysis Agent ─────────────────────────────────────────────────────
def analysis_agent_node(state: AgentState) -> AgentState:
    console.print("\n[bold magenta]📊  Analysis Agent[/bold magenta] computing insights…")
    df = state.get("query_result")
    if df is not None and not df.empty:
        summary = f"Shape: {df.shape}\nColumns: {list(df.columns)}\nSample:\n{df.head(5).to_string()}\n\nStats:\n{df.describe().to_string()}"
    else:
        summary = "No data available."

    prompt = ROLE_PROMPTS["analyst"].format(data_summary=summary, query=state["user_query"])
    response = llm.invoke([SystemMessage(content=prompt)])
    return {**state, "analysis_result": response.content}

# ─── Node: Visualization Agent ────────────────────────────────────────────────
def visualization_agent_node(state: AgentState) -> AgentState:
    console.print("\n[bold blue]📈  Visualization Agent[/bold blue] designing charts…")
    df = state.get("query_result")
    summary = df.head(5).to_string() if df is not None and not df.empty else "No data."
    prompt = ROLE_PROMPTS["visualization"].format(data_summary=summary, query=state["user_query"])
    response = llm.invoke([SystemMessage(content=prompt)])
    return {**state, "analysis_result": response.content}

# ─── Node: Search Agent ───────────────────────────────────────────────────────
def search_agent_node(state: AgentState) -> AgentState:
    console.print("\n[bold yellow]🔍  Search Agent[/bold yellow] searching the web…")
    results = web_search(state["user_query"])
    prompt  = ROLE_PROMPTS["search"].format(search_results=results, query=state["user_query"])
    response = llm.invoke([SystemMessage(content=prompt)])
    return {**state, "search_result": response.content}

# ─── Node: Response Synthesis Agent ──────────────────────────────────────────
def response_agent_node(state: AgentState) -> AgentState:
    console.print("\n[bold white]✨  Response Agent[/bold white] synthesising answer…")
    df = state.get("query_result")
    data_summary = ""
    if df is not None and not df.empty:
        data_summary = f"Rows returned: {len(df)}\nColumns: {list(df.columns)}\n\n{df.head(20).to_markdown(index=False)}"

    prompt = ROLE_PROMPTS["response"].format(
        query        = state["user_query"],
        task_type    = state.get("task_type",""),
        sql_query    = state.get("sql_query",""),
        data_summary = data_summary or "No tabular data.",
        analysis     = state.get("analysis_result",""),
        search_context=state.get("search_result",""),
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    return {**state, "final_response": response.content}

# ─── Routing Logic ────────────────────────────────────────────────────────────
def route_task(state: AgentState) -> str:
    t = state.get("task_type", "general")
    if t == "sql":          return "sql_agent"
    if t == "analysis":     return "analysis_agent"
    if t == "visualization":return "visualization_agent"
    if t == "search":       return "search_agent"
    return "response_agent"

def after_sql(state: AgentState) -> str:
    if state.get("error"):  return "response_agent"
    return "analysis_agent"

# ─── Build LangGraph ──────────────────────────────────────────────────────────
def build_graph() -> Any:
    g = StateGraph(AgentState)
    g.add_node("orchestrator",        orchestrator_node)
    g.add_node("sql_agent",           sql_agent_node)
    g.add_node("analysis_agent",      analysis_agent_node)
    g.add_node("visualization_agent", visualization_agent_node)
    g.add_node("search_agent",        search_agent_node)
    g.add_node("response_agent",      response_agent_node)

    g.set_entry_point("orchestrator")
    g.add_conditional_edges("orchestrator", route_task, {
        "sql_agent":           "sql_agent",
        "analysis_agent":      "analysis_agent",
        "visualization_agent": "visualization_agent",
        "search_agent":        "search_agent",
        "response_agent":      "response_agent",
    })
    g.add_conditional_edges("sql_agent", after_sql, {
        "analysis_agent": "analysis_agent",
        "response_agent": "response_agent",
    })
    g.add_edge("analysis_agent",      "response_agent")
    g.add_edge("visualization_agent", "response_agent")
    g.add_edge("search_agent",        "response_agent")
    g.add_edge("response_agent",      END)

    return g.compile()

graph = build_graph()

# ─── Display Helpers ──────────────────────────────────────────────────────────
def display_dataframe(df: pd.DataFrame, title: str = "Query Results"):
    if df is None or df.empty:
        console.print("[dim]No rows returned.[/dim]")
        return
    table = Table(title=title, show_header=True, header_style="bold cyan",
                  border_style="dim", show_lines=True)
    for col in df.columns:
        table.add_column(str(col), overflow="fold", max_width=25)
    for _, row in df.head(20).iterrows():
        table.add_row(*[str(v) for v in row])
    if len(df) > 20:
        table.add_row(*["…"]*len(df.columns))
        table.caption = f"Showing 20 of {len(df)} rows"
    console.print(table)

def display_welcome():
    console.print(Panel.fit(
        "[bold cyan]Advanced Data Scientist System[/bold cyan]\n"
        "[dim]LangGraph · Llama 3 (Groq) · CAMEL AI Role Playing · DuckDB · DuckDuckGo[/dim]\n\n"
        "Type your question in plain English. Examples:\n"
        "  • [yellow]Show total revenue by product[/yellow]\n"
        "  • [yellow]Who are the top 5 salespeople?[/yellow]\n"
        "  • [yellow]Analyse customer segmentation trends[/yellow]\n"
        "  • [yellow]Visualise monthly sales over time[/yellow]\n"
        "  • [yellow]Search for best practices in churn analysis[/yellow]\n\n"
        "Commands: [bold]schema[/bold] | [bold]tables[/bold] | [bold]load <csv> <table>[/bold] | [bold]exit[/bold]",
        title="🤖 Multi-Agent Data Scientist",
        border_style="cyan",
    ))

# ─── Main REPL ────────────────────────────────────────────────────────────────
def run():
    display_welcome()
    console.print(f"\n[dim]Model: {MODEL_NAME}  |  DB: {DB_PATH}  |  {datetime.now():%Y-%m-%d %H:%M}[/dim]\n")

    while True:
        try:
            query = Prompt.ask("\n[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold]Goodbye! 👋[/bold]")
            break

        if not query:
            continue

        # ── Built-in commands ──
        if query.lower() in ("exit", "quit", "q"):
            console.print("[bold]Goodbye! 👋[/bold]")
            break

        if query.lower() in ("schema", "tables"):
            console.print(Panel(db.get_schema(), title="Database Schema", border_style="blue"))
            continue

        if query.lower().startswith("load "):
            parts = query.split()
            if len(parts) >= 3:
                console.print(db.load_csv(parts[1], parts[2]))
            else:
                console.print("[red]Usage: load <csv_path> <table_name>[/red]")
            continue

        # ── Run multi-agent graph ──
        initial_state: AgentState = {
            "messages":         [HumanMessage(content=query)],
            "user_query":       query,
            "task_type":        "",
            "sql_query":        "",
            "query_result":     None,
            "analysis_result":  "",
            "search_result":    "",
            "final_response":   "",
            "error":            "",
            "iteration":        0,
            "schema":           db.get_schema(),
        }

        with console.status("[bold cyan]Agents thinking…[/bold cyan]", spinner="dots"):
            try:
                final_state = graph.invoke(initial_state)
            except Exception as e:
                console.print(f"[bold red]System error:[/bold red] {e}")
                continue

        # ── Display results ──
        console.print()
        if final_state.get("sql_query"):
            console.print(Panel(
                f"```sql\n{final_state['sql_query']}\n```",
                title="🗄️  Generated SQL", border_style="green", expand=False))

        if final_state.get("query_result") is not None:
            display_dataframe(final_state["query_result"])

        if final_state.get("final_response"):
            console.print(Panel(
                Markdown(final_state["final_response"]),
                title="🤖 Agent Response", border_style="cyan"))

        if final_state.get("error"):
            console.print(f"[red]⚠️  Error:[/red] {final_state['error']}")


if __name__ == "__main__":
    run()
