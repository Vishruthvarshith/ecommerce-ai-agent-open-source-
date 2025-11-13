"""
E-Commerce AI Agent - Core Agent Implementation
Uses LangChain with Ollama (Llama 3.1) for ReAct pattern with SQL Agent
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from langchain.sql_database import SQLDatabase
from langchain_community.llms import Ollama
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from typing import Dict, Any, Optional
import json
import os

# Configuration
DB_PATH = "ecommerce.db"
MODEL_NAME = "llama3.1:8b"

# Load schema documentation
SCHEMA_DOCS = ""
if os.path.exists("schema_docs.txt"):
    with open("schema_docs.txt", "r") as f:
        SCHEMA_DOCS = f.read()

# Initialize SQLAlchemy engine and LangChain SQL Database
try:
    engine = create_engine(f"sqlite:///{DB_PATH}")
    db = SQLDatabase(engine=engine)
except Exception as e:
    raise Exception(f"Failed to connect to database. Make sure {DB_PATH} exists.\nError: {str(e)}")

def get_schema_info() -> str:
    """Get detailed schema information for the agent"""
    schema = "DATABASE SCHEMA:\n\n"
    
    for table in db.get_table_names():
        schema += f"TABLE: {table}\n"
        columns = db.get_table_info([table])
        schema += columns + "\n\n"
    
    schema += "\nTRANSLATION GUIDANCE:\n"
    schema += "- For product categories, use JOIN with category_translation table\n"
    schema += "- Use product_category_name_english column for English translations\n"
    schema += "- Example: SELECT pt.product_category_name_english FROM category_translation pt WHERE pt.product_category_name = 'Beleza Saúde'\n"
    
    return schema

SCHEMA_INFO = get_schema_info()

# Store last query result for analysis/visualization tools
_last_query_result = None

def sql_query_tool(query: str) -> str:
    """
    Executes SQL query on the e-commerce database using SQLAlchemy.
    Input should be a valid SQL SELECT query string.
    Returns query results as formatted string.
    """
    global _last_query_result
    
    try:
        query = query.strip()
        if query.startswith('```'):
            query = query.split('```')[1]
            if query.startswith('sql'):
                query = query[3:].strip()
        
        query_upper = query.strip().upper()
        if not query_upper.startswith('SELECT'):
            return "Error: Only SELECT queries are allowed for safety."
        
        _last_query_result = pd.read_sql_query(query, engine)
        
        if _last_query_result.empty:
            return "Query returned no results."
        
        if len(_last_query_result) > 20:
            result = _last_query_result.head(20).to_string(index=False)
            result += f"\n\n(Showing first 20 of {len(_last_query_result)} total rows)"
        else:
            result = _last_query_result.to_string(index=False)
        
        numeric_cols = _last_query_result.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            result += "\n\nNumeric columns summary:"
            for col in numeric_cols[:3]:
                result += f"\n  {col}: min={_last_query_result[col].min():.2f}, max={_last_query_result[col].max():.2f}, mean={_last_query_result[col].mean():.2f}"
        
        return result
    
    except Exception as e:
        return f"SQL Error: {str(e)}\nPlease check your query syntax."

# ==================== TOOL FUNCTIONS ====================

def data_analysis_tool(instruction: str) -> str:
    """
    Performs data analysis on the last SQL query result using pandas.
    Input should describe the analysis operation.
    Examples: 'calculate mean and median', 'find correlation between columns', 'describe statistics'
    """
    try:
        if _last_query_result is None or len(_last_query_result) == 0:
            return "No data available for analysis. Please run a SQL query first."
        
        df = _last_query_result
        instruction_lower = instruction.lower()
        
        if 'mean' in instruction_lower or 'average' in instruction_lower:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return "No numeric columns found in the data."
            result = "Mean values:\n" + df[numeric_cols].mean().to_string()
            return result
        
        elif 'median' in instruction_lower:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return "No numeric columns found in the data."
            result = "Median values:\n" + df[numeric_cols].median().to_string()
            return result
        
        elif 'sum' in instruction_lower or 'total' in instruction_lower:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return "No numeric columns found in the data."
            result = "Sum totals:\n" + df[numeric_cols].sum().to_string()
            return result
        
        elif 'correlation' in instruction_lower or 'corr' in instruction_lower:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                return "Need at least 2 numeric columns for correlation analysis."
            corr = df[numeric_cols].corr()
            return "Correlation matrix:\n" + corr.to_string()
        
        elif 'describe' in instruction_lower or 'statistics' in instruction_lower or 'stats' in instruction_lower:
            return "Statistical summary:\n" + df.describe().to_string()
        
        elif 'count' in instruction_lower:
            return "Value counts:\n" + str(df.count())
        
        else:
            return "Statistical summary:\n" + df.describe().to_string()
    
    except Exception as e:
        return f"Analysis Error: {str(e)}"

def visualization_tool(chart_config: str) -> str:
    """
    Creates visualizations from the last SQL query result.
    Input should be JSON string with chart configuration:
    {"type": "bar/line/scatter/pie", "x": "column_name", "y": "column_name", "title": "Chart Title"}
    
    For pie charts: {"type": "pie", "names": "column_name", "values": "column_name", "title": "..."}
    
    Returns confirmation message if successful.
    """
    try:
        if _last_query_result is None or len(_last_query_result) == 0:
            return "No data available for visualization. Please run a SQL query first."
        
        df = _last_query_result
        
        try:
            config = json.loads(chart_config)
        except json.JSONDecodeError:
            if 'pie' in chart_config.lower():
                config = {"type": "pie"}
            elif 'line' in chart_config.lower():
                config = {"type": "line"}
            elif 'scatter' in chart_config.lower():
                config = {"type": "scatter"}
            else:
                config = {"type": "bar"}
        
        chart_type = config.get("type", "bar").lower()
        title = config.get("title", "Data Visualization")
        
        plot_df = df.head(50) if len(df) > 50 else df
        
        fig = None
        
        if chart_type == "bar":
            x_col = config.get("x", df.columns[0])
            y_col = config.get("y", df.columns[1] if len(df.columns) > 1 else df.columns[0])
            
            if x_col not in df.columns or y_col not in df.columns:
                return f"Error: Columns '{x_col}' or '{y_col}' not found in data."
            
            fig = px.bar(plot_df, x=x_col, y=y_col, title=title)
            fig.update_layout(xaxis_tickangle=-45)
        
        elif chart_type == "line":
            x_col = config.get("x", df.columns[0])
            y_col = config.get("y", df.columns[1] if len(df.columns) > 1 else df.columns[0])
            
            if x_col not in df.columns or y_col not in df.columns:
                return f"Error: Columns '{x_col}' or '{y_col}' not found in data."
            
            fig = px.line(df, x=x_col, y=y_col, title=title, markers=True)
        
        elif chart_type == "scatter":
            x_col = config.get("x", df.columns[0])
            y_col = config.get("y", df.columns[1] if len(df.columns) > 1 else df.columns[0])
            
            if x_col not in df.columns or y_col not in df.columns:
                return f"Error: Columns '{x_col}' or '{y_col}' not found in data."
            
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
        
        elif chart_type == "pie":
            names_col = config.get("names", df.columns[0])
            values_col = config.get("values", df.columns[1] if len(df.columns) > 1 else df.columns[0])
            
            if names_col not in df.columns or values_col not in df.columns:
                return f"Error: Columns '{names_col}' or '{values_col}' not found in data."
            
            pie_df = df.nlargest(10, values_col) if len(df) > 10 else df
            fig = px.pie(pie_df, names=names_col, values=values_col, title=title)
        
        else:
            return f"Error: Unsupported chart type '{chart_type}'. Use: bar, line, scatter, or pie."
        
        if fig is None:
            return "Error: Could not create visualization."
        
        fig.update_layout(
            template="plotly_white",
            font=dict(size=12),
            title_font=dict(size=16, color='#2c3e50'),
            height=500
        )
        
        visualization_tool.last_figure = fig
        
        return f"✓ Successfully created {chart_type} chart: '{title}'"
    
    except Exception as e:
        return f"Visualization Error: {str(e)}"

visualization_tool.last_figure = None

def auto_create_visualization() -> str:
    """
    Automatically create an appropriate visualization based on the last query result.
    Returns visualization configuration JSON string or empty string if no visualization needed.
    """
    try:
        if _last_query_result is None or len(_last_query_result) == 0 or len(_last_query_result.columns) < 2:
            return ""

        df = _last_query_result
        columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        # Skip visualization for single column results or very large datasets
        if len(columns) < 2 or len(df) > 100:
            return ""

        # Determine chart type based on data structure
        config = {}

        # Check for time series (columns with 'month', 'date', 'time' in name)
        time_cols = [col for col in columns if any(keyword in col.lower() for keyword in ['month', 'date', 'time', 'year'])]
        if time_cols and numeric_cols:
            config = {
                "type": "line",
                "x": time_cols[0],
                "y": numeric_cols[0],
                "title": f"{numeric_cols[0].replace('_', ' ').title()} Over Time"
            }
        # Check for categorical data with counts
        elif any(keyword in ' '.join(columns).lower() for keyword in ['count', 'total', 'sum', 'avg', 'average']):
            if len(df) <= 10:
                config = {
                    "type": "pie",
                    "names": columns[0],
                    "values": numeric_cols[0] if numeric_cols else columns[1],
                    "title": f"Distribution of {columns[0].replace('_', ' ').title()}"
                }
            else:
                config = {
                    "type": "bar",
                    "x": columns[0],
                    "y": numeric_cols[0] if numeric_cols else columns[1],
                    "title": f"{columns[0].replace('_', ' ').title()} vs {numeric_cols[0] if numeric_cols else columns[1]}"
                }
        # Default to bar chart for most data
        elif len(df) <= 20:
            config = {
                "type": "bar",
                "x": columns[0],
                "y": columns[1] if len(columns) > 1 else columns[0],
                "title": f"{columns[1] if len(columns) > 1 else columns[0]} by {columns[0]}"
            }
        else:
            return ""  # Skip visualization for large datasets

        return json.dumps(config)

    except Exception as e:
        return ""

# ==================== AGENT SETUP ====================

def initialize_agent() -> AgentExecutor:
    """Initialize the LangChain agent with SQLAlchemy-based SQL tool"""
    
    try:
        llm = Ollama(
            model=MODEL_NAME,
            temperature=0,
            num_ctx=4096
        )
    except Exception as e:
        raise Exception(f"Failed to initialize Ollama. Make sure Ollama is running and model '{MODEL_NAME}' is pulled.\nError: {str(e)}")
    
    tools = [
        Tool(
            name="SQLDatabase",
            func=sql_query_tool,
            description=f"""Execute SQL SELECT queries on the Brazilian e-commerce database.
Input: A valid SQL SELECT query.
Output: Query results formatted as a table.

CRITICAL - Use these EXACT column names:
{SCHEMA_INFO}

Key column names:
- orders: order_id, order_purchase_timestamp (NOT created_at or date)
- customers: customer_id, customer_state
- products: product_id, product_category_name
- order_items: order_id, product_id, price
- order_payments: order_id, payment_type, payment_value
- order_reviews: order_id, review_score

TRANSLATION: For product category results, JOIN with category_translation table and use product_category_name_english for English names.
Example: 
  SELECT pt.product_category_name_english, SUM(oi.price) as revenue
  FROM products p
  JOIN category_translation pt ON p.product_category_name = pt.product_category_name
  JOIN order_items oi ON p.product_id = oi.product_id
  GROUP BY pt.product_category_name_english

SELLER QUERIES: Join sellers with order_items, then orders, then order_reviews.
Example for seller ratings:
  SELECT s.seller_id, AVG(rev.review_score) as avg_rating
  FROM sellers s
  LEFT JOIN order_items oi ON s.seller_id = oi.seller_id
  LEFT JOIN orders o ON oi.order_id = o.order_id
  LEFT JOIN order_reviews rev ON o.order_id = rev.order_id
  GROUP BY s.seller_id
  ORDER BY avg_rating DESC

DATE AND TREND QUERIES: Use strftime for date filtering and grouping.
Example for monthly trends:
  SELECT strftime('%Y-%m', order_purchase_timestamp) as month, COUNT(*) as order_count
  FROM orders
  WHERE strftime('%Y', order_purchase_timestamp) = '2017'
  GROUP BY strftime('%Y-%m', order_purchase_timestamp)
  ORDER BY month

SQLite syntax examples:
- COUNT orders: SELECT COUNT(*) FROM orders;
- Date filter: WHERE order_purchase_timestamp >= '2017-01-01'
- Month extract: strftime('%Y-%m', order_purchase_timestamp)
"""
        ),
        Tool(
            name="DataAnalysis",
            func=data_analysis_tool,
            description="""Analyze data from the most recent SQL query result.
Input: Description of analysis like 'calculate mean', 'find correlation', 'describe statistics'
Output: Statistical analysis results.

Must run SQLDatabase tool first to have data available.
Examples: 'calculate mean and median', 'show correlation matrix', 'describe statistics'
"""
        ),
        Tool(
            name="CreateVisualization",
            func=visualization_tool,
            description="""Create charts from the most recent SQL query result.
Input: JSON configuration with chart details.
Format: {{"type": "bar/line/scatter/pie", "x": "column", "y": "column", "title": "Title"}}
For pie: {{"type": "pie", "names": "column", "values": "column", "title": "Title"}}

Must run SQLDatabase tool first to have data available.
Example: {{"type": "bar", "x": "category", "y": "count", "title": "Top Categories"}}
"""
        )
    ]
    
    react_prompt = PromptTemplate.from_template("""You are a data analyst. Answer questions using the SQLDatabase tool.

Use this format EXACTLY:
Thought: what data do I need?
Action: SQLDatabase
Action Input: SELECT query here
Observation: [results]
Final Answer: [answer based on results]

COLUMN NAMES:
- orders: order_id, order_purchase_timestamp, customer_id, order_status
- customers: customer_id, customer_state
- sellers: seller_id, seller_state
- order_items: order_id, product_id, seller_id, price
- order_reviews: order_id, review_score
- products: product_id, product_category_name
- category_translation: product_category_name, product_category_name_english

JOINS: sellers->order_items (seller_id), order_items->orders (order_id), orders->order_reviews (order_id)

CORRECT EXAMPLES (copy-paste these if needed):
- Monthly 2017: SELECT strftime('%Y-%m', order_purchase_timestamp) as month, COUNT(*) as count FROM orders WHERE strftime('%Y', order_purchase_timestamp)='2017' GROUP BY strftime('%Y-%m', order_purchase_timestamp)
- Seller ratings: SELECT s.seller_id, AVG(r.review_score) FROM sellers s JOIN order_items oi ON s.seller_id=oi.seller_id JOIN orders o ON oi.order_id=o.order_id JOIN order_reviews r ON o.order_id=r.order_id GROUP BY s.seller_id ORDER BY AVG(r.review_score) DESC
- Categories with reviews: SELECT pt.product_category_name_english, COUNT(rev.review_id) as review_count FROM products p JOIN category_translation pt ON p.product_category_name=pt.product_category_name JOIN order_items oi ON p.product_id=oi.product_id JOIN orders o ON oi.order_id=o.order_id LEFT JOIN order_reviews rev ON o.order_id=rev.order_id GROUP BY pt.product_category_name_english ORDER BY review_count DESC
- Reviews by payment: SELECT op.payment_type, AVG(rev.review_score) as avg_review FROM order_payments op JOIN orders o ON op.order_id=o.order_id LEFT JOIN order_reviews rev ON o.order_id=rev.order_id GROUP BY op.payment_type
- Revenue by month: SELECT strftime('%Y-%m', o.order_purchase_timestamp) as month, SUM(oi.price) as revenue FROM orders o JOIN order_items oi ON o.order_id=oi.order_id WHERE strftime('%Y', o.order_purchase_timestamp)='2017' AND strftime('%m', o.order_purchase_timestamp) IN ('01','02','03','04','05','06') GROUP BY strftime('%Y-%m', o.order_purchase_timestamp)

RULES:
- Only use Action and Action Input, never explain queries
- Execute the query immediately after Action Input
- Do not describe what you're about to do

{tools}

Tool Names: {tool_names}

Question: {input}

{agent_scratchpad}""")
    
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=20,
        max_execution_time=90,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        early_stopping_method="force"
    )
    
    return agent_executor

_agent_executor = None

def get_agent() -> AgentExecutor:
    """Get or create agent instance (singleton)"""
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = initialize_agent()
    return _agent_executor

def reset_agent():
    """Reset the agent instance to force reload"""
    global _agent_executor
    _agent_executor = None

def query_agent(question: str) -> Dict[str, Any]:
    """
    Query the agent with a question

    Args:
        question: User's question about the data

    Returns:
        Dictionary with 'answer', 'figure', and 'success' keys
    """
    try:
        visualization_tool.last_figure = None

        agent = get_agent()

        response = agent.invoke({"input": question})

        answer = response.get("output", "No response generated.")

        if "Agent stopped due to iteration" in answer or not answer.strip():
            if _last_query_result is not None and not _last_query_result.empty:
                answer = "Results:\n\n" + _last_query_result.to_string(index=False)

        # Auto-create visualization if SQL query was executed and no visualization was created by agent
        if _last_query_result is not None and not _last_query_result.empty and visualization_tool.last_figure is None:
            viz_config = auto_create_visualization()
            if viz_config:
                try:
                    visualization_tool(viz_config)
                except Exception as viz_error:
                    # Silently fail visualization creation to not break the main flow
                    pass

        return {
            "answer": answer,
            "figure": visualization_tool.last_figure,
            "data": _last_query_result,
            "success": True
        }
    
    except Exception as e:
        error_msg = str(e)
        
        if "Ollama" in error_msg or "connect" in error_msg.lower():
            error_msg = f"""❌ Cannot connect to Ollama. Please ensure:

1. Ollama is installed: curl -fsSL https://ollama.com/install.sh | sh
2. Ollama is running: ollama serve (in another terminal)
3. Model is downloaded: ollama pull {MODEL_NAME}

Original error: {error_msg}"""
        
        return {
            "answer": f"Error: {error_msg}",
            "figure": None,
            "data": None,
            "success": False
        }

# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing E-Commerce AI Agent...\n")
    
    test_questions = [
        "How many orders are in the database?",
        "What are the top 5 product categories by number of orders?",
        "What are the most common payment methods?",
        "Show me the monthly order trends in 2017",
        "Which states have the highest average order values?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {question}")
        print('='*60)
        
        result = query_agent(question)
        
        if result["success"]:
            print(f"\n✓ Answer:\n{result['answer']}\n")
            if result["figure"]:
                print("✓ Visualization created")
        else:
            print(f"\n✗ Error:\n{result['answer']}\n")
    
    print("\n" + "="*60)
    print("Testing complete!")
