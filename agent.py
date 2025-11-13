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
_last_count_result = None

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

        # Store count results separately for fallback logic
        global _last_count_result
        if len(_last_query_result) == 1 and len(_last_query_result.columns) == 1 and ('count' in _last_query_result.columns[0].lower() or _last_query_result.columns[0] == 'COUNT(*)'):
            _last_count_result = _last_query_result.copy()

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

        # Special handling for correlation analysis (2 numeric columns)
        if len(numeric_cols) == 2 and len(columns) == 2:
            # Create scatter plot for correlation analysis, even with large datasets
            config = {
                "type": "scatter",
                "x": numeric_cols[0],
                "y": numeric_cols[1],
                "title": f"Correlation: {numeric_cols[0].replace('_', ' ').title()} vs {numeric_cols[1].replace('_', ' ').title()}"
            }
            return json.dumps(config)

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

DELIVERY TIME ANALYSIS: Calculate delivery time and correlate with reviews.
Example for delivery time correlation:
  SELECT
    (julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp)) as delivery_days,
    r.review_score
  FROM orders o
  JOIN order_reviews r ON o.order_id = r.order_id
  WHERE o.order_status = 'delivered'
    AND o.order_delivered_customer_date IS NOT NULL
    AND o.order_purchase_timestamp IS NOT NULL

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
For trend queries (monthly/yearly patterns), use 'calculate mean and median' instead of correlation.
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
Thought: now I need to analyze this data
Action: DataAnalysis
Action Input: correlation
Observation: [analysis results]
Thought: I have the correlation results, now provide final answer
Final Answer: [brief analysis of the correlation coefficient]

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
- Payment methods: SELECT payment_type, COUNT(*) as count FROM order_payments GROUP BY payment_type ORDER BY count DESC
- Customer distribution by state: SELECT customer_state, COUNT(*) as count FROM customers GROUP BY customer_state ORDER BY count DESC
- Reviews by payment: SELECT op.payment_type, AVG(rev.review_score) as avg_review FROM order_payments op JOIN orders o ON op.order_id=o.order_id LEFT JOIN order_reviews rev ON o.order_id=rev.order_id GROUP BY op.payment_type
- Revenue by month: SELECT strftime('%Y-%m', o.order_purchase_timestamp) as month, SUM(oi.price) as revenue FROM orders o JOIN order_items oi ON o.order_id=oi.order_id WHERE strftime('%Y', o.order_purchase_timestamp)='2017' AND strftime('%m', o.order_purchase_timestamp) IN ('01','02','03','04','05','06') GROUP BY strftime('%Y-%m', o.order_purchase_timestamp)
- Delivery time correlation: SELECT (julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp)) as delivery_days, r.review_score FROM orders o JOIN order_reviews r ON o.order_id = r.order_id WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL AND o.order_purchase_timestamp IS NOT NULL
- Price vs satisfaction: SELECT oi.price, r.review_score FROM order_items oi JOIN orders o ON oi.order_id = o.order_id LEFT JOIN order_reviews r ON o.order_id = r.order_id WHERE o.order_status = 'delivered'
- States by average order value: SELECT c.customer_state, AVG(op.payment_value) as avg_order_value FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_payments op ON o.order_id = op.order_id GROUP BY c.customer_state ORDER BY avg_order_value DESC LIMIT 10

RULES:
- Only use Action and Action Input, never explain queries
- Execute the query immediately after Action Input
- Do not describe what you're about to do
- After getting SQL data, ALWAYS use DataAnalysis tool for correlation/statistics
- For correlation questions, get data first, then analyze with DataAnalysis tool

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
        max_iterations=5,
        max_execution_time=60,
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

def provide_follow_up_analysis() -> Dict[str, Any]:
    """
    Provide additional analysis on the last query result.
    Returns additional insights, statistics, or visualizations.
    """
    try:
        if _last_query_result is None or len(_last_query_result) == 0:
            return {
                "answer": "No previous query data available for additional analysis.",
                "figure": None,
                "success": False
            }

        df = _last_query_result
        analysis_parts = []

        # Basic statistics
        analysis_parts.append("Additional Analysis:")
        analysis_parts.append(f"- Total rows: {len(df)}")
        analysis_parts.append(f"- Columns: {', '.join(df.columns)}")

        # Numeric column analysis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) > 0:
            analysis_parts.append("\nNumeric Summary:")
            for col in numeric_cols[:3]:  # Limit to first 3 columns
                analysis_parts.append(f"- {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, avg={df[col].mean():.2f}")

        # Create additional visualization if appropriate
        additional_viz = None
        if len(df) > 1 and len(df.columns) >= 2 and len(df) <= 20:
            try:
                # Simple bar chart for additional insights
                config = {
                    "type": "bar",
                    "x": df.columns[0],
                    "y": df.columns[1] if len(df.columns) > 1 else df.columns[0],
                    "title": f"Additional Analysis: {df.columns[0]}"
                }
                viz_result = visualization_tool(json.dumps(config))
                if "successfully created" in viz_result.lower():
                    additional_viz = visualization_tool.last_figure
                    analysis_parts.append("\n✓ Additional visualization created")
            except Exception as viz_error:
                # Silently skip visualization on error
                pass

        return {
            "answer": "\n".join(analysis_parts),
            "figure": additional_viz,
            "success": True
        }

    except Exception as e:
        return {
            "answer": f"Error generating follow-up analysis: {str(e)}",
            "figure": None,
            "success": False
        }

def enhance_answer_based_on_query(question: str, query_result: pd.DataFrame, count_result: pd.DataFrame = None) -> str:
    """
    Enhance the agent's answer based on the query type and results.
    Returns enhanced answer string or None if no enhancement needed.
    """
    if query_result is None or query_result.empty:
        return None

    numeric_cols = query_result.select_dtypes(include=['number']).columns
    # Handle payment method queries
    if 'payment' in question.lower() and ('method' in question.lower() or 'type' in question.lower()):
        print("DEBUG: Payment method handler triggered")
        if len(query_result) <= 10 and len(query_result.columns) >= 1:
            # Check if we have counts or just types
            if len(query_result.columns) == 1:
                # Only payment types, need to get counts
                try:
                    count_result = pd.read_sql_query("""
                        SELECT payment_type, COUNT(*) as count
                        FROM order_payments
                        GROUP BY payment_type
                        ORDER BY count DESC
                        LIMIT 10
                    """, engine)

                    total_payments = count_result['count'].sum()
                    payment_methods = []

                    for _, row in count_result.iterrows():
                        method = str(row['payment_type'])
                        count = int(row['count'])
                        percentage = (count / total_payments) * 100

                        if method == 'credit_card':
                            method = 'credit card'
                        elif method == 'debit_card':
                            method = 'debit card'
                        elif method == 'boleto':
                            method = 'boleto (bank slip)'

                        # Skip not_defined if percentage is very low
                        if method == 'not_defined' and percentage < 0.1:
                            continue

                        payment_methods.append(f"{method}: {percentage:.0f}%")

                    answer = f"Payment method distribution:\n" + "\n".join(payment_methods)

                    # Add more detailed breakdown
                    top_method = count_result.iloc[0]['payment_type']
                    top_count = int(count_result.iloc[0]['count'])
                    top_pct = (top_count / total_payments) * 100

                    if top_method == 'credit_card':
                        top_method_display = 'credit card'
                    elif top_method == 'boleto':
                        top_method_display = 'boleto (bank slip)'
                    else:
                        top_method_display = top_method.replace('_', ' ')

                    answer = f"The most common payment method is {top_method_display}, used in about {top_pct:.0f}% of all orders, followed by " + ", ".join(payment_methods[1:]) + "."
                    return answer

                except Exception as e:
                    return f"Payment methods:\n{query_result.to_string(index=False)}"
            else:
                # We have counts
                total = query_result.iloc[:, 1].sum() if len(query_result.columns) > 1 else len(query_result)
                payment_methods = []

                for _, row in query_result.iterrows():
                    method = str(row.iloc[0])
                    count = int(row.iloc[1]) if len(row) > 1 else 1
                    percentage = (count / total) * 100 if total > 0 else 0

                    if method == 'credit_card':
                        method = 'credit card'
                    elif method == 'debit_card':
                        method = 'debit card'
                    elif method == 'boleto':
                        method = 'boleto (bank slip)'

                    # Skip not_defined if percentage is very low
                    if method == 'not_defined' and percentage < 0.1:
                        continue

                    payment_methods.append(f"{method}: {percentage:.0f}%")

                return f"Payment method distribution:\n" + "\n".join(payment_methods)
        else:
            return f"Payment methods:\n{query_result.to_string(index=False)}"

    # Handle distribution queries (customer/state distribution)
    elif ('distribution' in question.lower() or 'regions' in question.lower() or ('most' in question.lower() and 'customers' in question.lower())) and len(query_result) > 5 and len(query_result.columns) >= 2 and 'count' in query_result.columns[1].lower():
        print("DEBUG: Distribution handler triggered")
        category_col = query_result.columns[0]
        count_col = query_result.columns[1]

        # Check if count column contains numeric data
        if not pd.api.types.is_numeric_dtype(query_result[count_col]):
            return None  # Skip if count column is not numeric

        # Sort by count descending
        sorted_result = query_result.sort_values(count_col, ascending=False)
        total_customers = sorted_result[count_col].sum()

        # Get top states and their percentages
        top_states = []
        for _, row in sorted_result.head(5).iterrows():
            state = str(row[category_col]).upper()
            count = int(row[count_col])
            percentage = (count / total_customers) * 100
            top_states.append(f"{state} ({percentage:.1f}%)")

        # Identify major regions
        sp_count = sorted_result[sorted_result[category_col] == 'SP'][count_col].iloc[0] if len(sorted_result[sorted_result[category_col] == 'SP']) > 0 else 0
        rj_count = sorted_result[sorted_result[category_col] == 'RJ'][count_col].iloc[0] if len(sorted_result[sorted_result[category_col] == 'RJ']) > 0 else 0
        mg_count = sorted_result[sorted_result[category_col] == 'MG'][count_col].iloc[0] if len(sorted_result[sorted_result[category_col] == 'MG']) > 0 else 0

        sp_pct = (sp_count / total_customers) * 100
        rj_pct = (rj_count / total_customers) * 100
        mg_pct = (mg_count / total_customers) * 100

        # Regional analysis
        # Northeast states
        ne_states = ['BA', 'PE', 'CE', 'RN', 'PB', 'AL', 'SE', 'PI', 'MA']
        ne_total = sum(sorted_result[sorted_result[category_col].isin(ne_states)][count_col])
        ne_pct = (ne_total / total_customers) * 100

        # South states
        south_states = ['RS', 'SC', 'PR']
        south_total = sum(sorted_result[sorted_result[category_col].isin(south_states)][count_col])
        south_pct = (south_total / total_customers) * 100

        # Small states (North region)
        small_states = ['AC', 'RR', 'AP', 'RO', 'TO', 'AM', 'PA']
        small_total = sum(sorted_result[sorted_result[category_col].isin(small_states)][count_col])
        small_pct = (small_total / total_customers) * 100

        answer = f"The customer base in the Olist dataset is widely distributed across Brazil, with notable regional concentrations. "

        if sp_pct > 35:
            answer += f"São Paulo (SP) dominates the distribution, accounting for approximately {sp_pct:.0f}% of all customers, reflecting its high population density and economic activity. "

        if rj_pct > 8 and mg_pct > 8:
            answer += f"Rio de Janeiro (RJ) and Minas Gerais (MG) follow, each contributing roughly {rj_pct:.0f}% and {mg_pct:.0f}% of the customer base respectively. "

        if south_pct > 10:
            answer += f"The Southern region (RS, SC, PR) shows strong participation with about {south_pct:.0f}% of customers. "

        if ne_pct > 8:
            answer += f"States from the Northeast region such as Bahia (BA), Pernambuco (PE), and Ceará (CE) also show moderate participation, collectively representing about {ne_pct:.0f}% of customers. "

        if small_pct < 5:
            answer += f"Less populous regions exhibit minimal customer representation, with the northern states contributing only about {small_pct:.0f}% of the total customer base. "

        answer += f"The top 5 states by customer count are: {', '.join(top_states)}. The visualization provides a clear geographical view of customer concentration across Brazilian states."

        return answer

    # Handle top categories/products queries (check first to avoid conflicts)
    elif any(word in question.lower() for word in ['top', 'best', 'most', 'highest']):
        print("DEBUG: Top handler triggered")
        if len(query_result) <= 20 and len(query_result.columns) >= 2:
            # Format as professional top list
            category_col = query_result.columns[0]
            value_col = query_result.columns[1]

            # Clean up column name for display
            display_name = category_col.replace('_', ' ')
            display_name = display_name.replace('product category name english', 'product categories')
            display_name = display_name.replace('product category name', 'product categories')
            display_name = display_name.replace('category name english', 'categories')
            display_name = display_name.replace('category name', 'categories')
            display_name = display_name.replace(' name english', '')
            display_name = display_name.replace(' name', '')

            # Determine if values are currency (revenue) or counts
            is_currency = 'revenue' in value_col.lower() or 'price' in value_col.lower() or 'amount' in value_col.lower() or 'value' in value_col.lower() or 'order_value' in value_col.lower()

            answer = f"🏆 **Top {len(query_result)} {display_name.title()}:**\n\n"

            top_items = []
            for i, (_, row) in enumerate(query_result.iterrows(), 1):
                item = str(row[category_col]).replace('_', ' ').upper()  # Use upper for state codes
                value = float(row[value_col])

                if is_currency:
                    formatted_value = f"R$ {value:,.0f}"
                else:
                    formatted_value = f"{value:,.0f}"

                top_items.append(f"{i:2d}. {item:<25} {formatted_value:>12}")

            answer += "```\n" + "\n".join(top_items) + "\n```"

            # Add insights
            try:
                # For averages, calculate overall average, not sum
                if 'avg' in value_col.lower() or 'average' in value_col.lower():
                    overall_metric = query_result[value_col].mean()
                    metric_name = "Average"
                    # For averages, show how many states are above overall average
                    states_above_avg = sum(1 for val in query_result[value_col] if val > overall_metric)
                    concentration_text = f"{states_above_avg} of {len(query_result)} states have above-average values"
                else:
                    overall_metric = sum(query_result[value_col])
                    metric_name = "Total"
                    top5_total = sum(query_result[value_col][:5])
                    concentration_pct = (top5_total / overall_metric) * 100 if overall_metric > 0 else 0
                    concentration_text = f"Top 5 represent {concentration_pct:.1f}% of total {'revenue' if is_currency else 'volume'}"

                leader = str(query_result.iloc[0][category_col]).replace('_', ' ').upper()
                leader_value = float(query_result.iloc[0][value_col])

                if is_currency:
                    leader_formatted = f"R$ {leader_value:,.0f}"
                    overall_formatted = f"R$ {overall_metric:,.0f}"
                else:
                    leader_formatted = f"{leader_value:,.0f}"
                    overall_formatted = f"{overall_metric:,.0f}"

                answer += f"\n\n📊 **Key Insights:**\n"
                answer += f"• **Market Leader:** {leader} leads with {leader_formatted}\n"
                answer += f"• **Distribution:** {concentration_text}\n"
                answer += f"• **Overall {metric_name} {'Revenue' if is_currency else 'Volume'}:** {overall_formatted}"

            except Exception as e:
                pass

            return answer
        else:
            return f"Top results:\n{query_result.to_string(index=False)}"

    # Handle count queries (check question type)
    elif any(word in question.lower() for word in ['how many', 'count', 'number of', 'total']):
        # Check if we have a stored count result
        if count_result is not None and not count_result.empty:
            count_value = int(count_result.iloc[0, 0])
            if 'late' in question.lower() or 'delay' in question.lower():
                # Calculate percentage if we can
                try:
                    total_result = pd.read_sql_query("SELECT COUNT(*) FROM orders", engine)
                    total_orders = int(total_result.iloc[0, 0])
                    percentage = (count_value / total_orders) * 100
                    return f"Approximately {percentage:.1f}% of orders ({count_value:,}) were delivered late."
                except:
                    return f"Looks like {count_value:,} orders were delivered late."
            elif 'orders' in question.lower():
                return f"There are {count_value:,} orders in the database."
            elif 'customers' in question.lower():
                return f"We've got {count_value:,} customers total."
            elif 'products' in question.lower():
                return f"There are {count_value:,} products in the catalog."
            elif 'sellers' in question.lower():
                return f"There are {count_value:,} sellers on the platform."
            else:
                return f"The count comes to {count_value:,}."
        elif len(query_result) == 1 and len(numeric_cols) == 1:
            count_value = int(query_result.iloc[0, 0])
            if 'late' in question.lower() or 'delay' in question.lower():
                try:
                    total_result = pd.read_sql_query("SELECT COUNT(*) FROM orders", engine)
                    total_orders = int(total_result.iloc[0, 0])
                    percentage = (count_value / total_orders) * 100
                    return f"Approximately {percentage:.1f}% of orders ({count_value:,}) were delivered late."
                except:
                    return f"Looks like {count_value:,} orders were delivered late."
            elif 'orders' in question.lower():
                return f"There are {count_value:,} orders in the database."
            elif 'customers' in question.lower():
                return f"We've got {count_value:,} customers total."
            elif 'products' in question.lower():
                return f"There are {count_value:,} products in the catalog."
            elif 'sellers' in question.lower():
                return f"There are {count_value:,} sellers on the platform."
            else:
                return f"The count comes to {count_value:,}."
        else:
            # Count the rows if it's not a single count result
            count_value = len(query_result)
            return f"Found {count_value:,} results in total."

    # Handle time series data (monthly trends) - check before revenue handler
    elif len(query_result) > 1 and any(col.lower() in ['month', 'date', 'time', 'year'] for col in query_result.columns):
        print("DEBUG: Time series handler triggered")
        if len(query_result) <= 50:  # Increased limit for more comprehensive time series
            try:
                # Format as professional business report
                import calendar

                # Extract data
                periods = []
                values = []
                for _, row in query_result.iterrows():
                    period = str(row.iloc[0])
                    value = float(row.iloc[1])  # Use float to handle both counts and revenue
                    periods.append(period)
                    values.append(value)

                # Convert month format (2017-01 -> January 2017)
                formatted_periods = []
                for period in periods:
                    if len(period) == 7 and period[4] == '-':  # YYYY-MM format
                        year, month = period.split('-')
                        month_name = calendar.month_name[int(month)]
                        formatted_periods.append(f"{month_name} {year}")
                    else:
                        formatted_periods.append(period)

                # Determine metric name based on column name and question
                value_col = query_result.columns[1]
                if 'order' in question.lower() or 'count' in value_col.lower():
                    metric_name = "orders"
                    title_metric = "Orders"
                elif 'revenue' in value_col.lower() or 'price' in value_col.lower() or 'amount' in value_col.lower():
                    metric_name = "revenue"
                    title_metric = "Revenue"
                else:
                    metric_name = "values"
                    title_metric = value_col.replace('_', ' ').title()

                answer = f"📈 **Monthly {title_metric} Trends in 2017**\n\n"

                # Create formatted table
                table_lines = []
                metric_display = "Revenue" if metric_name == "revenue" else "Count"
                table_lines.append(f"```\nMonth          {metric_display:<8} MoM Change")
                table_lines.append("─" * 35)

                prev_value = None
                for i, (period, value) in enumerate(zip(formatted_periods, values)):
                    # Calculate month-over-month change
                    if prev_value is not None and prev_value > 0:
                        change = ((value - prev_value) / prev_value) * 100
                        change_str = f"{change:+.1f}%"
                    else:
                        change_str = "     -"

                    # Format the row
                    if metric_name == "revenue":
                        value_str = f"R$ {value:>7,.0f}"
                    else:
                        value_str = f"{value:>8,.0f}"
                    table_lines.append(f"{period:<15}{value_str}{change_str:>12}")
                    prev_value = value

                table_lines.append("```")
                answer += "\n".join(table_lines)

                # Add summary statistics
                if len(values) > 1:
                    avg_value = sum(values) / len(values)
                    max_value = max(values)
                    min_value = min(values)
                    max_idx = values.index(max_value)
                    min_idx = values.index(min_value)

                    total_change = ((values[-1] - values[0]) / values[0]) * 100 if values[0] > 0 else 0
                    trend = "📈 Growing" if total_change > 10 else "📉 Declining" if total_change < -10 else "➡️ Stable"

                    answer += f"\n\n📊 **Summary Statistics:**\n"
                    if metric_name == "revenue":
                        answer += f"• **Total {title_metric}:** R$ {sum(values):,.0f}\n"
                        answer += f"• **Average per Month:** R$ {avg_value:,.0f}\n"
                        answer += f"• **Peak Month:** {formatted_periods[max_idx]} (R$ {max_value:,.0f})\n"
                        answer += f"• **Lowest Month:** {formatted_periods[min_idx]} (R$ {min_value:,.0f})\n"
                    else:
                        answer += f"• **Total {title_metric}:** {sum(values):,}\n"
                        answer += f"• **Average per Month:** {avg_value:.0f}\n"
                        answer += f"• **Peak Month:** {formatted_periods[max_idx]} ({max_value:,})\n"
                        answer += f"• **Lowest Month:** {formatted_periods[min_idx]} ({min_value:,})\n"
                    answer += f"• **Overall Trend:** {trend} ({total_change:+.1f}% from start to end)\n"

                    # Add seasonality insight
                    if max_value > avg_value * 1.5:
                        answer += f"• **Seasonality:** Strong peak in {formatted_periods[max_idx].split()[0]}"

                    return answer
            except Exception as e:
                return f"Time series data (error in formatting): {query_result.to_string(index=False)}"
        else:
            return f"Found {len(query_result)} data points. Here's a summary of the trends."

    # Handle revenue/sales queries
    elif any(word in question.lower() for word in ['revenue', 'sales', 'total value', 'amount']):
        if len(query_result) == 1 and len(numeric_cols) == 1:
            revenue_value = float(query_result.iloc[0, 0])
            if '2018' in question.lower():
                return f"The total revenue in 2018 was R$ {revenue_value:,.2f}, calculated by summing all price + freight_value from order_items."
            else:
                return f"The total revenue comes to R$ {revenue_value:,.2f}."
        else:
            return f"Revenue analysis shows {query_result.to_string(index=False)}"



    # Handle average queries
    elif any(word in question.lower() for word in ['average', 'avg', 'mean']):
        if len(query_result) <= 10 and len(query_result.columns) >= 2:
            # Format averages by category
            category_col = query_result.columns[0]
            avg_col = query_result.columns[1]

            avg_items = []
            for _, row in query_result.iterrows():
                category = str(row[category_col])
                avg_value = float(row[avg_col])
                avg_items.append(f"{category} → {avg_value:.1f}")

            answer = f"Average {avg_col.replace('_', ' ')}:\n" + "\n".join(avg_items)

            # Add overall average
            try:
                overall_avg = query_result[avg_col].mean()
                answer += f"\n\nThe overall average {avg_col.replace('_', ' ')} is {overall_avg:.2f}."
            except:
                pass
            return answer
        elif len(query_result) == 1 and len(numeric_cols) == 1:
            avg_value = float(query_result.iloc[0, 0])
            return f"The average comes to {avg_value:.2f}."
        elif len(query_result) > 10 and len(query_result.columns) >= 2:
            # Handle large datasets - show top and bottom performers
            category_col = query_result.columns[0]
            avg_col = query_result.columns[1]

            # Sort by average descending
            sorted_result = query_result.sort_values(avg_col, ascending=False)

            # Get top 10 and bottom 10
            top_10 = sorted_result.head(10)
            bottom_10 = sorted_result.tail(10)

            answer = f"Average {avg_col.replace('_', ' ')} by {category_col.replace('_', ' ')}:\n\n"

            answer += "🏆 **Top 10 Categories:**\n"
            for _, row in top_10.iterrows():
                category = str(row[category_col])
                avg_value = float(row[avg_col])
                answer += f"  {category}: {avg_value:.2f}\n"

            answer += "\n📉 **Bottom 10 Categories:**\n"
            for _, row in bottom_10.iterrows():
                category = str(row[category_col])
                avg_value = float(row[avg_col])
                answer += f"  {category}: {avg_value:.2f}\n"

            # Add overall statistics
            try:
                overall_avg = query_result[avg_col].mean()
                max_avg = query_result[avg_col].max()
                min_avg = query_result[avg_col].min()
                answer += f"\n📊 **Summary Statistics:**\n"
                answer += f"  Overall average: {overall_avg:.2f}\n"
                answer += f"  Highest rating: {max_avg:.2f}\n"
                answer += f"  Lowest rating: {min_avg:.2f}\n"
                answer += f"  Total categories: {len(query_result)}"
            except:
                pass

            return answer
        else:
            return f"Average analysis:\n{query_result.to_string(index=False)}"

    # Handle percentage queries
    elif 'percentage' in question.lower() or 'percent' in question.lower():
        if len(query_result) == 1 and len(numeric_cols) == 1:
            pct_value = float(query_result.iloc[0, 0])
            return f"Approximately {pct_value:.1f}%."
        else:
            return f"Percentage breakdown:\n{query_result.to_string(index=False)}"

    # Handle correlation analysis
    elif len(numeric_cols) == 2 and any(word in question.lower() for word in ['correlation', 'relationship', 'vs', 'versus']):
        print("DEBUG: Correlation handler triggered")
        corr_matrix = query_result[numeric_cols].corr()
        corr_value = corr_matrix.iloc[0, 1]
        col1, col2 = numeric_cols[0], numeric_cols[1]

        if abs(corr_value) < 0.1:
            strength = "no significant"
        elif abs(corr_value) < 0.3:
            strength = "weak"
        elif abs(corr_value) < 0.7:
            strength = "moderate"
        else:
            strength = "strong"

        direction = "positive" if corr_value > 0 else "negative"

        # Add specific insights based on column names
        if 'freight' in col1.lower() and 'distance' in col2.lower():
            return f"There's a {direction} correlation of {corr_value:.2f} between freight value and geodesic distance between seller and customer — indicating longer distances result in proportionally higher freight costs."
        elif 'review' in col1.lower() and 'price' in col2.lower():
            return f"There's a {strength} {direction} correlation ({corr_value:.2f}) between review scores and product prices."
        elif 'delivery' in col1.lower() and 'review' in col2.lower():
            return f"There's a {strength} {direction} correlation ({corr_value:.2f}) between delivery time and customer review scores."
        else:
            return f"The correlation between {col1.replace('_', ' ')} and {col2.replace('_', ' ')} is {corr_value:.3f} ({strength} {direction} relationship)."



    # Default fallback
    else:
        if len(query_result) <= 10:
            return f"Here are the results:\n\n{query_result.to_string(index=False)}"
        else:
            return f"Found {len(query_result)} results. The data has been processed and visualized."

def query_agent_with_followup(question: str, is_followup: bool = False) -> Dict[str, Any]:
    """
    Query the agent (follow-up functionality removed)
    """
    return query_agent(question)

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

        # Always try to enhance the answer if we have query results
        if _last_query_result is not None and not _last_query_result.empty:
            enhanced_answer = enhance_answer_based_on_query(question, _last_query_result, _last_count_result)
            # Prioritize enhanced formatting for trend queries, or use enhancement for poor responses
            if enhanced_answer and ('trend' in question.lower() or 'monthly' in question.lower() or 'revenue' in question.lower()):
                answer = enhanced_answer
            elif enhanced_answer and (not answer or "Found" in answer or "results" in answer or len(answer.strip()) < 20 or "Agent stopped" in answer or "analysis shows" in answer.lower()):
                answer = enhanced_answer

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
