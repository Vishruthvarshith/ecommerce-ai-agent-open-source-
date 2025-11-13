# System Architecture

## Overview

This document provides a detailed technical overview of the E-Commerce AI Analyst system architecture, designed for the Maersk AI/ML Campus Hiring Assignment.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                      │
│                     (Streamlit Web App)                      │
│  ┌────────────┐  ┌─────────────┐  ┌────────────────────┐   │
│  │ Chat Input │  │   Display   │  │   Visualization    │   │
│  │            │  │   Engine    │  │      Renderer      │   │
│  └────────────┘  └─────────────┘  └────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────────┐
│                   AGENT LAYER                                │
│              (LangChain ReAct Agent)                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            Llama 3.1 8B (via Ollama)                │    │
│  │  • Query Understanding    • Multi-step Reasoning    │    │
│  │  • Tool Selection         • Context Management      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────┐   │
│  │   Memory     │  │    Prompt     │  │  Agent State   │   │
│  │   Buffer     │  │   Template    │  │   Management   │   │
│  └──────────────┘  └───────────────┘  └────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   TOOL LAYER                                 │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────┐   │
│  │   SQL Tool   │  │  Analysis     │  │ Visualization  │   │
│  │              │  │     Tool      │  │      Tool      │   │
│  │ • Generate   │  │ • Statistics  │  │ • Chart Type   │   │
│  │   Queries    │  │ • Correlation │  │   Selection    │   │
│  │ • Execute    │  │ • Aggregation │  │ • Plotly Gen   │   │
│  │ • Format     │  │               │  │                │   │
│  └──────────────┘  └───────────────┘  └────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   DATA LAYER                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              SQLite Database                         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌────────────────────┐   │   │
│  │  │ orders   │ │customers │ │   order_items      │   │   │
│  │  └──────────┘ └──────────┘ └────────────────────┘   │   │
│  │  ┌──────────┐ ┌──────────┐ ┌────────────────────┐   │   │
│  │  │ products │ │ sellers  │ │   order_reviews    │   │   │
│  │  └──────────┘ └──────────┘ └────────────────────┘   │   │
│  │  ┌──────────┐ ┌──────────┐ ┌────────────────────┐   │   │
│  │  │ payments │ │ geolocat │ │   categories       │   │   │
│  │  └──────────┘ └──────────┘ └────────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. User Interface Layer (Streamlit)

**Purpose:** Provide intuitive chat-based interface for users

**Components:**
- **Chat Input:** Captures user queries in natural language
- **Display Engine:** Renders text responses, tables, and formatted data
- **Visualization Renderer:** Displays Plotly charts interactively
- **Sidebar:** Example queries and system status indicators

**Key Features:**
- Real-time chat interface
- Responsive design for different screen sizes
- Error handling with user-friendly messages
- Session state management for conversation continuity

### 2. Agent Layer (LangChain ReAct)

**Core Component:** Llama 3.1 8B model via Ollama

**Capabilities:**
- **Query Understanding:** Parses natural language into actionable intents
- **Multi-step Reasoning:** Breaks complex queries into sequential steps
- **Tool Selection:** Intelligently chooses appropriate tools for each task
- **Context Management:** Maintains conversation context across interactions

**Agent Architecture:**
- **Memory Buffer:** Stores conversation history and intermediate results
- **Prompt Template:** Structured prompts for consistent AI behavior
- **State Management:** Tracks agent execution state and tool usage

### 3. Tool Layer

**SQL Tool:**
- Generates SQL queries from natural language descriptions
- Validates query syntax and security
- Executes queries on SQLite database
- Formats results for display and further analysis

**Analysis Tool:**
- Statistical analysis (mean, median, correlation, trends)
- Data aggregation and summarization
- Works on previous query results
- Provides numerical insights and business metrics

**Visualization Tool:**
- Automatic chart type selection based on data characteristics
- Configurable via JSON specifications
- Creates interactive Plotly charts
- Supports bar charts, line charts, scatter plots, and pie charts

### 4. Data Layer (SQLite Database)

**Structure:** Normalized relational database with 9 interconnected tables

**Key Tables:**
- `orders` (99,441 rows): Order information and status
- `order_items` (112,650 rows): Items within each order
- `order_payments` (103,886 rows): Payment details and methods
- `order_reviews` (99,224 rows): Customer reviews and ratings
- `customers` (99,441 rows): Customer information and location
- `products` (32,951 rows): Product catalog with categories
- `sellers` (3,095 rows): Seller information and performance
- `geolocation` (1,000,163 rows): Geographic coordinates
- `product_category_name_translation` (71 rows): Category translations

**Performance:** Indexed for fast queries, ~85MB total size

---

## Technical Stack

### AI/ML Components
- **LLM:** Llama 3.1 8B (Meta AI, open-source)
- **Agent Framework:** LangChain 0.1.0 with ReAct pattern
- **Local Inference:** Ollama for CPU/GPU model serving

### Data Processing
- **Database:** SQLite 3.x (embedded, serverless)
- **Data Processing:** Pandas 2.1.4, NumPy
- **Visualization:** Plotly 5.18.0 (interactive charts)
- **ORM:** SQLAlchemy 2.0.25

### Web Interface
- **Framework:** Streamlit 1.29.0
- **Styling:** Built-in responsive design
- **State Management:** Streamlit session state

### Development
- **Language:** Python 3.9+ to 3.11.10
- **Environment:** Virtual environments (venv) 
- **Package Management:** pip with requirements.txt

---

## ReAct Agent Pattern Implementation

The system implements the **Reasoning + Acting (ReAct)** pattern:

### 1. Question Analysis
- Parse user query for intent and entities
- Identify required data sources and operations
- Determine appropriate tool sequence

### 2. Thought Process
- Plan execution strategy based on query complexity
- Select optimal tool combination
- Consider data dependencies and relationships

### 3. Action Execution
- Execute SQL queries for data retrieval
- Perform statistical analysis on results
- Generate visualizations when appropriate

### 4. Observation & Iteration
- Review tool outputs for completeness
- Identify need for additional analysis
- Repeat process if more information required

### 5. Final Response
- Synthesize results into coherent answer
- Format data professionally for business use
- Provide actionable insights and recommendations

---

## Performance Characteristics

### Response Times
- **Simple queries:** 2-5 seconds (e.g., "How many orders?")
- **Complex analysis:** 10-20 seconds (multi-step reasoning)
- **Data visualization:** 5-15 seconds (chart generation)

### Resource Usage
- **CPU:** 2-4 cores recommended for Ollama
- **RAM:** 8GB+ required, 16GB recommended
- **Storage:** ~5GB for Ollama models, ~85MB for database
- **Network:** Required for initial setup, optional for operation

### Scalability
- **Concurrent users:** 1-2 simultaneous sessions
- **Query complexity:** Handles complex multi-table joins
- **Data size:** Optimized for 100K+ order datasets

---

## Why Open-Source Architecture?

### Advantages Over Proprietary Solutions

**Cost Benefits:**
- **Zero API costs:** No usage-based pricing
- **No rate limits:** Unlimited queries locally
- **Free model access:** Llama 3.1 available at no cost

**Privacy & Security:**
- **Data stays local:** No external API calls
- **Full control:** Complete data sovereignty
- **No third-party dependencies:** Self-contained system

**Technical Advantages:**
- **Reproducible results:** Same model version always
- **Customizable:** Can modify prompts and behavior
- **Offline operation:** Works without internet after setup

### Performance Comparison

| Metric | Llama 3.1 8B | GPT-3.5 | GPT-4 |
|--------|--------------|---------|-------|
| Cost | $0 | $0.002/1K tokens | $0.03/1K tokens |
| Speed | 2-5s | 1-3s | 3-10s |
| Privacy | Local | Cloud | Cloud |
| SQL Quality | Excellent | Good | Excellent |

---

## Future Enhancements

### 1. Advanced Analytics
- Predictive modeling for sales forecasting
- Customer churn prediction
- Demand forecasting algorithms

### 2. Enhanced Visualizations
- Geographic heatmaps with geolocation data
- Interactive dashboards with drill-down capabilities
- Time series decomposition and trend analysis

### 3. Performance Optimization
- Query result caching for frequently asked questions
- Database indexing optimization
- Async processing for long-running queries

### 4. Multi-modal Capabilities
- Product image analysis integration
- Review sentiment analysis
- Text mining on customer feedback

### 5. Production Deployment
- Docker containerization
- Cloud deployment options (AWS/GCP)
- Monitoring and logging infrastructure

---

## Conclusion

This architecture provides a solid foundation for AI-powered data analysis:

- **Scalable:** Handles complex queries across multiple data sources
- **Extensible:** Modular design allows easy addition of new tools
- **Cost-effective:** Open-source stack eliminates API costs
- **Privacy-preserving:** All processing happens locally
- **Production-ready:** Clean code, error handling, and documentation

The combination of LangChain's ReAct pattern with Llama 3.1 demonstrates modern AI engineering practices while maintaining accessibility and cost-effectiveness.