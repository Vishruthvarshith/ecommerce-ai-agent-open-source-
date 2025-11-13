"""
E-Commerce AI Agent - Streamlit Web Interface
A conversational AI interface for analyzing Brazilian e-commerce data
"""

import streamlit as st
from agent import query_agent, provide_follow_up_analysis, MODEL_NAME
import os

# Page configuration
st.set_page_config(
    page_title="E-Commerce AI Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .example-button {
        margin: 0.2rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🛒 E-Commerce AI Analyst</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Powered by {MODEL_NAME.title().replace(":", " ")} - Ask questions about Brazilian e-commerce data in natural language</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📚 About")
    st.markdown("""
    This AI agent analyzes **100,000+ orders** from Brazilian e-commerce marketplaces (2016-2018).
    
    **Dataset includes:**
    - Orders and order items
    - Customer information
    - Product catalog
    - Reviews and ratings
    - Payment details
    - Seller information
    - Geographic data
    """)
    
    st.markdown("---")
    
    st.header("💡 Example Questions")
    st.markdown("*Click any question to try it:*")
    
    example_questions = [
        "What are the top 5 product categories by revenue?",
        "Show me the monthly order trends in 2017",
        "Which states have the highest average order values?",
        "What's the correlation between delivery time and review scores?",
        "Which sellers have the best customer ratings?",
        "What are the most common payment methods?",
        "Show me customer distribution by state",
        "What's the average review score by product category?",
        "How many orders were delivered late?",
        "What's the relationship between price and customer satisfaction?"
    ]
    
    for i, question in enumerate(example_questions):
        if st.button(question, key=f"example_{i}", use_container_width=True):
            st.session_state.selected_question = question
    
    st.markdown("---")
    
    st.header("🛠️ Technical Stack")
    st.markdown(f"""
    - **LLM:** {MODEL_NAME}
    - **Framework:** LangChain (ReAct)
    - **Database:** SQLite
    - **Visualization:** Plotly
    - **Interface:** Streamlit
    """)
    
    st.markdown("---")
    
    # System status check
    st.header("🔍 System Status")
    
    db_exists = os.path.exists("ecommerce.db")
    schema_exists = os.path.exists("schema_docs.txt")
    
    if db_exists:
        st.success("✓ Database ready")
    else:
        st.error("✗ Database not found")
        st.info("Run: `python data_setup.py`")
    
    if schema_exists:
        st.success("✓ Schema loaded")
    else:
        st.warning("⚠ Schema docs missing")
    
    # Clear chat button
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None



# Welcome message
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""
        👋 **Hello! I'm your E-Commerce AI Analyst.**
        
        I can help you analyze the Brazilian e-commerce dataset. Here's what I can do:
        
        - 📊 **Query data** with natural language (no SQL knowledge needed!)
        - 📈 **Create visualizations** (charts, graphs, plots)
        - 🔍 **Find insights** and patterns in the data
        - 📉 **Analyze trends** over time and across regions
        - 💡 **Answer questions** about customers, products, orders, and more
        
        **Try asking something like:**
        - "What are the top selling products?"
        - "Show me monthly sales trends"
        - "Which regions have the most customers?"
        
        *Select an example from the sidebar or type your own question below!*
        """)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display figure if exists
        if message.get("figure"):
            st.plotly_chart(message["figure"], use_container_width=True)
        
        # Display data table if exists and not too large
        if message.get("data") is not None and len(message["data"]) > 0:
            with st.expander("📋 View Raw Data"):
                st.dataframe(message["data"], use_container_width=True)

# Always show chat input at the bottom
user_input_chat = st.chat_input("Ask a question about the e-commerce data...")

# Handle example question selection
if st.session_state.selected_question:
    user_input = st.session_state.selected_question
    st.session_state.selected_question = None  # Reset
    process_query = True
elif user_input_chat:
    user_input = user_input_chat
    process_query = True
else:
    process_query = False

# Process user input
if process_query:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Display assistant response
    with st.chat_message("assistant"):
        # Regular query
        with st.spinner("🤔 Analyzing data..."):
            result = query_agent(user_input)

        # Clean and display answer (remove verbose LangChain output)
        answer = result["answer"]

        # Extract just the final answer if it contains verbose output
        if "Final Answer:" in answer:
            # Find the last "Final Answer:" and take everything after it
            parts = answer.split("Final Answer:")
            if len(parts) > 1:
                answer = parts[-1].strip()

        # Remove verbose artifacts and chain messages
        answer = answer.replace("> Finished chain.", "").replace("Observation:", "").replace("Thought:", "").strip()

        # Clean up any remaining artifacts
        lines = [line.strip() for line in answer.split('\n') if line.strip() and not line.startswith('>')]
        answer = '\n'.join(lines).strip()

        st.markdown(answer)

        # Display visualization if created
        if result.get("figure"):
            st.plotly_chart(result["figure"], use_container_width=True)

        # Display data table if available
        if result.get("data") is not None and len(result["data"]) > 0 and len(result["data"]) <= 100:
            with st.expander("📋 View Raw Data"):
                st.dataframe(result["data"], use_container_width=True)

        # Save assistant message (with cleaned answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "figure": result.get("figure"),
            "data": result.get("data")
        })

        # Show error alert if failed
        if not result["success"]:
            st.error("⚠️ An error occurred. Please check the error message above.")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🎓 Maersk AI/ML Intern Assignment**")

with col2:
    st.markdown("**🗄️ Dataset:** [Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/)")

with col3:
    st.markdown(f"**🤖 Model:** {MODEL_NAME}")
