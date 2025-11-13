# Quick Start

Get the AI chatbot running in 5 minutes.

## What You Need First

- Computer with Python 3.9 or newer
- 8GB of memory or more
- Internet for downloading

## Step 1: Install the AI Brain

1. Open terminal
2. Run: `curl -fsSL https://ollama.com/install.sh | sh`
3. Run: `ollama pull llama3.1:8b`

## Step 2: Get the Shopping Data

1. Go to: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/
2. Click Download
3. Unzip the file
4. Create a folder called `data`
5. Put all CSV files in the `data` folder

## Step 3: Install Python Stuff

1. Run: `pip install -r requirements.txt`
2. Run: `python data_setup.py`

## Step 4: Start Everything

1. Open new terminal, run: `ollama serve` or keep the ollama Desktop GUI open and select the correct model. (llama3.1:8b)
2. In first terminal, run: `streamlit run app.py`
3. Open browser to http://localhost:8501

## Test It

Ask: "How many orders are there?"

## Problems?

- If no answers: Make sure both terminals are running
- If slow: Try simpler questions first
