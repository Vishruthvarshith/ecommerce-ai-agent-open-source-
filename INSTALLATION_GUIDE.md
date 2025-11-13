# Installation Guide

Follow these steps to install the AI chatbot.

## Step 1: Install Ollama

1. Go to https://ollama.com/download
2. Download and install for your computer
3. Open terminal and run: `ollama pull llama3.1:8b`

## Step 2: Get Data

1. Visit: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/
2. Click Download
3. Unzip the file
4. Make a folder named `data`
5. Put all the CSV files in the `data` folder

## Step 3: Install Python Packages

1. venv setup
    ```bash
    python -m venv .venv
    source venv/bin/activate # Linux/MacOS
    source venv\Scripts\Activate.ps1 # Windows PowerShell (please check once for the command appropriate to your system)
    ``` 
2. Make sure Python 3.9+  to 3.11.10 (not anything higher) is installed
3. Run: `pip install -r requirements.txt`

## Step 4: Setup Database

1. Run: `python data_setup.py`
2. Should see "Database created successfully"

## Step 5: Start the App

1. Terminal 1: `ollama serve`
2. Terminal 2: `streamlit run app.py`
3. Open browser to http://localhost:8501

## Test

Ask: "How many orders are there?"

## Problems?

- Restart terminal if commands not found
- Make sure both terminals are running
- Check that data files are in `data/` folder
