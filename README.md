# E-Commerce AI Agent

This is a simple AI chatbot that answers questions about online shopping data.

## What It Does

- You type questions in normal English
- It shows you answers with charts and numbers
- No need to know computer code

## How To Get It Running

Follow these steps exactly:

### Step 1: Install Ollama

Ollama is free software that runs the AI brain.

1. Open your computer terminal
2. Copy and paste this command:
   ```
   curl -fsSL https://ollama.com/install.sh | sh
   ```
3. Wait for it to finish
4. Download the AI model:
   ```
   ollama pull llama3.1:8b
   ```
5. Check it worked:
   ```
   ollama list
   ```

### Step 2: Get the Shopping Data

1. Go to this website: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/
2. Click the big blue "Download" button
3. Wait for the ZIP file to download
4. Unzip the file (double-click it)
5. Create a folder called `data` in this project
6. Move all the CSV files from the unzipped folder into the `data` folder

### Step 3: Install Python Stuff

1. Make sure you have Python 3.9 or newer
2. In terminal, go to this project folder
3. Run this command:
   ```
   python -m venv env
   ```
4. Activate the virtual environment:
   - Windows: `source venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
5. Run this command:
   ```
   pip install -r requirements.txt
   ```

### If you face dependency issues (especially with langchain):

- Update pip first: `pip install --upgrade pip`
- Install langchain separately: `pip install langchain==0.1.0`
- If still issues, try: `pip install langchain langchain-community --upgrade`
- Or install all packages one by one:
  ```
  pip install streamlit pandas plotly sqlalchemy
  pip install langchain langchain-community
  ```

### Step 4: Setup the Database

1. Run this command:
   ```
   python data_setup.py
   ```
2. Wait for it to say "Database created successfully"
3. It will show all the .csv files loaded
### Step 5: Start the AI Brain

1. Open a new terminal window
2. Run this command:
   ```
   ollama serve
   or 
   open ollama using Desktop gui and make sure the model needed is installed and responding.
   IF NOT:
   ollama pull llama3.1:8b
   ollama run llama3.1:8b
   ```
3. Keep this window open - don't close it

### Step 6: Start the App

1. In the first terminal, run:
   ```
   streamlit run app.py
   ```
2. Your web browser will open automatically
3. The app is now running!

## How To Use It

1. Open the web page that appeared
2. Type your question in the chat box
3. Press Enter
4. Wait a few seconds
5. See the answer with charts

## Example Questions You Can Ask

- "What are the top 5 product categories?"
- "Show me sales by month in 2017"
- "Which states have the most customers?"
- "What's the average order value?"
- "How many orders are there total?"

## If Something Goes Wrong

### App won't start?
- Make sure you ran `ollama serve` in another terminal or open the Desktop gui and select the correct model.
- Then check to ensure you have not quit your ollama session
- Check that all steps were followed exactly

### No answers to questions?
- Try a simple question first: "How many orders are there?"
- Wait 1-5 seconds for the answer (This depends on your internet speed)

### Questions about the data?
- The data is from Brazilian online shopping (2016-2018)
- It has 100,000+ orders from many stores


### Submitted by:
- Name: Vishruth H V
- Phone: 9880004466
- Email: vishruthvasu@gmail.com
- LinkedIn: https://www.linkedin.com/in/vishruth-hv-16870326b/
- SRN: PES1UG22AM193
- Branch: CSE-AIML
- College: PES University
- Year of Study: Final Year

Thankyou For Giving Me This Opportunity To Work On This project. I hope I Can Contribute More Projects Like This In Future.


Video Link:  https://drive.google.com/file/d/1NJS-sOlHAUc0AXiC_HgKkQD5ZgtCUvcP/view?usp=sharing
