AI Chatbot (RAG + Memory)

This is a simple chatbot built using Streamlit that can answer questions from your own documents. It uses RAG (retrieval augmented generation) so it pulls relevant info from whatever files you upload, and it also remembers the conversation as you go. Backend model used is Google Gemini.

No database, no separate server setup needed - everything runs in one Python file.

What it does


Chat normally with the AI (uses Gemini, gemini-2.5-flash by default)
Upload documents (pdf, docx, xlsx, csv, pptx, txt, md etc) and ask questions about them
Keeps memory of the conversation so follow up questions work properly
If you upload multiple files, it makes sure not just one file gets picked every time - tries to pull info from all of them
Settings in sidebar to control chunk size, overlap, how many chunks to retrieve etc
No vector database used, just a basic TF-IDF + cosine similarity search written in plain python


Files in this repo

rag_app.py          - the main app file
requirements.txt     - dependencies
.env.example          - sample env file for api key
README.md

How to run it locally

First clone this repo:

bashgit clone https://github.com/your-username/your-repo.git
cd your-repo

Make a virtual environment (not compulsory but good practice):

bashpython -m venv venv
source venv/bin/activate
# on windows: venv\Scripts\activate

Install the requirements:

bashpip install -r requirements.txt

You'll need a Gemini API key, get one free from Google AI Studio (aistudio.google.com). Once you have it, make a .env file in the project folder:

GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash

(You can also just paste your key directly into the app sidebar if you don't want to use a .env file)

Now run it:

bashstreamlit run rag_app.py

It'll open at localhost:8501 in your browser.

How the RAG part works

Basically when you upload a doc, it gets split into chunks (around 400 words each, with some overlap so context isn't lost between chunks). Each chunk gets a TF-IDF score. When you ask something, your question also gets converted into the same kind of vector and compared against all chunks using cosine similarity - whichever chunks score highest get pulled in and given to Gemini as context before it answers.

If you've added more than one document, the app makes sure every document gets at least one chunk represented in the results instead of letting one big document dominate everything.

Memory

Every message you send and every reply gets stored in the session, and the whole history gets sent back to Gemini each time so it has context of what was said before. This is only stored in memory though (Streamlit session state) - so if you refresh or close the tab, it's gone. Nothing is saved permanently right now.

Settings you can change

In the sidebar there's a settings panel where you can adjust:


chunk size
overlap between chunks
how many chunks get retrieved (top-k)
whether to show sources/citations
debug mode to see which chunks were actually picked


Deploying

Easiest way is Streamlit Community Cloud, it's free:


Push your code to GitHub
Go to share.streamlit.io and login with GitHub
Click new app, select your repo, set main file as rag_app.py
In advanced settings add your GEMINI_API_KEY as a secret
Deploy


Note: if deploying here, you might need to change the api key line in the code to also check st.secrets, since .env files don't really work the same way on Streamlit cloud.

You can also use Hugging Face Spaces (just rename rag_app.py to app.py) or any VPS/Railway type platform with a basic Procfile.

Pushing this to GitHub

bashgit init
echo ".env" >> .gitignore
git add .
git commit -m "initial commit"
git remote add origin https://github.com/your-username/your-repo.git
git branch -M main
git push -u origin main

Just don't commit your .env file or api key anywhere, keep that out of version control.

A few notes


Nothing here gets saved to disk, it's all in-memory for the session
Be a bit careful what documents you upload since the content does get sent to Gemini's API for processing
This was mostly a learning project to understand how RAG pipelines work without relying on heavier tools like LangChain or a proper vector DB


Credits

Built using Streamlit, Google Gemini API, and a few python libraries for reading files (pypdf, python-docx, openpyxl, python-pptx).
