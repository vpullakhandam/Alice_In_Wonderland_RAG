# Alice in Wonderland Q&A

A Streamlit application that uses RAG (Retrieval Augmented Generation) to answer questions about Alice in Wonderland.

## Features

- Question answering grounded in the original text
- Query expansion for better search results
- Reranking of retrieved passages
- Source citations for transparency
- Interactive UI with sample questions

## Project Structure

```
Alice_In_Wonderland_RAG/
├── Code Files/
│   ├── app.py          # Streamlit application
│   └── notebook.ipynb  # Knowledge base creation notebook
├── SourceFile/
│   └── alice_in_wonderland.md  # Source text
├── chroma_store/       # Vector database
├── .streamlit/        # Streamlit configuration
└── requirements.txt    # Python dependencies
```

## Setup for Local Development

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Alice_In_Wonderland_RAG.git
   cd Alice_In_Wonderland_RAG
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   - Create a `.env` file with:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```
   - Or set in Streamlit Cloud secrets

5. Run the app:
   ```bash
   streamlit run "Code Files/app.py"
   ```

## Deployment on Streamlit Cloud

1. Fork/push this repository to GitHub

2. Connect to Streamlit Cloud:

   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select this repository and branch
   - Set the path to: `Code Files/app.py`

3. Add secrets:

   - In your app's settings on Streamlit Cloud
   - Go to "Secrets"
   - Add your `GEMINI_API_KEY`

4. Advanced Settings:
   - Python version: 3.10
   - Packages: requirements.txt will be automatically detected

## Dependencies

- Python 3.10+
- Streamlit
- LangChain
- ChromaDB
- Google Gemini API
- See requirements.txt for full list

## Notes

- The knowledge base is built from the original text using BGE embeddings
- Uses Gemini for answer generation
- Includes cross-encoder reranking for better results
- SQLite compatibility layer for Streamlit Cloud deployment

## Troubleshooting

1. If you get SQLite errors on Streamlit Cloud:

   - The app includes a SQLite compatibility shim
   - No action needed, it's handled automatically

2. If the knowledge base fails to load:

   - Check that the chroma_store directory is properly pushed to GitHub
   - Ensure all binary files are included

3. If you get memory errors:
   - The app is optimized for Streamlit Cloud's free tier
   - Reduce batch sizes if needed

