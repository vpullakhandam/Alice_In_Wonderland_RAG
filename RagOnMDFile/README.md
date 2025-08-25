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
RagOnMDFile/
├── Code Files/
│   ├── app.py          # Streamlit application
│   └── notebook.ipynb  # Knowledge base creation notebook
├── SourceFile/
│   └── alice_in_wonderland.md  # Source text
├── chroma_store/       # Vector database
├── requirements.txt    # Python dependencies
└── .streamlit/        # Streamlit configuration
```

## Setup for Local Development

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:

   - Create a `.env` file with:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```
   - Or set in Streamlit Cloud secrets

3. Run the app:
   ```bash
   streamlit run Code\ Files/app.py
   ```

## Deployment on Streamlit Cloud

1. Fork/push this repository to GitHub
2. Connect to Streamlit Cloud
3. Add your `GEMINI_API_KEY` to Streamlit secrets
4. Deploy!

## Dependencies

- Python 3.8+
- See requirements.txt for full list

## Notes

- The knowledge base is built from the original text using BGE embeddings
- Uses Gemini for answer generation
- Includes cross-encoder reranking for better results
