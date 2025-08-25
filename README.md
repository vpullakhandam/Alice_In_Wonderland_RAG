# Alice in Wonderland Q&A

A Streamlit application that uses RAG (Retrieval Augmented Generation) to answer questions about Alice in Wonderland. The application implements robust data validation and structured outputs using Pydantic models.

## Features

- Question answering grounded in the original text
- Query expansion for better search results using Gemini
- Reranking of retrieved passages with BAAI/bge-reranker-base
- Source citations for transparency
- Interactive UI with sample questions
- Structured data validation with Pydantic models
- Multi-query expansion with JSON schema enforcement
- Token-aware text chunking with metadata preservation

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

## Data Models

The project uses Pydantic models for data validation and structure:

- `ChunkMetadata`: Validates chunk metadata (source, chapter info, position)
- `ChunkDoc`: Represents a document chunk with content and metadata
- `Expansions`: Validates query expansion results
- `RetrievedChunk`: Structures retrieved passages with scores
- `RerankedDocument`: Represents reranked documents with scores
- `GeminiResponse`: Structures final answers with citations

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
     OPENROUTER_API_KEY=your_openrouter_key_here
     HUGGINGFACE_TOKEN=your_huggingface_token_here
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
   - Add your API keys:
     - `GEMINI_API_KEY`
     - `OPENROUTER_API_KEY`
     - `HUGGINGFACE_TOKEN`

4. Advanced Settings:
   - Python version: 3.10
   - Packages: requirements.txt will be automatically detected

## Technical Implementation

- **Text Processing**:

  - Token-aware text chunking with position tracking
  - Chapter metadata extraction and preservation
  - Structured chunk summarization using OpenRouter's GPT models

- **Embeddings & Retrieval**:

  - BGE base embeddings (BAAI/bge-base-en-v1.5)
  - Persistent Chroma vectorstore with cosine similarity
  - Multi-query expansion using Gemini
  - Cross-encoder reranking with BAAI/bge-reranker-base

- **Answer Generation**:
  - Gemini 2.5 Flash for final answer generation
  - Structured output with source citations
  - JSON schema enforcement for consistent responses

## Dependencies

- Python 3.10+
- Streamlit
- LangChain & LangChain Community
- ChromaDB
- Google Gemini API
- Pydantic
- OpenRouter API
- Hugging Face Transformers
- See requirements.txt for full list

## Notes

- The knowledge base is built from the original text using BGE embeddings
- Uses Gemini for answer generation and query expansion
- Includes cross-encoder reranking for better results
- SQLite compatibility layer for Streamlit Cloud deployment
- Robust data validation with Pydantic models
- Structured JSON outputs for consistency

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

4. If you encounter API authentication issues:
   - Verify all required API keys are properly set in .env or Streamlit secrets
   - Check API key permissions and quotas
   - Ensure environment variables are properly loaded
