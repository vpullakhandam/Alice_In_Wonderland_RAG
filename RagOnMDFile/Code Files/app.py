import streamlit as st
import sys
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

# SQLite compatibility shim for Streamlit Cloud
try:
    import pysqlite3 as _sqlite3  # type: ignore
    sys.modules["sqlite3"] = _sqlite3
    sys.modules["_sqlite3"] = _sqlite3
except Exception:
    pass

# Handle different Chroma imports
try:
    from langchain_chroma import Chroma as ChromaStore
    CHROMA_DEPRECATED = False
except Exception:
    from langchain_community.vectorstores import Chroma as ChromaStore
    CHROMA_DEPRECATED = True

from langchain_huggingface import HuggingFaceEmbeddings
from FlagEmbedding import FlagReranker

# Pydantic models for structured output
class Expansions(BaseModel):
    items: List[str] = Field(..., min_items=1, description="List of paraphrased questions")
    
    @field_validator('items')
    @classmethod
    def validate_items(cls, v):
        if not all(isinstance(item, str) and len(item.strip()) > 0 for item in v):
            raise ValueError("All items must be non-empty strings")
        return [item.strip() for item in v]

class RetrievedMetadata(BaseModel):
    source: str
    chapter_number: Optional[str] = None
    chapter_title: Optional[str] = None
    position: Optional[int] = Field(None, alias="start_index")
    chunk_number: Optional[int] = None
    chunk_summary: Optional[str] = None

class RetrievedChunk(BaseModel):
    question: str
    score: float
    content: str
    metadata: RetrievedMetadata

class RetrievalResults(BaseModel):
    results: List[RetrievedChunk]

class RerankedDocument(BaseModel):
    content: str
    metadata: RetrievedMetadata
    rerank_score: float
    
class RerankedResults(BaseModel):
    results: List[RerankedDocument]
    original_question: str
    model_name: str = "BAAI/bge-reranker-base"

class Citation(BaseModel):
    source: str
    chapter_number: Optional[str] = None
    chapter_title: Optional[str] = None
    position: Optional[int] = Field(None, alias="start_index")
    chunk_number: Optional[int] = None

class GeminiResponse(BaseModel):
    answer: str = Field(..., description="The answer from Gemini")
    citations: List[Citation] = Field(default_factory=list, description="Citations from the context")
    context_headers: List[str] = Field(default_factory=list, description="Headers from the context")

# Setup page config
st.set_page_config(
    page_title="Alice in Wonderland Q&A",
    page_icon="🎩",
    layout="wide"
)

# Constants
PERSIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "chroma_store"))
COLLECTION = "alice"  # Make sure this matches the collection name from notebook.ipynb

def get_api_key_from_secrets() -> str:
    return st.secrets.get("GEMINI_API_KEY", "")

def get_api_key_from_env() -> str:
    return os.getenv("GEMINI_API_KEY", "")

def check_api_key():
    """Check if API key is properly set (prefer Streamlit Secrets by default)."""
    # Try Streamlit Secrets first
    api_key = get_api_key_from_secrets()
    
    # Fallback to environment variable
    if not api_key:
        api_key = get_api_key_from_env()

    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found. Add it in Settings → Secrets (Cloud) or .env (local).")
        return False
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    return True

@st.cache_resource
def load_vectordb():
    """Load the persisted Chroma DB built in the notebook using BGE embeddings."""
    try:
        # Check directory and SQLite file
        if not os.path.isdir(PERSIST_DIR):
            st.error(f"❌ Chroma store directory not found")
            return None
            
        sqlite_path = os.path.join(PERSIST_DIR, "chroma.sqlite3")
        if not os.path.exists(sqlite_path):
            st.error(f"❌ Chroma SQLite DB not found")
            return None
            
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize Chroma
        vectordb = ChromaStore(
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION,
        )
        
        # Verify collection
        try:
            count = vectordb._collection.count()
            if count == 0:
                st.warning("⚠️ Knowledge base is empty")
                return None
        except Exception as e:
            st.error(f"❌ Failed to verify collection: {str(e)}")
            return None
            
        if CHROMA_DEPRECATED:
            st.warning("⚠️ Using deprecated Chroma import. Consider installing 'langchain-chroma'.")
            
        return vectordb
        
    except Exception as e:
        st.error(f"❌ Error loading Chroma DB: {str(e)}")
        # Print more detailed error info
        import traceback
        st.error(f"Detailed error:\n```\n{traceback.format_exc()}\n```")
        return None

def expand_queries(llm: ChatGoogleGenerativeAI, question: str, n: int = 4) -> List[str]:
    """Generate query expansions using Gemini."""
    prompt = f"""
    Generate exactly {n} diverse paraphrases of the question below.
    Your response should be ONLY a valid JSON object with this exact format:
    {{"items": ["paraphrase1", "paraphrase2", ...]}}

    Question: {question}

    Remember: Return ONLY the JSON object, no other text.
    """
    response = llm.invoke(prompt)
    try:
        json_str = response.content.strip()
        start = json_str.find('{')
        end = json_str.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = json_str[start:end]
        result = json.loads(json_str)
        
        expansions = Expansions(items=result.get('items', []))
        items = expansions.items[:n] if len(expansions.items) > n else expansions.items + [question] * (n - len(expansions.items))
        return items
    except Exception:
        return [question] * n

def retrieve_candidates(vectordb, queries, per_query_k: int = 5) -> RetrievalResults:
    """Retrieve candidates with structured output."""
    results = []
    seen = set()
    
    # Ensure we have exactly 4 queries
    queries = queries[:4] if len(queries) > 4 else queries + [queries[0]] * (4 - len(queries))
    
    for q in queries:
        hits = vectordb.similarity_search_with_score(q, k=per_query_k)
        for doc, score in hits:
            key = (doc.metadata.get("start_index"), doc.metadata.get("chunk_number"))
            if key not in seen:
                seen.add(key)
                chunk = RetrievedChunk(
                    question=q,
                    score=float(score),
                    content=doc.page_content,
                    metadata=RetrievedMetadata(**doc.metadata)
                )
                results.append(chunk)
    
    return RetrievalResults(results=results)

def rerank_candidates(question: str, candidates: RetrievalResults, top_n: int = 8) -> RerankedResults:
    """Rerank candidates with structured output."""
    reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
    pairs = [[question, doc.content] for doc in candidates.results]
    scores = reranker.compute_score(pairs)
    reranked = sorted(zip(candidates.results, scores), key=lambda x: x[1], reverse=True)
    
    seen = set()
    top_docs = []
    for doc, rerank_score in reranked:
        sid = doc.metadata.position
        if sid not in seen:
            seen.add(sid)
            reranked_doc = RerankedDocument(
                content=doc.content,
                metadata=doc.metadata,
                rerank_score=float(rerank_score)
            )
            top_docs.append(reranked_doc)
        if len(top_docs) == top_n:
            break
    
    return RerankedResults(results=top_docs, original_question=question)

def build_context(docs: RerankedResults) -> tuple[str, List[Citation], List[str]]:
    """Build context with structured output."""
    parts = []
    citations = []
    headers = []
    
    for doc in docs.results:
        m = doc.metadata
        header = f"[Position: {m.position} | Chunk: {m.chunk_number}]"
        parts.append(header + "\n" + doc.content)
        headers.append(header)
        
        citations.append(Citation(
            source=m.source,
            chapter_number=m.chapter_number,
            chapter_title=m.chapter_title,
            start_index=m.position,
            chunk_number=m.chunk_number
        ))
    
    return "\n\n".join(parts), citations, headers

def get_answer(vectordb, query: str) -> GeminiResponse:
    """Get answer with structured output."""
    try:
        # Initialize Gemini
        llm_gemini = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        # 1. Query Expansion
        st.write("🔄 Generating query variations...")
        expansions = expand_queries(llm_gemini, query)
        queries = [query] + expansions
        st.write(f"Generated {len(expansions)} variations")

        # 2. Initial Retrieval
        st.write("🔍 Searching knowledge base...")
        candidates = retrieve_candidates(vectordb, queries, per_query_k=5)
        if not candidates.results:
            return GeminiResponse(
                answer="I couldn't find any relevant information in the book about that.",
                citations=[],
                context_headers=[]
            )

        # 3. Reranking
        st.write("📊 Reranking results...")
        top_docs = rerank_candidates(query, candidates, top_n=8)
        if not top_docs.results:
            return GeminiResponse(
                answer="I found some matches but couldn't rank them properly.",
                citations=[],
                context_headers=[]
            )

        # 4. Context Building
        context, citations, headers = build_context(top_docs)
        
        # 5. Answer Generation
        st.write("💭 Generating answer...")
        prompt = PromptTemplate.from_template("""
        You are a helpful assistant answering questions about Alice in Wonderland.
        Answer the user question using ONLY the provided context.
        Read the chunk summary carefully and if it matches with the question then check the chunk content and answer the question.
        Expand the answer into at least 2–3 sentences and don't use quotes from the content unless the question is asking for the quotes.

        Question: {question}
        Context:
        {context}
        """)

        response = llm_gemini.invoke(prompt.format(question=query, context=context))
        
        return GeminiResponse(
            answer=response.content,
            citations=citations,
            context_headers=headers
        )
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return GeminiResponse(
            answer="I encountered an error while processing your question.",
            citations=[],
            context_headers=[]
        )

def main():
    st.title("🎩 Alice in Wonderland Q&A")
    st.markdown(
        "Discover answers grounded in the original text of Alice in Wonderland.\n\n"
        "Ask a question in the box below. The app searches a persistent knowledge base built from the book, "
        "expands your query, reranks the most relevant passages, and answers using only the retrieved context."
    )

    # Check API key first
    if not check_api_key():
        return

    # Load persisted vector DB
    with st.spinner("Loading knowledge base..."):
        vectordb = load_vectordb()
        if vectordb is None:
            return

    # Sidebar with examples and settings
    with st.sidebar:
        st.header("How to use")
        st.markdown(
            "- Type your question or try a sample\n"
            "- View sources to see relevant passages\n"
            "- Results are based on the original text"
        )

        st.subheader("Settings")
        show_sources = st.checkbox("Show source passages", value=True)

        st.subheader("Sample Questions")
        examples = [
            "Why did Alice follow the White Rabbit?",
            "What happens at the Mad Hatter's tea party?",
            "How does the Cheshire Cat disappear?",
            "What does Alice find in the bottle labeled 'DRINK ME'?",
        ]
        if "user_query" not in st.session_state:
            st.session_state.user_query = ""
        for ex in examples:
            if st.button(ex):
                st.session_state.user_query = ex

    # Main query input
    query = st.text_input(
        "Ask your question",
        key="user_query",
        placeholder="e.g., Why did Alice follow the White Rabbit?"
    )

    if query:
        with st.spinner("Searching and thinking..."):
            response = get_answer(vectordb, query)
            
        st.markdown("### Answer")
        st.write(response.answer)

        if show_sources and response.citations:
            with st.expander("📚 Source Passages", expanded=False):
                for citation, header in zip(response.citations, response.context_headers):
                    st.markdown(f"**{header}**")
                    st.markdown("---")

if __name__ == "__main__":
    main()
