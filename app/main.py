import os
import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, List
from qdrant_client.http.models import VectorParams

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from dotenv import load_dotenv

# --- LlamaIndex Core Imports ---
from llama_index.core import (
    Settings,
    Document,
    VectorStoreIndex
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# --- LlamaIndex Integrations ---
from llama_index.readers.llama_parse import LlamaParse
# --- ⬇️ GEMINI IMPORTS ⬇️ ---
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
# --- ⬆️ GEMINI IMPORTS ⬆️ ---
from llama_index.vector_stores.qdrant import QdrantVectorStore

import qdrant_client

load_dotenv()

if os.getenv("GOOGLE_API_KEY") is None:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env file")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "financial_reports"

qdrant_client = qdrant_client.QdrantClient(url=QDRANT_URL)

parser = LlamaParse(
    api_key=os.getenv("LLAMA_PARSE_KEY"),
    result_type="markdown",
    num_workers=2,
    verbose=True
)

Settings.llm = GoogleGenAI(model_name="gemini-1.5-pro-latest")
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")
Settings.node_parser = MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True)


app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """On startup, ensure the Qdrant collection exists."""
    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION)
        print(f"Collection '{QDRANT_COLLECTION}' already exists.")
    except Exception:
        print(f"Collection '{QDRANT_COLLECTION}' not found. Creating...")
        qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=768,
                distance="Cosine"
            )
        )
        print(f"Collection '{QDRANT_COLLECTION}' created.")


async def index_document_pipeline(
    file_bytes: bytes,
    file_name: str,
    company_name: str,
    report_type: str,
    fiscal_year: int
):
    temp_pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file_bytes)
            temp_pdf_path = temp_pdf.name

        print(f"Starting LlamaParse for {file_name}...")
        docs = await parser.aload_data(temp_pdf_path)
        print("LlamaParse complete.")
        
        doc_id = str(uuid.uuid4())
        for doc in docs:
            doc.metadata = {
                "company_name": company_name,
                "report_type": report_type,
                "fiscal_year": fiscal_year,
                "filename": file_name,
                "doc_id": doc_id,
            }
            doc.id_ = doc_id

        print("Chunking documents into nodes...")
        nodes = await asyncio.to_thread(Settings.node_parser.get_nodes_from_documents, docs)
        print(f"Created {len(nodes)} nodes.")

        print("Connecting to vector store and indexing nodes...")
        print("Creating QdrantVectorStore...")
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=QDRANT_COLLECTION
        )
        print("Created QdrantVectorStore.")

        print("Creating VectorStoreIndex...")
        index = await asyncio.to_thread(
            VectorStoreIndex.from_vector_store,
            vector_store=vector_store,
        )
        print("Created VectorStoreIndex.")

        print("Inserting nodes...")
        await asyncio.to_thread(index.insert_nodes, nodes)
        print("Inserted nodes.")
        
        print(f"Successfully indexed {len(nodes)} nodes for doc_id {doc_id}.")
        return doc_id, len(nodes)

    except Exception as e:
        print(f"Error in pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)


@app.post("/index_document/")
async def index_document(
    pdf_file: Annotated[UploadFile, File(...)],
    company_name: Annotated[str, Form(...)],
    report_type: Annotated[str, Form(...)],
    fiscal_year: Annotated[int, Form(...)]
):
    if pdf_file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    
    pdf_bytes = await pdf_file.read()
    
    doc_id, nodes_added = await index_document_pipeline(
        file_bytes=pdf_bytes,
        file_name=pdf_file.filename,
        company_name=company_name,
        report_type=report_type,
        fiscal_year=fiscal_year
    )
    
    return {
        "message": "Document indexed successfully",
        "filename": pdf_file.filename,
        "doc_id": doc_id,
        "nodes_added": nodes_added
    }

class QueryRequest(BaseModel):
    query: str
    company_name: str | None = None
    fiscal_year: int | None = None
    report_type: str | None = None

# ...existing code...

from llama_index.llms.google_genai import GoogleGenAI

@app.post("/query/")
async def query_index(request: QueryRequest):
    print(f"Received query: {request.query}")
    
    filters = []
    if request.company_name:
        filters.append(ExactMatchFilter(key="company_name", value=request.company_name))
    if request.fiscal_year:
        filters.append(ExactMatchFilter(key="fiscal_year", value=request.fiscal_year))
    if request.report_type:
        filters.append(ExactMatchFilter(key="report_type", value=request.report_type))
    
    metadata_filters = MetadataFilters(filters=filters)
    print(f"Applying filters: {metadata_filters}")

    try:
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=QDRANT_COLLECTION
        )
        
        index = await asyncio.to_thread(
            VectorStoreIndex.from_vector_store,
            vector_store=vector_store
        )
        
        query_engine = index.as_query_engine(
            filters=metadata_filters,
            similarity_top_k=5,
            response_mode="compact"
        )

        # Get raw response from vector DB
        raw_response = await asyncio.to_thread(query_engine.query, request.query)
        raw_text = str(raw_response)

        # Use Gemini to make the response readable
        gemini_llm = GoogleGenAI(model_name="gemini-1.5-pro-latest")
        prompt = (
            "You are a financial analyst assistant. Please read the following excerpt from a financial report and rewrite it for clarity and readability. "
            "Summarize key points, highlight important figures, and present the information in a way that is easy for a non-expert to understand. "
            "Use bullet points or short paragraphs where helpful. Here is the excerpt:\n\n"
            f"{raw_text}"
        )
        readable_response = await asyncio.to_thread(gemini_llm.complete, prompt)

        return {
            "query": request.query,
            "response": readable_response.text
        }

    except Exception as e:
        print(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

# ...existing code...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)