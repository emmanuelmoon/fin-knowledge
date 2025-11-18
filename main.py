import os
import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from dotenv import load_dotenv

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.readers.llama_parse import LlamaParse
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams

load_dotenv()

if os.getenv("GOOGLE_API_KEY") is None:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env file")

QDRANT_COLLECTION = "financial_reports"

# Initialize Qdrant client
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"),
                      api_key=os.getenv("QDRANT_API_KEY"))  # renamed from qdrant_client to avoid conflicts

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
    """Ensure Qdrant collection and field indexes exist on startup."""
    try:
        qdrant.get_collection(collection_name=QDRANT_COLLECTION)
        print(f"Collection '{QDRANT_COLLECTION}' already exists.")
    except Exception:
        print(f"Collection '{QDRANT_COLLECTION}' not found. Creating...")
        qdrant.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=768, distance="Cosine")
        )
        print(f"Collection '{QDRANT_COLLECTION}' created.")

    # Create field indexes for filtering
    try:
        qdrant.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="company_name",
            field_type="keyword"
        )
        qdrant.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="report_type",
            field_type="keyword"
        )
        qdrant.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="fiscal_year",
            field_type="integer"
        )

        print("Field indexes created.")
    except Exception as e:
        print(f"Field index creation error: {e}")


async def index_document_pipeline(file_bytes: bytes, file_name: str, company_name: str, report_type: str, fiscal_year: int):
    temp_pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file_bytes)
            temp_pdf_path = temp_pdf.name

        docs = await parser.aload_data(temp_pdf_path)

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

        nodes = await asyncio.to_thread(Settings.node_parser.get_nodes_from_documents, docs)

        vector_store = QdrantVectorStore(
            client=qdrant,
            collection_name=QDRANT_COLLECTION
        )

        index = await asyncio.to_thread(VectorStoreIndex.from_vector_store, vector_store)
        await asyncio.to_thread(index.insert_nodes, nodes)

        return doc_id, len(nodes)

    except Exception as e:
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


@app.post("/query/")
async def query_index(request: QueryRequest):
    filters = []
    if request.company_name:
        filters.append(ExactMatchFilter(key="company_name", value=request.company_name))
    if request.fiscal_year:
        filters.append(ExactMatchFilter(key="fiscal_year", value=request.fiscal_year))
    if request.report_type:
        filters.append(ExactMatchFilter(key="report_type", value=request.report_type))
    
    metadata_filters = MetadataFilters(filters=filters)

    try:
        vector_store = QdrantVectorStore(
            client=qdrant,
            collection_name=QDRANT_COLLECTION
        )
        
        index = await asyncio.to_thread(VectorStoreIndex.from_vector_store, vector_store)
        
        query_engine = index.as_query_engine(
            filters=metadata_filters,
            similarity_top_k=5,
            response_mode="compact"
        )

        raw_response = await asyncio.to_thread(query_engine.query, request.query)
        raw_text = str(raw_response)

        gemini_llm = GoogleGenAI(model_name="gemini-1.5-pro-latest")
        prompt = (
            "You are a financial analyst assistant. Rewrite the following excerpt for clarity and readability, "
            "summarize key points, highlight important figures, and make it easy for a non-expert. "
            "Here is the excerpt:\n\n"
            f"{raw_text}"
        )
        readable_response = await asyncio.to_thread(gemini_llm.complete, prompt)

        return {
            "query": request.query,
            "response": readable_response.text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
