from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Annotated, List
from llama_index.readers.llama_parse import LlamaParse
import asyncio
import os
from pdf2image import convert_from_bytes
from pathlib import Path
from dotenv import load_dotenv

app = FastAPI()

load_dotenv() 

# --- Configuration ---
# We use a persistent directory so the images can be retrieved
# by the RAG system later. A temp dir would be deleted.
IMAGE_OUTPUT_DIR = Path("static/page_images")
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the parser once
parser = LlamaParse(
    api_key=os.getenv("LLAMA_PARSE_KEY"), # Your API key
    result_type="markdown",
    num_workers=1  # <-- Fix for 429 rate-limiting
)
# --- End Configuration ---


@app.post("/upload_pdf_multimodal/")
async def upload_pdf_multimodal(pdf_file: Annotated[UploadFile, File(...)]):
    
    if pdf_file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
        
    pdf_bytes = await pdf_file.read()

    # --- 1. Convert PDF to a list of PIL Images ---
    def convert_pdf_to_images(data):
        try:
            return convert_from_bytes(data)
        except Exception as e:
            return None

    images = await asyncio.to_thread(convert_pdf_to_images, pdf_bytes)
    
    if images is None:
        raise HTTPException(status_code=500, detail="Failed to convert PDF to images. Is 'poppler' installed?")

    # --- 2. Save images to our persistent directory ---
    safe_filename = Path(pdf_file.filename).stem
    page_image_dir = IMAGE_OUTPUT_DIR / safe_filename
    page_image_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    save_tasks = []

    def save_image(img, path):
        img.save(path, "PNG")

    for i, img in enumerate(images):
        page_num = i + 1
        # Use an OS-independent path
        image_path = page_image_dir / f"page_{page_num}.png"
        
        # Store the path as a string
        image_paths.append(str(image_path)) 
        
        save_tasks.append(asyncio.to_thread(save_image, img, image_path))
    
    await asyncio.gather(*save_tasks) # Wait for all images to be saved


    # --- 3. Parse the images with LlamaParse ---
    try:
        docs = await asyncio.to_thread(parser.load_data, image_paths)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LlamaParse API error: {e}")


    # --- 4. Return Structured Data (Using zip) ---
    #
    #    THIS IS THE FIX:
    #    We zip our known 'image_paths' list with the 'docs' list.
    #    They are guaranteed to be in the same order.
    #
    structured_data = []
    for img_path, doc in zip(image_paths, docs):
        structured_data.append({
            "source_image_path": img_path,  # <-- This is now correct and reliable
            "text_content": doc.text
        })

    if not structured_data:
        raise HTTPException(status_code=500, detail="LlamaParse returned no documents.")

    return {
        "filename": pdf_file.filename,
        "total_pages_processed": len(structured_data),
        "data": structured_data # This list is ready for Qdrant
    }