from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse
from sentence_transformers import SentenceTransformer
import whisper
import chromadb
from chromadb.config import Settings
from gtts import gTTS
import os
import shutil
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF for PDF parsing

app = FastAPI()

# Initialize models
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
stt_model = whisper.load_model("base")

# Setup Chroma vector store
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma"))
collection = chroma_client.get_or_create_collection(name="ciso_docs")

