# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import whisper, os, uvicorn
from gtts import gTTS
from TTS.api import TTS
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI()
# Load models and DB on startup for reuse
stt_model = whisper.load_model("base")
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", gpu=True)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection(name="ciso_docs")

class AskRequest(BaseModel):
    question: str

# Utility: speak_text
def speak_text(text, lang="en", out_path="response.mp3"):
    if lang == "ur":
        tts = gTTS(text=text, lang="ur")
        tts.save(out_path)
    else:
        tts_model.tts_to_file(text=text, speaker=tts_model.speakers[0],
                              language=lang, file_path=out_path)
    return out_path

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    content = await file.read()
    text = ""
    if ext in ("txt",):
        text = content.decode()
    else:
        return JSONResponse({"error": "Only .txt supported now"}, status_code=400)
    # chunk + embed
    for i, chunk in enumerate(text.split("\n\n")):
        embedding = embedder.encode(chunk).tolist()
        collection.add(documents=[chunk], embeddings=[embedding], ids=[f"doc_{i}"])
    return {"status": "indexed"}

@app.post("/ask")
async def ask(request: AskRequest, lang: str = "en"):
    q = request.question
    q_vec = embedder.encode(q).tolist()
    res = collection.query(query_embeddings=[q_vec], n_results=3)
    ctx = " ".join(res["documents"][0])
    prompt = f"You are a cybersecurity advisor.\nContext: {ctx}\nQuestion: {q}\nAnswer:"
    output = prompt  # TODO: integrate your LLM here (e.g., Hugging Face pipeline)
    mp3 = speak_text(output, lang=lang)
    return FileResponse(mp3, media_type="audio/mpeg")

@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    tmp = "input.wav"
    with open(tmp, "wb") as f: f.write(await audio.read())
    res = stt_model.transcribe(tmp)
    return {"text": res["text"]}

@app.post("/tts")
async def tts(text: str = Form(...), lang: str = Form("en")):
    out = speak_text(text, lang=lang)
    return FileResponse(out, media_type="audio/mpeg")

# Threat-intel ingestion route
@app.post("/ingest_url")
async def ingest_url(url: str = Form(...)):
    resp = requests.get(url, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    text = ' '.join(soup.get_text().split())
    qid = f"url_{datetime.utcnow().timestamp()}"
    embedding = embedder.encode(text).tolist()
    collection.add(documents=[text], embeddings=[embedding], ids=[qid])
    return {"status": "indexed", "chunks": 1}

@app.get("/health")
def health():
    return {"status": "OK"}
