# 🛡️ CISO Advisor Bot Backend

This FastAPI service powers a CISO Advisor with STT, TTS, document ingestion, and QA capabilities.

## Endpoints

- `POST /upload` – Upload `.txt` files to index Q&A context  
- `POST /stt` – Upload audio, receive transcription  
- `POST /tts` – Provide text + `lang`, receive TTS audio  
- `POST /ask` – Provide `{ "question": "..." }` + `lang`, receive audio answer  
- `POST /ingest_url` – Send a URL, retrieves & indexes content  
- `GET /health` – Health check

## 🔧 Setup

```bash
git clone <your-repo>
cd repo
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 10000
