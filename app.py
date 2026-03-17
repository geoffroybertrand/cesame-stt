import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

import config
from stt.realtime import transcribe_chunk
from stt.diarization import transcribe_and_diarize, export_txt, export_json

app = FastAPI(title="CESAME STT Test")

# Stockage des jobs de diarisation
jobs: dict = {}

# Créer le dossier recordings
Path(config.RECORDINGS_DIR).mkdir(exist_ok=True)


# --- Pages statiques ---

@app.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/realtime", response_class=HTMLResponse)
async def realtime_page():
    return FileResponse("static/realtime.html")


@app.get("/session", response_class=HTMLResponse)
async def session_page():
    return FileResponse("static/session.html")


@app.get("/diarization", response_class=HTMLResponse)
async def diarization_page():
    return FileResponse("static/diarization.html")


@app.get("/api/health")
async def health_check():
    """Vérifie la disponibilité des composants."""
    import shutil

    checks = {
        "mlx_whisper": False,
        "pyannote": False,
        "hf_token": bool(config.HF_TOKEN),
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }

    try:
        import mlx_whisper  # noqa: F401
        checks["mlx_whisper"] = True
    except ImportError:
        pass

    try:
        import pyannote.audio  # noqa: F401
        checks["pyannote"] = True
    except ImportError:
        pass

    return checks


@app.get("/api/models")
async def list_models():
    """Liste des modèles disponibles avec taille et backend."""
    models = []
    for name, info in config.WHISPER_MODELS.items():
        models.append({
            "name": name,
            "size": info["size"],
            "backend": config.WHISPER_BACKEND,
        })
    return {
        "models": models,
        "default": config.DEFAULT_MODEL_REALTIME,
        "backend": config.WHISPER_BACKEND,
    }


app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/recordings", StaticFiles(directory="recordings"), name="recordings")


# --- WebSocket temps réel ---

@app.websocket("/ws/realtime")
async def ws_realtime(websocket: WebSocket, model: str = None):
    await websocket.accept()

    selected_model = model or config.DEFAULT_MODEL_REALTIME
    all_audio: list[np.ndarray] = []
    overlap_buffer: np.ndarray = np.array([], dtype=np.float32)
    overlap_samples = int(config.OVERLAP_DURATION_S * config.SAMPLE_RATE)
    full_transcript = []

    try:
        while True:
            data = await websocket.receive_bytes()

            # Convertir bytes Float32 en numpy
            audio_chunk = np.frombuffer(data, dtype=np.float32).copy()

            if len(audio_chunk) == 0:
                continue

            all_audio.append(audio_chunk)

            # Ajouter l'overlap du chunk précédent
            if len(overlap_buffer) > 0:
                chunk_with_overlap = np.concatenate([overlap_buffer, audio_chunk])
            else:
                chunk_with_overlap = audio_chunk

            # Garder l'overlap pour le prochain chunk
            if len(audio_chunk) > overlap_samples:
                overlap_buffer = audio_chunk[-overlap_samples:]
            else:
                overlap_buffer = audio_chunk.copy()

            # Transcription
            t0 = time.time()
            text = await asyncio.to_thread(transcribe_chunk, chunk_with_overlap, selected_model)
            latency = time.time() - t0

            if text:
                full_transcript.append(text)

            await websocket.send_json({
                "type": "transcript",
                "text": text,
                "full_text": " ".join(full_transcript),
                "latency_ms": round(latency * 1000),
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # Sauvegarder l'audio complet
        if all_audio:
            full_audio = np.concatenate(all_audio)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"realtime_{ts}.wav"
            filepath = os.path.join(config.RECORDINGS_DIR, filename)
            sf.write(filepath, full_audio, config.SAMPLE_RATE)

            # Envoyer le nom du fichier sauvegardé (le client est peut-être déjà déconnecté)
            try:
                await websocket.send_json({
                    "type": "saved",
                    "filename": filename,
                    "transcript": " ".join(full_transcript),
                })
            except Exception:
                pass


# --- API Diarisation ---

@app.post("/api/diarize")
async def start_diarization(
    file: UploadFile = File(...),
    min_speakers: int = Form(2),
    max_speakers: int = Form(5),
    language: str = Form("fr"),
):
    if not config.HF_TOKEN:
        return JSONResponse(
            status_code=400,
            content={"error": "Token HuggingFace non configuré. Définissez HF_TOKEN."},
        )

    # Sauvegarder le fichier uploadé
    job_id = str(uuid.uuid4())[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = Path(file.filename).suffix or ".wav"
    saved_filename = f"diarize_{ts}_{job_id}{ext}"
    saved_path = os.path.join(config.RECORDINGS_DIR, saved_filename)

    content = await file.read()
    with open(saved_path, "wb") as f:
        f.write(content)

    # Initialiser le job
    jobs[job_id] = {
        "status": "processing",
        "step": 1,
        "step_name": "Transcription",
        "progress": 0,
        "audio_file": saved_filename,
        "result": None,
        "error": None,
        "speaker_names": {},
    }

    # Lancer le traitement en arrière-plan
    asyncio.create_task(
        _run_diarization(job_id, saved_path, min_speakers, max_speakers, language)
    )

    return {"job_id": job_id}


async def _run_diarization(
    job_id: str, audio_path: str, min_speakers: int, max_speakers: int, language: str
):
    def progress_callback(step, step_name, pct):
        jobs[job_id]["step"] = step
        jobs[job_id]["step_name"] = step_name
        jobs[job_id]["progress"] = pct

    try:
        result = await asyncio.to_thread(
            transcribe_and_diarize,
            audio_path,
            config.HF_TOKEN,
            min_speakers,
            max_speakers,
            language,
            progress_callback,
        )
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.get("/api/diarize/{job_id}")
async def get_diarization_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job non trouvé"})

    job = jobs[job_id]
    response = {
        "status": job["status"],
        "step": job["step"],
        "step_name": job["step_name"],
        "progress": job["progress"],
    }

    if job["status"] == "done":
        response["result"] = job["result"]
        response["speaker_names"] = job["speaker_names"]
    elif job["status"] == "error":
        response["error"] = job["error"]

    return response


@app.post("/api/rename-speaker")
async def rename_speaker(data: dict):
    job_id = data.get("job_id")
    speaker_id = data.get("speaker_id")
    new_name = data.get("new_name", "").strip()

    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job non trouvé"})

    if not speaker_id or not new_name:
        return JSONResponse(status_code=400, content={"error": "Paramètres manquants"})

    jobs[job_id]["speaker_names"][speaker_id] = new_name
    return {"ok": True}


@app.get("/api/export/{job_id}/{fmt}")
async def export_result(job_id: str, fmt: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job non trouvé"})

    job = jobs[job_id]
    if job["status"] != "done" or not job["result"]:
        return JSONResponse(status_code=400, content={"error": "Résultat non disponible"})

    turns = job["result"]["turns"]
    speaker_names = job.get("speaker_names", {})

    if fmt == "txt":
        content = export_txt(turns, speaker_names)
        return Response(
            content=content,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename=transcript_{job_id}.txt"},
        )
    elif fmt == "json":
        data = export_json(turns, speaker_names)
        content = json.dumps(data, ensure_ascii=False, indent=2)
        return Response(
            content=content,
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename=transcript_{job_id}.json"},
        )
    else:
        return JSONResponse(status_code=400, content={"error": "Format invalide (txt ou json)"})


# --- Upload d'enregistrement depuis le navigateur ---

@app.post("/api/upload-recording")
async def upload_recording(file: UploadFile = File(...)):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = Path(file.filename).suffix or ".wav"
    filename = f"session_{ts}{ext}"
    filepath = os.path.join(config.RECORDINGS_DIR, filename)

    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    return {"filename": filename}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
