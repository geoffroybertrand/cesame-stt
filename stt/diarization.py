import os
import subprocess
import tempfile

import torch
import numpy as np
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import config


def convert_to_wav(audio_path: str) -> str:
    """Convertit un fichier audio en WAV 16kHz mono via ffmpeg si nécessaire."""
    if audio_path.lower().endswith(".wav"):
        return audio_path

    wav_path = os.path.splitext(audio_path)[0] + ".wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
        capture_output=True,
        check=True,
    )
    return wav_path


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def transcribe_and_diarize(
    audio_path: str,
    hf_token: str,
    min_speakers: int = 2,
    max_speakers: int = 5,
    language: str = "fr",
    progress_callback=None,
):
    """Pipeline complet : transcription + diarisation.

    Utilise faster-whisper pour la transcription et pyannote directement
    pour la diarisation (sans passer par whisperX).
    """
    # Convertir en WAV si nécessaire (webm, m4a, mp3, etc.)
    audio_path = convert_to_wav(audio_path)
    device = get_device()

    def notify(step, name, pct):
        if progress_callback:
            progress_callback(step, name, pct)

    # Étape 1 : Transcription avec faster-whisper
    notify(1, "Transcription", 0)
    model = WhisperModel(
        config.WHISPER_MODEL_DIARIZATION,
        device="cpu",
        compute_type="int8",
    )
    segments_raw, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        word_timestamps=True,
    )
    # Matérialiser les segments (le générateur ne peut être lu qu'une fois)
    segments = []
    for seg in segments_raw:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": [
                {"start": w.start, "end": w.end, "word": w.word}
                for w in (seg.words or [])
            ],
        })
    del model
    notify(1, "Transcription", 100)

    # Étape 2 : Diarisation avec pyannote
    notify(2, "Diarisation", 0)
    diarize_device = torch.device(device)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline.to(diarize_device)

    diarization = pipeline(
        audio_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    del pipeline
    notify(2, "Diarisation", 100)

    # Étape 3 : Associer locuteurs aux segments transcrits
    notify(3, "Association locuteurs", 0)

    # Construire la timeline des locuteurs
    speaker_timeline = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_timeline.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    # Associer chaque segment transcrit au locuteur dominant
    for seg in segments:
        seg["speaker"] = _find_speaker(seg["start"], seg["end"], speaker_timeline)

    notify(3, "Association locuteurs", 100)

    turns = format_transcript(segments)
    speakers = set(t["speaker"] for t in turns)

    return {"turns": turns, "speakers": sorted(speakers)}


def _find_speaker(seg_start, seg_end, speaker_timeline):
    """Trouve le locuteur dominant pour un segment donné."""
    overlaps = {}
    for st in speaker_timeline:
        overlap_start = max(seg_start, st["start"])
        overlap_end = min(seg_end, st["end"])
        if overlap_start < overlap_end:
            duration = overlap_end - overlap_start
            overlaps[st["speaker"]] = overlaps.get(st["speaker"], 0) + duration

    if not overlaps:
        return "INCONNU"
    return max(overlaps, key=overlaps.get)


def format_transcript(segments):
    """Formater les segments en liste de tours de parole."""
    turns = []
    current_speaker = None
    current_text = []
    current_start = None

    for seg in segments:
        speaker = seg.get("speaker", "INCONNU")
        text = seg.get("text", "").strip()
        start = seg.get("start", 0)

        if speaker != current_speaker:
            if current_speaker and current_text:
                turns.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text),
                    "start": current_start,
                    "end": start,
                })
            current_speaker = speaker
            current_text = [text] if text else []
            current_start = start
        else:
            if text:
                current_text.append(text)

    if current_speaker and current_text:
        turns.append({
            "speaker": current_speaker,
            "text": " ".join(current_text),
            "start": current_start,
            "end": segments[-1].get("end", current_start),
        })

    return turns


def format_time(seconds: float) -> str:
    """Convertit des secondes en MM:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def export_txt(turns: list, speaker_names: dict = None) -> str:
    """Exporte les tours de parole en format texte."""
    lines = []
    for turn in turns:
        speaker = turn["speaker"]
        if speaker_names and speaker in speaker_names:
            speaker = speaker_names[speaker]
        ts = format_time(turn["start"])
        lines.append(f"[{ts}] {speaker} : {turn['text']}")
    return "\n".join(lines)


def export_json(turns: list, speaker_names: dict = None) -> list:
    """Exporte les tours de parole en format JSON structuré."""
    result = []
    for turn in turns:
        speaker = turn["speaker"]
        display_name = speaker
        if speaker_names and speaker in speaker_names:
            display_name = speaker_names[speaker]
        result.append({
            "speaker_id": speaker,
            "speaker_name": display_name,
            "start": turn["start"],
            "end": turn["end"],
            "start_formatted": format_time(turn["start"]),
            "end_formatted": format_time(turn["end"]),
            "text": turn["text"],
        })
    return result
