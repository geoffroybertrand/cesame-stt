import os
import subprocess

import config
from stt import engines


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
    return engines.get_device()


def _transcribe_mlx(audio_path: str, language: str) -> list:
    """Transcription via mlx-whisper (GPU Metal Apple Silicon)."""
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=config.get_model_id(config.WHISPER_MODEL_DIARIZATION),
        language=language,
        word_timestamps=True,
        verbose=False,
    )
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg.get("text", "").strip(),
            "words": [
                {"start": w["start"], "end": w["end"], "word": w["word"]}
                for w in seg.get("words", [])
            ],
        })
    return segments


def transcribe_and_diarize(
    audio_path: str,
    hf_token: str,
    min_speakers: int = 2,
    max_speakers: int = 5,
    language: str = "fr",
    progress_callback=None,
    engine: str = None,
    mode: str = None,
):
    """Pipeline complet : transcription + diarisation.

    Transcription par le moteur configuré (stt/engines.py : faster-whisper
    ou CrisperWhisper 2.0), diarisation par pyannote, puis attribution des
    locuteurs AU MOT quand les timestamps le permettent — un changement de
    locuteur en milieu de phrase n'est donc plus perdu.
    """
    # Convertir en WAV si nécessaire (webm, m4a, mp3, etc.)
    audio_path = convert_to_wav(audio_path)
    device = get_device()

    def notify(step, name, pct):
        if progress_callback:
            progress_callback(step, name, pct)

    # Étape 1 : Transcription
    notify(1, "Transcription", 0)
    # Support language="auto" : passer None pour détection automatique
    transcribe_language = None if language == "auto" else language
    segments = engines.transcribe(
        audio_path, transcribe_language, engine=engine, mode=mode, device=device
    )
    notify(1, "Transcription", 100)

    # Étape 2 : Diarisation avec pyannote (pipeline mis en cache)
    notify(2, "Diarisation", 0)
    pipeline = engines.get_diarization_pipeline(hf_token, device)
    diarization = pipeline(
        audio_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    notify(2, "Diarisation", 100)

    # Étape 3 : Associer locuteurs aux mots (repli : aux segments)
    notify(3, "Association locuteurs", 0)

    # Construire la timeline des locuteurs
    speaker_timeline = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_timeline.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    turns = _assign_speakers(segments, speaker_timeline)
    notify(3, "Association locuteurs", 100)

    speakers = set(t["speaker"] for t in turns)
    return {"turns": turns, "speakers": sorted(speakers)}


def _assign_speakers(segments: list, speaker_timeline: list) -> list:
    """Attribue les locuteurs et construit les tours de parole.

    Voie principale : mot à mot (précision des timestamps CrisperWhisper
    ~30 ms), ce qui découpe correctement les phrases où deux personnes
    s'enchaînent. Repli sur l'attribution par segment si le moteur n'a pas
    fourni de timestamps de mots.
    """
    words = []
    for seg in segments:
        for w in seg.get("words") or []:
            if w.get("start") is None or w.get("end") is None:
                continue
            text = (w.get("word") or "").strip()
            if text:
                words.append({"start": w["start"], "end": w["end"], "word": text})

    if not words:
        for seg in segments:
            seg["speaker"] = _find_speaker(seg["start"], seg["end"], speaker_timeline)
        return format_transcript(segments)

    turns = []
    for w in words:
        speaker = _find_speaker(w["start"], w["end"], speaker_timeline)
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] += " " + w["word"]
            turns[-1]["end"] = w["end"]
        else:
            turns.append({
                "speaker": speaker,
                "text": w["word"],
                "start": w["start"],
                "end": w["end"],
            })
    return turns


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
    """Formate des segments (déjà porteurs d'un « speaker ») en tours de
    parole — repli utilisé quand aucun timestamp de mot n'est disponible."""
    turns = []
    current_speaker = None
    current_text = []
    current_start = None
    current_end = None

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
                    "end": current_end,
                })
            current_speaker = speaker
            current_text = [text] if text else []
            current_start = start
        else:
            if text:
                current_text.append(text)
        current_end = seg.get("end", start)

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
