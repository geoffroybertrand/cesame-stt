"""Moteurs de transcription du pipeline batch, avec cache de modèles.

Deux moteurs produisent le MÊME contrat de sortie, consommé par
stt/diarization.py :

    [{"start": float, "end": float, "text": str,
      "words": [{"start": float, "end": float, "word": str}]}]

- "faster-whisper"  : large-v3 via CTranslate2 (historique).
- "crisperwhisper"  : CrisperWhisper 2.0 via CTranslate2 — timestamps au
  mot (~30 ms), long-form natif, modes verbatim/intended.

Les modèles sont chargés une seule fois et gardés en mémoire (ils étaient
auparavant reconstruits à chaque job, soit 10-20 s perdues par
transcription). Le pipeline pyannote suit la même logique.
"""

import threading

import torch
import torch.serialization

import config

# Cache { clé → modèle } protégé par un verrou : les jobs tournent dans des
# threads (asyncio.to_thread), deux chargements simultanés du même modèle
# doubleraient la VRAM.
_lock = threading.RLock()
_asr_models: dict = {}
_diarization_pipeline = None


def resolve_engine(engine: str | None = None) -> str:
    """Moteur effectif : paramètre de requête, sinon configuration."""
    name = (engine or "").strip().lower() or config.STT_ENGINE
    return name if name in ("faster-whisper", "crisperwhisper") else "faster-whisper"


def resolve_mode(mode: str | None = None) -> str:
    """Mode CrisperWhisper effectif (verbatim | intended)."""
    name = (mode or "").strip().lower() or config.STT_MODE
    return name if name in ("verbatim", "intended") else "verbatim"


def model_name(engine: str) -> str:
    if config.STT_MODEL:
        return config.STT_MODEL
    if engine == "crisperwhisper":
        return config.CRISPERWHISPER_DEFAULT_MODEL
    return config.WHISPER_MODEL_DIARIZATION


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- Chargement des modèles ----------------------------------------------

def _load_faster_whisper(name: str, device: str):
    from faster_whisper import WhisperModel

    if device == "cuda":
        fw_device, compute_type = "cuda", "float16"
    else:
        fw_device, compute_type = "cpu", "int8"
    return WhisperModel(name, device=fw_device, compute_type=compute_type)


def _load_crisperwhisper(name: str, device: str):
    from crisperwhisper import CrisperWhisperModel

    kwargs = {}
    if config.STT_DRAFT_MODEL:
        kwargs["draft_model"] = config.STT_DRAFT_MODEL
    # Le backend ct2 (GPU NVIDIA) est choisi automatiquement par la lib
    # quand l'extra [ct2] est installé ; premier chargement = téléchargement
    # depuis HuggingFace puis conversion CTranslate2 mise en cache.
    return CrisperWhisperModel(name, **kwargs)


def get_asr(engine: str | None = None, device: str | None = None):
    """Modèle ASR du moteur demandé, chargé une seule fois."""
    engine = resolve_engine(engine)
    device = device or get_device()
    name = model_name(engine)
    key = (engine, name, device)
    with _lock:
        if key not in _asr_models:
            loader = _load_crisperwhisper if engine == "crisperwhisper" else _load_faster_whisper
            _asr_models[key] = loader(name, device)
        return _asr_models[key]


def get_diarization_pipeline(hf_token: str, device: str | None = None):
    """Pipeline pyannote, chargé une seule fois et gardé sur le device."""
    global _diarization_pipeline
    device = device or get_device()
    with _lock:
        if _diarization_pipeline is None:
            from pyannote.audio import Pipeline

            # PyTorch 2.6+ : autoriser le chargement des poids pyannote
            torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            pipeline.to(torch.device(device))
            _diarization_pipeline = pipeline
        return _diarization_pipeline


def preload(hf_token: str = "") -> dict:
    """Précharge ASR (+ pyannote si le token est là). Jamais bloquant :
    une erreur de préchargement est rapportée, le job la refera surface."""
    state = {"asr": False, "diarization": False, "error": None}
    try:
        get_asr()
        state["asr"] = True
        if hf_token:
            get_diarization_pipeline(hf_token)
            state["diarization"] = True
    except Exception as e:  # pragma: no cover - dépend de l'environnement
        state["error"] = str(e)
    return state


def is_loaded() -> bool:
    with _lock:
        return bool(_asr_models)


# --- Transcription --------------------------------------------------------

def _segments_from_faster_whisper(model, audio_path: str, language: str | None) -> list:
    segments_raw, _ = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        word_timestamps=True,
    )
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
    return segments


def _words_of(result) -> list:
    """Mots normalisés depuis un résultat CrisperWhisper (objet ou dict)."""
    raw = getattr(result, "words", None)
    if raw is None and isinstance(result, dict):
        raw = result.get("words")
    words = []
    for w in raw or []:
        start = getattr(w, "start", None) if not isinstance(w, dict) else w.get("start")
        end = getattr(w, "end", None) if not isinstance(w, dict) else w.get("end")
        text = getattr(w, "word", None) if not isinstance(w, dict) else w.get("word")
        if text is None:
            text = getattr(w, "text", None) if not isinstance(w, dict) else w.get("text")
        if text is None:
            continue
        words.append({"start": start, "end": end, "word": text})
    return words


def _group_words(words: list, pause_s: float = 0.6) -> list:
    """Reconstruit des segments à partir des mots : coupure sur une pause
    marquée ou une ponctuation forte. Sert quand le moteur n'expose pas de
    segments — l'attribution des locuteurs, elle, travaille sur les mots."""
    segments = []
    current: list = []
    for w in words:
        if current and w.get("start") is not None and current[-1].get("end") is not None:
            if w["start"] - current[-1]["end"] > pause_s:
                segments.append(current)
                current = []
        current.append(w)
        if current[-1]["word"].strip().endswith((".", "!", "?", "…")):
            segments.append(current)
            current = []
    if current:
        segments.append(current)

    out = []
    for group in segments:
        if not group:
            continue
        text = "".join(w["word"] for w in group).strip()
        if not text:
            continue
        starts = [w["start"] for w in group if w.get("start") is not None]
        ends = [w["end"] for w in group if w.get("end") is not None]
        out.append({
            "start": starts[0] if starts else 0.0,
            "end": ends[-1] if ends else (starts[0] if starts else 0.0),
            "text": text,
            "words": group,
        })
    return out


def _segments_from_crisperwhisper(model, audio_path: str, language: str | None, mode: str) -> list:
    result = model.transcribe(
        audio_path,
        language=language,
        mode=mode,
        word_timestamps=True,
        **({"speculative_decoding": True} if config.STT_DRAFT_MODEL else {}),
    )

    # La lib expose au minimum result.text et result.words ; si elle fournit
    # aussi des segments, on les reprend tels quels, sinon on les reconstruit.
    raw_segments = getattr(result, "segments", None)
    if raw_segments is None and isinstance(result, dict):
        raw_segments = result.get("segments")

    if raw_segments:
        segments = []
        for seg in raw_segments:
            get = seg.get if isinstance(seg, dict) else (lambda k, d=None: getattr(seg, k, d))
            text = (get("text", "") or "").strip()
            words = _words_of(seg)
            starts = [w["start"] for w in words if w.get("start") is not None]
            ends = [w["end"] for w in words if w.get("end") is not None]
            segments.append({
                "start": get("start", starts[0] if starts else 0.0),
                "end": get("end", ends[-1] if ends else 0.0),
                "text": text,
                "words": words,
            })
        if segments:
            return segments

    words = _words_of(result)
    if words:
        return _group_words(words)

    # Dernier recours : le texte seul, sans timestamps (l'attribution des
    # locuteurs retombera alors sur « INCONNU »).
    text = (getattr(result, "text", "") or "").strip()
    return [{"start": 0.0, "end": 0.0, "text": text, "words": []}] if text else []


def transcribe(
    audio_path: str,
    language: str | None,
    engine: str | None = None,
    mode: str | None = None,
    device: str | None = None,
) -> list:
    """Transcrit un fichier et renvoie le contrat commun (segments + mots)."""
    engine = resolve_engine(engine)
    device = device or get_device()

    if engine == "crisperwhisper":
        model = get_asr(engine, device)
        return _segments_from_crisperwhisper(model, audio_path, language, resolve_mode(mode))

    if config.IS_MACOS_NATIVE:
        from stt.diarization import _transcribe_mlx

        return _transcribe_mlx(audio_path, language)

    model = get_asr(engine, device)
    return _segments_from_faster_whisper(model, audio_path, language)
