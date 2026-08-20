import os
import platform

# Charger le fichier .env s'il existe
from pathlib import Path as _Path
_env_path = _Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Détection de l'environnement d'exécution
# mlx-whisper ne fonctionne que nativement sur macOS Apple Silicon (pas dans Docker)
IS_MACOS_NATIVE = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and not os.environ.get("RUNNING_IN_DOCKER")
)

if IS_MACOS_NATIVE:
    WHISPER_BACKEND = "mlx"
else:
    WHISPER_BACKEND = "faster-whisper"

# Détection CUDA pour choisir le bon default
import torch as _torch
HAS_CUDA = _torch.cuda.is_available()

# Modèles disponibles avec leur taille approximative
WHISPER_MODELS = {
    "tiny":           {"size": "~75 Mo",  "mlx": "mlx-community/whisper-tiny",             "fw": "tiny"},
    "base":           {"size": "~140 Mo", "mlx": "mlx-community/whisper-base",             "fw": "base"},
    "small":          {"size": "~460 Mo", "mlx": "mlx-community/whisper-small",            "fw": "small"},
    "medium":         {"size": "~1.5 Go", "mlx": "mlx-community/whisper-medium",           "fw": "medium"},
    "large-v3-turbo": {"size": "~3 Go",   "mlx": "mlx-community/whisper-large-v3-turbo",   "fw": "large-v3-turbo"},
    "large-v3":       {"size": "~3 Go",   "mlx": "mlx-community/whisper-large-v3-mlx",     "fw": "large-v3"},
}

# Avec GPU (CUDA ou MLX), on peut utiliser les gros modèles en temps réel
DEFAULT_MODEL_REALTIME = "large-v3-turbo" if (IS_MACOS_NATIVE or HAS_CUDA) else "small"
WHISPER_MODEL_DIARIZATION = "large-v3"

# --- Moteur ASR du pipeline batch (/api/diarize) -------------------------
# "faster-whisper" : large-v3 via CTranslate2 (historique, licence MIT).
# "crisperwhisper" : CrisperWhisper 2.0 via CTranslate2 — timestamps au mot
#   (~30 ms), long-form sans artefacts, modes verbatim/intended.
#   ⚠️ Poids sous licence Nyra Health NON COMMERCIALE (recherche et usage
#   non commercial uniquement) : à revoir avant toute monétisation du
#   service — cf. README. Le temps réel (/ws/realtime) reste sur Whisper.
STT_ENGINE = os.environ.get("STT_ENGINE", "faster-whisper").strip().lower()

# Modèle du moteur choisi ; vide → défaut du moteur
# (faster-whisper : "large-v3" ; crisperwhisper : "large").
STT_MODEL = os.environ.get("STT_MODEL", "").strip()

# CrisperWhisper : "verbatim" (fidèle : hésitations, répétitions, faux
# départs) ou "intended" (texte propre, plus adapté au découpage RAG).
STT_MODE = os.environ.get("STT_MODE", "verbatim").strip().lower()

# CrisperWhisper : modèle brouillon du décodage spéculatif (ex. "turbo",
# +1,3-1,4× de vitesse, ~1,6 Go de VRAM en plus). Vide → désactivé.
STT_DRAFT_MODEL = os.environ.get("STT_DRAFT_MODEL", "").strip()

# Charger les modèles au démarrage plutôt qu'au premier job (le premier
# chargement coûte 10-20 s, et le téléchargement initial bien davantage).
STT_PRELOAD = os.environ.get("STT_PRELOAD", "1").strip().lower() not in ("0", "false", "no")

# Un seul job GPU à la fois : deux transcriptions simultanées chargeraient
# deux fois les modèles en VRAM.
STT_MAX_CONCURRENCY = max(1, int(os.environ.get("STT_MAX_CONCURRENCY", "1")))

CRISPERWHISPER_DEFAULT_MODEL = "large"

# --- Dictée utilisateur (POST /api/transcribe) ---------------------------
# Chemin court et synchrone : mono-locuteur, sans diarisation, sans job.
#
# ⚠️ TOUJOURS faster-whisper, JAMAIS CrisperWhisper : les poids de ce dernier
# sont sous licence non commerciale interdisant explicitement « production or
# operational deployment ». Servir des utilisateurs finaux en est. Whisper est
# sous MIT — et rend de surcroît nativement un texte propre et ponctué, sans
# les hésitations, ce qui est exactement ce qu'on veut dans un champ message.
STT_DICTEE_MODEL = os.environ.get("STT_DICTEE_MODEL", "large-v3-turbo").strip()

# File d'attente SÉPARÉE de STT_MAX_CONCURRENCY : sans cela, une ingestion
# d'une heure lancée depuis l'admin gèlerait le micro de tous les
# utilisateurs pendant toute sa durée.
STT_DICTEE_CONCURRENCY = max(1, int(os.environ.get("STT_DICTEE_CONCURRENCY", "2")))

# Garde-fous, vérifiés AVANT de toucher au GPU.
STT_DICTEE_MAX_SECONDS = max(10, int(os.environ.get("STT_DICTEE_MAX_SECONDS", "180")))
STT_DICTEE_MAX_BYTES = max(
    100_000, int(os.environ.get("STT_DICTEE_MAX_BYTES", str(25 * 1024 * 1024)))
)

# Biais de vocabulaire (faster-whisper ≥ 1.1). C'est le levier le plus direct
# sur la fidélité : sans lui, le lexique du projet est systématiquement
# massacré (« sola stalgie », « écho-anxiété », « Nardonne »).
HOTWORDS_DEFAUT = (
    "éco-anxiété, solastalgie, canicule, vague de chaleur, îlot de chaleur, "
    "adaptation climatique, dérèglement climatique, Global Adaptation, PRS, "
    "échelle de résolution, thérapie systémique et stratégique, Nardone, "
    "Watzlawick, Palo Alto, LACT, SYPRENE, CERMES3, CNRS"
)

# ⚠️ Une variable DÉFINIE MAIS VIDE doit retomber sur le défaut, pas le
# désactiver : les fichiers Compose passent `STT_HOTWORDS=${STT_HOTWORDS:-}`,
# donc la variable existe et vaut "" dès qu'elle n'est pas renseignée. Avec un
# simple os.environ.get(clé, défaut), le lexique se retrouvait desactivé en
# silence partout — y compris en production.
# Pour le désactiver réellement (comparaison A/B) : STT_HOTWORDS=none
_hotwords = os.environ.get("STT_HOTWORDS", "").strip()
STT_HOTWORDS = "" if _hotwords.lower() == "none" else (_hotwords or HOTWORDS_DEFAUT)

RECORDINGS_DIR = "recordings"
CHUNK_DURATION_S = 4
OVERLAP_DURATION_S = 0.5

SAMPLE_RATE = 16000


def get_model_id(model_name: str) -> str:
    """Retourne l'identifiant du modèle selon le backend."""
    info = WHISPER_MODELS.get(model_name, WHISPER_MODELS[DEFAULT_MODEL_REALTIME])
    return info["mlx"] if WHISPER_BACKEND == "mlx" else info["fw"]
