import os
import platform

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

RECORDINGS_DIR = "recordings"
CHUNK_DURATION_S = 4
OVERLAP_DURATION_S = 0.5

SAMPLE_RATE = 16000


def get_model_id(model_name: str) -> str:
    """Retourne l'identifiant du modèle selon le backend."""
    info = WHISPER_MODELS.get(model_name, WHISPER_MODELS[DEFAULT_MODEL_REALTIME])
    return info["mlx"] if WHISPER_BACKEND == "mlx" else info["fw"]
