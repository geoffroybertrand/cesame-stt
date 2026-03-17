#!/bin/bash
set -e

echo "============================================"
echo "  CESAME STT Test — Installation"
echo "============================================"
echo ""

# Vérifier homebrew
if ! command -v brew &> /dev/null; then
    echo "[ERREUR] Homebrew requis. Installer depuis https://brew.sh"
    exit 1
fi
echo "[OK] Homebrew"

# ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "[...] Installation de ffmpeg..."
    brew install ffmpeg
fi
echo "[OK] ffmpeg"

# Python 3.11+
PYTHON=""
for p in python3.11 python3.12 python3.13 python3; do
    if command -v "$p" &> /dev/null; then
        version=$("$p" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON="$p"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[ERREUR] Python 3.11+ requis."
    echo "  brew install python@3.11"
    exit 1
fi
echo "[OK] $PYTHON ($($PYTHON --version))"

# Virtualenv
if [ ! -d "venv" ]; then
    echo "[...] Creation du virtualenv..."
    $PYTHON -m venv venv
fi
source venv/bin/activate
echo "[OK] Virtualenv active"

# Dependencies
echo "[...] Installation des dependances Python (peut prendre quelques minutes)..."
pip install --upgrade pip -q
pip install -r requirements.txt

# mlx-whisper pour macOS Apple Silicon
if [ "$(uname -m)" = "arm64" ] && [ "$(uname -s)" = "Darwin" ]; then
    echo "[...] Installation de mlx-whisper (Apple Silicon)..."
    pip install -r requirements-mac.txt
fi

# Dossiers
mkdir -p recordings

echo ""
echo "============================================"
echo "  Installation terminee !"
echo "============================================"
echo ""

# Token HF
if [ -z "$HF_TOKEN" ]; then
    echo "IMPORTANT: Configurez votre token HuggingFace :"
    echo ""
    echo "  1. Creez un token sur https://huggingface.co/settings/tokens"
    echo "  2. Acceptez la licence pyannote :"
    echo "     https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "     https://huggingface.co/pyannote/segmentation-3.0"
    echo "  3. Exportez le token :"
    echo "     export HF_TOKEN=hf_votre_token"
    echo ""
fi

echo "Demarrer l'application :"
echo "  source venv/bin/activate"
echo "  export HF_TOKEN=votre_token"
echo "  python app.py"
echo ""
echo "Ou avec Docker :"
echo "  cp .env.example .env  # editez le token"
echo "  docker compose up --build"
echo ""
echo "Puis ouvrir http://localhost:8000"
