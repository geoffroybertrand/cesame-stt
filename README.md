# CESAME STT Test

Outil de test Speech-to-Text pour le projet CESAME (therapie systemique breve).
Optimise pour Apple Silicon (M4 Max) avec mlx-whisper et pyannote.

## Prerequis

- macOS 15+ (Tahoe) avec Apple Silicon (M4/M3/M2)
- Python 3.11+
- Homebrew
- Docker (optionnel, pour le deploiement conteneurise)
- Token HuggingFace (pour la diarisation multi-locuteur)

## 3 modes de test

| Mode | Description | Backend |
|------|-------------|---------|
| **Dictee temps reel** | Streaming mono-locuteur, texte en direct | mlx-whisper large-v3-turbo |
| **Session multi-locuteur** | Enregistrement live + diarisation post | mlx-whisper + whisperX + pyannote |
| **Batch post-seance** | Upload de fichier + diarisation complete | whisperX + pyannote |

## Installation

### Option 1 : Docker (recommande)

```bash
# 1. Cloner le projet
cd cesame-stt

# 2. Configurer le token HuggingFace
cp .env.example .env
# Editez .env et ajoutez votre token HF_TOKEN=hf_...

# 3. Lancer
docker compose up --build

# 4. Ouvrir http://localhost:8000
```

### Option 2 : Installation locale

```bash
# 1. Installer
bash install.sh

# 2. Activer le virtualenv
source venv/bin/activate

# 3. Configurer le token
export HF_TOKEN=hf_votre_token

# 4. Lancer
python app.py

# 5. Ouvrir http://localhost:8000
```

## Token HuggingFace

La diarisation multi-locuteur necessite un token HuggingFace et l'acceptation
des licences pyannote :

1. Creez un compte sur [huggingface.co](https://huggingface.co)
2. Generez un token : [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Acceptez les licences (cliquez "Agree") :
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

**Note :** Le mode "Dictee temps reel" fonctionne SANS token HuggingFace.

## Performances attendues (M4 Max 48 Go)

| Operation | Temps approximatif |
|-----------|--------------------|
| Chunk 4s temps reel (mlx-whisper) | ~0.4s |
| Transcription 1h (whisperX batch) | ~2-3 min |
| Diarisation 1h (pyannote) | ~1-2 min |
| Pipeline complet 1h | ~5-8 min |

## Premier lancement

Au premier lancement, les modeles seront telecharges automatiquement :
- mlx-whisper large-v3-turbo : ~3 Go
- whisperX large-v3 : ~3 Go
- pyannote segmentation + diarisation : ~200 Mo
- wav2vec2 (alignement) : ~1 Go

Total : environ **7 Go** de telechargement. Prevoyez une bonne connexion.

## Depannage

**Le micro ne fonctionne pas dans le navigateur**
- Utilisez Chrome ou Safari
- Autorisez l'acces au micro quand le navigateur le demande
- L'application doit etre servie en HTTP (localhost) ou HTTPS

**Erreur "HF_TOKEN non configure"**
- Verifiez que la variable d'environnement est definie : `echo $HF_TOKEN`
- Avec Docker : verifiez le fichier `.env`

**"ffmpeg not found"**
- `brew install ffmpeg`

**Erreur memoire pendant la diarisation**
- Le pipeline complet peut utiliser ~16 Go de RAM
- Fermez les applications gourmandes en memoire
- Sur 48 Go de memoire unifiee, pas de probleme en usage normal

**La transcription est lente**
- Au premier appel, le modele est telecharge et charge en memoire (~10-20s)
- Les appels suivants sont rapides (~0.4s par chunk de 4s)

## Architecture

```
cesame-stt/
├── app.py                 # Serveur FastAPI + WebSocket
├── config.py              # Configuration
├── stt/
│   ├── realtime.py        # Transcription streaming (mlx-whisper)
│   └── diarization.py     # Pipeline diarisation (whisperX + pyannote)
├── static/
│   ├── index.html         # Hub des 3 modes
│   ├── realtime.html      # Mode dictee temps reel
│   ├── session.html       # Mode session multi-locuteur
│   ├── diarization.html   # Mode batch post-seance
│   └── app.js             # Utilitaires JS partages
├── recordings/            # Fichiers audio sauvegardes
├── Dockerfile
├── docker-compose.yml
└── install.sh
```
