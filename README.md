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
| **Dictee temps reel** | Streaming mono-locuteur, texte en direct | mlx-whisper (Mac) / faster-whisper large-v3-turbo |
| **Session multi-locuteur** | Enregistrement live + diarisation post | idem + pyannote |
| **Batch post-seance** | Upload de fichier + diarisation complete | moteur ASR configurable + pyannote |

## Moteur ASR du pipeline batch (`/api/diarize`)

Deux moteurs, tous deux sur CTranslate2, choisis par `STT_ENGINE` :

| `STT_ENGINE` | Modele | Apport |
|---|---|---|
| `faster-whisper` (defaut) | `large-v3`, 1,55 Md | Historique, licence MIT |
| `crisperwhisper` | CrisperWhisper 2.0 `large`, 2 Md | Timestamps au mot (~30 ms) → **diarisation au mot** (un changement de locuteur en milieu de phrase n'est plus perdu), long-form sans artefacts, modes `verbatim` / `intended` |

> ⚠️ **Licence CrisperWhisper 2.0** — les poids sont sous *Nyra Health
> Non-Commercial Research License* : « non-commercial and research use only,
> no production or operational deployment, no use from which commercial gain
> is derived » (le code du paquet, lui, est MIT). Usage actuel : projet de
> recherche-action CNRS/PISE, service gratuit, corpus construit en interne.
> **A renegocier aupres de licensing@nyra-labs.com avant toute monetisation**
> (credits payants, offre mutuelles). Repli immediat : `STT_ENGINE=faster-whisper`.
> Les hotwords necessitent les modeles `*_pro`, sous licence commerciale.

Variables d'environnement (lues depuis `.env` ou l'environnement) :

| Variable | Defaut | Role |
|---|---|---|
| `STT_ENGINE` | `faster-whisper` | Moteur du pipeline batch |
| `STT_MODEL` | vide | Modele ; vide → `large` (crisperwhisper) ou `large-v3` |
| `STT_MODE` | `verbatim` | CrisperWhisper : `verbatim` (hesitations, reprises) ou `intended` (texte lisse) |
| `STT_DRAFT_MODEL` | vide | Modele brouillon du decodage speculatif (ex. `turbo`) : +1,3-1,4×, ~1,6 Go de VRAM |
| `STT_PRELOAD` | `1` | Charger les modeles au demarrage plutot qu'au premier job |
| `STT_MAX_CONCURRENCY` | `1` | Jobs GPU simultanes (ingestion) |
| `STT_DICTEE_MODEL` | `large-v3-turbo` | Modele de la dictee utilisateur — **toujours faster-whisper** |
| `STT_DICTEE_CONCURRENCY` | `2` | Dictees simultanees, **file separee de l'ingestion** |
| `STT_DICTEE_MAX_SECONDS` | `180` | Duree max d'une dictee |
| `STT_DICTEE_MAX_BYTES` | `26214400` | Taille max de l'upload (25 Mo) |
| `STT_HOTWORDS` | lexique du projet | Biais de vocabulaire de la dictee |

### Dictee utilisateur — `POST /api/transcribe`

Chemin court et synchrone pour le micro du chat : upload → texte, en une
reponse. Ni diarisation, ni job asynchrone, ni ecriture persistante — tout
transite par un repertoire temporaire detruit dans un `finally`, y compris en
cas d'echec. Contrairement a `/api/diarize`, **rien n'atterrit dans
`recordings/`** : ce sont des personnes qui decrivent leur detresse, et
l'application leur promet que rien n'est conserve.

```bash
curl -s -X POST http://localhost:8000/api/transcribe -F "file=@dictee.webm"
# {"text": "...", "duration": 8.6, "elapsed": 2.1}
```

Trois points de conception :

- **File d'attente separee.** Le semaphore de la dictee est distinct de celui
  de l'ingestion. Partages, une ingestion d'une heure gelerait le micro de
  tous les utilisateurs pendant toute sa duree. Verifie : deux dictees
  simultanees se chevauchent (17,9 s au mur pour 31,3 s cumulees), donc
  CTranslate2 tolere les appels concurrents sur une meme instance.
- **Toujours faster-whisper, jamais CrisperWhisper.** La licence de ce
  dernier interdit explicitement « production or operational deployment » ;
  servir des utilisateurs finaux en est. Whisper est sous MIT — et rend de
  surcroit nativement un texte propre et ponctue, sans les hesitations, ce
  qui est exactement ce qu'on veut dans un champ message.
- **`vad_filter` + `hotwords` + `condition_on_previous_text=False`.** Le VAD
  retire les silences, ou Whisper hallucine des phrases entieres — son echec
  le plus connu, et inevitable quand quelqu'un hesite au micro. Les hotwords
  biaisent le decodage vers le lexique du projet (solastalgie, eco-anxiete,
  PRS, Nardone). ⚠️ Leur apport reste **a mesurer sur voix reelle** : sur de
  la synthese vocale, `large-v3` transcrivait deja ces termes correctement
  sans eux, donc le test realise ne prouve rien de leur utilite.

`POST /api/diarize` accepte aussi les champs `engine` et `mode` pour comparer
deux moteurs sur le meme fichier sans redeployer. `GET /api/health` expose la
configuration active (moteur, modele, mode, **device reellement utilise**,
prechargement) — un repli silencieux en CPU devient ainsi visible.

### VRAM (NVIDIA L4 24 Go)

| Composant | float16 |
|---|---|
| Whisper large-v3 | ~3 Go |
| CrisperWhisper 2.0 large | ~4 Go |
| + brouillon `turbo` (speculatif) | ~1,6 Go |
| pyannote 3.1 | ~1-2 Go |

Soit ~5-8 Go selon la configuration : large marge sur 24 Go. Les modeles sont
charges **une seule fois** (auparavant reconstruits a chaque job, 10-20 s
perdues) et un semaphore serialise les jobs pour ne pas doubler la VRAM.

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

## Performances

### Mac (M4 Max 48 Go, backend MLX natif)

| Operation | Temps approximatif |
|-----------|--------------------|
| Chunk 4s temps reel (mlx-whisper) | ~0.4s |
| Transcription 1h | ~2-3 min |
| Diarisation 1h (pyannote) | ~1-2 min |
| Pipeline complet 1h | ~5-8 min |

### GPU NVIDIA L4 24 Go (production Scaleway) — a mesurer

Aucune mesure GPU n'a encore ete relevee. A completer au premier fichier reel
transcrit en production :

| Operation | `faster-whisper` large-v3 | `crisperwhisper` large | + brouillon `turbo` |
|---|---|---|---|
| Chargement des modeles (une seule fois au demarrage) | | | |
| Transcription 1h | | | |
| Diarisation 1h (pyannote) | | | |
| Pipeline complet 1h | | | |
| VRAM crete | | | |

Protocole de mesure, depuis le serveur :

```bash
# 1. Le service est-il bien sur GPU, avec le bon moteur, modeles precharges ?
docker exec global-adaptation \
  node -e "fetch('http://cesame-stt:8000/api/health').then(r=>r.json()).then(o=>console.log(o))"
# attendu : engine=crisperwhisper, device=cuda, preloaded=true

# 2. VRAM et occupation pendant un job (a lancer en parallele d'une transcription)
watch -n 2 nvidia-smi

# 3. Chronometrage d'un fichier local (contourne l'admin)
time docker exec cesame-stt python3 -c "
from stt.diarization import transcribe_and_diarize
import config, time
t0=time.time()
r=transcribe_and_diarize('recordings/VOTRE_FICHIER.wav', config.HF_TOKEN, 2, 5, 'fr')
print(len(r['turns']), 'tours,', len(r['speakers']), 'locuteurs en', round(time.time()-t0,1), 's')"
```

Le premier demarrage est plus long : telechargement du modele (~4 Go pour
CrisperWhisper) **et conversion CTranslate2 unique**, mise en cache ensuite
dans le volume `huggingface-cache`. Avec `STT_PRELOAD=1`, cela se produit au
demarrage du conteneur — suivre `docker logs -f cesame-stt` jusqu'a la ligne
`[stt] modeles precharges — moteur=... device=cuda`.

## Premier lancement

Au premier lancement, les modeles sont telecharges automatiquement :
- Whisper large-v3 (CTranslate2) : ~3 Go
- ou CrisperWhisper 2.0 large : ~4 Go, **plus une conversion CTranslate2 unique**
  (quelques minutes, mise en cache ensuite)
- pyannote segmentation + diarisation : ~200 Mo
- mlx-whisper large-v3-turbo (Mac natif uniquement) : ~3 Go

Prevoyez ~3 a 5 Go de telechargement selon le moteur, dans le volume
`huggingface-cache`. Avec `STT_PRELOAD=1` (defaut), tout se fait au demarrage
du service et non pendant la premiere transcription.

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
