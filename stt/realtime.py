import numpy as np
import config


def transcribe_chunk(audio_data: np.ndarray, model_name: str = None) -> str:
    """Transcrit un chunk audio numpy float32 mono 16kHz."""
    if audio_data is None or len(audio_data) < config.SAMPLE_RATE * 0.3:
        return ""

    audio_data = audio_data.astype(np.float32)
    if np.max(np.abs(audio_data)) > 1.0:
        audio_data = audio_data / np.max(np.abs(audio_data))

    model_id = config.get_model_id(model_name or config.DEFAULT_MODEL_REALTIME)

    if config.WHISPER_BACKEND == "mlx":
        return _transcribe_mlx(audio_data, model_id)
    else:
        return _transcribe_faster_whisper(audio_data, model_id)


def _transcribe_mlx(audio_data: np.ndarray, model_id: str) -> str:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio_data,
        path_or_hf_repo=model_id,
        language="fr",
        word_timestamps=False,
        verbose=False,
    )
    return result.get("text", "").strip()


_fw_models: dict = {}


def _transcribe_faster_whisper(audio_data: np.ndarray, model_id: str) -> str:
    global _fw_models
    if model_id not in _fw_models:
        from faster_whisper import WhisperModel
        _fw_models[model_id] = WhisperModel(
            model_id,
            device="cpu",
            compute_type="int8",
        )

    segments, _ = _fw_models[model_id].transcribe(
        audio_data,
        language="fr",
        beam_size=5,
        word_timestamps=False,
    )
    return " ".join(seg.text.strip() for seg in segments)
