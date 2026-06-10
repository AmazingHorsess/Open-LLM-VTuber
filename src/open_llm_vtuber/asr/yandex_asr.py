import io
import numpy as np
import scipy.io.wavfile as wav_io
import requests
from loguru import logger
from .asr_interface import ASRInterface


class VoiceRecognition(ASRInterface):
    """Yandex SpeechKit ASR engine (synchronous REST API v1)."""

    def __init__(self, api_key: str, lang: str = "ru-RU", folder_id: str = ""):
        self.api_key = api_key
        self.lang = lang
        self.folder_id = folder_id

    def transcribe_np(self, audio: np.ndarray) -> str:
        # Convert float32 [-1, 1] to 16-bit PCM
        audio_int16 = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767).astype(np.int16)

        buf = io.BytesIO()
        wav_io.write(buf, self.SAMPLE_RATE, audio_int16)
        # Yandex LPCM API expects raw PCM, skip WAV header (44 bytes)
        buf.seek(44)
        pcm_data = buf.read()

        params = {
            "lang": self.lang,
            "format": "lpcm",
            "sampleRateHertz": str(self.SAMPLE_RATE),
        }
        if self.folder_id:
            params["folderId"] = self.folder_id

        response = requests.post(
            "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize",
            headers={
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/octet-stream",
            },
            params=params,
            data=pcm_data,
        )
        response.raise_for_status()
        result = response.json().get("result", "")
        logger.debug(f"Yandex ASR result: {result!r}")
        return result
