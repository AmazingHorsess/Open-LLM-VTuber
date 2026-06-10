import requests
from loguru import logger
from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    """Yandex SpeechKit TTS engine."""

    def __init__(
        self,
        api_key: str,
        voice: str = "alena",
        lang: str = "ru-RU",
        speed: float = 1.0,
        emotion: str = "neutral",
        folder_id: str = "",
    ):
        self.api_key = api_key
        self.folder_id = folder_id
        self.voice = voice
        self.lang = lang
        self.speed = speed
        self.emotion = emotion
        self.file_extension = "ogg"

    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)

        data = {
            "text": text,
            "lang": self.lang,
            "voice": self.voice,
            "speed": str(self.speed),
            "emotion": self.emotion,
            "format": "oggopus",
        }
        if self.folder_id:
            data["folderId"] = self.folder_id

        response = requests.post(
            "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize",
            headers={"Authorization": f"Api-Key {self.api_key}"},
            data=data,
        )
        response.raise_for_status()

        with open(file_name, "wb") as f:
            f.write(response.content)

        logger.debug(f"Yandex TTS generated: {file_name}")
        return file_name
