import numpy as np
import grpc
from loguru import logger
from yandex.cloud.ai.stt.v3 import stt_pb2, stt_service_pb2_grpc
from .asr_interface import ASRInterface

_ENDPOINT = "stt.api.cloud.yandex.net:443"
_CHUNK_BYTES = 8000  # 250 ms at 16kHz int16 (2 bytes/sample)


class VoiceRecognition(ASRInterface):
    """Yandex SpeechKit ASR engine — gRPC streaming API v3 (FULL_DATA mode)."""

    def __init__(self, api_key: str, lang: str = "ru-RU", folder_id: str = ""):
        self.api_key = api_key
        self.lang = lang
        self.folder_id = folder_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_options(self) -> stt_pb2.StreamingOptions:
        return stt_pb2.StreamingOptions(
            recognition_model=stt_pb2.RecognitionModelOptions(
                audio_format=stt_pb2.AudioFormatOptions(
                    raw_audio=stt_pb2.RawAudio(
                        audio_encoding=stt_pb2.RawAudio.LINEAR16_PCM,
                        sample_rate_hertz=self.SAMPLE_RATE,
                        audio_channel_count=self.NUM_CHANNELS,
                    )
                ),
                language_restriction=stt_pb2.LanguageRestrictionOptions(
                    restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                    language_code=[self.lang],
                ),
                # FULL_DATA: wait for all audio before recognising (our VAD already
                # guarantees we send a complete utterance)
                audio_processing_type=stt_pb2.RecognitionModelOptions.FULL_DATA,
            )
        )

    def _request_iter(self, audio_bytes: bytes):
        yield stt_pb2.StreamingRequest(session_options=self._build_options())
        for offset in range(0, len(audio_bytes), _CHUNK_BYTES):
            yield stt_pb2.StreamingRequest(
                chunk=stt_pb2.AudioChunk(data=audio_bytes[offset : offset + _CHUNK_BYTES])
            )

    def _metadata(self) -> list:
        meta = [("authorization", f"Api-Key {self.api_key}")]
        if self.folder_id:
            meta.append(("x-folder-id", self.folder_id))
        return meta

    # ------------------------------------------------------------------
    # ASRInterface implementation
    # ------------------------------------------------------------------

    def transcribe_np(self, audio: np.ndarray) -> str:
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        ssl_creds = grpc.ssl_channel_credentials()
        with grpc.secure_channel(_ENDPOINT, ssl_creds) as channel:
            stub = stt_service_pb2_grpc.RecognizerStub(channel)
            responses = stub.RecognizeStreaming(
                self._request_iter(audio_bytes),
                metadata=self._metadata(),
                timeout=15,
            )

            text = ""
            for resp in responses:
                event = resp.WhichOneof("Event")
                # final_refinement carries the normalised full-sentence text
                if event == "final_refinement":
                    alts = resp.final_refinement.normalized_text.alternatives
                    if alts:
                        text = alts[0].text
                elif event == "final" and not text:
                    alts = resp.final.alternatives
                    if alts:
                        text = alts[0].text

        logger.debug(f"Yandex ASR (gRPC v3) result: {text!r}")
        return text.strip()
