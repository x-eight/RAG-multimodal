import os
import whisper
from deepgram import DeepgramClient
from ports.interfaces import ITranscriptionService
from domain.models import TranscriptionSegment
from typing import List, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class LocalWhisperTranscriptionAdapter(ITranscriptionService):
    def __init__(self, model_name: str = "base"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, file_path: str) -> str:
        result = self.model.transcribe(file_path)
        return result["text"].strip()

    def transcribe_with_timestamps(self, file_path: str) -> List[TranscriptionSegment]:
        result = self.model.transcribe(file_path)
        segments = []
        for s in result["segments"]:
            segments.append(
                TranscriptionSegment(
                    text=s["text"].strip(), start=s["start"], end=s["end"]
                )
            )
        return segments


class DeepgramTranscriptionAdapter(ITranscriptionService):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        # Use api_key parameter to ensure standard "Token <key>" authentication.
        # "access_token" would use Bearer authentication, which can cause 401s for some project keys.
        self.client = DeepgramClient(api_key=self.api_key)

    def transcribe(self, file_path: str) -> str:
        segments = self.transcribe_with_timestamps(file_path)
        return " ".join([s.text for s in segments])

    def transcribe_with_timestamps(self, file_path: str) -> List[TranscriptionSegment]:
        with open(file_path, "rb") as f:
            buffer_data = f.read()

        # In deepgram-sdk 6.x (Fern-generated), we pass options as keyword arguments.
        # model="nova-2" is widely supported; nova-3 may be available depending on the account.
        response = self.client.listen.v1.media.transcribe_file(
            request=buffer_data,
            model="nova-2",
            smart_format=True,
            utterances=True,
            punctuate=True,
        )

        segments = []
        # Support for the Fern-generated response structure
        if hasattr(response, "results") and response.results.channels:
            for channel in response.results.channels:
                for alternative in channel.alternatives:
                    # Prefer word-level timestamps if available for precise alignment
                    if hasattr(alternative, "words") and alternative.words:
                        for word in alternative.words:
                            segments.append(
                                TranscriptionSegment(
                                    text=word.word, start=word.start, end=word.end
                                )
                            )
                    # Fallback to paragraph-based segments if words are missing
                    elif hasattr(alternative, "paragraphs") and alternative.paragraphs:
                        if hasattr(alternative.paragraphs, "paragraphs"):
                            for para in alternative.paragraphs.paragraphs:
                                for sentence in para.sentences:
                                    segments.append(
                                        TranscriptionSegment(
                                            text=sentence.text,
                                            start=sentence.start,
                                            end=sentence.end,
                                        )
                                    )

        return segments
