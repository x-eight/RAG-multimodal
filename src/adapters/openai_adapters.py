import os
from openai import OpenAI
from ports.interfaces import (
    ITranscriptionService,
    IVisualDescriptionService,
    IEmbeddingService,
    IChatService,
)
from domain.models import TranscriptionSegment
import numpy as np
from PIL import Image
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()


class OpenAIAdapter:
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))


class OpenAITranscriptionAdapter(OpenAIAdapter, ITranscriptionService):
    def transcribe(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", file=f
            )
        return transcript.text

    def transcribe_with_timestamps(self, file_path: str) -> List[TranscriptionSegment]:
        with open(file_path, "rb") as f:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="verbose_json"
            )
        segments = []
        if hasattr(transcript, "segments"):
            for s in transcript.segments:
                segments.append(
                    TranscriptionSegment(
                        text=s["text"].strip(), start=s["start"], end=s["end"]
                    )
                )
        return segments


class OpenAIVisualDescriptionAdapter(OpenAIAdapter, IVisualDescriptionService):
    def describe_frame(self, frame: np.ndarray) -> str:
        temp_path = "temp/temp_frame.png"
        Image.fromarray(frame).save(temp_path)
        with open(temp_path, "rb") as f:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "user", "content": "Describe this scene in detail."}
                ],
                files=[{"name": temp_path, "data": f.read()}],
            )
        os.remove(temp_path)
        return response.choices[0].message.content


class OpenAIEmbeddingAdapter(OpenAIAdapter, IEmbeddingService):
    def generate_embedding(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)


class OpenAIChatAdapter(OpenAIAdapter, IChatService):
    def ask(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
