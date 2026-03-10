import os
import time
from typing import Optional, List
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from ports.interfaces import (
    ITranscriptionService,
    IVisualDescriptionService,
    IEmbeddingService,
    IChatService,
)
from domain.models import TranscriptionSegment
from dotenv import load_dotenv

load_dotenv()


class GeminiAdapter:
    def __init__(
        self, api_key: Optional[str] = None, model_id: str = "gemini-2.0-flash"
    ):
        self.client = genai.Client(
            api_key=api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        self.model_id = model_id


class GeminiTranscriptionAdapter(GeminiAdapter, ITranscriptionService):
    def transcribe(self, file_path: str) -> str:
        """
        Transcribe audio from a file using Gemini.
        Note: Gemini can process audio files directly.
        """
        myfile = self.client.files.upload(file=file_path)

        # Wait for file to be processed if it's a video/audio (though for small audio it's usually fast)
        # For simplicity in this adapter, we just send it.
        # In a production scenario, we might need to poll myfile.state

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[
                myfile,
                "Transcribe the audio from this file exactly. Just return the text.",
            ],
        )
        return response.text

    def transcribe_with_timestamps(self, file_path: str) -> List[TranscriptionSegment]:
        """
        Transcribe with timestamps and sound descriptions using Gemini JSON output.
        """
        myfile = self.client.files.upload(file=file_path)

        # Wait for file to be processed if it's large (though for small audio it's usually fast)
        # For simplicity in this adapter, we just send it.
        
        prompt = """
        Analyze this audio and provide a transcription with timestamps.
        If there is speech, transcribe it into segments with start and end times.
        If there are significant environmental sounds (birds, traffic, music, etc.), 
        describe them as segments with start and end times.
        
        Return the result as a JSON list of segments:
        [
          {"text": "transcribed text or sound description", "start": 0.0, "end": 1.5, "type": "speech"},
          {"text": "ononomatopoeia or sound desc", "start": 1.5, "end": 3.0, "type": "sound"},
          ...
        ]
        The "type" field must be either "speech" or "sound".
        Only return the JSON.
        """

        response = self.client.models.generate_content(
            model=self.model_id,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
            contents=[
                myfile,
                prompt,
            ],
        )

        import json

        try:
            data = json.loads(response.text)
            segments = []
            for item in data:
                segments.append(
                    TranscriptionSegment(
                        text=item["text"],
                        start=float(item["start"]),
                        end=float(item["end"]),
                        type=item.get("type", "speech"),
                    )
                )
            return segments
        except Exception as e:
            print(f"Error parsing Gemini transcription JSON: {e}")
            # Fallback to simple transcription if JSON fails
            full_text = self.transcribe(file_path)
            return [TranscriptionSegment(text=full_text, start=0.0, end=0.0, type="speech")]


class GeminiVisualDescriptionAdapter(GeminiAdapter, IVisualDescriptionService):
    def describe_frame(self, frame: np.ndarray) -> str:
        temp_path = "temp/temp_gemini_frame.png"
        Image.fromarray(frame).save(temp_path)

        with open(temp_path, "rb") as f:
            image_bytes = f.read()

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                "Describe this scene in detail.",
            ],
        )

        os.remove(temp_path)
        return response.text

    def describe_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                "Describe this image in detail.",
            ],
        )
        return response.text


class GeminiEmbeddingAdapter(GeminiAdapter, IEmbeddingService):
    def generate_embedding(self, text: str) -> np.ndarray:
        result = self.client.models.embed_content(
            model="gemini-embedding-001", contents=text  # or gemini-embedding-001
        )
        # The new SDK returns a list of embeddings if contents is a list,
        # or a single embedding object if it's a string.
        # Based on docs: result.embeddings is a list/object.
        return np.array(
            (
                result.embeddings[0].values
                if isinstance(result.embeddings, list)
                else result.embeddings.values
            ),
            dtype=np.float32,
        )


class GeminiChatAdapter(GeminiAdapter, IChatService):
    def ask(self, system_prompt: str, user_prompt: str) -> str:
        # Gemini often uses a single prompt or system_instruction
        # In the new SDK, we can pass system_instruction in config
        response = self.client.models.generate_content(
            model=self.model_id,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
            contents=user_prompt,
        )
        return response.text
