from typing import Protocol, List, Dict, Any, Optional
import numpy as np


class ITranscriptionService(Protocol):
    def transcribe(self, file_path: str) -> str:
        """Transcribe audio from a file."""
        ...

    def transcribe_with_timestamps(
        self, file_path: str
    ) -> List[Any]:  # List[TranscriptionSegment]
        """Transcribe audio with timestamps."""
        ...


class IVisualDescriptionService(Protocol):
    def describe_frame(self, frame: np.ndarray) -> str:
        """Generate a description for a given video frame or image."""
        ...

    def describe_image(self, image_path: str) -> str:
        """Generate a description for a given image file."""
        ...


class IOCRService(Protocol):
    def extract_text(self, frame: np.ndarray) -> str:
        """Extract text from a video frame or image using OCR."""
        ...

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image file using OCR."""
        ...


class IEmbeddingService(Protocol):
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding vector for the given text."""
        ...


class IVectorStore(Protocol):
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Add embeddings to the store with associated metadata."""
        ...

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Search for the most similar embeddings."""
        ...

    def save(self, path: str):
        """Save the vector index to a file."""
        ...

    def load(self, path: str):
        """Load the vector index from a file."""
        ...


class IDatasetRepository(Protocol):
    def save(
        self, source_path: str, dataset: Any
    ):  # Any to avoid circular import in some cases, or use Dict
        """Save the dataset metadata for a given source (video, audio, image)."""
        ...

    def load(self, source_path: str) -> Optional[Any]:
        """Load the dataset metadata for a given source."""
        ...

    def exists(self, source_path: str) -> bool:
        """Check if the source has already been indexed."""
        ...


class IChatService(Protocol):
    def ask(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the chat service and get a response."""
        ...
