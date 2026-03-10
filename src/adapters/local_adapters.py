import pytesseract
from PIL import Image
import numpy as np
import faiss
import json
import os
from typing import List, Dict, Any, Optional
from ports.interfaces import IOCRService, IVectorStore, IDatasetRepository


class PytesseractOCRAdapter(IOCRService):
    def extract_text(self, frame: np.ndarray) -> str:
        img = Image.fromarray(frame)
        return pytesseract.image_to_string(img).strip()

    def extract_text_from_image(self, image_path: str) -> str:
        return pytesseract.image_to_string(Image.open(image_path)).strip()

class FAISSVectorStoreAdapter(IVectorStore):
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata: List[Dict[str, Any]] = []

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        D, I = self.index.search(query_embedding.reshape(1, -1), k)
        return [self.metadata[i] for i in I[0] if i < len(self.metadata) and i >= 0]

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4)

    def load(self, path: str):
        if os.path.exists(f"{path}.index"):
            self.index = faiss.read_index(f"{path}.index")
        if os.path.exists(f"{path}.metadata.json"):
            with open(f"{path}.metadata.json", "r", encoding="utf-8") as f:
                self.metadata = json.load(f)


class JSONDatasetRepository(IDatasetRepository):
    def __init__(self, storage_path: str = "database/indexed_videos.json"):
        self.storage_path = storage_path
        self._db = self._load_db()

    def _load_db(self) -> Dict[str, Any]:
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_db(self):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self._db, f, indent=4)

    def save(self, video_path: str, dataset: Any):
        # Convert dataset to dict if it's a domain object
        if hasattr(dataset, "__dict__"):
            # Deep conversion for segments
            data = {
                "source_path": dataset.source_path,
                "media_type": dataset.media_type,
                "segments": [vars(s) for s in dataset.segments],
            }
        else:
            data = dataset

        self._db[video_path] = data
        self._save_db()

    def load(self, video_path: str) -> Optional[Dict[str, Any]]:
        return self._db.get(video_path)

    def exists(self, video_path: str) -> bool:
        return video_path in self._db
