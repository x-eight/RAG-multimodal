import numpy as np
from typing import List, Dict, Any
from domain.models import MultimodalDataset, MediaSegment
from ports.interfaces import IEmbeddingService, IVectorStore, IChatService


class RAGOrchestrator:
    def __init__(
        self,
        embedding_service: IEmbeddingService,
        vector_store: IVectorStore,
        chat_service: IChatService,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.chat_service = chat_service

    def index_dataset(self, dataset: MultimodalDataset):
        embeddings = []
        metadata = []
        for segment in dataset.segments:
            emb = self.embedding_service.generate_embedding(segment.combined_text)
            embeddings.append(emb)
            metadata.append(vars(segment))

        if embeddings:
            self.vector_store.add_embeddings(np.stack(embeddings), metadata)

    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        q_emb = self.embedding_service.generate_embedding(question)
        context_fragments = self.vector_store.search(q_emb, k)

        context_text = " ".join([f["combined_text"] for f in context_fragments])

        system_prompt = "Answer using only the provided information."
        user_prompt = f"Context: {context_text}\nQuestion: {question}"

        answer_text = self.chat_service.ask(system_prompt, user_prompt)

        return {
            "answer": answer_text,
            "start_time": (
                context_fragments[0]["start_time"] if context_fragments else 0
            ),
            "end_time": context_fragments[0]["end_time"] if context_fragments else 0,
        }
