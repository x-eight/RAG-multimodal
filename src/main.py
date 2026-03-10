from infrastructure.scene_detector import SceneDetector
from adapters.openai_adapters import (
    OpenAITranscriptionAdapter,
    OpenAIVisualDescriptionAdapter,
    OpenAIEmbeddingAdapter,
    OpenAIChatAdapter,
)
from adapters.gemini_adapters import (
    GeminiTranscriptionAdapter,
    GeminiVisualDescriptionAdapter,
    GeminiEmbeddingAdapter,
    GeminiChatAdapter,
)
from adapters.local_adapters import (
    PytesseractOCRAdapter,
    FAISSVectorStoreAdapter,
)
from adapters.postgres_adapters import PostgresDatasetRepository
from adapters.stt_adapters import DeepgramTranscriptionAdapter
from application.video_processor import VideoProcessor
from application.audio_processor import AudioProcessor
from application.image_processor import ImageProcessor
from application.rag_orchestrator import RAGOrchestrator
from domain.models import MultimodalDataset, MediaSegment

from typing import Optional
import argparse
import hashlib
import sys


def generate_index_id(
    video_path: str, interval: Optional[float], start: float, end: Optional[float]
) -> str:
    """Generate a unique and stable ID for the given indexing parameters."""
    base_name = video_path.split("/")[-1].split(".")[0]
    config_str = f"{video_path}_{interval}_{start}_{end}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return f"{base_name}_{config_hash}"


def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: index
    index_parser = subparsers.add_parser("index", help="Index a media file")
    index_parser.add_argument("--file", required=True, help="Path to the file (video, audio, image) to index")
    index_parser.add_argument(
        "--interval", type=float, default=1.0, help="Indexing interval in seconds (video/audio only)"
    )
    index_parser.add_argument(
        "--start", type=float, default=0.0, help="Start time in seconds (video/audio only)"
    )
    index_parser.add_argument(
        "--end", type=float, default=None, help="End time in seconds (video/audio only)"
    )

    # Command: query
    query_parser = subparsers.add_parser("query", help="Query an existing index")
    query_parser.add_argument("--id", required=True, help="The index ID to query")
    query_parser.add_argument("--question", required=True, help="The question to ask")
    query_parser.add_argument(
        "--k", type=int, default=3, help="Number of segments to retrieve"
    )

    args = parser.parse_args()

    # 1. Initialize Adapters
    # User requested Deepgram as primary, but we'll use Gemini as fallback for robustness
    transcription_service = DeepgramTranscriptionAdapter() 
    fallback_transcription = GeminiTranscriptionAdapter() 
    
    visual_service = GeminiVisualDescriptionAdapter()
    embedding_service = GeminiEmbeddingAdapter()
    chat_service = GeminiChatAdapter()
    
    ocr_service = PytesseractOCRAdapter()
    scene_detector = SceneDetector()
    
    # We use a placeholder dimension, it will be updated or loaded
    vector_store = FAISSVectorStoreAdapter(dimension=3072) # 3072 for Gemini/OpenAI
    
    # Use PostgreSQL instead of JSON
    # dataset_repo = JSONDatasetRepository()
    dataset_repo = PostgresDatasetRepository()

    if args.command == "index":
        video_processor = VideoProcessor(
            transcription_service=transcription_service,
            visual_service=visual_service,
            ocr_service=ocr_service,
            scene_detector=scene_detector,
            fallback_service=fallback_transcription
        )
        audio_processor = AudioProcessor(transcription_service=transcription_service)
        image_processor = ImageProcessor(visual_service=visual_service, ocr_service=ocr_service)

        file_path = args.file
        ext = file_path.lower().split('.')[-1]
        
        video_exts = ["mp4", "avi", "mkv", "mov", "webm"]
        audio_exts = ["mp3", "wav", "aac", "flac", "m4a", "ogg"]
        image_exts = ["jpg", "jpeg", "png", "bmp", "gif", "webp"]

        if ext in video_exts:
            media_type = "video"
            index_id = generate_index_id(file_path, args.interval, args.start, args.end)
        elif ext in audio_exts:
            media_type = "audio"
            index_id = generate_index_id(file_path, args.interval, args.start, args.end)
        elif ext in image_exts:
            media_type = "image"
            index_id = generate_index_id(file_path, None, 0.0, None)
        else:
            print(f"Error: Unsupported file extension '.{ext}'")
            sys.exit(1)

        storage_key = f"database/{index_id}"

        if dataset_repo.exists(storage_key):
            print(f"File already indexed with ID: {index_id}")
            print(
                f'You can query it using: python src/main.py query --id {index_id} --question "..."'
            )
            return

        print(f"🚀 Indexing {media_type}: {file_path}")
        if media_type in ["video", "audio"]:
            if args.end is not None:
                print(f"⏱️  Range: {args.start}s to {args.end}s (Duration: {args.end - args.start}s)")
            else:
                print(f" [From {args.start}s to end]")
        
        if media_type in ["video", "audio"]:
            print(f"📊 Interval: {args.interval}s")

        if media_type == "video":
            dataset = video_processor.process(
                file_path, interval=args.interval, start_time=args.start, end_time=args.end
            )
        elif media_type == "audio":
            dataset = audio_processor.process(
                file_path, interval=args.interval, start_time=args.start, end_time=args.end
            )
        elif media_type == "image":
            dataset = image_processor.process(file_path)

        rag_orchestrator = RAGOrchestrator(
            embedding_service=embedding_service,
            vector_store=vector_store,
            chat_service=chat_service,
        )

        print("Generating embeddings and indexing...")
        rag_orchestrator.index_dataset(dataset)

        print("Saving index...")
        dataset_repo.save(storage_key, dataset)
        vector_store.save(storage_key)

        print(f"\n✅ Indexing complete! Index ID: {index_id}")
        print(
            f'Query command: python src/main.py query --id {index_id} --question "Your question here"'
        )

    elif args.command == "query":
        storage_key = f"database/{args.id}"

        if not dataset_repo.exists(storage_key):
            print(f"Error: Index ID '{args.id}' not found in database.")
            sys.exit(1)

        print(f"Loading index '{args.id}'...")
        vector_store.load(storage_key)

        rag_orchestrator = RAGOrchestrator(
            embedding_service=embedding_service,
            vector_store=vector_store,
            chat_service=chat_service,
        )

        print(f"Querying: {args.question}")
        result = rag_orchestrator.query(args.question, k=args.k)

        print("\n🤖 Answer:", result["answer"])
        print(f"📍 Timestamp: {result['start_time']:.2f}s - {result['end_time']:.2f}s")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
