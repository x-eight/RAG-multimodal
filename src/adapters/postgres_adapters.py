import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional
from domain.models import MediaSegment, MultimodalDataset
from ports.interfaces import IDatasetRepository
from dotenv import load_dotenv


class PostgresDatasetRepository(IDatasetRepository):
    def __init__(self, db_url: Optional[str] = None):
        if db_url is None:
            load_dotenv()
            db_url = os.getenv("DATABASE_URL")
            
        if not db_url:
            raise ValueError("DATABASE_URL must be provided or set in environment variables.")
            
        self.db_url = db_url

    def _get_connection(self):
        return psycopg2.connect(self.db_url)

    def save(self, source_path: str, dataset: MultimodalDataset):
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            dataset_id = source_path.split("/")[-1]

            # Clear existing data if any (due to ON DELETE CASCADE on segments)
            cursor.execute("DELETE FROM datasets WHERE id = %s", (dataset_id,))

            # Insert dataset metadata
            cursor.execute(
                "INSERT INTO datasets (id, source_path, media_type) VALUES (%s, %s, %s)",
                (dataset_id, dataset.source_path, dataset.media_type)
            )

            # Insert segments
            for s in dataset.segments:
                cursor.execute(
                    """
                    INSERT INTO media_segments 
                    (dataset_id, segment_id, start_time, end_time, transcript, audio_description, ocr_text, visual_description, combined_text)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        dataset_id,
                        s.segment_id,
                        s.start_time,
                        s.end_time,
                        s.transcript,
                        s.audio_description,
                        s.ocr_text,
                        s.visual_description,
                        s.combined_text
                    )
                )

            conn.commit()
            print(f"Dataset '{dataset_id}' successfully saved to PostgreSQL database.")

        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error saving dataset to database: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def load(self, source_path: str) -> Optional[MultimodalDataset]:
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            dataset_id = source_path.split("/")[-1]

            cursor.execute("SELECT * FROM datasets WHERE id = %s", (dataset_id,))
            dataset_row = cursor.fetchone()

            if not dataset_row:
                return None

            dataset = MultimodalDataset(
                source_path=dataset_row["source_path"],
                media_type=dataset_row["media_type"]
            )

            cursor.execute("SELECT * FROM media_segments WHERE dataset_id = %s ORDER BY segment_id ASC", (dataset_id,))
            segment_rows = cursor.fetchall()

            for row in segment_rows:
                segment = MediaSegment(
                    segment_id=row["segment_id"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    transcript=row["transcript"],
                    audio_description=row["audio_description"],
                    ocr_text=row["ocr_text"],
                    visual_description=row["visual_description"],
                    combined_text=row["combined_text"],
                )
                dataset.segments.append(segment)

            return dataset
            
        except Exception as e:
            print(f"Error loading dataset from database: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def exists(self, source_path: str) -> bool:
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            dataset_id = source_path.split("/")[-1]
            cursor.execute("SELECT 1 FROM datasets WHERE id = %s", (dataset_id,))
            return cursor.fetchone() is not None

        except Exception as e:
            print(f"Error checking if dataset exists: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
