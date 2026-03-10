import os
import psycopg2
from dotenv import load_dotenv

def init_db():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        print("Error: DATABASE_URL not found in environment variables.")
        return

    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()

        # Create datasets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id VARCHAR(255) PRIMARY KEY,
                source_path VARCHAR(255) NOT NULL,
                media_type VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create media_segments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS media_segments (
                id SERIAL PRIMARY KEY,
                dataset_id VARCHAR(255) REFERENCES datasets(id) ON DELETE CASCADE,
                segment_id INTEGER NOT NULL,
                start_time FLOAT NOT NULL,
                end_time FLOAT NOT NULL,
                transcript TEXT,
                audio_description TEXT,
                ocr_text TEXT,
                visual_description TEXT,
                combined_text TEXT
            );
        """)

        conn.commit()
        print("Successfully created database tables (datasets, media_segments).")

    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    init_db()
