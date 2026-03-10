from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class MediaSegment:
    segment_id: int
    start_time: float
    end_time: float
    transcript: str = ""
    audio_description: str = ""
    ocr_text: str = ""
    visual_description: str = ""
    combined_text: str = ""


@dataclass
class TranscriptionSegment:
    text: str
    start: float
    end: float
    type: str = "speech"  # "speech" or "sound"


@dataclass
class MultimodalDataset:
    source_path: str
    media_type: str  # "video", "audio", "image"
    segments: List[MediaSegment] = field(default_factory=list)
