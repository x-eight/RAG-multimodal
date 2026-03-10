import os
import json
from typing import Optional
from domain.models import MediaSegment, MultimodalDataset
from ports.interfaces import (
    ITranscriptionService,
    IVisualDescriptionService,
    IOCRService,
)
from infrastructure.scene_detector import SceneDetector


from infrastructure.audio_analyzer import AudioAnalyzer


class VideoProcessor:
    def __init__(
        self,
        transcription_service: ITranscriptionService,
        visual_service: IVisualDescriptionService,
        ocr_service: IOCRService,
        scene_detector: SceneDetector,
        fallback_service: Optional[ITranscriptionService] = None,
    ):
        self.transcription_service = transcription_service
        self.visual_service = visual_service
        self.ocr_service = ocr_service
        self.scene_detector = scene_detector
        self.fallback_service = fallback_service
        self.audio_analyzer = AudioAnalyzer()

    def process(
        self,
        video_path: str,
        interval: Optional[float] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> MultimodalDataset:
        # Extract audio and transcribe only the requested portion
        audio_path = "temp/temp_audio.wav"

        # ffmpeg duration (-t)
        duration_arg = ""
        if end_time is not None:
            duration = end_time - start_time
            if duration > 0:
                duration_arg = f"-t {duration}"

        # ffmpeg -ss (seek) and optional -t (duration)
        os.system(
            f"ffmpeg -y -ss {start_time} {duration_arg} -i {video_path} -ar 16000 -ac 1 {audio_path}"
        )

        # Get transcription segments with timestamps
        audio_segments = self.transcription_service.transcribe_with_timestamps(
            audio_path
        )

        # Case 1: Fallback for short audio or failure if sound is present
        is_silent = self.audio_analyzer.is_silent(audio_path)
        if not is_silent and (not audio_segments or len(audio_segments) == 0):
             if self.fallback_service:
                 print("⚠️ Primary STT failed or returned nothing. Using fallback AI...")
                 audio_segments = self.fallback_service.transcribe_with_timestamps(audio_path)

        os.remove(audio_path)

        scenes = self.scene_detector.detect_scenes(
            video_path, max_interval=interval, start_time=start_time, end_time=end_time
        )
        dataset = MultimodalDataset(source_path=video_path, media_type="video")

        for idx, scene in enumerate(scenes):
            frame = scene["frame"]
            ocr_text = self.ocr_service.extract_text(frame)
            visual_desc = self.visual_service.describe_frame(frame)

            # Align audio: filter transcription segments that overlap with this scene
            # Note: scenes are absolute relative to video start
            scene_start = scene["start_time"]
            scene_end = scene["end_time"]

            # Transcription segments are relative to the extracted audio (which starts at start_time)
            # So we adjust by adding start_time to transcription segment times
            scene_transcripts = []
            scene_audio_descriptions = []
            
            for s in audio_segments:
                abs_s_start = s.start + start_time
                abs_s_end = s.end + start_time

                # Check for overlap
                if max(abs_s_start, scene_start) < min(abs_s_end, scene_end):
                    if s.type == "sound":
                        scene_audio_descriptions.append(s.text)
                    else:
                        scene_transcripts.append(s.text)

            transcript = " ".join(scene_transcripts).strip()
            audio_description = " ".join(scene_audio_descriptions).strip()
            
            combined_text = " ".join([transcript, audio_description, ocr_text, visual_desc]).strip()

            segment = MediaSegment(
                segment_id=idx,
                start_time=scene_start,
                end_time=scene_end,
                transcript=transcript,
                audio_description=audio_description,
                ocr_text=ocr_text,
                visual_description=visual_desc,
                combined_text=combined_text,
            )
            dataset.segments.append(segment)

        # Save JSON (to maintain legacy behavior if needed)
        segments_dict = [vars(s) for s in dataset.segments]
        with open("video_multimodal_dataset.json", "w", encoding="utf-8") as f:
            json.dump(segments_dict, f, indent=4)

        return dataset
