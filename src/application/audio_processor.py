import os
from typing import Optional
from domain.models import MediaSegment, MultimodalDataset
from ports.interfaces import ITranscriptionService


class AudioProcessor:
    def __init__(self, transcription_service: ITranscriptionService):
        self.transcription_service = transcription_service

    def process(
        self,
        audio_path: str,
        interval: Optional[float] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> MultimodalDataset:
        
        # Extract/slice audio only the requested portion
        temp_audio_path = "temp/temp_audio_slice.wav"

        # ffmpeg duration (-t)
        duration_arg = ""
        if end_time is not None:
            duration = end_time - start_time
            if duration > 0:
                duration_arg = f"-t {duration}"

        # ffmpeg -ss (seek) and optional -t (duration)
        os.system(
            f"ffmpeg -y -ss {start_time} {duration_arg} -i {audio_path} -ar 16000 -ac 1 {temp_audio_path}"
        )

        segments = self.transcription_service.transcribe_with_timestamps(temp_audio_path)
        os.remove(temp_audio_path)
        
        dataset = MultimodalDataset(source_path=audio_path, media_type="audio")

        import librosa
        
        # Determine actual end time if not provided
        actual_end_time = end_time
        if actual_end_time is None:
            try:
                y, sr = librosa.load(audio_path, sr=None)
                actual_end_time = len(y) / sr
            except Exception:
                # Fallback to the last segment's end time if available, or just use a large number
                actual_end_time = max([s.end for s in segments] + [start_time]) if segments else start_time + 60.0

        if actual_end_time <= start_time:
             return dataset

        # If no interval is provided, treat the entire range as one segment
        # Or alternatively, just use the transcription segments directly as before.
        # But to be consistent with video (interval = 1.0 default if not specified in CLI, but let's handle None)
        computed_interval = interval if interval is not None and interval > 0 else (actual_end_time - start_time)

        current_time = start_time
        segment_id = 0

        while current_time < actual_end_time:
            next_time = min(current_time + computed_interval, actual_end_time)
            
            # Find overlapping segments
            scene_transcripts = []
            scene_audio_descriptions = []
            
            for s in segments:
                abs_s_start = s.start + start_time
                abs_s_end = s.end + start_time

                # Check for overlap
                if max(abs_s_start, current_time) < min(abs_s_end, next_time):
                    if hasattr(s, "type") and s.type == "sound":
                        scene_audio_descriptions.append(s.text)
                    else:
                        scene_transcripts.append(s.text)
                        
            transcript = " ".join(scene_transcripts).strip()
            audio_description = " ".join(scene_audio_descriptions).strip()
            combined_text = " ".join([transcript, audio_description]).strip()

            # For audio, we might not want to create empty segments if there is no sound/text at all in this interval, 
            # but for consistency with video indexing, we create the segment.
            segment = MediaSegment(
                segment_id=segment_id,
                start_time=current_time,
                end_time=next_time,
                transcript=transcript,
                audio_description=audio_description,
                ocr_text="",
                visual_description="",
                combined_text=combined_text,
            )
            dataset.segments.append(segment)
            
            current_time = next_time
            segment_id += 1
            
        return dataset
