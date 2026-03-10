import cv2
import numpy as np
from typing import List, Dict, Optional


class SceneDetector:
    def detect_scenes(
        self,
        video_path: str,
        threshold: float = 30.0,
        max_interval: Optional[float] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> List[Dict]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Seek to start_time
        start_frame = int(start_time * fps)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
        else:
            frame_idx = 0

        scenes = []
        prev_frame_gray = None
        last_frame_bgr = None
        current_segment_start = start_time

        while True:
            ret, frame = cap.read()
            current_time = frame_idx / fps

            # Check if we reached the end_time or the end of the video
            if not ret or (end_time is not None and current_time >= end_time):
                if last_frame_bgr is not None:
                    scenes.append(
                        {
                            "start_time": current_segment_start,
                            "end_time": (
                                min(current_time, end_time)
                                if end_time
                                else current_time
                            ),
                            "frame": last_frame_bgr,
                        }
                    )
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Check for time interval split
            is_interval_reached = (
                max_interval and (current_time - current_segment_start) >= max_interval
            )

            # 2. Check for scene change
            is_scene_change = False
            if prev_frame_gray is not None:
                diff = cv2.absdiff(gray, prev_frame_gray)
                score = np.mean(diff)
                if score > threshold:
                    is_scene_change = True

            if is_interval_reached or is_scene_change:
                # If it's a scene change, we use the PREVIOUS frame
                # If it's just an interval reached, we can use the CURRENT frame
                frame_to_save = last_frame_bgr if is_scene_change else frame
                scenes.append(
                    {
                        "start_time": current_segment_start,
                        "end_time": current_time,
                        "frame": frame_to_save,
                    }
                )
                current_segment_start = current_time
                prev_frame_gray = gray

            last_frame_bgr = frame
            prev_frame_gray = gray if prev_frame_gray is None else prev_frame_gray
            frame_idx += 1

        cap.release()
        return scenes
