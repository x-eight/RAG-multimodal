import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from typing import Dict, Any, Optional


class AudioAnalyzer:
    def __init__(self, silence_thresh: int = -40, min_silence_len: int = 500):
        self.silence_thresh = silence_thresh
        self.min_silence_len = min_silence_len

    def is_silent(self, file_path: str) -> bool:
        """Check if the audio file is mostly silent."""
        try:
            audio = AudioSegment.from_file(file_path)
            nonsilent_ranges = detect_nonsilent(
                audio,
                min_silence_len=self.min_silence_len,
                silence_thresh=self.silence_thresh,
            )
            return len(nonsilent_ranges) == 0
        except Exception as e:
            print(f"Error checking silence: {e}")
            return False

    def analyze_audio_activity(self, file_path: str) -> Dict[str, Any]:
        """Detect if there is environmental noise or other audio events."""
        try:
            # Load with librosa for more detailed analysis
            y, sr = librosa.load(file_path, sr=None)
            if len(y) == 0:
                return {"has_sound": False, "is_noisy": False, "mean_rms": 0.0}

            rms = librosa.feature.rms(y=y)
            flatness = librosa.feature.spectral_flatness(y=y)

            mean_rms = np.mean(rms)
            mean_flatness = np.mean(flatness)

            return {
                "has_sound": mean_rms > 0.0005,
                "is_noisy": mean_flatness > 0.05,
                "mean_rms": float(mean_rms),
            }
        except Exception as e:
            print(f"Error analyzing audio activity: {e}")
            return {"has_sound": True, "is_noisy": False, "mean_rms": 0.0}
