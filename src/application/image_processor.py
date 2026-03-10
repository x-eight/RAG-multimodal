from domain.models import MediaSegment, MultimodalDataset
from ports.interfaces import IVisualDescriptionService, IOCRService


class ImageProcessor:
    def __init__(
        self, visual_service: IVisualDescriptionService, ocr_service: IOCRService
    ):
        self.visual_service = visual_service
        self.ocr_service = ocr_service

    def process(self, image_path: str) -> MultimodalDataset:
        visual_desc = self.visual_service.describe_image(image_path)
        ocr_text = self.ocr_service.extract_text_from_image(image_path)

        combined_text = " ".join([ocr_text, visual_desc]).strip()

        segment = MediaSegment(
            segment_id=0,
            start_time=0.0,
            end_time=0.0,
            transcript="",
            audio_description="",
            ocr_text=ocr_text,
            visual_description=visual_desc,
            combined_text=combined_text,
        )

        dataset = MultimodalDataset(source_path=image_path, media_type="image")
        dataset.segments.append(segment)
        return dataset
