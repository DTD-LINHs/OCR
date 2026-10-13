"""PaddleOCR wrapper. The model is loaded lazily, once per process."""

import logging
import threading
from typing import Optional

from app.schemas.sche_ocr_pipeline import OcrBox
from app.utils.image import decode_image

logger = logging.getLogger(__name__)


class PaddleOcrEngine:
    def __init__(
        self,
        lang: str = "en",
        device: str = "cpu",
        det_model: Optional[str] = None,
        rec_model: Optional[str] = None,
    ):
        self.lang = lang
        self.device = device
        self.det_model = det_model
        self.rec_model = rec_model
        self._ocr = None
        # Paddle predictors are not thread-safe; FastAPI background tasks run in a thread pool.
        self._lock = threading.Lock()

    def _load(self):
        from paddleocr import PaddleOCR

        kwargs = {
            "lang": self.lang,
            "device": self.device,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
        }
        if self.det_model:
            kwargs["text_detection_model_name"] = self.det_model
        if self.rec_model:
            kwargs["text_recognition_model_name"] = self.rec_model
        logger.info("Loading PaddleOCR %s", kwargs)
        return PaddleOCR(**kwargs)

    def predict(self, image: bytes) -> list[OcrBox]:
        array = decode_image(image)
        with self._lock:
            if self._ocr is None:
                self._ocr = self._load()
            results = self._ocr.predict(array)
        return [box for result in results for box in self._parse(result)]

    @staticmethod
    def _parse(result) -> list[OcrBox]:
        def field(key):
            try:
                return result[key]
            except (KeyError, TypeError):
                data = getattr(result, "json", None) or {}
                return data.get("res", data).get(key)

        texts = field("rec_texts") or []
        scores = field("rec_scores") or []
        polys = field("rec_polys")
        if polys is None:
            polys = field("dt_polys") or []
        return [
            OcrBox(
                text=str(text),
                polygon=[[float(x), float(y)] for x, y in poly],
                confidence=float(score),
            )
            for text, score, poly in zip(texts, scores, polys)
        ]
