import cv2
import numpy as np

LLM_MAX_SIDE = 2048


class InvalidImage(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImage("File is not a readable image")
    return image


def to_llm_jpeg(data: bytes, max_side: int = LLM_MAX_SIDE) -> bytes:
    """Downscale big phone photos: GPT-4o tiles images, so pixels beyond ~2k only cost tokens."""
    image = decode_image(data)
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise InvalidImage("Could not encode image")
    return encoded.tobytes()
