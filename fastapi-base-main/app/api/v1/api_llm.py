from typing import Any

from fastapi import APIRouter, status

from app.schemas.sche_llm import (
    CorrectOcrRequest,
    CorrectOcrResponse,
    ReviewRequest,
    ReviewResponse,
)
from app.schemas.sche_response import DataResponse
from app.services.srv_llm import LLMService
from app.utils.exception_handler import CustomException, ExceptionType

router = APIRouter(prefix="/llm")

llm_service: LLMService = LLMService()


@router.post(
    "/correct-ocr",
    response_model=DataResponse[CorrectOcrResponse],
    status_code=status.HTTP_200_OK,
    summary="Sửa lỗi nhận dạng trong output OCR (không sửa lỗi của sinh viên)",
)
def correct_ocr(body: CorrectOcrRequest) -> Any:
    try:
        data = llm_service.correct_ocr(body)
        return DataResponse(http_code=status.HTTP_200_OK, data=data)
    except CustomException:
        raise
    except Exception:
        raise CustomException(exception=ExceptionType.INTERNAL_SERVER_ERROR)


@router.post(
    "/review",
    response_model=DataResponse[ReviewResponse],
    status_code=status.HTTP_200_OK,
    summary="Nhận xét bài làm dựa trên code và kết quả chạy test",
)
def review(body: ReviewRequest) -> Any:
    try:
        data = llm_service.review(body)
        return DataResponse(http_code=status.HTTP_200_OK, data=data)
    except CustomException:
        raise
    except Exception:
        raise CustomException(exception=ExceptionType.INTERNAL_SERVER_ERROR)


@router.get(
    "/info",
    status_code=status.HTTP_200_OK,
    summary="Xem đang dùng provider / model nào",
)
def info() -> Any:
    c = llm_service.client
    return DataResponse(http_code=status.HTTP_200_OK,
                        data={"provider": c.provider, "model": c.model, "base_url": c.base_url})
