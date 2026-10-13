from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile, status

from app.schemas.sche_base import PaginationParams, SortParams
from app.schemas.sche_ocr import OcrSubmissionResponse, OcrTraceResponse
from app.schemas.sche_response import DataResponse
from app.services.srv_ocr_submission import OcrSubmissionService, process_submission
from app.utils.exception_handler import CustomException

router = APIRouter(prefix=f"/ocr")

ocr_service: OcrSubmissionService = OcrSubmissionService()


@router.post(
    "/submissions",
    response_model=DataResponse[OcrSubmissionResponse],
    status_code=status.HTTP_202_ACCEPTED,
)
def create_submission(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> Any:
    try:
        submission = ocr_service.create_from_upload(file)
        background_tasks.add_task(process_submission, submission.id)
        return DataResponse(
            http_code=status.HTTP_202_ACCEPTED,
            data=OcrSubmissionResponse.model_validate(submission),
        )
    except Exception as e:
        raise CustomException(exception=e)


@router.get(
    "/submissions",
    response_model=DataResponse[List[OcrSubmissionResponse]],
    status_code=status.HTTP_200_OK,
)
def get_submissions(
    sort_params: SortParams = Depends(),
    pagination_params: PaginationParams = Depends(),
) -> Any:
    try:
        data, metadata = ocr_service.get_by_filter(
            pagination_params=pagination_params, sort_params=sort_params
        )
        return DataResponse(
            http_code=status.HTTP_200_OK,
            data=[OcrSubmissionResponse.model_validate(item) for item in data],
            metadata=metadata,
        )
    except Exception as e:
        raise CustomException(exception=e)


@router.get(
    "/submissions/{submission_id}",
    response_model=DataResponse[OcrSubmissionResponse],
    status_code=status.HTTP_200_OK,
)
def get_submission(submission_id: UUID) -> Any:
    try:
        submission = ocr_service.get_by_id(id=submission_id)
        return DataResponse(
            http_code=status.HTTP_200_OK,
            data=OcrSubmissionResponse.model_validate(submission),
        )
    except Exception as e:
        raise CustomException(exception=e)


@router.get(
    "/submissions/{submission_id}/trace",
    response_model=DataResponse[OcrTraceResponse],
    status_code=status.HTTP_200_OK,
)
def get_submission_trace(submission_id: UUID, run: Optional[int] = None) -> Any:
    try:
        trace = ocr_service.get_trace(id=submission_id, run=run)
        return DataResponse(http_code=status.HTTP_200_OK, data=OcrTraceResponse.model_validate(trace, from_attributes=True))
    except Exception as e:
        raise CustomException(exception=e)


@router.post(
    "/submissions/{submission_id}/retry",
    response_model=DataResponse[OcrSubmissionResponse],
    status_code=status.HTTP_202_ACCEPTED,
)
def retry_submission(submission_id: UUID, background_tasks: BackgroundTasks) -> Any:
    try:
        submission = ocr_service.prepare_retry(id=submission_id)
        background_tasks.add_task(process_submission, submission.id)
        return DataResponse(
            http_code=status.HTTP_202_ACCEPTED,
            data=OcrSubmissionResponse.model_validate(submission),
        )
    except Exception as e:
        raise CustomException(exception=e)
