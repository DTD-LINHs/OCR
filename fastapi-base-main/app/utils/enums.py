import enum


class UserRole(enum.Enum):
    ADMIN = "Administrator"
    USER = "Normal user"
    SPECIALIST = "Specialist"
    GUEST = "Guest"


class UserGender(enum.Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"


class OcrStatus(str, enum.Enum):
    PENDING = "PENDING"
    OCR_RUNNING = "OCR_RUNNING"
    SANDBOX_CHECKING = "SANDBOX_CHECKING"
    LLM_CORRECTING = "LLM_CORRECTING"
    DONE = "DONE"
    NEED_REVIEW = "NEED_REVIEW"
    FAILED = "FAILED"


class CodeStage(str, enum.Enum):
    OCR_RAW = "OCR_RAW"
    SERVICE_NORMALIZED = "SERVICE_NORMALIZED"
    LLM_CORRECTED = "LLM_CORRECTED"


class LlmPurpose(str, enum.Enum):
    TRANSCRIBE = "TRANSCRIBE"
