import os
from typing import Optional
from pydantic import model_validator
from pydantic_settings import BaseSettings
from keycloak.keycloak_openid import KeycloakOpenID


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

_OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
_CONFIGURED_LLM_PROVIDER = os.environ.get("LLM_PROVIDER")
_DEFAULT_LLM_PROVIDER = _CONFIGURED_LLM_PROVIDER or ("openai" if _OPENAI_KEY else "mock")


class Settings(BaseSettings):
    PROJECT_NAME: str = os.environ.get("PROJECT_NAME", "FASTAPI_BASE")
    SECRET_KEY: str = os.environ.get("SECRET_KEY", None)
    API_PREFIX: str = os.environ.get("API_PREFIX", "/api")
    API_VERSIONS: str = os.environ.get("API_VERSIONS", "")
    API_VERSION: str = os.environ.get("API_VERSION", "v1")
    BACKEND_CORS_ORIGINS: str = os.environ.get("BACKEND_CORS_ORIGINS", '["*"]')
    DATABASE_URL: str = (
        f"postgresql+psycopg2://{os.environ.get('POSTGRES_USER')}:{os.environ.get('POSTGRES_PASSWORD')}@{os.environ.get('POSTGRES_HOST')}:{os.environ.get('POSTGRES_PORT')}/{os.environ.get('POSTGRES_DB')}"
    )
    ACCESS_TOKEN_EXPIRE_SECONDS: int = 60 * 60 * 24 * 7  # Token expired after 7 days
    SECURITY_ALGORITHM: str = "HS256"
    LOGGING_CONFIG_FILE: str = os.path.join(BASE_DIR, "logging.ini")
    DEBUG: bool = os.environ.get("DEBUG", "False").lower() == "true"
    KEYCLOAK_SERVER_URL: Optional[str] = os.environ.get("KEYCLOAK_SERVER_URL", None)
    KEYCLOAK_REALM: Optional[str] = os.environ.get("KEYCLOAK_REALM", None)
    KEYCLOAK_CLIENT_ID: Optional[str] = os.environ.get("KEYCLOAK_CLIENT_ID", None)
    KEYCLOAK_CLIENT_SECRET: Optional[str] = os.environ.get("KEYCLOAK_CLIENT_SECRET", None)
    KEYCLOAK_VERIFY: Optional[bool] = os.environ.get("KEYCLOAK_VERIFY", "False").lower() == "true"
    GOOGLE_CLIENT_ID: Optional[str] = os.environ.get("GOOGLE_CLIENT_ID", None)

    # OCR pipeline
    UPLOAD_DIR: str = os.environ.get("UPLOAD_DIR", os.path.join(BASE_DIR, "mount-data/uploads"))
    OCR_MAX_UPLOAD_MB: int = int(os.environ.get("OCR_MAX_UPLOAD_MB", "10"))
    OCR_LANG: str = os.environ.get("OCR_LANG", "en")
    OCR_DEVICE: str = os.environ.get("OCR_DEVICE", "cpu")
    OCR_DET_MODEL: Optional[str] = os.environ.get("OCR_DET_MODEL") or None
    OCR_REC_MODEL: Optional[str] = os.environ.get("OCR_REC_MODEL") or None
    OCR_LOW_CONFIDENCE: float = float(os.environ.get("OCR_LOW_CONFIDENCE", "0.8"))
    # Mean PaddleOCR box confidence below this -> "low_confidence" review reason
    OCR_REVIEW_CONFIDENCE: float = float(os.environ.get("OCR_REVIEW_CONFIDENCE", "0.75"))
    SANDBOX_TIMEOUT: int = int(os.environ.get("SANDBOX_TIMEOUT", "15"))
    SANDBOX_MEMORY: str = os.environ.get("SANDBOX_MEMORY", "256m")
    SANDBOX_CPUS: float = float(os.environ.get("SANDBOX_CPUS", "0.5"))
    SANDBOX_IMAGE: str = os.environ.get("SANDBOX_IMAGE", "sandbox-python:latest")

    # ---- LLM (sửa lỗi OCR + nhận xét bài) ----
    # LLM_PROVIDER=mock      : không gọi LLM thật, dùng để dev/test khi chưa có API key
    # LLM_PROVIDER=openai    : mọi API theo chuẩn OpenAI Chat Completions
    #                          (OpenAI, Gemini, Groq, OpenRouter, Ollama, vLLM, LM Studio...)
    # If OPENAI_API_KEY is present and no provider is specified, use OpenAI.
    # An explicit LLM_PROVIDER=mock still keeps the offline mock mode.
    LLM_PROVIDER: str = _DEFAULT_LLM_PROVIDER
    LLM_BASE_URL: str = os.environ.get(
        "LLM_BASE_URL",
        "https://api.openai.com/v1" if _DEFAULT_LLM_PROVIDER == "openai" else "http://localhost:11434/v1",
    )
    # Dùng LLM_API_KEY; nếu để trống thì lấy OPENROUTER_API_KEY (tên biến nhóm đang dùng)
    LLM_API_KEY: Optional[str] = (
        os.environ.get("LLM_API_KEY")
        or _OPENAI_KEY
        or os.environ.get("OPENROUTER_API_KEY")
        or None
    )
    LLM_MODEL: str = os.environ.get(
        "LLM_MODEL",
        "gpt-4o-mini" if _DEFAULT_LLM_PROVIDER == "openai" else "qwen2.5-coder:7b",
    )
    LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0"))
    LLM_TIMEOUT_SECONDS: float = float(os.environ.get("LLM_TIMEOUT_SECONDS", "180"))
    LLM_JSON_MODE: bool = os.environ.get("LLM_JSON_MODE", "true").lower() == "true"
    # Nếu code sau khi sửa giống bản OCR dưới ngưỡng này -> đánh dấu cần giảng viên xem
    LLM_MIN_SIMILARITY: float = float(os.environ.get("LLM_MIN_SIMILARITY", "0.7"))
    # Từng dòng: dòng LLM sửa giống dòng OCR dưới ngưỡng này (bỏ khoảng trắng, chữ thường) -> coi là sửa quá nhiều
    LLM_MIN_LINE_SIMILARITY: float = float(os.environ.get("LLM_MIN_LINE_SIMILARITY", "0.5"))
    # Xử lý thay đổi đáng ngờ của LLM ở /correct-ocr:
    #   flag   : giữ nguyên code LLM, chỉ đánh dấu (suspicious_changes, uncertain, review_reasons)
    #   revert : đặt lại token / dòng OCR gốc cho các thay đổi đáng ngờ
    LLM_GUARD_MODE: str = os.environ.get("LLM_GUARD_MODE", "flag").lower()

    @model_validator(mode="after")
    def _llm_key_fallback(self):
        if not self.LLM_API_KEY:
            self.LLM_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or None
        return self


settings = Settings()

if (
    settings.KEYCLOAK_SERVER_URL != None
    and settings.KEYCLOAK_REALM != None
    and settings.KEYCLOAK_CLIENT_ID != None
    and settings.KEYCLOAK_CLIENT_SECRET != None
    and settings.KEYCLOAK_VERIFY != None
):
    keycloak_openid = KeycloakOpenID(
        server_url=settings.KEYCLOAK_SERVER_URL,
        realm_name=settings.KEYCLOAK_REALM,
        client_id=settings.KEYCLOAK_CLIENT_ID,
        client_secret_key=settings.KEYCLOAK_CLIENT_SECRET,
        verify=settings.KEYCLOAK_VERIFY,
    )
else:
    keycloak_openid = None


def get_openid_config():
    if keycloak_openid == None:
        return {}
    return keycloak_openid.well_known()
