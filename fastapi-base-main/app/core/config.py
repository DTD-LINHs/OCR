import os
from typing import Optional
from pydantic_settings import BaseSettings
from keycloak.keycloak_openid import KeycloakOpenID


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


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
    OCR_REVIEW_CONFIDENCE: float = float(os.environ.get("OCR_REVIEW_CONFIDENCE", "0.75"))
    OCR_MAX_FIX_ROUNDS: int = int(os.environ.get("OCR_MAX_FIX_ROUNDS", "2"))
    OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY") or None
    OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o")
    OPENAI_TIMEOUT: int = int(os.environ.get("OPENAI_TIMEOUT", "120"))
    SANDBOX_TIMEOUT: int = int(os.environ.get("SANDBOX_TIMEOUT", "15"))
    SANDBOX_MEMORY: str = os.environ.get("SANDBOX_MEMORY", "256m")
    SANDBOX_CPUS: float = float(os.environ.get("SANDBOX_CPUS", "0.5"))
    SANDBOX_RUNTIME: Optional[str] = os.environ.get("SANDBOX_RUNTIME") or None


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
