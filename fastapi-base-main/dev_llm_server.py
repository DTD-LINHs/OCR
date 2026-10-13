"""Chạy RIÊNG phần LLM, không cần PostgreSQL. Dùng để dev / demo / cho nhóm gọi thử.

    python dev_llm_server.py
    -> mở http://127.0.0.1:8001/docs

Cấu hình LLM đọc từ file .env (nếu có) hoặc biến môi trường. Mặc định LLM_PROVIDER=mock.
"""

import os

try:  # đọc .env nếu có cài python-dotenv
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
os.environ.setdefault("SECRET_KEY", "dev-secret")  # bộ khung bắt buộc có biến này

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from starlette.middleware.cors import CORSMiddleware  # noqa: E402

from app.api.v1 import api_llm  # noqa: E402
from app.utils.exception_handler import CustomException, custom_error_handler  # noqa: E402

app = FastAPI(title="LLM service (dev, không cần DB)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(api_llm.router, prefix="/api/v1")
app.add_exception_handler(CustomException, custom_error_handler)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
