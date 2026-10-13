import os

# Settings của bộ khung bắt buộc có SECRET_KEY; test LLM không dùng DB nên đặt giá trị giả.
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("LLM_PROVIDER", "mock")
