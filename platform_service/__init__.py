# platform_service/__init__.py
"""
platform_service 패키지 초기화
외부에서 core 기능을 바로 import 할 수 있도록 re-export
"""

from .core import (
    pipeline,
    build_or_load_vectorstore,
    AZURE_AVAILABLE,
)

from . import constants

# 로깅 설정 초기화
from .logging_config import setup_logging
setup_logging()

__all__ = [
    "pipeline",
    "build_or_load_vectorstore",
    "AZURE_AVAILABLE",
    "constants",
]