import logging
import logging.config
from pathlib import Path
import os
import json

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

class JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
            "source": f"{os.path.basename(record.pathname)}:{record.lineno}",
            "function": record.funcName
        }
        if hasattr(record, "extra_data"):
            base.update(record.extra_data)
        return json.dumps(base, ensure_ascii=False)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {"()": JsonFormatter},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": LOG_DIR / "app.log",
            "encoding": "utf-8",
            "formatter": "json",
            "level": "INFO",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
