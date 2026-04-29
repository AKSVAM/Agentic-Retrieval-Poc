import logging
import logging.handlers
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> None:
    """Configure console + rotating-file logging for the GraphRAG server."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    fmt = "%(asctime)s [%(levelname)-8s] %(name)-40s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Rotating file: 5 MB per file, keep 5 backups → max 25 MB on disk
    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "graphrag.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Silence noisy third-party loggers — their warnings still come through
    for noisy in ("httpx", "httpcore", "openai._base_client", "chromadb",
                  "uvicorn.access", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
