from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

APP_LOGGER = "ler_nf"

_FORMATO = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def setup_logging(
    log_dir: str = "data/logs",
    level: int = logging.INFO,
    max_bytes: int = 1_000_000,
    backups: int = 5,
    console: bool = False,
) -> logging.Logger:
    """Centraliza a configuração de handlers do app principal.

    Os módulos declaram apenas ``logging.getLogger(__name__)`` com
    ``NullHandler``; quem decide destinos e níveis é o app.

    Exemplo:
        >>> setup_logging("data/logs")
        <Logger ler_nf (INFO)>
    """
    raiz = logging.getLogger(APP_LOGGER)
    if any(isinstance(h, RotatingFileHandler) for h in raiz.handlers):
        return raiz
    os.makedirs(log_dir, exist_ok=True)
    handler = RotatingFileHandler(
        os.path.join(log_dir, "ler_nf.log"),
        maxBytes=max_bytes,
        backupCount=backups,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_FORMATO))
    raiz.addHandler(handler)
    if console:
        c_handler = logging.StreamHandler()
        c_handler.setFormatter(logging.Formatter(_FORMATO))
        raiz.addHandler(c_handler)
    raiz.setLevel(level)
    return raiz
