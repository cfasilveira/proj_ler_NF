import logging

from .errors import LeNfError

__version__ = "0.1.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["LeNfError", "__version__"]
