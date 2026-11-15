from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import magic
from PIL import Image, UnidentifiedImageError
import safeuploads
from safeuploads import FileValidator
from safeuploads.config import FileSecurityConfig
from safeuploads.exceptions import FileSecurityError, FileSignatureError

from .contracts import GuardResult
from .errors import LeNfError

logger = logging.getLogger(__name__)


class ConfigImagensEstendida(FileSecurityConfig):
    """Formatos de imagem aceitos, incluindo containers de capturas de tela."""

    ALLOWED_IMAGE_TYPES = {
        "image/jpeg": frozenset({".jpg", ".jpeg"}),
        "image/png": frozenset({".png"}),
        "image/webp": frozenset({".webp"}),
        "image/avif": frozenset({".avif"}),
        "image/heif": frozenset({".heif", ".heic"}),
        "image/heic": frozenset({".heif", ".heic"}),
    }


_CODIGOS_ERRO: list[tuple[type[Exception], str]] = [
    (safeuploads.FilenameSecurityError, "nome_invalido"),
    (safeuploads.ExtensionSecurityError, "extensao_nao_permitida"),
    (safeuploads.FileSizeError, "tamanho_invalido"),
    (safeuploads.MimeTypeError, "tipo_nao_permitido"),
    (safeuploads.FileSignatureError, "conteudo_corrompido"),
    (safeuploads.ImageDimensionError, "dimensoes_invalidas"),
    (safeuploads.ZipBombError, "zip_bomb"),
    (safeuploads.ZipContentError, "zip_suspeito"),
    (safeuploads.CompressionSecurityError, "compressao_suspeita"),
    (safeuploads.ContentThreatError, "conteudo_suspeito"),
    (safeuploads.ResourceLimitError, "limite_excedido"),
    (safeuploads.FileProcessingError, "processamento_falhou"),
    (safeuploads.FileValidationError, "validacao_falhou"),
]


class FileGuardError(LeNfError):
    """Falha inesperada na validação de segurança do arquivo."""


class _LocalUpload:
    """Adaptador de arquivo local para o protocolo UploadFile do safeuploads."""

    def __init__(self, path: Path):
        self.path = path
        self.filename = path.name
        self.size = path.stat().st_size if path.exists() else 0
        self._io = None

    def _abrir(self):
        if self._io is None:
            self._io = self.path.open("rb")
        return self._io

    async def read(self, size: int = -1) -> bytes:
        return self._abrir().read(size if size >= 0 else -1)

    async def seek(self, offset: int) -> int:
        return self._abrir().seek(offset)


class FileGuard:
    """Validação de segurança de arquivos de imagem (M2).

    Verifica magic bytes, MIME, extensão, tamanho, dimensões e
    ameaças de conteúdo via ``safeuploads.FileValidator``.

    Exemplo:
        >>> guard = FileGuard()
        >>> resultado = guard.validate("foto.png")
        >>> resultado.ok
        True
    """

    def __init__(self, config: "safeuploads.FileSecurityConfig | None" = None):
        self._validator = FileValidator(config=config or ConfigImagensEstendida())

    def validate(self, path: "str | Path") -> GuardResult:
        arquivo = Path(path)
        if not arquivo.is_file():
            logger.error("Arquivo não encontrado: %s", arquivo)
            return GuardResult(ok=False, filename=arquivo.name, error_code="arquivo_inexistente")
        upload = _LocalUpload(arquivo)
        try:
            resultado = asyncio.run(self._validator.validate_image_file(upload))
        except FileSignatureError:
            if self._decodifica_como_imagem(arquivo):
                mime = self._mime_de_conteudo(arquivo)
                logger.warning(
                    "Assinatura não catalogada, mas o conteúdo decodifica como imagem: %s (%s)",
                    arquivo.name,
                    mime,
                )
                return GuardResult(ok=True, filename=arquivo.name, mime_type=mime)
            logger.warning("Assinatura inválida e conteúdo não decodificável: %s", arquivo.name)
            return GuardResult(ok=False, filename=arquivo.name, error_code="conteudo_corrompido")
        except FileSecurityError as exc:
            return self._rejeitar(arquivo, exc)
        except Exception as exc:
            logger.exception("Falha inesperada validando %s", arquivo.name)
            raise FileGuardError(f"Falha inesperada validando '{arquivo.name}': {exc}") from exc
        finally:
            if upload._io is not None:
                upload._io.close()
        logger.info(
            "Arquivo aceito: %s (mime=%s)",
            resultado.filename,
            resultado.mime_type,
        )
        return GuardResult(ok=True, filename=resultado.filename, mime_type=resultado.mime_type)

    @staticmethod
    def _decodifica_como_imagem(arquivo: Path) -> bool:
        try:
            with Image.open(arquivo) as imagem:
                imagem.verify()
            with Image.open(arquivo) as imagem:
                imagem.load()
            return True
        except (UnidentifiedImageError, OSError, ValueError):
            return False

    @staticmethod
    def _mime_de_conteudo(arquivo: Path) -> str | None:
        try:
            return magic.from_file(str(arquivo), mime=True)
        except Exception:
            return None

    @staticmethod
    def _rejeitar(arquivo: Path, exc: FileSecurityError) -> GuardResult:
        codigo = "arquivo_rejeitado"
        for tipo, codigo_mapeado in _CODIGOS_ERRO:
            if isinstance(exc, tipo):
                codigo = codigo_mapeado
                break
        logger.warning(
            "Arquivo rejeitado: %s (codigo=%s): %s",
            arquivo.name,
            codigo,
            exc,
        )
        return GuardResult(ok=False, filename=arquivo.name, error_code=codigo)
