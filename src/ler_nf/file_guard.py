from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import safeuploads
from safeuploads import FileValidator
from safeuploads.exceptions import FileSecurityError

from .contracts import GuardResult
from .errors import LeNfError

logger = logging.getLogger(__name__)

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
        self._validator = FileValidator(config=config)

    def validate(self, path: "str | Path") -> GuardResult:
        arquivo = Path(path)
        if not arquivo.is_file():
            logger.error("Arquivo não encontrado: %s", arquivo)
            return GuardResult(ok=False, filename=arquivo.name, error_code="arquivo_inexistente")
        upload = _LocalUpload(arquivo)
        try:
            resultado = asyncio.run(self._validator.validate_image_file(upload))
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
