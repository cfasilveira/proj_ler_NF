from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .contracts import ExtractionResult, GuardResult, QualityCheckResult, TemplateRecord
from .errors import LeNfError
from .exporter import DataExporter, ExportError
from .extractor import ExtractionError, NfseExtractor
from .file_guard import FileGuard, FileGuardError
from .image_preprocess import ImagePreprocessor, PreprocessError, ProcessedImage
from .log_config import setup_logging
from .quality_checker import QualityChecker, QualityCheckError
from .template_store import TemplateNotFoundError, TemplateStore, TemplateStoreError

logger = logging.getLogger(__name__)


class Status(Enum):
    OK = "ok"
    ARQUIVO_REJEITADO = "arquivo_rejeitado"
    QUALIDADE_INSUFICIENTE = "qualidade_insuficiente"
    PRE_PROCESSAMENTO_FALHOU = "pre_processamento_falhou"
    LAYOUT_DESCONHECIDO = "layout_desconhecido"
    EXTRACAO_FALHOU = "extracao_falhou"


@dataclass
class PipelineResult:
    status: Status
    message: str | None = None
    data: ExtractionResult | None = None
    template: TemplateRecord | None = None
    template_distance: int | None = None
    guard: GuardResult | None = None
    quality: QualityCheckResult | None = None


class NfsePipeline:
    """Orquestrador do fluxo: validar → qualidade → pré-processar →
    identificar layout → extrair → exportar.

    Exemplo:
        >>> pipeline = NfsePipeline(data_dir="data")
        >>> resultado = pipeline.processar("foto.jpg")
        >>> resultado.status
        <Status.OK: 'ok'>
    """

    def __init__(
        self,
        data_dir: "str | Path" = "data",
        limiar_hamming: int = 12,
        output_dir: "str | Path | None" = None,
        quality_thresholds: "dict[str, float] | None" = None,
    ):
        self.data_dir = Path(data_dir)
        self.guard = FileGuard()
        self.quality = QualityChecker(thresholds=quality_thresholds)
        self.preprocess = ImagePreprocessor()
        self.templates = TemplateStore(
            self.data_dir / "templates.sqlite", limiar_hamming=limiar_hamming
        )
        self.extractor = NfseExtractor()
        self.exporter = DataExporter(output_dir)
        logger.info("Pipeline iniciado (data_dir=%s)", self.data_dir)

    def processar(
        self, caminho: "str | Path", progresso: "Callable[[str], None] | None" = None
    ) -> PipelineResult:
        caminho = Path(caminho)
        logger.info("Processando: %s", caminho)

        progresso and progresso("Validando arquivo")
        guard = self.guard.validate(caminho)
        if not guard.ok:
            return PipelineResult(
                status=Status.ARQUIVO_REJEITADO,
                message=self._mensagem_arquivo_rejeitado(guard),
                guard=guard,
            )

        progresso and progresso("Analisando qualidade da foto")
        quality = self.quality.check(str(caminho))
        if not quality.ok:
            logger.info("Qualidade insuficiente: %s", quality.message)
            return PipelineResult(
                status=Status.QUALIDADE_INSUFICIENTE,
                message=quality.message,
                guard=guard,
                quality=quality,
            )

        progresso and progresso("Preparando imagem")
        try:
            processada = self.preprocess.processar_desde_arquivo(str(caminho))
        except PreprocessError as exc:
            return PipelineResult(
                status=Status.PRE_PROCESSAMENTO_FALHOU,
                message=str(exc),
                guard=guard,
                quality=quality,
            )

        progresso and progresso("Identificando layout")
        template, distancia = None, None
        try:
            template, distancia = self.templates.buscar(
                Image.fromarray(processada.grayscale)
            )
        except TemplateNotFoundError as exc:
            logger.info("Layout não reconhecido: %s", exc)
            return PipelineResult(
                status=Status.LAYOUT_DESCONHECIDO,
                message=str(exc),
                guard=guard,
                quality=quality,
                template_distance=exc.distance,
            )
        except TemplateStoreError as exc:
            raise LeNfError(f"Falha no banco de templates: {exc}") from exc

        progresso and progresso("Extraindo dados")
        try:
            dados = self.extractor.extract(processada.ocr)
        except ExtractionError as exc:
            return PipelineResult(
                status=Status.EXTRACAO_FALHOU,
                message=str(exc),
                guard=guard,
                quality=quality,
                template=template,
                template_distance=distancia,
            )

        logger.info("Processamento concluído para %s (layout '%s')", caminho.name, template.nome)
        return PipelineResult(
            status=Status.OK,
            data=dados,
            template=template,
            template_distance=distancia,
            guard=guard,
            quality=quality,
        )

    def exportar_e_descartar(
        self, dados: "dict | ExtractionResult", fmt: str, caminho_original: "str | Path"
    ) -> Path:
        """Exporta os dados e apaga a imagem original (descarte automático)."""
        destino = self.exporter.export(dados, fmt)
        original = Path(caminho_original)
        if original.is_file():
            original.unlink()
            logger.info("Imagem original descartada: %s", original)
        return destino

    def cadastrar_layout(
        self,
        caminho_imagem: "str | Path",
        nome: str,
        municipio: str,
        campos_mapeados: dict | None = None,
    ):
        try:
            processada = self.preprocess.processar_desde_arquivo(str(caminho_imagem))
            return self.templates.cadastrar(
                Image.fromarray(processada.grayscale), nome, municipio, campos_mapeados
            )
        except TemplateStoreError as exc:
            raise LeNfError(f"Falha ao cadastrar layout: {exc}") from exc

    @staticmethod
    def _mensagem_arquivo_rejeitado(guard: GuardResult) -> str:
        mensagens = {
            "arquivo_inexistente": "Arquivo não encontrado.",
            "extensao_nao_permitida": "Formato de arquivo não permitido. Envie uma imagem (PNG, JPEG ou WebP).",
            "tamanho_invalido": "Arquivo vazio ou maior que o limite aceito.",
            "tipo_nao_permitido": "Este tipo de arquivo não é aceito. Envie uma imagem.",
            "conteudo_corrompido": "O arquivo parece corrompido ou adulterado.",
            "dimensoes_invalidas": "A imagem tem dimensões inválidas.",
            "zip_bomb": "Arquivo suspeito (compactação perigosa detectada).",
            "conteudo_suspeito": "Arquivo suspeito e bloqueado por segurança.",
        }
        return mensagens.get(guard.error_code or "", "Arquivo rejeitado por segurança.")
