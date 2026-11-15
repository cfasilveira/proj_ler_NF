from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import tablib

from .contracts import ExtractionResult
from .errors import LeNfError

logger = logging.getLogger(__name__)

_FORMATOS = ("xlsx", "csv", "txt")

_PADRAO_ARQUIVO = re.compile(r"nfse_(\d{4}-\d{2}-\d{2})_(\d{3})\.(xlsx|csv|txt)$")

_CABECALHOS_PT_BR = {
    "cnpj_prestador": "CNPJ",
    "razao_social": "Razão Social",
    "data_emissao": "Data Emissão",
    "numero_nfse": "Número",
    "valor_total": "Valor Total",
    "iss": "ISS",
}


class ExportError(LeNfError):
    """Falha na exportação dos dados."""


class DataExporter:
    """Exportação dos dados extraídos para xlsx/csv/txt (M5).

    Gera arquivos ``nfse_YYYY-MM-DD_NNN.<fmt>`` no diretório
    informado (padrão: ``~/Downloads``), incrementando ``NNN``
    quando já existirem arquivos do mesmo dia.

    Exemplo:
        >>> exporter = DataExporter()
        >>> destino = exporter.export(dados, "xlsx")
        >>> destino.exists()
        True
    """

    def __init__(self, output_dir: "str | Path | None" = None):
        self.output_dir = Path(output_dir) if output_dir else Path.home() / "Downloads"

    def export(self, data: "dict | ExtractionResult", fmt: str = "xlsx") -> Path:
        fmt = fmt.lower()
        if fmt not in _FORMATOS:
            logger.error("Formato de exportação inválido: %s", fmt)
            raise ExportError(f"Formato '{fmt}' não suportado. Use: {', '.join(_FORMATOS)}.")
        registro = data.to_dict() if isinstance(data, ExtractionResult) else dict(data)
        if fmt == "txt":
            destino = self._proximo_caminho(fmt)
            self._escrever_txt(registro, destino)
        else:
            dataset = tablib.Dataset()
            dataset.headers = [_CABECALHOS_PT_BR[c] for c in ExtractionResult.CAMPOS]
            dataset.append([self._formatar(registro.get(c)) for c in ExtractionResult.CAMPOS])
            destino = self._proximo_caminho(fmt)
            try:
                destino.parent.mkdir(parents=True, exist_ok=True)
                conteudo = dataset.export(fmt)
                if isinstance(conteudo, str):
                    conteudo = conteudo.encode("utf-8")
                with open(destino, "wb") as arquivo:
                    arquivo.write(conteudo)
            except OSError as exc:
                logger.exception("Falha de escrita em %s", destino)
                raise ExportError(f"Falha ao escrever '{destino}': {exc}") from exc
        logger.info("Exportado: %s (formato=%s)", destino, fmt)
        return destino

    def _proximo_caminho(self, fmt: str) -> Path:
        hoje = datetime.now().strftime("%Y-%m-%d")
        maior = 0
        if self.output_dir.is_dir():
            for nome in self.output_dir.iterdir():
                match = _PADRAO_ARQUIVO.match(nome.name)
                if match and match.group(1) == hoje:
                    maior = max(maior, int(match.group(2)))
        destino = self.output_dir / f"nfse_{hoje}_{maior + 1:03d}.{fmt}"
        logger.debug("Próximo arquivo: %s", destino)
        return destino

    @staticmethod
    def _formatar(valor) -> str:
        return "" if valor is None else str(valor)

    @staticmethod
    def _escrever_txt(registro: dict, destino: Path) -> None:
        linhas = [
            f"{_CABECALHOS_PT_BR[campo]}: {registro.get(campo, '')}"
            for campo in ExtractionResult.CAMPOS
        ]
        try:
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
        except OSError as exc:
            logger.exception("Falha de escrita em %s", destino)
            raise ExportError(f"Falha ao escrever '{destino}': {exc}") from exc
