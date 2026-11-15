from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image

from .contracts import ExtractionResult
from .errors import LeNfError

logger = logging.getLogger(__name__)

_CNPJ_RE = re.compile(r"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}")
_DATA_RE = re.compile(r"\d{2}/\d{2}/\d{4}")
_NUMERO_RE = re.compile(r"\b(\d{1,6})\s*/\s*(\d{2,6})\b")
_MONEY_RE = re.compile(r"(?:R\$)?\s*(\d{1,3}(?:\.\d{3})+,\d{2}|\d+,\d{2})")

_ANCORAS_VALOR_TOTAL = ("VALOR TOTAL", "TOTAL DA NOTA", "VALOR LIQUIDO", "VALOR LÍQUIDO", "TOTAL")
_ANCORAS_RAZAO = ("RAZAO SOCIAL", "RAZÃO SOCIAL", "PRESTADOR")
_ANCORAS_DATA = ("DATA EMISSAO", "DATA DE EMISSÃO", "DATA EMISSÃO", "DATA")
_ANCORAS_NUMERO = ("NUMERO", "NÚMERO", "Nº", "N°")
_ANCORAS_ISS = ("ISS",)


class ExtractionError(LeNfError):
    """Falha no processo de extração de dados."""


@dataclass
class _Linha:
    texto: str
    confianca: float
    topo: int


class NfseExtractor:
    """Extração de campos de NFS-e via Tesseract + âncoras (M4).

    Procura termos-chave ("CNPJ", "VALOR TOTAL", ...) e captura os
    valores adjacentes com regexes tolerantes. Cada campo recebe um
    índice de confiança baseado na confiança média do OCR na linha
    usada.

    Exemplo:
        >>> extractor = NfseExtractor()
        >>> dados = extractor.extract("nota.png")
        >>> dados.valor_total
        1500.0
    """

    def __init__(
        self,
        lang: str = "por",
        config: str = "--oem 3 --psm 6",
        janela_ancora_linhas: int = 2,
    ):
        self.lang = lang
        self.config = config
        self.janela = janela_ancora_linhas

    def extract(self, imagem: "str | Path | np.ndarray | Image.Image") -> ExtractionResult:
        linhas, texto_completo = self._ocr(imagem)
        resultado = ExtractionResult()
        resultado.confianca = {campo: 0.0 for campo in ExtractionResult.CAMPOS}

        resultado.cnpj_prestador = self._extrair_cnpj(linhas, texto_completo, resultado.confianca)
        resultado.razao_social = self._extrair_ancorada(
            linhas, _ANCORAS_RAZAO, "razao_social", resultado.confianca
        )
        resultado.data_emissao = self._extrair_data(linhas, resultado.confianca)
        resultado.numero_nfse = self._extrair_numero(linhas, resultado.confianca)
        resultado.valor_total = self._extrair_valor(linhas, _ANCORAS_VALOR_TOTAL, "valor_total", resultado.confianca)
        resultado.iss = self._extrair_valor(linhas, _ANCORAS_ISS, "iss", resultado.confianca)

        preenchidos = sum(1 for c in ExtractionResult.CAMPOS if getattr(resultado, c) is not None)
        logger.info(
            "Extração concluída: %d/%d campos preenchidos (confiança média %.2f)",
            preenchidos,
            len(ExtractionResult.CAMPOS),
            sum(resultado.confianca.values()) / len(ExtractionResult.CAMPOS),
        )
        return resultado

    def _ocr(self, imagem) -> tuple[list[_Linha], str]:
        try:
            imagem = self._converter(imagem)
            dados = pytesseract.image_to_data(
                imagem,
                lang=self.lang,
                config=self.config,
                output_type=pytesseract.Output.DICT,
            )
            texto_completo = pytesseract.image_to_string(
                imagem, lang=self.lang, config=self.config
            )
        except Exception as exc:
            logger.exception("Falha no OCR")
            raise ExtractionError(f"Falha no OCR: {exc}") from exc
        linhas = self._agrupar_linhas(dados)
        logger.debug("OCR produziu %d linhas", len(linhas))
        return linhas, texto_completo

    @staticmethod
    def _converter(imagem):
        if isinstance(imagem, np.ndarray):
            return Image.fromarray(imagem)
        if isinstance(imagem, (str, Path)):
            return Image.open(imagem)
        if isinstance(imagem, Image.Image):
            return imagem
        raise ExtractionError(
            f"imagem deve ser caminho, np.ndarray ou PIL.Image; recebi {type(imagem).__name__}"
        )

    @staticmethod
    def _agrupar_linhas(dados: dict) -> list[_Linha]:
        grupos: dict[tuple, dict] = {}
        for i in range(len(dados["text"])):
            palavra = (dados["text"][i] or "").strip()
            conf = float(dados["conf"][i])
            if not palavra or conf < 0:
                continue
            chave = (
                dados["block_num"][i],
                dados["par_num"][i],
                dados["line_num"][i],
            )
            grupo = grupos.setdefault(
                chave,
                {"palavras": [], "confs": [], "topo": dados["top"][i]},
            )
            grupo["palavras"].append(palavra)
            grupo["confs"].append(conf)
        linhas = [
            _Linha(
                texto=" ".join(g["palavras"]),
                confianca=sum(g["confs"]) / len(g["confs"]) / 100.0,
                topo=g["topo"],
            )
            for g in grupos.values()
        ]
        linhas.sort(key=lambda l: l.topo)
        return linhas

    @staticmethod
    def parse_money(valor: str) -> float | None:
        match = _MONEY_RE.search(valor)
        if not match:
            return None
        bruto = match.group(1)
        if "," in bruto:
            bruto = bruto.replace(".", "").replace(",", ".")
        try:
            return float(bruto)
        except ValueError:
            return None

    def _extrair_cnpj(self, linhas, texto: str, confiancas: dict) -> str | None:
        match = _CNPJ_RE.search(texto)
        if not match:
            logger.debug("CNPJ não encontrado.")
            return None
        bruto = match.group(0).replace(".", "").replace("/", "").replace("-", "")
        formatado = f"{bruto[:2]}.{bruto[2:5]}.{bruto[5:8]}/{bruto[8:12]}-{bruto[12:]}"
        linha = next((l for l in linhas if _CNPJ_RE.search(l.texto)), None)
        conf = max(0.9, linha.confianca) if linha else 0.9
        confiancas["cnpj_prestador"] = round(conf, 2)
        logger.debug("CNPJ extraído: %s (conf %.2f)", formatado, conf)
        return formatado

    def _extrair_ancorada(self, linhas, ancoras, campo: str, confiancas: dict) -> str | None:
        idx = self._indice_da_ancora(linhas, ancoras)
        if idx is None:
            logger.debug("%s: âncora não encontrada.", campo)
            return None
        linha = linhas[idx]
        texto = linha.texto
        for ancora in ancoras:
            padrao = re.compile(rf"\b{re.escape(ancora)}\b", re.IGNORECASE)
            match_ancora = padrao.search(texto)
            if match_ancora:
                restante = texto[match_ancora.end():].lstrip(":").strip()
                if len(restante) >= 3:
                    confiancas[campo] = round(linha.confianca, 2)
                    return restante
        candidatos = [
            linhas[j].texto
            for j in range(idx + 1, min(idx + 1 + self.janela, len(linhas)))
        ]
        for candidato in candidatos:
            if len(candidato) >= 3 and not self._parece_outra_ancora(candidato):
                confiancas[campo] = round(linha.confianca * 0.9, 2)
                return candidato
        logger.debug("%s: âncora encontrada mas valor não identificado.", campo)
        return None

    @staticmethod
    def _parece_outra_ancora(texto: str) -> bool:
        superior = texto.upper()
        for ancora in (*_ANCORAS_VALOR_TOTAL, *_ANCORAS_DATA, *_ANCORAS_NUMERO, "CNPJ"):
            if ancora in superior:
                return True
        return False

    def _extrair_data(self, linhas, confiancas: dict) -> str | None:
        match = _DATA_RE.search(" ".join(l.texto for l in linhas))
        if not match:
            logger.debug("Data não encontrada.")
            return None
        idx = self._indice_da_ancora(linhas, _ANCORAS_DATA)
        linha = linhas[idx] if idx is not None else next(
            (l for l in linhas if _DATA_RE.search(l.texto)), None
        )
        conf = linha.confianca if linha else 0.7
        confiancas["data_emissao"] = round(min(0.95, conf), 2)
        logger.debug("Data extraída: %s", match.group(0))
        return match.group(0)

    def _extrair_numero(self, linhas, confiancas: dict) -> str | None:
        idx = self._indice_da_ancora(linhas, _ANCORAS_NUMERO)
        janela = [linhas[idx].texto] if idx is not None else []
        if idx is not None:
            janela += [l.texto for l in linhas[idx + 1: idx + 1 + self.janela]]
        for texto in janela:
            match = _NUMERO_RE.search(texto)
            if match:
                confiancas["numero_nfse"] = round(
                    (linhas[idx].confianca if idx is not None else 0.7), 2
                )
                logger.debug("Número extraído (âncora): %s/%s", match.group(1), match.group(2))
                return f"{match.group(1)}/{match.group(2)}"
        match = _NUMERO_RE.search(" ".join(l.texto for l in linhas))
        if match:
            numero, ano = match.group(1), match.group(2)
            if len(ano) == 4 and 1900 <= int(ano) <= 2100:
                confiancas["numero_nfse"] = 0.7
                logger.debug("Número extraído (texto completo): %s/%s", numero, ano)
                return f"{numero}/{ano}"
        logger.debug("Número da NFS-e não encontrado.")
        return None

    def _extrair_valor(self, linhas, ancoras, campo: str, confiancas: dict) -> float | None:
        idx = self._indice_da_ancora(linhas, ancoras)
        if idx is not None:
            janela = linhas[idx: idx + 1 + self.janela]
            for linha in janela:
                valor = self.parse_money(linha.texto)
                if valor is not None:
                    confiancas[campo] = round(linha.confianca, 2)
                    logger.debug("%s extraído: %.2f (conf %.2f)", campo, valor, linha.confianca)
                    return valor
        logger.debug("%s: âncora ou valor não encontrado.", campo)
        return None

    def _indice_da_ancora(self, linhas, ancoras) -> int | None:
        for ancora in ancoras:
            padrao = re.compile(rf"\b{re.escape(ancora)}\b", re.IGNORECASE)
            for i, linha in enumerate(linhas):
                if padrao.search(linha.texto):
                    return i
        return None
