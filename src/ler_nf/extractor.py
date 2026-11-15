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
_DATA_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
_NUMERO_RE = re.compile(r"\b(\d{1,9})\s*/\s*(\d{2,9})\b")
_NUMERO_SO_RE = re.compile(r"\b\d{5,9}\b")
_MONEY_RE = re.compile(r"(?:R\$)?\s*(\d{1,3}(?:\.\d{3})+,\d{2}|\d+,\d{2})")
_ANCORA_ISS_RE = re.compile(r"\bISS\b", re.IGNORECASE)

_ANCORAS_VALOR_TOTAL = ("VALOR TOTAL", "TOTAL DA NOTA", "VALOR LIQUIDO", "VALOR LÍQUIDO", "TOTAL")
_ANCORAS_RAZAO = ("RAZAO SOCIAL", "RAZÃO SOCIAL", "PRESTADOR")
_ANCORAS_DATA = ("DATA EMISSAO", "DATA DE EMISSÃO", "DATA EMISSÃO", "DATA")
_ANCORAS_NUMERO = ("NUMERO DA NOTA", "NÚMERO DA NOTA", "NUMERO", "NÚMERO", "Nº", "N°")
_ANCORAS_ISS = ("ISS",)


class ExtractionError(LeNfError):
    """Falha no processo de extração de dados."""


@dataclass
class _Linha:
    texto: str
    confianca: float
    topo: int


@dataclass
class _Palavra:
    texto: str
    confianca: float
    left: int
    top: int
    width: int
    height: int
    linha: tuple


class NfseExtractor:
    """Extração de campos de NFS-e via Tesseract + âncoras (M4).

    Combina busca por termos-chave com regexes tolerantes e, quando
    disponível, re-OCR posicional (recortes da imagem em tons de
    cinza) para campos difíceis como o número da nota. Cada campo
    recebe um índice de confiança baseado na confiança do OCR.

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

    def extract(
        self,
        imagem: "str | Path | np.ndarray | Image.Image",
        imagem_cinza: "np.ndarray | None" = None,
    ) -> ExtractionResult:
        linhas, texto_completo, palavras = self._ocr(imagem)
        resultado = ExtractionResult()
        resultado.confianca = {campo: 0.0 for campo in ExtractionResult.CAMPOS}

        resultado.cnpj_prestador = self._extrair_cnpj(linhas, texto_completo, resultado.confianca)
        resultado.razao_social = self._extrair_ancorada(
            linhas, _ANCORAS_RAZAO, "razao_social", resultado.confianca
        )
        resultado.data_emissao = self._extrair_data(linhas, resultado.confianca)
        resultado.numero_nfse = self._extrair_numero(
            linhas, palavras, imagem_cinza, resultado.confianca
        )
        resultado.valor_total = self._extrair_valor(
            linhas, _ANCORAS_VALOR_TOTAL, "valor_total", resultado.confianca
        )
        resultado.iss = self._extrair_iss(
            linhas, palavras, resultado.confianca
        )

        preenchidos = sum(1 for c in ExtractionResult.CAMPOS if getattr(resultado, c) is not None)
        logger.info(
            "Extração concluída: %d/%d campos preenchidos (confiança média %.2f)",
            preenchidos,
            len(ExtractionResult.CAMPOS),
            sum(resultado.confianca.values()) / len(ExtractionResult.CAMPOS),
        )
        return resultado

    def _ocr(self, imagem) -> tuple[list[_Linha], str, list[_Palavra]]:
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
        palavras = self._coletar_palavras(dados)
        logger.debug("OCR produziu %d linhas e %d palavras", len(linhas), len(palavras))
        return linhas, texto_completo, palavras

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
    def _coletar_palavras(dados: dict) -> list[_Palavra]:
        palavras = []
        for i in range(len(dados["text"])):
            texto = (dados["text"][i] or "").strip()
            conf = float(dados["conf"][i])
            if not texto or conf < 0:
                continue
            palavras.append(
                _Palavra(
                    texto=texto,
                    confianca=conf / 100.0,
                    left=dados["left"][i],
                    top=dados["top"][i],
                    width=dados["width"][i],
                    height=dados["height"][i],
                    linha=(
                        dados["block_num"][i],
                        dados["par_num"][i],
                        dados["line_num"][i],
                    ),
                )
            )
        return palavras

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
        for match in _DATA_RE.finditer(" ".join(l.texto for l in linhas)):
            bruto = match.group(0)
            dia, mes, ano = bruto.split("/")
            normalizada = f"{int(dia):02d}/{int(mes):02d}/{ano}"
            idx = self._indice_da_ancora(linhas, _ANCORAS_DATA)
            linha = linhas[idx] if idx is not None else next(
                (l for l in linhas if bruto in l.texto), None
            )
            conf = linha.confianca if linha else 0.7
            confiancas["data_emissao"] = round(min(0.95, conf), 2)
            logger.debug("Data extraída: %s -> %s", bruto, normalizada)
            return normalizada
        logger.debug("Data não encontrada.")
        return None

    def _extrair_numero(
        self, linhas, palavras, imagem_cinza, confiancas: dict
    ) -> str | None:
        idx = self._indice_da_ancora(linhas, _ANCORAS_NUMERO)
        janela = [linhas[idx].texto] if idx is not None else []
        if idx is not None:
            janela += [l.texto for l in linhas[idx + 1: idx + 1 + self.janela]]
        for texto in janela:
            candidato = self._candidato_numero(texto, excluir_datas=True)
            if candidato:
                confiancas["numero_nfse"] = round(
                    (linhas[idx].confianca if idx is not None else 0.7), 2
                )
                logger.debug("Número extraído (âncora): %s", candidato)
                return candidato
        if imagem_cinza is not None:
            candidato = self._reocr_numero(palavras, imagem_cinza, confiancas)
            if candidato:
                return candidato
        for linha in linhas:
            candidato = self._candidato_numero(linha.texto, excluir_datas=True)
            if candidato:
                confiancas["numero_nfse"] = 0.7
                logger.debug("Número extraído (texto completo): %s", candidato)
                return candidato
        logger.debug("Número da NFS-e não encontrado.")
        return None

    @staticmethod
    def _candidato_numero(texto: str, excluir_datas: bool = False) -> str | None:
        padrao = re.compile(r"(?<![\d.])(\d{1,9})\s*/\s*(\d{2,9})(?![\d])")
        match = padrao.search(texto)
        if not match:
            return None
        numero, sufixo = match.group(1), match.group(2)
        if excluir_datas:
            trecho = match.group(0)
            for data_match in _DATA_RE.finditer(texto):
                if data_match.group(0) in trecho or trecho in data_match.group(0):
                    return None
            if len(sufixo) == 4 and 1900 <= int(sufixo) <= 2100:
                return None
            if len(sufixo) < 3:
                return None
        return f"{numero}/{sufixo}"

    def _reocr_numero(self, palavras, imagem_cinza, confiancas: dict) -> str | None:
        """Segunda passada: recorte ao redor da âncora 'Número da Nota'."""
        for palavra in palavras:
            if not re.fullmatch(r"N[úuÚÚ]|N[º°]|NUMERO|NÚMERO", palavra.texto):
                continue
            left = max(0, palavra.left - 40)
            top = max(0, palavra.top - int(palavra.height * 0.6))
            right = palavra.left + int(palavra.width * 12)
            bottom = palavra.top + int(palavra.height * 4.0)
            altura_img, largura_img = imagem_cinza.shape
            recorte = imagem_cinza[top: min(bottom, altura_img), left: min(right, largura_img)]
            if recorte.size == 0:
                continue
            pilha = Image.fromarray(recorte)
            pilha = pilha.resize((pilha.width * 2, pilha.height * 2), Image.LANCZOS)
            try:
                texto = pytesseract.image_to_string(
                    pilha, lang=self.lang, config="--oem 3 --psm 6"
                )
            except Exception:
                logger.exception("Falha no re-OCR do número da nota")
                continue
            match = _NUMERO_RE.search(texto) or _NUMERO_SO_RE.search(texto)
            if not match:
                continue
            bruto = match.group(0).strip("| ")
            digits = re.sub(r"\D", "", bruto)
            if len(digits) >= 5:
                confiancas["numero_nfse"] = 0.75
                logger.debug("Número extraído (re-OCR posicional): %s", bruto)
                return bruto
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

    def _extrair_iss(self, linhas, palavras, confiancas: dict) -> float | None:
        alvo = self._valor_posicional(palavras, _ANCORA_ISS_RE, janela_linhas=2.5)
        if alvo is not None:
            valor, conf = alvo
            confiancas["iss"] = round(conf, 2)
            logger.debug("ISS extraído (posicional): %.2f (conf %.2f)", valor, conf)
            return valor
        return self._extrair_valor(linhas, _ANCORAS_ISS, "iss", confiancas)

    def _valor_posicional(
        self, palavras, padrao_ancora: re.Pattern, janela_linhas: float = 2.5
    ) -> "tuple[float, float] | None":
        """Busca um valor monetário na coluna abaixo da âncora encontrada."""
        ancora = next((p for p in palavras if padrao_ancora.search(p.texto)), None)
        if ancora is None:
            return None
        altura_media = ancora.height or 20
        limite_topo = ancora.top + altura_media * janela_linhas
        x_esq = ancora.left - ancora.width * 2.0
        x_dir = ancora.left + ancora.width * 6.0
        candidatos = []
        for palavra in palavras:
            if palavra.top <= ancora.top:
                continue
            if palavra.top > limite_topo:
                continue
            centro = palavra.left + palavra.width / 2
            if centro < x_esq or centro > x_dir:
                continue
            valor = self.parse_money(palavra.texto)
            if valor is not None:
                candidatos.append((palavra.top, valor, palavra.confianca))
        if not candidatos:
            return None
        candidatos.sort()
        _, valor, conf = candidatos[0]
        return valor, conf

    def _indice_da_ancora(self, linhas, ancoras) -> int | None:
        for ancora in ancoras:
            padrao = re.compile(rf"\b{re.escape(ancora)}\b", re.IGNORECASE)
            for i, linha in enumerate(linhas):
                if padrao.search(linha.texto):
                    return i
        return None
