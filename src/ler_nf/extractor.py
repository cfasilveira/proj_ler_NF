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
_ANCORA_ISS_RE = re.compile(r"\bI[S5]S\b", re.IGNORECASE)

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

        resultado.cnpj_prestador = self._extrair_cnpj(
            linhas, texto_completo, palavras, imagem_cinza, resultado.confianca
        )
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
            palavras = self._coletar_palavras(dados)
            config_psm4 = self.config.replace("psm 6", "psm 4")
            if config_psm4 != self.config:
                dados_psm4 = pytesseract.image_to_data(
                    imagem,
                    lang=self.lang,
                    config=config_psm4,
                    output_type=pytesseract.Output.DICT,
                )
                palavras = self._mesclar_palavras(palavras, self._coletar_palavras(dados_psm4))
        except Exception as exc:
            logger.exception("Falha no OCR")
            raise ExtractionError(f"Falha no OCR: {exc}") from exc
        linhas = self._agrupar_linhas(dados)
        logger.debug("OCR produziu %d linhas e %d palavras", len(linhas), len(palavras))
        return linhas, texto_completo, palavras

    @staticmethod
    def _mesclar_palavras(a: list[_Palavra], b: list[_Palavra]) -> list[_Palavra]:
        """Une palavras de duas passadas, descartando duplicatas espaciais."""
        resultado = list(a)
        for palavra in b:
            duplicada = any(
                abs(p.top - palavra.top) <= max(p.height, palavra.height) * 0.5
                and p.left <= palavra.left + palavra.width
                and palavra.left <= p.left + p.width
                for p in resultado
            )
            if not duplicada:
                resultado.append(palavra)
        return resultado

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

    @staticmethod
    def _cnpj_valido(cnpj: str) -> bool:
        digitos = re.sub(r"\D", "", cnpj)
        if len(digitos) != 14 or len(set(digitos)) == 1:
            return False
        pesos1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
        pesos2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]

        def dv(numeros, pesos):
            soma = sum(int(n) * p for n, p in zip(numeros, pesos))
            resto = soma % 11
            return "0" if resto < 2 else str(11 - resto)

        base = digitos[:12]
        return digitos[12] == dv(base, pesos1) and digitos[13] == dv(base + digitos[12], pesos2)

    def _extrair_cnpj(
        self, linhas, texto: str, palavras, imagem_cinza, confiancas: dict
    ) -> str | None:
        match = _CNPJ_RE.search(texto)
        if not match:
            logger.debug("CNPJ não encontrado.")
            return None
        bruto = match.group(0).replace(".", "").replace("/", "").replace("-", "")
        formatado = f"{bruto[:2]}.{bruto[2:5]}.{bruto[5:8]}/{bruto[8:12]}-{bruto[12:]}"
        linha = next((l for l in linhas if _CNPJ_RE.search(l.texto)), None)
        conf = max(0.9, linha.confianca) if linha else 0.9
        if self._cnpj_valido(bruto):
            confiancas["cnpj_prestador"] = round(conf, 2)
        else:
            logger.warning("CNPJ '%s' falhou no dígito verificador; tentando re-OCR.", formatado)
            corrigido = None
            if linha is not None and imagem_cinza is not None:
                banda = self._banda_da_linha(linha, palavras, imagem_cinza)
                if banda is not None:
                    for psm in (6, 7):
                        texto_recorte = self._ocr_recorte(
                            imagem_cinza, banda[0], banda[1], banda[2], banda[3], psm=psm
                        )
                        if not texto_recorte:
                            continue
                        for candidato in _CNPJ_RE.finditer(texto_recorte):
                            bruto_cand = (
                                candidato.group(0)
                                .replace(".", "")
                                .replace("/", "")
                                .replace("-", "")
                            )
                            if self._cnpj_valido(bruto_cand):
                                corrigido = bruto_cand
                                break
                        if corrigido:
                            break
            if corrigido:
                formatado = f"{corrigido[:2]}.{corrigido[2:5]}.{corrigido[5:8]}/{corrigido[8:12]}-{corrigido[12:]}"
                confiancas["cnpj_prestador"] = 0.8
                logger.info("CNPJ corrigido via re-OCR: %s", formatado)
            else:
                confiancas["cnpj_prestador"] = 0.5
                logger.warning("CNPJ mantido com confiança reduzida: %s", formatado)
        logger.debug("CNPJ extraído: %s", formatado)
        return formatado

    def _banda_da_linha(self, linha, palavras, imagem_cinza):
        altura_img, largura_img = imagem_cinza.shape
        palavras_linha = [
            p for p in palavras
            if p.top <= linha.topo + 15 and p.top + p.height >= linha.topo - 15
        ]
        if not palavras_linha:
            return None
        altura = max(p.height for p in palavras_linha) or 25
        left = 0
        right = largura_img
        top = max(0, linha.topo - int(altura * 0.8))
        bottom = linha.topo + int(altura * 1.8)
        return (left, top, right, bottom)

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
        candidatos = []
        for match in _DATA_RE.finditer(" ".join(l.texto for l in linhas)):
            bruto = match.group(0)
            linha = next((l for l in linhas if bruto in l.texto), None)
            if linha is None:
                continue
            superior = linha.texto.upper()
            if "VENCIMENTO" in superior:
                continue
            tem_ancora = "EMISS" in superior
            candidatos.append((not tem_ancora, linha.topo, bruto, linha))
        if not candidatos:
            logger.debug("Data não encontrada.")
            return None
        candidatos.sort(key=lambda c: (c[0], c[1]))
        _, _, bruto, linha = candidatos[0]
        dia, mes, ano = bruto.split("/")
        normalizada = f"{int(dia):02d}/{int(mes):02d}/{ano}"
        conf = linha.confianca
        confiancas["data_emissao"] = round(min(0.95, conf), 2)
        logger.debug("Data extraída: %s -> %s", bruto, normalizada)
        return normalizada

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
        recortes = []
        for palavra in palavras:
            if re.fullmatch(r"N[úuùûüÚUÙÛÜ][º°]?|N[º°]|NUMERO|NÚMERO", palavra.texto, re.IGNORECASE):
                left = max(0, palavra.left - 40)
                top = max(0, palavra.top - int(palavra.height * 0.6))
                right = palavra.left + int(palavra.width * 12)
                bottom = palavra.top + int(palavra.height * 4.0)
                recortes.append((left, top, right, bottom))
        for palavra in palavras:
            if re.search(r"EMISS", palavra.texto, re.IGNORECASE) and palavra.confianca > 0.4:
                altura_img, largura_img = imagem_cinza.shape
                if palavra.left > largura_img * 0.55 and palavra.top < altura_img * 0.25:
                    left = max(0, palavra.left - int(palavra.width * 8))
                    right = palavra.left + int(palavra.width * 5)
                    top = max(0, palavra.top - int(palavra.height * 4.5))
                    bottom = max(top, palavra.top - int(palavra.height * 0.4))
                    recortes.append((left, top, right, bottom))
        for left, top, right, bottom in recortes:
            texto_recorte = self._ocr_recorte(imagem_cinza, left, top, right, bottom)
            if not texto_recorte:
                continue
            match = _NUMERO_RE.search(texto_recorte) or _NUMERO_SO_RE.search(texto_recorte)
            if not match:
                continue
            bruto = match.group(0).strip("| ")
            if len(re.sub(r"\D", "", bruto)) >= 5:
                confiancas["numero_nfse"] = 0.75
                logger.debug("Número extraído (re-OCR posicional): %s", bruto)
                return bruto
        return None

    def _ocr_recorte(self, imagem_cinza, left, top, right, bottom, psm: int = 6) -> str | None:
        altura_img, largura_img = imagem_cinza.shape
        recorte = imagem_cinza[top: min(bottom, altura_img), left: min(right, largura_img)]
        if recorte.size == 0:
            return None
        pilha = Image.fromarray(recorte)
        pilha = pilha.resize((pilha.width * 2, pilha.height * 2), Image.LANCZOS)
        try:
            return pytesseract.image_to_string(
                pilha, lang=self.lang, config=f"--oem 3 --psm {psm}"
            )
        except Exception:
            logger.exception("Falha no re-OCR de recorte")
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
