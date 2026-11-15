from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PreprocessError(Exception):
    """Falha no pré-processamento da imagem."""


@dataclass
class ProcessedImage:
    grayscale: np.ndarray
    ocr: np.ndarray


class ImagePreprocessor:
    """Pré-processamento de fotos de NFS-e para OCR e pHash.

    Produz duas variantes: ``grayscale`` (cinzenta redimensionada,
    estável para cálculo de pHash) e ``ocr`` (binarizada, alto
    contraste para o Tesseract).

    Exemplo:
        >>> pre = ImagePreprocessor()
        >>> processada = pre.processar_desde_arquivo("foto.jpg")
        >>> processada.ocr.shape[0] > 0
        True
    """

    def __init__(
        self,
        largura_minima: int = 1400,
        angulo_maximo: float = 15.0,
        aplicar_binarizacao: bool = True,
    ):
        self.largura_minima = largura_minima
        self.angulo_maximo = angulo_maximo
        self.aplicar_binarizacao = aplicar_binarizacao

    def processar_desde_arquivo(self, path: str) -> ProcessedImage:
        imagem = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if imagem is None:
            logger.error("OpenCV não conseguiu ler a imagem: %s", path)
            raise PreprocessError(f"Não foi possível carregar a imagem '{path}'.")
        return self.processar(imagem)

    def processar(self, imagem: np.ndarray) -> ProcessedImage:
        if imagem is None or imagem.size == 0:
            raise PreprocessError("Imagem vazia recebida pelo pré-processamento.")
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        cinza = self._redimensionar(cinza)
        cinza = self._corrigir_inclinacao(cinza)
        ocr = self._binarizar(cinza) if self.aplicar_binarizacao else cinza
        logger.debug(
            "Pré-processamento concluído: grayscale=%sx%s ocr=%sx%s",
            *cinza.shape[::-1],
            *ocr.shape[::-1],
        )
        return ProcessedImage(grayscale=cinza, ocr=ocr)

    def _redimensionar(self, cinza: np.ndarray) -> np.ndarray:
        altura, largura = cinza.shape
        if largura >= self.largura_minima:
            return cinza
        fator = self.largura_minima / largura
        nova = cv2.resize(
            cinza,
            (int(largura * fator), int(altura * fator)),
            interpolation=cv2.INTER_CUBIC,
        )
        logger.debug("Upscale aplicado: %dx%d -> %dx%d", largura, altura, nova.shape[1], nova.shape[0])
        return nova

    def _corrigir_inclinacao(self, cinza: np.ndarray) -> np.ndarray:
        angulo = self._estimar_angulo(cinza)
        if angulo is None or abs(angulo) < 0.5:
            return cinza
        if abs(angulo) > self.angulo_maximo:
            logger.debug("Ângulo %.1f acima do limite, mantendo original.", angulo)
            return cinza
        altura, largura = cinza.shape
        centro = (largura // 2, altura // 2)
        matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
        rotacionada = cv2.warpAffine(
            cinza,
            matriz,
            (largura, altura),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.debug("Correção de inclinação: %.2f graus", angulo)
        return rotacionada

    @staticmethod
    def _estimar_angulo(cinza: np.ndarray) -> float | None:
        suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)
        _, binaria = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(binaria)
        if coords is None:
            return None
        retângulo = cv2.minAreaRect(coords)
        angulo = retângulo[2]
        if angulo > 45:
            angulo -= 90
        return angulo

    @staticmethod
    def _binarizar(cinza: np.ndarray) -> np.ndarray:
        binária = cv2.adaptiveThreshold(
            cinza,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=35,
            C=15,
        )
        binária = cv2.medianBlur(binária, 3)
        return binária
