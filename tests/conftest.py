from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def imagem_sintetica_nfse(caminho, com_inclinacao: bool = False):
    """Gera uma NFS-e sintética com os campos principais."""
    img = Image.new("RGB", (1400, 900), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=30)
    linhas = [
        "NOTA FISCAL DE SERVICOS ELETRONICA",
        "RAZAO SOCIAL: EMPRESA EXEMPLO LTDA",
        "CNPJ: 12.345.678/0001-90",
        "DATA EMISSAO: 15/11/2026",
        "NUMERO: 2026/000123",
        "VALOR TOTAL: R$ 1.500,00",
        "ISS: R$ 75,00",
    ]
    y = 60
    for linha in linhas:
        d.text((80, y), linha, fill="black", font=font)
        y += 60
    if com_inclinacao:
        img = img.rotate(3, fillcolor="white")
    img.save(caminho)
    return caminho


def imagem_ruido(caminho, seed: int = 42):
    """Gera uma imagem de ruído colorido (para templates/pHash)."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    Image.fromarray(arr).save(caminho)
    return caminho
