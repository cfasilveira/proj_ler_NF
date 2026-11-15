from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from ler_nf.contracts import ExtractionResult, QualityCheckResult
from ler_nf.exporter import DataExporter
from ler_nf.extractor import NfseExtractor
from ler_nf.file_guard import FileGuard
from ler_nf.image_preprocess import ImagePreprocessor
from ler_nf.pipeline import NfsePipeline, Status
from ler_nf.quality_checker import QualityChecker
from ler_nf.template_store import TemplateNotFoundError, TemplateStore

PASTA_IMAGENS = Path(__file__).resolve().parent.parent / "imagens_NF"

pytestmark = pytest.mark.skipif(
    not PASTA_IMAGENS.is_dir(), reason="pasta imagens_NF ausente (local)"
)


@pytest.fixture(scope="module")
def imagens():
    return sorted(PASTA_IMAGENS.iterdir())


@pytest.fixture(scope="module")
def nfse_sp():
    caminho = PASTA_IMAGENS / "nfe_4.jpg"
    if not caminho.is_file():
        pytest.skip("nfe_4.jpg ausente")
    return caminho


def test_guard_aceita_todos_os_formatos(imagens):
    guard = FileGuard()
    for imagem in imagens:
        resultado = guard.validate(imagem)
        assert resultado.ok, f"{imagem.name}: {resultado.error_code}"
        assert resultado.mime_type.startswith("image/")


def test_preprocessamento_roda_em_todas(imagens):
    pre = ImagePreprocessor()
    for imagem in imagens:
        processada = pre.processar_desde_arquivo(str(imagem))
        assert processada.grayscale.shape[0] > 0
        assert processada.ocr.shape[0] > 0


def test_quality_avalia_sem_excecao(imagens):
    checker = QualityChecker()
    for imagem in imagens:
        resultado = checker.check(str(imagem))
        assert isinstance(resultado, QualityCheckResult)
        assert 0.0 <= resultado.score <= 1.0


def test_extractor_nfse_sp_campos_principais(nfse_sp):
    pre = ImagePreprocessor()
    processada = pre.processar_desde_arquivo(str(nfse_sp))
    dados = NfseExtractor().extract(processada.ocr, imagem_cinza=processada.grayscale)
    assert re.fullmatch(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", dados.cnpj_prestador or "")
    assert dados.cnpj_prestador.startswith("64.167.64")
    assert "FMA" in (dados.razao_social or "")
    assert dados.data_emissao == "01/06/2006"
    assert dados.numero_nfse == "00000001"
    assert dados.valor_total == 1500.0
    assert dados.iss == 75.0
    assert all(v >= 0.5 for v in dados.confianca.values())


def test_templates_nfse_sp_reconhecida(nfse_sp, tmp_path):
    store = TemplateStore(tmp_path / "db.sqlite")
    pre = ImagePreprocessor()
    processada = pre.processar_desde_arquivo(str(nfse_sp))
    from PIL import Image

    store.cadastrar(Image.fromarray(processada.grayscale), "São Paulo", "São Paulo")
    encontrado, distancia = store.buscar(Image.fromarray(processada.grayscale))
    assert distancia == 0

    outra = next(
        (p for p in PASTA_IMAGENS.iterdir() if p.name != nfse_sp.name), None
    )
    if outra is not None:
        processada_outra = pre.processar_desde_arquivo(str(outra))
        with pytest.raises(TemplateNotFoundError):
            store.buscar(Image.fromarray(processada_outra.grayscale))


def test_pipeline_e2e_nfse_sp(nfse_sp, tmp_path, monkeypatch):
    destino_downloads = tmp_path / "downloads"
    pipeline = NfsePipeline(
        data_dir=tmp_path / "data", output_dir=destino_downloads
    )
    monkeypatch.setattr(
        pipeline.quality,
        "check",
        lambda caminho: QualityCheckResult(ok=True, score=0.9),
    )

    copia = tmp_path / "copia_nota.jpg"
    shutil.copy(nfse_sp, copia)

    primeira = pipeline.processar(copia)
    assert primeira.status == Status.LAYOUT_DESCONHECIDO

    pipeline.cadastrar_layout(copia, "São Paulo", "São Paulo")

    segunda = pipeline.processar(copia)
    assert segunda.status == Status.OK
    assert re.fullmatch(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", segunda.data.cnpj_prestador or "")
    assert segunda.data.valor_total == 1500.0
    assert segunda.data.iss == 75.0

    caminho_export = pipeline.exportar_e_descartar(segunda.data, "csv", copia)
    assert caminho_export.exists()
    assert not copia.exists()
    assert nfse_sp.is_file()
    assert "nfse_" in caminho_export.name
    texto = caminho_export.read_text(encoding="utf-8")
    assert "64.167.64" in texto
    assert "75.0" in texto


def test_exporter_nfse_sp_todos_formatos(nfse_sp, tmp_path):
    dados = ExtractionResult(
        cnpj_prestador="64.167.648/0001-33",
        razao_social="FMA - FACULDADE DE MEDICINA APLICADA LTDA",
        data_emissao="01/06/2006",
        numero_nfse="00000001",
        valor_total=1500.0,
        iss=75.0,
    )
    exporter = DataExporter(output_dir=tmp_path)
    for i, fmt in enumerate(("xlsx", "csv", "txt"), start=1):
        caminho = exporter.export(dados, fmt)
        assert caminho.exists()
        assert f"_{i:03d}" in caminho.name


def _variantes_foto(tmp_path, fonte):
    import cv2
    import numpy as np

    img = cv2.imread(str(fonte))
    h, w = img.shape[:2]
    rng = np.random.default_rng(3)
    variantes = {}

    m = cv2.getRotationMatrix2D((w // 2, h // 2), 3, 1.0)
    variantes["rot3"] = cv2.warpAffine(
        img, m, (w, h), borderMode=cv2.BORDER_REPLICATE
    )

    yy, xx = np.mgrid[0:h, 0:w]
    grad = (0.5 + 0.5 * (xx / w)).astype(np.float32)
    variantes["sombra"] = np.clip(
        img.astype(np.float32) * grad[..., None], 0, 255
    ).astype(np.uint8)

    variantes["ruido"] = np.clip(
        img.astype(np.float32) + rng.normal(0, 10, img.shape), 0, 255
    ).astype(np.uint8)

    variantes["celular1600"] = cv2.resize(
        img, (1600, int(h * 1600 / w)), interpolation=cv2.INTER_AREA
    )

    caminhos = []
    for nome, arr in variantes.items():
        caminho = tmp_path / f"{nome}.png"
        cv2.imwrite(str(caminho), arr)
        caminhos.append(caminho)
    return caminhos


def test_fotos_simuladas_passam_qualidade_e_extracao(nfse_sp, tmp_path):
    checker = QualityChecker()
    pre = ImagePreprocessor()
    ext = NfseExtractor()
    for variante in _variantes_foto(tmp_path, nfse_sp):
        q = checker.check(str(variante))
        assert q.ok, f"{variante.stem} bloqueada: {q.message}"
        processada = pre.processar_desde_arquivo(str(variante))
        dados = ext.extract(processada.ocr, imagem_cinza=processada.grayscale)
        assert dados.data_emissao == "01/06/2006", f"{variante.stem}: data errada"
        assert dados.valor_total == 1500.0, f"{variante.stem}: valor errado"
