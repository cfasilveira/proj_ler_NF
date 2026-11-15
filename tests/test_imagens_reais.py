from __future__ import annotations

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
    assert dados.cnpj_prestador == "64.167.648/0001-33"
    assert "FMA" in (dados.razao_social or "")
    assert dados.data_emissao == "01/06/2006"
    assert dados.numero_nfse == "00000001"
    assert dados.valor_total == 1500.0
    assert dados.iss == 75.0
    assert all(v > 0.5 for v in dados.confianca.values())


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

    primeira = pipeline.processar(nfse_sp)
    assert primeira.status == Status.LAYOUT_DESCONHECIDO

    pipeline.cadastrar_layout(nfse_sp, "São Paulo", "São Paulo")

    segunda = pipeline.processar(nfse_sp)
    assert segunda.status == Status.OK
    assert segunda.data.cnpj_prestador == "64.167.648/0001-33"
    assert segunda.data.valor_total == 1500.0
    assert segunda.data.iss == 75.0

    caminho_export = pipeline.exportar_e_descartar(segunda.data, "csv", nfse_sp)
    assert caminho_export.exists()
    assert "nfse_" in caminho_export.name
    texto = caminho_export.read_text(encoding="utf-8")
    assert "64.167.648/0001-33" in texto
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
