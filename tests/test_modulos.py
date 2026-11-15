from __future__ import annotations

from pathlib import Path

from ler_nf.file_guard import FileGuard
from ler_nf.template_store import TemplateStore
from ler_nf.quality_checker import QualityChecker, traduzir_motivo
from ler_nf.image_preprocess import ImagePreprocessor
from ler_nf.extractor import NfseExtractor
from ler_nf.exporter import DataExporter, ExportError
from ler_nf.contracts import ExtractionResult
import pytest
import numpy as np
from PIL import Image

from conftest import imagem_sintetica_nfse, imagem_ruido


@pytest.fixture(scope="module")
def guard():
    return FileGuard()


def test_file_guard_aceita_png_valido(guard, tmp_path):
    caminho = str(imagem_ruido(tmp_path / "valida.png"))
    resultado = guard.validate(caminho)
    assert resultado.ok is True
    assert resultado.mime_type.startswith("image/")


def test_file_guard_rejeita_conteudo_falsificado(guard, tmp_path):
    caminho = tmp_path / "falso.png"
    caminho.write_bytes(b"isto nao e uma imagem de verdade" * 100)
    resultado = guard.validate(str(caminho))
    assert resultado.ok is False
    assert resultado.error_code in ("conteudo_corrompido", "tipo_nao_permitido", "arquivo_rejeitado")


def test_file_guard_rejeita_arquivo_inexistente(guard, tmp_path):
    resultado = guard.validate(str(tmp_path / "inexistente.png"))
    assert resultado.ok is False
    assert resultado.error_code == "arquivo_inexistente"


def test_quality_checker_traducoes():
    assert "tremida" in traduzir_motivo("blurry").lower() or "tremida" in traduzir_motivo("blur")
    assert "escura" in traduzir_motivo("underexposed").lower()
    assert "clara" in traduzir_motivo("overexposed").lower()
    assert traduzir_motivo(None)  # mensagem padrão


def test_quality_checker_avalia_imagem(tmp_path):
    checker = QualityChecker()
    caminho = str(imagem_ruido(tmp_path / "ruido.png"))
    resultado = checker.check(caminho)
    assert isinstance(resultado.score, float)
    assert 0.0 <= resultado.score <= 1.0


def test_quality_checker_imagem_inexistente():
    resultado = QualityChecker().check("/nao/existe/foto.png")
    assert resultado.ok is False


def test_template_store_cadastrar_e_buscar(tmp_path):
    store = TemplateStore(tmp_path / "db.sqlite")
    imagem_ruido(tmp_path / "layout.png", seed=7)
    registro = store.cadastrar(str(tmp_path / "layout.png"), "Layout A", "Cidade A")
    assert registro.id == 1

    encontrado, distancia = store.buscar(str(tmp_path / "layout.png"))
    assert encontrado.nome == "Layout A"
    assert distancia == 0

    imagem_ruido(tmp_path / "outro.png", seed=99)
    with pytest.raises(Exception) as info:
        store.buscar(str(tmp_path / "outro.png"))
    assert getattr(info.value, "threshold", None) == store.limiar_hamming


def test_template_store_duplicado(tmp_path):
    store = TemplateStore(tmp_path / "db.sqlite")
    imagem_ruido(tmp_path / "a.png", seed=5)
    store.cadastrar(str(tmp_path / "a.png"), "A", "M")
    with pytest.raises(Exception):
        store.cadastrar(str(tmp_path / "a.png"), "B", "M")


def test_image_preprocess_produz_variantes(tmp_path):
    pre = ImagePreprocessor()
    processada = pre.processar_desde_arquivo(str(imagem_sintetica_nfse(tmp_path / "nf.png")))
    assert processada.grayscale.ndim == 2
    assert processada.ocr.ndim == 2
    assert processada.ocr.min() >= 0 and processada.ocr.max() <= 255


def test_image_preprocess_erro_arquivo_invalido(tmp_path):
    pre = ImagePreprocessor()
    caminho = tmp_path / "lixo.png"
    caminho.write_bytes(b"nao-imagem" * 50)
    with pytest.raises(Exception):
        pre.processar_desde_arquivo(str(caminho))


@pytest.mark.ocr
def test_extractor_campo_por_campo(tmp_path):
    extractor = NfseExtractor()
    caminho = str(imagem_sintetica_nfse(tmp_path / "nf.png"))
    dados = extractor.extract(caminho)
    assert dados.cnpj_prestador == "12.345.678/0001-90"
    assert "EMPRESA" in (dados.razao_social or "")
    assert dados.data_emissao == "15/11/2026"
    assert dados.numero_nfse == "2026/000123"
    assert dados.valor_total == 1500.00
    assert dados.iss == 75.00
    assert all(campo in dados.confianca for campo in ExtractionResult.CAMPOS)


def test_extractor_parse_money():
    parse = NfseExtractor.parse_money
    assert parse("VALOR TOTAL: R$ 1.500,00") == 1500.0
    assert parse("R$ 75,00") == 75.0
    assert parse("1.234.567,89") == 1234567.89
    assert parse("sem valor") is None


def test_extractor_cnpj_valido():
    validar = NfseExtractor._cnpj_valido
    assert validar("11.222.333/0001-81")
    assert not validar("11.222.333/0001-82")
    assert not validar("11111111111111")
    assert not validar("1234")


def test_exporter_formatos(tmp_path):
    exporter = DataExporter(output_dir=tmp_path)
    dados = ExtractionResult(
        cnpj_prestador="12.345.678/0001-90",
        razao_social="Empresa Exemplo Ltda",
        data_emissao="2026-11-15",
        numero_nfse="2026/000123",
        valor_total=1500.0,
        iss=75.0,
    )
    caminho_xlsx = exporter.export(dados, "xlsx")
    caminho_csv = exporter.export(dados, "csv")
    caminho_txt = exporter.export(dados, "txt")
    assert caminho_xlsx.exists() and caminho_xlsx.suffix == ".xlsx"
    assert caminho_csv.exists() and "002" in caminho_csv.name and "003" in caminho_txt.name
    conteudo = caminho_txt.read_text(encoding="utf-8")
    assert "CNPJ: 12.345.678/0001-90" in conteudo
    assert "Valor Total: 1500.0" in conteudo


def test_exporter_formato_invalido(tmp_path):
    with pytest.raises(ExportError):
        DataExporter(output_dir=tmp_path).export(ExtractionResult(), "pdf")


def test_contracts_round_trip():
    dados = ExtractionResult(cnpj_prestador="12.345.678/0001-90", valor_total=1500.0)
    reconstruido = ExtractionResult.from_dict(dados.to_dict())
    assert reconstruido.cnpj_prestador == dados.cnpj_prestador
    assert reconstruido.valor_total == dados.valor_total
