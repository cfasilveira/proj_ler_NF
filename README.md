# proj_ler_NF — Leitor de NFS-e

Aplicativo desktop local (Linux) para contadores e administradores extraírem
dados de **Notas Fiscais de Serviço Eletrônicas (NFS-e)** a partir de fotos
enviadas manualmente. 100% offline, processamento em CPU, descarte automático
da imagem original após a exportação.

## Pipeline

```
Upload → Validação de segurança (M2) → Qualidade visual (M1)
       → Pré-processamento (deskew/binarização) → Identificação do layout via pHash (M3)
       → Extração OCR (M4) → Revisão manual → Exportação xlsx/csv/txt (M5) + descarte
```

Escopo do MVP: um layout/município fixo, extensível por cadastro de templates.
Autenticação (M6) adiada para a fase multiusuário.

## Stack

| Papel | Biblioteca |
|---|---|
| Qualidade de imagem | `imageguard` |
| Segurança de arquivo | `safeuploads` |
| Hash perceptual | `ImageHash` |
| OCR | `pytesseract` (Tesseract + `por`) |
| Pré-processamento | `opencv-python-headless` |
| Exportação | `tablib[xlsx]` |
| Interface | `CustomTkinter` |

Requisitos de sistema: Python ≥ 3.13 (desenvolvido no 3.14 com `uv`),
binário `tesseract` com idioma `por`:

```bash
sudo apt install tesseract-ocr tesseract-ocr-por
```

## Como executar

```bash
uv sync
uv run python -m ler_nf
```

## Arquitetura modular

Cada módulo é independente, reutilizável fora do app e define:

- `logger = logging.getLogger(__name__)` com `NullHandler` — o app principal
  centraliza os handlers (`log_config.setup_logging`, `RotatingFileHandler`);
- exceção própria herdando de `ler_nf.errors.LeNfError`;
- contrato de entrada/saída via dataclasses (`ler_nf.contracts`).

| Módulo | Responsabilidade |
|---|---|
| `file_guard` (M2) | magic bytes, MIME, extensão, tamanho, ZIP bombs, polyglots |
| `quality_checker` (M1) | blur, ruído, exposição, resolução → `{ok, reason, score}` + mensagem em PT-BR |
| `image_preprocess` | upscale, correção de inclinação, binarização adaptativa |
| `template_store` (M3) | SQLite + pHash + distância de Hamming; blob WebP comprimido do template (módulo de referência) |
| `extractor` (M4) | Tesseract + âncoras ("CNPJ", "VALOR TOTAL", ...) e regex tolerantes, com índice de confiança por campo |
| `exporter` (M5) | `~/Downloads/nfse_YYYY-MM-DD_NNN.{xlsx,csv,txt}` |
| `pipeline` | orquestração das etapas e descarte da imagem original |
| `gui` (M7) | janela única, extração em thread, grade editável, cadastro de layout |

### Uso como biblioteca

```python
from ler_nf.template_store import TemplateStore
from ler_nf.extractor import NfseExtractor

store = TemplateStore("data/templates.sqlite", limiar_hamming=12)
store.cadastrar("nota_sp.png", "Layout SP", "São Paulo")

extractor = NfseExtractor()
dados = extractor.extract("nota.png")
print(dados.to_dict())  # inclui 'confianca' por campo
```

Ou pelo orquestrador:

```python
from ler_nf.pipeline import NfsePipeline

pipeline = NfsePipeline(data_dir="data", limiar_hamming=12)
resultado = pipeline.processar("foto.jpg")   # progresso opcional: progresso=callback
if resultado.status.name == "OK":
    pipeline.exportar_e_descartar(resultado.data, "xlsx", "foto.jpg")
```

## Logs

Os erros de cada módulo são registrados em `data/logs/ler_nf.log`
(`RotatingFileHandler`, 1 MB × 5 backups). Configure no entrypoint:

```python
from ler_nf.log_config import setup_logging
setup_logging("data/logs", level=logging.INFO, console=True)
```

## Testes

```bash
uv run pytest -v
```

## Decisões em aberto / roadmap

- **Motor de extração**: Tesseract no MVP; `TinyDoc-VLM` como evolução
  (contrato de saída já compatível — troca isolada do módulo M4).
- **Calibração**: limiar de Hamming (padrão 12), limiares do `imageguard`
  (parâmetro `quality_thresholds` do pipeline) e âncoras do extrator devem
  ser calibrados com amostras reais de NFS-e do município-alvo.
- **Drag-and-drop** na GUI: depende de `tkinterdnd2` (fora do MVP).
- **Autenticação (M6)**: adiada para a fase multiusuário.
