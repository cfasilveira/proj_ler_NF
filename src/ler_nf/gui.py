from __future__ import annotations

import logging
import threading
import tkinter.filedialog
from pathlib import Path

import customtkinter as ctk

from .contracts import ExtractionResult
from .errors import LeNfError
from .exporter import ExportError
from .log_config import setup_logging
from .pipeline import NfsePipeline, PipelineResult, Status
from .template_store import TemplateDuplicadoError

logger = logging.getLogger(__name__)

_MENSAGENS_STATUS = {
    Status.OK: "Dados extraídos. Revise os campos e exporte.",
    Status.ARQUIVO_REJEITADO: "Arquivo não aceito.",
    Status.QUALIDADE_INSUFICIENTE: "Qualidade da foto insuficiente.",
    Status.PRE_PROCESSAMENTO_FALHOU: "Não foi possível processar a imagem.",
    Status.LAYOUT_DESCONHECIDO: "Layout não reconhecido. Você pode cadastrá-lo.",
    Status.EXTRACAO_FALHOU: "Falha ao extrair os dados.",
}

_CAMPOS_UI = (
    ("cnpj_prestador", "CNPJ"),
    ("razao_social", "Razão Social"),
    ("data_emissao", "Data de Emissão"),
    ("numero_nfse", "Número da NFS-e"),
    ("valor_total", "Valor Total (R$)"),
    ("iss", "ISS (R$)"),
)

_ROTULO_FORMATOS = {"xlsx": "Excel (.xlsx)", "csv": "CSV (.csv)", "txt": "Texto (.txt)"}


class LeNfApp(ctk.CTk):
    """Janela única em fluxo vertical para o MVP do leitor de NFS-e."""

    def __init__(self, pipeline: NfsePipeline | None = None):
        super().__init__()
        self.pipeline = pipeline or NfsePipeline()
        self.title("Leitor de NFS-e")
        self.geometry("760x640")
        self.minsize(680, 560)
        self._imagem_original: Path | None = None
        self._dados: ExtractionResult | None = None
        self._campos: dict[str, ctk.CTkEntry] = {}
        self._montar_widgets()

    def _montar_widgets(self) -> None:
        ctk.CTkLabel(
            self, text="Leitor de NFS-e", font=ctk.CTkFont(size=22, weight="bold")
        ).pack(pady=(18, 6))

        frame_upload = ctk.CTkFrame(self)
        frame_upload.pack(fill="x", padx=24, pady=6)
        self.btn_upload = ctk.CTkButton(
            frame_upload, text="Selecionar foto da NFS-e", command=self._selecionar_arquivo
        )
        self.btn_upload.pack(side="left", padx=12, pady=12)

        frame_progresso = ctk.CTkFrame(self)
        frame_progresso.pack(fill="x", padx=24, pady=6)
        self.barra = ctk.CTkProgressBar(frame_progresso, mode="determinate")
        self.barra.set(0)
        self.barra.pack(fill="x", padx=12, pady=(12, 2))
        self.lbl_status = ctk.CTkLabel(frame_progresso, text="Envie uma foto para começar.")
        self.lbl_status.pack(fill="x", padx=12, pady=(0, 12))

        frame_campos = ctk.CTkFrame(self, fg_color="transparent")
        frame_campos.pack(fill="both", expand=True, padx=24, pady=6)
        for i, (chave, rotulo) in enumerate(_CAMPOS_UI):
            ctk.CTkLabel(frame_campos, text=rotulo, anchor="e", width=150).grid(
                row=i, column=0, sticky="e", padx=(0, 8), pady=4
            )
            entrada = ctk.CTkEntry(frame_campos, width=380)
            entrada.grid(row=i, column=1, sticky="w", pady=4)
            self._campos[chave] = entrada

        frame_exportar = ctk.CTkFrame(self)
        frame_exportar.pack(fill="x", padx=24, pady=(10, 18))
        for fmt, rotulo in _ROTULO_FORMATOS.items():
            ctk.CTkButton(
                frame_exportar, text=rotulo, width=140, command=lambda f=fmt: self._exportar(f)
            ).pack(side="left", padx=8, pady=12)

    def _selecionar_arquivo(self) -> None:
        caminho = tkinter.filedialog.askopenfilename(
            title="Selecione a foto da NFS-e",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.webp"), ("Todos os arquivos", "*.*")],
        )
        if not caminho:
            return
        self._imagem_original = Path(caminho)
        self._dados = None
        for entrada in self._campos.values():
            entrada.delete(0, "end")
            entrada.configure(state="normal")
        self.barra.set(0.0)
        self.lbl_status.configure(text="Processando...")
        self.btn_upload.configure(state="disabled")
        threading.Thread(target=self._processar, daemon=True).start()

    def _processar(self) -> None:
        try:
            resultado = self.pipeline.processar(
                self._imagem_original, progresso=self._progresso_da_thread
            )
        except LeNfError as exc:
            logger.exception("Falha no pipeline")
            self.after(0, self._finalizar_com_erro, str(exc))
            return
        self.after(0, self._finalizar_processamento, resultado)

    def _progresso_da_thread(self, etapa: str) -> None:
        self.after(0, self._atualizar_etapa, etapa)

    def _atualizar_etapa(self, etapa: str) -> None:
        self.lbl_status.configure(text=f"Processando: {etapa}...")
        self.barra.set(min(1.0, self.barra.get() + 0.2))

    def _finalizar_processamento(self, resultado: PipelineResult) -> None:
        self.btn_upload.configure(state="normal")
        self.barra.set(1.0 if resultado.status == Status.OK else self.barra.get())
        base = _MENSAGENS_STATUS[resultado.status]
        if resultado.status == Status.OK:
            self._dados = resultado.data
            self._preencher_campos(resultado.data)
            self.lbl_status.configure(text=base, text_color="green")
        else:
            mensagem = f"{base}\n{resultado.message or ''}".strip()
            self.lbl_status.configure(text=mensagem, text_color="orange")
            if resultado.status == Status.LAYOUT_DESCONHECIDO:
                self.after(300, self._prompt_cadastrar_layout)

    def _finalizar_com_erro(self, mensagem: str) -> None:
        self.btn_upload.configure(state="normal")
        self.lbl_status.configure(
            text=f"Erro: {mensagem}", text_color="red"
        )

    def _preencher_campos(self, dados: ExtractionResult) -> None:
        for chave, entrada in self._campos.items():
            valor = getattr(dados, chave)
            entrada.delete(0, "end")
            if valor is not None:
                entrada.insert(0, str(valor))

    def _ler_campos(self) -> dict:
        valores: dict = {}
        for chave, entrada in self._campos.items():
            bruto = entrada.get().strip()
            if chave in ("valor_total", "iss") and bruto:
                bruto = bruto.replace(".", "").replace(",", ".")
                try:
                    valores[chave] = float(bruto)
                except ValueError:
                    valores[chave] = None
            else:
                valores[chave] = bruto or None
        confianca = dict(self._dados.confianca) if self._dados else {}
        valores["confianca"] = confianca
        return valores

    def _exportar(self, fmt: str) -> None:
        if self._imagem_original is None:
            self.lbl_status.configure(
                text="Nenhuma foto processada ainda.", text_color="orange"
            )
            return
        valores = self._ler_campos()
        dados = ExtractionResult.from_dict(valores)
        try:
            destino = self.pipeline.exportar_e_descartar(dados, fmt, self._imagem_original)
        except ExportError as exc:
            self.lbl_status.configure(text=str(exc), text_color="red")
            return
        self._imagem_original = None
        self.lbl_status.configure(
            text=f"Exportado com sucesso: {destino.name}", text_color="green"
        )
        logger.info("Exportação concluída pela GUI: %s", destino)

    def _prompt_cadastrar_layout(self) -> None:
        dialogo = ctk.CTkInputDialog(
            text="Layout não reconhecido.\nDigite o nome do layout (município/órgão):",
            title="Cadastrar novo layout",
        )
        nome = dialogo.get_input()
        if not nome or not nome.strip() or self._imagem_original is None:
            return
        dialogo_municipio = ctk.CTkInputDialog(
            text="Digite o município (opcional):", title="Cadastrar novo layout"
        )
        municipio = (dialogo_municipio.get_input() or "").strip() or "—"
        try:
            registro = self.pipeline.cadastrar_layout(
                self._imagem_original, nome.strip(), municipio
            )
        except LeNfError as exc:
            logger.warning("Falha ao cadastrar layout: %s", exc)
            self.lbl_status.configure(
                text=f"Não foi possível cadastrar: {exc}", text_color="red"
            )
            return
        self.lbl_status.configure(
            text=f"Layout '{registro.nome}' cadastrado. Envie a foto novamente.",
            text_color="green",
        )


def run_app(pipeline: NfsePipeline | None = None) -> None:
    setup_logging(console=False)
    LeNfApp(pipeline).mainloop()
