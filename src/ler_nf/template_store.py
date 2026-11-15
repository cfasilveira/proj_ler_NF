from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import imagehash
from PIL import Image

from .contracts import TemplateRecord
from .errors import LeNfError

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT NOT NULL,
    municipio TEXT NOT NULL,
    imagem_path TEXT,
    imagem_blob BLOB,
    phash TEXT NOT NULL UNIQUE,
    campos_mapeados TEXT NOT NULL,
    data_cadastro TEXT NOT NULL
);
"""

_SQL_INSERIR = (
    "INSERT INTO templates "
    "(nome, municipio, imagem_path, imagem_blob, phash, campos_mapeados, data_cadastro) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)
_SQL_TODOS = (
    "SELECT id, nome, municipio, imagem_path, phash, campos_mapeados, data_cadastro "
    "FROM templates"
)


class TemplateStoreError(LeNfError):
    """Falha genérica do armazém de templates."""


class TemplateNotFoundError(TemplateStoreError):
    """Nenhum layout reconhecido dentro do limiar configurado."""

    def __init__(self, mensagem: str, distance: int | None = None, threshold: int | None = None):
        super().__init__(mensagem)
        self.distance = distance
        self.threshold = threshold


class TemplateDuplicadoError(TemplateStoreError):
    """Já existe um template com o mesmo pHash."""


class TemplateStore:
    """Armazém local de layouts de NFS-e indexados por pHash perceptual.

    Módulo de referência do projeto: define o padrão de logging,
    exceções estruturadas e contratos usado pelos demais módulos.

    Exemplo:
        >>> store = TemplateStore("data/templates.sqlite")
        >>> rec = store.cadastrar("nota_sp.png", "Layout SP", "São Paulo")
        >>> achado, distancia = store.buscar("nota_sp.png")
        >>> distancia
        0
    """

    def __init__(self, db_path: str | os.PathLike = "data/templates.sqlite", limiar_hamming: int = 12):
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self.limiar_hamming = int(limiar_hamming)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.executescript(_SCHEMA)
        logger.info(
            "TemplateStore iniciado (db=%s, limiar_hamming=%d)",
            self._db_path,
            self.limiar_hamming,
        )

    @staticmethod
    def phash_de(imagem: "str | os.PathLike | Image.Image") -> imagehash.ImageHash:
        """Calcula o pHash a partir de um caminho ou PIL.Image."""
        if isinstance(imagem, imagehash.ImageHash):
            return imagem
        if isinstance(imagem, (str, os.PathLike)):
            imagem = Image.open(imagem)
        if not isinstance(imagem, Image.Image):
            raise TypeError(
                f"imagem deve ser caminho, PIL.Image ou ImageHash; recebi {type(imagem).__name__}"
            )
        return imagehash.phash(imagem.convert("RGB"))

    def cadastrar(
        self,
        imagem: "str | os.PathLike | Image.Image",
        nome: str,
        municipio: str,
        campos_mapeados: dict | None = None,
    ) -> TemplateRecord:
        """Cadastra um novo layout, guardando cópia WebP comprimida no banco."""
        p_hash = self.phash_de(imagem)
        caminho = str(imagem) if isinstance(imagem, (str, os.PathLike)) else None
        blob = self._blob_webp(imagem)
        agora = datetime.now().isoformat()
        try:
            with self._conn:
                cur = self._conn.execute(
                    _SQL_INSERIR,
                    (
                        nome,
                        municipio,
                        caminho,
                        blob,
                        str(p_hash),
                        json.dumps(campos_mapeados or {}, ensure_ascii=False),
                        agora,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            logger.warning("Template duplicado: phash=%s", p_hash)
            raise TemplateDuplicadoError(f"Já existe um template com o mesmo pHash ({p_hash}).") from exc
        registro = TemplateRecord(
            id=cur.lastrowid,
            nome=nome,
            municipio=municipio,
            imagem_path=caminho or "",
            phash=str(p_hash),
            campos_mapeados=dict(campos_mapeados or {}),
            data_cadastro=agora,
        )
        logger.info("Template cadastrado: id=%d nome=%s municipio=%s phash=%s", registro.id, nome, municipio, p_hash)
        return registro

    def buscar(self, imagem: "str | os.PathLike | Image.Image") -> tuple[TemplateRecord, int]:
        """Busca o layout com menor distância de Hamming abaixo do limiar.

        Retorna ``(template, distancia)``. Levanta
        ``TemplateNotFoundError`` com ``distance``/``threshold``
        preenchidos quando nada é reconhecido.
        """
        p_hash = self.phash_de(imagem)
        melhor: TemplateRecord | None = None
        melhor_dist: int | None = None
        for row in self._conn.execute(_SQL_TODOS):
            registro = self._linha_para_registro(row)
            distancia = p_hash - imagehash.hex_to_hash(registro.phash)
            if melhor_dist is None or distancia < melhor_dist:
                melhor, melhor_dist = registro, distancia
        if melhor is None:
            logger.info("Busca sem candidatos: nenhum template cadastrado.")
            raise TemplateNotFoundError("Nenhum template cadastrado.")
        if melhor_dist > self.limiar_hamming:
            logger.info(
                "Layout desconhecido: melhor candidato '%s' a distância %d (limiar %d)",
                melhor.nome,
                melhor_dist,
                self.limiar_hamming,
            )
            raise TemplateNotFoundError(
                f"Layout não reconhecido. Melhor candidato '{melhor.nome}' "
                f"(distância {melhor_dist}, limiar {self.limiar_hamming}).",
                distance=melhor_dist,
                threshold=self.limiar_hamming,
            )
        logger.info(
            "Layout reconhecido: '%s' (id=%d) a distância %d",
            melhor.nome,
            melhor.id,
            melhor_dist,
        )
        return melhor, melhor_dist

    def listar(self) -> list[TemplateRecord]:
        """Retorna todos os templates cadastrados."""
        return [self._linha_para_registro(row) for row in self._conn.execute(_SQL_TODOS)]

    def close(self) -> None:
        self._conn.close()

    def _linha_para_registro(self, row: tuple) -> TemplateRecord:
        return TemplateRecord(
            id=row[0],
            nome=row[1],
            municipio=row[2],
            imagem_path=row[3] or "",
            phash=row[4],
            campos_mapeados=json.loads(row[5]),
            data_cadastro=row[6],
        )

    @staticmethod
    def _blob_webp(imagem) -> bytes:
        if isinstance(imagem, (str, os.PathLike)):
            imagem = Image.open(imagem)
        if not isinstance(imagem, Image.Image):
            raise TypeError("imagem deve ser caminho ou PIL.Image")
        from io import BytesIO

        buffer = BytesIO()
        imagem.convert("RGB").save(buffer, format="WEBP", quality=60, method=6)
        return buffer.getvalue()
