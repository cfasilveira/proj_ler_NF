from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    filename: str | None = None
    mime_type: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class QualityCheckResult:
    ok: bool
    score: float
    reason: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class TemplateRecord:
    id: int
    nome: str
    municipio: str
    imagem_path: str
    phash: str
    campos_mapeados: dict
    data_cadastro: str


@dataclass
class ExtractionResult:
    cnpj_prestador: str | None = None
    razao_social: str | None = None
    data_emissao: str | None = None
    numero_nfse: str | None = None
    valor_total: float | None = None
    iss: float | None = None
    confianca: dict[str, float] = field(default_factory=dict)

    CAMPOS = (
        "cnpj_prestador",
        "razao_social",
        "data_emissao",
        "numero_nfse",
        "valor_total",
        "iss",
    )

    def to_dict(self) -> dict:
        saida = {campo: getattr(self, campo) for campo in self.CAMPOS}
        saida["confianca"] = dict(self.confianca)
        return saida

    @classmethod
    def from_dict(cls, dados: dict) -> ExtractionResult:
        campos = {c: dados.get(c) for c in cls.CAMPOS}
        resultado = cls(**campos)
        resultado.confianca = dict(dados.get("confianca", {}))
        return resultado
