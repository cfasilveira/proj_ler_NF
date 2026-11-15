from __future__ import annotations

import logging

import imageguard

from .contracts import QualityCheckResult
from .errors import LeNfError

logger = logging.getLogger(__name__)

_MENSAGENS_PT_BR: list[tuple[str, str]] = [
    ("blur", "A foto está tremida ou fora de foco."),
    ("pixelat", "A imagem está pixelada ou com resolução baixa."),
    ("low_resolution", "A foto tem resolução muito baixa."),
    ("underexposed", "A imagem está muito escura."),
    ("overexposed", "A imagem está muito clara."),
    ("bad_exposure", "A iluminação da foto está inadequada."),
    ("compress", "A imagem está muito comprimida ou com artefatos."),
    ("noise", "A imagem está com muito ruído."),
    ("low_quality", "A qualidade geral da foto está abaixo do aceitável."),
    ("clipped", "A imagem parece estar cortada ou incompleta."),
    ("analysis_failed", "Não foi possível analisar a imagem."),
]


def traduzir_motivo(reason: str | None, issues: list[str] | None = None) -> str:
    """Traduz o diagnóstico técnico do imageguard para linguagem natural."""
    texto = " ".join(filter(None, [reason, *(issues or [])])).lower()
    mensagens = [msg for chave, msg in _MENSAGENS_PT_BR if chave in texto]
    if mensagens:
        return " ".join(dict.fromkeys(mensagens))
    return "A foto não atende aos critérios de qualidade." + (
        f" ({reason})" if reason else ""
    )


class QualityCheckError(LeNfError):
    """Falha inesperada na análise de qualidade da imagem."""


class QualityChecker:
    """Validação de qualidade visual da foto (M1).

    Envolve ``imageguard.validate`` e devolve um resultado
    estruturado com mensagem em linguagem natural para o usuário.

    Exemplo:
        >>> checker = QualityChecker()
        >>> resultado = checker.check("foto.jpg")
        >>> resultado.ok
        True
    """

    def __init__(self, thresholds: dict[str, float] | None = None):
        self._thresholds = thresholds

    def check(self, path: "str") -> QualityCheckResult:
        try:
            resultado = imageguard.validate(str(path), thresholds=self._thresholds)
        except FileNotFoundError:
            logger.error("Imagem não encontrada: %s", path)
            return QualityCheckResult(ok=False, score=0.0, reason="not_found", message="Arquivo não encontrado.")
        except ValueError as exc:
            logger.warning("Imagem ilegível: %s (%s)", path, exc)
            return QualityCheckResult(
                ok=False,
                score=0.0,
                reason="undecodable",
                message="Não foi possível abrir esta imagem.",
            )
        except Exception as exc:
            logger.exception("Falha inesperada na análise de %s", path)
            raise QualityCheckError(f"Falha inesperada na análise de qualidade: {exc}") from exc
        mensagem = None if resultado.ok else traduzir_motivo(resultado.reason, resultado.issues)
        logger.info(
            "Qualidade de %s: ok=%s score=%.2f reason=%s",
            path,
            resultado.ok,
            resultado.score,
            resultado.reason,
        )
        return QualityCheckResult(
            ok=resultado.ok,
            score=float(resultado.score),
            reason=resultado.reason,
            message=mensagem,
        )
