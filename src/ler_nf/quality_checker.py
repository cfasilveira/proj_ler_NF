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

    Avisos não bloqueiam a foto: artefatos de compressão, fundo
    branco superexposto e bordas cortadas são normais em fotos de
    documentos. Bloqueiam apenas sinais que tornam o texto
    ilegível (blur, ruído, subexposição, resolução).

    Exemplo:
        >>> checker = QualityChecker()
        >>> resultado = checker.check("foto.jpg")
        >>> resultado.ok
        True
    """

    AVISOS_PADRAO = ("compressed", "overexposed", "clipped", "bad_exposure")

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        bloquear: "tuple[str, ...] | None" = None,
    ):
        self._thresholds = thresholds
        self._bloquear = bloquear if bloquear is not None else self.AVISOS_PADRAO

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
        ok, mensagem = self._avaliar(resultado)
        logger.info(
            "Qualidade de %s: ok=%s score=%.2f reason=%s",
            path,
            ok,
            resultado.score,
            resultado.reason,
        )
        return QualityCheckResult(ok=ok, score=float(resultado.score), reason=resultado.reason, message=mensagem)

    def _avaliar(self, resultado) -> "tuple[bool, str | None]":
        issues = [i.lower() for i in (resultado.issues or [])]
        if resultado.reason:
            issues.append(resultado.reason.lower())
        bloqueios = [
            issue for issue in issues
            if not any(aviso in issue for aviso in self._bloquear)
        ]
        if resultado.ok:
            return True, None
        if not bloqueios:
            logger.debug("Somente avisos de qualidade (compressão): aceito com score %.2f", resultado.score)
            return True, None
        return False, traduzir_motivo(bloqueios[0], bloqueios[1:])
