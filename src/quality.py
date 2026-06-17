"""Input minőség-validáció a /classify endpoint előtt.

A modul fatálisnak nem minősülő figyelmeztetéseket ad vissza:
a request **továbbra is fel lesz dolgozva**, de a logban / metrikában
megjelenik, hogy az input gyanús (pl. túl rövid, valószínűleg nem angol).

A célja: production-ben látható legyen, ha az input-distribution eltér
attól, amire a modell kalibrálva van — ez egyfajta upstream drift-jelző.

Dependencia-mentes — heurisztikák. Ha pontosabb nyelv-detektálás kell,
később bevezethető (pl. langdetect csomag).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


MIN_INPUT_CHARS = 50
NON_ASCII_RATIO_THRESHOLD = 0.10  # 10%+ nem-ASCII karakter → gyanús (nem angol?)
HTML_TAG_PATTERN = re.compile(r"<[a-zA-Z/][^>]*>")
MIN_HTML_TAGS_FOR_FLAG = 3


@dataclass
class QualityIssue:
    """Egy adatminőségi probléma — ne állítson le, csak jelezze."""

    issue: str
    details: dict


def assess_input_quality(text: str) -> list[QualityIssue]:
    """Az input szöveg gyanús jeleinek ellenőrzése.

    A visszaadott lista lehet üres (minden rendben) vagy egy/több issue-t
    tartalmazhat. Az endpoint ezeket loggolja, de a request feldolgozásra
    kerül.
    """
    issues: list[QualityIssue] = []

    # 1) Túl rövid input
    char_count = len(text)
    if char_count < MIN_INPUT_CHARS:
        issues.append(QualityIssue(
            issue="input_too_short",
            details={
                "char_count": char_count,
                "min_recommended": MIN_INPUT_CHARS,
            },
        ))

    # 2) Nem-ASCII arány (heurisztika nem-angol input-ra)
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ratio = non_ascii / max(len(text), 1)
    if ratio > NON_ASCII_RATIO_THRESHOLD:
        issues.append(QualityIssue(
            issue="possibly_non_english",
            details={
                "non_ascii_ratio": round(ratio, 4),
                "threshold": NON_ASCII_RATIO_THRESHOLD,
                "comment": (
                    "A modell csak angolra van kalibrálva. Magas nem-ASCII "
                    "arány gyengébb predikció-minőséget eredményezhet."
                ),
            },
        ))

    # 3) HTML/markup tag-ek — a tisztítást a hívó félnek kell megoldania
    tag_count = len(HTML_TAG_PATTERN.findall(text))
    if tag_count >= MIN_HTML_TAGS_FOR_FLAG:
        issues.append(QualityIssue(
            issue="html_tags_detected",
            details={
                "tag_count": tag_count,
                "comment": (
                    "Az input HTML/markup-ot tartalmaz. Tisztított plain-text "
                    "input pontosabb predikciót eredményez."
                ),
            },
        ))

    return issues
