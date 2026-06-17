"""Input minőség-validáció tesztjei."""

from __future__ import annotations

from src.quality import assess_input_quality


def test_normal_english_article_has_no_issues() -> None:
    text = (
        "Apple unveiled its new M5 chip today, claiming a 40 percent "
        "performance boost over the previous generation while consuming "
        "30 percent less power."
    )
    issues = assess_input_quality(text)
    assert issues == []


def test_short_input_flagged() -> None:
    issues = assess_input_quality("Apple chip news.")
    assert any(i.issue == "input_too_short" for i in issues)


def test_high_non_ascii_input_flagged_as_possibly_non_english() -> None:
    # Magyar szöveg: ékezetek miatt magas nem-ASCII arány
    text = (
        "Az új okostelefon érzékenyen reagál a felhasználói igényekre, "
        "minden tekintetben kiemelkedő teljesítményt nyújt és igazán "
        "innovatív megoldásokat kínál a mindennapokban."
    )
    issues = assess_input_quality(text)
    assert any(i.issue == "possibly_non_english" for i in issues)


def test_html_tags_flagged_when_present() -> None:
    text = (
        "<p>Apple unveiled its new M5 chip today.</p><br/><div>"
        "Performance gains are significant.</div>"
    )
    issues = assess_input_quality(text)
    assert any(i.issue == "html_tags_detected" for i in issues)


def test_quality_issues_have_helpful_details() -> None:
    issues = assess_input_quality("short")
    issue = next(i for i in issues if i.issue == "input_too_short")
    assert "char_count" in issue.details
    assert "min_recommended" in issue.details


def test_long_clean_english_article_with_one_special_char_passes() -> None:
    """A 10%-os küszöb alatti speciális karakterek elfogadottak."""
    text = (
        "The European Central Bank raised interest rates by 25 basis points "
        "today, citing persistent inflation. The decision was widely "
        "anticipated by markets, though some analysts argued for a larger "
        "increase to combat ongoing price pressures."
    )
    issues = assess_input_quality(text)
    assert all(i.issue != "possibly_non_english" for i in issues)
