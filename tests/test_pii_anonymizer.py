"""Unit tests untuk modul PII Anonymizer (Blind Screening & Mitigasi Bias)."""

from src.pii_anonymizer import anonymize_cv_text


def test_redacts_email_phone_dob_and_name():
    sample_text = (
        "Dani\n"
        "ARCHITECTURAL DRAFTER\n"
        "South Jakarta-Indonesia | +6285523692189 | Dani19@gmail.com\n"
        "PROFILE\n"
        "I was born in Kuningan, 19 February 2003. I can use AutoCAD and SketchUp.\n"
    )
    result = anonymize_cv_text(sample_text, candidate_id="CAND-005", redact_institutions=False)

    assert result.detected_name == "Dani"
    assert "Dani19@gmail.com" not in result.anonymized_text
    assert "+6285523692189" not in result.anonymized_text
    assert "19 February 2003" not in result.anonymized_text
    assert "[EMAIL_REDACTED]" in result.anonymized_text
    assert "[PHONE_REDACTED]" in result.anonymized_text
    assert "[DOB_REDACTED]" in result.anonymized_text
    assert "AutoCAD" in result.anonymized_text
    assert result.total_redactions >= 4


def test_redacts_university_names_when_blind_mode_enabled():
    sample_text = (
        "Permana Abadi\n"
        "Bandung, Indonesia 40141 | 0821272986 | permanaabadi@gmail.com\n"
        "EDUCATION\n"
        "Bandung Institute of Technology\n"
        "Bachelors of Architecture (B.Arch.)\n"
    )
    result = anonymize_cv_text(sample_text, candidate_id="CAND-002", redact_institutions=True)

    assert result.detected_name == "Permana Abadi"
    assert "Bandung Institute of Technology" not in result.anonymized_text
    assert "[INSTITUTION_REDACTED]" in result.anonymized_text
    assert "Bachelors of Architecture" in result.anonymized_text


def test_preserves_year_ranges_and_section_headings_during_redaction():
    sample_text = (
        "Mohamad\n"
        "+62-822-4040-1234 | mohamad@gmail.com\n"
        "PT Grand Utama Cemerlang, Drafter 2021 - 2023\n"
        "Education Level\n"
        "Tarumanagara University - Indonesia Jul 2018 - Jul 2022\n"
        "Bachelor of Architecture\n"
        "University of Engineering Tantri Abeng\n"
        "2010-2013\n"
    )
    result = anonymize_cv_text(sample_text, candidate_id="CAND-003", redact_institutions=True)
    assert "2021 - 2023" in result.anonymized_text
    assert "2010-2013" in result.anonymized_text
    assert "Education Level" in result.anonymized_text
    assert "Bachelor of Architecture" in result.anonymized_text
    assert "+62-822-4040-1234" not in result.anonymized_text
