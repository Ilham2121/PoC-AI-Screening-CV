"""Regression & edge-case tests: generalisasi format CV, confidence gate HITL, bias rubrik, dan resilience."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from src.pii_anonymizer import anonymize_cv_text
from src.pipeline import run_screening_from_payloads
from src.timeline import parse_month_year

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "Soal Test BP AI Specialist 3"


def _vacancy_bytes() -> bytes:
    return (SAMPLE_DIR / "Vacancy Detail.pdf").read_bytes()


def test_generalizes_to_diverse_english_and_indonesian_cv_formats():
    """CV di luar 5 sampel dengan heading 'Work History', 'Pengalaman Kerja', MM/YYYY, S.Ars., dan Sarjana Teknik."""
    cv_rina = """Rina Kartika
rina.k@mail.com | 0812 3456 7890 | Sudirman 45, Jakarta Selatan
Professional Summary
Architect with 4 years of experience in design and build projects.
Work History
Senior Architect - PT Bangun Jaya
01/2020 - 12/2023
- Led conceptual design, working drawings and site supervision for 12 residential projects.
- Coordinated with suppliers on interior materials, lighting and furniture specs; ensured building code compliance.
Education
S.Ars. Arsitektur, Universitas Indonesia, 2015 - 2019
Software
AutoCAD, SketchUp, Revit, Lumion
"""
    cv_andi = """Andi Wijaya
andi@mail.com | 0813-9988-7766
Ringkasan
Arsitek dengan pengalaman 3 tahun di proyek desain dan bangun.
Pengalaman Kerja
Arsitek Junior, CV Karya Desain
Maret 2021 - Sekarang
- Membuat gambar kerja, konsep desain, dan mengawasi pelaksanaan proyek renovasi rumah tinggal.
- Berkoordinasi dengan pemasok material interior, pencahayaan, dan memastikan kepatuhan kode bangunan (PBG).
Pendidikan
Sarjana Teknik Arsitektur, Institut Teknologi Sepuluh Nopember, lulus 2020
Keahlian
AutoCAD, SketchUp, Revit, Enscape
"""
    report = run_screening_from_payloads(
        vacancy_bytes=_vacancy_bytes(),
        vacancy_filename="Vacancy Detail.pdf",
        cv_payloads=[
            ("cv_rina.txt", cv_rina.encode("utf-8")),
            ("cv_andi.txt", cv_andi.encode("utf-8")),
        ],
        engine_mode="local_hybrid",
        reference_date=date(2024, 10, 1),
        blind_screening=True,
    )

    by_name = {ev.candidate_name: ev for ev in report.evaluations}
    assert set(by_name.keys()) == {"Rina Kartika", "Andi Wijaya"}

    for name in ("Rina Kartika", "Andi Wijaya"):
        ev = by_name[name]
        assert ev.status == "Shortlisted (Wawancara)"
        assert ev.degree_requirement_met is True
        assert ev.missing_required_software == []
        assert ev.extracted_cv.timeline.merged_years_relevant >= 3.0
        assert ev.needs_manual_review is False

    # Pastikan PII institusi & alamat umum ikut tersensor tanpa hardcode
    assert "Universitas Indonesia" not in by_name["Rina Kartika"].pii_audit.anonymized_text
    assert "Sudirman 45, Jakarta Selatan" not in by_name["Rina Kartika"].pii_audit.anonymized_text
    assert "Institut Teknologi Sepuluh Nopember" not in by_name["Andi Wijaya"].pii_audit.anonymized_text


def test_confidence_gate_routes_scanned_or_unparseable_cvs_to_manual_review():
    """CV hasil scan (teks sangat minim) dan CV format non-standar harus masuk 'Perlu Review Manual', bukan ditolak."""
    scanned_cv = b"Nama: Budi"
    unusual_cv = (
        "Budi Santoso\n"
        "Portfolio link: https://example.com/portfolio\n"
        "Saya adalah desainer ruang kreatif yang menyukai arsitektur modern dan proyek hunian.\n"
        "Berpengalaman menggunakan AutoCAD, SketchUp, dan Revit pada beberapa proyek mandiri.\n"
    ).encode("utf-8")

    report = run_screening_from_payloads(
        vacancy_bytes=_vacancy_bytes(),
        vacancy_filename="Vacancy Detail.pdf",
        cv_payloads=[
            ("scanned_budi.txt", scanned_cv),
            ("unusual_budi.txt", unusual_cv),
        ],
        engine_mode="local_hybrid",
        reference_date=date(2024, 10, 1),
    )

    assert report.metadata.manual_review_count == 2
    assert report.metadata.rejected_count == 0
    for ev in report.evaluations:
        assert ev.status == "Perlu Review Manual"
        assert ev.needs_manual_review is True
        assert "masih kuliah/lulusan SMK" not in ev.decision_rationale


def test_rejection_rationale_distinguishes_missing_education_from_student_or_smk():
    """Kandidat dengan riwayat kerja tapi tanpa teks pendidikan tidak boleh diberi alasan 'masih kuliah/lulusan SMK'."""
    cv_no_edu = """Citra Dewi
citra@mail.com
Work Experience
Interior Drafter - PT Ruang Indah
Jan 2023 - Present
- Creating 3D models and drawings with AutoCAD and SketchUp.
Skills
AutoCAD, SketchUp
"""
    report = run_screening_from_payloads(
        vacancy_bytes=_vacancy_bytes(),
        vacancy_filename="Vacancy Detail.pdf",
        cv_payloads=[("citra.txt", cv_no_edu.encode("utf-8"))],
        engine_mode="local_hybrid",
        reference_date=date(2024, 10, 1),
    )
    ev = report.evaluations[0]
    assert ev.status == "Tidak Lolos Kualifikasi"
    assert "masih kuliah/lulusan SMK" not in ev.decision_rationale
    assert "tidak terdeteksi" in ev.decision_rationale


def test_corrupt_file_in_batch_does_not_crash_other_candidates():
    """Satu PDF rusak di tengah batch harus masuk 'Perlu Review Manual' sementara kandidat lain tetap diproses."""
    valid_cv = (SAMPLE_DIR / "CV Sample 1.pdf").read_bytes()
    corrupt_pdf = b"%PDF-1.4-CORRUPTED-GARBAGE-BYTES"

    report = run_screening_from_payloads(
        vacancy_bytes=_vacancy_bytes(),
        vacancy_filename="Vacancy Detail.pdf",
        cv_payloads=[
            ("CV Sample 1.pdf", valid_cv),
            ("corrupt_upload.pdf", corrupt_pdf),
        ],
        engine_mode="local_hybrid",
        reference_date=date(2024, 10, 1),
    )
    assert report.metadata.total_candidates == 2
    assert report.metadata.shortlisted_count == 1
    assert report.metadata.manual_review_count == 1
    corrupt_ev = [e for e in report.evaluations if e.source_filename == "corrupt_upload.pdf"][0]
    assert corrupt_ev.status == "Perlu Review Manual"
    assert corrupt_ev.needs_manual_review is True


def test_deepseek_runtime_failure_falls_back_gracefully_per_cv():
    """Error runtime DeepSeek pada satu CV harus fallback ke local_hybrid; error API key tetap raise ValueError."""
    valid_cv = (SAMPLE_DIR / "CV Sample 1.pdf").read_bytes()

    with patch("src.pipeline.extract_cv_with_deepseek", side_effect=RuntimeError("API timeout")):
        report = run_screening_from_payloads(
            vacancy_bytes=_vacancy_bytes(),
            vacancy_filename="Vacancy Detail.pdf",
            cv_payloads=[("CV Sample 1.pdf", valid_cv)],
            engine_mode="deepseek",
            reference_date=date(2024, 10, 1),
            api_key="sk-dummy-key",
        )
    ev = report.evaluations[0]
    assert ev.extracted_cv.fallback_used is True
    assert any("DeepSeek API gagal merespons" in flag for flag in ev.anomaly_flags)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        run_screening_from_payloads(
            vacancy_bytes=_vacancy_bytes(),
            vacancy_filename="Vacancy Detail.pdf",
            cv_payloads=[("CV Sample 1.pdf", valid_cv)],
            engine_mode="deepseek",
            reference_date=date(2024, 10, 1),
            api_key="",
        )


def test_symmetric_education_scoring_and_post_graduation_audit_flags():
    """S1 selesai setara 100 dengan S1+S2 in-progress; pengalaman pra-lulus & gelar expected tercatat di audit."""
    cv_files = sorted(SAMPLE_DIR.glob("CV Sample *.pdf"))
    payloads = [(p.name, p.read_bytes()) for p in cv_files]
    report = run_screening_from_payloads(
        vacancy_bytes=_vacancy_bytes(),
        vacancy_filename="Vacancy Detail.pdf",
        cv_payloads=payloads,
        engine_mode="local_hybrid",
        reference_date=date(2024, 10, 1),
    )
    by_name = {e.candidate_name: e for e in report.evaluations}

    # Tidak ada bias over-qualification antara S1 selesai (Bayu, Permana) dan S1 selesai + sedang S2 (Indri)
    assert by_name["Bayu"].dimension_scores.education == 100.0
    assert by_name["Indri"].dimension_scores.education == 100.0
    assert by_name["Permana Abadi"].dimension_scores.education == 100.0

    # Bayu memiliki flag expected graduation + catatan selisih durasi total vs pasca-lulus
    assert any("expected" in flag.lower() for flag in by_name["Bayu"].anomaly_flags)
    assert any("pasca-lulus" in flag.lower() for flag in by_name["Bayu"].anomaly_flags)
    assert by_name["Bayu"].extracted_cv.timeline.post_graduation_years_relevant is not None
    assert by_name["Bayu"].extracted_cv.timeline.post_graduation_years_relevant < by_name["Bayu"].extracted_cv.timeline.merged_years_relevant


def test_parse_numeric_dates_and_safe_pii_redaction():
    ref = date(2024, 10, 1)
    assert parse_month_year("01/2020", ref) == (2020, 1)
    assert parse_month_year("12.2023", ref) == (2023, 12)
    assert parse_month_year("Nop 2022", ref) == (2022, 11)
    assert parse_month_year("Kini", ref) == (2024, 10)

    # Pastikan kata bahasa Inggris "its" tidak ikut tersensor sebagai singkatan kampus
    text = "Andi\nThe project reached its deadline at Institut Teknologi Sepuluh Nopember."
    res = anonymize_cv_text(text, candidate_id="CAND-099", redact_institutions=True)
    assert "reached its deadline" in res.anonymized_text
    assert "Institut Teknologi Sepuluh Nopember" not in res.anonymized_text


def test_multi_domain_ai_engineer_vacancy_never_falls_back_to_cad_and_shortlists_relevant_cv():
    """Menguji Vacancy Detail part 2.pdf (AI Engineer) terhadap 5 CV arsitek + 1 CV AI Engineer."""
    ai_vac_path = SAMPLE_DIR / "Vacancy Detail part 2.pdf"
    if not ai_vac_path.exists():
        pytest.skip("Vacancy Detail part 2.pdf tidak tersedia di folder dataset.")

    cv_ai_engineer = """Ilham Pratama
ilham.ai@mail.com | 0812-3456-7890 | Jakarta, Indonesia
Summary
AI Engineer with hands-on experience in Python, Machine Learning, Generative AI/LLM, RAG pipelines, and FastAPI backend deployment.
Work Experience
AI Engineer Intern - PT Teknologi Cerdas
Jan 2024 - Sep 2024
- Built end-to-end machine learning workflows (EDA, data cleaning, feature engineering, model training, and evaluation) using Python, Pandas, NumPy, and Scikit-learn.
- Developed Generative AI and RAG chatbot integrating OpenAI API, LangChain, and ChromaDB vector database with unit testing and data privacy guardrails.
- Created RESTful APIs with FastAPI, SQL queries (PostgreSQL), and Git version control to deploy models in real time for cross-functional teams.
Education
Bachelor of Informatics (S.Kom.), Institut Teknologi Bandung, 2020 - 2024
Skills
Python, Pandas, NumPy, Scikit-learn, PyTorch, LangChain, RAG, FastAPI, SQL, Git, Docker
"""
    dummy_cvs = [(p.name, p.read_bytes()) for p in sorted(SAMPLE_DIR.glob("CV Sample *.pdf"))]
    payloads = dummy_cvs + [("cv_ai_engineer.txt", cv_ai_engineer.encode("utf-8"))]

    report = run_screening_from_payloads(
        vacancy_bytes=ai_vac_path.read_bytes(),
        vacancy_filename=ai_vac_path.name,
        cv_payloads=payloads,
        engine_mode="local_hybrid",
        reference_date=date(2024, 10, 1),
        blind_screening=True,
    )

    # 1. Pastikan parser mendeteksi domain AI Engineer & TIDAK pernah meminta AutoCAD/SketchUp/Revit
    assert report.vacancy.domain == "ai_data"
    assert report.vacancy.accepts_fresh_graduate is True
    for cad_tool in ("AutoCAD", "SketchUp", "Revit"):
        assert cad_tool not in report.vacancy.required_software
    assert "Python" in report.vacancy.required_software
    assert "SQL" in report.vacancy.required_software
    assert "Git" in report.vacancy.required_software

    # 2. Kandidat AI Engineer harus peringkat #1 dan Shortlisted
    top = report.evaluations[0]
    assert top.candidate_name == "Ilham Pratama"
    assert top.status == "Shortlisted (Wawancara)"
    assert top.degree_requirement_met is True
    assert top.missing_required_software == []

    # 3. Kelima CV arsitek harus Tidak Lolos untuk posisi AI Engineer
    for ev in report.evaluations[1:]:
        assert ev.status == "Tidak Lolos Kualifikasi"
        assert "AutoCAD" not in ev.decision_rationale
        assert "Revit" not in ev.decision_rationale

