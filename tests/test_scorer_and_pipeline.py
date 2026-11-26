"""Integration & unit tests untuk Hybrid CV Screening Pipeline (Mode Lokal Hybrid & DeepSeek)."""

from datetime import date
from pathlib import Path

from src.deepseek_client import parse_deepseek_json_response
from src.pipeline import run_screening_pipeline

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "Soal Test BP AI Specialist 3"


def test_end_to_end_local_hybrid_pipeline_ranks_five_sample_cvs_accurately():
    cv_files = sorted(SAMPLE_DIR.glob("CV Sample *.pdf"))
    assert len(cv_files) == 5

    report = run_screening_pipeline(
        vacancy_path=SAMPLE_DIR / "Vacancy Detail.pdf",
        cv_paths=cv_files,
        engine_mode="local_hybrid",
        reference_date=date(2024, 10, 1),
        blind_screening=True,
    )

    assert report.metadata.total_candidates == 5
    assert len(report.evaluations) == 5

    by_name = {ev.candidate_name: ev for ev in report.evaluations}
    assert set(by_name.keys()) == {"Bayu", "Permana Abadi", "Mohamad", "Indri", "Dani"}

    # 1. Top 2 harus Indri dan Bayu (keduanya S1 Arsitektur + punya AutoCAD, SketchUp, Revit)
    top2_names = {report.evaluations[0].candidate_name, report.evaluations[1].candidate_name}
    assert top2_names == {"Indri", "Bayu"}
    assert by_name["Indri"].status == "Shortlisted (Wawancara)"
    assert by_name["Bayu"].status == "Shortlisted (Wawancara)"
    assert set(by_name["Indri"].matched_required_software) == {"AutoCAD", "SketchUp", "Revit"}
    assert set(by_name["Bayu"].matched_required_software) == {"AutoCAD", "SketchUp", "Revit"}

    # 2. Bayu dan Indri harus terdeteksi memiliki overlapping roles yang sudah dideduplikasi
    assert by_name["Bayu"].extracted_cv.timeline.has_overlapping_roles is True
    assert by_name["Indri"].extracted_cv.timeline.has_overlapping_roles is True

    # 3. Permana Abadi berada di peringkat 3 (KIV / Pertimbangan), kurang Revit, organisasi dipisahkan
    assert report.evaluations[2].candidate_name == "Permana Abadi"
    assert "Revit" in by_name["Permana Abadi"].missing_required_software
    assert by_name["Permana Abadi"].degree_requirement_met is True
    assert by_name["Permana Abadi"].extracted_cv.organizational_experiences_count >= 2

    # 4. Mohamad dan Dani belum memenuhi syarat gelar S1 (masih kuliah) dan tidak punya Revit
    assert by_name["Mohamad"].degree_requirement_met is False
    assert by_name["Dani"].degree_requirement_met is False
    assert "Revit" in by_name["Mohamad"].missing_required_software
    assert "Revit" in by_name["Dani"].missing_required_software
    assert by_name["Mohamad"].status == "Tidak Lolos Kualifikasi"
    assert by_name["Dani"].status == "Tidak Lolos Kualifikasi"

    # 5. Mohamad harus memicu flag inkonsistensi klaim 10 tahun vs riwayat tertulis
    assert by_name["Mohamad"].extracted_cv.timeline.claim_discrepancy_flag is True

    # 6. Dani: pengalaman structural/concrete testing tidak boleh dianggap penuh sebagai design & build
    assert by_name["Dani"].extracted_cv.timeline.merged_years_relevant < 1.5
    assert "Tanggal Lahir / Usia" in by_name["Dani"].pii_redacted_categories


def test_parse_deepseek_json_response_strips_markdown_fences():
    raw_content = '```json\n{"education": [], "work_experiences": [], "skills_raw": ["AutoCAD"]}\n```'
    parsed = parse_deepseek_json_response(raw_content)
    assert parsed["skills_raw"] == ["AutoCAD"]
