"""Unit tests untuk layout-aware PDF/DOCX parser."""

from pathlib import Path
from src.pdf_parser import extract_text_from_file, parse_vacancy_document

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "Soal Test BP AI Specialist 3"


def test_extracts_multicolumn_cv_without_scrambling_education_and_skills():
    """CV Sample 5 memiliki 2 kolom di bagian bawah (Education di kiri, Skills di kanan)."""
    text = extract_text_from_file(SAMPLE_DIR / "CV Sample 5.pdf")
    assert "Dani" in text
    assert "Borobudur University" in text
    assert "AutoCAD" in text
    # Pastikan blok Formal Education tidak tercampur langsung di baris yang sama dengan Soft Skills
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    joined = "\n".join(lines)
    assert "FORMAL EDUCATION" in joined
    assert "AutoCAD" in joined
    # Blok Education harus muncul dalam satu kesatuan sebelum atau sesudah blok Skills, tidak selang-seling
    edu_idx = joined.find("Borobudur University")
    smk_idx = joined.find("State Vocational High School 3 Kuningan")
    autocad_idx = joined.find("V-Ray Render")
    assert abs(smk_idx - edu_idx) < 250
    assert autocad_idx != -1


def test_extracts_vacancy_criteria_from_pdf():
    """Vacancy Detail.pdf harus terurai menjadi kriteria terstruktur."""
    vacancy = parse_vacancy_document(SAMPLE_DIR / "Vacancy Detail.pdf")
    assert "Junior Architect" in vacancy.title
    assert vacancy.min_years_relevant_experience == 2.0
    assert "AutoCAD" in vacancy.required_software
    assert "SketchUp" in vacancy.required_software
    assert "Revit" in vacancy.required_software
    assert len(vacancy.responsibilities) >= 4
    assert vacancy.requires_asap_join is True
