"""Unit tests untuk generator laporan Excel multi-sheet."""

from datetime import date
from pathlib import Path

from openpyxl import load_workbook

from src.excel_exporter import export_screening_report_to_excel
from src.pipeline import run_screening_pipeline

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "Soal Test BP AI Specialist 3"


def test_exports_structured_four_sheet_excel_report(tmp_path: Path):
    cv_files = sorted(SAMPLE_DIR.glob("CV Sample *.pdf"))
    report = run_screening_pipeline(
        vacancy_path=SAMPLE_DIR / "Vacancy Detail.pdf",
        cv_paths=cv_files,
        engine_mode="local_hybrid",
        reference_date=date(2024, 10, 1),
        blind_screening=True,
    )

    out_xlsx = tmp_path / "Hasil_Screening_CV.xlsx"
    export_screening_report_to_excel(report, out_xlsx)

    assert out_xlsx.exists()
    wb = load_workbook(out_xlsx)
    assert wb.sheetnames == [
        "Ringkasan_Ranking",
        "Detail_Skoring",
        "Audit_Bias_Anomali",
        "Panduan_Wawancara",
    ]
    ws_summary = wb["Ringkasan_Ranking"]
    # Pastikan semua 5 nama kandidat muncul di sheet ringkasan
    all_values = [
        str(cell.value)
        for row in ws_summary.iter_rows(values_only=False)
        for cell in row
        if cell.value is not None
    ]
    joined = " ".join(all_values)
    for name in ["Bayu", "Permana Abadi", "Mohamad", "Indri", "Dani"]:
        assert name in joined
