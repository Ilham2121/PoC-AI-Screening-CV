"""Generator laporan Excel multi-sheet yang rapi dan siap diserahkan ke tim HR."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from src.schemas import ScreeningReport

HEADER_FILL = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
SUBHEADER_FILL = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")
GREEN_FILL = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")
AMBER_FILL = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
BLUE_FILL = PatternFill(start_color="DBEAFE", end_color="DBEAFE", fill_type="solid")
RED_FILL = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")

HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
TITLE_FONT = Font(name="Calibri", size=14, bold=True, color="0F172A")
BOLD_FONT = Font(name="Calibri", size=10, bold=True, color="0F172A")
BODY_FONT = Font(name="Calibri", size=10, color="1E293B")
GREEN_FONT = Font(name="Calibri", size=10, bold=True, color="14532D")
AMBER_FONT = Font(name="Calibri", size=10, bold=True, color="92400E")
BLUE_FONT = Font(name="Calibri", size=10, bold=True, color="1E40AF")
RED_FONT = Font(name="Calibri", size=10, bold=True, color="991B1B")

THIN_BORDER = Border(
    left=Side(style="thin", color="CBD5E1"),
    right=Side(style="thin", color="CBD5E1"),
    top=Side(style="thin", color="CBD5E1"),
    bottom=Side(style="thin", color="CBD5E1"),
)


def _style_table_headers(ws, row_idx: int, col_count: int) -> None:
    ws.row_dimensions[row_idx].height = 28
    for col in range(1, col_count + 1):
        cell = ws.cell(row=row_idx, column=col)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = THIN_BORDER


def _apply_status_style(cell, status: str) -> None:
    if "Shortlisted" in status:
        cell.fill = GREEN_FILL
        cell.font = GREEN_FONT
    elif "Pertimbangan" in status:
        cell.fill = AMBER_FILL
        cell.font = AMBER_FONT
    elif "Review Manual" in status:
        cell.fill = BLUE_FILL
        cell.font = BLUE_FONT
    else:
        cell.fill = RED_FILL
        cell.font = RED_FONT


def _set_col_widths(ws, widths: dict[int, float]) -> None:
    for col_idx, width in widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = width


def build_screening_workbook(report: ScreeningReport) -> Workbook:
    """Membangun objek OpenPyXL Workbook 4 sheet dari objek ScreeningReport."""
    wb = Workbook()

    # =========================================================================
    # SHEET 1: Ringkasan_Ranking
    # =========================================================================
    ws1 = wb.active
    ws1.title = "Ringkasan_Ranking"
    ws1.views.sheetView[0].showGridLines = True

    ws1["A1"] = f"Laporan Otomatisasi Seleksi CV: {report.vacancy.title}"
    ws1["A1"].font = TITLE_FONT
    ws1.merge_cells("A1:K1")

    meta_info = [
        ("Waktu Eksekusi", report.metadata.run_timestamp, "Engine AI", report.metadata.engine_mode.upper()),
        (
            "Tanggal Acuan Masa Kerja",
            report.metadata.reference_date.isoformat(),
            "Blind Screening (PII Redaction)",
            "Aktif" if report.metadata.blind_screening_enabled else "Nonaktif",
        ),
        (
            "Total CV Diproses",
            report.metadata.total_candidates,
            "Shortlisted / KIV / Review Manual / Tidak Lolos",
            f"{report.metadata.shortlisted_count} / {report.metadata.kiv_count} / {report.metadata.manual_review_count} / {report.metadata.rejected_count}",
        ),
        (
            "Durasi Pemrosesan",
            f"{report.metadata.processing_seconds:.2f} detik",
            "Estimasi Biaya Token DeepSeek",
            f"${report.metadata.total_estimated_cost_usd:.5f} ({report.metadata.total_tokens_used} token)",
        ),
    ]

    for offset, (k1, v1, k2, v2) in enumerate(meta_info, start=3):
        ws1.cell(row=offset, column=1, value=k1).font = BOLD_FONT
        ws1.cell(row=offset, column=2, value=v1).font = BODY_FONT
        ws1.cell(row=offset, column=5, value=k2).font = BOLD_FONT
        ws1.cell(row=offset, column=6, value=v2).font = BODY_FONT
        for c in (1, 2, 5, 6):
            ws1.cell(row=offset, column=c).fill = SUBHEADER_FILL

    headers_s1 = [
        "Rank",
        "ID Anonim",
        "Nama Kandidat",
        "Sumber Kanal",
        "Skor Total (0-100)",
        "Status Rekomendasi",
        "Gelar S1 Terpenuhi",
        "Pengalaman Relevan Bersih (Thn)",
        "Software Wajib Terpenuhi",
        "Software Wajib Kurang",
        "Alasan Keputusan Utama",
    ]
    start_row = 8
    for c_idx, title in enumerate(headers_s1, start=1):
        ws1.cell(row=start_row, column=c_idx, value=title)
    _style_table_headers(ws1, start_row, len(headers_s1))

    for row_offset, ev in enumerate(report.evaluations, start=start_row + 1):
        values = [
            ev.rank,
            ev.candidate_id,
            ev.candidate_name,
            ev.source_channel,
            ev.overall_score,
            ev.status,
            "Ya" if ev.degree_requirement_met else "Belum",
            ev.extracted_cv.timeline.merged_years_relevant,
            ", ".join(ev.matched_required_software) or "-",
            ", ".join(ev.missing_required_software) or "-",
            ev.decision_rationale,
        ]
        ws1.row_dimensions[row_offset].height = 48
        for c_idx, val in enumerate(values, start=1):
            cell = ws1.cell(row=row_offset, column=c_idx, value=val)
            cell.font = BODY_FONT
            cell.border = THIN_BORDER
            horiz = "center" if c_idx in {1, 2, 5, 6, 7, 8} else "left"
            cell.alignment = Alignment(horizontal=horiz, vertical="center", wrap_text=True)
            if c_idx == 5:
                cell.number_format = "0.0"
                cell.font = BOLD_FONT
            elif c_idx == 6:
                _apply_status_style(cell, str(val))
            elif c_idx == 8:
                cell.number_format = "0.0"

    ws1.freeze_panes = "A9"
    _set_col_widths(
        ws1,
        {1: 7, 2: 12, 3: 18, 4: 22, 5: 14, 6: 24, 7: 14, 8: 16, 9: 22, 10: 18, 11: 55},
    )

    # =========================================================================
    # SHEET 2: Detail_Skoring
    # =========================================================================
    ws2 = wb.create_sheet(title="Detail_Skoring")
    headers_s2 = [
        "Rank",
        "Nama Kandidat",
        "Skor Total",
        "Pengalaman Relevan (30%)",
        "Software & Teknis (25%)",
        "Pendidikan (15%)",
        "Kecocokan Tugas (20%)",
        "Regulasi & Kesiapan (10%)",
        "Skor Kemiripan Leksikal (TF-IDF)",
        "Durasi Mentah Semua (Bln)",
        "Durasi Bersih Relevan (Bln)",
        "Kekuatan Utama",
        "Celah / Risiko",
    ]
    for c_idx, title in enumerate(headers_s2, start=1):
        ws2.cell(row=1, column=c_idx, value=title)
    _style_table_headers(ws2, 1, len(headers_s2))

    for r_idx, ev in enumerate(report.evaluations, start=2):
        d = ev.dimension_scores
        row_vals = [
            ev.rank,
            ev.candidate_name,
            ev.overall_score,
            d.relevant_experience,
            d.software_technical,
            d.education,
            d.responsibility_fit,
            d.regulatory_readiness,
            ev.semantic_similarity_score,
            ev.extracted_cv.timeline.raw_sum_months_all,
            ev.extracted_cv.timeline.merged_months_relevant,
            "\n".join(f"- {s}" for s in ev.strengths),
            "\n".join(f"- {g}" for g in ev.gaps_and_risks),
        ]
        ws2.row_dimensions[r_idx].height = 76
        for c_idx, val in enumerate(row_vals, start=1):
            cell = ws2.cell(row=r_idx, column=c_idx, value=val)
            cell.font = BODY_FONT
            cell.border = THIN_BORDER
            horiz = "center" if c_idx in {1, 3, 4, 5, 6, 7, 8, 9, 10, 11} else "left"
            cell.alignment = Alignment(horizontal=horiz, vertical="top", wrap_text=True)
            if c_idx in {3, 4, 5, 6, 7, 8, 9}:
                cell.number_format = "0.0"

    ws2.freeze_panes = "A2"
    _set_col_widths(
        ws2,
        {1: 7, 2: 18, 3: 11, 4: 14, 5: 14, 6: 13, 7: 14, 8: 14, 9: 14, 10: 14, 11: 14, 12: 45, 13: 45},
    )

    # =========================================================================
    # SHEET 3: Audit_Bias_Anomali
    # =========================================================================
    ws3 = wb.create_sheet(title="Audit_Bias_Anomali")
    headers_s3 = [
        "ID Anonim",
        "Nama Kandidat",
        "File Sumber",
        "Total Sensor PII",
        "Kategori PII Disensor (Blind Screening)",
        "Terdeteksi Overlap Kerja",
        "Flag Inkonsistensi Klaim",
        "Catatan Anomali & Audit Data",
    ]
    for c_idx, title in enumerate(headers_s3, start=1):
        ws3.cell(row=1, column=c_idx, value=title)
    _style_table_headers(ws3, 1, len(headers_s3))

    for r_idx, ev in enumerate(report.evaluations, start=2):
        row_vals = [
            ev.candidate_id,
            ev.candidate_name,
            ev.source_filename,
            ev.pii_redactions_count,
            ", ".join(ev.pii_redacted_categories) or "-",
            "Ya" if ev.extracted_cv.timeline.has_overlapping_roles else "Tidak",
            "Ya" if ev.extracted_cv.timeline.claim_discrepancy_flag else "Tidak",
            "\n".join(f"- {a}" for a in ev.anomaly_flags) or "Tidak ada anomali signifikan.",
        ]
        ws3.row_dimensions[r_idx].height = 60
        for c_idx, val in enumerate(row_vals, start=1):
            cell = ws3.cell(row=r_idx, column=c_idx, value=val)
            cell.font = BODY_FONT
            cell.border = THIN_BORDER
            horiz = "center" if c_idx in {1, 4, 6, 7} else "left"
            cell.alignment = Alignment(horizontal=horiz, vertical="top", wrap_text=True)

    ws3.freeze_panes = "A2"
    _set_col_widths(ws3, {1: 12, 2: 18, 3: 18, 4: 14, 5: 38, 6: 16, 7: 16, 8: 55})

    # =========================================================================
    # SHEET 4: Panduan_Wawancara
    # =========================================================================
    ws4 = wb.create_sheet(title="Panduan_Wawancara")
    headers_s4 = [
        "Rank",
        "Nama Kandidat",
        "Status Rekomendasi",
        "Sinyal Ketersediaan (Join ASAP)",
        "Daftar Pertanyaan Konfirmasi Wawancara HR",
    ]
    for c_idx, title in enumerate(headers_s4, start=1):
        ws4.cell(row=1, column=c_idx, value=title)
    _style_table_headers(ws4, 1, len(headers_s4))

    for r_idx, ev in enumerate(report.evaluations, start=2):
        row_vals = [
            ev.rank,
            ev.candidate_name,
            ev.status,
            "\n".join(f"- {s}" for s in ev.extracted_cv.availability_signals) or "Tidak ada kendala jadwal yang tercatat.",
            "\n".join(f"{i}. {q}" for i, q in enumerate(ev.interview_questions, start=1)),
        ]
        ws4.row_dimensions[r_idx].height = 75
        for c_idx, val in enumerate(row_vals, start=1):
            cell = ws4.cell(row=r_idx, column=c_idx, value=val)
            cell.font = BODY_FONT
            cell.border = THIN_BORDER
            horiz = "center" if c_idx in {1, 3} else "left"
            cell.alignment = Alignment(horizontal=horiz, vertical="top", wrap_text=True)
            if c_idx == 3:
                _apply_status_style(cell, str(val))

    ws4.freeze_panes = "A2"
    _set_col_widths(ws4, {1: 7, 2: 18, 3: 24, 4: 38, 5: 68})

    return wb


def export_screening_report_to_excel(report: ScreeningReport, output_path: Union[str, Path]) -> Path:
    """Menyimpan laporan Excel ke disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb = build_screening_workbook(report)
    wb.save(out)
    return out


def export_screening_report_to_excel_bytes(report: ScreeningReport) -> bytes:
    """Menghasilkan bytes file Excel untuk tombol unduh di aplikasi Streamlit."""
    wb = build_screening_workbook(report)
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()
