"""Skrip CLI end-to-end untuk menjalankan pipeline otomatisasi seleksi CV dan mengekspor Excel + JSON.

Contoh penggunaan:
    python run_pipeline.py
    python run_pipeline.py --engine deepseek --reference-date 2024-10-01
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from src.excel_exporter import export_screening_report_to_excel
from src.pipeline import run_screening_pipeline

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BASE_DIR.parent / "Soal Test BP AI Specialist 3"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Platform Otomatisasi Seleksi CV Berbasis Hybrid AI/ML (Multi-Domain)."
    )
    parser.add_argument(
        "--vacancy",
        type=Path,
        default=DEFAULT_DATASET_DIR / "Vacancy Detail.pdf",
        help="Path dokumen lowongan (PDF/DOCX/TXT).",
    )
    parser.add_argument(
        "--cv-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Folder yang berisi file CV kandidat (PDF/DOCX).",
    )
    parser.add_argument(
        "--engine",
        choices=["local_hybrid", "deepseek"],
        default="local_hybrid",
        help="Mesin ekstraksi & evaluasi: 'local_hybrid' (offline) atau 'deepseek' (DeepSeek API).",
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        default="2024-10-01",
        help="Tanggal acuan perhitungan masa kerja dalam format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--no-blind",
        action="store_true",
        help="Nonaktifkan Blind Screening (penyensoran PII & institusi).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder tujuan penyimpanan laporan Excel dan JSON.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(BASE_DIR / ".env")
    args = parse_args()

    ref_date = date.fromisoformat(args.reference_date)
    cv_paths = sorted(
        [
            p
            for p in args.cv_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".pdf", ".docx"}
            and p.name.lower().startswith("cv")
        ]
    )
    if not cv_paths:
        raise SystemExit(f"Tidak ditemukan file CV (*.pdf / *.docx) di folder: {args.cv_dir}")

    print(f"[INFO] Lowongan : {args.vacancy.name}")
    print(f"[INFO] Jumlah CV: {len(cv_paths)} dokumen")
    print(f"[INFO] Engine   : {args.engine} | Blind Screening: {not args.no_blind} | Ref Date: {ref_date}")

    def _progress(curr: int, total: int, msg: str) -> None:
        print(f"  [{curr}/{total}] {msg}")

    report = run_screening_pipeline(
        vacancy_path=args.vacancy,
        cv_paths=cv_paths,
        engine_mode=args.engine,
        reference_date=ref_date,
        blind_screening=not args.no_blind,
        on_progress=_progress,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "DeepSeek" if args.engine == "deepseek" else "LocalHybrid"
    xlsx_path = args.output_dir / f"Hasil_Screening_CV_{suffix}_Ilham_A_A.xlsx"
    json_path = args.output_dir / f"Hasil_Screening_CV_{suffix}_Ilham_A_A.json"

    export_screening_report_to_excel(report, xlsx_path)
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    print("\n=== RINGKASAN HASIL RANKING KANDIDAT ===")
    for ev in report.evaluations:
        print(
            f"#{ev.rank} | {ev.candidate_name:<15} | Skor: {ev.overall_score:5.1f} | "
            f"{ev.status:<26} | Relevan: {ev.extracted_cv.timeline.merged_years_relevant:.1f} thn | "
            f"SW Kurang: {', '.join(ev.missing_required_software) or '-'}"
        )

    print(
        f"\n[OK] Ringkasan Status: Shortlisted={report.metadata.shortlisted_count} | "
        f"KIV={report.metadata.kiv_count} | Review Manual={report.metadata.manual_review_count} | "
        f"Tidak Lolos={report.metadata.rejected_count}"
    )
    print(f"[OK] Laporan Excel tersimpan di: {xlsx_path}")
    print(f"[OK] Output JSON tersimpan di  : {json_path}")
    print(
        f"[OK] Durasi: {report.metadata.processing_seconds:.2f} detik | "
        f"Token: {report.metadata.total_tokens_used} | "
        f"Estimasi Biaya: ${report.metadata.total_estimated_cost_usd:.5f}"
    )


if __name__ == "__main__":
    main()
