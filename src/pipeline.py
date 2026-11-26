"""Orkestrator end-to-end untuk Otomatisasi Seleksi CV (Ingestion -> Parsing -> Blind PII -> Ekstraksi -> Ranking)."""

from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from src.deepseek_client import extract_cv_with_deepseek
from src.fallback_extractor import extract_cv_local_hybrid
from src.pdf_parser import extract_text_from_bytes, parse_vacancy_text
from src.pii_anonymizer import anonymize_cv_text
from src.schemas import (
    ExtractedCV,
    PIIAuditResult,
    ScoringWeights,
    ScreeningReport,
    ScreeningRunMetadata,
    VacancyCriteria,
)
from src.scorer import evaluate_candidates

ProgressCallback = Callable[[int, int, str], None]


def _detect_source_channel(raw_bytes: bytes, raw_text: str, filename: str) -> str:
    """Mendeteksi simulasi kanal masuk CV (JobStreet, LinkedIn, atau Portal Karir Langsung)."""
    lower_bytes = raw_bytes.decode("latin-1", errors="ignore").lower()
    lower_txt = raw_text.lower()
    if "jobstreet" in lower_bytes or "jobstreet" in lower_txt:
        return "JobStreet Profile Export"
    if "linkedin" in lower_bytes or "linkedin" in lower_txt:
        return "LinkedIn Jobs"
    if "glints" in lower_txt:
        return "Glints"
    return "Direct Portal / Email Intake"


def run_screening_from_payloads(
    vacancy_bytes: bytes,
    vacancy_filename: str,
    cv_payloads: List[Tuple[str, bytes]],
    engine_mode: str = "local_hybrid",
    reference_date: Optional[date] = None,
    blind_screening: bool = True,
    weights: Optional[ScoringWeights] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> ScreeningReport:
    """Menjalankan seluruh pipeline seleksi CV dari bytes dokumen (cocok untuk Streamlit maupun CLI)."""
    t0 = time.perf_counter()
    ref_date = reference_date or date(2024, 10, 1)
    active_weights = weights or ScoringWeights()

    vacancy_text = extract_text_from_bytes(vacancy_bytes, vacancy_filename)
    vacancy: VacancyCriteria = parse_vacancy_text(vacancy_text)

    extracted_list: List[ExtractedCV] = []
    pii_audits: Dict[str, PIIAuditResult] = {}
    total_cvs = len(cv_payloads)

    for idx, (filename, file_bytes) in enumerate(cv_payloads, start=1):
        cand_id = f"CAND-{idx:03d}"
        if on_progress:
            on_progress(idx, total_cvs, f"Memproses {filename} ({cand_id})...")

        try:
            raw_text = extract_text_from_bytes(file_bytes, filename)
        except Exception as exc:
            # Jangan hentikan seluruh batch jika satu file rusak/terenkripsi; arahkan ke review manual
            source_channel = _detect_source_channel(file_bytes, "", filename)
            pii_audits[cand_id] = PIIAuditResult(
                candidate_id=cand_id,
                detected_name=Path(filename).stem,
                anonymized_text="",
                redacted_items=[],
                total_redactions=0,
            )
            extracted_list.append(
                ExtractedCV(
                    candidate_id=cand_id,
                    source_filename=filename,
                    source_channel=source_channel,
                    detected_name=Path(filename).stem,
                    raw_text_char_count=0,
                    extraction_warnings=[
                        f"Gagal membaca dokumen '{filename}' ({type(exc).__name__}: {exc}). Perlu pemeriksaan manual."
                    ],
                    extraction_engine="local_hybrid",
                )
            )
            continue

        source_channel = _detect_source_channel(file_bytes, raw_text, filename)

        pii_result = anonymize_cv_text(
            raw_text=raw_text,
            candidate_id=cand_id,
            redact_institutions=blind_screening,
        )
        pii_audits[cand_id] = pii_result

        text_for_ai = pii_result.anonymized_text if blind_screening else raw_text

        # Hindari panggilan API LLM yang sia-sia jika teks dokumen kosong/pindaian gambar (<80 karakter)
        if engine_mode == "deepseek" and len(raw_text.strip()) >= 80:
            last_err: Optional[Exception] = None
            extracted: Optional[ExtractedCV] = None
            for attempt in range(2):
                try:
                    extracted = extract_cv_with_deepseek(
                        anonymized_text=text_for_ai,
                        raw_text=raw_text,
                        candidate_id=cand_id,
                        detected_name=pii_result.detected_name,
                        source_filename=filename,
                        source_channel=source_channel,
                        vacancy=vacancy,
                        reference_date=ref_date,
                        api_key=api_key,
                        model=model,
                    )
                    break
                except ValueError:
                    # Kesalahan konfigurasi API Key harus langsung dihentikan agar jelas ke pengguna
                    raise
                except Exception as exc:
                    last_err = exc
                    if attempt == 0:
                        time.sleep(0.5)

            if extracted is None:
                extracted = extract_cv_local_hybrid(
                    anonymized_text=text_for_ai,
                    raw_text=raw_text,
                    candidate_id=cand_id,
                    detected_name=pii_result.detected_name,
                    source_filename=filename,
                    source_channel=source_channel,
                    vacancy=vacancy,
                    reference_date=ref_date,
                )
                extracted.fallback_used = True
                extracted.extraction_warnings.append(
                    f"DeepSeek API gagal merespons ({type(last_err).__name__}), dialihkan otomatis ke mode Lokal Hybrid."
                )
        else:
            extracted = extract_cv_local_hybrid(
                anonymized_text=text_for_ai,
                raw_text=raw_text,
                candidate_id=cand_id,
                detected_name=pii_result.detected_name,
                source_filename=filename,
                source_channel=source_channel,
                vacancy=vacancy,
                reference_date=ref_date,
            )
        extracted_list.append(extracted)

    if on_progress:
        on_progress(total_cvs, total_cvs, "Menghitung skor Hybrid, TF-IDF, dan meranking kandidat...")

    evaluations = evaluate_candidates(
        vacancy=vacancy,
        extracted_list=extracted_list,
        pii_audits=pii_audits,
        weights=active_weights,
        engine_mode=engine_mode,
        api_key=api_key,
        model=model,
    )

    elapsed = round(time.perf_counter() - t0, 2)
    total_tokens = sum(ev.extracted_cv.llm_tokens_used for ev in evaluations)
    total_cost = round(sum(ev.extracted_cv.estimated_cost_usd for ev in evaluations), 6)

    metadata = ScreeningRunMetadata(
        run_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        reference_date=ref_date,
        vacancy_title=vacancy.title,
        engine_mode="deepseek" if engine_mode == "deepseek" else "local_hybrid",
        blind_screening_enabled=blind_screening,
        total_candidates=len(evaluations),
        shortlisted_count=sum(1 for e in evaluations if e.status == "Shortlisted (Wawancara)"),
        kiv_count=sum(1 for e in evaluations if e.status == "Pertimbangan Khusus (KIV)"),
        manual_review_count=sum(1 for e in evaluations if e.status == "Perlu Review Manual"),
        rejected_count=sum(1 for e in evaluations if e.status == "Tidak Lolos Kualifikasi"),
        processing_seconds=elapsed,
        total_tokens_used=total_tokens,
        total_estimated_cost_usd=total_cost,
        weights_used=active_weights,
    )

    return ScreeningReport(metadata=metadata, vacancy=vacancy, evaluations=evaluations)


def run_screening_pipeline(
    vacancy_path: Path | str,
    cv_paths: List[Path | str],
    engine_mode: str = "local_hybrid",
    reference_date: Optional[date] = None,
    blind_screening: bool = True,
    weights: Optional[ScoringWeights] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> ScreeningReport:
    """Wrapper praktis untuk menjalankan pipeline dari daftar path file di disk."""
    v_path = Path(vacancy_path)
    payloads: List[Tuple[str, bytes]] = []
    for p in cv_paths:
        cv_p = Path(p)
        payloads.append((cv_p.name, cv_p.read_bytes()))

    return run_screening_from_payloads(
        vacancy_bytes=v_path.read_bytes(),
        vacancy_filename=v_path.name,
        cv_payloads=payloads,
        engine_mode=engine_mode,
        reference_date=reference_date,
        blind_screening=blind_screening,
        weights=weights,
        api_key=api_key,
        model=model,
        on_progress=on_progress,
    )
