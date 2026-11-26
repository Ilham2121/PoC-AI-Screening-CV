"""Kalkulator durasi pengalaman kerja bersih (deduplikasi overlap tanggal + deteksi anomali klaim)."""

from __future__ import annotations

import re
from datetime import date
from typing import List, Optional, Tuple

from src.schemas import EducationEntry, TimelineSummary, WorkExperienceEntry

MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "januari": 1,
    "feb": 2,
    "february": 2,
    "februari": 2,
    "mar": 3,
    "march": 3,
    "maret": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "mei": 5,
    "jun": 6,
    "june": 6,
    "juni": 6,
    "jul": 7,
    "july": 7,
    "juli": 7,
    "aug": 8,
    "august": 8,
    "agu": 8,
    "ags": 8,
    "agustus": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "okt": 10,
    "oktober": 10,
    "nov": 11,
    "november": 11,
    "nop": 11,
    "nopember": 11,
    "dec": 12,
    "december": 12,
    "des": 12,
    "desember": 12,
}

PRESENT_TOKENS = {"present", "saat ini", "sekarang", "current", "now", "ongoing", "kini", "today"}


def parse_month_year(raw: Optional[str], reference_date: date) -> Optional[Tuple[int, int]]:
    """Mengubah string tanggal (YYYY-MM, MM/YYYY, 'Sep 2023', 'Des 2022', 'Saat ini', '2021') ke tuple (tahun, bulan)."""
    if not raw:
        return None
    cleaned = raw.strip()
    low = cleaned.lower()
    if low in PRESENT_TOKENS:
        return (reference_date.year, reference_date.month)

    # Format ISO YYYY-MM
    iso_match = re.match(r"^(\d{4})-(\d{1,2})$", cleaned)
    if iso_match:
        return (int(iso_match.group(1)), min(12, max(1, int(iso_match.group(2)))))

    # Format numerik MM/YYYY atau MM.YYYY
    num_match = re.match(r"^(\d{1,2})[/.](\d{4})$", cleaned)
    if num_match and 1 <= int(num_match.group(1)) <= 12:
        return (int(num_match.group(2)), int(num_match.group(1)))

    # Format "Month YYYY" / "Month, YYYY" (Inggris / Indonesia)
    my_match = re.match(r"^([A-Za-z]+)\.?,?\s+(\d{4})$", cleaned)
    if my_match:
        m_key = my_match.group(1).lower()
        year = int(my_match.group(2))
        if m_key in MONTH_MAP:
            return (year, MONTH_MAP[m_key])

    # Format tahun saja "2021"
    y_match = re.match(r"^(\d{4})$", cleaned)
    if y_match:
        return (int(y_match.group(1)), 1)

    return None


def _to_month_index(ym: Tuple[int, int]) -> int:
    return ym[0] * 12 + (ym[1] - 1)


def _merge_intervals(intervals: List[Tuple[int, int]]) -> int:
    """Menghitung total bulan unik dari kumpulan interval [start, end] inklusif."""
    if not intervals:
        return 0
    sorted_iv = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int]] = [sorted_iv[0]]
    for start, end in sorted_iv[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return sum((end - start + 1) for start, end in merged)


def _extract_claimed_years(summary_text: str) -> Optional[float]:
    """Menemukan klaim jumlah tahun pengalaman di paragraf ringkasan CV."""
    patterns = [
        r"(?:with\s+)?(\d+(?:\.\d+)?)\s*\+?\s*years?\s+of\s+experience",
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+experience",
        r"pengalaman\s+(?:kerja\s+)?(?:selama\s+)?(?:lebih\s+dari\s+)?(\d+(?:\.\d+)?)\s*\+?\s*tahun",
        r"(\d+(?:\.\d+)?)\s*\+?\s*tahun\s+pengalaman",
    ]
    for pat in patterns:
        match = re.search(pat, summary_text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def relevant_months_after(
    experiences: List[WorkExperienceEntry],
    start_ym: Tuple[int, int],
    reference_date: date,
) -> int:
    """Menghitung bulan unik pengalaman relevan yang terjadi mulai bulan tertentu (mis. setelah lulus)."""
    cutoff = _to_month_index(start_ym)
    ref_idx = _to_month_index((reference_date.year, reference_date.month))
    intervals: List[Tuple[int, int]] = []
    for exp in experiences:
        if not exp.is_relevant_domain:
            continue
        s_ym = parse_month_year(exp.start_date, reference_date)
        if not s_ym:
            continue
        e_ym = parse_month_year(exp.end_date, reference_date) or (reference_date.year, reference_date.month)
        s_idx = max(_to_month_index(s_ym), cutoff)
        e_idx = min(_to_month_index(e_ym), ref_idx)
        if e_idx >= s_idx:
            intervals.append((s_idx, e_idx))
    return _merge_intervals(intervals)


def infer_bachelor_graduation_ym(education: List[EducationEntry]) -> Optional[Tuple[int, int]]:
    """Menentukan perkiraan (tahun, bulan) kelulusan S1 relevan pertama yang berstatus Completed."""
    completed_bachelors = [
        e for e in education if e.degree_level == "Bachelor" and e.status == "Completed" and e.end_year
    ]
    if not completed_bachelors:
        return None
    grad_year = min(e.end_year for e in completed_bachelors if e.end_year is not None)
    # Gunakan pertengahan tahun (Juli) sebagai pendekatan standar bila bulan kelulusan tidak tercatat
    return (grad_year, 7)


def calculate_timeline_summary(
    experiences: List[WorkExperienceEntry],
    summary_text: str,
    reference_date: date,
    graduation_ym: Optional[Tuple[int, int]] = None,
) -> TimelineSummary:
    """Menghitung masa kerja mentah vs bersih (deduplikasi overlap) serta deteksi inkonsistensi klaim."""
    all_intervals: List[Tuple[int, int, str]] = []
    relevant_intervals: List[Tuple[int, int, str]] = []

    ref_idx = _to_month_index((reference_date.year, reference_date.month))

    for exp in experiences:
        start_ym = parse_month_year(exp.start_date, reference_date)
        end_ym = parse_month_year(exp.end_date, reference_date)
        if not start_ym:
            continue
        if not end_ym:
            end_ym = (reference_date.year, reference_date.month)

        s_idx = _to_month_index(start_ym)
        e_idx = min(_to_month_index(end_ym), ref_idx)
        if e_idx < s_idx:
            continue

        label = f"{exp.role} ({exp.company})" if exp.company else exp.role
        all_intervals.append((s_idx, e_idx, label))
        if exp.is_relevant_domain:
            relevant_intervals.append((s_idx, e_idx, label))

    raw_sum_all = sum((e - s + 1) for s, e, _ in all_intervals)
    merged_all = _merge_intervals([(s, e) for s, e, _ in all_intervals])

    raw_sum_rel = sum((e - s + 1) for s, e, _ in relevant_intervals)
    merged_rel = _merge_intervals([(s, e) for s, e, _ in relevant_intervals])

    # Identifikasi pasangan pekerjaan yang saling tumpang tindih
    overlap_notes: List[str] = []
    sorted_all = sorted(all_intervals, key=lambda x: (x[0], x[1]))
    for i in range(len(sorted_all)):
        s1, e1, l1 = sorted_all[i]
        for j in range(i + 1, len(sorted_all)):
            s2, e2, l2 = sorted_all[j]
            if s2 <= e1:
                overlap_months = min(e1, e2) - s2 + 1
                if overlap_months >= 2:
                    overlap_notes.append(
                        f"Periode tumpang tindih ({overlap_months} bulan) antara '{l1}' dan '{l2}' telah dideduplikasi."
                    )

    has_overlap = raw_sum_all > merged_all and len(overlap_notes) > 0
    merged_years_all = round(merged_all / 12.0, 2)
    merged_years_rel = round(merged_rel / 12.0, 2)

    post_grad_years_rel: Optional[float] = None
    if graduation_ym is not None:
        post_grad_months = relevant_months_after(experiences, graduation_ym, reference_date)
        post_grad_years_rel = round(post_grad_months / 12.0, 2)

    claimed = _extract_claimed_years(summary_text)
    discrepancy_flag = False
    discrepancy_note = ""
    if claimed is not None and (claimed - merged_years_all) >= 2.0:
        discrepancy_flag = True
        discrepancy_note = (
            f"Ringkasan CV mengklaim {claimed:g} tahun pengalaman, tetapi riwayat kerja yang terdokumentasi "
            f"hanya mencakup {merged_years_all:.1f} tahun (selisih {claimed - merged_years_all:.1f} tahun). "
            "Perlu verifikasi bukti kerja pada tahap wawancara."
        )

    return TimelineSummary(
        raw_sum_months_all=raw_sum_all,
        merged_months_all=merged_all,
        raw_sum_months_relevant=raw_sum_rel,
        merged_months_relevant=merged_rel,
        merged_years_relevant=merged_years_rel,
        post_graduation_years_relevant=post_grad_years_rel,
        has_overlapping_roles=has_overlap,
        overlap_notes=overlap_notes[:6],
        claimed_years_in_summary=claimed,
        claim_discrepancy_flag=discrepancy_flag,
        claim_discrepancy_note=discrepancy_note,
    )
