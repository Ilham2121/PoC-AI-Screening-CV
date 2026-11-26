"""Modul Etika & Mitigasi Bias: Penyensoran PII (Blind Screening) sebelum tahap penilaian AI."""

from __future__ import annotations

import re
from typing import List, Tuple

from src.schemas import PIIAuditItem, PIIAuditResult

HEADER_SKIP_WORDS = {
    "curriculum vitae",
    "resume",
    "cv",
    "profile",
    "about me",
    "ringkasan",
    "summary",
    "dibuat dengan profil jobstreet",
}

COMMON_NON_NAME_WORDS = {
    "architect",
    "architectural",
    "drafter",
    "interior",
    "designer",
    "junior",
    "senior",
    "visualization",
    "profile",
    "indonesia",
    "jakarta",
    "bandung",
    "bekasi",
    "experience",
    "education",
    "skills",
}

# Gunakan [ \t]+ (spasi horizontal saja) agar regex tidak pernah melintasi baris baru (\n)
INSTITUTION_PATTERNS = [
    r"\b(?:[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+){0,2}[ \t]+Institute[ \t]+of[ \t]+[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+){0,2})\b",
    r"\b(?:Institut[ \t]+(?:Teknologi|Pertanian|Seni|Sains)[ \t]+[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+){0,3})\b",
    r"\b(?:Universitas[ \t]+[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+){0,3})\b",
    r"\b(?:University[ \t]+of[ \t]+[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+){0,4})\b",
    r"\b(?:[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+){0,2}[ \t]+University)\b",
    r"\b(?:Politeknik[ \t]+[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+){0,3})\b",
    r"\b(?:Sekolah[ \t]+Tinggi[ \t]+[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+){0,3})\b",
    r"\b(?:State[ \t]+Vocational[ \t]+High[ \t]+School[ \t]+\d+[ \t]+[A-Z][A-Za-z]+)\b",
    r"\b(?:Vocational[ \t]+High[ \t]+School[ \t]+\d+[ \t]+[A-Z][A-Za-z]+)\b",
    r"\b(?:Vocational[ \t]+School[ \t]+SMKN?[ \t]*\d+[ \t]+[A-Z][A-Za-z]+)\b",
    r"\b(?:SMKN?[ \t]*\d+[ \t]+[A-Z][A-Za-z]+)\b",
    r"\b(?:ITB|UGM|UNDIP|UNPAR|BINUS|UNAIR)\b",
]


def detect_candidate_name(raw_text: str) -> str:
    """Mendeteksi nama kandidat dari baris awal dokumen CV."""
    for raw_line in raw_text.splitlines()[:8]:
        line = raw_line.strip().strip("|").strip()
        if not line:
            continue
        low = line.lower()
        if low in HEADER_SKIP_WORDS:
            continue
        if "@" in line or re.search(r"\d{5,}", line):
            continue
        if re.match(r"^(?:[A-Z]\s+){4,}[A-Z]$", line):
            continue
        words = [w for w in re.split(r"\s+", line) if w]
        if 1 <= len(words) <= 4 and all(re.match(r"^[A-Za-z.\-']+$", w) for w in words):
            if not all(w.lower() in COMMON_NON_NAME_WORDS for w in words):
                return line.title() if line.isupper() else line
    return "Kandidat Tanpa Nama"


def _sub_and_count(pattern: str, repl: str, text: str, flags: int = 0) -> Tuple[str, int]:
    return re.subn(pattern, repl, text, flags=flags)


def anonymize_cv_text(
    raw_text: str,
    candidate_id: str,
    redact_institutions: bool = True,
) -> PIIAuditResult:
    """Menyensor data pribadi (PII) dan penanda bias dari teks CV untuk Blind Screening."""
    detected_name = detect_candidate_name(raw_text)
    text = raw_text
    audit_items: List[PIIAuditItem] = []

    # 1. Sensor alamat email
    text, email_count = _sub_and_count(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "[EMAIL_REDACTED]",
        text,
    )
    if email_count:
        audit_items.append(PIIAuditItem(category="Email", count=email_count, sample_redacted_label="[EMAIL_REDACTED]"))

    # 2. Sensor nomor telepon (Indonesia & internasional, aman dari rentang tahun seperti 2021 - 2023)
    text, phone_count = _sub_and_count(
        r"(?<!\d)(?:\+[ \t]*62[ \t\-]*\d|628|08)[\d\- \t]{6,14}\d(?!\d)",
        "[PHONE_REDACTED]",
        text,
    )
    if phone_count:
        audit_items.append(PIIAuditItem(category="Nomor Telepon", count=phone_count, sample_redacted_label="[PHONE_REDACTED]"))

    # 3. Sensor tempat & tanggal lahir / usia (mencegah diskriminasi usia)
    dob_count = 0
    dob_patterns = [
        r"(?:born[ \t]+in[ \t]+[A-Za-z \t]+,[ \t]*\d{1,2}[ \t]+[A-Za-z]+[ \t]+\d{4})",
        r"(?:lahir[ \t]+di[ \t]+[A-Za-z \t]+,?[ \t]*\d{1,2}[ \t]+[A-Za-z]+[ \t]+\d{4})",
        r"(?:(?:date[ \t]+of[ \t]+birth|dob|tanggal[ \t]+lahir)[ \t]*[:\-]?[ \t]*\d{1,2}[ \t/\-][A-Za-z0-9]+[ \t/\-]\d{2,4})",
        r"\b\d{1,2}[ \t]+(?:January|February|March|April|May|June|July|August|September|October|November|December|Januari|Februari|Maret|Mei|Juni|Juli|Agustus|Oktober|Desember)[ \t]+(?:19\d{2}|200\d)\b",
    ]
    for pat in dob_patterns:
        text, c = _sub_and_count(pat, "[DOB_REDACTED]", text, flags=re.IGNORECASE)
        dob_count += c
    if dob_count:
        audit_items.append(
            PIIAuditItem(category="Tanggal Lahir / Usia", count=dob_count, sample_redacted_label="[DOB_REDACTED]")
        )

    # 4. Sensor alamat domisili lengkap di baris kontak (pola umum jalan/kota, spasi horizontal saja)
    addr_count = 0
    addr_patterns = [
        r"\b(?:Jl\.|Jalan)[ \t]+[A-Za-z0-9. \t,\-]{5,55}",
        r"\b[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+){0,2}[ \t]+\d{1,4}[ \t]*,[ \t]*(?:Jakarta|Bandung|Surabaya|Bekasi|Tangerang|Depok|Semarang|Yogyakarta|Medan|Makassar)(?:[ \t]+(?:Barat|Timur|Utara|Selatan|Pusat))?\b",
        r"\b(?:South|North|West|East|Central)[ \t]+Jakarta[ \t]*[\-,][ \t]*Indonesia\b",
        r"\b(?:Bandung|Jakarta|Bekasi|Surabaya|Tangerang|Depok|Semarang|Yogyakarta|Medan|Makassar)[ \t]*,[ \t]*Indonesia(?:[ \t]+\d{5})?\b",
    ]
    for pat in addr_patterns:
        text, c = _sub_and_count(pat, "[ADDRESS_REDACTED]", text, flags=re.IGNORECASE)
        addr_count += c
    if addr_count:
        audit_items.append(
            PIIAuditItem(category="Alamat / Domisili", count=addr_count, sample_redacted_label="[ADDRESS_REDACTED]")
        )

    # 5. Sensor nama kandidat (mencegah bias gender, etnis, dan preferensi personal)
    name_count = 0
    if detected_name and detected_name != "Kandidat Tanpa Nama":
        text, c = _sub_and_count(
            rf"\b{re.escape(detected_name)}\b",
            f"[{candidate_id}]",
            text,
            flags=re.IGNORECASE,
        )
        name_count += c
        for part in detected_name.split():
            if len(part) >= 4 and part.lower() not in COMMON_NON_NAME_WORDS:
                text, c2 = _sub_and_count(
                    rf"\b{re.escape(part)}\b",
                    f"[{candidate_id}]",
                    text,
                    flags=re.IGNORECASE,
                )
                name_count += c2
    if name_count:
        audit_items.append(
            PIIAuditItem(category="Nama Kandidat", count=name_count, sample_redacted_label=f"[{candidate_id}]")
        )

    # 6. Opsional: Sensor nama institusi pendidikan (mencegah halo effect / bias almamater)
    if redact_institutions:
        inst_count = 0
        for pat in INSTITUTION_PATTERNS:
            text, c = _sub_and_count(pat, "[INSTITUTION_REDACTED]", text, flags=re.IGNORECASE)
            inst_count += c
        if inst_count:
            audit_items.append(
                PIIAuditItem(
                    category="Nama Institusi Pendidikan (Anti Halo-Effect)",
                    count=inst_count,
                    sample_redacted_label="[INSTITUTION_REDACTED]",
                )
            )

    total_redactions = sum(item.count for item in audit_items)
    return PIIAuditResult(
        candidate_id=candidate_id,
        detected_name=detected_name,
        anonymized_text=text,
        redacted_items=audit_items,
        total_redactions=total_redactions,
    )
