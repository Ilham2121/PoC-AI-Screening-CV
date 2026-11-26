"""Klien integrasi DeepSeek API (OpenAI-compatible) untuk ekstraksi CV terstruktur & evaluasi rubrik."""

from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from src.schemas import (
    DimensionScores,
    EducationEntry,
    ExtractedCV,
    VacancyCriteria,
    WorkExperienceEntry,
)
from src.skill_normalizer import extract_known_skills_from_text, normalize_skills
from src.timeline import calculate_timeline_summary, infer_bachelor_graduation_ym

# Harga acuan DeepSeek-V3 (deepseek-chat) per 1 juta token (USD, cek ulang sesuai tarif terbaru)
DEEPSEEK_INPUT_COST_PER_M = 0.27
DEEPSEEK_OUTPUT_COST_PER_M = 1.10


def _safe_int_year(value: Any) -> Optional[int]:
    """Mengonversi nilai tahun dari respons LLM secara aman ke int atau None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1950 <= value <= 2100 else None
    m = re.search(r"\b(19\d{2}|20\d{2})\b", str(value))
    return int(m.group(1)) if m else None


def parse_deepseek_json_response(raw_content: str) -> Dict[str, Any]:
    """Mengurai respons JSON dari LLM dengan aman meski dibungkus blok markdown ```json."""
    cleaned = raw_content.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    return json.loads(cleaned)


def estimate_deepseek_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Menghitung estimasi biaya API DeepSeek dalam USD berdasarkan jumlah token."""
    return round(
        (prompt_tokens / 1_000_000.0) * DEEPSEEK_INPUT_COST_PER_M
        + (completion_tokens / 1_000_000.0) * DEEPSEEK_OUTPUT_COST_PER_M,
        6,
    )


def _build_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    key = (api_key or os.getenv("DEEPSEEK_API_KEY", "")).strip()
    if not key or key.startswith("sk-masukkan"):
        raise ValueError("DEEPSEEK_API_KEY belum diisi. Masukkan API Key DeepSeek yang valid.")
    url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).strip()
    return OpenAI(api_key=key, base_url=url, timeout=45.0)


def extract_cv_with_deepseek(
    anonymized_text: str,
    raw_text: str,
    candidate_id: str,
    detected_name: str,
    source_filename: str,
    source_channel: str,
    vacancy: VacancyCriteria,
    reference_date: date,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> ExtractedCV:
    """Mengirim teks CV yang sudah dianonimisasi ke DeepSeek untuk ekstraksi JSON terstruktur."""
    client = _build_client(api_key=api_key)
    model_name = (model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")).strip()

    system_prompt = (
        "Anda adalah sistem ekstraksi informasi CV lintas domain yang objektif dan presisi untuk rekrutmen. "
        "Teks CV sudah melalui penyensoran PII (Blind Screening). "
        "Tugas Anda HANYA mengekstrak fakta yang tertulis di CV ke format JSON yang valid, tanpa mengarang data. "
        "Bedakan secara ketat antara: (1) pengalaman kerja profesional atau proyek teknis relevan, "
        "(2) pengalaman organisasi/kepanitiaan kampus (JANGAN masukkan ke work_experiences), "
        "dan (3) status pendidikan yang sudah lulus (Completed) vs masih berkuliah (In Progress). "
        "Untuk setiap work_experience dan education, nilai relevansinya (is_relevant_domain / is_relevant_field) "
        "SECARA KETAT terhadap posisi dan persyaratan LOWONGAN TARGET yang diberikan (bukan posisi lain). "
        "JANGAN gunakan tanda em dash pada teks penjelasan."
    )

    user_prompt = f"""
LOWONGAN TARGET:
Posisi: {vacancy.title}
Domain Terdeteksi: {vacancy.domain}
Bidang Gelar yang Diterima: {json.dumps(vacancy.required_degree_fields, ensure_ascii=False)}
Tools/Skills Utama: {json.dumps(vacancy.required_software, ensure_ascii=False)}
Tools/Skills Pendukung: {json.dumps(vacancy.preferred_skills, ensure_ascii=False)}
Persyaratan Utama: {json.dumps(vacancy.requirements_raw, ensure_ascii=False)}
Tanggung Jawab: {json.dumps(vacancy.responsibilities, ensure_ascii=False)}
Tanggal Acuan Evaluasi: {reference_date.isoformat()}

TEKS CV (ANONIM):
{anonymized_text}

Kembalikan HANYA JSON valid dengan struktur persis berikut:
{{
  "language_mix": "Bahasa Inggris / Bahasa Indonesia / Campuran Indonesia & Inggris",
  "summary_text": "ringkasan singkat profil dari CV",
  "education": [
    {{
      "degree_level": "SMK/SMA | Diploma | Bachelor | Master | Other",
      "field_of_study": "nama jurusan atau gelar",
      "status": "Completed | In Progress | Unknown",
      "start_year": 2018,
      "end_year": 2022,
      "is_relevant_field": true
    }}
  ],
  "work_experiences": [
    {{
      "role": "nama jabatan",
      "company": "nama perusahaan atau Freelance",
      "employment_type": "Full-time | Freelance | Intern | Contract | Unknown",
      "start_date": "YYYY-MM",
      "end_date": "YYYY-MM atau Present",
      "is_relevant_domain": true,
      "relevance_reason": "alasan singkat dalam Bahasa Indonesia mengapa relevan atau tidak relevan dengan {vacancy.title}",
      "highlights": ["poin tanggung jawab utama"]
    }}
  ],
  "organizational_experiences_count": 0,
  "skills_raw": ["daftar seluruh software, bahasa pemrograman, framework, dan skill teknis yang disebut di CV"],
  "has_building_codes_evidence": false,
  "building_codes_evidence": ["bukti kepatuhan regulasi/standar teknis/testing/validasi/keamanan data yang relevan dengan {vacancy.title}"],
  "has_design_and_build_evidence": false,
  "design_and_build_evidence": ["bukti implementasi proyek end-to-end/hands-on yang langsung relevan dengan {vacancy.title}"],
  "has_supplier_or_material_evidence": false,
  "availability_signals": ["catatan singkat terkait kesiapan bergabung, misalnya masih kerja full-time atau sedang kuliah aktif"]
}}
"""

    resp = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = resp.choices[0].message.content or "{}"
    data = parse_deepseek_json_response(content)

    prompt_toks = getattr(resp.usage, "prompt_tokens", 0) or 0
    comp_toks = getattr(resp.usage, "completion_tokens", 0) or 0
    total_toks = prompt_toks + comp_toks
    cost_usd = estimate_deepseek_cost(prompt_toks, comp_toks)

    # Konversi aman ke Pydantic models
    edu_entries: List[EducationEntry] = []
    has_expected_marker = bool(
        re.search(r"(?:perkiraan\s+tanggal\s+selesai|expected\s+graduation|\(expected\))", anonymized_text, flags=re.I)
    )
    for item in data.get("education", []):
        lvl = item.get("degree_level", "Other")
        if lvl not in {"SMK/SMA", "Diploma", "Bachelor", "Master", "Other"}:
            lvl = "Other"
        st = item.get("status", "Unknown")
        if st not in {"Completed", "In Progress", "Unknown"}:
            st = "Unknown"
        edu_entries.append(
            EducationEntry(
                degree_level=lvl,
                field_of_study=str(item.get("field_of_study", "")),
                institution_redacted="[INSTITUTION_REDACTED]",
                status=st,
                start_year=_safe_int_year(item.get("start_year")),
                end_year=_safe_int_year(item.get("end_year")),
                is_relevant_field=bool(item.get("is_relevant_field", False)),
                is_expected_graduation=bool(item.get("is_expected_graduation", False))
                or (lvl == "Bachelor" and st == "Completed" and has_expected_marker),
            )
        )

    work_entries: List[WorkExperienceEntry] = []
    for item in data.get("work_experiences", []):
        emp_type = item.get("employment_type", "Unknown")
        if emp_type not in {"Full-time", "Freelance", "Intern", "Contract", "Unknown"}:
            emp_type = "Unknown"
        work_entries.append(
            WorkExperienceEntry(
                role=str(item.get("role", "Unknown Role")),
                company=str(item.get("company", "")),
                employment_type=emp_type,
                start_date=item.get("start_date"),
                end_date=item.get("end_date"),
                is_relevant_domain=bool(item.get("is_relevant_domain", False)),
                relevance_reason=str(item.get("relevance_reason", "")),
                highlights=[str(h) for h in item.get("highlights", [])][:10],
            )
        )

    skills_raw = [str(s) for s in data.get("skills_raw", [])]
    # Gabungkan dengan pemindai deterministik agar tidak ada software yang terlewat atau typo
    skills_normalized = normalize_skills(skills_raw + extract_known_skills_from_text(anonymized_text))

    summary_text = str(data.get("summary_text", ""))
    # Jalankan kalkulator timeline deterministik agar deduplikasi overlap & deteksi klaim tetap presisi 100%
    timeline = calculate_timeline_summary(
        experiences=work_entries,
        summary_text=f"{summary_text}\n{anonymized_text[:700]}",
        reference_date=reference_date,
        graduation_ym=infer_bachelor_graduation_ym(edu_entries),
    )

    currently_employed = any(
        w.end_date == "Present" and w.employment_type == "Full-time" for w in work_entries
    )
    currently_studying = any(e.status == "In Progress" for e in edu_entries)
    raw_char_count = len(raw_text.strip())
    extraction_warnings: List[str] = []
    if raw_char_count < 80:
        extraction_warnings.append(
            "Teks yang berhasil diekstrak sangat sedikit (<80 karakter); indikasi PDF hasil pindaian gambar tanpa OCR."
        )

    return ExtractedCV(
        candidate_id=candidate_id,
        source_filename=source_filename,
        source_channel=source_channel,
        detected_name=detected_name,
        language_mix=str(data.get("language_mix", "Bahasa Inggris")),
        summary_text=summary_text,
        education=edu_entries,
        work_experiences=work_entries,
        organizational_experiences_count=int(data.get("organizational_experiences_count", 0) or 0),
        skills_raw=skills_raw,
        skills_normalized=skills_normalized,
        has_building_codes_evidence=bool(data.get("has_building_codes_evidence", False)),
        building_codes_evidence=[str(x) for x in data.get("building_codes_evidence", [])][:4],
        has_design_and_build_evidence=bool(data.get("has_design_and_build_evidence", False)),
        design_and_build_evidence=[str(x) for x in data.get("design_and_build_evidence", [])][:4],
        has_supplier_or_material_evidence=bool(data.get("has_supplier_or_material_evidence", False)),
        availability_signals=[str(x) for x in data.get("availability_signals", [])][:4],
        currently_employed=currently_employed,
        currently_studying=currently_studying,
        timeline=timeline,
        raw_text_char_count=raw_char_count,
        extraction_warnings=extraction_warnings,
        extraction_engine="deepseek",
        llm_tokens_used=total_toks,
        estimated_cost_usd=cost_usd,
    )


def enrich_evaluation_with_deepseek(
    extracted: ExtractedCV,
    vacancy: VacancyCriteria,
    dimension_scores: DimensionScores,
    semantic_score: float,
    matched_sw: List[str],
    missing_sw: List[str],
    degree_met: bool,
    status: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[Dict[str, Any], int, float]:
    """Menghasilkan narasi keputusan, kelebihan, kekurangan, dan pertanyaan wawancara HR berbasis bukti."""
    client = _build_client(api_key=api_key)
    model_name = (model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")).strip()

    system_prompt = (
        "Anda adalah asisten evaluasi rekrutmen HR yang objektif dan transparan. "
        "Tulis alasan keputusan seleksi, poin kekuatan, celah/risiko, dan pertanyaan wawancara dalam Bahasa Indonesia "
        "yang jelas, mengalir, profesional, dan menggunakan istilah teknis dalam Bahasa Inggris secara natural. "
        "DILARANG menggunakan tanda em dash (—). DILARANG menyebut identitas pribadi, usia, gender, atau nama kampus."
    )

    payload = {
        "vacancy_title": vacancy.title,
        "status_keputusan": status,
        "skor_dimensi": dimension_scores.model_dump(),
        "semantic_similarity_tfidf": semantic_score,
        "gelar_terpenuhi": degree_met,
        "software_terpenuhi": matched_sw,
        "software_kurang": missing_sw,
        "durasi_pengalaman_relevan_bersih_tahun": extracted.timeline.merged_years_relevant,
        "has_overlapping_roles": extracted.timeline.has_overlapping_roles,
        "claim_discrepancy_note": extracted.timeline.claim_discrepancy_note,
        "education": [e.model_dump(exclude={"institution_original"}) for e in extracted.education],
        "work_experiences": [w.model_dump() for w in extracted.work_experiences],
        "building_codes_evidence": extracted.building_codes_evidence,
        "availability_signals": extracted.availability_signals,
    }

    user_prompt = f"""
Berdasarkan data terstruktur kandidat anonim berikut, susun penjelasan evaluasi untuk tim HR dalam format JSON valid:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Format JSON yang wajib dikembalikan:
{{
  "decision_rationale": "Ringkasan 2 sampai 3 kalimat objektif yang menjelaskan alasan status keputusan berdasarkan bukti fakta CV.",
  "strengths": ["3 sampai 4 poin kekuatan utama kandidat yang relevan dengan lowongan"],
  "gaps_and_risks": ["2 sampai 4 poin celah kualifikasi atau hal yang perlu diwaspadai"],
  "interview_questions": ["3 pertanyaan wawancara spesifik dan tajam untuk mengklarifikasi pengalaman atau potensi risiko kandidat ini"]
}}
"""

    resp = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    parsed = parse_deepseek_json_response(content)

    # Bersihkan berjaga-jaga jika LLM menyelipkan karakter em dash
    def _strip_em_dash(val: Any) -> Any:
        if isinstance(val, str):
            return val.replace("—", ", ").replace("–", " s.d. ")
        if isinstance(val, list):
            return [_strip_em_dash(x) for x in val]
        return val

    cleaned = {k: _strip_em_dash(v) for k, v in parsed.items()}
    prompt_toks = getattr(resp.usage, "prompt_tokens", 0) or 0
    comp_toks = getattr(resp.usage, "completion_tokens", 0) or 0
    return cleaned, (prompt_toks + comp_toks), estimate_deepseek_cost(prompt_toks, comp_toks)
