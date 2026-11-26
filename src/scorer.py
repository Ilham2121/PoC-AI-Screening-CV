"""Hybrid Scoring Engine: menggabungkan Rule-Based Gate, TF-IDF Lexical Similarity, Confidence Gate, dan Explainability."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.deepseek_client import enrich_evaluation_with_deepseek
from src.schemas import (
    CandidateEvaluation,
    DimensionScores,
    ExtractedCV,
    PIIAuditResult,
    ScoringWeights,
    VacancyCriteria,
)
from src.skill_normalizer import match_required_skills


def compute_batch_semantic_scores(
    vacancy: VacancyCriteria,
    extracted_list: List[ExtractedCV],
) -> Dict[str, float]:
    """Menghitung skor kemiripan leksikal TF-IDF (0-100) antara profil kerja kandidat dan lowongan.

    Catatan: TF-IDF digunakan sebagai baseline leksikal yang cepat dan deterministik pada PoC;
    pada fase produksi disarankan memakai multilingual dense embeddings (mis. BGE-M3 / multilingual-e5)
    agar kesamaan makna lintas Bahasa Indonesia dan Inggris tertangkap lebih akurat.
    """
    if not extracted_list:
        return {}

    vacancy_doc = " ".join(
        [vacancy.title]
        + vacancy.requirements_raw
        + vacancy.responsibilities
        + vacancy.required_software
        + vacancy.key_technical_skills
    )

    candidate_docs: List[str] = []
    for ext in extracted_list:
        relevant_work_text = " ".join(
            f"{w.role} {' '.join(w.highlights)}" for w in ext.work_experiences if w.is_relevant_domain
        )
        # Jika tidak ada pengalaman relevan, tetap gunakan summary + skills agar tidak kosong
        cand_doc = " ".join(
            [
                ext.summary_text,
                relevant_work_text,
                " ".join(ext.skills_normalized),
                " ".join(ext.building_codes_evidence),
            ]
        ).strip()
        candidate_docs.append(cand_doc or "no relevant experience")

    corpus = [vacancy_doc] + candidate_docs
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Skalakan cosine similarity (umumnya berkisar 0.05 - 0.45 untuk teks CV vs JD) ke skala 0-100 yang terbaca
    result: Dict[str, float] = {}
    for ext, sim in zip(extracted_list, similarities):
        scaled = min(100.0, round(float(sim) * 230.0, 1))
        result[ext.candidate_id] = scaled
    return result


def _score_education(extracted: ExtractedCV, vacancy: VacancyCriteria) -> Tuple[float, bool, str, str]:
    """Menilai kesesuaian gelar pendidikan (Completed Bachelor/Master di bidang relevan terhadap lowongan).

    Mengembalikan (skor, degree_met, catatan_lengkap, alasan_singkat_celah).
    Skor S1 selesai setara (100) baik dengan maupun tanpa studi S2 lanjutan agar tidak menimbulkan
    bias over-qualification terhadap syarat lowongan S1.
    """
    req_fields_label = (
        "/".join(vacancy.required_degree_fields[:3])
        if vacancy.required_degree_fields
        else f"bidang relevan ({vacancy.title})"
    )

    completed_relevant_bachelor = any(
        e.degree_level in {"Bachelor", "Master"} and e.status == "Completed" and e.is_relevant_field
        for e in extracted.education
    )
    in_progress_master = any(
        e.degree_level == "Master" and e.status == "In Progress" and e.is_relevant_field
        for e in extracted.education
    )
    in_progress_bachelor = any(
        e.degree_level == "Bachelor" and e.status == "In Progress" and e.is_relevant_field
        for e in extracted.education
    )
    has_diploma_relevant = any(
        e.degree_level == "Diploma" and e.is_relevant_field for e in extracted.education
    )
    has_smk_relevant = any(
        e.degree_level == "SMK/SMA" and e.is_relevant_field for e in extracted.education
    )
    has_other_completed_degree = any(
        e.degree_level in {"Bachelor", "Master"} and e.status == "Completed" and not e.is_relevant_field
        for e in extracted.education
    )

    if completed_relevant_bachelor and in_progress_master:
        return (
            100.0,
            True,
            f"Memenuhi syarat gelar S1 bidang {req_fields_label} (dan sedang menempuh studi S2).",
            "",
        )
    if completed_relevant_bachelor:
        return 100.0, True, f"Memenuhi syarat gelar S1 di bidang {req_fields_label}.", ""
    if in_progress_bachelor:
        return (
            45.0,
            False,
            f"Belum memenuhi syarat gelar sarjana (S1 {req_fields_label} masih berstatus aktif kuliah / belum lulus).",
            "belum menyelesaikan gelar sarjana (masih berstatus kuliah aktif)",
        )
    if has_diploma_relevant:
        return (
            60.0,
            False,
            f"Memiliki latar pendidikan Diploma di bidang {req_fields_label}, namun belum memenuhi syarat gelar sarjana (S1).",
            "berlatar belakang Diploma (belum memenuhi syarat gelar sarjana S1)",
        )
    if has_other_completed_degree:
        return (
            50.0,
            False,
            f"Memiliki gelar sarjana, namun bidang studi tidak sesuai dengan {req_fields_label}.",
            f"bidang gelar sarjana tidak sesuai dengan {req_fields_label}",
        )
    if has_smk_relevant:
        return (
            30.0,
            False,
            "Latar pendidikan terakhir SMK/setara, belum memiliki gelar sarjana.",
            "latar pendidikan terakhir SMK/setara (belum memiliki gelar sarjana)",
        )
    return (
        20.0,
        False,
        "Informasi gelar pendidikan formal tidak terdeteksi pada teks CV (perlu verifikasi manual).",
        "informasi pendidikan formal tidak terdeteksi di teks CV",
    )


def _score_relevant_experience(
    extracted: ExtractedCV,
    vacancy: VacancyCriteria,
) -> Tuple[float, bool]:
    """Menilai durasi bersih pengalaman kerja relevan + kualitas eksposur proyek end-to-end sesuai domain."""
    years = extracted.timeline.merged_years_relevant

    # Kasus lowongan menerima Fresh Graduate tanpa syarat minimal tahun kerja
    if vacancy.accepts_fresh_graduate and vacancy.min_years_relevant_experience == 0.0:
        if years >= 1.0:
            base = min(92.0, 82.0 + years * 4.0)
        elif years > 0.0:
            base = 74.0 + years * 8.0
        elif extracted.has_design_and_build_evidence:
            base = 76.0
        else:
            base = 18.0

        if extracted.has_design_and_build_evidence and years > 0.0:
            base += 8.0
        if extracted.timeline.claim_discrepancy_flag:
            base -= 12.0

        exp_met = years > 0.0 or extracted.has_design_and_build_evidence
        return round(min(100.0, max(0.0, base)), 1), exp_met

    min_years = max(0.5, vacancy.min_years_relevant_experience)

    # Skor dasar berbasis rasio durasi relevan terhadap syarat minimal
    if years >= min_years:
        base = min(85.0, 75.0 + (years - min_years) * 4.0)
    elif years >= min_years * 0.8:
        base = 68.0
    else:
        base = max(10.0, (years / min_years) * 65.0)

    # Bonus eksposur nyata ke proyek end-to-end sesuai domain
    if extracted.has_design_and_build_evidence:
        base += 13.0

    # Penalti ringan jika pengalaman hanya murni freelance/intern tanpa tim proyek penuh waktu
    has_fulltime_relevant = any(
        w.is_relevant_domain and w.employment_type == "Full-time" for w in extracted.work_experiences
    )
    if not has_fulltime_relevant and years > 0:
        base -= 10.0

    # Penalti jika ada anomali klaim pengalaman tidak didukung riwayat tertulis
    if extracted.timeline.claim_discrepancy_flag:
        base -= 12.0

    # Toleransi >= (min_years - 0.25) tahun jika ada bukti eksposur proyek end-to-end yang kuat
    exp_met = years >= min_years or (years >= (min_years - 0.25) and extracted.has_design_and_build_evidence)
    return round(min(100.0, max(0.0, base)), 1), exp_met


def _score_software_technical(
    extracted: ExtractedCV,
    vacancy: VacancyCriteria,
) -> Tuple[float, List[str], List[str]]:
    """Menilai kelengkapan software/tools wajib + ekosistem pendukung secara dinamis dari lowongan."""
    required = vacancy.required_software
    if not required:
        return 80.0, [], []

    matched, missing = match_required_skills(extracted.skills_normalized, required)

    req_ratio = len(matched) / max(1, len(required))
    score = req_ratio * 82.0

    # Pool bonus dinamis berdasarkan preferred_skills dari dokumen lowongan atau ekosistem domain
    if vacancy.preferred_skills:
        bonus_pool = set(vacancy.preferred_skills)
    elif vacancy.domain == "architecture":
        bonus_pool = {"Lumion", "Enscape", "V-Ray", "Twinmotion", "Adobe Photoshop", "D5 Render", "BIM"}
    else:
        req_lower = {r.lower() for r in required}
        bonus_pool = {s for s in extracted.skills_normalized if s.lower() not in req_lower}

    bonus_hits = len([s for s in extracted.skills_normalized if s in bonus_pool])
    score += min(18.0, bonus_hits * 4.5)

    return round(min(100.0, score), 1), matched, missing


def _score_responsibility_fit(
    extracted: ExtractedCV,
    semantic_score: float,
    vacancy: VacancyCriteria,
) -> float:
    """Menggabungkan TF-IDF lexical similarity dengan cakupan pilar tanggung jawab sesuai domain lowongan."""
    relevant_parts = [
        f"{w.role} {' '.join(w.highlights)}" for w in extracted.work_experiences if w.is_relevant_domain
    ]
    if vacancy.domain != "architecture":
        # Untuk peran tech/fresh-grad, ikut sertakan summary & skills yang sudah dinormalisasi
        relevant_parts.extend([extracted.summary_text, " ".join(extracted.skills_normalized)])
    relevant_blob = " ".join(relevant_parts).lower()

    if vacancy.domain == "architecture":
        pillars = [
            any(k in relevant_blob for k in ["concept", "conceptual", "designing", "design proposal", "konsep", "desain", "perencanaan"]),
            any(k in relevant_blob for k in ["monitor", "supervis", "timeline", "coordina", "project manager", "mengawasi", "pengawasan", "koordinasi", "jadwal"]),
            any(k in relevant_blob for k in ["3d", "modeling", "modelling", "rendering", "working drawing", "shop drawing", "detailed drawing", "gambar kerja", "pemodelan", "visualisasi"]),
            any(k in relevant_blob for k in ["interior", "material", "spec", "supplier", "lighting", "furniture", "pemasok", "pencahayaan", "furnitur", "spesifikasi"]),
        ]
        pillar_score = (sum(1 for p in pillars if p) / 4.0) * 100.0
    elif vacancy.domain == "ai_data":
        pillars = [
            any(k in relevant_blob for k in ["eda", "data analysis", "preprocessing", "data cleaning", "feature engineering", "pandas", "numpy", "sql"]),
            any(k in relevant_blob for k in ["machine learning", "deep learning", "scikit", "tensorflow", "pytorch", "training", "evaluation", "model", "algorithm"]),
            any(k in relevant_blob for k in ["llm", "generative ai", "genai", "langchain", "rag", "openai", "gemini", "vector database", "prompt", "nlp", "agent"]),
            any(k in relevant_blob for k in ["fastapi", "flask", "rest api", "api", "pipeline", "deploy", "git", "docker", "testing", "pytest", "streamlit"]),
        ]
        pillar_score = (sum(1 for p in pillars if p) / 4.0) * 100.0
    elif vacancy.domain == "software_engineering":
        pillars = [
            any(k in relevant_blob for k in ["architecture", "design", "system", "microservice", "scalable"]),
            any(k in relevant_blob for k in ["backend", "frontend", "fullstack", "python", "javascript", "typescript", "java", "golang", "react", "node"]),
            any(k in relevant_blob for k in ["api", "rest", "database", "sql", "postgresql", "mysql", "mongodb", "redis", "integration"]),
            any(k in relevant_blob for k in ["testing", "unit test", "ci/cd", "docker", "kubernetes", "git", "deploy", "cloud"]),
        ]
        pillar_score = (sum(1 for p in pillars if p) / 4.0) * 100.0
    else:
        # Domain general: ukur kecocokan kata kunci langsung dari daftar tanggung jawab lowongan
        resp_hits = 0
        resp_list = vacancy.responsibilities[:4] or ["project", "analysis", "report", "team"]
        for resp in resp_list:
            tokens = [t for t in resp.lower().split() if len(t) >= 5]
            if any(t in relevant_blob for t in tokens):
                resp_hits += 1
        pillar_score = (resp_hits / max(1, len(resp_list))) * 100.0

    # Hybrid 55% cakupan pilar tanggung jawab + 45% TF-IDF lexical cosine similarity
    combined = 0.55 * pillar_score + 0.45 * semantic_score
    return round(min(100.0, max(0.0, combined)), 1)


def _score_regulatory_readiness(extracted: ExtractedCV, vacancy: VacancyCriteria) -> float:
    """Menilai kepatuhan standar/regulasi atau kesiapan produksi serta kesiapan bergabung."""
    if vacancy.requires_regulatory_compliance:
        score = 35.0
        if extracted.has_building_codes_evidence:
            score += 45.0
        if extracted.has_supplier_or_material_evidence:
            score += 10.0
    else:
        # Untuk posisi non-regulasi bangunan (mis. AI/Software): nilai standar testing/keamanan data & kolaborasi
        score = 45.0
        if extracted.has_building_codes_evidence:
            score += 35.0
        if extracted.has_supplier_or_material_evidence:
            score += 10.0

    # Pertimbangan kesiapan jadwal secara simetris
    if not extracted.currently_employed and not extracted.currently_studying:
        score += 10.0
    elif extracted.currently_employed and extracted.currently_studying:
        score -= 5.0

    return round(min(100.0, max(0.0, score)), 1)


def _check_manual_review_gate(extracted: ExtractedCV) -> Tuple[bool, List[str]]:
    """Confidence gate (Human-in-the-Loop): mencegah penolakan otomatis pada CV yang gagal/minim terbaca."""
    reasons: List[str] = []
    if extracted.raw_text_char_count < 80 and not extracted.work_experiences and not extracted.education:
        reasons.append(
            "Teks dokumen sangat sedikit atau gagal dibaca (indikasi PDF hasil pindaian gambar tanpa OCR / file rusak)."
        )
    elif len(extracted.work_experiences) == 0 and len(extracted.education) == 0:
        reasons.append(
            "Riwayat pengalaman kerja dan pendidikan formal tidak terdeteksi otomatis (kemungkinan format CV tidak standar)."
        )
    elif extracted.fallback_used and (len(extracted.work_experiences) == 0 or len(extracted.education) == 0):
        reasons.append(
            "Pemanggilan LLM gagal dan fallback lokal belum menangkap seluruh struktur CV secara utuh."
        )

    for warn in extracted.extraction_warnings:
        if warn not in reasons and any(k in warn.lower() for k in ["gagal membaca", "ocr", "tidak standar"]):
            reasons.append(warn)

    return len(reasons) > 0, reasons


def _determine_status(
    overall_score: float,
    degree_met: bool,
    matched_sw: List[str],
    missing_sw: List[str],
    exp_met: bool,
    needs_manual_review: bool = False,
) -> str:
    """Menentukan kategori rekomendasi HR secara objektif dan adaptif lintas ukuran tech stack."""
    if needs_manual_review:
        return "Perlu Review Manual"

    total_req_sw = len(matched_sw) + len(missing_sw)
    if total_req_sw == 0:
        sw_met = True
    elif total_req_sw <= 4:
        sw_met = len(missing_sw) == 0
    else:
        # Untuk posisi dengan stack luas (>4 tool/skill), toleransi >= 65% penguasaan stack inti
        sw_met = (len(matched_sw) / total_req_sw) >= 0.65

    if overall_score >= 72.0 and degree_met and sw_met and exp_met:
        return "Shortlisted (Wawancara)"
    if overall_score >= 60.0 and degree_met and (exp_met or len(matched_sw) >= max(1, total_req_sw // 2)):
        return "Pertimbangan Khusus (KIV)"
    if overall_score >= 72.0 and exp_met and sw_met:
        return "Pertimbangan Khusus (KIV)"
    return "Tidak Lolos Kualifikasi"


def _build_local_explanations(
    extracted: ExtractedCV,
    vacancy: VacancyCriteria,
    dim_scores: DimensionScores,
    matched_sw: List[str],
    missing_sw: List[str],
    degree_met: bool,
    edu_note: str,
    edu_gap_reason: str,
    status: str,
    manual_review_reasons: List[str],
) -> Tuple[str, List[str], List[str], List[str]]:
    """Menyusun alasan keputusan, kelebihan, risiko, dan pertanyaan wawancara dalam Bahasa Indonesia."""
    strengths: List[str] = []
    gaps: List[str] = []
    questions: List[str] = []

    if status == "Perlu Review Manual":
        rationale = (
            "Dialihkan ke antrean review manual HR (tidak ditolak otomatis) karena "
            + "; ".join(manual_review_reasons)
        )
        if matched_sw:
            strengths.append(f"Terdapat sebagian software/tools yang terdeteksi di teks CV: {', '.join(matched_sw)}.")
        else:
            strengths.append("Dokumen berhasil diterima sistem; memerlukan pemeriksaan visual oleh tim HR.")
        gaps.extend(manual_review_reasons[:4])
        req_tools_str = ", ".join(vacancy.required_software) or f"tools utama posisi {vacancy.title}"
        questions.extend(
            [
                "Mohon konfirmasi latar belakang pendidikan formal dan tahun kelulusan kandidat dari dokumen CV asli.",
                f"Mohon verifikasi riwayat pengalaman kerja dan proyek yang relevan dengan posisi {vacancy.title}.",
                f"Konfirmasi penguasaan keahlian/tools utama lowongan ({req_tools_str}) serta kesiapan bergabung kandidat.",
            ]
        )
        return rationale, strengths[:4], gaps[:4], questions[:3]

    domain_label = (
        "arsitektur/design & build"
        if vacancy.domain == "architecture"
        else f"bidang {vacancy.title}"
    )

    if degree_met:
        strengths.append(edu_note)
    else:
        gaps.append(edu_note)

    if vacancy.required_software:
        if len(missing_sw) == 0:
            strengths.append(
                f"Menguasai seluruh software/tools utama lowongan ({', '.join(matched_sw)}) beserta teknologi pendukung."
            )
        else:
            if matched_sw:
                strengths.append(f"Menguasai sebagian software/tools yang diminta ({', '.join(matched_sw)}).")
            gaps.append(f"Belum mencantumkan penguasaan software/tools wajib: {', '.join(missing_sw)}.")

    rel_years = extracted.timeline.merged_years_relevant
    if vacancy.accepts_fresh_graduate and vacancy.min_years_relevant_experience == 0.0:
        if rel_years > 0:
            strengths.append(
                f"Memiliki {rel_years:.1f} tahun pengalaman relevan bersih di {domain_label} (posisi ini juga terbuka untuk Fresh Graduate)."
            )
        elif extracted.has_design_and_build_evidence:
            strengths.append(
                f"Memiliki bukti portofolio/proyek hands-on yang relevan dengan posisi {vacancy.title}."
            )
        else:
            gaps.append(
                f"Belum ditemukan pengalaman kerja maupun proyek portofolio yang relevan dengan posisi {vacancy.title}."
            )
    elif rel_years >= vacancy.min_years_relevant_experience:
        strengths.append(
            f"Memiliki {rel_years:.1f} tahun pengalaman relevan bersih (setelah deduplikasi periode tumpang tindih)."
        )
    elif rel_years >= vacancy.min_years_relevant_experience - 0.3:
        strengths.append(
            f"Memiliki {rel_years:.1f} tahun pengalaman relevan langsung pada proyek {domain_label}."
        )
        gaps.append(
            f"Durasi pengalaman kerja relevan bersih ({rel_years:.1f} tahun) berada tipis di batas minimal {vacancy.min_years_relevant_experience:g} tahun pada tanggal acuan."
        )
    else:
        gaps.append(
            f"Pengalaman kerja bersih yang relevan di {domain_label} baru sekitar {rel_years:.1f} tahun (di bawah syarat minimal {vacancy.min_years_relevant_experience:g} tahun)."
        )

    if vacancy.requires_regulatory_compliance:
        if extracted.has_building_codes_evidence:
            strengths.append("Terdapat bukti eksplisit kepatuhan terhadap kode bangunan, standar keselamatan, atau regulasi lokal.")
        else:
            gaps.append("Belum terlihat bukti eksplisit penguasaan building codes dan standar regulasi bangunan di CV.")
    else:
        if extracted.has_building_codes_evidence:
            strengths.append("Terdapat bukti penerapan pengujian, validasi, standar kualitas, atau keamanan data pada proyek.")
        else:
            gaps.append("Belum terlihat bukti eksplisit terkait pengujian (testing/validasi) atau standar kualitas produksi di CV.")

    if extracted.has_design_and_build_evidence:
        if vacancy.domain == "architecture":
            strengths.append("Memiliki pengalaman nyata lintas tahapan design & build (survei lokasi, gambar kerja/shop drawing, dan pengawasan proyek).")
        else:
            strengths.append(f"Memiliki bukti implementasi end-to-end dan delivery proyek yang relevan dengan {vacancy.title}.")
    else:
        if vacancy.domain == "architecture":
            gaps.append("Pengalaman proyek masih dominan pada tahap desain/visualisasi, minim bukti pengawasan eksekusi konstruksi end-to-end.")
        else:
            gaps.append(f"Belum terlihat bukti implementasi proyek end-to-end yang langsung relevan dengan ruang lingkup {vacancy.title}.")

    if extracted.timeline.claim_discrepancy_flag:
        gaps.append(extracted.timeline.claim_discrepancy_note)

    for sig in extracted.availability_signals:
        gaps.append(sig)

    # Susun pertanyaan wawancara spesifik berbasis fakta CV & domain lowongan
    if extracted.timeline.has_overlapping_roles:
        questions.append(
            "Terdapat beberapa peran kerja dengan rentang waktu bersamaan di CV Anda. Bagaimana pembagian peran penuh waktu vs freelance/proyek pada periode tersebut?"
        )
    if extracted.timeline.claim_discrepancy_flag:
        questions.append(
            f"Di ringkasan CV tertulis pengalaman beberapa tahun lebih panjang dibanding riwayat yang tercantum. Bisa jelaskan proyek {domain_label} yang dikerjakan pada periode yang belum tertulis?"
        )
    if any(e.is_expected_graduation and e.status == "Completed" for e in extracted.education):
        questions.append(
            "Pada CV tercantum perkiraan tanggal kelulusan sarjana. Apakah ijazah S1 sudah resmi diterbitkan dan siap dilampirkan saat verifikasi?"
        )
    if extracted.currently_studying:
        questions.append(
            "Saat ini Anda masih tercatat menempuh pendidikan aktif. Bagaimana pengaturan jadwal kuliah apabila diterima untuk bergabung secara penuh waktu?"
        )
    if extracted.currently_employed:
        questions.append(
            "Mengingat posisi saat ini masih aktif bekerja penuh waktu, berapa lama masa notice period Anda dan kapan tanggal paling cepat dapat bergabung?"
        )
    if missing_sw:
        questions.append(
            f"Lowongan {vacancy.title} membutuhkan {', '.join(missing_sw)}. Sejauh mana familiaritas Anda dan berapa lama waktu adaptasi yang dibutuhkan?"
        )
    if not extracted.has_supplier_or_material_evidence or len(questions) < 3:
        if vacancy.domain == "architecture":
            questions.append(
                "Ceritakan pengalaman Anda berkoordinasi dengan supplier material interior, menyusun timeline penyelesaian proyek, serta menangani revisi di lapangan."
            )
        elif vacancy.domain == "ai_data":
            questions.append(
                "Ceritakan satu proyek AI/ML atau LLM end-to-end yang paling menantang: mulai dari pengolahan data, evaluasi model, hingga integrasi API atau deployment ke pengguna."
            )
        else:
            questions.append(
                f"Ceritakan satu proyek end-to-end yang paling relevan dengan posisi {vacancy.title}, termasuk tantangan teknis dan kolaborasi lintas tim yang Anda tangani."
            )

    # Ringkasan keputusan objektif
    if status == "Shortlisted (Wawancara)":
        sw_desc = f"menguasai stack utama ({', '.join(matched_sw)})" if matched_sw else "memenuhi kompetensi teknis yang diminta"
        exp_desc = (
            f"dengan durasi relevan bersih {rel_years:.1f} tahun"
            if rel_years > 0
            else "dengan bukti portofolio proyek relevan (sesuai kriteria Fresh Graduate)"
        )
        rationale = (
            f"Direkomendasikan untuk tahap wawancara posisi {vacancy.title} karena memenuhi gelar yang disyaratkan, "
            f"{sw_desc}, serta memiliki bukti pengalaman nyata di {domain_label} {exp_desc}."
        )
    elif status == "Pertimbangan Khusus (KIV)":
        gap_hint = ", ".join(missing_sw) if missing_sw else "kedalaman pengalaman end-to-end"
        rationale = (
            f"Masuk kategori pertimbangan khusus untuk posisi {vacancy.title} karena memiliki dasar pendidikan dan latar relevan "
            f"({rel_years:.1f} tahun bersih), namun belum memenuhi seluruh kriteria utama (kekurangan: {gap_hint})."
        )
    else:
        reasons_short = []
        if not degree_met and edu_gap_reason:
            reasons_short.append(edu_gap_reason)
        if missing_sw:
            reasons_short.append(f"belum menguasai {', '.join(missing_sw)}")
        if vacancy.min_years_relevant_experience > 0 and rel_years < vacancy.min_years_relevant_experience:
            reasons_short.append(f"pengalaman relevan {domain_label} baru {rel_years:.1f} tahun")
        elif vacancy.min_years_relevant_experience == 0 and rel_years == 0 and not extracted.has_design_and_build_evidence:
            reasons_short.append(f"belum memiliki pengalaman maupun proyek relevan di {domain_label}")
        if not reasons_short:
            reasons_short.append("skor kecocokan keseluruhan berada di bawah ambang minimum rubrik")
        rationale = (
            f"Belum dapat dilanjutkan ke tahap wawancara untuk posisi {vacancy.title} karena "
            + ", ".join(reasons_short)
            + "."
        )

    return rationale, strengths[:4], gaps[:4], questions[:3]


def evaluate_candidates(
    vacancy: VacancyCriteria,
    extracted_list: List[ExtractedCV],
    pii_audits: Dict[str, PIIAuditResult],
    weights: ScoringWeights,
    engine_mode: str = "local_hybrid",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> List[CandidateEvaluation]:
    """Menjalankan penilaian Hybrid (Rule-Based + TF-IDF + Confidence Gate + opsional DeepSeek) lalu meranking kandidat."""
    semantic_scores = compute_batch_semantic_scores(vacancy, extracted_list)
    norm_weights = weights.normalized()

    evaluations: List[CandidateEvaluation] = []
    for ext in extracted_list:
        sem_score = semantic_scores.get(ext.candidate_id, 0.0)
        edu_score, degree_met, edu_note, edu_gap_reason = _score_education(ext, vacancy)
        exp_score, exp_met = _score_relevant_experience(ext, vacancy)
        sw_score, matched_sw, missing_sw = _score_software_technical(ext, vacancy)
        resp_score = _score_responsibility_fit(ext, sem_score, vacancy)
        reg_score = _score_regulatory_readiness(ext, vacancy)

        dim_scores = DimensionScores(
            relevant_experience=exp_score,
            software_technical=sw_score,
            education=edu_score,
            responsibility_fit=resp_score,
            regulatory_readiness=reg_score,
        )

        overall = round(
            dim_scores.relevant_experience * norm_weights["relevant_experience"]
            + dim_scores.software_technical * norm_weights["software_technical"]
            + dim_scores.education * norm_weights["education"]
            + dim_scores.responsibility_fit * norm_weights["responsibility_fit"]
            + dim_scores.regulatory_readiness * norm_weights["regulatory_readiness"],
            1,
        )

        needs_review, review_reasons = _check_manual_review_gate(ext)
        status = _determine_status(
            overall,
            degree_met,
            matched_sw,
            missing_sw,
            exp_met,
            needs_manual_review=needs_review,
        )

        # Kumpulkan flag anomali dari timeline, pendidikan & ekstraksi
        anomaly_flags: List[str] = []
        for warn in ext.extraction_warnings:
            anomaly_flags.append(warn)
        if ext.timeline.has_overlapping_roles:
            anomaly_flags.append(
                f"Overlap tanggal dideduplikasi: mentah {ext.timeline.raw_sum_months_all} bln menjadi bersih {ext.timeline.merged_months_all} bln."
            )
        if ext.timeline.claim_discrepancy_flag:
            anomaly_flags.append(ext.timeline.claim_discrepancy_note)
        if any(e.is_expected_graduation and e.status == "Completed" for e in ext.education):
            anomaly_flags.append(
                "Gelar pendidikan tercantum dengan keterangan perkiraan kelulusan (expected) - perlu verifikasi ijazah resmi."
            )
        if (
            ext.timeline.post_graduation_years_relevant is not None
            and (ext.timeline.merged_years_relevant - ext.timeline.post_graduation_years_relevant) >= 1.0
        ):
            anomaly_flags.append(
                f"Sebagian pengalaman relevan dijalani sebelum/paralel kuliah "
                f"(total bersih {ext.timeline.merged_years_relevant:.1f} thn; pasca-lulus {ext.timeline.post_graduation_years_relevant:.1f} thn)."
            )
        if ext.organizational_experiences_count > 0:
            anomaly_flags.append(
                f"{ext.organizational_experiences_count} kegiatan organisasi/kepanitiaan dipisahkan dari perhitungan pengalaman kerja."
            )
        non_rel_count = sum(1 for w in ext.work_experiences if not w.is_relevant_domain)
        if non_rel_count > 0:
            anomaly_flags.append(
                f"{non_rel_count} riwayat kerja di luar domain posisi {vacancy.title} tidak dihitung sebagai pengalaman relevan."
            )

        rationale, strengths, gaps, questions = _build_local_explanations(
            extracted=ext,
            vacancy=vacancy,
            dim_scores=dim_scores,
            matched_sw=matched_sw,
            missing_sw=missing_sw,
            degree_met=degree_met,
            edu_note=edu_note,
            edu_gap_reason=edu_gap_reason,
            status=status,
            manual_review_reasons=review_reasons,
        )

        # Jika mode DeepSeek aktif & kandidat tidak masuk review manual karena gagal baca, perkaya narasi dengan LLM
        if engine_mode == "deepseek" and not needs_review and not ext.fallback_used:
            try:
                enriched, extra_toks, extra_cost = enrich_evaluation_with_deepseek(
                    extracted=ext,
                    vacancy=vacancy,
                    dimension_scores=dim_scores,
                    semantic_score=sem_score,
                    matched_sw=matched_sw,
                    missing_sw=missing_sw,
                    degree_met=degree_met,
                    status=status,
                    api_key=api_key,
                    model=model,
                )
                ext.llm_tokens_used += extra_toks
                ext.estimated_cost_usd = round(ext.estimated_cost_usd + extra_cost, 6)
                rationale = str(enriched.get("decision_rationale") or rationale)
                if enriched.get("strengths"):
                    strengths = [str(s) for s in enriched["strengths"]][:4]
                if enriched.get("gaps_and_risks"):
                    gaps = [str(g) for g in enriched["gaps_and_risks"]][:4]
                if enriched.get("interview_questions"):
                    questions = [str(q) for q in enriched["interview_questions"]][:3]
            except Exception:
                # Tetap gunakan penjelasan lokal yang rapi apabila pemanggilan narasi gagal
                pass

        audit = pii_audits[ext.candidate_id]
        evaluations.append(
            CandidateEvaluation(
                rank=0,
                candidate_id=ext.candidate_id,
                candidate_name=ext.detected_name,
                source_filename=ext.source_filename,
                source_channel=ext.source_channel,
                overall_score=overall,
                semantic_similarity_score=sem_score,
                status=status,
                needs_manual_review=needs_review,
                manual_review_reasons=review_reasons,
                dimension_scores=dim_scores,
                matched_required_software=matched_sw,
                missing_required_software=missing_sw,
                degree_requirement_met=degree_met,
                experience_requirement_met=exp_met,
                strengths=strengths,
                gaps_and_risks=gaps,
                decision_rationale=rationale,
                interview_questions=questions,
                anomaly_flags=anomaly_flags,
                pii_redactions_count=audit.total_redactions,
                pii_redacted_categories=[item.category for item in audit.redacted_items],
                extracted_cv=ext,
                pii_audit=audit,
            )
        )

    # Urutkan kandidat: prioritas status (Shortlisted > KIV > Review Manual > Tidak Lolos), lalu skor tertinggi
    status_priority = {
        "Shortlisted (Wawancara)": 0,
        "Pertimbangan Khusus (KIV)": 1,
        "Perlu Review Manual": 2,
        "Tidak Lolos Kualifikasi": 3,
    }
    evaluations.sort(
        key=lambda ev: (
            status_priority.get(ev.status, 9),
            -ev.overall_score,
            -ev.semantic_similarity_score,
        )
    )
    for idx, ev in enumerate(evaluations, start=1):
        ev.rank = idx

    return evaluations
