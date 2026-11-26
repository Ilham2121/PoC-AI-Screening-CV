"""Ekstraktor CV terstruktur berbasis aturan NLP lokal (mode offline / hybrid fallback)."""

from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Optional, Tuple

from src.schemas import EducationEntry, ExtractedCV, VacancyCriteria, WorkExperienceEntry
from src.skill_normalizer import extract_known_skills_from_text, normalize_skills
from src.timeline import MONTH_MAP, calculate_timeline_summary, infer_bachelor_graduation_ym, parse_month_year

MONTH_REGEX = (
    r"(?:Jan(?:uary|uari)?|Feb(?:ruary|ruari)?|Mar(?:ch|et)?|Apr(?:il)?|"
    r"May|Mei|Jun(?:e|i)?|Jul(?:y|i)?|Aug(?:ust)?|Agu(?:stus)?|Ags|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Okt(?:ober)?|Nov(?:ember)?|Nop(?:ember)?|Dec(?:ember)?|Des(?:ember)?)"
)

DATE_TOKEN_REGEX = rf"(?:{MONTH_REGEX}\.?,?\s+\d{{4}}|\d{{4}}-\d{{1,2}}|\d{{1,2}}[/.]\d{{4}}|\d{{4}})"

DATE_RANGE_REGEX = re.compile(
    rf"\(?({DATE_TOKEN_REGEX})\s*[\-\u2013\u2014]\s*"
    rf"({DATE_TOKEN_REGEX}|Present|Saat ini|Sekarang|Current|Ongoing|Kini|Today|Now)\)?",
    flags=re.IGNORECASE,
)

SECTION_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "summary",
        re.compile(
            r"^(?:ringkasan|tentang saya|about me|profile|profil|summary|professional summary|career summary|objective)$",
            re.I,
        ),
    ),
    (
        "experience",
        re.compile(
            r"^(?:riwayat karier|riwayat pekerjaan|pengalaman kerja|pengalaman profesional|"
            r"experiences?|work experiences?|work history|employment history|employment|"
            r"professional experiences?|career history)$",
            re.I,
        ),
    ),
    (
        "organization",
        re.compile(
            r"^(?:organi[sz]ational experiences?|pengalaman organisasi|aktivitas organisasi|"
            r"extracurricular|volunteering|volunteer experience)$",
            re.I,
        ),
    ),
    (
        "education",
        re.compile(
            r"^(?:pendidikan|riwayat pendidikan|education|education level|educational background|"
            r"formal education|academic background)$",
            re.I,
        ),
    ),
    (
        "skills",
        re.compile(
            r"^(?:keahlian|keterampilan|kemampuan|skills?|soft and hard skills|hard skills?|"
            r"area of expertise|technical skills?|software|software skills?|core competencies|kompetensi)$",
            re.I,
        ),
    ),
    (
        "awards",
        re.compile(
            r"^(?:awards?|achievements?|key achievements?|penghargaan|prestasi|certifications?|sertifikasi)$",
            re.I,
        ),
    ),
    (
        "projects",
        re.compile(
            r"^(?:projects|selected projects?|key projects?|technical projects?|personal projects?|proyek|portofolio proyek)$",
            re.I,
        ),
    ),
    ("additional", re.compile(r"^(?:bahasa|languages?|additional information|informasi tambahan)$", re.I)),
]


def _normalize_spaced_heading(line: str) -> str:
    """Mengubah judul huruf berjarak seperti 'W O R K E X P E R I E N C E' menjadi teks normal."""
    cleaned = line.strip().strip(":").strip("•-* ")
    if re.match(r"^(?:[A-Z]\s+){3,}[A-Z](?:\s+(?:[A-Z]\s+)*[A-Z])?$", cleaned):
        # Jika ada pemisah >= 2 spasi antar kata (mis. 'W O R K   E X P E R I E N C E'), gabung per kata secara dinamis
        if re.search(r"\s{2,}", cleaned):
            words = [re.sub(r"\s+", "", w) for w in re.split(r"\s{2,}", cleaned)]
            return " ".join(words)
        # Fallback jika spasi antar kata sudah terkompresi menjadi 1 spasi oleh PDF parser
        compact = re.sub(r"\s+", "", cleaned).lower()
        mapping = {
            "profile": "PROFILE",
            "workexperience": "WORK EXPERIENCE",
            "workhistory": "WORK HISTORY",
            "pengalamankerja": "PENGALAMAN KERJA",
            "education": "EDUCATION",
            "educationsoftandhardskills": "EDUCATION",
            "softandhardskills": "SOFT AND HARD SKILLS",
            "achievements": "ACHIEVEMENTS",
            "skills": "SKILLS",
            "architecturaldrafter": "ARCHITECTURAL DRAFTER",
        }
        return mapping.get(compact, compact.upper())
    return cleaned


def _split_into_sections(text: str) -> Dict[str, List[str]]:
    """Memecah teks CV menjadi beberapa seksi utama berdasarkan heading."""
    sections: Dict[str, List[str]] = {
        "header": [],
        "summary": [],
        "experience": [],
        "projects": [],
        "organization": [],
        "education": [],
        "skills": [],
        "awards": [],
        "additional": [],
    }
    current = "header"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        norm_heading = _normalize_spaced_heading(line)
        matched_section: Optional[str] = None
        for sec_name, pat in SECTION_PATTERNS:
            if pat.match(norm_heading):
                matched_section = sec_name
                break
        if matched_section:
            current = matched_section
            continue

        # Tangani kasus JobStreet / CV ringkas di mana heading berada di awal baris yang sama
        inline_handled = False
        for prefix, sec_name in [
            ("Ringkasan ", "summary"),
            ("Ringkasan:", "summary"),
            ("Riwayat karier ", "experience"),
            ("Pengalaman Kerja ", "experience"),
            ("Pengalaman Kerja:", "experience"),
            ("Keahlian ", "skills"),
            ("Keahlian:", "skills"),
            ("Software:", "skills"),
            ("Skills:", "skills"),
            ("Pendidikan ", "education"),
            ("Pendidikan:", "education"),
            ("Bahasa ", "additional"),
            ("Bahasa:", "additional"),
        ]:
            if line.lower().startswith(prefix.lower()):
                current = sec_name
                rest = line[len(prefix) :].strip()
                if rest:
                    sections[current].append(rest)
                inline_handled = True
                break
        if not inline_handled:
            sections[current].append(line)

    # Jika tidak ada heading summary eksplisit, ambil paragraf awal di header (panjang > 80 karakter)
    if not sections["summary"]:
        for line in sections["header"]:
            if len(line) >= 80 and "@" not in line and "[EMAIL_REDACTED]" not in line:
                sections["summary"].append(line)

    return sections


def _format_iso_ym(raw_date: str, reference_date: date) -> Optional[str]:
    low = raw_date.strip().lower()
    if low in {"present", "saat ini", "sekarang", "current", "ongoing", "kini", "today", "now"}:
        return "Present"
    ym = parse_month_year(raw_date, reference_date)
    if not ym:
        return None
    return f"{ym[0]:04d}-{ym[1]:02d}"


def _classify_experience_relevance(
    role: str,
    company: str,
    highlights: List[str],
    vacancy: VacancyCriteria,
) -> Tuple[bool, str, str]:
    """Menilai apakah suatu pengalaman kerja relevan dengan domain & kriteria lowongan target."""
    combined = f"{role} {company} {' '.join(highlights)}".lower()

    # Tentukan tipe pekerjaan
    if any(k in combined for k in ["intern", "magang"]):
        emp_type = "Intern"
    elif any(k in combined for k in ["freelance", "independet", "independent"]):
        emp_type = "Freelance"
    else:
        emp_type = "Full-time"

    if vacancy.domain == "architecture":
        # Deteksi pekerjaan non-arsitektur / pengujian struktur sipil murni
        structural_testing_signals = [
            "testing technical",
            "concrete integrity",
            "strain gauge",
            "accelerometer",
            "loading test",
            "bridge test equipment",
            "structural health monitoring",
            "sensor placement",
        ]
        if any(sig in combined for sig in structural_testing_signals):
            return (
                False,
                "Fokus utama pada pengujian struktur sipil/beton dan alat uji, bukan perancangan arsitektur atau interior design & build.",
                emp_type,
            )

        arch_positive_signals = [
            "architect", "arsitek", "interior", "drafter", "kontraktor", "contractor",
            "quality control", "working drawing", "shop drawing", "gambar kerja",
            "3d model", "conceptual design", "konsep desain", "desain bangunan",
            "desain rumah", "residential", "building", "site survey", "survei lokasi",
            "site supervision", "pengawasan proyek", "renovation", "renovasi", "desain dan bangun",
        ]
        if any(sig in combined for sig in arch_positive_signals):
            if "virtual tour" in combined and "architect" not in role.lower() and "interior" not in role.lower():
                return (
                    True,
                    "Relevan sebagian (fokus pada drafting teknis dan visualisasi 3D pemasaran properti, bukan perancangan konseptual penuh).",
                    emp_type,
                )
            return (
                True,
                "Relevan dengan domain arsitektur, desain interior, drafting teknis, atau pelaksanaan konstruksi.",
                emp_type,
            )
        return (False, "Tidak berkaitan langsung dengan bidang arsitektur atau interior design & build.", emp_type)

    if vacancy.domain == "ai_data":
        ai_signals = [
            "ai ", "artificial intelligence", "machine learning", "deep learning", "data scien",
            "data analy", "data engineer", "llm", "genai", "generative ai", "rag", "langchain",
            "nlp", "computer vision", "pytorch", "tensorflow", "scikit", "pandas", "numpy",
            "python", "fastapi", "flask", "mlops", "feature engineering", "model training",
            "software engineer", "backend", "automation", "otomasi", "api ", "sql", "analytics",
            "research assistant", "asisten praktikum", "asisten laboratorium",
        ]
        if any(sig in combined for sig in ai_signals):
            return (
                True,
                f"Relevan dengan domain AI/ML, pengolahan data, atau rekayasa perangkat lunak untuk posisi {vacancy.title}.",
                emp_type,
            )
        return (
            False,
            f"Tidak berkaitan langsung dengan bidang AI/ML, data, atau rekayasa perangkat lunak ({vacancy.title}).",
            emp_type,
        )

    if vacancy.domain == "software_engineering":
        swe_signals = [
            "software", "developer", "engineer", "backend", "frontend", "fullstack", "full-stack",
            "devops", "cloud", "api", "python", "javascript", "typescript", "react", "node",
            "java", "golang", "docker", "kubernetes", "database", "sql", "programmer", "web ",
        ]
        if any(sig in combined for sig in swe_signals):
            return (
                True,
                f"Relevan dengan pengembangan perangkat lunak dan sistem teknis untuk posisi {vacancy.title}.",
                emp_type,
            )
        return (
            False,
            f"Tidak berkaitan langsung dengan rekayasa perangkat lunak untuk posisi {vacancy.title}.",
            emp_type,
        )

    # Domain general: cek kecocokan kata kunci terhadap judul, software wajib, dan tanggung jawab lowongan
    vac_tokens = {
        tok
        for tok in re.findall(r"[a-z0-9+#]{3,}", f"{vacancy.title} {' '.join(vacancy.required_software)} {' '.join(vacancy.preferred_skills)}".lower())
        if tok not in {"and", "the", "for", "with", "junior", "senior", "staff", "intern", "specialist", "officer"}
    }
    if any(tok in combined for tok in vac_tokens):
        return (True, f"Relevan dengan kompetensi dan ruang lingkup posisi {vacancy.title}.", emp_type)
    return (False, f"Tidak ditemukan keterkaitan langsung dengan posisi {vacancy.title}.", emp_type)


def _extract_work_experiences(
    exp_lines: List[str],
    reference_date: date,
    vacancy: VacancyCriteria,
) -> List[WorkExperienceEntry]:
    """Mengekstrak daftar pengalaman kerja dari seksi experience beserta tanggal dan highlight."""
    experiences: List[WorkExperienceEntry] = []
    if not exp_lines:
        return experiences

    # Temukan indeks baris yang memuat rentang tanggal (penanda entri pekerjaan baru)
    date_indices: List[Tuple[int, re.Match[str]]] = []
    for idx, line in enumerate(exp_lines):
        # Hindari tahun proyek di dalam bullet seperti "(2022)" yang bukan rentang tanggal
        match = DATE_RANGE_REGEX.search(line)
        if match:
            date_indices.append((idx, match))

    for pos, (line_idx, match) in enumerate(date_indices):
        start_raw, end_raw = match.group(1), match.group(2)
        start_iso = _format_iso_ym(start_raw, reference_date)
        end_iso = _format_iso_ym(end_raw, reference_date)

        # Tentukan batas blok baris untuk pekerjaan ini
        prev_bound = date_indices[pos - 1][0] + 1 if pos > 0 else 0
        next_date_idx = date_indices[pos + 1][0] if pos + 1 < len(date_indices) else len(exp_lines)

        # Ambil header role/company dari baris yang sama (sebelum tanggal) atau 1-2 baris tepat sebelum tanggal
        line_before_date = exp_lines[line_idx][: match.start()].strip(" -|•()")
        header_candidates: List[str] = []

        # Cek 1-2 baris sebelumnya jika bukan bullet deskripsi panjang
        lookback = []
        for k in range(max(prev_bound, line_idx - 2), line_idx):
            cand = exp_lines[k].strip()
            if (
                len(cand) <= 75
                and not cand.startswith(("•", "-", "*"))
                and not cand.endswith(".")
                and "PROJECT" not in cand.upper()
            ):
                lookback.append(cand)
        header_candidates.extend(lookback)
        if line_before_date:
            header_candidates.append(line_before_date)

        # Cek juga 1-2 baris tepat sesudah tanggal jika itu adalah jabatan/perusahaan (contoh: CV Indri & Dani)
        after_start = line_idx + 1
        post_headers: List[str] = []
        while after_start < next_date_idx and len(post_headers) < 2:
            nxt = exp_lines[after_start].strip()
            if (
                len(nxt) <= 65
                and not nxt.startswith(("•", "-", "*"))
                and not nxt.endswith(".")
                and not nxt.upper().startswith("PROJECT")
                and not any(
                    v in nxt.lower()
                    for v in [
                        "designed ",
                        "studied ",
                        "developed ",
                        "conducting ",
                        "creating ",
                        "produced ",
                        "drawing ",
                        "planning ",
                        "collaborat",
                        "worked as",
                        "led ",
                        "coordinated ",
                        "membuat ",
                        "merancang ",
                        "mengawasi ",
                        "mengelola ",
                        "mendesain ",
                        "bertanggung jawab",
                    ]
                )
            ):
                post_headers.append(nxt)
                after_start += 1
            else:
                break

        # Susun role & company dari kandidat header
        role = ""
        company = ""
        all_hdrs = header_candidates + post_headers
        company_markers = ["pt.", "pt ", "cv.", "cv ", "studio", "center", "firm", "corp", "inc", "ltd", "konsultan"]
        for h in all_hdrs:
            if " di " in h:
                parts = h.split(" di ", 1)
                role = parts[0].strip()
                company = parts[1].strip()
                break
            if " - " in h:
                parts = [p.strip() for p in h.split(" - ", 1)]
                if parts[0] and parts[1] and "[ADDRESS_REDACTED]" not in parts[1]:
                    if any(k in parts[0].lower() for k in company_markers):
                        company, role = parts[0], parts[1]
                    else:
                        role, company = parts[0], parts[1]
                    break
            if "," in h and any(k in h.lower() for k in company_markers + ["freelance"]) and "developer" not in h.lower():
                parts = [p.strip() for p in h.split(",", 1)]
                if any(k in parts[0].lower() for k in company_markers):
                    company, role = parts[0], parts[1]
                else:
                    role, company = parts[0], parts[1]
                break

        if not role:
            for h in all_hdrs:
                h_clean = re.sub(
                    r"\b(?:Full Time|Part Time|Independet|Independent)\b|\[ADDRESS_REDACTED\]",
                    "",
                    h,
                    flags=re.I,
                ).strip(" -|,")
                if not h_clean:
                    continue
                if any(k in h_clean.lower() for k in ["pt.", "pt ", "cv ", "studio", "center", "firm", "developer", "integrated port"]):
                    if not company:
                        company = h_clean
                elif not role:
                    role = h_clean
                elif not company:
                    company = h_clean

        if not role and company:
            role = company
            company = ""
        role = role or "Professional Role"

        # Kumpulkan deskripsi / highlights sampai sebelum blok header pekerjaan berikutnya
        end_body = next_date_idx
        if pos + 1 < len(date_indices):
            # Sisakan 1-2 baris header milik entri berikutnya
            for back in range(1, 3):
                cand_idx = next_date_idx - back
                if cand_idx <= after_start:
                    break
                cand_txt = exp_lines[cand_idx].strip()
                if (
                    len(cand_txt) <= 70
                    and not cand_txt.startswith(("•", "-", "*"))
                    and not cand_txt.endswith(".")
                    and "PROJECT" not in cand_txt.upper()
                ):
                    end_body = cand_idx
                else:
                    break

        highlights = [
            exp_lines[i].lstrip("•-* ").strip()
            for i in range(after_start, end_body)
            if len(exp_lines[i].strip()) > 10 and not exp_lines[i].strip().upper().startswith("PROJECT")
        ]

        is_rel, rel_reason, emp_type = _classify_experience_relevance(role, company, highlights, vacancy)
        experiences.append(
            WorkExperienceEntry(
                role=role,
                company=company,
                employment_type=emp_type,
                start_date=start_iso,
                end_date=end_iso,
                is_relevant_domain=is_rel,
                relevance_reason=rel_reason,
                highlights=highlights[:12],
            )
        )

    return experiences


def _is_education_field_relevant(field_text: str, vacancy: VacancyCriteria) -> bool:
    """Mengevaluasi apakah bidang pendidikan kandidat relevan dengan kriteria lowongan."""
    low = field_text.lower()
    if not vacancy.required_degree_fields:
        return True

    field_equivalents: Dict[str, List[str]] = {
        "architecture": ["architect", "arsitek", "s.ars", "sars", "m.ars", "building drawing", "building information"],
        "interior design": ["interior", "desain interior"],
        "architectural engineering": ["architectural engineering", "teknik arsitektur", "arsitek"],
        "civil engineering": ["civil", "sipil"],
        "information technology": [
            "information technology", "teknologi informasi", "informatika", "informatics",
            "teknik informatika", "sistem informasi", "information system", "s.kom", "s.tr.kom",
            "komputer", "computer", "software", "rekayasa perangkat lunak", "teknik elektro",
        ],
        "computer science": [
            "computer science", "ilmu komputer", "informatika", "informatics", "teknik informatika",
            "information technology", "teknologi informasi", "sistem informasi", "s.kom",
            "software engineering", "rekayasa perangkat lunak", "artificial intelligence", "kecerdasan buatan",
        ],
        "data science": [
            "data science", "sains data", "statistika", "statistics", "matematika", "mathematics",
            "informatika", "computer science", "ilmu komputer", "information technology", "s.si", "s.kom",
        ],
        "information systems": ["sistem informasi", "information system", "teknologi informasi", "informatika", "s.kom"],
        "software engineering": ["software", "rekayasa perangkat lunak", "informatika", "computer science", "s.kom"],
        "statistics": ["statistika", "statistics", "matematika", "data science"],
        "mathematics": ["matematika", "mathematics", "statistika", "data science"],
        "electrical engineering": ["electrical", "teknik elektro", "elektronika", "komputer"],
        "industrial engineering": ["industrial", "teknik industri"],
        "mechanical engineering": ["mechanical", "teknik mesin"],
        "economics & finance": ["ekonomi", "economics", "finance", "keuangan", "akuntansi", "accounting"],
        "management & business": ["manajemen", "management", "bisnis", "business"],
    }

    for req_field in vacancy.required_degree_fields:
        req_low = req_field.lower()
        if req_low in low:
            return True
        for eq in field_equivalents.get(req_low, []):
            if eq in low:
                return True
    return False


def _extract_education(
    edu_lines: List[str],
    full_text: str,
    reference_date: date,
    vacancy: VacancyCriteria,
) -> List[EducationEntry]:
    """Mengekstrak tingkat pendidikan, jurusan, dan status kelulusan (Completed vs In Progress)."""
    entries: List[EducationEntry] = []
    source_lines = edu_lines if edu_lines else full_text.splitlines()
    joined = "\n".join(source_lines)

    # Pola gelar lintas domain (Inggris & Indonesia, dengan batas kata ketat agar tidak salah cocok di dalam kata)
    degree_patterns = [
        (
            "Master",
            r"(?:\bMaster(?:'?s|s)?\s+(?:of|in)\s+[A-Za-z][A-Za-z &]{2,45}|"
            r"\bMagister\s+(?:Teknik\s+)?[A-Za-z][A-Za-z &]{2,40}|"
            r"\bM\.(?:Ars|Kom|Sc|Eng|T|Si)\.?(?=\s|,|$)|"
            r"\bS2\s+(?:Teknik\s+)?[A-Za-z][A-Za-z &]{2,40})",
        ),
        (
            "Bachelor",
            r"(?:\bBachelor(?:'?s|s)?\s+(?:of|in)\s+[A-Za-z][A-Za-z &]{2,45}(?:\s*\([A-Za-z.]+\))?|"
            r"\bB\.(?:Sc|Eng|Arch|Tech|IT|CS)\.?(?:\s+(?:in\s+)?[A-Za-z][A-Za-z &]{2,35})?(?=\s|,|$)|"
            r"\bSarjana\s+(?:Teknik\s+|Komputer\s+|Sains\s+)?[A-Za-z][A-Za-z &]{2,40}(?:\s*\([^)]+\))?|"
            r"\bS\.(?:Ars|Kom|Tr\.?Kom|Si|T)\.?(?:\s+[A-Za-z][A-Za-z &]{2,35})?(?=\s|,|$)|"
            r"\bS1\s+(?:Teknik\s+)?[A-Za-z][A-Za-z &]{2,40}|"
            r"\b(?:Department|Faculty|Major)\s+(?:of|in)\s+(?:Architectural\s+Engineering|Informatics|Computer\s+Science|Information\s+Technology|Data\s+Science|Information\s+Systems|Software\s+Engineering))",
        ),
        (
            "Diploma",
            r"(?:\bDiploma\s+(?:III|IV|3|4)?\s*(?:of\s+|in\s+)?[A-Za-z][A-Za-z &]{2,35}|"
            r"\bD[34]\s+(?:Teknik\s+)?[A-Za-z][A-Za-z &]{2,35}|"
            r"\bAhli\s+Madya\s+[A-Za-z][A-Za-z &]{2,35})",
        ),
        (
            "SMK/SMA",
            r"(?:\bState\s+Vocational\s+(?:High\s+)?School[^\n]*|\bVocational\s+(?:High\s+)?School[^\n]*|"
            r"\bBuilding\s+Drawing\s+Techniques|\bSMKN?\s*\d+[^\n]*)",
        ),
    ]

    for deg_level, pat in degree_patterns:
        for match in re.finditer(pat, joined, flags=re.IGNORECASE):
            # Ambil jendela konteks sekitar entri pendidikan (sebelum & sesudah ~140 karakter)
            ctx_start = max(0, match.start() - 120)
            ctx_end = min(len(joined), match.end() + 140)
            context = joined[ctx_start:ctx_end]

            # Cek apakah masih berkuliah / belum lulus
            is_in_progress = bool(
                re.search(
                    r"(?:20\d{2}\s*[\-\u2013]\s*(?:Present|Saat ini|Sekarang|Expected|Kini))",
                    context,
                    flags=re.I,
                )
                or re.search(r"\bExpected\b", context, flags=re.I)
            )

            # Cek kasus "Perkiraan tanggal selesai Okt 2023" terhadap reference_date
            est_match = re.search(
                rf"(?:perkiraan\s+tanggal\s+selesai|expected\s+graduation)\s*:?\s*({MONTH_REGEX}\.?\s+\d{{4}})",
                context,
                flags=re.I,
            )
            is_expected_grad = bool(est_match or re.search(r"\bExpected\b", context, flags=re.I))

            start_year: Optional[int] = None
            end_year: Optional[int] = None

            years_found = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", context)]
            if years_found:
                start_year = min(years_found)
                end_year = max(years_found)

            if est_match:
                est_ym = parse_month_year(est_match.group(1), reference_date)
                if est_ym:
                    end_year = est_ym[0]
                    if (est_ym[0], est_ym[1]) <= (reference_date.year, reference_date.month):
                        is_in_progress = False
                    else:
                        is_in_progress = True

            # Cek pola eksplisit "lulus 2020" / "graduated 2020"
            grad_match = re.search(
                r"\b(?:lulus|graduated?)\s*(?:tahun\s*)?(19\d{2}|20\d{2})\b",
                context,
                flags=re.I,
            )
            if grad_match:
                end_year = int(grad_match.group(1))
                is_in_progress = False

            # Cek khusus jika ada "Aug 2025 (Expected)" atau rentang tahun vs reference_date
            range_m = DATE_RANGE_REGEX.search(context)
            if range_m:
                s_ym = parse_month_year(range_m.group(1), reference_date)
                e_ym = parse_month_year(range_m.group(2), reference_date)
                if s_ym:
                    start_year = s_ym[0]
                if e_ym and range_m.group(2).lower() not in {"present", "saat ini", "sekarang", "kini"}:
                    end_year = e_ym[0]
                    if (e_ym[0], e_ym[1]) > (reference_date.year, reference_date.month):
                        is_in_progress = True
                elif range_m.group(2).lower() in {"present", "saat ini", "sekarang", "kini"}:
                    is_in_progress = True

            # Cek juga paragraf summary untuk konfirmasi sedang kuliah S2 ("currently pursuing a Master's degree")
            if deg_level == "Master" and re.search(r"currently\s+pursuing\s+a\s+master", full_text, flags=re.I):
                is_in_progress = True

            field_text = re.sub(r"\s+", " ", match.group(0)).strip(" ,.-|")
            field_text = re.split(r"\s+(?:dari|from|at|di)\s+", field_text, maxsplit=1, flags=re.I)[0].strip(" ,.-|")
            is_relevant = _is_education_field_relevant(field_text, vacancy)
            status = "In Progress" if is_in_progress else "Completed"

            # Hindari duplikasi entri dengan level & tahun yang sama
            if not any(e.degree_level == deg_level and e.start_year == start_year for e in entries):
                entries.append(
                    EducationEntry(
                        degree_level=deg_level,
                        field_of_study=field_text,
                        institution_redacted="[INSTITUTION_REDACTED]",
                        status=status,
                        start_year=start_year,
                        end_year=end_year,
                        is_relevant_field=is_relevant,
                        is_expected_graduation=is_expected_grad,
                    )
                )

    # Jika kandidat sudah menempuh S1/S2/Diploma, jenjang SMK/SMA sebelumnya pasti sudah selesai
    has_higher_edu = any(e.degree_level in {"Diploma", "Bachelor", "Master"} for e in entries)
    if has_higher_edu:
        for e in entries:
            if e.degree_level == "SMK/SMA" and e.status == "In Progress":
                e.status = "Completed"

    return entries


def extract_cv_local_hybrid(
    anonymized_text: str,
    raw_text: str,
    candidate_id: str,
    detected_name: str,
    source_filename: str,
    source_channel: str,
    vacancy: VacancyCriteria,
    reference_date: date,
) -> ExtractedCV:
    """Mengekstrak CV secara terstruktur menggunakan parser NLP lokal (tanpa panggilan LLM)."""
    sections = _split_into_sections(anonymized_text)

    summary_text = " ".join(sections["summary"]).strip()
    work_experiences = _extract_work_experiences(sections["experience"], reference_date, vacancy)
    # Jika seksi projects memuat proyek bertanggal dan lowongan menerima fresh graduate atau riwayat kerja masih kosong
    if sections["projects"] and (not work_experiences or vacancy.accepts_fresh_graduate):
        proj_entries = _extract_work_experiences(sections["projects"], reference_date, vacancy)
        for pe in proj_entries:
            if pe.employment_type == "Full-time":
                pe.employment_type = "Freelance"
            work_experiences.append(pe)
    education = _extract_education(sections["education"], anonymized_text, reference_date, vacancy)

    # Hitung entri organisasi (terpisah dari pengalaman kerja profesional)
    org_text = "\n".join(sections["organization"])
    org_count = len(DATE_RANGE_REGEX.findall(org_text))
    if org_count == 0 and sections["organization"]:
        org_count = max(1, sum(1 for l in sections["organization"] if len(l) < 60 and not l.startswith(("•", "-")) and "20" in l))

    # Ekstrak & normalisasi skills dari seksi skills + seluruh teks CV
    skills_raw_lines = sections["skills"]
    skills_from_section = normalize_skills(skills_raw_lines)
    skills_from_text = extract_known_skills_from_text(anonymized_text)
    skills_normalized = normalize_skills(skills_from_section + skills_from_text)

    # Deteksi bukti kepatuhan regulasi / standar kualitas / testing sesuai domain lowongan
    if vacancy.domain == "architecture":
        reg_keywords = [
            "building code", "building codes", "kode bangunan", "peraturan bangunan",
            "local regulations", "building regulations", "regulatory requirements",
            "safety standards", "standar keselamatan", "industry standards",
            "imb", "pbg", "slf", "quality standards", "environmental regulations",
        ]
        db_keywords = [
            "site supervision", "site survey", "survei lokasi", "site measurements",
            "shop drawings", "as-built drawings", "working drawings", "gambar kerja",
            "construction", "konstruksi", "kontraktor", "renovation", "renovasi",
            "overseeing the project from concept through to completion",
            "monitoring the project", "supervise", "mengawasi", "pengawasan proyek",
            "pelaksanaan proyek", "desain dan bangun",
        ]
        sup_keywords = [
            "material", "materials", "supplier", "pemasok", "lighting", "pencahayaan",
            "furniture", "furnitur", "interior", "specifications", "spesifikasi",
        ]
    elif vacancy.domain == "ai_data":
        reg_keywords = [
            "testing", "pytest", "validation", "unit test", "evaluation", "benchmark",
            "data security", "privacy", "pii", "anonymiz", "scalab", "accuracy", "f1", "roc", "auc",
        ]
        db_keywords = [
            "end-to-end", "end to end", "deployment", "deploy", "production", "pipeline",
            "training", "fine-tun", "rag", "langchain", "fastapi", "flask", "rest api",
            "feature engineering", "preprocessing", "eda", "model", "llm", "generative ai",
        ]
        sup_keywords = [
            "cross-functional", "stakeholder", "collaborat", "business", "non-technical",
            "api", "integration", "integrasi", "real-time", "real time", "tim",
        ]
    else:
        reg_keywords = [
            "testing", "unit test", "qa", "quality", "security", "compliance",
            "standard", "validation", "audit", "scalab", "documentation",
        ]
        db_keywords = [
            "end-to-end", "deployment", "production", "implementation", "implementasi",
            "delivered", "built", "developed", "launched", "architecture", "pipeline",
        ]
        sup_keywords = [
            "stakeholder", "client", "vendor", "cross-functional", "collaborat", "team", "koordinasi",
        ]

    reg_evidence: List[str] = []
    for line in anonymized_text.splitlines():
        low = line.lower()
        if any(k in low for k in reg_keywords):
            cleaned_line = line.lstrip("•-* ").strip()
            if cleaned_line and cleaned_line not in reg_evidence:
                reg_evidence.append(cleaned_line[:180])

    db_evidence: List[str] = []
    for exp in work_experiences:
        if not exp.is_relevant_domain:
            continue
        blob = f"{exp.role} {' '.join(exp.highlights)}".lower()
        if any(k in blob for k in db_keywords):
            db_evidence.append(f"{exp.role} ({exp.start_date} s.d. {exp.end_date})")
    # Untuk domain non-arsitektur (mis. AI Engineer fresh grad), cek juga seksi projects & summary
    if not db_evidence and vacancy.domain != "architecture":
        proj_blob = " ".join(sections["projects"] + sections["summary"]).lower()
        matched_hits = [k for k in db_keywords if k in proj_blob]
        if len(matched_hits) >= 2:
            db_evidence.append(f"Bukti proyek/portofolio relevan ({', '.join(matched_hits[:4])})")

    has_supplier_material = any(k in anonymized_text.lower() for k in sup_keywords)

    # Sinyal ketersediaan (currently employed / studying)
    currently_employed = any(
        exp.end_date == "Present" and exp.employment_type == "Full-time" for exp in work_experiences
    )
    currently_studying = any(ed.status == "In Progress" for ed in education)

    availability_signals: List[str] = []
    if currently_employed:
        availability_signals.append("Masih aktif bekerja penuh waktu (perlu konfirmasi masa notice period untuk join ASAP).")
    elif any(exp.end_date == "Present" and exp.employment_type == "Freelance" for exp in work_experiences):
        availability_signals.append("Berstatus Freelancer aktif (fleksibilitas tinggi untuk bergabung ASAP).")
    if currently_studying:
        in_prog = [f"{ed.degree_level} ({ed.field_of_study})" for ed in education if ed.status == "In Progress"]
        availability_signals.append(
            f"Sedang menempuh studi aktif: {', '.join(in_prog)} (perlu konfirmasi jadwal kuliah vs kerja penuh waktu)."
        )

    # Deteksi bahasa CV
    indo_hits = sum(
        1
        for w in [
            "ringkasan",
            "riwayat karier",
            "pengalaman kerja",
            "keahlian",
            "pendidikan",
            "saat ini",
            "sekarang",
            "tahun",
            "bulan",
        ]
        if w in raw_text.lower()
    )
    eng_hits = sum(
        1
        for w in ["experience", "design", "project", "drawing", "skills", "education"]
        if w in raw_text.lower()
    )
    if indo_hits >= 2 and eng_hits >= 2:
        lang_mix = "Campuran Indonesia & Inggris"
    elif indo_hits >= 2:
        lang_mix = "Bahasa Indonesia"
    else:
        lang_mix = "Bahasa Inggris"

    grad_ym = infer_bachelor_graduation_ym(education)
    timeline = calculate_timeline_summary(
        experiences=work_experiences,
        summary_text=summary_text or anonymized_text[:600],
        reference_date=reference_date,
        graduation_ym=grad_ym,
    )

    raw_char_count = len(raw_text.strip())
    extraction_warnings: List[str] = []
    if raw_char_count < 80:
        extraction_warnings.append(
            "Teks yang berhasil diekstrak sangat sedikit (<80 karakter); indikasi PDF hasil pindaian gambar tanpa OCR atau dokumen kosong."
        )
    elif not work_experiences and not education:
        extraction_warnings.append(
            "Format CV tidak standar: riwayat pengalaman kerja dan pendidikan formal tidak terdeteksi oleh parser lokal."
        )

    return ExtractedCV(
        candidate_id=candidate_id,
        source_filename=source_filename,
        source_channel=source_channel,
        detected_name=detected_name,
        language_mix=lang_mix,
        summary_text=summary_text,
        education=education,
        work_experiences=work_experiences,
        organizational_experiences_count=org_count,
        skills_raw=skills_raw_lines,
        skills_normalized=skills_normalized,
        has_building_codes_evidence=len(reg_evidence) > 0,
        building_codes_evidence=reg_evidence[:4],
        has_design_and_build_evidence=len(db_evidence) > 0,
        design_and_build_evidence=db_evidence[:4],
        has_supplier_or_material_evidence=has_supplier_material,
        availability_signals=availability_signals,
        currently_employed=currently_employed,
        currently_studying=currently_studying,
        timeline=timeline,
        raw_text_char_count=raw_char_count,
        extraction_warnings=extraction_warnings,
        extraction_engine="local_hybrid",
        llm_tokens_used=0,
        estimated_cost_usd=0.0,
    )
