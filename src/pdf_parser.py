"""Parser dokumen PDF dan DOCX berbasis tata letak (layout-aware) untuk CV dan deskripsi lowongan."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import List, Tuple, Union

import pymupdf
from docx import Document

from src.schemas import VacancyCriteria
from src.skill_normalizer import CORE_TOOL_SKILLS, extract_known_skills_from_text

BOILERPLATE_LINES = [
    "dibuat dengan profil jobstreet",
    "created with jobstreet profile",
]


def _clean_block_text(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if line.lower() in BOILERPLATE_LINES:
            continue
        lines.append(line)
    return "\n".join(lines)


def _reconstruct_block_lines(block: dict) -> str:
    """Merekonstruksi baris teks dari dict block PyMuPDF agar teks justified tidak terpecah per kata."""
    spans: List[Tuple[float, float, str]] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            txt = str(span.get("text", "")).strip()
            if not txt:
                continue
            bbox = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
            spans.append((float(bbox[1]), float(bbox[0]), txt))

    if not spans:
        return ""

    # Kelompokkan span yang berada pada garis vertikal (y0) yang sama (toleransi 3.5 pt)
    spans.sort(key=lambda item: (item[0], item[1]))
    grouped_lines: List[List[Tuple[float, float, str]]] = []
    for sp in spans:
        if not grouped_lines or abs(sp[0] - grouped_lines[-1][0][0]) > 3.5:
            grouped_lines.append([sp])
        else:
            grouped_lines[-1].append(sp)

    reconstructed: List[str] = []
    for row in grouped_lines:
        row.sort(key=lambda item: item[1])
        line_text = re.sub(r"\s+", " ", " ".join(item[2] for item in row)).strip()
        # Hilangkan spasi berlebih setelah '+' pada nomor telepon ("+ 628..." -> "+628...")
        line_text = re.sub(r"\+\s+(\d)", r"+\1", line_text)
        if not line_text or line_text.lower() in BOILERPLATE_LINES:
            continue
        reconstructed.append(line_text)
    return "\n".join(reconstructed)


def _extract_page_text_layout_aware(page: pymupdf.Page) -> str:
    """Membaca teks halaman PDF sambil mendeteksi band 2-kolom agar urutan baca tetap utuh."""
    page_width = float(page.rect.width) if page.rect.width else 600.0
    page_dict = page.get_text("dict")

    blocks: List[Tuple[float, float, float, float, str]] = []
    for b in page_dict.get("blocks", []):
        if b.get("type") != 0:
            continue
        bbox = b.get("bbox", (0.0, 0.0, 0.0, 0.0))
        cleaned = _reconstruct_block_lines(b)
        if cleaned:
            blocks.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), cleaned))

    if not blocks:
        return ""

    # Urutkan awal berdasarkan posisi vertikal
    blocks.sort(key=lambda item: (round(item[1], 1), round(item[0], 1)))

    # Klasifikasi setiap blok: full-width vs kolom kiri vs kolom kanan
    # Bedakan layout sidebar-label (JobStreet: kolom kiri sangat sempit, kolom kanan lebar >55%)
    # dari layout 2-kolom seimbang (seperti CV Dani: kolom kiri & kanan masing-masing <52% lebar halaman).
    classified = []
    for x0, y0, x1, y1, text in blocks:
        width_ratio = (x1 - x0) / page_width
        center_ratio = ((x0 + x1) / 2.0) / page_width
        if width_ratio > 0.54 or (x0 < 0.38 * page_width and x1 > 0.62 * page_width):
            col_type = "full"
        elif x1 <= 0.54 * page_width and center_ratio < 0.48:
            col_type = "left"
        elif x0 >= 0.44 * page_width and center_ratio >= 0.50:
            col_type = "right"
        else:
            col_type = "full"
        classified.append((x0, y0, x1, y1, text, col_type))

    # Pisahkan blok menjadi band berdasarkan pembatas blok full-width
    output_parts: List[str] = []
    pending_left: List[Tuple[float, float, float, float, str]] = []
    pending_right: List[Tuple[float, float, float, float, str]] = []

    def flush_columns() -> None:
        nonlocal pending_left, pending_right
        if not pending_left and not pending_right:
            return
        if pending_left and pending_right:
            # Cek apakah ini 2 kolom seimbang (masing-masing punya konten cukup signifikan)
            left_max_width = max((b[2] - b[0]) / page_width for b in pending_left)
            right_max_width = max((b[2] - b[0]) / page_width for b in pending_right)
            if left_max_width >= 0.18 and right_max_width >= 0.15:
                pending_left.sort(key=lambda item: (item[1], item[0]))
                pending_right.sort(key=lambda item: (item[1], item[0]))
                output_parts.extend(b[4] for b in pending_left)
                output_parts.extend(b[4] for b in pending_right)
            else:
                merged = pending_left + pending_right
                merged.sort(key=lambda item: (round(item[1], 1), item[0]))
                output_parts.extend(b[4] for b in merged)
        else:
            merged = pending_left + pending_right
            merged.sort(key=lambda item: (item[1], item[0]))
            output_parts.extend(b[4] for b in merged)
        pending_left = []
        pending_right = []

    for x0, y0, x1, y1, text, col_type in classified:
        if col_type == "full":
            flush_columns()
            output_parts.append(text)
        elif col_type == "left":
            pending_left.append((x0, y0, x1, y1, text))
        else:
            pending_right.append((x0, y0, x1, y1, text))

    flush_columns()
    return "\n".join(output_parts)


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Mengekstrak teks bersih dari bytes file PDF, DOCX, atau TXT."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        pages_text: List[str] = []
        with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                page_txt = _extract_page_text_layout_aware(page)
                if page_txt.strip():
                    pages_text.append(page_txt)
        return "\n\n".join(pages_text).strip()

    if ext == ".docx":
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()

    if ext in {".txt", ".md"}:
        return file_bytes.decode("utf-8", errors="ignore").strip()

    raise ValueError(f"Format file tidak didukung: {ext}. Gunakan PDF, DOCX, atau TXT.")


def extract_text_from_file(path: Union[str, Path]) -> str:
    """Mengekstrak teks dari file PDF, DOCX, atau TXT di disk."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    return extract_text_from_bytes(file_path.read_bytes(), file_path.name)


def _detect_vacancy_domain(raw_text: str) -> str:
    """Mendeteksi domain utama lowongan secara dinamis dari judul dan deskripsi pekerjaan."""
    low = raw_text.lower()
    scores = {
        "ai_data": sum(
            1
            for k in [
                "ai engineer", "machine learning", "deep learning", "data scien", "data analy",
                "generative ai", "large language model", "llm", "pytorch", "tensorflow",
                "scikit-learn", "feature engineering", "rag", "langchain", "nlp", "mlops",
            ]
            if k in low
        ),
        "software_engineering": sum(
            1
            for k in [
                "software engineer", "backend engineer", "frontend engineer", "fullstack",
                "full-stack", "devops", "microservices", "kubernetes", "react", "next.js",
                "typescript", "golang", "spring boot", "ci/cd",
            ]
            if k in low
        ),
        "architecture": sum(
            1
            for k in [
                "architect", "arsitek", "interior design", "autocad", "sketchup", "revit",
                "design and build", "design & build", "building codes", "shop drawing", "working drawing",
            ]
            if k in low
        ),
    }
    best_domain, best_score = max(scores.items(), key=lambda x: x[1])
    return best_domain if best_score >= 2 else "general"


def parse_vacancy_text(raw_text: str) -> VacancyCriteria:
    """Mengurai teks deskripsi lowongan lintas domain menjadi objek VacancyCriteria terstruktur."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    title = lines[0] if lines else "Open Position"

    requirements: List[str] = []
    responsibilities: List[str] = []
    section = "header"

    for line in lines[1:]:
        upper = line.upper().rstrip(":")
        if "REQUIREMENT" in upper or "QUALIFICATION" in upper or "KUALIFIKASI" in upper:
            section = "requirements"
            continue
        if "RESPONSIBILIT" in upper or "JOB DESCRIPTION" in upper or "TANGGUNG JAWAB" in upper:
            section = "responsibilities"
            continue

        cleaned_line = line.lstrip("•·-*.").strip()
        if not cleaned_line:
            continue
        if section == "requirements":
            requirements.append(cleaned_line)
        elif section == "responsibilities":
            responsibilities.append(cleaned_line)

    domain = _detect_vacancy_domain(raw_text)

    # Deteksi penerimaan fresh graduate & minimal tahun pengalaman
    accepts_fresh_grad = bool(
        re.search(
            r"\b(?:fresh\s+graduates?|recent\s+graduates?|entry[\s\-]level|lulusan\s+baru)\b",
            raw_text,
            flags=re.IGNORECASE,
        )
    )
    min_years = 0.0
    exp_match = re.search(r"minimum\s+(\d+(?:\.\d+)?)\s*\+?\s*year", raw_text, flags=re.IGNORECASE)
    if not exp_match:
        exp_match = re.search(r"min(?:imal|\.)?\s*(\d+(?:\.\d+)?)\s*tahun", raw_text, flags=re.IGNORECASE)
    if not exp_match:
        exp_match = re.search(r"at\s+least\s+(\d+(?:\.\d+)?)\s*\+?\s*year", raw_text, flags=re.IGNORECASE)
    if exp_match:
        min_years = float(exp_match.group(1))

    # Pisahkan tools/skills wajib (di luar klausa contoh "(such as ... or ...)") vs preferred/alternatif
    req_block = "\n".join(requirements) if requirements else raw_text
    paren_alt_pattern = re.compile(r"\((?:[^)]*(?:such as|e\.g\.|seperti|misalnya)[^)]*\bor\b[^)]*)\)", flags=re.I)
    paren_chunks = paren_alt_pattern.findall(req_block)
    core_req_text = paren_alt_pattern.sub(" ", req_block)

    core_detected = [s for s in extract_known_skills_from_text(core_req_text) if s in CORE_TOOL_SKILLS]
    alt_detected = [s for s in extract_known_skills_from_text(" ".join(paren_chunks)) if s in CORE_TOOL_SKILLS]
    resp_detected = [s for s in extract_known_skills_from_text("\n".join(responsibilities)) if s in CORE_TOOL_SKILLS]

    required_software = core_detected if core_detected else alt_detected
    preferred_skills = [
        s for s in dict.fromkeys(alt_detected + resp_detected) if s not in required_software
    ]

    # Ekstrak bidang pendidikan yang diterima lintas domain
    degree_field_catalog = [
        ("Architecture", ["architecture", "arsitektur"]),
        ("Interior Design", ["interior design", "desain interior"]),
        ("Architectural Engineering", ["architectural engineering", "teknik arsitektur"]),
        ("Civil Engineering", ["civil engineering", "teknik sipil"]),
        ("Information Technology", ["information technology", "teknologi informasi"]),
        ("Computer Science", ["computer science", "ilmu komputer", "informatics", "informatika"]),
        ("Data Science", ["data science", "sains data"]),
        ("Information Systems", ["information systems", "sistem informasi"]),
        ("Software Engineering", ["software engineering", "rekayasa perangkat lunak"]),
        ("Statistics", ["statistics", "statistika"]),
        ("Mathematics", ["mathematics", "matematika"]),
        ("Electrical Engineering", ["electrical engineering", "teknik elektro"]),
        ("Industrial Engineering", ["industrial engineering", "teknik industri"]),
        ("Mechanical Engineering", ["mechanical engineering", "teknik mesin"]),
        ("Economics & Finance", ["economics", "finance", "accounting", "akuntansi", "ekonomi", "keuangan"]),
        ("Management & Business", ["management", "business", "manajemen", "bisnis"]),
    ]
    degree_fields: List[str] = []
    degree_context_lines = [
        r for r in requirements if any(k in r.lower() for k in ["degree", "sarjana", "bachelor", "s1", "graduate", "pendidikan"])
    ]
    deg_blob = " ".join(degree_context_lines).lower() if degree_context_lines else raw_text.lower()
    for label, kws in degree_field_catalog:
        if any(kw in deg_blob for kw in kws):
            degree_fields.append(label)

    # Ekstrak keahlian teknis kunci lintas domain
    key_technical_skills: List[str] = []
    tech_checks = [
        ("Technical Drawing", ["technical drawing", "working drawing", "shop drawing", "gambar kerja"]),
        ("3D Modeling & Visualization", ["visualization", "modelling", "modeling", "3d"]),
        ("Design & Build", ["design and build", "design & build"]),
        ("Building Codes & Regulatory Standards", ["building codes", "regulatory standards", "regulations"]),
        ("Material & Supplier Sourcing", ["suppliers", "source and procure", "materials"]),
        ("Data Analysis & EDA", ["data analysis", "exploratory data analysis", "eda", "data cleaning", "preprocessing"]),
        ("Machine Learning & Algorithms", ["machine learning", "algorithm", "feature engineering", "training", "evaluation"]),
        ("Generative AI & LLM Systems", ["generative ai", "large language model", "llm", "rag", "langchain", "vector database"]),
        ("Backend API & Data Pipelines", ["restful api", "rest api", "backend api", "fastapi", "flask", "data pipeline"]),
        ("Model Testing, Scalability & Security", ["testing", "validating", "scalability", "security of company data"]),
        ("Project Management & Collaboration", ["project management", "monitor projects", "timelines", "cross-functional", "collaborate"]),
    ]
    lower_all = raw_text.lower()
    for label, keywords in tech_checks:
        if any(kw in lower_all for kw in keywords):
            key_technical_skills.append(label)

    requires_asap = bool(re.search(r"\b(asap|immediately|secepatnya)\b", raw_text, flags=re.IGNORECASE))
    requires_regulatory = bool(
        re.search(
            r"\b(building codes?|regulatory standards?|building regulations?|iso\s*\d+|compliance|kepatuhan regulasi)\b",
            raw_text,
            flags=re.IGNORECASE,
        )
    )

    return VacancyCriteria(
        title=title,
        domain=domain,
        required_degree_fields=degree_fields,
        min_years_relevant_experience=min_years,
        accepts_fresh_graduate=accepts_fresh_grad,
        required_software=required_software,
        preferred_skills=preferred_skills,
        key_technical_skills=key_technical_skills,
        responsibilities=responsibilities,
        requirements_raw=requirements,
        raw_text=raw_text,
        requires_asap_join=requires_asap,
        requires_regulatory_compliance=requires_regulatory,
    )


def parse_vacancy_document(path: Union[str, Path]) -> VacancyCriteria:
    """Membaca file lowongan (PDF/DOCX/TXT) dan mengurainya ke VacancyCriteria."""
    text = extract_text_from_file(path)
    return parse_vacancy_text(text)
