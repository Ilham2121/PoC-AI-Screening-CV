"""Normalisasi nama software, bahasa pemrograman, framework, dan keahlian teknis lintas domain."""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Set, Tuple

CANONICAL_SKILL_ALIASES: Dict[str, List[str]] = {
    # --- Architecture, Interior & Engineering Design ---
    "AutoCAD": ["autocad", "auto cad", "autodesk autocad", "cad", "2d cad", "3d cad"],
    "SketchUp": ["sketchup", "sketch up", "skechup", "skethcup", "3d sketchup", "google sketchup"],
    "Revit": ["revit", "revit architecture", "autodesk revit", "revit mep", "revit bim"],
    "Lumion": ["lumion", "lumion 3d"],
    "Enscape": ["enscape", "enscape 3d"],
    "V-Ray": ["v-ray", "vray", "v ray", "v-ray render", "vray render"],
    "Twinmotion": ["twinmotion", "twin motion"],
    "D5 Render": ["d5 render", "d5", "d5render"],
    "Adobe Photoshop": ["photoshop", "adobe photoshop"],
    "Adobe Illustrator": ["illustrator", "adobe illustrator"],
    "ArchiCAD": ["archicad", "archi cad"],
    "3ds Max": ["3ds max", "3dsmax", "3d max", "autodesk 3ds max"],
    "Rhino": ["rhino", "rhinoceros", "rhino 3d"],
    "BIM": ["bim", "building information modeling", "building information modelling"],
    "SolidWorks": ["solidworks", "solid works"],
    "Figma": ["figma"],
    "2D & 3D Modeling": ["2d & 3d modeling", "3d modeling", "3d visualization", "3d visualization & modeling"],
    "Technical Drawing": [
        "technical drawing",
        "drafting & technical drawing",
        "architectural drafting & detailing",
        "working drawings",
        "shop drawings",
        "as-built drawings",
    ],
    "Landscape Architecture": ["landscape architecture", "landscape design"],
    "Green Building & Sustainability": ["sustainability and green building design", "green building"],
    # --- AI, Machine Learning, GenAI & Data Science ---
    "Python": ["python", "python3", "py"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "TensorFlow": ["tensorflow", "tensor flow", "keras"],
    "PyTorch": ["pytorch", "py torch"],
    "Machine Learning": ["machine learning", "ml", "supervised learning", "unsupervised learning"],
    "Deep Learning": ["deep learning", "neural network", "neural networks", "cnn", "rnn", "transformer"],
    "LLM / GenAI": [
        "generative ai",
        "genai",
        "gen ai",
        "large language model",
        "large language models",
        "llm",
        "llms",
        "prompt engineering",
        "ai agent",
        "ai agents",
    ],
    "LangChain": ["langchain", "lang chain", "langgraph"],
    "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "Vector Database": ["vector database", "vector db", "chromadb", "chroma", "faiss", "pinecone", "qdrant", "milvus", "weaviate"],
    "OpenAI API": ["openai", "openai api", "gpt-4", "chatgpt api"],
    "Gemini API": ["gemini", "gemini api", "google gemini", "vertex ai"],
    "Hugging Face": ["hugging face", "huggingface", "transformers"],
    "NLP": ["nlp", "natural language processing", "text mining"],
    "Computer Vision": ["computer vision", "opencv", "yolo", "object detection"],
    "Feature Engineering": ["feature engineering", "data preprocessing", "data cleaning"],
    "EDA": ["exploratory data analysis", "eda", "data analysis", "data analytics"],
    "MLOps": ["mlops", "mlflow", "kubeflow", "model deployment"],
    "SQL": ["sql", "mysql", "postgresql", "postgres", "sqlite", "sql server", "bigquery", "t-sql"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Apache Spark": ["spark", "pyspark", "apache spark"],
    "Streamlit": ["streamlit"],
    # --- Software Engineering, Backend, Cloud & DevOps ---
    "REST API": ["restful api", "restful apis", "rest api", "rest apis", "api development", "backend api", "backend apis"],
    "FastAPI": ["fastapi", "fast api"],
    "Flask": ["flask"],
    "Django": ["django"],
    "Git": ["git", "github", "gitlab", "bitbucket", "version control"],
    "Docker": ["docker", "containerization"],
    "Kubernetes": ["kubernetes", "k8s"],
    "JavaScript": ["javascript", "js", "es6"],
    "TypeScript": ["typescript", "ts"],
    "React": ["react", "reactjs", "react.js"],
    "Next.js": ["next.js", "nextjs"],
    "Node.js": ["node.js", "nodejs", "express.js"],
    "Java": ["java", "spring boot"],
    "Go": ["golang"],
    "C++": ["c++"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "AWS": ["aws", "amazon web services"],
    "GCP": ["gcp", "google cloud", "google cloud platform"],
    "Azure": ["azure", "microsoft azure"],
    "CI/CD": ["ci/cd", "github actions", "jenkins", "gitlab ci"],
    "Linux": ["linux", "ubuntu", "bash"],
    # --- General / Office / Management ---
    "Microsoft Office": ["microsoft office", "ms office", "ms. office"],
    "Microsoft Excel": ["microsoft excel", "ms excel", "excel"],
    "Project Management": [
        "project management",
        "construction management & site supervision",
        "site supervision",
    ],
    "Agile / Scrum": ["agile", "scrum", "kanban", "jira"],
}

# Peta ekuivalensi kompetensi: menguasai salah satu tool spesifik memenuhi kategori kompetensi induknya
SKILL_SATISFIED_BY: Dict[str, Set[str]] = {
    "Machine Learning": {"Scikit-learn", "TensorFlow", "PyTorch", "Deep Learning", "XGBoost", "NLP", "Computer Vision", "MLOps"},
    "LLM / GenAI": {"LangChain", "RAG", "OpenAI API", "Gemini API", "Vector Database", "Hugging Face"},
    "REST API": {"FastAPI", "Flask", "Django", "Node.js"},
    "EDA": {"Pandas", "NumPy", "Power BI", "Tableau"},
}

# Daftar skill/tool konkret yang valid sebagai kriteria teknis lowongan lintas domain
CORE_TOOL_SKILLS: Set[str] = {
    # Architecture / Design
    "AutoCAD", "SketchUp", "Revit", "Lumion", "Enscape", "V-Ray", "Twinmotion", "D5 Render",
    "ArchiCAD", "3ds Max", "Rhino", "BIM", "SolidWorks", "Figma", "Adobe Photoshop", "Adobe Illustrator",
    # AI / ML / Data
    "Python", "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch",
    "Machine Learning", "Deep Learning", "LLM / GenAI", "LangChain", "RAG", "Vector Database",
    "OpenAI API", "Gemini API", "Hugging Face", "NLP", "Computer Vision", "MLOps",
    "SQL", "Power BI", "Tableau", "Apache Spark", "Streamlit",
    # Software / Backend / DevOps
    "REST API", "FastAPI", "Flask", "Django", "Git", "Docker", "Kubernetes",
    "JavaScript", "TypeScript", "React", "Next.js", "Node.js", "Java", "Go", "C++",
    "MongoDB", "Redis", "AWS", "GCP", "Azure", "CI/CD", "Linux",
    # Analytics / Office
    "Microsoft Excel",
}

_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for _canonical, _aliases in CANONICAL_SKILL_ALIASES.items():
    _ALIAS_TO_CANONICAL[_canonical.lower()] = _canonical
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias.lower()] = _canonical

_ALL_ALIAS_KEYS: List[str] = list(_ALIAS_TO_CANONICAL.keys())


def canonicalize_skill(raw_skill: str) -> str:
    """Memetakan satu nama skill/software ke bentuk kanoniknya."""
    cleaned = re.sub(r"\s+", " ", raw_skill.strip().strip("•-*.,;:()[]"))
    if not cleaned:
        return ""
    lower = cleaned.lower()

    if lower in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[lower]

    # Cek substring untuk nama dengan tambahan kata seperti "Autodesk AutoCAD 2023" atau "Python 3.11"
    for alias, canonical in sorted(_ALIAS_TO_CANONICAL.items(), key=lambda x: len(x[0]), reverse=True):
        if len(alias) >= 4 and re.search(rf"\b{re.escape(alias)}\b", lower):
            return canonical

    # Fuzzy matching untuk menangani typo ringan seperti "Skechup" atau "Tensorflw"
    matches = difflib.get_close_matches(lower, _ALL_ALIAS_KEYS, n=1, cutoff=0.84)
    if matches:
        return _ALIAS_TO_CANONICAL[matches[0]]

    return cleaned


def normalize_skills(raw_skills: List[str]) -> List[str]:
    """Menormalisasi dan menduplikasi daftar skill sambil menjaga urutan kemunculan."""
    seen = set()
    result: List[str] = []
    for item in raw_skills:
        parts = re.split(r"[,;\n/|]+", item)
        for part in parts:
            sub_parts = [part]
            if "&" in part and "2d" not in part.lower() and "drafting" not in part.lower() and "management" not in part.lower():
                sub_parts = part.split("&")
            for sub in sub_parts:
                canon = canonicalize_skill(sub)
                if canon and len(canon) <= 60 and canon.lower() not in seen:
                    seen.add(canon.lower())
                    result.append(canon)
    return result


def extract_known_skills_from_text(text: str) -> List[str]:
    """Memindai teks bebas CV/lowongan untuk menemukan seluruh software/skill kanonik yang disebut."""
    found: List[str] = []
    seen = set()
    lower_text = text.lower()

    # Token pendek yang tetap aman dipindai dengan word boundary ketat (mis. SQL, Git, RAG, LLM, NLP, AWS, GCP, BIM, EDA, C++)
    safe_short_aliases = {"sql", "git", "rag", "llm", "nlp", "aws", "gcp", "bim", "eda", "cad", "c++", "k8s"}

    for canonical, aliases in CANONICAL_SKILL_ALIASES.items():
        for alias in aliases:
            if len(alias) <= 2 and alias not in safe_short_aliases:
                continue
            if alias == "c++":
                matched = "c++" in lower_text
            else:
                pattern = rf"\b{re.escape(alias)}\b"
                matched = bool(re.search(pattern, lower_text))
            if matched:
                if canonical.lower() not in seen:
                    seen.add(canonical.lower())
                    found.append(canonical)
                break

    # Pindai token tunggal untuk menangkap typo software/framework di tengah paragraf
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{4,}", text)
    fuzzy_targets = [
        "autocad", "sketchup", "revit", "lumion", "enscape", "twinmotion", "archicad",
        "python", "pandas", "numpy", "tensorflow", "pytorch", "fastapi", "langchain", "docker", "kubernetes",
    ]
    for token in tokens:
        t_low = token.lower()
        if t_low in _ALIAS_TO_CANONICAL:
            canon = _ALIAS_TO_CANONICAL[t_low]
        else:
            close = difflib.get_close_matches(t_low, fuzzy_targets, n=1, cutoff=0.85)
            canon = _ALIAS_TO_CANONICAL[close[0]] if close else ""
        if canon and canon.lower() not in seen:
            seen.add(canon.lower())
            found.append(canon)

    return found


def match_required_skills(
    candidate_skills: List[str],
    required_skills: List[str],
) -> Tuple[List[str], List[str]]:
    """Membandingkan skill kandidat (sudah dinormalisasi) terhadap syarat wajib lowongan, termasuk ekuivalensi stack."""
    cand_norm_list = normalize_skills(candidate_skills)
    cand_norm = {s.lower() for s in cand_norm_list}
    req_norm = normalize_skills(required_skills)

    matched: List[str] = []
    missing: List[str] = []
    for req in req_norm:
        if req.lower() in cand_norm:
            matched.append(req)
            continue
        # Cek apakah skill kategori induk terpenuhi oleh salah satu tool spesifik kandidat (mis. FastAPI -> REST API)
        equivalents = SKILL_SATISFIED_BY.get(req, set())
        if any(eq.lower() in cand_norm for eq in equivalents):
            matched.append(req)
        else:
            missing.append(req)
    return matched, missing
