"""Skema data terstruktur (Pydantic) untuk pipeline otomatisasi seleksi CV."""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ScoringWeights(BaseModel):
    """Bobot penilaian tiap dimensi (total harus 100)."""

    relevant_experience: float = Field(default=30.0, ge=0, le=100)
    software_technical: float = Field(default=25.0, ge=0, le=100)
    education: float = Field(default=15.0, ge=0, le=100)
    responsibility_fit: float = Field(default=20.0, ge=0, le=100)
    regulatory_readiness: float = Field(default=10.0, ge=0, le=100)

    def normalized(self) -> Dict[str, float]:
        total = (
            self.relevant_experience
            + self.software_technical
            + self.education
            + self.responsibility_fit
            + self.regulatory_readiness
        )
        if total <= 0:
            raise ValueError("Total bobot penilaian harus lebih besar dari 0.")
        return {
            "relevant_experience": self.relevant_experience / total,
            "software_technical": self.software_technical / total,
            "education": self.education / total,
            "responsibility_fit": self.responsibility_fit / total,
            "regulatory_readiness": self.regulatory_readiness / total,
        }


class VacancyCriteria(BaseModel):
    """Kriteria terstruktur dari dokumen lowongan kerja (multi-domain)."""

    title: str
    domain: Literal["architecture", "ai_data", "software_engineering", "general"] = "general"
    required_degree_fields: List[str] = Field(default_factory=list)
    min_years_relevant_experience: float = 0.0
    accepts_fresh_graduate: bool = False
    required_software: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    key_technical_skills: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    requirements_raw: List[str] = Field(default_factory=list)
    raw_text: str = ""
    requires_asap_join: bool = False
    requires_regulatory_compliance: bool = False


class EducationEntry(BaseModel):
    """Entri pendidikan formal kandidat."""

    degree_level: Literal["SMK/SMA", "Diploma", "Bachelor", "Master", "Other"] = "Other"
    field_of_study: str = ""
    institution_redacted: str = "[INSTITUTION_REDACTED]"
    institution_original: str = ""
    status: Literal["Completed", "In Progress", "Unknown"] = "Unknown"
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    is_relevant_field: bool = False
    is_expected_graduation: bool = False


class WorkExperienceEntry(BaseModel):
    """Entri riwayat pekerjaan profesional (bukan kepanitiaan/organisasi)."""

    role: str
    company: str = ""
    employment_type: Literal["Full-time", "Freelance", "Intern", "Contract", "Unknown"] = "Unknown"
    start_date: Optional[str] = None  # Format YYYY-MM
    end_date: Optional[str] = None  # Format YYYY-MM atau "Present"
    is_relevant_domain: bool = False
    relevance_reason: str = ""
    highlights: List[str] = Field(default_factory=list)


class PIIAuditItem(BaseModel):
    """Catatan penyensoran satu kategori data pribadi (PII) untuk audit bias."""

    category: str
    count: int
    sample_redacted_label: str


class PIIAuditResult(BaseModel):
    """Hasil penyensoran PII untuk tahap Blind Screening."""

    candidate_id: str
    detected_name: str
    anonymized_text: str
    redacted_items: List[PIIAuditItem] = Field(default_factory=list)
    total_redactions: int = 0


class TimelineSummary(BaseModel):
    """Ringkasan hasil perhitungan masa kerja setelah deduplikasi overlap."""

    raw_sum_months_all: int = 0
    merged_months_all: int = 0
    raw_sum_months_relevant: int = 0
    merged_months_relevant: int = 0
    merged_years_relevant: float = 0.0
    post_graduation_years_relevant: Optional[float] = None
    has_overlapping_roles: bool = False
    overlap_notes: List[str] = Field(default_factory=list)
    claimed_years_in_summary: Optional[float] = None
    claim_discrepancy_flag: bool = False
    claim_discrepancy_note: str = ""


class ExtractedCV(BaseModel):
    """Informasi terstruktur yang diekstrak dari satu CV."""

    candidate_id: str
    source_filename: str
    source_channel: str = "Job Portal Upload"
    detected_name: str = "Unknown"
    language_mix: str = "English"
    summary_text: str = ""
    education: List[EducationEntry] = Field(default_factory=list)
    work_experiences: List[WorkExperienceEntry] = Field(default_factory=list)
    organizational_experiences_count: int = 0
    skills_raw: List[str] = Field(default_factory=list)
    skills_normalized: List[str] = Field(default_factory=list)
    has_building_codes_evidence: bool = False
    building_codes_evidence: List[str] = Field(default_factory=list)
    has_design_and_build_evidence: bool = False
    design_and_build_evidence: List[str] = Field(default_factory=list)
    has_supplier_or_material_evidence: bool = False
    availability_signals: List[str] = Field(default_factory=list)
    currently_employed: bool = False
    currently_studying: bool = False
    timeline: TimelineSummary = Field(default_factory=TimelineSummary)
    raw_text_char_count: int = 0
    extraction_warnings: List[str] = Field(default_factory=list)
    fallback_used: bool = False
    extraction_engine: Literal["deepseek", "local_hybrid"] = "local_hybrid"
    llm_tokens_used: int = 0
    estimated_cost_usd: float = 0.0


class DimensionScores(BaseModel):
    """Rincian skor 0-100 untuk setiap dimensi evaluasi."""

    relevant_experience: float = Field(ge=0, le=100)
    software_technical: float = Field(ge=0, le=100)
    education: float = Field(ge=0, le=100)
    responsibility_fit: float = Field(ge=0, le=100)
    regulatory_readiness: float = Field(ge=0, le=100)


class CandidateEvaluation(BaseModel):
    """Hasil evaluasi lengkap, ranking, dan penjelasan untuk satu kandidat."""

    rank: int = 0
    candidate_id: str
    candidate_name: str
    source_filename: str
    source_channel: str = "Job Portal Upload"
    overall_score: float = Field(ge=0, le=100)
    semantic_similarity_score: float = Field(ge=0, le=100)
    status: Literal[
        "Shortlisted (Wawancara)",
        "Pertimbangan Khusus (KIV)",
        "Perlu Review Manual",
        "Tidak Lolos Kualifikasi",
    ]
    needs_manual_review: bool = False
    manual_review_reasons: List[str] = Field(default_factory=list)
    dimension_scores: DimensionScores
    matched_required_software: List[str] = Field(default_factory=list)
    missing_required_software: List[str] = Field(default_factory=list)
    degree_requirement_met: bool = False
    experience_requirement_met: bool = False
    strengths: List[str] = Field(default_factory=list)
    gaps_and_risks: List[str] = Field(default_factory=list)
    decision_rationale: str = ""
    interview_questions: List[str] = Field(default_factory=list)
    anomaly_flags: List[str] = Field(default_factory=list)
    pii_redactions_count: int = 0
    pii_redacted_categories: List[str] = Field(default_factory=list)
    extracted_cv: ExtractedCV
    pii_audit: PIIAuditResult


class ScreeningRunMetadata(BaseModel):
    """Metadata eksekusi batch screening untuk transparansi dan audit."""

    run_timestamp: str
    reference_date: date
    vacancy_title: str
    engine_mode: Literal["deepseek", "local_hybrid"]
    blind_screening_enabled: bool = True
    total_candidates: int = 0
    shortlisted_count: int = 0
    kiv_count: int = 0
    manual_review_count: int = 0
    rejected_count: int = 0
    processing_seconds: float = 0.0
    total_tokens_used: int = 0
    total_estimated_cost_usd: float = 0.0
    weights_used: ScoringWeights = Field(default_factory=ScoringWeights)


class ScreeningReport(BaseModel):
    """Hasil akhir satu sesi screening untuk semua kandidat."""

    metadata: ScreeningRunMetadata
    vacancy: VacancyCriteria
    evaluations: List[CandidateEvaluation] = Field(default_factory=list)
