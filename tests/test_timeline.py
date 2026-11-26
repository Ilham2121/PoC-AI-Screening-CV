"""Unit tests untuk timeline deduplication & anomaly detection."""

from datetime import date
from src.schemas import WorkExperienceEntry
from src.timeline import calculate_timeline_summary, parse_month_year


def test_parses_indonesian_and_english_dates():
    ref = date(2024, 10, 1)
    assert parse_month_year("Sep 2023", ref) == (2023, 9)
    assert parse_month_year("Des 2022", ref) == (2022, 12)
    assert parse_month_year("Mei 2020", ref) == (2020, 5)
    assert parse_month_year("Okt 2023", ref) == (2023, 10)
    assert parse_month_year("Saat ini", ref) == (2024, 10)
    assert parse_month_year("Present", ref) == (2024, 10)


def test_merges_overlapping_intervals_without_double_counting():
    """Simulasi pola CV Bayu: beberapa pekerjaan berjalan tumpang tindih."""
    ref = date(2024, 10, 1)
    experiences = [
        WorkExperienceEntry(
            role="Interior Designer",
            company="PT Lumino",
            start_date="2023-09",
            end_date="Present",
            is_relevant_domain=True,
        ),
        WorkExperienceEntry(
            role="Architectural Drafter",
            company="PT Inkoneksi",
            start_date="2023-01",
            end_date="2023-09",
            is_relevant_domain=True,
        ),
        WorkExperienceEntry(
            role="Interior Architect",
            company="Skezma Studio",
            start_date="2018-02",
            end_date="2023-09",
            is_relevant_domain=True,
        ),
        WorkExperienceEntry(
            role="Kontraktor",
            company="CV Media Sarana",
            start_date="2018-06",
            end_date="2019-07",
            is_relevant_domain=True,
        ),
    ]
    summary = calculate_timeline_summary(
        experiences=experiences,
        summary_text="Started working in 2018.",
        reference_date=ref,
    )
    # Rentang bersih Feb 2018 sampai Okt 2024 = 81 bulan (inklusi batas), jauh lebih kecil dari jumlah mentah (~105 bulan)
    assert summary.has_overlapping_roles is True
    assert summary.merged_months_relevant < summary.raw_sum_months_relevant
    assert 79 <= summary.merged_months_relevant <= 81


def test_detects_claimed_vs_documented_experience_discrepancy():
    """Simulasi pola CV Mohamad: klaim 10 tahun sejak 2014, tapi riwayat tertulis mulai 2021."""
    ref = date(2024, 10, 1)
    experiences = [
        WorkExperienceEntry(
            role="Drafter & 3D Visualization",
            company="PT Serpong Indah",
            start_date="2023-01",
            end_date="Present",
            is_relevant_domain=True,
        ),
        WorkExperienceEntry(
            role="Drafter & 3D Visualization",
            company="PT Grand Utama",
            start_date="2021-01",
            end_date="2023-01",
            is_relevant_domain=True,
        ),
    ]
    summary = calculate_timeline_summary(
        experiences=experiences,
        summary_text="With 10 years of experience since 2014, specializes in 3D models.",
        reference_date=ref,
    )
    assert summary.claimed_years_in_summary == 10.0
    assert summary.claim_discrepancy_flag is True
    assert "10" in summary.claim_discrepancy_note
