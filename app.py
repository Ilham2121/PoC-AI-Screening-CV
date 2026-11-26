"""Aplikasi Web Streamlit: Platform Otomatisasi Seleksi CV Berbasis Hybrid AI/ML (Multi-Domain).

Dirancang sebagai purwarupa portofolio AI/ML yang adaptif untuk berbagai posisi lowongan
(AI/Data, Software Engineering, Arsitektur, hingga peran bisnis umum).
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.excel_exporter import export_screening_report_to_excel_bytes
from src.pdf_parser import extract_text_from_bytes, parse_vacancy_text
from src.pipeline import run_screening_from_payloads
from src.schemas import CandidateEvaluation, ScoringWeights, ScreeningReport

BASE_DIR = Path(__file__).resolve().parent
LOCAL_SAMPLE_DIR = BASE_DIR / "sample_data"
FALLBACK_SAMPLE_DIR = BASE_DIR.parent / "Soal Test BP AI Specialist 3"
SAMPLE_DIR = LOCAL_SAMPLE_DIR if LOCAL_SAMPLE_DIR.exists() else FALLBACK_SAMPLE_DIR
load_dotenv(BASE_DIR / ".env")


def _get_config_value(key: str, default: str = "") -> str:
    """Ambil konfigurasi dari environment variable atau Streamlit Secrets secara aman."""
    val = os.getenv(key, "").strip()
    if val:
        return val
    secrets_paths = (
        BASE_DIR / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    )
    if any(p.exists() for p in secrets_paths):
        try:
            secret_val = st.secrets.get(key, default)
            return str(secret_val).strip() if secret_val is not None else default
        except Exception:
            return default
    return default

st.set_page_config(
    page_title="HR CV Screening Workbench | PoC AI/ML",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling bersih dan tenang (Warm Paper #F7F6F2 + Deep Slate #18212F + Deep Forest #14532D)
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }
    h1, h2, h3 {
        color: #18212F;
        letter-spacing: -0.015em;
    }
    .page-subtitle {
        color: #4B5563;
        font-size: 0.98rem;
        margin-top: -0.6rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E0D8;
        border-radius: 6px;
        padding: 0.95rem 1.1rem;
    }
    .metric-label {
        color: #4B5563;
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        color: #18212F;
        font-size: 1.55rem;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        line-height: 1.2;
    }
    .metric-sub {
        color: #6B7280;
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .status-shortlisted {
        background-color: #DCFCE7;
        color: #14532D;
        border: 1px solid #86EFAC;
    }
    .status-kiv {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
    }
    .status-review {
        background-color: #DBEAFE;
        color: #1E40AF;
        border: 1px solid #93C5FD;
    }
    .status-rejected {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
    .panel-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E0D8;
        border-radius: 6px;
        padding: 1.2rem 1.35rem;
        margin-bottom: 1rem;
    }
    .section-kicker {
        color: #14532D;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _status_badge_html(status: str) -> str:
    if "Shortlisted" in status:
        cls = "status-shortlisted"
    elif "Pertimbangan" in status:
        cls = "status-kiv"
    elif "Review Manual" in status:
        cls = "status-review"
    else:
        cls = "status-rejected"
    return f'<span class="status-pill {cls}">{status}</span>'


def _load_sample_payloads() -> Tuple[bytes, str, List[Tuple[str, bytes]]]:
    vacancy_path = SAMPLE_DIR / "Vacancy Detail.pdf"
    cv_paths = sorted(SAMPLE_DIR.glob("CV Sample *.pdf"))
    if not vacancy_path.exists() or not cv_paths:
        raise FileNotFoundError(f"Folder dataset studi kasus tidak ditemukan di: {SAMPLE_DIR}")
    cv_payloads = [(p.name, p.read_bytes()) for p in cv_paths]
    return vacancy_path.read_bytes(), vacancy_path.name, cv_payloads


def _render_dimension_comparison_chart(evaluations: List[CandidateEvaluation]) -> go.Figure:
    """Grafik perbandingan skor kandidat pada 5 dimensi kualifikasi."""
    names = [f"#{ev.rank} {ev.candidate_name}" for ev in evaluations]
    dims = [
        ("Pengalaman Relevan", [ev.dimension_scores.relevant_experience for ev in evaluations], "#14532D"),
        ("Software & Teknis", [ev.dimension_scores.software_technical for ev in evaluations], "#1E3A5F"),
        ("Pendidikan", [ev.dimension_scores.education for ev in evaluations], "#4B5563"),
        ("Kecocokan Tugas", [ev.dimension_scores.responsibility_fit for ev in evaluations], "#92400E"),
        ("Regulasi & Kesiapan", [ev.dimension_scores.regulatory_readiness for ev in evaluations], "#64748B"),
    ]
    fig = go.Figure()
    for label, vals, color in dims:
        fig.add_trace(
            go.Bar(
                name=label,
                x=names,
                y=vals,
                marker_color=color,
                text=[f"{v:.0f}" for v in vals],
                textposition="outside",
            )
        )
    fig.update_layout(
        title=dict(
            text="Bagaimana perbandingan skor kandidat pada 5 dimensi kualifikasi?",
            font=dict(size=15, color="#18212F"),
            x=0,
        ),
        barmode="group",
        yaxis=dict(title="Skor (0-100)", range=[0, 114], gridcolor="#E5E7EB"),
        xaxis=dict(title=""),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=75, b=40),
        height=390,
    )
    return fig


def _render_candidate_detail(ev: CandidateEvaluation) -> None:
    st.markdown(
        f"""
        <div class="panel-box">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                <div>
                    <div class="section-kicker">Peringkat #{ev.rank} | {ev.candidate_id} | {ev.source_channel}</div>
                    <h3 style="margin:0;">{ev.candidate_name}</h3>
                </div>
                <div style="text-align:right;">
                    {_status_badge_html(ev.status)}
                    <div style="font-size:1.35rem; font-weight:700; color:#18212F; margin-top:0.25rem;">
                        Skor Total: {ev.overall_score:.1f} / 100
                    </div>
                </div>
            </div>
            <p style="margin-top:0.85rem; margin-bottom:0; color:#1F2937; line-height:1.55;">
                <strong>Alasan Keputusan:</strong> {ev.decision_rationale}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Kekuatan Utama (Berbasis Bukti CV)**")
        for item in ev.strengths:
            st.markdown(f"- {item}")

        st.markdown("**Kesesuaian Software Wajib**")
        matched_str = ", ".join(ev.matched_required_software) if ev.matched_required_software else "Tidak ada"
        missing_str = ", ".join(ev.missing_required_software) if ev.missing_required_software else "Lengkap"
        st.markdown(f"- Terpenuhi: `{matched_str}`")
        st.markdown(f"- Belum Tercantum: `{missing_str}`")
        st.markdown(f"- Seluruh Skill Ternormalisasi: {', '.join(ev.extracted_cv.skills_normalized) or '-'}")

    with c_right:
        st.markdown("**Celah Kualifikasi & Catatan Risiko**")
        for item in ev.gaps_and_risks:
            st.markdown(f"- {item}")

        st.markdown("**Pertanyaan Konfirmasi Wawancara HR**")
        for idx, q in enumerate(ev.interview_questions, start=1):
            st.markdown(f"{idx}. {q}")

    st.markdown("**Rincian Riwayat Kerja & Deduplikasi Timeline**")
    tl = ev.extracted_cv.timeline
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Durasi Mentah (Semua)", f"{tl.raw_sum_months_all} bln")
    t2.metric("Durasi Bersih (Semua)", f"{tl.merged_months_all} bln")
    post_grad_delta = (
        f"Pasca-lulus: {tl.post_graduation_years_relevant:.1f} thn"
        if tl.post_graduation_years_relevant is not None
        else None
    )
    t3.metric("Pengalaman Relevan Bersih", f"{tl.merged_years_relevant:.1f} thn", delta=post_grad_delta, delta_color="off")
    t4.metric("TF-IDF Lexical Score", f"{ev.semantic_similarity_score:.1f} / 100")

    if ev.extracted_cv.work_experiences:
        exp_rows = [
            {
                "Jabatan": w.role,
                "Perusahaan / Tipe": f"{w.company or '-'} ({w.employment_type})",
                "Periode": f"{w.start_date or '?'} s.d. {w.end_date or '?'}",
                "Relevan dengan Posisi": "Ya" if w.is_relevant_domain else "Tidak",
                "Catatan Relevansi": w.relevance_reason,
            }
            for w in ev.extracted_cv.work_experiences
        ]
        st.dataframe(pd.DataFrame(exp_rows), use_container_width=True, hide_index=True)

    if ev.anomaly_flags:
        st.markdown("**Bendera Anomali & Audit Data**")
        for flag in ev.anomaly_flags:
            st.warning(flag)


def main() -> None:
    st.title("Hybrid AI/ML CV Screening & Candidate Ranking")
    st.markdown(
        '<div class="page-subtitle">'
        "Seleksi CV otomatis berbasis AI: Blind Screening anti-bias, ekstraksi terstruktur, "
        "deduplikasi timeline, confidence gate (Human-in-the-Loop), dan perankingan berbasis bukti."
        "</div>",
        unsafe_allow_html=True,
    )

    # =========================================================================
    # SIDEBAR KONFIGURASI
    # =========================================================================
    with st.sidebar:
        st.header("Konfigurasi Seleksi")

        data_source = st.radio(
            "Sumber Dokumen",
            options=["Dataset Demo Bawaan (5 CV Sample)", "Unggah Lowongan & CV Sendiri"],
            index=0,
        )

        uploaded_vacancy = None
        uploaded_cvs = None
        if data_source == "Unggah Lowongan & CV Sendiri":
            uploaded_vacancy = st.file_uploader(
                "Unggah Deskripsi Lowongan (PDF/DOCX/TXT)",
                type=["pdf", "docx", "txt"],
            )
            uploaded_cvs = st.file_uploader(
                "Unggah File CV Kandidat (bisa lebih dari satu)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
            )

        st.divider()
        engine_label = st.radio(
            "Mesin Evaluasi AI",
            options=[
                "Mode Lokal Hybrid (Offline)",
                "DeepSeek API + Hybrid",
            ],
            index=0,
            help="Mode Lokal Hybrid berjalan instan tanpa API Key. Mode DeepSeek menggunakan LLM deepseek-chat ditambah guardrail deterministik.",
        )
        engine_mode = "deepseek" if "DeepSeek" in engine_label else "local_hybrid"

        env_api_key = _get_config_value("DEEPSEEK_API_KEY", "")
        if env_api_key.startswith("sk-masukkan"):
            env_api_key = ""

        env_base_url = _get_config_value("DEEPSEEK_BASE_URL", "")
        if env_base_url and not os.getenv("DEEPSEEK_BASE_URL"):
            os.environ["DEEPSEEK_BASE_URL"] = env_base_url

        deepseek_api_key = ""
        deepseek_model = _get_config_value("DEEPSEEK_MODEL", "deepseek-chat")
        if engine_mode == "deepseek":
            if env_api_key:
                st.caption("API Key DeepSeek terdeteksi dari `.env` / Streamlit Secrets.")
            deepseek_api_key = st.text_input(
                "DeepSeek API Key (Opsional jika sudah di Secrets)" if env_api_key else "DeepSeek API Key",
                value="",
                type="password",
                placeholder="sk-... (biarkan kosong untuk memakai Secrets)" if env_api_key else "sk-...",
                help="Kunci API hanya dipakai selama sesi berlangsung dan tidak disimpan ke kode sumber.",
            )
            effective_api_key = deepseek_api_key.strip() or env_api_key
            model_options = ["deepseek-chat", "deepseek-reasoner"]
            model_idx = model_options.index(deepseek_model) if deepseek_model in model_options else 0
            deepseek_model = st.selectbox(
                "Model DeepSeek",
                options=model_options,
                index=model_idx,
            )
        else:
            effective_api_key = ""

        st.divider()
        blind_screening = st.checkbox(
            "Aktifkan Blind Screening (Sensor PII & Institusi)",
            value=True,
            help="Menyensor nama, email, telepon, tanggal lahir, alamat, dan nama universitas sebelum penilaian AI untuk mencegah bias.",
        )

        ref_date = st.date_input(
            "Tanggal Acuan Masa Kerja",
            value=date(2024, 10, 1),
            help="Dataset 5 CV sampel diekspor sekitar Oktober 2024. Ubah tanggal ini untuk simulasi evaluasi pada waktu berbeda.",
        )

        with st.expander("Atur Bobot Rubrik Penilaian"):
            w_exp = st.slider("Pengalaman Relevan (%)", 0, 50, 30, 5)
            w_sw = st.slider("Software & Teknis (%)", 0, 50, 25, 5)
            w_resp = st.slider("Kecocokan Tugas (%)", 0, 50, 20, 5)
            w_edu = st.slider("Pendidikan (%)", 0, 50, 15, 5)
            w_reg = st.slider("Regulasi & Kesiapan (%)", 0, 50, 10, 5)
            total_w = w_exp + w_sw + w_resp + w_edu + w_reg
            st.caption(f"Total bobot saat ini: {total_w}% (otomatis dinormalisasi menjadi 100%).")

        run_btn = st.button("Jalankan Seleksi CV", type="primary", use_container_width=True)

    # =========================================================================
    # EKSEKUSI PIPELINE
    # =========================================================================
    if run_btn:
        try:
            if data_source == "Dataset Demo Bawaan (5 CV Sample)":
                vac_bytes, vac_name, cv_payloads = _load_sample_payloads()
            else:
                if not uploaded_vacancy or not uploaded_cvs:
                    st.error(
                        "Unggah minimal 1 dokumen deskripsi lowongan dan 1 file CV terlebih dahulu di panel sebelah kiri."
                    )
                    st.stop()
                vac_bytes = uploaded_vacancy.read()
                vac_name = uploaded_vacancy.name
                cv_payloads = [(f.name, f.read()) for f in uploaded_cvs]

            if engine_mode == "deepseek" and not deepseek_api_key.strip():
                st.error(
                    "DeepSeek API Key belum diisi. Masukkan API Key DeepSeek Anda di panel kiri, "
                    "atau pilih 'Mode Lokal Hybrid' untuk mencoba langsung tanpa API Key."
                )
                st.stop()

            weights = ScoringWeights(
                relevant_experience=float(w_exp),
                software_technical=float(w_sw),
                education=float(w_edu),
                responsibility_fit=float(w_resp),
                regulatory_readiness=float(w_reg),
            )

            progress_bar = st.progress(0.0, text="Memulai pipeline seleksi CV...")
            raw_texts_map = {}
            for idx, (fname, fbytes) in enumerate(cv_payloads, start=1):
                try:
                    raw_texts_map[f"CAND-{idx:03d}"] = extract_text_from_bytes(fbytes, fname)
                except Exception as parse_exc:
                    raw_texts_map[f"CAND-{idx:03d}"] = f"[Gagal membaca teks dokumen: {parse_exc}]"

            def _on_progress(curr: int, total: int, msg: str) -> None:
                pct = min(0.95, curr / max(1, total))
                progress_bar.progress(pct, text=msg)

            report = run_screening_from_payloads(
                vacancy_bytes=vac_bytes,
                vacancy_filename=vac_name,
                cv_payloads=cv_payloads,
                engine_mode=engine_mode,
                reference_date=ref_date,
                blind_screening=blind_screening,
                weights=weights,
                api_key=effective_api_key or None,
                model=deepseek_model,
                on_progress=_on_progress,
            )
            progress_bar.progress(1.0, text="Seleksi selesai.")
            progress_bar.empty()

            st.session_state["screening_report"] = report
            st.session_state["raw_texts_map"] = raw_texts_map

        except Exception as exc:
            st.error(
                f"Gagal menjalankan proses seleksi ({type(exc).__name__}): {exc}. "
                "Periksa koneksi internet dan validitas API Key DeepSeek, atau beralih ke Mode Lokal Hybrid."
            )

    # =========================================================================
    # EMPTY STATE (SEBELUM EKSEKUSI PERTAMA)
    # =========================================================================
    report: ScreeningReport | None = st.session_state.get("screening_report")
    if report is None:
        st.markdown(
            """
            <div class="panel-box">
                <div class="section-kicker">Sistem Siap Digunakan</div>
                <h3 style="margin-top:0;">Belum ada hasil seleksi pada sesi ini</h3>
                <p style="color:#4B5563; margin-bottom:0.6rem;">
                    Klik tombol <strong>Jalankan Seleksi CV</strong> di panel sebelah kiri untuk mencoba
                    <strong>Dataset Demo Bawaan</strong>, atau pilih <strong>Unggah Lowongan &amp; CV Sendiri</strong>
                    untuk mengevaluasi posisi apapun.
                </p>
                <p style="color:#4B5563; margin-bottom:0;">
                    Gunakan <strong>Mode Lokal Hybrid</strong> untuk hasil cepat tanpa biaya, lalu bandingkan
                    dengan <strong>DeepSeek API + Hybrid</strong> menggunakan API Key.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return

    # =========================================================================
    # HASIL SELEKSI (4 TAB TERFOKUS)
    # =========================================================================
    meta = report.metadata
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Total CV</div>'
            f'<div class="metric-value">{meta.total_candidates}</div>'
            f'<div class="metric-sub">{meta.vacancy_title}</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Shortlisted</div>'
            f'<div class="metric-value" style="color:#14532D;">{meta.shortlisted_count}</div>'
            f'<div class="metric-sub">Siap tahap wawancara</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        review_note = (
            f"Cadangan HR | Review Manual: {meta.manual_review_count}"
            if meta.manual_review_count > 0
            else "Cadangan / review HR"
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Pertimbangan (KIV)</div>'
            f'<div class="metric-value" style="color:#92400E;">{meta.kiv_count}</div>'
            f'<div class="metric-sub">{review_note}</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Tidak Lolos</div>'
            f'<div class="metric-value" style="color:#991B1B;">{meta.rejected_count}</div>'
            f'<div class="metric-sub">Belum penuhi syarat utama</div></div>',
            unsafe_allow_html=True,
        )
    with m5:
        cost_line = (
            f"${meta.total_estimated_cost_usd:.5f} ({meta.total_tokens_used} tok)"
            if meta.engine_mode == "deepseek"
            else "Lokal ($0.00)"
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Waktu & Biaya</div>'
            f'<div class="metric-value">{meta.processing_seconds:.2f} dtk</div>'
            f'<div class="metric-sub">{cost_line}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    tab_rank, tab_bias, tab_vac, tab_export = st.tabs(
        [
            "Keputusan & Ranking HR",
            "Audit Etika & Blind Screening",
            "Kriteria Lowongan",
            "Ekspor Laporan",
        ]
    )

    # -------------------------------------------------------------------------
    # TAB 1: KEPUTUSAN & RANKING HR
    # -------------------------------------------------------------------------
    with tab_rank:
        table_rows = [
            {
                "Rank": ev.rank,
                "ID Anonim": ev.candidate_id,
                "Nama Kandidat": ev.candidate_name,
                "Status Rekomendasi": ev.status,
                "Skor Total": ev.overall_score,
                "Gelar S1": "Terpenuhi" if ev.degree_requirement_met else "Belum",
                "Pengalaman Relevan (Thn)": ev.extracted_cv.timeline.merged_years_relevant,
                "Software Kurang": ", ".join(ev.missing_required_software) or "Lengkap",
                "Sumber Kanal": ev.source_channel,
            }
            for ev in report.evaluations
        ]
        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Skor Total": st.column_config.ProgressColumn(
                    "Skor Total", min_value=0, max_value=100, format="%.1f"
                )
            },
        )

        st.plotly_chart(_render_dimension_comparison_chart(report.evaluations), use_container_width=True)

        st.subheader("Bedah Bukti & Rekomendasi per Kandidat")
        selected_label = st.selectbox(
            "Pilih kandidat untuk melihat alasan detail dan panduan wawancara:",
            options=[f"#{ev.rank} - {ev.candidate_name} ({ev.overall_score:.1f} | {ev.status})" for ev in report.evaluations],
        )
        selected_idx = int(selected_label.split(" - ")[0].lstrip("#")) - 1
        _render_candidate_detail(report.evaluations[selected_idx])

    # -------------------------------------------------------------------------
    # TAB 2: AUDIT ETIKA & BLIND SCREENING
    # -------------------------------------------------------------------------
    with tab_bias:
        st.subheader("Ringkasan Mitigasi Bias & Deteksi Anomali")
        st.write(
            "Sebelum teks CV dinilai oleh mesin AI, modul PII Anonymizer menyensor identitas personal "
            "(nama, email, nomor telepon, tempat/tanggal lahir, alamat, serta nama universitas) agar penilaian "
            "berfokus murni pada kompetensi dan rekam jejak proyek."
        )

        audit_rows = [
            {
                "ID Anonim": ev.candidate_id,
                "Nama Asli (Dibuka untuk HR)": ev.candidate_name,
                "Total Sensor PII": ev.pii_redactions_count,
                "Kategori Disensor": ", ".join(ev.pii_redacted_categories) or "-",
                "Overlap Tanggal": "Ya (Dideduplikasi)" if ev.extracted_cv.timeline.has_overlapping_roles else "Tidak",
                "Anomali Klaim": "Ya (Perlu Verifikasi)" if ev.extracted_cv.timeline.claim_discrepancy_flag else "Tidak",
            }
            for ev in report.evaluations
        ]
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

        st.subheader("Inspeksi Sebelum vs Sesudah Blind Screening")
        inspect_name = st.selectbox(
            "Pilih kandidat untuk memeriksa teks yang disensor:",
            options=[f"{ev.candidate_id} ({ev.candidate_name})" for ev in report.evaluations],
            key="pii_inspect_select",
        )
        inspect_id = inspect_name.split(" ")[0]
        target_ev = next(ev for ev in report.evaluations if ev.candidate_id == inspect_id)
        raw_texts_map = st.session_state.get("raw_texts_map", {})

        col_raw, col_anon = st.columns(2)
        with col_raw:
            st.markdown("**Teks Asli Hasil Layout-Aware Parser**")
            st.text_area(
                "Teks Asli",
                value=raw_texts_map.get(inspect_id, "Tidak tersedia"),
                height=360,
                label_visibility="collapsed",
            )
        with col_anon:
            st.markdown("**Teks Anonim yang Diterima Mesin Evaluasi AI**")
            st.text_area(
                "Teks Anonim",
                value=target_ev.pii_audit.anonymized_text,
                height=360,
                label_visibility="collapsed",
            )

    # -------------------------------------------------------------------------
    # TAB 3: KRITERIA LOWONGAN
    # -------------------------------------------------------------------------
    with tab_vac:
        vac = report.vacancy
        st.subheader(f"Profil Lowongan Terekstrak: {vac.title}")
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("**Persyaratan Utama (Requirements)**")
            for r in vac.requirements_raw:
                st.markdown(f"- {r}")
        with v2:
            st.markdown("**Tanggung Jawab (Responsibilities)**")
            for resp in vac.responsibilities:
                st.markdown(f"- {resp}")

        st.markdown("**Parameter Terstruktur yang Dipakai Mesin Penilai**")
        st.json(
            {
                "title": vac.title,
                "domain_detected": vac.domain,
                "required_degree_fields": vac.required_degree_fields,
                "min_years_relevant_experience": vac.min_years_relevant_experience,
                "accepts_fresh_graduate": vac.accepts_fresh_graduate,
                "required_software": vac.required_software,
                "preferred_skills": vac.preferred_skills,
                "key_technical_skills": vac.key_technical_skills,
                "requires_asap_join": vac.requires_asap_join,
                "weights_normalized": meta.weights_used.normalized(),
            }
        )

    # -------------------------------------------------------------------------
    # TAB 4: EKSPOR LAPORAN
    # -------------------------------------------------------------------------
    with tab_export:
        st.subheader("Unduh Laporan Hasil Seleksi")
        st.write(
            "Laporan Excel berisi 4 lembar kerja terstruktur: Ringkasan_Ranking, Detail_Skoring, "
            "Audit_Bias_Anomali, dan Panduan_Wawancara."
        )

        xlsx_bytes = export_screening_report_to_excel_bytes(report)
        json_bytes = report.model_dump_json(indent=2).encode("utf-8")
        suffix = "DeepSeek" if meta.engine_mode == "deepseek" else "LocalHybrid"

        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button(
                label="Unduh Laporan Excel (.xlsx)",
                data=xlsx_bytes,
                file_name=f"Hasil_Screening_CV_{suffix}_Ilham_A_A.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )
        with c_dl2:
            st.download_button(
                label="Unduh Data Terstruktur (.json)",
                data=json_bytes,
                file_name=f"Hasil_Screening_CV_{suffix}_Ilham_A_A.json",
                mime="application/json",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
