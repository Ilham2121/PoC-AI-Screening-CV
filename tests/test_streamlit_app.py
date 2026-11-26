"""Headless UI verification test untuk aplikasi Streamlit (memastikan tidak ada error runtime & seluruh kontrol bekerja)."""

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_empty_state_and_runs_screening_cleanly():
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception

    # Pastikan empty state muncul sebelum tombol diklik
    assert any("Belum ada hasil seleksi" in md.value for md in at.markdown)

    # Klik tombol "Jalankan Seleksi CV" (mode default: Dataset Studi Kasus + Mode Lokal Hybrid)
    run_buttons = [b for b in at.sidebar.button if b.label == "Jalankan Seleksi CV"]
    assert len(run_buttons) == 1
    run_buttons[0].click().run()

    assert not at.exception
    # Pastikan hasil tersimpan di session_state dengan 5 kandidat ter-ranking
    assert "screening_report" in at.session_state
    report = at.session_state["screening_report"]
    assert report.metadata.total_candidates == 5
    assert report.metadata.shortlisted_count == 2
    assert report.metadata.kiv_count == 1
    assert report.metadata.rejected_count == 2
    # Pastikan tidak ada pesan error UI
    assert len(at.error) == 0
