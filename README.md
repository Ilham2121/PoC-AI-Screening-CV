# Hybrid AI/ML CV Screening & Candidate Ranking Platform

Project ini merupana Proof of Concept untuk otomatisasi seleksi dan perankingan CV berbasis **Hybrid AI/ML** yang dirancang untuk berbagai type lowongan

Proyek ini menggabungkan pemahaman konteks **LLM (DeepSeek API)** dengan **guardrail deterministik Python** untuk menjaga transparansi, reproduktibilitas skor, privasi data (*Blind Screening*), dan mekanisme *Human-in-the-Loop* (*Confidence Gate*).

## Fitur Utama

1. **Multi-Domain Vacancy Parser (`src/pdf_parser.py`)**: Membaca deskripsi lowongan (PDF/DOCX/TXT), mendeteksi domain posisi secara otomatis (`ai_data`, `software_engineering`, `architecture`, `general`), memisahkan *core required tools* vs *preferred/alternative stack*, serta mengenali kriteria *Fresh Graduate*.
2. **Layout-Aware Document Parsing (`src/pdf_parser.py`)**: Menggunakan PyMuPDF berbasis blok koordinat sehingga tata letak CV multi-kolom tetap terbaca utuh tanpa teks tercampur antar kolom.
3. **Blind Screening & PII Anonymization (`src/pii_anonymizer.py`)**: Menyensor otomatis nama kandidat, email, nomor telepon, tempat/tanggal lahir, alamat domisili, dan nama institusi pendidikan sebelum tahap evaluasi AI untuk meminimalkan bias serta mendukung kepatuhan UU PDP.
4. **Hybrid Extraction & Deterministic Guardrails (`src/deepseek_client.py`, `src/fallback_extractor.py`, `src/timeline.py`, `src/skill_normalizer.py`)**:
   - **DeepSeek API (`deepseek-chat`)** dengan *Structured JSON Output* untuk pemahaman konteks proyek lintas domain, dilengkapi *retry* & *per-CV fallback* otomatis.
   - **Mode Lokal Hybrid** (offline tanpa API Key) sebagai *baseline* cepat.
   - **Guardrail Python**: normalisasi alias & *typo* lintas domain (`PyTorch`, `FastAPI`, `LangChain`, `AutoCAD`, `SketchUp`, dll.), ekuivalensi stack (`FastAPI`/`Flask` → `REST API`), pemisahan pengalaman organisasi, *interval merging* untuk mendeduplikasi masa kerja tumpang tindih, pelacakan pengalaman pasca-lulus, dan deteksi anomali klaim durasi.
5. **Hybrid Scoring, Confidence Gate & Pelaporan (`src/scorer.py`, `src/excel_exporter.py`)**:
   - Rubrik 5 dimensi yang bobotnya dapat disesuaikan HR + *TF-IDF Lexical Cosine Similarity* (baseline cepat pada PoC, dapat ditingkatkan ke *multilingual dense embeddings* pada produksi).
   - **Confidence Gate (`Perlu Review Manual`)**: CV hasil pindaian tanpa OCR, file rusak, atau format non-standar otomatis diarahkan ke review manual HR, bukan ditolak otomatis.
   - Output **Explainable AI**: alasan keputusan berbasis bukti, pertanyaan wawancara spesifik, dashboard **Streamlit**, serta ekspor laporan **Excel 4 Sheet** dan **JSON**.

## Struktur Folder

```text
This_project/
├── app.py                        # Aplikasi Web Streamlit interaktif (multi-posisi + dataset demo)
├── run_pipeline.py               # Skrip CLI end-to-end untuk batch processing + ekspor Excel/JSON
├── requirements.txt              # Daftar dependensi Python
├── .env.example                  # Template konfigurasi DEEPSEEK_API_KEY
├── src/                          # Modul inti pipeline (parser, anonymizer, timeline, scorer, DeepSeek, Excel)
├── tests/                        # Unit test, edge-case test & headless Streamlit AppTest (pytest)
├── output/                       # Folder hasil laporan Excel (.xlsx) dan JSON (.json)
└── docs/                         # Diagram arsitektur Mermaid & aset presentasi
```

## Cara Menjalankan

### 1. Instalasi Dependensi
```powershell
pip install -r requirements.txt
```

### 2. Konfigurasi API Key DeepSeek (Opsional untuk Mode DeepSeek)
Salin `.env.example` menjadi `.env`, lalu masukkan API Key :
```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```
Anda juga dapat memasukkan API Key langsung di sidebar aplikasi Streamlit tanpa membuat file `.env`.

### 3. Menjalankan Aplikasi Web Streamlit
```powershell
streamlit run app.py
```

### 4. Menjalankan Pipeline CLI
> **Catatan `--reference-date`:** Untuk dataset demo bawaan (Q4 2024), gunakan default `--reference-date 2024-10-01`. Untuk lowongan terbaru Anda sendiri, atur ke tanggal hari ini (misalnya `--reference-date 2026-11-25`).

Mode Lokal Hybrid (offline, tanpa biaya API):
```powershell
python run_pipeline.py
```

Mode DeepSeek API dengan lowongan kustom:
```powershell
python run_pipeline.py --vacancy "path/ke/Vacancy_AI_Engineer.pdf" --cv-dir "path/ke/folder_cv" --engine deepseek --reference-date 2026-11-25
```

### 5. Menjalankan Pengujian Otomatis
```powershell
pytest tests/ -v
```
