# Panduan Topik & Narasi Slide Presentasi (Question 3)

Dokumen ini disusun sebagai bahan dasar bagi Anda untuk merakit file presentasi PDF dengan rapi. Seluruh struktur slide dirancang untuk menjawab langsung **5 kriteria penilaian** pada soal (Kualitas Solusi AI/ML, Kelayakan Implementasi, Efisiensi & Efektivitas, Pertimbangan Etika & Bias, serta Kejelasan Penyampaian).

---

## Struktur Rekomendasi: 11 Slide Ringkas & Tajam

### Slide 1: Halaman Judul
* **Judul Utama**: Otomatisasi Seleksi CV Berbasis Hybrid AI/ML
* **Subjudul**: Solusi Skala Besar yang Cepat, Objektif, Transparan, dan Rendah Bias untuk Tim HR Manufaktur
* **Info Penyusun**: Ilham Aly Abdillah | Technical Assessment: AI/ML Specialist (PT Allure Industries)

### Slide 2: Latar Belakang & Tantangan Rekrutmen Saat Ini
* **Poin Masalah**:
  * Pertumbuhan cepat perusahaan manufaktur memicu lonjakan kebutuhan rekrutmen lintas posisi teknis dan operasional secara bersamaan.
  * Seleksi manual menghabiskan rata-rata 5 hingga 8 menit per CV untuk membaca, mencatat kualifikasi, dan membandingkan kandidat. Untuk 500 pelamar per posisi, dibutuhkan sekitar 40 hingga 65 jam kerja HR.
  * Risiko kelelahan (*reviewer fatigue*) dan bias bawah sadar (nama almamater, usia, format CV visual, atau urutan baca).
* **Tantangan Nyata dari 5 Sampel CV Studi Kasus**:
  * Format beragam: ekspor JobStreet bahasa campuran, CV 2 kolom padat, hingga variasi *typo* penulisan software (`Skechup`, `Sketch Up`).
  * Jebakan data tersembunyi: pekerjaan paralel/tumpang tindih (berisiko terhitung ganda), klaim pengalaman di ringkasan yang tidak cocok dengan riwayat tertulis, pengalaman organisasi yang panjang, hingga pekerjaan lintas bidang (pengujian beton sipil vs arsitek *design & build*).
* **Narasi Singkat**: Masalah seleksi CV bukan hanya soal volume dokumen, melainkan kompleksitas dan jebakan data di dalam CV yang tidak cukup diselesaikan dengan filter kata kunci biasa.

### Slide 3: Visi Solusi & 6 Kapabilitas Utama
* **Pesan Kunci**: Merancang pipeline **Hybrid AI/ML** yang menggabungkan fleksibilitas pemahaman konteks LLM dengan ketepatan matematika aturan deterministik dan *information retrieval*.
* **Pemetaan ke 6 Tujuan Soal**:
  1. **Akuisisi Otomatis**: Integrasi kanal masuk dari JobStreet, LinkedIn, Glints, dan email rekrutmen dengan pelabelan *Vacancy ID*.
  2. **Ekstraksi Layout-Aware**: Pembacaan PDF/DOCX multi-kolom menjadi profil JSON terstruktur.
  3. **Pencocokan Multi-Dimensi**: Evaluasi pada 5 dimensi terukur plus *TF-IDF Semantic Similarity*.
  4. **Penyaringan & Perankingan**: Klasifikasi otomatis ke *Shortlisted*, *Pertimbangan Khusus (KIV)*, dan *Tidak Lolos*.
  5. **Siap Wawancara Minim Intervensi**: Menghasilkan daftar *shortlist* sekaligus pertanyaan klarifikasi wawancara otomatis.
  6. **Output Transparan (*Explainable AI*)**: Skor rinci, alasan penerimaan/penolakan berbasis bukti CV, dan laporan Excel multi-sheet.

### Slide 4: Arsitektur Alur Sistem End-to-End
* **Visual**: Tempelkan diagram yang dirender dari `docs/arsitektur_sistem.mmd` (bisa langsung di-paste ke `mermaid.live` atau Draw.io untuk diekspor menjadi PNG/SVG resolusi tinggi).
* **Penjelasan 5 Lapisan Utama**:
  1. *Ingestion Layer*: Webhook / Email Parser / n8n mengambil CV otomatis berdasarkan kode lowongan.
  2. *Document & Privacy Layer*: Layout-Aware Parser (PyMuPDF) + **PII Anonymizer** (memisahkan identitas asli ke *vault* terenkripsi sebelum masuk ke AI).
  3. *Hybrid Extraction & Guardrail Layer*: DeepSeek LLM (*Structured JSON*) + *Timeline Deduplicator* + *Skill Normalizer*.
  4. *Matching & Ranking Layer*: Hard/Soft Gate + TF-IDF Cosine Similarity + Weighted Scoring Rubric.
  5. *HR Workbench & Integration Layer*: Dashboard Streamlit, Laporan Excel 4 Sheet, dan integrasi API ke ATS/HRIS/Email.

### Slide 5: Mengapa Pendekatan Hybrid AI/ML? (Pemilihan Model & Algoritma)
* **Poin Penting (Menjawab Kriteria Kualitas Solusi AI/ML)**:
  * Mengapa tidak murni Regex/Keyword? Keyword gagal membedakan pengalaman *Structural Testing* sipil dari *Architectural Design & Build*, serta gagal menyaring pengalaman kepanitiaan kampus.
  * Mengapa tidak murni LLM (ChatGPT/DeepSeek saja)? LLM murni rentan salah hitung matematika durasi kerja yang tumpang tindih (*overlapping dates*) dan skornya bisa sedikit berubah antar eksekusi.
  * **Keunggulan Hybrid Pipeline**:
    * **PyMuPDF Coordinate Block Parser**: Menjaga urutan baca CV 2 kolom agar teks Pendidikan dan Keahlian tidak tercampur.
    * **DeepSeek-V3 (`deepseek-chat`)**: Rasio akurasi pemahaman konteks terhadap biaya terbaik di kelasnya (~$0.27 / 1 juta token input), mendukung *JSON mode* untuk ekstraksi dan narasi evaluasi Bahasa Indonesia.
    * **Deterministic Guardrails (Python)**: Normalisasi *fuzzy* nama software (`Skechup` -> `SketchUp`), penggabungan interval bulan (*interval merging*) agar pekerjaan paralel tidak dihitung ganda, dan deteksi anomali klaim pengalaman.
    * **TF-IDF Cosine Similarity (`scikit-learn`)**: Mengukur kedekatan leksikal dan semantik antara deskripsi proyek kandidat dengan tanggung jawab lowongan secara konsisten dan cepat.

### Slide 6: Rubrik Skoring & Logika Pengambilan Keputusan
* **Tampilkan Tabel Bobot 5 Dimensi (Konfigurabel oleh HR)**:
  * Pengalaman Relevan (*Design & Build*, durasi bersih): **30%**
  * Penguasaan Software & Teknis (AutoCAD, SketchUp, Revit + pendukung): **25%**
  * Kecocokan Tanggung Jawab (Pilar tugas + TF-IDF Similarity): **20%**
  * Pendidikan (Gelar selesai di bidang relevan): **15%**
  * Pemahaman Regulasi Bangunan & Kesiapan (*Join ASAP*): **10%**
* **Aturan Transparansi Status**:
  * **Shortlisted (Wawancara)**: Skor >= 72, gelar S1 terpenuhi, dan seluruh software wajib lengkap.
  * **Pertimbangan Khusus (KIV)**: Skor >= 60 dengan syarat gelar terpenuhi (atau pengalaman relevan kuat), disimpan sebagai cadangan.
  * **Tidak Lolos Kualifikasi**: Syarat fundamental tidak terpenuhi, disertai alasan objektif berbasis fakta CV.

### Slide 7: Studi Kasus Junior Architect (Hasil PoC pada 5 CV Dummy)
* **Visual**: Ambil tangkapan layar tabel ranking dan grafik perbandingan 5 dimensi dari aplikasi Streamlit (`app.py`).
* **Ringkasan Temuan Nyata pada 5 CV (Acuan Tanggal Dataset: Okt 2024)**:
  1. **#1 Bayu (Skor 90.7 | Shortlisted)**: S1 Arsitektur, lengkap AutoCAD/SketchUp/Revit, pengalaman lapangan kuat (kontraktor, QC, drafter, interior). Sistem mendeteksi dan mendeduplikasi periode kerja tumpang tindih dari 114 bulan mentah menjadi **81 bulan bersih (6.8 tahun)**.
  2. **#2 Indri (Skor 86.7 | Shortlisted)**: S1 Arsitektur (sedang S2), lengkap AutoCAD/SketchUp/Revit, skor kesesuaian semantik tertinggi (33.3) dengan bukti eksplisit *local building codes* dan supervisi lapangan. Pekerjaan paralel penuh waktu + *freelance* dideduplikasi dari 45 bulan menjadi **24 bulan bersih (2.0 tahun)**. Catatan wawancara: konfirmasi jadwal kuliah S2 terhadap syarat *join ASAP*.
  3. **#3 Permana Abadi (Skor 71.0 | Pertimbangan Khusus / KIV)**: S1 Arsitektur, pengalaman *freelance* 3.1 tahun bersih (proyek rumah & lampu jalan IKN). Kegiatan organisasi kampus berhasil dipisahkan dari pengalaman kerja. Kurang **Revit** dan minim pengalaman supervisi konstruksi lapangan.
  4. **#4 Mohamad (Skor 64.8 | Tidak Lolos)**: Fokus kuat di *Drafter & 3D Virtual Tour*, tetapi masih aktif kuliah S1 (belum bergelar), tidak memiliki Revit, dan sistem menandai **anomali klaim** (mengklaim 10 tahun pengalaman sejak 2014 namun riwayat tertulis hanya 3.8 tahun sejak 2021).
  5. **#5 Dani (Skor 47.2 | Tidak Lolos)**: Masih kuliah S1 (lulusan SMK), tidak memiliki Revit, dan pengalaman utama 3 tahun adalah *Testing Technical* pengujian beton sipil (bukan arsitektur *design & build*), sehingga pengalaman arsitektur relevannya hanya **0.6 tahun**. Tanggal lahir eksplisit di CV berhasil disensor otomatis sebelum penilaian.

### Slide 8: Pertimbangan Etika, Privasi Data, dan Mitigasi Bias
* **Menjawab Kriteria Etika & Bias + Kepatuhan UU PDP (UU No. 27 Tahun 2022)**:
  * **Blind Screening Otomatis**: Sebelum masuk ke mesin AI, sistem menyensor Nama (anti bias gender/etnis), Tanggal Lahir/Usia (terbukti menangkap `19 February 2003` pada CV Dani), Email/Telepon/Alamat, serta Nama Universitas (mencegah *halo effect* kampus ternama seperti ITB vs universitas lain).
  * **Human-in-the-Loop (HITL)**: AI bertindak sebagai asisten penyaring dan pemberi rekomendasi beserta bukti kutipan CV. Keputusan akhir penolakan dan penerimaan tetap memiliki jalur audit dan dapat ditinjau HR.
  * **Zero-Data Retention & Audit Trail**: Menggunakan API dengan kebijakan tanpa pelatihan ulang data pelanggan (atau model *open-weights self-hosted* untuk data sangat sensitif), disertai log audit penyensoran PII pada Sheet 3 laporan Excel.

### Slide 9: Analisis Efisiensi Waktu & Penghematan Biaya (ROI)
* **Simulasi Skenario Rekrutmen Manufaktur (500 CV per Bulan / 10 Posisi Terbuka)**:
  * **Proses Manual HR**: 500 CV x 6 menit/CV = **50 jam kerja/bulan** (~6,2 hari kerja penuh 1 staf HR hanya untuk skrining awal).
  * **Proses Otomatis Hybrid AI**: 500 CV diproses dalam **< 15 menit** (batch paralel), staf HR cukup meninjau 15% kandidat teratas (~75 CV *shortlisted/KIV*, sekitar 4 jam kerja). **Penghematan waktu skrining awal: ~85% hingga 90%.**
  * **Estimasi Biaya Komputasi API DeepSeek-V3**:
    * Rata-rata ~2.000 token input + ~600 token output per CV = **~$0.0012 per CV** (sekitar **Rp19 per CV**).
    * Untuk 500 CV/bulan, total biaya API hanya **~$0.60/bulan (~Rp9.500/bulan)**, jauh lebih hemat dibanding jam lembur atau penambahan tenaga rekruter manual.
    * Total biaya infrastruktur ringan (server/cloud container kecil + n8n): sangat terjangkau dengan ROI positif sejak bulan pertama.

### Slide 10: Kelayakan Implementasi & Roadmap Produksi (3 Fase)
* **Fase 1 (Bulan 1 - Pilot & Shadow Mode)**: Jalankan PoC berdampingan dengan seleksi manual HR pada 2 hingga 3 posisi prioritas untuk kalibrasi bobot rubrik dan validasi akurasi *shortlist* (target kesesuaian >= 85% dengan keputusan HR senior).
* **Fase 2 (Bulan 2 - Integrasi Kanal & Otomasi)**: Hubungkan n8n / webhook ke email rekrutmen dan portal karir, aktifkan notifikasi otomatis ke Hiring Manager via Slack/Teams/Email, serta ekspor langsung ke HRIS/ATS perusahaan.
* **Fase 3 (Bulan 3+ - Skala Penuh & Continuous Learning)**: Tambahkan *feedback loop* dari hasil wawancara HR untuk menyempurnakan bobot kriteria per divisi (Manufaktur, Engineering, Arsitektur, Finance), pemantauan *bias drift* berkala, dan dukungan OCR untuk CV hasil pindaian gambar.

### Slide 11: Penutup & Demo Proof of Concept
* **Ringkasan Nilai Tambah**: Solusi tidak berhenti di konsep slide, melainkan sudah dibuktikan melalui **Aplikasi Web Streamlit interaktif**, **Skrip Pipeline Python**, **Unit Test otomatis**, dan **Laporan Excel 4 Sheet** yang siap dipakai langsung oleh tim HR.
* **Ajakan Diskusi / Demo**: Siap mendemonstrasikan alur unggah CV, toggle *Blind Screening*, perbandingan Mode Lokal Hybrid vs DeepSeek API, hingga unduh laporan Excel secara langsung.
