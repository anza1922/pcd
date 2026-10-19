# Pipeline Bab 4 — Evaluasi Perbaikan Citra

## Urutan Penggunaan

### Langkah 1 — Anotasi Ground Truth

```
python buat_ground_truth.py
```

Jendela OpenCV akan muncul untuk setiap gambar di `dataset_wajah/dataset_wajah_rapi/orang_*`.

| Tombol | Aksi |
|--------|------|
| Klik-drag | Gambar bounding box wajah |
| ENTER / S | Simpan kotak, lanjut ke gambar berikutnya |
| Z | Hapus kotak terakhir |
| R | Reset semua kotak pada gambar ini |
| Q / ESC | Lewati gambar (tidak perlu dianotasi) |

Output: `ground_truth.csv`

> Anotasi bisa dilanjutkan — gambar yang sudah dianotasi dilewati otomatis.

---

### Langkah 2 — Download Model YuNet

Download file berikut dan letakkan di folder ini:

```
face_detection_yunet_2023mar.onnx
```

Link:  
https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

---

### Langkah 3 — Jalankan Evaluasi Utama

```
python evaluasi_bab4.py
```

Program ini:
- Membuat 4 kondisi citra sintetis (normal, gelap, terang, kontras rendah)
- Menerapkan 6 skenario perbaikan (baseline, brightness, kontras, HE, AHE, CLAHE)
- Mendeteksi wajah dengan MediaPipe dan YuNet
- Menghitung metrik: Precision, Recall, F1-score, IoU, Waktu inferensi
- Menyimpan visualisasi panel (citra + histogram + bounding box)

Output di `hasil_bab4/`:
```
hasil_bab4/
├── rekap_metrik.csv             ← Tabel 4.7 (semua kondisi digabung)
├── rekap_per_kondisi.csv        ← Metrik per kondisi × skenario
├── grafik_f1_skenario.png       ← Bar chart F1
└── visualisasi/
    ├── normal/
    │   ├── baseline/
    │   ├── brightness/
    │   ├── kontras/
    │   ├── HE/
    │   ├── AHE/
    │   └── CLAHE/
    ├── gelap/
    ├── terang/
    └── kontras_rendah/
```

---

### Langkah 4 — Analisis & Grafik Tambahan

```
python analisis_bab4.py
```

Membuat grafik dan tabel tambahan untuk Bab 4.8 & 4.9:

| File | Kegunaan |
|------|----------|
| `grafik_per_kondisi.png` | F1 per kondisi per skenario (Gambar 4.x) |
| `grafik_iou_heatmap.png` | Heatmap IoU semua kombinasi |
| `grafik_waktu.png` | Waktu inferensi per skenario |
| `tabel_rekap_md.txt` | Tabel Markdown siap tempel di paper |
| `tabel_rekap_latex.txt` | Tabel LaTeX siap tempel di Overleaf |

Juga mencetak analisis otomatis ke terminal: skenario terbaik/terburuk,
delta F1 terhadap baseline, dll.

---

## Parameter yang Bisa Diubah

Edit di bagian atas `evaluasi_bab4.py`:

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `GAMMA_GELAP` | 2.5 | γ untuk kondisi gelap |
| `GAMMA_TERANG` | 0.4 | γ untuk kondisi terang |
| `C_KONTRAS` | 0.4 | c untuk kontras rendah |
| `BRIGHTNESS_BETA` | 50 | nilai tambah brightness |
| `CONTRAST_ALPHA` | 1.5 | faktor pengali kontras |
| `CLAHE_CLIP` | 2.0 | clip limit CLAHE |
| `CLAHE_TILE` | (8,8) | ukuran tile CLAHE |
| `MP_CONFIDENCE` | 0.5 | threshold MediaPipe |
| `YUNET_SCORE` | 0.5 | score_threshold YuNet |
| `IOU_THRESHOLD` | 0.5 | ambang TP (0.3/0.4/0.5) |

---

## Instalasi Dependensi

```
pip install mediapipe opencv-python numpy matplotlib pandas
```
