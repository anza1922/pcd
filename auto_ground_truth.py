"""
auto_ground_truth.py
====================
Buat ground_truth.csv secara otomatis menggunakan YuNet
dengan score_threshold tinggi (0.85) agar hanya deteksi
yang sangat yakin yang dijadikan ground truth.

Hanya memproses folder orang_01 sampai orang_14
(bukan _perlu_dicek karena ada multi-wajah & ambigus).

Jalankan:
    python auto_ground_truth.py
"""

import os
import csv
import cv2
import numpy as np

INPUT_DIR   = os.path.join("dataset_wajah", "dataset_wajah_rapi")
OUTPUT_CSV  = "ground_truth.csv"
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
SCORE_THR   = 0.60   # ambang cukup untuk menangkap semua wajah
NMS_THR     = 0.3
TOP_K       = 5000
EXT         = (".jpg", ".jpeg", ".png")

# Inisialisasi YuNet
detector = cv2.FaceDetectorYN_create(
    YUNET_MODEL, "", (320, 320),
    score_threshold=SCORE_THR,
    nms_threshold=NMS_THR,
    top_k=TOP_K,
)

rows = []
total_wajah = 0
total_gambar = 0
gagal = []

for folder in sorted(os.listdir(INPUT_DIR)):
    folder_path = os.path.join(INPUT_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    if folder.startswith("_"):          # skip _perlu_dicek
        continue

    for fn in sorted(os.listdir(folder_path)):
        if not fn.lower().endswith(EXT):
            continue

        img_path = os.path.join(folder_path, fn)
        raw = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            continue

        H, W = img.shape[:2]
        detector.setInputSize((W, H))
        _, faces = detector.detect(img)

        total_gambar += 1
        n = 0
        if faces is not None:
            for face in faces:
                x = max(0, int(face[0]))
                y = max(0, int(face[1]))
                w = int(face[2])
                h = int(face[3])
                rows.append({
                    "img_path": img_path,
                    "x": x, "y": y, "w": w, "h": h
                })
                n += 1
                total_wajah += 1

        status = f"{n} wajah"
        if n == 0:
            gagal.append(img_path)
            status = "TIDAK TERDETEKSI"
        print(f"  {folder}/{fn:20s}  {status}")

# Simpan CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["img_path", "x", "y", "w", "h"])
    w.writeheader()
    w.writerows(rows)

print(f"\n{'='*55}")
print(f"Ground truth disimpan : {OUTPUT_CSV}")
print(f"Total gambar diproses : {total_gambar}")
print(f"Total wajah dianotasi : {total_wajah}")
if gagal:
    print(f"\nGambar tanpa deteksi ({len(gagal)}) -- perlu anotasi manual:")
    for p in gagal:
        print(f"  {p}")
    print("\nJalankan: python buat_ground_truth.py  untuk anotasi manual gambar di atas.")
else:
    print("Semua gambar berhasil dianotasi otomatis.")
print("="*55)
