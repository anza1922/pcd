"""
buat_ground_truth.py
====================
Tool anotasi bounding box manual untuk membuat ground_truth.csv.

Cara pakai:
    python buat_ground_truth.py

Kontrol:
    - Klik-drag  : gambar kotak bounding box baru
    - R          : reset kotak pada gambar ini
    - ENTER/S    : simpan kotak dan lanjut ke gambar berikutnya
    - Q/ESC      : lewati gambar (tidak ada wajah / abaikan)
    - Z          : hapus kotak terakhir

Output:
    ground_truth.csv  --  kolom: img_path, x, y, w, h
"""

import os
import csv
import cv2
import numpy as np

INPUT_DIR = os.path.join("dataset_wajah", "dataset_wajah_rapi")
OUTPUT_CSV = "ground_truth.csv"
EXT = (".jpg", ".jpeg", ".png")

# Kumpulkan semua path gambar (hanya folder orang_*)
img_paths = []
for folder in sorted(os.listdir(INPUT_DIR)):
    folder_path = os.path.join(INPUT_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    if folder.startswith("_"):          # lewati _perlu_dicek
        continue
    for fn in sorted(os.listdir(folder_path)):
        if fn.lower().endswith(EXT):
            img_paths.append(os.path.join(folder_path, fn))

# Muat anotasi yang sudah ada agar bisa dilanjutkan
existing = {}
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row["img_path"]
            if key not in existing:
                existing[key] = []
            existing[key].append((int(row["x"]), int(row["y"]),
                                   int(row["w"]), int(row["h"])))

# ── State klik-drag ──────────────────────────────────────────────────────────
drawing = False
ix, iy = -1, -1
boxes_current = []
canvas = None
orig = None


def draw_boxes(img, boxes, temp_box=None):
    disp = img.copy()
    for (bx, by, bw, bh) in boxes:
        cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(disp, f"({bx},{by},{bw}x{bh})",
                    (bx, max(by - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1)
    if temp_box:
        x1, y1, x2, y2 = temp_box
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 165, 255), 2)
    return disp


def mouse_cb(event, x, y, flags, param):
    global drawing, ix, iy, boxes_current, canvas, orig
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            canvas = draw_boxes(orig, boxes_current, (ix, iy, x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        bw, bh = x2 - x1, y2 - y1
        if bw > 5 and bh > 5:
            boxes_current.append((x1, y1, bw, bh))
        canvas = draw_boxes(orig, boxes_current)


# ── Loop anotasi ─────────────────────────────────────────────────────────────
WIN = "Anotasi Ground Truth -- ENTER=simpan  Q=lewati  R=reset  Z=hapus terakhir"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WIN, mouse_cb)

all_rows = []
# Sertakan anotasi yang sudah ada dulu
already_done = set()
for path, blist in existing.items():
    for (bx, by, bw, bh) in blist:
        all_rows.append({"img_path": path, "x": bx, "y": by, "w": bw, "h": bh})
    already_done.add(path)

for img_path in img_paths:
    norm_path = img_path.replace("\\", "/")
    if norm_path in already_done or img_path in already_done:
        print(f"[SKIP already done] {img_path}")
        continue

    raw = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        continue

    # Resize tampilan agar muat di layar (maks 900px)
    H, W = raw.shape[:2]
    scale = min(900 / max(H, W), 1.0)
    disp_w, disp_h = int(W * scale), int(H * scale)
    orig = cv2.resize(raw, (disp_w, disp_h))
    canvas = orig.copy()
    boxes_current = []

    print(f"\n[{img_paths.index(img_path)+1}/{len(img_paths)}] {img_path}  ({W}x{H})")
    print("  Klik-drag untuk gambar kotak. ENTER=simpan, Q=lewati, R=reset, Z=hapus terakhir.")

    while True:
        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord('s'), ord('S')):   # ENTER atau S
            # Skala balik ke koordinat asli
            for (bx, by, bw, bh) in boxes_current:
                rx = int(bx / scale)
                ry = int(by / scale)
                rw = int(bw / scale)
                rh = int(bh / scale)
                all_rows.append({"img_path": img_path, "x": rx, "y": ry, "w": rw, "h": rh})
            print(f"  -> Disimpan {len(boxes_current)} kotak.")
            break
        elif key in (ord('q'), ord('Q'), 27):  # Q atau ESC = lewati
            print("  -> Dilewati.")
            break
        elif key in (ord('r'), ord('R')):
            boxes_current = []
            canvas = draw_boxes(orig, boxes_current)
        elif key in (ord('z'), ord('Z')):
            if boxes_current:
                boxes_current.pop()
                canvas = draw_boxes(orig, boxes_current)

cv2.destroyAllWindows()

# Simpan CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["img_path", "x", "y", "w", "h"])
    w.writeheader()
    w.writerows(all_rows)

print(f"\nGround truth disimpan ke {OUTPUT_CSV}  ({len(all_rows)} baris)")
