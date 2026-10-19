"""
evaluasi_bab4.py
================
Pipeline lengkap untuk Bab 4 paper:
  "Analisis Pengaruh Perbaikan Citra terhadap Kinerja Deteksi Wajah
   Menggunakan MediaPipe dan YuNet"

Alur:
  1. Baca ground_truth.csv
  2. Buat 4 kondisi citra (normal, gelap, terang, kontras-rendah)
  3. Terapkan 6 skenario perbaikan citra per kondisi
  4. Deteksi wajah: MediaPipe (Tasks API) & YuNet (OpenCV)
  5. Hitung metrik: TP/FP/FN, Precision, Recall, F1, IoU rata-rata, waktu
  6. Simpan hasil CSV rekapitulasi (siap tempel ke Tabel 4.7)
  7. Hasilkan visualisasi per skenario (citra + histogram + bounding box)

Prasyarat:
    pip install mediapipe opencv-python numpy matplotlib pandas

Jalankan:
    python evaluasi_bab4.py
"""

import os
import csv
import time
import warnings
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # render tanpa layar (simpan ke file)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ─── Konfigurasi ──────────────────────────────────────────────────────────────

GT_CSV          = "ground_truth.csv"
INPUT_DIR       = os.path.join("dataset_wajah", "dataset_wajah_rapi")
OUT_ROOT        = "hasil_bab4"
REKAP_CSV       = os.path.join(OUT_ROOT, "rekap_metrik.csv")
IOU_THRESHOLD   = 0.5       # ambang TP (paper memakai 0.3, 0.4, 0.5)

# Parameter kondisi sintetis
GAMMA_GELAP     = 2.5       # f^gamma  → gelap
GAMMA_TERANG    = 0.4       # f^gamma  → terang
C_KONTRAS       = 0.4       # g = 0.5 + c*(f-0.5) → kontras rendah

# Parameter perbaikan citra
BRIGHTNESS_BETA = 50        # g = f + beta  (tambah kecerahan)
CONTRAST_ALPHA  = 1.5       # g = alpha*f   (naikkan kontras)
CLAHE_CLIP      = 2.0       # clip limit CLAHE (Naufal dkk., 2026: 1.2–2.0)
CLAHE_TILE      = (8, 8)    # ukuran tile CLAHE
AHE_TILE        = (8, 8)    # ukuran tile AHE (clip limit sangat besar = full AHE)

# MediaPipe
MP_CONFIDENCE   = 0.5
MP_MODEL_SEL    = 0         # 0 = jarak dekat

# YuNet
YUNET_SCORE     = 0.5
YUNET_NMS       = 0.3
YUNET_TOPK      = 5000
YUNET_MODEL     = "face_detection_yunet_2023mar.onnx"   # harus ada di folder ini

N_WARMUP        = 3         # pemanggilan pemanasan sebelum ukur waktu
N_REPEAT        = 3         # pengulangan pengukuran waktu per citra

EXT = (".jpg", ".jpeg", ".png")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def baca_img(path: str) -> np.ndarray | None:
    """Baca gambar dengan dukungan path Unicode."""
    raw = np.fromfile(path, np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def simpan_img(path: str, img: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        buf.tofile(path)


def hitung_iou(boxA, boxB) -> float:
    """boxA/boxB = (x, y, w, h). Kembalikan IoU [0..1]."""
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = boxA[0] + boxA[2], boxA[1] + boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = boxB[0] + boxB[2], boxB[1] + boxB[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def match_detections(gt_list, pred_list, iou_thr):
    """
    Greedy matching: setiap pred dipasangkan ke GT terbaik (IoU >= iou_thr).
    Kembalikan (tp, fp, fn, iou_list).
    """
    matched_gt = set()
    iou_vals   = []
    tp = fp = 0
    for pred in pred_list:
        best_iou, best_idx = 0.0, -1
        for j, gt in enumerate(gt_list):
            if j in matched_gt:
                continue
            iou = hitung_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_idx = iou, j
        if best_iou >= iou_thr:
            tp += 1
            matched_gt.add(best_idx)
            iou_vals.append(best_iou)
        else:
            fp += 1
    fn = len(gt_list) - tp
    return tp, fp, fn, iou_vals


# ─── Kondisi Citra ────────────────────────────────────────────────────────────

def kondisi_normal(img: np.ndarray) -> np.ndarray:
    return img.copy()


def kondisi_gelap(img: np.ndarray) -> np.ndarray:
    """Koreksi gamma dengan gamma > 1  → gelap."""
    norm = img.astype(np.float32) / 255.0
    out  = np.power(norm, GAMMA_GELAP)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def kondisi_terang(img: np.ndarray) -> np.ndarray:
    """Koreksi gamma dengan gamma < 1  → terang."""
    norm = img.astype(np.float32) / 255.0
    out  = np.power(norm, GAMMA_TERANG)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def kondisi_kontras_rendah(img: np.ndarray) -> np.ndarray:
    """Penyempitan rentang intensitas: g = 0.5 + c*(f - 0.5)."""
    norm = img.astype(np.float32) / 255.0
    out  = 0.5 + C_KONTRAS * (norm - 0.5)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


KONDISI = {
    "normal"        : kondisi_normal,
    "gelap"         : kondisi_gelap,
    "terang"        : kondisi_terang,
    "kontras_rendah": kondisi_kontras_rendah,
}

# ─── Perbaikan Citra ──────────────────────────────────────────────────────────

def terapkan_brightness(img: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.int32) + BRIGHTNESS_BETA, 0, 255).astype(np.uint8)


def terapkan_kontras(img: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.float32) * CONTRAST_ALPHA, 0, 255).astype(np.uint8)


def _eq_luminans(img: np.ndarray, fn_eq) -> np.ndarray:
    """Terapkan fn_eq (fungsi equalization) hanya pada kanal L di ruang LAB."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    L_eq = fn_eq(L)
    lab_eq = cv2.merge([L_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)


def terapkan_he(img: np.ndarray) -> np.ndarray:
    return _eq_luminans(img, cv2.equalizeHist)


def terapkan_ahe(img: np.ndarray) -> np.ndarray:
    ahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=AHE_TILE)
    return _eq_luminans(img, ahe.apply)


def terapkan_clahe(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    return _eq_luminans(img, clahe.apply)


SKENARIO = {
    "baseline"   : lambda img: img.copy(),
    "brightness" : terapkan_brightness,
    "kontras"    : terapkan_kontras,
    "HE"         : terapkan_he,
    "AHE"        : terapkan_ahe,
    "CLAHE"      : terapkan_clahe,
}

# ─── Detektor ─────────────────────────────────────────────────────────────────

class DetectorMediaPipe:
    def __init__(self):
        try:
            import mediapipe as mp
        except ImportError as e:
            raise ImportError("Install mediapipe: pip install mediapipe") from e

        # Pakai Solutions API (bekerja di mediapipe 0.10.x)
        self._mp = mp
        self.model = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=MP_CONFIDENCE,
            model_selection=MP_MODEL_SEL,
        )

    def detect(self, img_bgr: np.ndarray):
        """Kembalikan list (x, y, w, h, confidence)."""
        results = []
        H, W = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.model.process(rgb)
        if res.detections:
            for d in res.detections:
                bb = d.location_data.relative_bounding_box
                (conf,) = d.score
                x = max(0, int(bb.xmin * W))
                y = max(0, int(bb.ymin * H))
                w = int(bb.width  * W)
                h = int(bb.height * H)
                results.append((x, y, w, h, float(conf)))
        return results


class DetectorYuNet:
    def __init__(self):
        # Cari model .onnx
        search_dirs = [
            ".",
            os.path.join(os.path.expanduser("~"), ".deepface", "weights"),
        ]
        self.model_path = None
        for d in search_dirs:
            p = os.path.join(d, YUNET_MODEL)
            if os.path.exists(p):
                self.model_path = p
                break
        if self.model_path is None:
            raise FileNotFoundError(
                f"Model YuNet '{YUNET_MODEL}' tidak ditemukan.\n"
                "Download dari:\n"
                "  https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_detection_yunet/face_detection_yunet_2023mar.onnx\n"
                "dan letakkan di folder yang sama dengan script ini."
            )
        self.detector = cv2.FaceDetectorYN_create(
            self.model_path, "", (320, 320),
            score_threshold=YUNET_SCORE,
            nms_threshold=YUNET_NMS,
            top_k=YUNET_TOPK,
        )

    def detect(self, img_bgr: np.ndarray):
        """Kembalikan list (x, y, w, h, confidence)."""
        H, W = img_bgr.shape[:2]
        r = 1.0
        img = img_bgr
        if H > 640 or W > 640:
            r = 640.0 / max(H, W)
            img = cv2.resize(img_bgr, (int(W * r), int(H * r)))
            H2, W2 = img.shape[:2]
        else:
            W2, H2 = W, H

        self.detector.setInputSize((W2, H2))
        _, faces = self.detector.detect(img)
        results = []
        if faces is None:
            return results
        for face in faces:
            x  = max(0, int(face[0] / r))
            y  = max(0, int(face[1] / r))
            w  = int(face[2] / r)
            h  = int(face[3] / r)
            conf = float(face[-1])
            results.append((x, y, w, h, conf))
        return results


# ─── Visualisasi ──────────────────────────────────────────────────────────────

WARNA_GT   = (0,   255,  0)   # hijau
WARNA_MP   = (255, 165,  0)   # oranye
WARNA_YN   = (0,   0,   255)  # merah
LABEL_SIZE = 0.45


def gambar_kotak(img, gt_list, mp_list, yn_list):
    out = img.copy()
    for (x, y, w, h) in gt_list:
        cv2.rectangle(out, (x, y), (x+w, y+h), WARNA_GT, 2)
        cv2.putText(out, "GT", (x, max(y-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, WARNA_GT, 1)
    for (x, y, w, h, c) in mp_list:
        cv2.rectangle(out, (x, y), (x+w, y+h), WARNA_MP, 2)
        cv2.putText(out, f"MP {c:.2f}", (x, max(y-14, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, WARNA_MP, 1)
    for (x, y, w, h, c) in yn_list:
        cv2.rectangle(out, (x, y), (x+w, y+h), WARNA_YN, 2)
        cv2.putText(out, f"YN {c:.2f}", (x+w//2, max(y-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, WARNA_YN, 1)
    return out


def simpan_visualisasi(img_asli, img_perbaikan, gt_list, mp_list, yn_list,
                        out_path: str, judul: str):
    """
    Simpan panel 2x2:
      [0,0] citra asli          [0,1] citra hasil perbaikan + bounding box
      [1,0] histogram asli      [1,1] histogram perbaikan
    """
    kotak_disp = gambar_kotak(img_perbaikan, gt_list, mp_list, yn_list)

    # OpenCV BGR → RGB untuk matplotlib
    def bgr2rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    fig.suptitle(judul, fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Citra asli
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(bgr2rgb(img_asli))
    ax0.set_title("Citra Asli")
    ax0.axis("off")

    # Citra perbaikan + bounding box
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(bgr2rgb(kotak_disp))
    ax1.set_title("Hasil Perbaikan + Deteksi\n"
                  "  GT=hijau  MediaPipe=oranye  YuNet=merah")
    ax1.axis("off")

    # Histogram asli
    ax2 = fig.add_subplot(gs[1, 0])
    gray_asli = cv2.cvtColor(img_asli, cv2.COLOR_BGR2GRAY)
    ax2.hist(gray_asli.ravel(), bins=256, range=(0, 256),
             color="steelblue", alpha=0.8)
    ax2.set_title("Histogram Citra Asli")
    ax2.set_xlabel("Intensitas")
    ax2.set_ylabel("Frekuensi")
    ax2.set_xlim(0, 255)

    # Histogram perbaikan
    ax3 = fig.add_subplot(gs[1, 1])
    gray_prb = cv2.cvtColor(img_perbaikan, cv2.COLOR_BGR2GRAY)
    ax3.hist(gray_prb.ravel(), bins=256, range=(0, 256),
             color="tomato", alpha=0.8)
    ax3.set_title("Histogram Citra Perbaikan")
    ax3.set_xlabel("Intensitas")
    ax3.set_ylabel("Frekuensi")
    ax3.set_xlim(0, 255)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    # 1. Baca ground truth
    if not os.path.exists(GT_CSV):
        print(f"[ERROR] File ground truth '{GT_CSV}' tidak ditemukan.")
        print("  Jalankan dulu: python buat_ground_truth.py")
        return

    gt_data: dict[str, list] = {}
    with open(GT_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row["img_path"].replace("\\", "/")
            gt_data.setdefault(key, []).append(
                (int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"]))
            )

    # Kumpulkan path gambar (hanya folder orang_*)
    img_paths = []
    for folder in sorted(os.listdir(INPUT_DIR)):
        fp = os.path.join(INPUT_DIR, folder)
        if not os.path.isdir(fp) or folder.startswith("_"):
            continue
        for fn in sorted(os.listdir(fp)):
            if fn.lower().endswith(EXT):
                img_paths.append(os.path.join(fp, fn))

    total = len(img_paths)
    print(f"Dataset: {total} gambar | GT: {sum(len(v) for v in gt_data.values())} wajah")

    # 2. Inisialisasi detektor
    print("Inisialisasi MediaPipe...", end=" ", flush=True)
    try:
        det_mp = DetectorMediaPipe()
        print("OK")
    except Exception as e:
        print(f"GAGAL ({e})")
        det_mp = None

    print("Inisialisasi YuNet...", end=" ", flush=True)
    try:
        det_yn = DetectorYuNet()
        print("OK")
    except Exception as e:
        print(f"GAGAL ({e})")
        det_yn = None

    if det_mp is None and det_yn is None:
        print("[ERROR] Kedua detektor gagal diinisialisasi. Cek instalasi.")
        return

    # 3. Pemanasan detektor
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    if det_mp:
        for _ in range(N_WARMUP):
            det_mp.detect(dummy)
    if det_yn:
        for _ in range(N_WARMUP):
            det_yn.detect(dummy)

    # 4. Kumpulkan metrik
    # Struktur: rekap[(detektor, kondisi, skenario)] = {"tp":0, "fp":0, "fn":0, "iou":[], "ms":[]}
    rekap = {}
    for det_name in (["mediapipe"] if det_mp else []) + (["yunet"] if det_yn else []):
        for kond in KONDISI:
            for sk in SKENARIO:
                rekap[(det_name, kond, sk)] = {"tp": 0, "fp": 0, "fn": 0,
                                                "iou": [], "ms": []}

    # Juga kumpulkan metrik agregat per skenario (semua kondisi digabung)
    rekap_agg = {}
    for det_name in (["mediapipe"] if det_mp else []) + (["yunet"] if det_yn else []):
        for sk in SKENARIO:
            rekap_agg[(det_name, sk)] = {"tp": 0, "fp": 0, "fn": 0,
                                          "iou": [], "ms": []}

    print(f"\nMemulai evaluasi ({len(KONDISI)} kondisi × {len(SKENARIO)} skenario × "
          f"{len([d for d in [det_mp, det_yn] if d])} detektor)...\n")

    # Tentukan contoh gambar per skenario untuk disimpan visualisasi (maks 3)
    contoh_per_sk = {}

    for i, img_path in enumerate(img_paths):
        norm_key = img_path.replace("\\", "/")
        gt_list  = gt_data.get(norm_key, gt_data.get(img_path, []))

        if not gt_list:
            continue   # lewati gambar tanpa anotasi

        img_asli = baca_img(img_path)
        if img_asli is None:
            continue

        folder_rel = os.path.relpath(
            os.path.dirname(img_path), INPUT_DIR).replace("\\", "/")
        fn_base = os.path.splitext(os.path.basename(img_path))[0]

        print(f"  [{i+1:3d}/{total}] {folder_rel}/{fn_base}")

        for kond_nama, fn_kond in KONDISI.items():
            img_kondisi = fn_kond(img_asli)

            for sk_nama, fn_sk in SKENARIO.items():
                img_prb = fn_sk(img_kondisi)

                # Deteksi MediaPipe
                if det_mp:
                    times_mp = []
                    for _ in range(N_REPEAT):
                        t0 = time.perf_counter()
                        mp_boxes = det_mp.detect(img_prb)
                        times_mp.append((time.perf_counter() - t0) * 1000)
                    ms_mp = float(np.median(times_mp))
                    pred_mp = [(x, y, w, h) for (x, y, w, h, _) in mp_boxes]
                    tp, fp, fn, iou_v = match_detections(gt_list, pred_mp, IOU_THRESHOLD)
                    rekap[("mediapipe", kond_nama, sk_nama)]["tp"] += tp
                    rekap[("mediapipe", kond_nama, sk_nama)]["fp"] += fp
                    rekap[("mediapipe", kond_nama, sk_nama)]["fn"] += fn
                    rekap[("mediapipe", kond_nama, sk_nama)]["iou"].extend(iou_v)
                    rekap[("mediapipe", kond_nama, sk_nama)]["ms"].append(ms_mp)
                    rekap_agg[("mediapipe", sk_nama)]["tp"] += tp
                    rekap_agg[("mediapipe", sk_nama)]["fp"] += fp
                    rekap_agg[("mediapipe", sk_nama)]["fn"] += fn
                    rekap_agg[("mediapipe", sk_nama)]["iou"].extend(iou_v)
                    rekap_agg[("mediapipe", sk_nama)]["ms"].append(ms_mp)
                else:
                    mp_boxes = []

                # Deteksi YuNet
                if det_yn:
                    times_yn = []
                    for _ in range(N_REPEAT):
                        t0 = time.perf_counter()
                        yn_boxes = det_yn.detect(img_prb)
                        times_yn.append((time.perf_counter() - t0) * 1000)
                    ms_yn = float(np.median(times_yn))
                    pred_yn = [(x, y, w, h) for (x, y, w, h, _) in yn_boxes]
                    tp, fp, fn, iou_v = match_detections(gt_list, pred_yn, IOU_THRESHOLD)
                    rekap[("yunet", kond_nama, sk_nama)]["tp"] += tp
                    rekap[("yunet", kond_nama, sk_nama)]["fp"] += fp
                    rekap[("yunet", kond_nama, sk_nama)]["fn"] += fn
                    rekap[("yunet", kond_nama, sk_nama)]["iou"].extend(iou_v)
                    rekap[("yunet", kond_nama, sk_nama)]["ms"].append(ms_yn)
                    rekap_agg[("yunet", sk_nama)]["tp"] += tp
                    rekap_agg[("yunet", sk_nama)]["fp"] += fp
                    rekap_agg[("yunet", sk_nama)]["fn"] += fn
                    rekap_agg[("yunet", sk_nama)]["iou"].extend(iou_v)
                    rekap_agg[("yunet", sk_nama)]["ms"].append(ms_yn)
                else:
                    yn_boxes = []

                # Simpan visualisasi untuk beberapa contoh per skenario
                vis_key = (kond_nama, sk_nama)
                if contoh_per_sk.get(vis_key, 0) < 3:
                    vis_dir = os.path.join(OUT_ROOT, "visualisasi",
                                            kond_nama, sk_nama)
                    vis_path = os.path.join(vis_dir,
                                             f"{folder_rel.replace('/','_')}_{fn_base}.jpg")
                    judul = (f"Kondisi: {kond_nama} | Skenario: {sk_nama}\n"
                             f"{folder_rel}/{fn_base}.jpg")
                    simpan_visualisasi(img_kondisi, img_prb, gt_list,
                                       mp_boxes, yn_boxes, vis_path, judul)
                    contoh_per_sk[vis_key] = contoh_per_sk.get(vis_key, 0) + 1

    # 5. Hitung metrik akhir
    def hitung_metrik(d):
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        iou_m = float(np.mean(d["iou"])) if d["iou"] else 0.0
        ms_m  = float(np.mean(d["ms"])) if d["ms"] else 0.0
        return prec, rec, f1, iou_m, ms_m, tp, fp, fn

    # 6. Tulis CSV per kondisi
    rows_per_kond = []
    for (det, kond, sk), d in sorted(rekap.items()):
        prec, rec, f1, iou_m, ms_m, tp, fp, fn = hitung_metrik(d)
        rows_per_kond.append({
            "detektor": det, "kondisi": kond, "skenario": sk,
            "TP": tp, "FP": fp, "FN": fn,
            "precision": f"{prec:.4f}", "recall": f"{rec:.4f}",
            "f1_score": f"{f1:.4f}", "iou_rata": f"{iou_m:.4f}",
            "waktu_ms": f"{ms_m:.2f}",
        })

    os.makedirs(OUT_ROOT, exist_ok=True)
    df_kond = pd.DataFrame(rows_per_kond)
    df_kond.to_csv(os.path.join(OUT_ROOT, "rekap_per_kondisi.csv"),
                   index=False, encoding="utf-8-sig")

    # 7. Tulis CSV agregat (Tabel 4.7 paper)
    rows_agg = []
    for (det, sk), d in sorted(rekap_agg.items()):
        prec, rec, f1, iou_m, ms_m, tp, fp, fn = hitung_metrik(d)
        rows_agg.append({
            "detektor": det, "skenario": sk,
            "TP": tp, "FP": fp, "FN": fn,
            "precision": f"{prec:.4f}", "recall": f"{rec:.4f}",
            "f1_score": f"{f1:.4f}", "iou_rata": f"{iou_m:.4f}",
            "waktu_ms": f"{ms_m:.2f}",
        })
    df_agg = pd.DataFrame(rows_agg)
    df_agg.to_csv(REKAP_CSV, index=False, encoding="utf-8-sig")

    # 8. Cetak ringkasan Tabel 4.7
    print("\n" + "="*80)
    print("TABEL 4.7 — Rekapitulasi Seluruh Skenario (semua kondisi, mikro)")
    print("="*80)
    hdr = f"{'Detektor':<12} {'Skenario':<12} {'Prec':>7} {'Rec':>7} {'F1':>7} {'IoU':>7} {'ms':>7}"
    print(hdr)
    print("-"*80)
    for r in rows_agg:
        print(f"{r['detektor']:<12} {r['skenario']:<12} "
              f"{r['precision']:>7} {r['recall']:>7} {r['f1_score']:>7} "
              f"{r['iou_rata']:>7} {r['waktu_ms']:>7}")
    print("="*80)
    print(f"\nHasil lengkap disimpan di folder '{OUT_ROOT}/'")
    print(f"  {REKAP_CSV}")
    print(f"  {os.path.join(OUT_ROOT, 'rekap_per_kondisi.csv')}")
    print(f"  {os.path.join(OUT_ROOT, 'visualisasi', '<kondisi>', '<skenario>', '*.jpg')}")

    # 9. Plot bar chart F1 per skenario (MediaPipe vs YuNet)
    _plot_rekap(df_agg)


def _plot_rekap(df: "pd.DataFrame"):
    """Bar chart F1-score MediaPipe vs YuNet per skenario."""
    skenario_urut = list(SKENARIO.keys())
    mp_f1 = []
    yn_f1 = []
    for sk in skenario_urut:
        row_mp = df[(df["detektor"] == "mediapipe") & (df["skenario"] == sk)]
        row_yn = df[(df["detektor"] == "yunet")     & (df["skenario"] == sk)]
        mp_f1.append(float(row_mp["f1_score"].values[0]) if len(row_mp) else 0)
        yn_f1.append(float(row_yn["f1_score"].values[0]) if len(row_yn) else 0)

    x   = np.arange(len(skenario_urut))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, mp_f1, w, label="MediaPipe", color="steelblue", alpha=0.85)
    b2 = ax.bar(x + w/2, yn_f1, w, label="YuNet",     color="tomato",    alpha=0.85)
    ax.set_xlabel("Skenario Perbaikan Citra")
    ax.set_ylabel("F1-score")
    ax.set_title("Perbandingan F1-score MediaPipe vs YuNet\n(Agregat semua kondisi)")
    ax.set_xticks(x)
    ax.set_xticklabels(skenario_urut, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(OUT_ROOT, "grafik_f1_skenario.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  {out_path}")

    # Plot juga per kondisi
    for kond in KONDISI:
        mp_f1k, yn_f1k = [], []
        for sk in skenario_urut:
            # baca dari rekap_per_kondisi jika sudah ada
            pass   # cukup dengan grafik agregat untuk versi ini

    print("\nGrafik disimpan.")


if __name__ == "__main__":
    main()
