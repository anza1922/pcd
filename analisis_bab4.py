"""
analisis_bab4.py
================
Buat grafik dan tabel analisis dari hasil evaluasi_bab4.py.
Jalankan SETELAH evaluasi_bab4.py selesai.

Output:
  hasil_bab4/grafik_per_kondisi.png    -- F1 per kondisi per skenario
  hasil_bab4/grafik_iou_heatmap.png    -- heatmap IoU semua kombinasi
  hasil_bab4/grafik_waktu.png          -- waktu inferensi
  hasil_bab4/tabel_rekap_latex.txt     -- tabel LaTeX siap tempel
  hasil_bab4/tabel_rekap_md.txt        -- tabel Markdown siap tempel
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_ROOT   = "hasil_bab4"
CSV_AGG    = os.path.join(OUT_ROOT, "rekap_metrik.csv")
CSV_KOND   = os.path.join(OUT_ROOT, "rekap_per_kondisi.csv")

KONDISI_URUT  = ["normal", "gelap", "terang", "kontras_rendah"]
SKENARIO_URUT = ["baseline", "brightness", "kontras", "HE", "AHE", "CLAHE"]
DETEKTOR      = ["mediapipe", "yunet"]

KONDISI_LABEL = {
    "normal"        : "Normal",
    "gelap"         : "Gelap",
    "terang"        : "Terang",
    "kontras_rendah": "Kontras Rendah",
}
SKENARIO_LABEL = {
    "baseline"  : "Baseline",
    "brightness": "Brightness",
    "kontras"   : "Kontras",
    "HE"        : "HE",
    "AHE"       : "AHE",
    "CLAHE"     : "CLAHE",
}


def cek_file():
    for p in [CSV_AGG, CSV_KOND]:
        if not os.path.exists(p):
            print(f"[ERROR] File tidak ditemukan: {p}")
            print("  Jalankan dulu: python evaluasi_bab4.py")
            return False
    return True


def get_val(df, detektor, kondisi_or_none, skenario, kolom):
    if kondisi_or_none:
        row = df[(df["detektor"] == detektor) &
                 (df["kondisi"]  == kondisi_or_none) &
                 (df["skenario"] == skenario)]
    else:
        row = df[(df["detektor"] == detektor) &
                 (df["skenario"] == skenario)]
    if len(row) == 0:
        return 0.0
    return float(row[kolom].values[0])


# ── 1. Grafik F1 per kondisi ──────────────────────────────────────────────────

def plot_f1_per_kondisi(df_kond):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    fig.suptitle("F1-score per Kondisi Citra: MediaPipe vs YuNet", fontsize=13)

    x  = np.arange(len(SKENARIO_URUT))
    bw = 0.35

    for idx, kond in enumerate(KONDISI_URUT):
        ax = axes[idx // 2][idx % 2]
        mp_f1 = [get_val(df_kond, "mediapipe", kond, sk, "f1_score")
                 for sk in SKENARIO_URUT]
        yn_f1 = [get_val(df_kond, "yunet",     kond, sk, "f1_score")
                 for sk in SKENARIO_URUT]
        b1 = ax.bar(x - bw/2, mp_f1, bw, label="MediaPipe",
                    color="steelblue", alpha=0.85)
        b2 = ax.bar(x + bw/2, yn_f1, bw, label="YuNet",
                    color="tomato", alpha=0.85)
        ax.set_title(KONDISI_LABEL[kond])
        ax.set_xticks(x)
        ax.set_xticklabels([SKENARIO_LABEL[s] for s in SKENARIO_URUT],
                           rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1-score")
        ax.legend(fontsize=8)
        ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=7)
        ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=7)
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out = os.path.join(OUT_ROOT, "grafik_per_kondisi.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Disimpan: {out}")


# ── 2. Heatmap IoU ────────────────────────────────────────────────────────────

def plot_heatmap_iou(df_kond):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Heatmap IoU Rata-rata", fontsize=13)

    for idx, det in enumerate(DETEKTOR):
        mat = np.zeros((len(KONDISI_URUT), len(SKENARIO_URUT)))
        for ci, kond in enumerate(KONDISI_URUT):
            for si, sk in enumerate(SKENARIO_URUT):
                mat[ci, si] = get_val(df_kond, det, kond, sk, "iou_rata")

        ax = axes[idx]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
        ax.set_xticks(range(len(SKENARIO_URUT)))
        ax.set_xticklabels([SKENARIO_LABEL[s] for s in SKENARIO_URUT],
                           rotation=25, ha="right", fontsize=9)
        ax.set_yticks(range(len(KONDISI_URUT)))
        ax.set_yticklabels([KONDISI_LABEL[k] for k in KONDISI_URUT], fontsize=9)
        ax.set_title(det.capitalize())
        fig.colorbar(im, ax=ax, label="IoU rata-rata")

        # Anotasi nilai di setiap sel
        for ci in range(len(KONDISI_URUT)):
            for si in range(len(SKENARIO_URUT)):
                val = mat[ci, si]
                color = "white" if val > 0.6 else "black"
                ax.text(si, ci, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    fig.tight_layout()
    out = os.path.join(OUT_ROOT, "grafik_iou_heatmap.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Disimpan: {out}")


# ── 3. Grafik Waktu Inferensi ─────────────────────────────────────────────────

def plot_waktu(df_agg):
    fig, ax = plt.subplots(figsize=(10, 5))
    x  = np.arange(len(SKENARIO_URUT))
    bw = 0.35

    mp_ms = [get_val(df_agg, "mediapipe", None, sk, "waktu_ms")
             for sk in SKENARIO_URUT]
    yn_ms = [get_val(df_agg, "yunet",     None, sk, "waktu_ms")
             for sk in SKENARIO_URUT]

    b1 = ax.bar(x - bw/2, mp_ms, bw, label="MediaPipe",
                color="steelblue", alpha=0.85)
    b2 = ax.bar(x + bw/2, yn_ms, bw, label="YuNet",
                color="tomato", alpha=0.85)
    ax.set_xlabel("Skenario")
    ax.set_ylabel("Waktu Inferensi Rata-rata (ms)")
    ax.set_title("Perbandingan Waktu Inferensi MediaPipe vs YuNet")
    ax.set_xticks(x)
    ax.set_xticklabels([SKENARIO_LABEL[s] for s in SKENARIO_URUT],
                       rotation=15, ha="right")
    ax.legend()
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(OUT_ROOT, "grafik_waktu.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Disimpan: {out}")


# ── 4. Tabel Rekap (LaTeX & Markdown) ────────────────────────────────────────

def buat_tabel(df_agg):
    """Tabel 4.7: Detektor x Skenario, semua kondisi."""

    # Markdown
    header_md = ("| Detektor   | Skenario   | Precision | Recall | F1-score | "
                 "IoU rata-rata | Waktu (ms) |")
    sep_md    = ("|------------|------------|-----------|--------|----------|"
                 "--------------|------------|")
    rows_md   = [header_md, sep_md]

    # LaTeX
    rows_latex = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Rekapitulasi Kinerja Deteksi Wajah Seluruh Skenario}",
        r"\label{tab:rekap}",
        r"\begin{tabular}{llccccc}",
        r"\hline",
        r"Detektor & Skenario & Precision & Recall & F1-score & "
        r"IoU rata-rata & Waktu (ms) \\",
        r"\hline",
    ]

    prev_det = None
    for det in DETEKTOR:
        for sk in SKENARIO_URUT:
            row = df_agg[(df_agg["detektor"] == det) & (df_agg["skenario"] == sk)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            det_label = det.capitalize() if det != prev_det else ""
            prev_det  = det

            rows_md.append(
                f"| {det_label:<10} | {SKENARIO_LABEL[sk]:<10} | "
                f"{float(r['precision']):.4f}    | "
                f"{float(r['recall']):.4f} | "
                f"{float(r['f1_score']):.4f}   | "
                f"{float(r['iou_rata']):.4f}        | "
                f"{float(r['waktu_ms']):.2f}       |"
            )
            rows_latex.append(
                f"{det_label} & {SKENARIO_LABEL[sk]} & "
                f"{float(r['precision']):.4f} & "
                f"{float(r['recall']):.4f} & "
                f"{float(r['f1_score']):.4f} & "
                f"{float(r['iou_rata']):.4f} & "
                f"{float(r['waktu_ms']):.2f} \\\\"
            )
        rows_latex.append(r"\hline")

    rows_latex += [
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_md = os.path.join(OUT_ROOT, "tabel_rekap_md.txt")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(rows_md))
    print(f"Disimpan: {out_md}")

    out_tex = os.path.join(OUT_ROOT, "tabel_rekap_latex.txt")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(rows_latex))
    print(f"Disimpan: {out_tex}")


# ── 5. Analisis Teks (Bab 4.8 draft) ─────────────────────────────────────────

def analisis_teks(df_agg, df_kond):
    """Cetak temuan otomatis untuk membantu menulis Bab 4.8."""
    print("\n" + "="*70)
    print("ANALISIS OTOMATIS untuk Bab 4.8")
    print("="*70)

    for det in DETEKTOR:
        print(f"\n>>> {det.upper()}")
        sub = df_agg[df_agg["detektor"] == det].copy()
        sub["f1_float"] = sub["f1_score"].astype(float)
        best  = sub.loc[sub["f1_float"].idxmax()]
        worst = sub.loc[sub["f1_float"].idxmin()]
        base  = sub[sub["skenario"] == "baseline"]["f1_float"].values[0]

        print(f"  Baseline F1      : {base:.4f}")
        print(f"  Skenario terbaik : {best['skenario']} "
              f"(F1={float(best['f1_score']):.4f}, "
              f"delta={float(best['f1_score'])-base:+.4f})")
        print(f"  Skenario terburuk: {worst['skenario']} "
              f"(F1={float(worst['f1_score']):.4f}, "
              f"delta={float(worst['f1_score'])-base:+.4f})")

        for sk in SKENARIO_URUT:
            f1 = float(sub[sub["skenario"] == sk]["f1_score"].values[0])
            delta = f1 - base
            tanda = "naik" if delta > 0.005 else ("turun" if delta < -0.005 else "sama")
            print(f"    {SKENARIO_LABEL[sk]:<12}: F1={f1:.4f}  {tanda}{delta:+.4f}")

    print("\n--- Per Kondisi ---")
    for kond in KONDISI_URUT:
        print(f"\n  Kondisi: {KONDISI_LABEL[kond]}")
        for det in DETEKTOR:
            best_sk, best_f1 = None, -1
            for sk in SKENARIO_URUT:
                f1 = get_val(df_kond, det, kond, sk, "f1_score")
                if f1 > best_f1:
                    best_f1, best_sk = f1, sk
            base_f1 = get_val(df_kond, det, kond, "baseline", "f1_score")
            print(f"    {det:<12}: baseline={base_f1:.4f}  "
                  f"terbaik={SKENARIO_LABEL.get(best_sk,'?')} ({best_f1:.4f})")
    print("="*70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not cek_file():
        return

    df_agg  = pd.read_csv(CSV_AGG)
    df_kond = pd.read_csv(CSV_KOND)

    print("Membuat grafik dan tabel analisis...\n")
    os.makedirs(OUT_ROOT, exist_ok=True)

    plot_f1_per_kondisi(df_kond)
    plot_heatmap_iou(df_kond)
    plot_waktu(df_agg)
    buat_tabel(df_agg)
    analisis_teks(df_agg, df_kond)

    print(f"\nSemua output di folder: {OUT_ROOT}/")


if __name__ == "__main__":
    main()
