"""
@file: utils.py
@author: Mert-chan
@date: 2026-11-26
@brief: Helpers around denoiser.py for the 200 graded frames of data/frames.csv. run_denoiser writes each echo mask to data/denoised/ as PNG (echo black on white, 315 x 396, same crop as the method). make_table counts Table I of the letter: per set (tuning, random), subset and background the number of frames, echo kept / partial / lost and noise clean / left, output data/table1.csv. make_examples draws numbered Raw | Denoised pairs, 24 per page, raw titled with subset and background, denoised with the grade (red if partial, lost or noise left); pages never mix tuning (1-100) and random (101-200); needs data/denoised/; outputs examples/tuning_<page>.png, examples/random_<page>.png, examples/manual_grade.csv (the grade of every frame and the sheet it is on). Args: denoise, table and/or examples, default all three in that order.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from denoiser import denoise, load_crop

HERE    = Path(__file__).parent
ROWS    = [("Echo trace", "Moderate"), ("Echo trace", "Heavy RFI / speckle"), ("Echo trace", "Sparse / weak echo"),
           ("Spread F", "Moderate"), ("Spread F", "Heavy RFI / speckle"), ("Spread F", "Sparse / weak echo"),
           ("No echo", None), ("Total", None)]
EXTENT  = (2.0, 17.84, 100.0, 800.0)                    # MHz, km of the crop
PER_ROW = 4
N_ROWS  = 6


def load_frames():
    return pd.read_csv(HERE / "data" / "frames.csv", dtype=str, keep_default_na=False)


# Denoise
#==================================
def run_denoiser(out=HERE / "data" / "denoised"):
    frames = load_frames()
    out.mkdir(parents=True, exist_ok=True)
    for f in frames.itertuples():
        mask = denoise(load_crop(HERE / "data" / f.file))
        Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(out / Path(f.file).name)
    print("%d masks -> %s" % (len(frames), out))
#==================================


# Table I
#==================================
def make_table(out=HERE / "data" / "table1.csv"):
    frames = load_frames()
    table  = []
    for sub, bg in ROWS:
        row = {"subset": sub, "background": bg or ""}
        for name in ("tuning", "random"):
            d = frames[frames.set == name]
            if sub != "Total":
                d = d[d.subset == sub]
            if bg:
                d = d[d.background == bg]
            row.update({f"{name}_n": len(d), f"{name}_kept": (d.echo == "kept").sum(), f"{name}_partial": (d.echo == "partial").sum(),
                        f"{name}_lost": (d.echo == "lost").sum(), f"{name}_clean": (d.noise == "clean").sum(),
                        f"{name}_left": (d.noise == "noise left").sum()})
        table.append(row)
    table = pd.DataFrame(table)
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
#==================================


# Example sheets
#==================================
def show(ax, img):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto", extent=EXTENT, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def make_examples(out=HERE / "examples"):
    frames = load_frames()
    frames["id"] = frames.id.astype(int)
    out.mkdir(parents=True, exist_ok=True)
    per    = PER_ROW * N_ROWS
    chunks = [(name, i + 1, part.iloc[s:s + per]) for name, part in (("tuning", frames[frames.id <= 100]), ("random", frames[frames.id > 100]))
              for i, s in enumerate(range(0, len(part), per))]
    sheet  = {}
    for name, page, chunk in chunks:
        sheet.update({i: "%s_%d" % (name, page) for i in chunk.id})
        fig, axs = plt.subplots(N_ROWS, 2 * PER_ROW, figsize=(11, 8.8), gridspec_kw=dict(wspace=0.08, hspace=0.55, top=0.93, bottom=0.01))
        for ax in axs.flat:
            ax.axis("off")
        for k, f in enumerate(chunk.itertuples()):
            r, p = divmod(k, PER_ROW)
            raw  = load_crop(HERE / "data" / f.file)
            den  = np.array(Image.open(HERE / "data" / "denoised" / Path(f.file).name)) / 255.0
            for c, img in enumerate((raw, den)):
                axs[r, 2 * p + c].axis("on")
                show(axs[r, 2 * p + c], img)
            axs[r, 2 * p].set_title("%d  %s %s\n%s, %s" % (f.id, f.station, f.time_utc, f.subset, f.background) if f.background else
                                    "%d  %s %s\n%s" % (f.id, f.station, f.time_utc, f.subset), fontsize=6)
            grade = ("%s, %s" % (f.echo, f.noise) if f.echo else f.noise).capitalize()
            bad   = f.echo in ("partial", "lost") or f.noise == "noise left"
            axs[r, 2 * p + 1].set_title("\n" + grade, fontsize=6, color="#c0392b" if bad else "k")
        fig.suptitle("Tuning frames (1-100)" if chunk.id.iloc[0] <= 100 else "Random frames (101-200)", fontsize=11, y=0.985)
        fig.savefig(out / ("%s_%d.png" % (name, page)), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(out / ("%s_%d.png" % (name, page)))
    frames.assign(sheet=frames.id.map(sheet))[["id", "set", "station", "time_utc", "sheet", "subset", "background", "echo", "noise"]].to_csv(
        out / "manual_grade.csv", index=False)
#==================================


if __name__ == "__main__":
    steps = {"denoise": run_denoiser, "table": make_table, "examples": make_examples}
    for arg in sys.argv[1:] or list(steps):
        steps[arg]()
