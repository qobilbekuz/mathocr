"""Template bankini yig'ish (CLI).

Ishlatish:
    python -m app.train --dir samples --labels labels.json --out models/templates.npz

labels.json formati — fayl nomi -> ifoda satri:
    {
      "0803774a-....webp": "11+18=",
      "a740762c-....webp": "4+16="
    }

Belgilash juda arzon: rasmga qarab ko'rgan ifodangizni yozasiz. Trainer
rasmni segmentatsiya qiladi va glyph'larni chapdan o'ngga tartibda ifoda
belgilariga moslaydi. Agar glyph soni belgilar soniga to'g'ri kelmasa, o'sha
rasm o'tkazib yuboriladi va sababi ko'rsatiladi — bu segmentatsiya
xatolarini erta ushlash uchun ataylab qattiq qoida.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import features
from .pipeline import load_and_segment
from .recognizer import CHARSET, TemplateRecognizer

CANON = {"×": "x", "*": "x", "÷": "/", ":": "/", "−": "-", "–": "-"}


def canonical(expr: str) -> str:
    out = []
    for ch in expr:
        ch = CANON.get(ch, ch)
        if ch.isspace():
            continue
        out.append(ch)
    return "".join(out)


def build(sample_dir: Path, labels_path: Path, augment: bool = True) -> tuple[TemplateRecognizer, list[str]]:
    labels: dict[str, str] = json.loads(labels_path.read_text())
    vectors: list[np.ndarray] = []
    names: list[str] = []
    report: list[str] = []
    counts: dict[str, int] = {}

    for fname, expr in sorted(labels.items()):
        path = sample_dir / fname
        if not path.exists():
            report.append(f"SKIP {fname}: fayl yo'q")
            continue
        expr = canonical(expr)
        wanted = [c for c in expr if c != "="]
        seg = load_and_segment(path.read_bytes())
        glyphs = [g for g in seg.glyphs if g.kind == "glyph"]
        if len(glyphs) != len(wanted):
            report.append(
                f"SKIP {fname}: segmentatsiya {len(glyphs)} ta belgi topdi, "
                f"belgilashda esa {len(wanted)} ta ({expr})"
            )
            continue
        bad = [c for c in wanted if c not in CHARSET]
        if bad:
            report.append(f"SKIP {fname}: noma'lum belgi(lar) {bad}")
            continue
        for ch, g in zip(wanted, glyphs):
            variants = features.augment(g.mask) if augment else [g.mask]
            for aug in variants:
                vectors.append(features.vector(aug))
                names.append(ch)
            counts[ch] = counts.get(ch, 0) + 1
        report.append(f"OK   {fname}: {expr}")

    meta = {"samples_per_char": counts, "n_templates": len(vectors)}
    rec = TemplateRecognizer(np.array(vectors, dtype=np.float32) if vectors else np.zeros((0, 1), np.float32),
                             names, meta)
    missing = [c for c in CHARSET if c != "=" and c not in counts]
    if missing:
        report.append("DIQQAT: bu belgilar uchun hech qanday namuna yo'q: " + " ".join(missing))
    return rec, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="samples")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--out", default="models/templates.npz")
    ap.add_argument("--no-augment", action="store_true",
                    help="augmentatsiyasiz — namuna ko'p bo'lganda (masalan open-budget dataseti) tavsiya etiladi")
    args = ap.parse_args()

    rec, report = build(Path(args.dir), Path(args.labels), augment=not args.no_augment)
    for line in report:
        print(line)
    rec.save(args.out)
    print(f"\nsaqlandi: {args.out}  |  {len(rec.vectors)} template, "
          f"belgilar: {' '.join(rec.known_chars)}")
    print("namunalar soni:", rec.meta.get("samples_per_char"))


if __name__ == "__main__":
    main()
