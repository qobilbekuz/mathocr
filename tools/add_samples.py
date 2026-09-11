"""Yangi namuna qo'shish va template bankni qayta qurish.

Ishlatish (fayl yoki URL, ifoda bilan birga):

    python tools/add_samples.py \
        "https://cdn.edugames.uz/f/xxxx.webp=25-17=" \
        "/tmp/captcha2.webp=36:6="

Ifodadagi '×' o'rniga 'x', '÷' o'rniga '/' yoki ':' yozish mumkin.
Skript rasmni samples/ ga ko'chiradi, labels.json ni yangilaydi, bankni
qayta quradi va qaysi belgilar hali yetishmayotganini ko'rsatadi.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline import load_and_segment  # noqa: E402
from app.recognizer import CHARSET  # noqa: E402
from app.train import build, canonical  # noqa: E402


def fetch(src: str, dest_dir: Path) -> Path:
    if src.startswith(("http://", "https://")):
        name = src.rstrip("/").split("/")[-1] or "sample.webp"
        dest = dest_dir / name
        with urllib.request.urlopen(src, timeout=20) as r, dest.open("wb") as f:
            shutil.copyfileobj(r, f)
        return dest
    src_path = Path(src).expanduser().resolve()
    dest = dest_dir / src_path.name
    if src_path != dest:
        shutil.copyfile(src_path, dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", metavar="MANBA=IFODA",
                    help="fayl yo'li yoki URL, so'ng '=' va ifoda (masalan 25-17=)")
    ap.add_argument("--dir", default=str(ROOT / "samples"))
    ap.add_argument("--labels", default=str(ROOT / "labels.json"))
    ap.add_argument("--out", default=str(ROOT / "models" / "templates.npz"))
    args = ap.parse_args()

    sample_dir = Path(args.dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    labels_path = Path(args.labels)
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}

    for pair in args.pairs:
        # oxirgi '=' ifodaning bir qismi bo'lishi mumkin, shuning uchun
        # BIRINCHI '=' bo'yicha ajratamiz
        if "=" not in pair:
            print(f"o'tkazildi (format noto'g'ri): {pair}")
            continue
        src, expr = pair.split("=", 1)
        path = fetch(src, sample_dir)
        expr = canonical(expr)
        if not expr.endswith("="):
            expr += "="

        seg = load_and_segment(path.read_bytes())
        n_glyph = len([g for g in seg.glyphs if g.kind == "glyph"])
        n_want = len([c for c in expr if c != "="])
        flag = "OK" if n_glyph == n_want else f"DIQQAT: segmentatsiya {n_glyph} ta topdi, siz {n_want} ta yozdingiz"
        labels[path.name] = expr
        print(f"{path.name}  <-  {expr}   [{flag}]")

    labels_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False))
    rec, report = build(sample_dir, labels_path)
    print()
    for line in report:
        if line.startswith(("SKIP", "DIQQAT")):
            print(line)
    rec.save(args.out)
    counts = rec.meta.get("samples_per_char", {})
    print(f"\nbank yangilandi: {len(rec.vectors)} template")
    print("belgi bo'yicha namunalar:", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    missing = [c for c in CHARSET if c != "=" and c not in counts]
    print("hali yetishmayotgan belgilar:", " ".join(missing) if missing else "yo'q — to'liq")


if __name__ == "__main__":
    main()
