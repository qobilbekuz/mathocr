#!/usr/bin/env python3
"""Yig'ilgan trafik namunalarini sarhisob qiladi.

    python3 tools/traffic_report.py                # umumiy holat
    python3 tools/traffic_report.py --review       # ko'rib chiqilishi kerak bo'lganlar
    python3 tools/traffic_report.py --chars        # qaysi belgilar qancha uchradi

MAQSAD: bankdagi bo'shliqni (ayniqsa `÷`) real trafikdan topib olish.
Ko'rib chiqilgan namuna `tools/add_samples.py "fayl=36/6="` bilan bankka qo'shiladi.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

ROOT = Path(os.getenv("MATHOCR_COLLECT_DIR", "/var/lib/mathocr/traffic"))


def load() -> list[dict]:
    idx = ROOT / "index.jsonl"
    if not idx.exists():
        return []
    rows = []
    for line in idx.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # yarim yozilgan qatorni jimgina o'tkazamiz
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review", action="store_true", help="faqat ko'rib chiqilishi kerak bo'lganlar")
    ap.add_argument("--chars", action="store_true", help="belgilar chastotasi")
    ap.add_argument("--limit", type=int, default=25)
    a = ap.parse_args()

    rows = load()
    if not rows:
        print(f"Hali namuna yig'ilmagan ({ROOT}).")
        return

    buckets = collections.Counter(r["bucket"] for r in rows)
    total_bytes = sum(
        (ROOT / r["file"]).stat().st_size for r in rows if (ROOT / r["file"]).exists()
    )
    print(f"Katalog : {ROOT}")
    print(f"Jami    : {len(rows)} namuna, {total_bytes / 1024 / 1024:.1f} MB")
    print("Toifalar: " + ", ".join(f"{k}={v}" for k, v in buckets.most_common()))

    if a.chars:
        cnt = collections.Counter(c for r in rows for c in (r.get("chars") or []) if c)
        print("\nBelgilar chastotasi:")
        for ch, n in cnt.most_common():
            print(f"  {ch!r:5} {n}")
        div = cnt.get("/", 0) + cnt.get("÷", 0)
        print(f"\n  ÷ (bo'lish): {div} — " + ("KELDI! bankka qo'shing" if div else "hali uchramadi"))
        return

    if a.review:
        rev = [r for r in rows if r["bucket"] == "review"]
        print(f"\nKo'rib chiqilishi kerak: {len(rev)}")
        for r in rev[-a.limit:]:
            print(f"  {r['confidence']:.3f}  valid={str(r['valid']):5}  "
                  f"{str(r.get('expression')):10}  {r['file']}")
        if rev:
            print("\nBelgilagach bankka qo'shish:")
            print(f'  python3 tools/add_samples.py "{ROOT}/{rev[-1]["file"]}=36/6="')
        return

    print("\nBatafsil: --review (ko'rib chiqish kerak) yoki --chars (belgilar)")


if __name__ == "__main__":
    main()
