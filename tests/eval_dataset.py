"""Haqiqiy open-budget dataseti ustida sizishsiz baholash.

Dataset yorliqlari (datasets/open-budget/labels.json) 0.80 chegarasida
avtomatik yaratilgan va qo'lda tekshirilgan (eng qiyin 0/8 holatlari
kiritilib, 54 ta namuna 100% to'g'ri chiqqan). Bu yerda rasmlar IMAGE
darajasida ikkiga bo'linadi: yarmidan bank quriladi, ikkinchi yarmida
sinaladi — shu tariqa bir rasmning glyphi ham o'rgatishda, ham sinovda
qatnashmaydi (ma'lumot sizishi bo'lmaydi).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline import MathRecognizer  # noqa: E402
from app.recognizer import DISPLAY  # noqa: E402
from app.train import build, canonical  # noqa: E402

DATA = ROOT / "datasets" / "open-budget"


def main() -> None:
    labels = json.loads((DATA / "labels.json").read_text())
    names = sorted(labels)
    train = {n: labels[n] for i, n in enumerate(names) if i % 2 == 0}
    test = {n: labels[n] for i, n in enumerate(names) if i % 2 == 1}

    tmp = DATA / "_train.json"
    tmp.write_text(json.dumps(train))
    bank, _ = build(DATA / "img", tmp, augment=False)
    tmp.unlink()
    rec = MathRecognizer(bank)

    solved = correct = wrong = rejected = 0
    wrong_cases = []
    for name, expr in test.items():
        res = rec.process((DATA / "img" / name).read_bytes())
        want = canonical(expr).rstrip("=")
        if not res["valid"]:
            rejected += 1
            continue
        solved += 1
        op = {"×": "x", "÷": "/"}.get(res["operator"], res["operator"])
        got = f"{res['left_operand']}{op}{res['right_operand']}"
        if got == want:
            correct += 1
        else:
            wrong += 1
            wrong_cases.append((name, want, got, res["confidence"]))

    n = len(test)
    print(f"Sizishsiz test (bank yarim rasmdan, sinov ikkinchi yarmida):")
    print(f"  test rasm:      {n}")
    print(f"  yechildi:       {solved} ({100*solved/n:.1f}%)")
    print(f"  rad etildi:     {rejected} ({100*rejected/n:.1f}%)")
    print(f"  yechilganidan to'g'ri: {correct}/{solved} = {100*correct/max(solved,1):.2f}%  (aniqlik/precision)")
    print(f"  NOTO'G'RI javob: {wrong}")
    for name, want, got, c in wrong_cases[:20]:
        print(f"     {name}: kutilgan {want}, chiqdi {got} (conf {c:.2f})")


if __name__ == "__main__":
    main()
