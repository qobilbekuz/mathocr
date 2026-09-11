"""Click-captcha yechuvchini CSV ground-truth ustida baholaydi.

Ishlatish:
    python3 clickcaptcha/evaluate.py clickcaptcha/samples.csv

CSV ustunlari: cache_key, body_b64, instructions_b64, answer_json, created_at

Har namuna uchun yechuvchi 2 koordinata qaytaradi; ular haqiqiy javob
nuqtalariga qancha yaqinligi o'lchanadi. "To'g'ri" deb hisoblanadi, agar
koordinata haqiqiy nuqtadan `TOL` piksel ichida bo'lsa (token o'lchamiga mos).
"""
from __future__ import annotations

import base64
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clickcaptcha.solver import locate  # noqa: E402

csv.field_size_limit(sys.maxsize)

TOL = 22  # piksel: bosish token ustiga tushsa yetarli (token ~30-50 px)


def main(path: str) -> None:
    rows = list(csv.DictReader(open(path)))
    print(f"namunalar: {len(rows)}\n")

    per_point_ok = 0
    per_point_tot = 0
    full_ok = 0
    for i, r in enumerate(rows):
        body = base64.b64decode(r["body_b64"])
        instr = base64.b64decode(r["instructions_b64"])
        truth = json.loads(r["answer_json"])["coordinates"]
        out = locate(instr, body)
        got = out["coordinates"]

        both = True
        line = []
        for j in range(len(truth)):
            per_point_tot += 1
            if j < len(got) and got[j]["x"] is not None:
                dx = got[j]["x"] - truth[j]["x"]
                dy = got[j]["y"] - truth[j]["y"]
                dist = (dx * dx + dy * dy) ** 0.5
                ok = dist <= TOL
                per_point_ok += ok
                both &= ok
                line.append(f"({got[j]['x']},{got[j]['y']})~{dist:.0f}px {'OK' if ok else 'XATO'}")
            else:
                both = False
                line.append("—")
        full_ok += both
        truth_s = " ".join(f"({c['x']},{c['y']})" for c in truth)
        print(f"row{i}: haqiqiy[{truth_s}]  nomzod={out['n_candidates']}  "
              f"scores={out['scores']}  -> {' | '.join(line)}")

    print(f"\nnuqta darajasida: {per_point_ok}/{per_point_tot} "
          f"({100*per_point_ok/max(per_point_tot,1):.1f}%) {TOL}px ichida")
    print(f"to'liq (ikkala nuqta): {full_ok}/{len(rows)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "click" / "samples.csv"))
