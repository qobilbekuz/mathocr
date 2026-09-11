"""Baholash: (a) to'liq bankda, (b) leave-one-image-out (LOIO).

LOIO — halol o'lchov: har bir rasm o'zi hissa qo'shmagan template bank
bilan sinaladi. Bank kichik bo'lganda bu qattiq sinov, chunki ba'zi belgi
(masalan '8') faqat bitta rasmda uchraydi va o'sha rasmni chiqarib
tashlaganda uni tanish uchun umuman namuna qolmaydi.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import features  # noqa: E402
from app.pipeline import MathRecognizer  # noqa: E402
from app.recognizer import TemplateRecognizer  # noqa: E402
from app.train import build, canonical  # noqa: E402


def expected(expr: str) -> dict:
    e = canonical(expr).rstrip("=")
    for op in "+-x/":
        if op in e[1:]:
            i = e.index(op, 1)
            return {"left": int(e[:i]), "op": op, "right": int(e[i + 1:]),
                    "expr": e}
    raise ValueError(expr)


DISPLAY = {"x": "×", "/": "÷"}


def run(bank: TemplateRecognizer, labels: dict, sample_dir: Path, title: str) -> tuple[int, int]:
    rec = MathRecognizer(bank)
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    head = f"{'Rasm':<12}{'Kutilgan':<12}{'Tanilgan':<12}{'Chap':>6}{'Op':>4}{'O‘ng':>6}{'Natija':>8}{'Conf':>7}  {'Holat'}"
    print(head)
    print("-" * 100)
    ok = 0
    for fname, expr in sorted(labels.items()):
        exp = expected(expr)
        res = MathRecognizer.process(rec, (sample_dir / fname).read_bytes())
        want_disp = f"{exp['left']}{DISPLAY.get(exp['op'], exp['op'])}{exp['right']}"
        got = res.get("expression") or "—"
        passed = (
            res.get("valid")
            and res.get("left_operand") == exp["left"]
            and res.get("operator") == DISPLAY.get(exp["op"], exp["op"])
            and res.get("right_operand") == exp["right"]
        )
        ok += bool(passed)
        status = "PASS" if passed else ("FAIL " + str(res.get("error") or res.get("detail") or ""))
        print(f"{fname[:8]:<12}{want_disp:<12}{got:<12}"
              f"{str(res.get('left_operand') or '—'):>6}{str(res.get('operator') or '—'):>4}"
              f"{str(res.get('right_operand') or '—'):>6}{str(res.get('result') if res.get('result') is not None else '—'):>8}"
              f"{res.get('confidence', 0):>7.2f}  {status}")
    print("-" * 100)
    print(f"Natija: {ok}/{len(labels)} PASS")
    return ok, len(labels)


def main() -> None:
    sample_dir = ROOT / "samples"
    labels = json.loads((ROOT / "labels.json").read_text())

    full, _ = build(sample_dir, ROOT / "labels.json")
    run(full, labels, sample_dir, "A) To'liq template bank (sanity check — bank rasmning o'zidan qurilgan)")

    # --- LOIO ---
    print(f"\n{'=' * 100}\nB) LEAVE-ONE-IMAGE-OUT (halol o'lchov)\n{'=' * 100}")
    total_c = correct_c = 0
    per_char_err: dict[str, list[str]] = {}
    for held in sorted(labels):
        keep = {k: v for k, v in labels.items() if k != held}
        tmp = ROOT / "models" / "_loio.json"
        tmp.write_text(json.dumps(keep))
        bank, _ = build(sample_dir, tmp)
        rec = MathRecognizer(bank)
        chars, _, _ = rec.recognize_chars((sample_dir / held).read_bytes())
        want = [c for c in canonical(labels[held]) if c != "="]
        got = [c.char for c in chars if c.char != "="]
        for w, g_, ch in zip(want, got, [c for c in chars if c.char != "="]):
            total_c += 1
            if w == g_:
                correct_c += 1
            else:
                per_char_err.setdefault(w, []).append(f"{g_} (conf {ch.confidence:.2f})")
        res = rec.process((sample_dir / held).read_bytes())
        if want == got:
            mark = "PASS"
        elif not res.get("valid"):
            mark = f"RAD ETILDI (to'g'ri xatti-harakat) — {res.get('error')}"
        else:
            mark = "FAIL — noto'g'ri natija qaytdi!"
        print(f"  {held[:8]}  kutilgan={''.join(want):<6} tanilgan={''.join(got):<6} "
              f"bank={' '.join(bank.known_chars):<20} {mark}")
        tmp.unlink()
    print(f"\n  Belgi darajasida aniqlik: {correct_c}/{total_c} = {100*correct_c/max(total_c,1):.1f}%")
    if per_char_err:
        print("  Xatolar:")
        for ch, errs in sorted(per_char_err.items()):
            print(f"    '{ch}' -> {', '.join(errs)}")


if __name__ == "__main__":
    main()
