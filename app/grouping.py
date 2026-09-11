"""Bosqich 5-7: spatial tartiblash, operator aniqlash, operandlarni guruhlash.

Loyihaning eng muhim qoidasi shu yerda amalga oshiriladi:

    yonma-yon turgan raqamlar — BITTA son.

    "1 9 + 6 0 ="  ->  19 + 60      ("1 + 9 + 6 + 0" EMAS)
    "4 + 1 6 ="    ->  4 + 16       ("4 + 1 + 6" EMAS)

Buning uchun raqamlar orasidagi bo'shliqqa emas, OPERATOR pozitsiyasiga
tayanamiz: operator — operandlar orasidagi yagona chegara. Bo'shliq faqat
tekshiruv (sanity check) uchun ishlatiladi, chunki generator belgilar orasini
tasodifiy o'zgartiradi va "bo'shliq katta bo'lsa — yangi son" degan qoida
ishonchsiz bo'lardi.
"""
from __future__ import annotations

from dataclasses import dataclass, field

OPERATORS = {"+", "-", "x", "/"}
DISPLAY_OP = {"x": "×", "/": "÷"}


@dataclass
class Char:
    char: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    runner_up: str | None = None

    def as_dict(self, display: bool = True) -> dict:
        return {
            "char": DISPLAY_OP.get(self.char, self.char) if display else self.char,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class Grouped:
    left: str | None = None
    operator: str | None = None
    right: str | None = None
    valid: bool = False
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    has_equals: bool = False


def _pitch(chars: list[Char]) -> float:
    if len(chars) < 2:
        return 0.0
    centers = sorted(c.x + c.width / 2 for c in chars)
    gaps = [b - a for a, b in zip(centers, centers[1:])]
    gaps.sort()
    return gaps[len(gaps) // 2]


def group(chars: list[Char], max_gap_ratio: float = 2.2) -> Grouped:
    """Tanilgan belgilardan chap operand / operator / o'ng operandni ajratadi."""
    chars = sorted(chars, key=lambda c: c.x)
    res = Grouped()

    # --- '=' ifoda oxirini bildiradi ---
    eq_positions = [i for i, c in enumerate(chars) if c.char == "="]
    if eq_positions:
        res.has_equals = True
        if eq_positions[0] != len(chars) - 1:
            res.warnings.append("'=' ifoda oxirida emas — undan keyingi belgilar tashlab yuborildi")
        chars = chars[: eq_positions[0]]  # '=' hisobga OLINMAYDI
    else:
        res.warnings.append("'=' belgisi topilmadi")

    if not chars:
        res.error = "Belgilar topilmadi"
        return res

    unknown = [c for c in chars if c.char not in OPERATORS and not c.char.isdigit()]
    if unknown:
        res.error = f"Noma'lum belgi: {unknown[0].char!r}"
        return res

    # --- operator chegaralarini topish ---
    op_idx = [i for i, c in enumerate(chars) if c.char in OPERATORS]
    if not op_idx:
        res.error = "Operator topilmadi"
        return res
    if len(op_idx) > 1:
        res.error = f"{len(op_idx)} ta operator topildi, 1 tasi kutilgan edi"
        return res

    i = op_idx[0]
    left_chars, right_chars = chars[:i], chars[i + 1:]
    if not left_chars:
        res.error = "Chap operand bo'sh"
        return res
    if not right_chars:
        res.error = "O'ng operand bo'sh"
        return res

    # --- raqamlarni bitta songa BIRLASHTIRISH (asosiy qoida) ---
    res.left = "".join(c.char for c in left_chars)
    res.right = "".join(c.char for c in right_chars)
    res.operator = chars[i].char

    # --- spatial tekshiruv: faqat ogohlantirish, guruhlashni buzmaydi ---
    pitch = _pitch(chars)
    if pitch:
        for side, group_chars in (("chap", left_chars), ("o'ng", right_chars)):
            for a, b in zip(group_chars, group_chars[1:]):
                gap = (b.x + b.width / 2) - (a.x + a.width / 2)
                if gap > max_gap_ratio * pitch:
                    res.warnings.append(
                        f"{side} operandda raqamlar orasi odatdagidan keng "
                        f"({gap:.0f} px, o'rtacha {pitch:.0f} px)"
                    )

    res.valid = True
    return res
