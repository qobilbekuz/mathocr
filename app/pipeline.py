"""To'liq quvur: rasm -> ifoda -> natija.

Tanish (recognition) va hisoblash (calculation) ATAYLAB ajratilgan:
recognition oraliq ko'rinish (`characters`) qaytaradi, undan keyingi
bosqichlar faqat shu ko'rinish bilan ishlaydi. Shu sababli recognition
modelini almashtirish uchun `Recognizer` protokolini qondirish yetarli,
grouping/parser/calculator kodiga tegilmaydi.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import CFG
from . import preprocess as pre
from . import parser as safe_parser
from .grouping import Char, DISPLAY_OP, group
from .recognizer import EqualsRecognizer, Recognizer, TemplateRecognizer
from .segment import Segmentation, segment

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "templates.npz"


def load_and_segment(data: bytes) -> Segmentation:
    gray = pre.decode(data)
    work, _ = pre.normalize_scale(gray)
    return segment(work)


class MathRecognizer:
    """Yuqori darajali xizmat obyekti (bir marta yuklanadi, qayta ishlatiladi)."""

    def __init__(self, recognizer: Recognizer | None = None):
        self.recognizer = recognizer or TemplateRecognizer.load(MODEL_PATH)

    # --- 1-bosqich: rasm -> belgilar ---

    def recognize_chars(self, data: bytes) -> tuple[list[Char], Segmentation, float]:
        gray = pre.decode(data)
        work, scale = pre.normalize_scale(gray)
        seg = segment(work)
        inv = 1.0 / scale if scale else 1.0

        chars: list[Char] = []
        for g in seg.glyphs:
            if g.kind == "equals":
                bars = 2 if g.h > CFG.eq_max_height else 1
                pred = EqualsRecognizer.predict(bars)
            else:
                pred = self.recognizer.predict(g.mask)
            chars.append(
                Char(
                    char=pred.char,
                    x=int(round(g.x * inv)),
                    y=int(round(g.y * inv)),
                    width=int(round(g.w * inv)),
                    height=int(round(g.h * inv)),
                    confidence=pred.confidence,
                    runner_up=pred.runner_up,
                )
            )
        return chars, seg, scale

    # --- 2-bosqich: belgilar -> ifoda -> natija ---

    def process(self, data: bytes, debug: bool = False) -> dict[str, Any]:
        empty: dict[str, Any] = {
            "expression": None,
            "left_operand": None,
            "operator": None,
            "right_operand": None,
            "result": None,
            "confidence": 0.0,
            "valid": False,
        }

        try:
            chars, seg, _ = self.recognize_chars(data)
        except Exception as exc:
            return {**empty, "error": f"Rasmni o'qib bo'lmadi: {exc}"}

        out: dict[str, Any] = dict(empty)
        out["characters"] = [c.as_dict() for c in chars]

        if not chars:
            return {**out, "error": "Rasmda belgi topilmadi"}

        confidences = [c.confidence for c in chars]
        min_conf = float(min(confidences))
        out["confidence"] = round(min_conf, 4)

        weak = [c for c in chars if c.confidence < CFG.min_confidence]
        if weak:
            worst = min(weak, key=lambda c: c.confidence)
            return {
                **out,
                "error": "Low recognition confidence",
                "detail": (
                    f"x={worst.x} dagi belgi ishonchsiz tanildi: "
                    f"{worst.char!r} ({worst.confidence:.2f}), "
                    f"ikkinchi nomzod: {worst.runner_up!r}"
                ),
            }

        g = group(chars, CFG.max_intra_operand_gap_ratio)
        if g.warnings:
            out["warnings"] = g.warnings
        if not g.valid:
            return {**out, "error": g.error or "Ifodani qurib bo'lmadi"}

        internal = f"{g.left}{g.operator}{g.right}"
        display = f"{g.left}{DISPLAY_OP.get(g.operator, g.operator)}{g.right}"
        try:
            result = safe_parser.compute(internal)
        except safe_parser.ExpressionError as exc:
            return {**out, "expression": display, "error": f"Ifoda xato: {exc}"}

        out.update(
            {
                "expression": display,
                "left_operand": int(g.left),
                "operator": DISPLAY_OP.get(g.operator, g.operator),
                "right_operand": int(g.right),
                "result": result,
                "valid": True,
            }
        )
        if debug:
            out["debug"] = {
                "glyph_count": len(chars),
                "segmentation_notes": seg.notes,
                "mean_confidence": round(float(np.mean(confidences)), 4),
                "runner_ups": [
                    {"char": c.char, "runner_up": c.runner_up} for c in chars
                ],
            }
        return out
