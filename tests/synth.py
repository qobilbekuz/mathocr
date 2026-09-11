"""Sinov rasmlarini yig'ish — sample'lardan kesib olingan HAQIQIY dog'lardan.

Nima uchun kerak: siz bergan 5 ta rasmda 3 xonali operand ham, yopishib
qolgan dog'lar ham yo'q. Lekin dog'larni qayta joylashtirish orqali aynan
shunday holatlarni yasash mumkin — glyph piksellari haqiqiy bo'lib qoladi,
faqat joylashuvi o'zgaradi. Bu grouping algoritmini ("yonma-yon raqamlar =
bitta son") sample'da uchramagan konfiguratsiyalarda tekshiradi.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.preprocess import binarize, decode  # noqa: E402
from app.segment import _components, segment  # noqa: E402
from app.train import canonical  # noqa: E402

LABELS = {
    "0803774a-bf81-4d22-81a4-81d58d278adf.webp": "11+18=",
    "a740762c-d57d-48b4-9b24-a0a586d6b383.webp": "4+16=",
    "ce71b009-8626-45dc-8ace-cc55de052a51.webp": "19-14=",
    "d376ec0a-549f-4817-9bef-21e915bc19d9.webp": "19-7=",
    "f9555414-91a3-4939-bd1f-208af845ca35.webp": "7-6=",
}


def _blob_crops() -> dict[str, list[np.ndarray]]:
    """char -> [gray crop] (dog' + uning o'yig'i, atrofi oq).

    Dog' ASL (yopilmagan) maskadagi bog'langan komponent bo'yicha kesiladi:
    yopilgan maskada qo'shni dog' yoki '=' chizig'i bilan qo'shilib ketishi
    mumkin va kesma ikkita belgini o'z ichiga olib qolardi.
    """
    from app.segment import _holes, _owner_label

    bank: dict[str, list[np.ndarray]] = {}
    for fname, expr in LABELS.items():
        gray = decode((ROOT / "samples" / fname).read_bytes())
        ink = binarize(gray)
        _, labels_u, _ = _components(ink)

        seg = segment(gray)
        chars = [c for c in canonical(expr) if c != "="]
        glyphs = [g for g in seg.glyphs if g.kind == "glyph"]
        for ch, g in zip(chars, glyphs):
            lab = _owner_label(labels_u, ink, (g.x, g.y, g.w, g.h))
            if lab == 0:
                continue
            comp = (labels_u == lab).astype(np.uint8)
            filled = (comp | _holes(comp)).astype(np.uint8)
            region = cv2.dilate(filled, np.ones((5, 5), np.uint8))
            ys, xs = np.where(region > 0)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop = np.full((y1 - y0, x1 - x0), 255, np.uint8)
            m = region[y0:y1, x0:x1] > 0
            crop[m] = gray[y0:y1, x0:x1][m]
            bank.setdefault(ch, []).append(crop)

        eq = [g for g in seg.glyphs if g.kind == "equals"]
        if eq:
            e = eq[0]
            bank.setdefault("=", []).append(gray[e.y - 1:e.y + e.h + 1, e.x - 1:e.x + e.w + 1])
    return bank


_BANK: dict[str, list[np.ndarray]] | None = None


def available_chars() -> set[str]:
    global _BANK
    if _BANK is None:
        _BANK = _blob_crops()
    return set(_BANK)


def compose(expr: str, gap: int = 4, seed: int = 0, height: int = 57) -> np.ndarray:
    """Ifodadan captcha ko'rinishidagi rasm yasaydi.

    gap — dog'lar orasidagi bo'shliq (px). Manfiy qiymat dog'larni bir-biriga
    kiritib yuboradi — segmentatsiyaning ajratish qobiliyatini sinash uchun.
    """
    global _BANK
    if _BANK is None:
        _BANK = _blob_crops()
    rng = np.random.default_rng(seed)
    chars = [c for c in canonical(expr)]
    if not chars or chars[-1] != "=":
        chars.append("=")
    crops = []
    for ch in chars:
        pool = _BANK.get(ch)
        if not pool:
            raise KeyError(f"'{ch}' uchun namuna yo'q (sample'da uchramagan)")
        crops.append(pool[int(rng.integers(len(pool)))])

    total_w = sum(c.shape[1] for c in crops) + gap * (len(crops) - 1) + 16
    canvas = np.full((height, total_w), 255, np.uint8)
    x = 8
    for c in crops:
        h, w = c.shape
        y = int(np.clip((height - h) // 2 + rng.integers(-3, 4), 1, height - h - 1))
        win = canvas[y:y + h, x:x + w]
        canvas[y:y + h, x:x + w] = np.minimum(win, c)
        x += w + gap
    return canvas


def encode(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()
