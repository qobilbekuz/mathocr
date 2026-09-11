"""Bosqich 1: rasmni o'qish va binarizatsiya."""
from __future__ import annotations

import io

import cv2
import numpy as np

from .config import CFG


class ImageDecodeError(ValueError):
    pass


def decode(data: bytes) -> np.ndarray:
    """Baytlardan gray-scale rasm. WebP/PNG/JPEG/BMP qo'llab-quvvatlanadi."""
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # WebP ba'zi buildlarda imdecode orqali o'qilmasligi mumkin -> Pillow
        try:
            from PIL import Image

            img = np.array(Image.open(io.BytesIO(data)).convert("L"))
        except Exception as exc:  # pragma: no cover
            raise ImageDecodeError(f"rasmni dekod qilib bo'lmadi: {exc}") from exc
    return img


def normalize_scale(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Rasmni ish balandligiga keltiradi. (rasm, scale) qaytaradi.

    scale — original -> ish koordinatasiga o'tish koeffitsienti; bounding box'lar
    javobda ORIGINAL koordinatada qaytishi uchun kerak.
    """
    if not CFG.work_height or gray.shape[0] == CFG.work_height:
        return gray, 1.0
    scale = CFG.work_height / gray.shape[0]
    w = max(1, int(round(gray.shape[1] * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, (w, CFG.work_height), interpolation=interp), scale


def binarize(gray: np.ndarray) -> np.ndarray:
    """Siyoh (qora) = 1, fon (oq) = 0 bo'lgan uint8 maska.

    Generator har doim oq fonga qora siyoh chizadi, shuning uchun qutblik
    aniq. Baribir fon yorug'ligini tekshirib, teskari rasmni ham to'g'rilaymiz.
    """
    # DIQQAT: bu yerda blur QILINMAYDI. Glyph'larning eng ingichka shtrixlari
    # (masalan '4' ning diagonali yoki '-' ning tanasi) atigi 1-2 px; 3x3
    # Gaussian ularni atrofdagi qora siyohga qo'shib yuboradi va belgi yo'qoladi.
    thr, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if not 20 < thr < 240:  # Otsu degenerativ bo'lsa
        _, mask = cv2.threshold(gray, CFG.fallback_threshold, 1, cv2.THRESH_BINARY_INV)
    # Fon qora bo'lib qolgan bo'lsa (teskari rasm) — qutblikni almashtiramiz.
    if mask.mean() > 0.6:
        mask = 1 - mask
    return mask.astype(np.uint8)


def seal_cracks(ink: np.ndarray) -> np.ndarray:
    """Distress teksturasi hosil qilgan ingichka yoriqlarni yopadi.

    Bu MUHIM: yoriq dog' chetigacha yetib borsa, ichkaridagi oq glyph tashqi
    fon bilan tutashib ketadi va "o'yiq" sifatida topilmay qoladi.
    """
    k = CFG.crack_close_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
