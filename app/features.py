"""Bosqich 4a: glyph maskasini o'lchamga bog'liq bo'lmagan vektorga aylantirish."""
from __future__ import annotations

import cv2
import numpy as np

from .config import CFG


def normalize(mask: np.ndarray, size: int | None = None) -> np.ndarray:
    """Glyph maskasini size x size kvadratga soladi.

    Nisbat (aspect ratio) SAQLANADI va bo'sh joy to'ldiriladi — bu muhim,
    chunki nisbat o'zi kuchli belgi: '1' ingichka (~0.45), '8'/'0' esa keng.
    Cho'zib yuborilsa bu ma'lumot yo'qoladi.
    """
    n = size or CFG.norm_size
    h, w = mask.shape
    if h == 0 or w == 0:
        return np.zeros((n, n), np.float32)
    inner = n - 4  # chetlarida 2 px joy qoldiramiz
    scale = inner / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = cv2.resize(mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((n, n), np.float32)
    y0, x0 = (n - nh) // 2, (n - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def vector(mask: np.ndarray) -> np.ndarray:
    """Solishtirish uchun L2-normallashtirilgan xususiyat vektori.

    Yengil blur qo'yiladi: shu tufayli 1 px siljish yoki distress yeb ketgan
    chekka piksel o'xshashlik ballini keskin tushirmaydi.
    """
    # Blur ATAYLAB kichik (0.5): 1 px siljishga chidamlilik uchun yetadi, lekin
    # 0/9/6/8 ni farqlaydigan mayda kontur detallarini yuvib yubormaydi.
    # Katta blur (1.2) '0' ni '9' ga o'xshatib, xato javobga sabab bo'lgan edi.
    norm = normalize(mask)
    blurred = cv2.GaussianBlur(norm, (3, 3), 0.5)
    v = blurred.ravel().astype(np.float32)
    nrm = float(np.linalg.norm(v))
    return v / nrm if nrm > 1e-6 else v


def augment(mask: np.ndarray) -> list[np.ndarray]:
    """Template bankini kengaytirish: kichik burilish va masshtab o'zgarishlari.

    Generator har bir belgini biroz boshqacha o'lchamda chizadi (sample'larda
    '4' bir joyda 12x18, boshqasida 10x14 bo'lgan), shuning uchun bitta
    namunadan bir nechta variant hosil qilamiz.
    """
    out = [mask]
    h, w = mask.shape
    if h < 3 or w < 3:
        return out
    pad = 3
    padded = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    ph, pw = padded.shape
    for angle in (-6.0, 6.0):
        for sc in (0.92, 1.0, 1.08):
            m = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, sc)
            warped = cv2.warpAffine(padded, m, (pw, ph), flags=cv2.INTER_NEAREST)
            ys, xs = np.where(warped > 0)
            if len(ys) == 0:
                continue
            out.append(warped[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    return out
