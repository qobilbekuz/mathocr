"""Click-captcha yechuvchi (koordinata qaytaradi, bosmaydi).

Captcha turi
------------
Foydalanuvchiga ikkita rasm beriladi:
  1. SAHNA (body)      — 345x230 manzara, ichiga ~7-9 ta 2-harfli token
                         yarim-shaffof singdirilgan (kamuflyaj).
  2. KO'RSATMA (instr) — 210x70 oq fon, ustida ikkita 2-harfli NISHON token
                         (+ chalg'ituvchi nuqtalar va egri chiziq).

Vazifa: ko'rsatmadagi 2 nishon tokenning SAHNADAGI joylashuvini topib,
ularning (x, y) koordinatalarini KO'RSATMA TARTIBIDA qaytarish.

Javob formati (backend bilan bir xil):
  {"coordinates": [{"id": "..", "x": .., "y": ..}, ...]}
  `id` = str(x) + str(y) — foydalanuvchi bergan haqiqiy javobdan aniqlangan
  taxmin (pastda `locate()` da izoh bor).

Yondashuv
---------
Harf OCR'siz, eng kam ma'lumot talab qiladigan yo'l:
  1. Ko'rsatmani ikkita token-rasmga ajratish (ustun proyeksiyasidagi katta
     bo'shliq bo'yicha).
  2. Sahnadagi token-nomzod hududlarini aniqlash (yuqori-o'tkazgich + morfologiya).
  3. Har nishon token-rasmini nomzodlar bilan ko'p masshtabli, qirralarga
     asoslangan (rangdan mustaqil) o'xshashlik bo'yicha solishtirish.
  4. Eng mos nomzodning markazini qaytarish.

DIQQAT — kalibrlash: bu modul 5 ta namuna asosida yozilgan. Chegaralar
(threshold) va parametrlar ko'proq namuna kelganda `tune()` bilan qayta
sozlanishi kerak. Hozircha bu ISHONCHLI EMAS, PROTOTIP.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------
# Yordamchi: rasmni yuklash
# --------------------------------------------------------------------------

def _to_gray(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("L"))


def _to_rgb(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def _edge_mag(gray: np.ndarray) -> np.ndarray:
    """Rangdan mustaqil qirra kuchi (0..1). Token boshqa rangda bo'lsa ham
    uning SHAKLI (qirralari) bir xil qoladi."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.sqrt(gx * gx + gy * gy)
    return m / (m.max() + 1e-6)


# --------------------------------------------------------------------------
# 1-bosqich: ko'rsatmadan ikkita nishon token-rasmini ajratish
# --------------------------------------------------------------------------

@dataclass
class TokenPatch:
    gray: np.ndarray   # token kulrang rasmi (oq fonda)
    x0: int            # ko'rsatmadagi joyi (faqat ma'lumot uchun)


def split_instruction(instr_gray: np.ndarray) -> list[TokenPatch]:
    """Ko'rsatmani ikkita token-rasmga ajratadi.

    Ikkala token orasida katta bo'sh vertikal yo'lak bor. Egri chiziq ingichka
    bo'lgani uchun uni masofa-transformatsiya bilan olib tashlab, harf
    yadrolarining ustun proyeksiyasidan bo'shliqni topamiz.
    """
    mask = (instr_gray < 200).astype(np.uint8)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    core = ((dt >= 2.2) & (mask > 0)).astype(np.uint8)  # qalin harf tanasi
    colsum = core.sum(0)
    nz = np.where(colsum > 0)[0]
    if len(nz) < 2:
        return []

    # ichki bo'shliqlarni topamiz, eng kengini ajratuvchi deb olamiz
    runs = []
    inrun = False
    for x in range(nz.min(), nz.max() + 1):
        if colsum[x] == 0:
            if not inrun:
                start = x
                inrun = True
        elif inrun:
            runs.append((start, x - 1))
            inrun = False
    if not runs:
        # bitta blok — o'rtadan bo'lamiz
        mid = (nz.min() + nz.max()) // 2
        splits = [(nz.min(), mid), (mid, nz.max())]
    else:
        sep = max(runs, key=lambda r: r[1] - r[0])
        splits = [(nz.min(), sep[0]), (sep[1], nz.max())]

    patches = []
    for x0, x1 in splits:
        sub = instr_gray[:, x0:x1 + 1]
        m = (sub < 200).astype(np.uint8)
        # kichik chalg'ituvchi nuqtalarni olib tashlash
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        ys, xs = np.where(m > 0)
        if len(ys) == 0:
            continue
        pad = 2
        y0, y1 = max(0, ys.min() - pad), ys.max() + pad
        xa, xb = max(0, xs.min() - pad), xs.max() + pad
        patches.append(TokenPatch(gray=sub[y0:y1 + 1, xa:xb + 1], x0=x0))
    return patches


# --------------------------------------------------------------------------
# 2-bosqich: sahnadagi token-nomzod hududlarini aniqlash
# --------------------------------------------------------------------------

def scene_candidates(body_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Sahnadagi ehtimoliy token markazlari (cx, cy, w, h).

    Yarim-shaffof tokenlar mahalliy qirra to'plami sifatida ko'rinadi.
    Yuqori-o'tkazgich bilan ularni ajratib, harf shtrixlarini token bloklariga
    birlashtiramiz. Filtrlar token o'lchamiga moslangan (~14-70 px keng).
    """
    a = body_rgb.astype(np.float32)
    hp = a - cv2.GaussianBlur(a, (0, 0), 2.5)
    mag = np.sqrt((hp ** 2).sum(2))
    mag = (mag / (mag.max() + 1e-6) * 255).astype(np.uint8)
    th = cv2.threshold(mag, 40, 255, cv2.THRESH_BINARY)[1]
    close = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))
    n, _, st, cent = cv2.connectedComponentsWithStats(close, 8)
    cands = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if 12 <= h <= 46 and 14 <= w <= 72 and area >= 110 and w < 3.6 * h:
            cands.append((int(cent[i][0]), int(cent[i][1]), int(w), int(h)))
    return cands


# --------------------------------------------------------------------------
# 3-bosqich: nishon tokenni sahnada joylashtirish
# --------------------------------------------------------------------------

def _match_score(patch_edges: np.ndarray, scene_edges: np.ndarray,
                 center: tuple[int, int], scales) -> float:
    """Nomzod markazi atrofida token-rasm bilan eng yaxshi moslik bali."""
    cx, cy = center
    best = -1.0
    H, W = scene_edges.shape
    for sc in scales:
        nh = max(6, int(patch_edges.shape[0] * sc))
        nw = max(6, int(patch_edges.shape[1] * sc))
        y0, y1 = cy - nh, cy + nh
        x0, x1 = cx - nw, cx + nw
        if y0 < 0 or x0 < 0 or y1 >= H or x1 >= W:
            y0, x0 = max(0, y0), max(0, x0)
            y1, x1 = min(H, y1), min(W, x1)
        window = scene_edges[y0:y1, x0:x1]
        if window.shape[0] <= nh or window.shape[1] <= nw:
            continue
        tmpl = cv2.resize(patch_edges, (nw, nh)).astype(np.float32)
        tmpl = tmpl - tmpl.mean()
        res = cv2.matchTemplate(window.astype(np.float32), tmpl, cv2.TM_CCOEFF_NORMED)
        best = max(best, float(res.max()))
    return best


def locate(instr_data: bytes, body_data: bytes) -> dict:
    """Asosiy funksiya: ikkita nishon tokenning koordinatalarini qaytaradi.

    Qaytaradi:
      {"coordinates": [{"x","y"}, {"x","y"}], "scores": [...], "valid": bool}
    """
    instr = _to_gray(instr_data)
    body_rgb = _to_rgb(body_data)
    body = cv2.cvtColor(body_rgb, cv2.COLOR_RGB2GRAY)

    patches = split_instruction(instr)
    cands = scene_candidates(body_rgb)
    scene_edges = _edge_mag(body)
    scales = np.linspace(0.7, 1.4, 15)

    coords, scores = [], []
    used = set()
    for tp in patches[:2]:
        pe = _edge_mag(tp.gray)
        ranked = []
        for ci, (cx, cy, w, h) in enumerate(cands):
            s = _match_score(pe, scene_edges, (cx, cy), scales)
            ranked.append((s, ci, cx, cy))
        ranked.sort(reverse=True)
        # bitta nomzod ikki tokenga berilmasin
        pick = next((r for r in ranked if r[1] not in used), ranked[0] if ranked else None)
        if pick is None:
            coords.append({"id": None, "x": None, "y": None})
            scores.append(0.0)
            continue
        used.add(pick[1])
        # `id` — mijoz backendi kutadigan maydon. 2026-08-25 da foydalanuvchi
        # bergan haqiqiy javobda id = str(x) + str(y) ekani ikkala nuqtada ham
        # tasdiqlandi (113,175 -> "113175"; 208,75 -> "20875").
        # DIQQAT: bu 2 ta namunadan chiqarilgan TAXMIN. Agar backend id ni
        # o'zi tayinlasa (koordinatadan kelib chiqmasa), bu qiymat noto'g'ri
        # bo'ladi — mijozdan tasdiqlash kerak.
        coords.append({"id": f"{pick[2]}{pick[3]}", "x": pick[2], "y": pick[3]})
        scores.append(round(pick[0], 3))

    valid = len(coords) == 2 and all(c["x"] is not None for c in coords)
    return {"coordinates": coords, "scores": scores, "valid": valid,
            "n_candidates": len(cands)}
