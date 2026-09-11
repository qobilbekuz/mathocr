"""Bosqich 2-3: belgi detektsiyasi va segmentatsiya.

Asosiy g'oya
------------
Bu generatorda belgi qora siyoh dog'i ustiga chizilgan EMAS — belgi dog'
ichidan OQ o'yiq (knockout) qilib kesilgan. Shuning uchun:

    dog' (ink component)  ->  bitta belgi
    dog' ichidagi oq teshiklar birlashmasi  ->  o'sha belgining glyph maskasi

Bu oq o'yiqlar bo'yicha to'g'ridan-to'g'ri segmentatsiya qilishdan ancha
ishonchli, chunki distress teksturasi glyph'ni 2-3 bo'lakka parchalab
yuboradi ('9' -> 3 bo'lak, '4' -> 2 bo'lak), lekin dog' bitta bo'lib qoladi.

'=' belgisi esa istisno: u dog'siz, oddiy ikkita qora chiziq.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import CFG
from . import preprocess as pre


@dataclass
class Glyph:
    """Segmentatsiya birligi — bitta belgiga nomzod."""

    x: int
    y: int
    w: int
    h: int
    mask: np.ndarray = field(repr=False)   # glyph shakli (0/1), bbox o'lchamida
    kind: str = "glyph"                    # "glyph" | "equals"
    blob: tuple[int, int, int, int] | None = None  # ona dog'ning bbox'i

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass
class Segmentation:
    glyphs: list[Glyph]
    ink: np.ndarray = field(repr=False)
    sealed: np.ndarray = field(repr=False)
    notes: list[str] = field(default_factory=list)


def _components(mask: np.ndarray):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    return n, labels, stats


def _find_equals(stats: np.ndarray, ink_shape) -> list[int]:
    """'=' ni tashkil qiluvchi ingichka gorizontal chiziqlarning indekslari."""
    bars = []
    for i in range(1, len(stats)):
        x, y, w, h, area = stats[i]
        if h <= CFG.eq_max_height and w >= CFG.eq_min_width and w / max(h, 1) >= CFG.eq_min_aspect:
            # to'ldirish zichligi yuqori bo'lsa (to'g'ri chiziq), qabul qilamiz
            if area / float(w * h) > 0.55:
                bars.append(i)
    return bars


def _holes(ink_mask: np.ndarray) -> np.ndarray:
    """Siyoh ichidagi yopiq oq sohalar (glyph o'yiqlari)."""
    h, w = ink_mask.shape
    free = (1 - ink_mask).astype(np.uint8).copy()
    for pt in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if free[pt[1], pt[0]] == 1:
            cv2.floodFill(free, np.zeros((h + 2, w + 2), np.uint8), pt, 2)
    # chetdagi hamma erkin pikselni ham fon deb belgilaymiz
    border = np.concatenate([free[0], free[-1], free[:, 0], free[:, -1]])
    if (border == 1).any():
        ys, xs = np.where(free == 1)
        for y, x in zip(ys, xs):
            if y in (0, h - 1) or x in (0, w - 1):
                cv2.floodFill(free, np.zeros((h + 2, w + 2), np.uint8), (int(x), int(y)), 2)
    return (free == 1).astype(np.uint8)


def _knockouts(ink: np.ndarray, sealed: np.ndarray) -> np.ndarray:
    """Glyph o'yiqlarining to'liq maskasi.

    Ikki manbani birlashtiradi:
      1. `holes(ink)` — asl maskadagi to'liq yopiq o'yiqlar. Yupqa belgilar
         ('-' atigi 2 px balandlikda) faqat shu yerda saqlanib qoladi;
         morfologik yopish ularni butunlay yo'q qiladi.
      2. yopilgan maskadagi o'yiqlar — distress yorig'i dog' chetigacha yetib,
         glyph tashqi fon bilan tutashib ketgan holatlar uchun. Bunda yopilgan
         o'yiq faqat "urug'" bo'lib xizmat qiladi, glyph shakli esa asl
         maskadan geodezik rekonstruksiya bilan tiklanadi (dog' chegarasidan
         tashqariga chiqmaydi).
    """
    holes_u = _holes(ink)
    holes_s = _holes(sealed)
    seeds = holes_s & (1 - holes_u)
    if not seeds.any():
        return holes_u

    bound = ((sealed | holes_s) > 0).astype(np.uint8)
    cand = ((1 - ink) & bound).astype(np.uint8)
    n, labels = cv2.connectedComponents(cand, 8)
    keep = np.unique(labels[seeds > 0])
    out = holes_u.copy()
    for lab in keep:
        if lab:
            out |= (labels == lab).astype(np.uint8)
    return out


def _owner_label(ink_labels: np.ndarray, ink: np.ndarray, box) -> int:
    """Teshik qaysi dog' ichida joylashganini aniqlaydi."""
    x, y, w, h = box
    H, W = ink_labels.shape
    x0, y0 = max(0, x - 2), max(0, y - 2)
    x1, y1 = min(W, x + w + 2), min(H, y + h + 2)
    win = ink_labels[y0:y1, x0:x1][ink[y0:y1, x0:x1] > 0]
    if win.size == 0:
        return 0
    vals, counts = np.unique(win, return_counts=True)
    return int(vals[np.argmax(counts)])


def _split_merged(frags: list[tuple], blob_h: int) -> list[list[tuple]]:
    """Yopishib qolgan dog'larni ichidagi o'yiqlar orasidagi bo'shliq bo'yicha ajratadi.

    Chegara o'lchovga asoslangan. Sample'larda bo'laklar orasidagi x-bo'shliq
    (dog' balandligiga nisbatan):

        BITTA belgi ichida:      -0.29 ... +0.04   (deyarli har doim ustma-ust)
        QO'SHILIB ketgan dog'da:  0.42 ... 0.67

    Ikki taqsimot orasida keng bo'sh oraliq bor, shuning uchun 0.25 chegarasi
    ikkalasidan ham uzoq turadi. Bo'shliq dog' BALANDLIGIGA nisbatlanadi, chunki
    balandlik gorizontal qo'shilishdan buzilmaydi — kenglik esa buziladi.
    """
    frags = sorted(frags, key=lambda f: f[0])
    if len(frags) < 2 or blob_h <= 0:
        return [frags]

    limit = CFG.split_gap_ratio * blob_h
    groups: list[list[tuple]] = [[frags[0]]]
    reach = frags[0][0] + frags[0][2]
    for f in frags[1:]:
        if f[0] - reach > limit:
            groups.append([f])
        else:
            groups[-1].append(f)
        reach = max(reach, f[0] + f[2])

    if len(groups) == 1:
        return groups

    # Himoya choragi: ajratish natijasida juda kichik bo'lak paydo bo'lsa, u
    # alohida belgi emas, shrift elementi (serif, nuqta) bo'lishi ehtimoli
    # yuqori — uni eng yaqin qo'shnisiga qaytarib qo'shamiz.
    areas = [sum(f[3] * f[2] for f in g) for g in groups]
    biggest = max(areas)
    merged: list[list[tuple]] = []
    for g, area in zip(groups, areas):
        if merged and area < CFG.min_split_area_ratio * biggest:
            merged[-1].extend(g)
        else:
            merged.append(g)
    return merged


def segment(gray: np.ndarray) -> Segmentation:
    raw_ink = pre.binarize(gray)
    notes: list[str] = []

    # 1) '=' ni ENG AVVAL, asl maskada topamiz va uni siyohdan olib tashlaymiz.
    #    Sababi: '=' chizig'i yonidagi dog'ga 1-2 px yaqin turishi mumkin
    #    (sample a740762c'da shunday) va morfologik yopish ikkalasini bitta
    #    komponentga qo'shib yuboradi. Bu esa dog'ni sun'iy ravishda
    #    kengaytirib, uni "yopishib qolgan ikkita dog'" deb ajratishga olib
    #    keladi.
    _, _, raw_stats = _components(raw_ink)
    eq_bars = [tuple(int(v) for v in raw_stats[i][:4]) for i in _find_equals(raw_stats, raw_ink.shape)]
    ink = raw_ink.copy()
    for bx0, by0, bw0, bh0 in eq_bars:
        ink[by0:by0 + bh0, bx0:bx0 + bw0] = 0

    sealed = pre.seal_cracks(ink)
    n, ink_labels, stats = _components(sealed)

    holes = _knockouts(ink, sealed)

    # --- teshiklarni dog'lar bo'yicha guruhlash ---
    hn, hlabels, hstats = _components(holes)
    per_blob: dict[int, list[tuple]] = {}
    for i in range(1, hn):
        x, y, w, h, area = (int(v) for v in hstats[i])
        if area < CFG.min_hole_area:
            continue
        owner = _owner_label(ink_labels, sealed, (x, y, w, h))
        if owner == 0:
            continue
        per_blob.setdefault(owner, []).append((x, y, w, h, i))

    glyphs: list[Glyph] = []
    for blob_id, frags in per_blob.items():
        bx, by, bw, bh, barea = (int(v) for v in stats[blob_id])
        if barea < CFG.min_blob_area or min(bw, bh) < CFG.min_blob_dim:
            continue
        for group in _split_merged(frags, bh):
            if not group:
                continue
            gx = min(f[0] for f in group)
            gy = min(f[1] for f in group)
            gx1 = max(f[0] + f[2] for f in group)
            gy1 = max(f[1] + f[3] for f in group)
            m = np.zeros((gy1 - gy, gx1 - gx), np.uint8)
            for f in group:
                m |= (hlabels[gy:gy1, gx:gx1] == f[4]).astype(np.uint8)
            glyphs.append(
                Glyph(x=gx, y=gy, w=gx1 - gx, h=gy1 - gy, mask=m, blob=(bx, by, bw, bh))
            )

    # --- '=' chiziqlarini bitta belgiga birlashtirish ---
    bars = list(eq_bars)
    if bars:
        bars.sort(key=lambda b: b[1])
        used = [False] * len(bars)
        for a in range(len(bars)):
            if used[a]:
                continue
            grp = [bars[a]]
            used[a] = True
            for b in range(a + 1, len(bars)):
                if used[b]:
                    continue
                # vertikal yaqin va gorizontal ustma-ust tushsa -> bir '='
                dy = abs(bars[b][1] - grp[-1][1])
                ox = min(grp[0][0] + grp[0][2], bars[b][0] + bars[b][2]) - max(grp[0][0], bars[b][0])
                if dy <= CFG.eq_max_gap_y and ox > 0.4 * min(grp[0][2], bars[b][2]):
                    grp.append(bars[b])
                    used[b] = True
            ex = min(g[0] for g in grp)
            ey = min(g[1] for g in grp)
            ex1 = max(g[0] + g[2] for g in grp)
            ey1 = max(g[1] + g[3] for g in grp)
            glyphs.append(
                Glyph(x=ex, y=ey, w=ex1 - ex, h=ey1 - ey,
                      mask=np.ones((ey1 - ey, ex1 - ex), np.uint8), kind="equals")
            )
            if len(grp) == 1:
                notes.append("'=' faqat bitta chiziqdan topildi")

    glyphs.sort(key=lambda g: g.x)
    return Segmentation(glyphs=glyphs, ink=raw_ink, sealed=sealed, notes=notes)
