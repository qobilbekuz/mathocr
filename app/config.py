"""Pipeline sozlamalari — sample rasmlar tahlilidan olingan qiymatlar.

Sample statistikasi (5 ta rasm, 180x57 px):
  - siyoh dog'i (blob):  21..40 px kenglik, 22..40 px balandlik, area 320..1005
  - glyph (oq o'yiq):    4..14 px kenglik, 2..23 px balandlik
  - '=' chiziqlari:      w 23..27, h 5..7, y ~29 va ~41, doim eng o'ngda
  - belgilar orasi pitch: ~29 px
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- binarizatsiya ---
    # Otsu ishlatiladi; bu qiymat faqat fallback (Otsu degenerativ bo'lsa).
    fallback_threshold: int = 128
    # Ichki qayta ishlash uchun rasm shu balandlikka keltiriladi (0 = tegilmasin).
    # Sample'lar 57 px; kattaroq rasm kelsa normallashtiriladi.
    work_height: int = 57

    # --- komponent filtrlari (work_height masshtabida) ---
    min_blob_area: int = 90        # bundan kichigi distress "chachragi"
    min_blob_dim: int = 10         # blob eng kichik o'lchami
    speck_max_area: int = 80       # shovqin sifatida tashlanadi

    # --- '=' aniqlash ---
    eq_max_height: int = 10        # chiziq balandligi
    eq_min_aspect: float = 2.0     # w/h
    eq_min_width: int = 10
    eq_max_gap_y: int = 22         # ikki chiziq orasidagi vertikal masofa

    # --- glyph (oq o'yiq) filtrlari ---
    min_hole_area: int = 6         # bundan kichigi tekstura teshigi
    crack_close_kernel: int = 3    # distress yoriqlarini yopish uchun


    # --- yopishib qolgan dog'larni ajratish ---
    # Dog' ichidagi o'yiqlar orasidagi x-bo'shliq (dog' balandligiga nisbatan)
    # shu qiymatdan katta bo'lsa, ular alohida belgilar deb ajratiladi.
    # O'lchangan: bitta belgi ichida <= 0.04, qo'shilgan dog'da >= 0.42.
    split_gap_ratio: float = 0.25
    # Ajratilgan bo'lak eng kattasining shu ulushidan kichik bo'lsa, u alohida
    # belgi emas (serif/nuqta) deb hisoblanadi va qaytarib qo'shiladi.
    min_split_area_ratio: float = 0.15

    # --- klassifikator ---
    norm_size: int = 32            # 0<->9/6/8 farqini ushlash uchun 24 dan oshirildi (piksellar ko'p)
    # LOIO baholashda kalibrlangan: to'g'ri tanishlar >= 0.88, yagona xato
    # (bank'da namunasi yo'q belgi) 0.71 bo'lgan. 0.80 — ikkisi orasidagi
    # xavfsiz chegara. Bank kengaygach `MATHOCR_MIN_CONFIDENCE` bilan
    # qayta sozlash mumkin.
    min_confidence: float = 0.80   # open-budget datasetida kalibrlangan; 0.80 dan pastda 0<->8 chalkashligi xato javob berishi mumkin, shuning uchun bu chegara qat'iy

    # --- grouping ---
    # Ikki raqam orasidagi bo'shliq shu qiymatdan katta bo'lsa ham ular
    # operator bilan ajratilmagan bo'lsa BITTA operand hisoblanadi (qoida 1/2
    # ustuvor), lekin bo'shliq juda katta bo'lsa ogohlantirish beriladi.
    max_intra_operand_gap_ratio: float = 2.2  # o'rtacha pitch'ga nisbatan


def _from_env() -> Config:
    """Ishlab chiqarishda qayta kompilyatsiyasiz sozlash uchun env override."""
    import os

    kwargs = {}
    for f in Config.__dataclass_fields__.values():
        raw = os.getenv("MATHOCR_" + f.name.upper())
        if raw is None:
            continue
        try:
            kwargs[f.name] = float(raw) if f.type == "float" else int(raw)
        except ValueError:
            continue
    return Config(**kwargs)


CFG = _from_env()
