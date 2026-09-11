# Click-captcha (2 tokenni topib bosish) — tahlil

## Captcha mexanizmi (to'liq ochilgan)

Foydalanuvchiga 2 rasm beriladi:

| Rasm | O'lcham | Mazmun |
|------|---------|--------|
| **body** (sahna) | 345×230 JPEG | manzara, ichiga ~7-9 ta 2-harfli token yarim-shaffof singdirilgan |
| **instructions** | 210×70 PNG | oq fon, ustida **2 ta nishon token** (+ chalg'ituvchi nuqta va egri chiziq) |

**Javob** = nishon tokenlarning sahnadagi koordinatalari, **ko'rsatma tartibida**:

```json
{"coordinates": [{"x": 289, "y": 70}, {"x": 123, "y": 79}]}
```

Tekshirilgan (row2): ko'rsatma "KY EP" → javob[0]=(289,70)=KY joyi, javob[1]=(123,79)=EP joyi.

5 namunadagi nishonlar: `MC KI` / `OZ SN` / `KY EP` / `XR UO` / `LR NU`.

## Yechuvchi arxitekturasi

```
instructions ──► ikkita nishon token-rasmga ajratish (split_instruction)
                                                          │
body ──────────► sahnadagi token-nomzodlarni aniqlash (scene_candidates)
                                                          │
                 har nishonni nomzodlar bilan qirra-mosligi bo'yicha
                 solishtirish (locate) ──► (x,y) koordinatalar
```

Harf OCR ishlatilmaydi — nishon token-RASMI to'g'ridan-to'g'ri sahnada
qidiriladi (rangdan mustaqil, qirralarga asoslangan, ko'p masshtabli).

## Hozirgi natija (5 namuna, klassik CV)

`python3 clickcaptcha/evaluate.py clickcaptcha/samples.csv`

```
nuqta darajasida aniqlik:  3/10 = 30%   (22px ichida)
to'liq (ikkala nuqta):     1/5
token aniqlash recall:     4/10 = 40%
```

## Asosiy topilma (audit uchun muhim)

**Bu captcha avtomatik yechishga qarshi KUCHLI.** Sabab — tokenlar ataylab
kamuflyaj qiladigan fonga joylashtirilgan:

- **Osmondagi tokenlar** (och fon) oson topiladi.
- **O't / buta / tosh fonidagi tokenlar** deyarli ko'rinmaydi — klassik
  aniqlash usullari (yuqori-o'tkazgich, kanal-qirralari, CLAHE) ularni
  60% holatda umuman topa olmadi.

Ya'ni bu captcha matematik captchadan **ancha kuchliroq**: matematikni
klassik 1-NN deyarli 100% yechardi; bu yerda esa klassik usul ~20-30% da
qoladi.

## Nima uchun bu chegara (tub sabab)

Nishon topishning asosiy to'sig'i — **aniqlash (detection)**, tanish emas.
Tokenlar past-kontrastli, teksturali fon bilan qo'shilib ketadi. Buni
klassik piksel/qirra usullari bilan yechib bo'lmaydi — bu aynan **o'rgatilgan
matn-deteksiya modellari** (CRAFT, DBNet, EAST) uchun mo'ljallangan masala.

## Ishonchli yechuvchi uchun nima kerak

1. **Ko'proq namuna: 100-200+** (rasm + ko'rsatma + javob, xuddi shu CSV
   formatida). Backend allaqachon saqlayapti — shu jadvaldan eksport qiling.
2. **Matn-deteksiya modeli** (CRAFT/DBNet) — kamuflyajlangan tokenlarni
   topish uchun. Kerak bo'lsa shu namunalarda fine-tune qilinadi.
3. So'ng: har token kesib olinib, harf-klassifikatori (matematik bankdagidek
   template usuli, endi A-Z uchun) bilan o'qiladi va ko'rsatma nishonlariga
   moslashtiriladi.

## Fayllar

```
clickcaptcha/solver.py     yechuvchi (locate() koordinata qaytaradi, BOSMAYDI)
clickcaptcha/evaluate.py   CSV ground-truth ustida baholash
clickcaptcha/samples.csv   5 ta namuna (javoblari bilan)
```

FastAPI endpoint: `POST /click-captcha` (body + instructions rasmlarini oladi,
koordinata qaytaradi). Hozircha PROTOTIP — aniqligi past, ko'proq ma'lumot
kutilmoqda.
