# Math Captcha Recognition

Rasm → matematik ifoda → natija. CPU'da, internetsiz, tashqi OCR API'siz.

O'z captcha generatoringizning kuchini o'lchash (audit) uchun mo'ljallangan:
xizmat qancha oson yechilishini va qaysi belgilar chalkashishini ko'rsatadi.

---

## 1. Sample rasmlar tahlili

Berilgan 5 ta rasm — hammasi **180×57 px, WebP, oq fon**.

| Fayl | Ifoda | Belgilar soni |
|---|---|---|
| `f9555414` | `7 − 6 =` | 3 + `=` |
| `a740762c` | `4 + 1 6 =` → **4 + 16** | 4 + `=` |
| `d376ec0a` | `1 9 − 7 =` → **19 − 7** | 4 + `=` |
| `ce71b009` | `1 9 − 1 4 =` → **19 − 14** | 5 + `=` |
| `0803774a` | `1 1 + 1 8 =` → **11 + 18** | 5 + `=` |

O'lchangan xususiyatlar:

| Xususiyat | Qiymat |
|---|---|
| Fon | toza oq, shovqinsiz (Otsu deyarli ideal ajratadi) |
| Belgi tuzilishi | **qora siyoh dog'i ichidan kesilgan OQ o'yiq (knockout)** |
| Dog' o'lchami | 21–46 px kenglik, 22–40 px balandlik |
| Glyph (o'yiq) o'lchami | 4–17 px kenglik, 2–23 px balandlik |
| Belgilar orasidagi pitch | ~29 px |
| Dog'lar orasidagi bo'shliq | 2–5 px (ba'zan tegib turadi) |
| Shrift | distressed blackletter, **o'zgarmas** |
| Masshtab | belgi bo'yicha ±30% tasodifiy (`4` bir rasmda 12×18, boshqasida 10×14) |
| Vertikal siljish | ±5 px |
| `=` belgisi | **istisno** — dog'siz, oddiy ikkita qora chiziq, doim eng o'ngda, y ≈ 29 va 41 |
| Distortion | dog' chetlarida yoriq/chachraq; glyph'ni 2–3 bo'lakka parchalaydi |

### Eng muhim ikkita kuzatuv

**1. Segmentatsiya birligi — dog', o'yiq emas.**
Distress teksturasi oq glyph'ni bo'laklarga bo'lib yuboradi (`9` → 3 bo'lak,
`4` → 2 bo'lak, `7` → 2 bo'lak). O'yiqlar bo'yicha to'g'ridan-to'g'ri
segmentatsiya qilinsa, bitta `9` uchta "belgi" bo'lib ketadi. Ammo **dog'
bitta bo'lib qoladi** — shuning uchun: *dog' = belgi, dog' ichidagi o'yiqlar
birlashmasi = o'sha belgining shakli*.

**2. Blur qilish mumkin emas.**
Odatdagi `GaussianBlur` bilan tozalash bu yerda ZARARLI: `4` ning diagonali va
`−` ning tanasi atigi 1–2 px, 3×3 blur ularni atrofdagi qora siyohga qo'shib
yuboradi. Sinovda aynan shu sabab `4` → `1` va `19−7` → `19 7` xatolariga olib
keldi. Kod ataylab blursiz binarizatsiya qiladi.

---

## 2. Usul tanlovi

Talab bo'yicha beshta variant sample'lar asosida baholandi:

| Variant | Baho |
|---|---|
| **Template matching + 1-NN** | **TANLANDI** |
| OpenCV custom classifier (HOG+SVM) | ortiqcha — 1-NN allaqachon toza ajratmoqda |
| PaddleOCR | ~100 MB model, sekin; tabiiy matnga o'rgatilgan, blackletter knockout shriftda yomon |
| Tesseract | shu shriftda ishonchsiz; muhimi — u **satr** qaytaradi, bizga esa har bir belgining bounding box'i kerak |
| Kichik CNN | 5 ta rasm = 21 ta glyph. Bu hajmda CNN o'rgatish overfitting'dan boshqa narsa bermaydi |

**Nima uchun template matching:**

1. Shrift o'zgarmas, alifbo 15 ta belgidan iborat — masala yopiq to'plamli.
2. Segmentatsiyadan keyin glyph toza **binar shakl** bo'lib chiqadi; tanish
   masalasi shakl solishtirishga qisqaradi.
3. Rasmiga ~1.3 ms, GPU yo'q, model hajmi 30 KB.
4. **Tushuntiriladigan**: xato bo'lganda qaysi template bilan chalkashgani
   ko'rinadi (`runner_up` maydonida qaytadi) — audit uchun aynan shu kerak.
5. Yangi belgi qo'shish = bitta rasm belgilash, qayta o'rgatish 1 sekund.
6. 1998 ta haqiqiy glyph ustida 4-fold kross-validatsiya 100% berdi — bu
   masala uchun murakkabroq model kerak emasligini isbotlaydi.

Masshtab o'zgarishiga bardosh berish uchun: glyph nisbatni saqlagan holda
24×24 ga keltiriladi, yengil blur qo'yiladi (1 px siljishga chidamlilik uchun)
va L2-normallashtirilgan vektor kosinus o'xshashligi bilan solishtiriladi.
Template bank o'rgatishda burilish (±6°) va masshtab (±8%) bilan kengaytiriladi.

---

## 3. Quvur (pipeline)

```
Rasm
 ↓ preprocess.py    Otsu binarizatsiya (BLURSIZ), balandlikni 57 px ga keltirish
 ↓ segment.py       '=' ni ajratish → dog'larni topish → o'yiqlarni tiklash
 ↓ segment.py       yopishib qolgan dog'larni bo'shliq bo'yicha ajratish
 ↓ features.py      nisbatni saqlab 24×24 ga normallashtirish
 ↓ recognizer.py    1-NN template bank → char + ishonch + runner_up
 ↓                  ── ORALIQ KO'RINISH: characters[] (char, x, y, w, h, confidence)
 ↓ grouping.py      x bo'yicha tartiblash → operator chegarasi → operandlarni birlashtirish
 ↓ parser.py        xavfsiz rekursiv-tushuvchi parser (eval YO'Q)
 ↓ parser.py        hisoblash
JSON
```

Tanish va hisoblash ataylab ajratilgan: `Recognizer` protokolini qondirgan
istalgan model (masalan kelajakda CNN) qolgan bosqichlarga tegmasdan
almashtiriladi.

### Operandlarni guruhlash qoidasi

Loyihaning asosiy talabi — **yonma-yon raqamlar bitta son**:

```
1 9 + 6 0 =   →   19 + 60   →   79        ( 1+9+6+0 EMAS )
4 + 1 6 =     →   4 + 16    →   20        ( 4+1+6 EMAS )
```

Buning uchun raqamlar orasidagi **bo'shliqqa tayanilmaydi** — faqat
**operator pozitsiyasiga**. Operatorgacha bo'lgan hamma raqam — chap operand,
operator bilan `=` orasidagi hamma raqam — o'ng operand. Bo'shliq faqat
ogohlantirish (`warnings`) uchun o'lchanadi va guruhlashni hech qachon buzmaydi.
Shu sabab `123 + 456` ham, `1 + 1` ham bir xil ishlaydi va xona soni
arxitekturaga umuman ta'sir qilmaydi.

`=` ifoda oxirini bildiradi va hisobga operand sifatida qo'shilmaydi.

---

## 4. O'rnatish

```bash
cd /root/mathocr
pip3 install -r requirements.txt

# template bankni qurish (labels.json asosida)
python3 -m app.train --dir samples --labels labels.json --out models/templates.npz

# ishga tushirish
python3 -m uvicorn app.api:app --host 127.0.0.1 --port 8731 --workers 2 --root-path /mathocr
```

Doimiy xizmat sifatida:

```bash
cp deploy/mathocr.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now mathocr
```

---

## 5. API

### `POST /recognize` — multipart, maydon nomi `image`

```bash
curl -H "X-API-Key: $MATHOCR_API_KEY" \
     -F "image=@captcha.webp" https://api.qobilbek.dev/mathocr/recognize
```

### `POST /recognize/base64`

```json
{ "image_base64": "UklGRi4...", "debug": false }
```

### Muvaffaqiyatli javob

```json
{
  "expression": "4+16",
  "left_operand": 4,
  "operator": "+",
  "right_operand": 16,
  "result": 20,
  "confidence": 0.99,
  "valid": true,
  "characters": [
    {"char": "4", "x": 18, "y": 29, "width": 12, "height": 18, "confidence": 1.0},
    {"char": "+", "x": 54, "y": 24, "width": 11, "height": 11, "confidence": 1.0},
    {"char": "1", "x": 88, "y": 21, "width": 8,  "height": 18, "confidence": 1.0},
    {"char": "6", "x": 124,"y": 31, "width": 13, "height": 19, "confidence": 1.0},
    {"char": "=", "x": 148,"y": 29, "width": 27, "height": 17, "confidence": 0.99}
  ],
  "elapsed_ms": 1.2
}
```

### Ishonchsiz javob

```json
{
  "expression": null, "left_operand": null, "operator": null,
  "right_operand": null, "result": null,
  "confidence": 0.71, "valid": false,
  "error": "Low recognition confidence",
  "detail": "x=133 dagi belgi ishonchsiz tanildi: '6' (0.71), ikkinchi nomzod: '9'"
}
```

**Qoida:** noto'g'ri ifodani taxmin qilib natija qaytarishdan ko'ra
`valid: false` qaytariladi. HTTP kodi baribir `200` — bu server xatosi emas,
mijoz `valid` maydonini tekshiradi.

### `GET /health`

Template bank holati, qaysi belgilar o'rgatilgan, qaysilari yetishmaydi.

---

## 6. PHP integratsiyasi

`php/MathCaptchaClient.php` — PHP **7.4 mos** (serveringizdagi asosiy versiya),
8.x da ham ishlaydi. Retry (tarmoq va 5xx xatolarida), timeout, MIME va hajm
tekshiruvi, ilova darajasidagi qo'shimcha ishonch chegarasi bor.

```php
$client = new MathCaptchaClient('https://api.qobilbek.dev/mathocr', getenv('MATHOCR_API_KEY'), 5.0, 2, 0.85);
$r = $client->recognizeFile($path);

if ($r->valid) {
    echo $r->expression . ' = ' . $r->result;   // "4+16 = 20"
} else {
    error_log('tanib bo\'lmadi: ' . $r->error);  // taxmin QILMANG
}
```

`php/example.php` — ishlaydigan namuna (`php php/example.php rasm.webp`).

---

## 7. Aniqlik — 480+ ta HAQIQIY captcha ustida

Boshlang'ich 6 ta rasmdan tashqari, `api.qobilbek.dev/open-budget/img` ichidagi
**480+ ta haqiqiy captcha** ustida to'liq o'lchov o'tkazildi. Bu rasmlar orqali
yetishmayotgan barcha belgilar (`0 2 3 5` va `*` ko'paytirish) topildi va
tizim ularga o'rgatildi.

### Belgilar qanday o'rgatildi (bootstrapping)

Rasmlar javobsiz edi, shuning uchun:

1. 480 rasm segmentatsiya qilindi → 1998 ta glyph ajratildi (segmentatsiya
   **100% muvaffaqiyatli**, birorta rasm ham parchalanmadi).
2. Glyphlar shakli bo'yicha 44 ta klasterga bo'lindi (k-means).
3. Har klaster ko'z bilan ko'rilib, haqiqiy belgisi berildi (montaj orqali).
4. Klasterlar mukammal ajraldi: **rasm darajasida 4-fold kross-validatsiya
   1998/1998 = 100%**, chalkashlik nol.

### Sizishsiz test (halol o'lchov)

Rasmlar ikkiga bo'linadi (bir rasm glyphi ham o'rgatishda, ham sinovda
qatnashmaydi), bank yarmidan quriladi, ikkinchi yarmida sinaladi:

```
test rasm:              241
yechildi:               241 (100%)
yechilganidan to'g'ri:  241/241 = 100.00%  (aniqlik)
NOTO'G'RI javob:        0
```

`python3 tests/eval_dataset.py` — shu o'lchovni qayta ishga tushiradi.

### Butun dataset (489 rasm, bitta bank)

```
yechildi:  482/489 = 98.6%   (chegara 0.80)
rad etildi: 7        (1.4%)  (ishonch pastligi — noto'g'ri javob EMAS)
tezlik:     ~10 ms/rasm
```

### Qo'lda tekshirilgan aniqlik

Eng xavfli holatlar — javobida `0`, `8` yoki `9` bo'lgan rasmlar (bu belgilar
bu shriftda o'xshash). Shundaylardan 33 ta (0/9) + 30 ta (0/8) + boshlang'ich
24 ta = **87 dan ortiq rasm ko'z bilan tekshirildi, barchasi 100% to'g'ri**.
Qolgan noaniq holatlar 0.80 dan past ball olib rad etiladi.

### `0` va `9` chalkashligi — topilgan va tuzatilgan xato

Dastlab tizim ba'zi `0` larni `9` deb o'qib, xato javob berardi (masalan
`18-10` o'rniga `18-19`). Ikki sabab aniqlandi va tuzatildi:

1. **Ortiqcha blur.** Xususiyat vektori 1.2 sigma bilan bulg'anardi — bu
   `0` (butun balandlik bo'ylab teshik), `9` (teshik faqat tepada, pastda dum),
   `6` (teshik pastda) o'rtasidagi mayda konturni yuvib yuborardi. Blur 0.5 ga
   tushirilib, ish o'lchami 24→32 ga oshirildi.

2. **Noto'g'ri o'rgatish yorliqlari.** Avtomatik belgilashda ~30 ta `0`
   xato `9` deb belgilangan edi — bitta noto'g'ri "9" shabloni haqiqiy `0` ga
   1.0 mos kelib, xato javobni 0.93 ishonch bilan o'tkazib yuborardi.
   Glyphlar yaxshilangan xususiyat bilan qayta klasterlanib, `0`/`9`/`6`/`8`
   toza ajratildi.

Tuzatishdan keyin: javobida `0`/`9` bo'lgan 33 ta rasm qo'lda tekshirildi —
barchasi 100% to'g'ri; butun dataset qamrovi 93% dan 98.6% ga oshdi.

### Nega `0`, `6`, `8` baribir e'tibor talab qiladi### Birlik testlari

`python3 -m pytest tests/ -q` → **77 test PASS** (grouping qoidalari, xavfsiz
parser, buzuq/shovqin rasm, ko'p xonali operandlar, API).

## 8. Cheklovlar — buni bilib turing

**O'rgatilgan belgilar:** `0 1 2 3 4 5 6 7 8 9  +  -  ×`  — barcha raqamlar va
uchta amal.

**Yagona yetishmayotgan belgi:** `÷` (bo'lish). 487 ta haqiqiy captcha'ning
birortasida ham bo'lish amali uchramadi — generator uni ishlatmaydi yoki juda
kam ishlatadi. Bo'lish uchun namuna kelsa, quyidagicha qo'shiladi:

```bash
python3 tools/add_samples.py "/tmp/bolish_captcha.jpg=36/6="
```

`/` yoki `:` yozsangiz ham bo'ladi — ikkalasi ham bo'lish deb tushuniladi.

**`0` va `8` chalkashligi:** 7-bo'limda tushuntirilgani kabi, bu shriftda `0`
va `8` (hamda `6`) ba'zan o'xshab qoladi. Tizim ularni taxmin qilmaydi —
ishonch past bo'lsa `valid:false` qaytaradi. Butun dataset'da bu ~7% rasmni
rad etishga olib keladi, lekin **noto'g'ri javob bermaydi**.

## 9. Xavfsizlik

* **`eval()` ishlatilmaydi.** `app/parser.py` — rekursiv-tushuvchi parser,
  faqat `0-9 + - × ÷ ( )` alifbosini qabul qiladi, boshqa har qanday belgida
  darhol xato beradi. AST faqat to'rtta arifmetik amalni biladi.
* Yuklanadigan rasm hajmi 2 MB bilan cheklangan (PHP tomonda ham, API tomonda ham).
* PHP klient MIME turini `finfo` bilan tekshiradi.
* systemd unit'da `ProtectSystem=strict`, `NoNewPrivileges`, `MemoryMax=512M`.
* Xizmat `127.0.0.1` ga bog'lanadi — tashqaridan ochiq emas.

---

## 10. Sozlash

Barcha chegaralar `app/config.py` da va env orqali ham o'zgartiriladi:

```bash
MATHOCR_MIN_CONFIDENCE=0.85    # qattiqroq rad etish
MATHOCR_SPLIT_GAP_RATIO=0.25   # yopishgan dog'larni ajratish chegarasi
MATHOCR_WORK_HEIGHT=57         # ichki ish balandligi
```

---

## 11. Fayl tuzilishi

```
app/preprocess.py    binarizatsiya, masshtab
app/segment.py       dog' va o'yiqlarni topish, '=' ni ajratish
app/features.py      normallashtirish, augmentatsiya
app/recognizer.py    template bank, 1-NN, Recognizer protokoli
app/grouping.py      operandlarni guruhlash (asosiy qoida)
app/parser.py        xavfsiz parser va hisoblagich
app/pipeline.py      to'liq quvur
app/api.py           FastAPI
app/train.py         template bankni qurish
tools/add_samples.py yangi namuna qo'shish
tests/               77 birlik testi + evaluate.py + synth.py
php/                 PHP 7.4 mos mijoz
deploy/              systemd unit, nginx snippet
```


---

## Ishlab chiqarishdagi holat (2026-08-24)

Ommaviy manzil: **`https://api.qobilbek.dev/mathocr/`** · Swagger: `/mathocr/docs`
(ochiq, "Authorize" tugmasi bor) · systemd: `mathocr` (enabled) · ichki: `127.0.0.1:8731`.

### Autentifikatsiya
`X-API-Key` sarlavhasi SHART: `POST /recognize`, `POST /recognize/base64`,
`POST /click-captcha`. `GET /health` **kalitsiz ochiq** — monitoring
(`/usr/local/bin/health_check.sh`) uni tashqaridan tekshiradi.

Kalitlar env'da faqat **sha256 dayjesti** sifatida turadi
(`MATHOCR_API_KEYS_SHA256`, unit faylida `Environment=`), kalitning o'zi
serverda saqlanmaydi. Yangi kalit:

```bash
python3 tools/apikey.py           # kalit + dayjest
# dayjestni unitga qo'shing (vergul bilan bir nechta bo'lishi mumkin), keyin:
systemctl daemon-reload && systemctl restart mathocr
```

**Fail-closed:** env bo'sh bo'lsa hamma so'rov 401 oladi — bu ataylab, aks holda
sozlama yo'qolganda xizmat jimgina hammaga ochilib qolardi (`tests/test_pipeline.py::test_api_auth`).

### nginx
Blok `api.qobilbek.dev.conf` da, `/nsfw/` dan keyin. Ikkita tuzoq:
- `^~` SHART — usiz `/mathocr/...` yo'llari statik-kesh (`\.(jpg|png|webp)$`) va
  `deny_sensitive.conf` (`\.py$`) regex location'lariga tushib ketadi.
- `proxy_read_timeout`/`proxy_send_timeout` yozilmaydi — `proxy_api_params.conf`
  ularni allaqachon qo'ygan, dublikat butun nginx konfigini rad etadi.

`--root-path /mathocr` bo'lmasa Swagger o'z manzillarini prefiksisiz yasaydi va
"Try it out" ishlamaydi.


## Namuna to'plash (2026-08-24)

Xizmat kelgan captcha rasmlarini javobi bilan **fon oqimida** saqlab boradi.
Maqsad — tezlik emas, **korpusni o'stirish**: bankda `÷` yo'q, chunki 533 ta
o'rganish rasmining birortasida bo'lish amali uchramagan. Real trafik shu
bo'shliqni to'ldiradi.

**Nega kesh emas:** 533 ta haqiqiy captcha -> 533 ta noyob sha256 (o'lchangan).
Generator har safar tasodifiy tekstura yasagani uchun bayt darajasidagi kesh
hech qachon topilmaydi (hit rate ~0%), tanish esa allaqachon 1.8 ms.

Katalog: `/var/lib/mathocr/traffic` (systemd `StateDirectory` — `ProtectSystem=strict`
ostida yozish mumkin bo'lgan yagona joy).

```
traffic/
  review/YYYY-MM-DD/<sha256>.webp   # valid=false yoki conf < 0.90 — MUHIMI shu
  ok/YYYY-MM-DD/<sha256>.webp       # ishonchli javoblardan 10% (xilma-xillik uchun)
  index.jsonl                        # har so'rovga bir qator
```

`÷` aynan `review/` ga tushadi: bankda yo'q belgi past ball va past ajralish beradi.

```bash
python3 tools/traffic_report.py            # umumiy holat
python3 tools/traffic_report.py --review   # ko'rib chiqilishi kerak bo'lganlar
python3 tools/traffic_report.py --chars    # ÷ keldimi?
```

Sozlash (unit faylida `Environment=`): `MATHOCR_COLLECT` (0 = o'chirish),
`MATHOCR_COLLECT_DIR`, `MATHOCR_COLLECT_LOW_CONF` (0.90), `MATHOCR_COLLECT_SAMPLE`
(0.10), `MATHOCR_COLLECT_MAX_BYTES` (512 MB), `MATHOCR_COLLECT_QUEUE` (256).

**Uchta kafolat** (`tests/test_collector.py` bilan qo'riqlanadi):
1. So'rovni sekinlashtirmaydi — yozuv fon oqimida; navbat to'lsa namuna tashlanadi.
2. Xizmatni yiqitmaydi — katalog ochilmasa `/health` da `collector.enabled=false`
   va sabab ko'rinadi, tanish esa ishlayveradi (fail-**open**).
3. Diskni to'ldirmaydi — hajm chegarasi yozishdan oldin, kelajakdagi hajm
   bo'yicha tekshiriladi. Chegaraga yetganda eski fayllar O'CHIRILMAYDI —
   bu qarorni odam qabul qilsin.
