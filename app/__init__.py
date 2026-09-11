"""mathocr paketi.

DIQQAT — bu yerdagi `cv2.setNumThreads(1)` ishlash uchun KRITIK.

Serverda 20 yadro bor. OpenCV ham, numpy ostidagi OpenBLAS ham har bir
amal uchun 20 ta oqim ochadi. Bizning yuk esa juda kichik: 180x57 rasm va
2004x1024 matritsa-vektor ko'paytmasi — bir yadroda ~1.7 ms. Oqimlarni
sinxronlash (spin-barrier) foydali ishdan qimmatroq tushadi va server band
bo'lganda (mariadbd, php-fpm) bu oqimlar navbatga tushib, bitta so'rov
2 ms o'rniga 60-180 ms davom etadi.

O'lchangan: ko'p oqim -> 2.5..143 ms (beqaror), bir oqim -> 2.3..3.2 ms.
Parallellik uvicorn worker'lari darajasida bo'ladi, BLAS darajasida emas.

numpy/OpenBLAS tomonini env boshqaradi (`OPENBLAS_NUM_THREADS=1` va h.k.,
systemd unit faylida) — u IMPORTDAN OLDIN qo'yilishi shart, shuning uchun
kodda emas, unitda.
"""
try:  # cv2 bo'lmasa paket importi yiqilmasin (masalan hujjat yig'ishda)
    import cv2 as _cv2

    _cv2.setNumThreads(1)
except Exception:  # pragma: no cover
    pass
