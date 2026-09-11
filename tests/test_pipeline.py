"""Birlik testlari: grouping qoidalari, xavfsiz parser, quvur va API."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.grouping import Char, group  # noqa: E402
from app.parser import ExpressionError, compute  # noqa: E402
from app.pipeline import MathRecognizer  # noqa: E402

SAMPLES = ROOT / "samples"


# --------------------------------------------------------------------------
# 1. ENG MUHIM QOIDA: yonma-yon raqamlar = bitta son
# --------------------------------------------------------------------------

def chars(spec: str, pitch: int = 29, start: int = 16) -> list[Char]:
    """'19+60=' -> real captcha'dagiga o'xshash koordinatali Char ro'yxati."""
    out = []
    for i, ch in enumerate(spec):
        out.append(Char(char=ch, x=start + i * pitch, y=25, width=10, height=16,
                        confidence=0.99))
    return out


@pytest.mark.parametrize(
    "spec,left,op,right,result",
    [
        ("7+6=", 7, "+", 6, 13),
        ("19+60=", 19, "+", 60, 79),      # 1+9+6+0 EMAS
        ("4+16=", 4, "+", 16, 20),        # 4+1+6 EMAS
        ("25-17=", 25, "-", 17, 8),
        ("12x4=", 12, "×", 4, 48),
        ("36/6=", 36, "÷", 6, 6),
        ("123+456=", 123, "+", 456, 579),
        ("197+25=", 197, "+", 25, 222),
        ("4+168=", 4, "+", 168, 172),
        ("100/25=", 100, "÷", 25, 4),
        ("9-3=", 9, "-", 3, 6),
        ("7x12=", 7, "×", 12, 84),
        ("20x3=", 20, "×", 3, 60),
        ("18+10=", 18, "+", 10, 28),
        ("30-5=", 30, "-", 5, 25),
    ],
)
def test_operand_grouping(spec, left, op, right, result):
    g = group(chars(spec))
    assert g.valid, g.error
    assert int(g.left) == left
    assert int(g.right) == right
    expr = f"{g.left}{g.operator}{g.right}"
    assert compute(expr) == result


def test_equals_hisobga_olinmaydi():
    g = group(chars("19+60="))
    assert g.has_equals
    assert "=" not in f"{g.left}{g.operator}{g.right}"


def test_equalssiz_ham_ishlaydi_lekin_ogohlantiradi():
    g = group(chars("19+60"))
    assert g.valid and not g.has_equals
    assert any("=" in w for w in g.warnings)


def test_keng_bosh_liq_guruhlashni_buzmaydi():
    """Raqamlar orasi keng bo'lsa ham operator chegara bo'lib qoladi."""
    cs = chars("1960=")
    cs.insert(2, Char(char="+", x=cs[1].x + 60, y=25, width=10, height=16, confidence=0.99))
    for c in cs[3:]:
        c.x += 120
    g = group(cs)
    assert g.valid and g.left == "19" and g.right == "60"


# --------------------------------------------------------------------------
# 2. Yaroqsiz holatlar — taxmin qilishdan ko'ra rad etish
# --------------------------------------------------------------------------

@pytest.mark.parametrize("spec", ["19=", "+19=", "19+=", "1+2+3=", "="])
def test_yaroqsiz_ifodalar_rad_etiladi(spec):
    assert not group(chars(spec)).valid


# --------------------------------------------------------------------------
# 3. Xavfsiz parser — eval() yo'q
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "expr,val",
    [("19+60", 79), ("36/6", 6), ("12x4", 48), ("2x(3+4)", 14),
     ("100/25", 4), ("36÷6", 6), ("12×4", 48), ("7-9", -2),
     ("20*3", 60), ("0+5", 5), ("30-5", 25), ("2*1", 2), ("10/2", 5)],
)
def test_parser_hisoblaydi(expr, val):
    assert compute(expr) == val


@pytest.mark.parametrize(
    "evil",
    ["__import__('os').system('id')", "1+1;print(1)", "eval('1')", "2**64",
     "open('/etc/passwd')", "1+", "(1+2", "abc", "1 if 1 else 2"],
)
def test_parser_zararli_kirishni_rad_etadi(evil):
    with pytest.raises(ExpressionError):
        compute(evil)


def test_nolga_bolish():
    with pytest.raises(ExpressionError):
        compute("5/0")


def test_butun_bolinmasa_kasr_qaytadi():
    assert compute("7/2") == 3.5


# --------------------------------------------------------------------------
# 4. Uchidan-uchiga: haqiqiy sample rasmlar
# --------------------------------------------------------------------------

REAL = [
    ("f9555414-91a3-4939-bd1f-208af845ca35.webp", "7−6", 7, "-", 6, 1),
    ("a740762c-d57d-48b4-9b24-a0a586d6b383.webp", "4+16", 4, "+", 16, 20),
    ("d376ec0a-549f-4817-9bef-21e915bc19d9.webp", "19−7", 19, "-", 7, 12),
    ("ce71b009-8626-45dc-8ace-cc55de052a51.webp", "19−14", 19, "-", 14, 5),
    ("0803774a-bf81-4d22-81a4-81d58d278adf.webp", "11+18", 11, "+", 18, 29),
    ("cbfd93cc-ecf8-4187-a98d-17d15bc2b686.webp", "2×1", 2, "×", 1, 2),
]


@pytest.fixture(scope="module")
def service() -> MathRecognizer:
    return MathRecognizer()


@pytest.mark.parametrize("fname,_expr,left,op,right,result", REAL)
def test_real_samples(service, fname, _expr, left, op, right, result):
    res = service.process((SAMPLES / fname).read_bytes())
    assert res["valid"], res.get("error")
    assert res["left_operand"] == left
    assert res["operator"] == op
    assert res["right_operand"] == right
    assert res["result"] == result
    assert res["confidence"] >= 0.8


def test_bounding_boxlar_qaytadi(service):
    res = service.process((SAMPLES / REAL[1][0]).read_bytes())
    boxes = res["characters"]
    assert [c["char"] for c in boxes] == ["4", "+", "1", "6", "="]
    assert all(c["width"] > 0 and c["height"] > 0 for c in boxes)
    # chapdan o'ngga tartiblangan bo'lishi kerak
    assert [c["x"] for c in boxes] == sorted(c["x"] for c in boxes)


def test_shovqin_rasm_rad_etiladi(service):
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    noise = (rng.random((57, 180)) * 255).astype(np.uint8)
    res = service.process(cv2.imencode(".png", noise)[1].tobytes())
    assert not res["valid"]
    assert res["result"] is None


def test_bosh_rasm_rad_etiladi(service):
    import cv2
    import numpy as np

    blank = np.full((57, 180), 255, np.uint8)
    res = service.process(cv2.imencode(".png", blank)[1].tobytes())
    assert not res["valid"] and res["error"]


def test_buzuq_fayl_rad_etiladi(service):
    res = service.process(b"bu rasm emas")
    assert not res["valid"] and res["result"] is None


# --------------------------------------------------------------------------
# 5. Sintetik joylashuvlar (haqiqiy dog'lardan qayta yig'ilgan)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "expr,result",
    [("119-46", 73), ("741+18", 759), ("1+1", 2), ("1911+8", 1919),
     ("111+111", 222), ("98-761", -663), ("86-17", 69)],
)
@pytest.mark.parametrize("gap", [10, 6, 3, 0])
def test_kop_xonali_operandlar(service, expr, result, gap):
    from tests.synth import compose, encode

    res = service.process(encode(compose(expr, gap=gap, seed=3)))
    assert res["valid"], res.get("error")
    assert res["expression"] == expr
    assert res["result"] == result


# --------------------------------------------------------------------------
# 6. API
# --------------------------------------------------------------------------

#: Testlar uchun kalit — haqiqiy kalit emas, faqat shu fayl doirasida.
TEST_KEY = "mk_test_key"
AUTH = {"X-API-Key": TEST_KEY}


@pytest.fixture()
def api_client(monkeypatch):
    """Env'ga test kalitining dayjestini qo'yib, TestClient qaytaradi."""
    from fastapi.testclient import TestClient

    from app.api import app
    from app.auth import ENV_VAR, digest_of

    monkeypatch.setenv(ENV_VAR, digest_of(TEST_KEY))
    with TestClient(app) as client:
        yield client


def test_api_recognize(api_client):
    client = api_client
    assert client.get("/health").json()["status"] in {"ok", "partial"}
    data = (SAMPLES / REAL[1][0]).read_bytes()
    r = client.post("/recognize", files={"image": ("c.webp", data, "image/webp")}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["expression"] == "4+16" and body["result"] == 20

    import base64

    r2 = client.post("/recognize/base64",
                     json={"image_base64": base64.b64encode(data).decode()}, headers=AUTH)
    assert r2.json()["result"] == 20
    assert client.post("/recognize/base64",
                       json={"image_base64": "@@@"}, headers=AUTH).status_code == 400


def test_api_auth(api_client, monkeypatch):
    """Kalitsiz/xato kalit 401; /health esa doim ochiq."""
    client = api_client
    data = (SAMPLES / REAL[1][0]).read_bytes()
    files = {"image": ("c.webp", data, "image/webp")}

    assert client.post("/recognize", files=files).status_code == 401
    assert client.post("/recognize", files=files,
                       headers={"X-API-Key": "mk_boshqa"}).status_code == 401
    assert client.post("/click-captcha").status_code == 401
    assert client.get("/health").status_code == 200

    # FAIL-CLOSED: env bo'sh bo'lsa to'g'ri kalit ham o'tmaydi.
    from app.auth import ENV_VAR

    monkeypatch.setenv(ENV_VAR, "")
    assert client.post("/recognize", files=files, headers=AUTH).status_code == 401


def test_openapi_security_scheme(api_client):
    """Swagger'da "Authorize" tugmasi chiqishi uchun sxema e'lon qilinishi shart."""
    spec = api_client.get("/openapi.json").json()
    assert spec["components"]["securitySchemes"]["APIKeyHeader"]["name"] == "X-API-Key"
    assert spec["paths"]["/recognize"]["post"]["security"] == [{"APIKeyHeader": []}]
    assert "security" not in spec["paths"]["/health"]["get"]


# --------------------------------------------------------------------------
# 7. Ishlash regressiyasidan qo'riqlash
# --------------------------------------------------------------------------

def test_cv2_single_thread():
    """`app/__init__.py` cv2 ni bir oqimga tushirgan bo'lishi shart.

    20 yadroli serverda ko'p oqim so'rovni 2 ms dan 60-180 ms ga cho'zadi
    (2026-08-24 da o'lchangan). numpy/OpenBLAS tomonini unit faylidagi
    `OPENBLAS_NUM_THREADS=1` hal qiladi.
    """
    import cv2

    import app  # noqa: F401  — import yon ta'siri sifatida setNumThreads(1)

    assert cv2.getNumThreads() == 1


def test_predict_is_vectorised():
    """Belgi bo'yicha indeks oldindan qurilgan bo'lishi shart.

    Usiz `predict()` 2004 ta shablon ustidan Python sikliga qaytadi va
    bitta glyph 18 ms oladi (bitta rasm ~90 ms).
    """
    from app.pipeline import MathRecognizer

    rec = MathRecognizer().recognizer
    assert set(rec._label_index) == set(rec.labels)
    assert sum(len(v) for v in rec._label_index.values()) == len(rec.labels)
