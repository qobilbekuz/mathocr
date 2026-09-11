"""FastAPI xizmati.

Ishga tushirish (nginx orqasida, api.qobilbek.dev/mathocr/):
    uvicorn app.api:app --host 127.0.0.1 --port 8731 --workers 2 --root-path /mathocr

Autentifikatsiya: /recognize*, /click-captcha uchun `X-API-Key` shart — app/auth.py.

Endpoint'lar:
    POST /recognize        multipart/form-data, maydon nomi: image
    POST /recognize/base64 JSON: {"image_base64": "..."}
    GET  /health           tayyorlik va template bank holati (KALITSIZ — monitoring uchun)
    POST /click-captcha    body + instructions rasmlari -> koordinatalar (PROTOTIP)
"""
from __future__ import annotations

import base64
import binascii
import time
from typing import Any

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .auth import require_key
from .collector import collector
from .pipeline import MODEL_PATH, MathRecognizer
from .recognizer import CHARSET

MAX_BYTES = 2 * 1024 * 1024  # rasm 180x57 ~5 KB; 2 MB dan kattasi shubhali

_service: MathRecognizer | None = None
_load_error: str | None = None


def _init() -> None:
    """Template bankni yuklaydi. Xato bo'lsa xizmat ishlashda davom etadi,
    lekin /health `degraded` qaytaradi — deploy paytida model yo'qligi
    butun konteynerni yiqitmasligi uchun."""
    global _service, _load_error
    if _service is not None:
        return
    try:
        _service = MathRecognizer()
        _load_error = None
    except Exception as exc:
        _load_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init()  # sovuq startni oldini olish uchun oldindan yuklab qo'yamiz
    collector.start()  # namuna to'plovchi fon oqimi (xato bo'lsa jim o'chadi)
    yield


app = FastAPI(
    title="Math Captcha Recognition",
    version="1.0",
    description="Rasm -> matematik ifoda -> natija. CPU'da, tashqi OCR API'siz.",
    lifespan=lifespan,
)


def _svc() -> MathRecognizer:
    _init()
    if _service is None:
        raise HTTPException(status_code=503, detail=f"Model yuklanmagan: {_load_error}")
    return _service


def _handle(data: bytes, debug: bool) -> JSONResponse:
    if not data:
        raise HTTPException(status_code=400, detail="Bo'sh fayl")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Rasm juda katta")
    started = time.perf_counter()
    result: dict[str, Any] = _svc().process(data, debug=debug)
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    # Korpusni o'stirish uchun namunani fon oqimiga uzatamiz. `submit`
    # bloklamaydi va hech qachon xato bermaydi — so'rovga ta'sir qilmaydi.
    collector.submit(data, result)
    # Ishonchsiz tanish — bu server xatosi emas, shuning uchun 200 qaytadi;
    # mijoz `valid` maydonini tekshiradi.
    return JSONResponse(result)


@app.post("/recognize")
async def recognize(
    image: UploadFile = File(..., description="captcha rasmi (webp/png/jpg)"),
    debug: bool = Query(False, description="oraliq ma'lumotni ham qaytarish"),
    _key: str = Depends(require_key),
) -> JSONResponse:
    return _handle(await image.read(), debug)


class Base64Request(BaseModel):
    image_base64: str = Field(..., description="rasm baytlari base64 ko'rinishida")
    debug: bool = False


@app.post("/recognize/base64")
async def recognize_base64(
    req: Base64Request,
    _key: str = Depends(require_key),
) -> JSONResponse:
    payload = req.image_base64
    if "," in payload[:64] and payload.lstrip().startswith("data:"):
        payload = payload.split(",", 1)[1]  # data: URI prefiksini olib tashlash
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="base64 noto'g'ri")
    return _handle(data, req.debug)


@app.get("/health")
def health() -> dict[str, Any]:
    _init()
    if _service is None:
        return {"status": "degraded", "error": _load_error, "model": str(MODEL_PATH)}
    rec = _service.recognizer
    known = set(getattr(rec, "known_chars", []))
    missing = [c for c in CHARSET if c not in known and c != "="]
    return {
        "status": "ok" if not missing else "partial",
        "model": str(MODEL_PATH),
        "templates": len(getattr(rec, "vectors", [])),
        "known_chars": sorted(known),
        "missing_chars": missing,
        "samples_per_char": getattr(rec, "meta", {}).get("samples_per_char", {}),
        "collector": collector.stats(),
    }


# ==========================================================================
# Click-captcha (2 tokenni topib bosish) — PROTOTIP endpoint
# ==========================================================================
# DIQQAT: bu hozircha past aniqlikli PROTOTIP (~30%, 5 namuna asosida).
# Faqat koordinata qaytaradi, bosmaydi. Ko'proq namuna + matn-deteksiya
# modeli bilan yaxshilanishi kerak — batafsil: clickcaptcha/README.md

@app.post("/click-captcha")
async def click_captcha(
    body: UploadFile = File(..., description="sahna rasmi (345x230)"),
    instructions: UploadFile = File(..., description="ko'rsatma rasmi (210x70)"),
    _key: str = Depends(require_key),
) -> JSONResponse:
    try:
        from clickcaptcha.solver import locate
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"click yechuvchi yuklanmadi: {exc}")

    body_data = await body.read()
    instr_data = await instructions.read()
    if not body_data or not instr_data:
        raise HTTPException(status_code=400, detail="ikkala rasm ham kerak")
    if len(body_data) > MAX_BYTES or len(instr_data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="rasm juda katta")

    started = time.perf_counter()
    result: dict[str, Any] = locate(instr_data, body_data)
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    result["note"] = "PROTOTIP: aniqlik past (~30%), ko'proq namuna kutilmoqda"
    return JSONResponse(result)
