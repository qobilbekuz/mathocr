"""X-API-Key autentifikatsiyasi.

Kalitlar env'da FAQAT sha256 dayjesti bo'lib turadi (`MATHOCR_API_KEYS_SHA256`),
kalitning o'zi hech qayerda saqlanmaydi — shu sababli `systemctl show mathocr`,
`ps auxe` yoki unit faylini o'qish orqali sizib chiqmaydi.

Redis ATAYLAB ishlatilmadi: qo'shni `nsfw-service` da kalitlar Redis DB 4 da
turadi va Redis o'chsa/tozalansa BARCHA mijoz 401 oladi (o'sha servisning
hal qilinmagan zaifligi). Env bilan bu nosozlik turi umuman yo'q.

FAIL-CLOSED: env bo'sh bo'lsa hech kim kira olmaydi (foydalanuvchi qarori,
2026-08-24) — aks holda env yo'qolib qolganda xizmat jimgina ochilib qolardi.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sys

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = "X-API-Key"
ENV_VAR = "MATHOCR_API_KEYS_SHA256"

#: Swagger UI'dagi "Authorize" tugmasi shu sxemadan paydo bo'ladi.
#: `auto_error=False` ataylab: sxema o'zi 401 qaytarmasin, tekshiruvni va
#: xato matnini `require_key()` boshqarsin.
api_key_scheme = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    description="`tools/apikey.py` bilan olingan kalit (`mk_...`).",
)

_UNAUTH_HEADERS = {"WWW-Authenticate": "ApiKey"}


def allowed_digests() -> set[str]:
    """Env'dagi vergul bilan ajratilgan sha256 dayjestlar ro'yxati."""
    raw = os.getenv(ENV_VAR, "")
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def digest_of(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


async def require_key(key: str | None = Security(api_key_scheme)) -> str:
    """Kalitni tekshiradi va uning qisqa identifikatorini qaytaradi (log uchun)."""
    digests = allowed_digests()
    if not digests:
        # Sababni mijozga aytmaymiz (konfiguratsiya holati sir), lekin
        # jurnalga yozamiz — aks holda 401 sababini topish qiyin bo'ladi.
        print(f"MATHOCR: {ENV_VAR} bo'sh — barcha so'rov rad etilmoqda", file=sys.stderr)
        raise HTTPException(status_code=401, detail="Kalit noto'g'ri", headers=_UNAUTH_HEADERS)

    if not key:
        raise HTTPException(
            status_code=401,
            detail=f"{API_KEY_HEADER} sarlavhasi kerak",
            headers=_UNAUTH_HEADERS,
        )

    supplied = digest_of(key)
    for allowed in digests:
        # compare_digest — dayjestlarni taqqoslashda vaqt bo'yicha sizishning oldini oladi.
        if hmac.compare_digest(supplied, allowed):
            return supplied[:12]

    raise HTTPException(status_code=401, detail="Kalit noto'g'ri", headers=_UNAUTH_HEADERS)
