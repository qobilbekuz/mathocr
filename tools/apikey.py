#!/usr/bin/env python3
"""mathocr uchun API kalit generatori.

    python3 tools/apikey.py            # yangi kalit + uning sha256 dayjesti
    python3 tools/apikey.py --digest mk_xxx   # mavjud kalitning dayjestini hisoblash

Kalitning O'ZI hech qayerda saqlanmaydi — faqat dayjest unit faylidagi
`MATHOCR_API_KEYS_SHA256` ga yoziladi. Kalitni yo'qotsangiz tiklab bo'lmaydi,
yangisi yaratiladi.
"""
from __future__ import annotations

import argparse
import hashlib
import secrets

PREFIX = "mk_"


def new_key() -> str:
    return PREFIX + secrets.token_urlsafe(32)


def digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--digest", metavar="KALIT", help="mavjud kalitning dayjestini chiqarish")
    a = ap.parse_args()

    if a.digest:
        print(digest(a.digest))
        return

    key = new_key()
    print("KALIT (bir marta ko'rsatiladi, saqlab qo'ying):")
    print(f"  {key}")
    print("\nUnit faylga yoziladigan dayjest:")
    print(f"  {digest(key)}")
    print("\nBir nechta kalit bo'lsa dayjestlar vergul bilan ajratiladi:")
    print('  Environment="MATHOCR_API_KEYS_SHA256=<dayjest1>,<dayjest2>"')


if __name__ == "__main__":
    main()
