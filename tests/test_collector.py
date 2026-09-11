"""Namuna to'plovchi testlari.

Eng muhim talab — to'plovchi HECH QANDAY holatda so'rovni buzmasligi kerak.
Shuning uchun testlarning yarmi "yomon sharoit" (katalog yo'q, navbat to'la,
yozuv xatosi) da xizmat ishlayverishini tekshiradi.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import collector as C  # noqa: E402


def _mk(tmp_path, monkeypatch, *, low_conf=0.90, sample=1.0, max_bytes=10**9):
    monkeypatch.setattr(C, "ROOT", tmp_path / "traffic")
    monkeypatch.setattr(C, "ENABLED", True)
    monkeypatch.setattr(C, "LOW_CONF", low_conf)
    monkeypatch.setattr(C, "SAMPLE", sample)
    monkeypatch.setattr(C, "MAX_BYTES", max_bytes)
    c = C.SampleCollector()
    c.start()
    return c


def _flush(c):
    c._q.join()


PNG = b"\x89PNG\r\n\x1a\n" + b"soxta-rasm-baytlari"
WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"yana-baytlar"


def test_ok_va_review_ajratiladi(tmp_path, monkeypatch):
    c = _mk(tmp_path, monkeypatch)
    c.submit(PNG, {"valid": True, "confidence": 0.99, "expression": "1+1",
                   "result": 2, "characters": [{"char": "1"}]})
    c.submit(WEBP, {"valid": False, "confidence": 0.4, "expression": None,
                    "result": None, "characters": []})
    _flush(c)

    root = tmp_path / "traffic"
    assert len(list((root / "ok").rglob("*.png"))) == 1
    assert len(list((root / "review").rglob("*.webp"))) == 1

    rows = [json.loads(x) for x in (root / "index.jsonl").read_text().splitlines()]
    assert {r["bucket"] for r in rows} == {"ok", "review"}


def test_past_ishonch_review_ga_tushadi(tmp_path, monkeypatch):
    """`÷` aynan shu yo'l bilan tutiladi: bankda yo'q belgi past ball beradi."""
    c = _mk(tmp_path, monkeypatch, low_conf=0.90, sample=0.0)
    c.submit(PNG, {"valid": True, "confidence": 0.55, "characters": []})
    _flush(c)
    files = [p for p in (tmp_path / "traffic" / "review").rglob("*") if p.is_file()]
    assert len(files) == 1


def test_ayni_rasm_ikki_marta_yozilmaydi(tmp_path, monkeypatch):
    c = _mk(tmp_path, monkeypatch)
    payload = {"valid": True, "confidence": 0.99, "characters": []}
    for _ in range(4):
        c.submit(PNG, dict(payload))
    _flush(c)
    assert c.written == 1


def test_hajm_chegarasi_hurmat_qilinadi(tmp_path, monkeypatch):
    c = _mk(tmp_path, monkeypatch, max_bytes=1)
    c.submit(PNG, {"valid": True, "confidence": 0.99, "characters": []})
    _flush(c)
    assert c.written == 0 and c.skipped_full == 1


def test_katalog_ochilmasa_xizmat_ishlayveradi(tmp_path, monkeypatch):
    """FAIL-OPEN: yozib bo'lmaydigan yo'l — `submit` jim o'tishi shart."""
    monkeypatch.setattr(C, "ROOT", Path("/proc/yoq/bunday/katalog"))
    monkeypatch.setattr(C, "ENABLED", True)
    c = C.SampleCollector()
    c.start()
    assert c.stats()["enabled"] is False
    assert c.stats()["reason"]
    c.submit(PNG, {"valid": True, "confidence": 0.99})  # xato bermasligi kerak


def test_submit_hech_qachon_xato_bermaydi(tmp_path, monkeypatch):
    """Buzuq natija lug'ati kelsa ham so'rov yiqilmasin."""
    c = _mk(tmp_path, monkeypatch)
    for bad in ({}, {"valid": "ha"}, {"confidence": None}, {"characters": None}):
        c.submit(PNG, bad)  # istisno chiqmasligi kerak


def test_navbat_tolganda_tashlanadi(tmp_path, monkeypatch):
    """Navbat to'lsa so'rov KUTMAYDI — namuna tashlab yuboriladi."""
    monkeypatch.setattr(C, "QUEUE_SIZE", 1)
    c = _mk(tmp_path, monkeypatch)
    c._thread = object()          # ishchi oqimni to'xtatib turamiz
    c._q = C.queue.Queue(maxsize=1)
    payload = {"valid": True, "confidence": 0.99, "characters": []}
    for _ in range(5):
        c.submit(PNG, dict(payload))
    assert c.dropped >= 3


def test_rasm_turi_sarlavhadan_aniqlanadi():
    assert C._ext(WEBP) == ".webp"
    assert C._ext(PNG) == ".png"
    assert C._ext(b"\xff\xd8\xff\xe0") == ".jpg"
    assert C._ext(b"nomalum") == ".bin"
