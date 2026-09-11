"""Kelgan captcha rasmlarini javobi bilan saqlab boruvchi fon yozuvchisi.

MAQSAD — tezlik emas, KORPUS O'STIRISH. Ochiq ish: bankda `÷` (bo'lish)
belgisi yo'q, chunki 533 ta o'rganish rasmining birortasida uchramagan.
Real trafikdan yig'ilgan namunalar shu bo'shliqni to'ldiradi.

NEGA KESH EMAS: 533 ta haqiqiy captcha -> 533 ta noyob sha256 (o'lchangan,
2026-08-24). Generator har safar tasodifiy tekstura yasagani uchun bayt
darajasidagi kesh hech qachon topilmaydi. Shuning uchun bu modul natijani
qayta ishlatmaydi — faqat yozib boradi.

UCHTA KAFOLAT:
1. So'rovni SEKINLASHTIRMAYDI — yozuv fon oqimida, navbat orqali. Navbat
   to'lsa namuna tashlab yuboriladi (so'rov hech qachon kutmaydi).
2. Xizmatni YIQITMAYDI — har qanday xato yutiladi (fail-open). Disk to'lsa,
   ruxsat bo'lmasa yoki katalog yo'q bo'lsa xizmat oddiy ishlayveradi.
3. Diskni TO'LDIRMAYDI — umumiy hajm chegarasi bor, oshsa yozish to'xtaydi.

NIMA SAQLANADI: past ishonchli/yaroqsiz javoblarning HAMMASI (eng qimmatlisi —
`÷` aynan shu toifaga tushadi, chunki bankda yo'q belgi past ball beradi),
ishonchli javoblardan esa faqat ulush (`MATHOCR_COLLECT_SAMPLE`).
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

# --- sozlamalar (hammasi env orqali, qayta kompilyatsiyasiz) ---
ENABLED = os.getenv("MATHOCR_COLLECT", "1") not in {"0", "false", "no"}
ROOT = Path(os.getenv("MATHOCR_COLLECT_DIR", "/var/lib/mathocr/traffic"))
#: Shu balldan past ishonch "ko'rib chiqilsin" deb belgilanadi va DOIM saqlanadi.
LOW_CONF = float(os.getenv("MATHOCR_COLLECT_LOW_CONF", "0.90"))
#: Ishonchli javoblardan qanchasi saqlansin (0..1). Xilma-xillik uchun zaxira.
SAMPLE = float(os.getenv("MATHOCR_COLLECT_SAMPLE", "0.10"))
#: Umumiy hajm chegarasi. Oshsa yangi namuna yozilmaydi (eskisi o'chirilmaydi —
#: o'chirish qarorini odam qabul qilsin).
MAX_BYTES = int(os.getenv("MATHOCR_COLLECT_MAX_BYTES", str(512 * 1024 * 1024)))
#: Navbat chuqurligi. To'lsa namuna tashlanadi — so'rov KUTMAYDI.
QUEUE_SIZE = int(os.getenv("MATHOCR_COLLECT_QUEUE", "256"))

_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"BM", ".bmp"),
    (b"GIF8", ".gif"),
)


def _ext(data: bytes) -> str:
    """Rasm turini sarlavhasidan aniqlaydi (kengaytmaga ishonmaymiz)."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    for magic, ext in _MAGIC:
        if data.startswith(magic):
            return ext
    return ".bin"


class SampleCollector:
    """Fon oqimida rasm + javobni diskka yozadi."""

    def __init__(self) -> None:
        self._q: queue.Queue[tuple[bytes, dict[str, Any]]] = queue.Queue(maxsize=QUEUE_SIZE)
        self._thread: threading.Thread | None = None
        self._bytes = 0
        self._seen: set[str] = set()
        self.dropped = 0        # navbat to'lganda tashlangan
        self.written = 0
        self.skipped_full = 0   # hajm chegarasi tufayli yozilmagan
        self._disabled_reason: str | None = None

    # --- hayotiy sikl ---

    def start(self) -> None:
        if not ENABLED:
            self._disabled_reason = "MATHOCR_COLLECT=0"
            return
        try:
            ROOT.mkdir(parents=True, exist_ok=True)
            self._scan()
        except Exception as exc:  # ruxsat yo'q / disk / read-only fs
            self._disabled_reason = f"katalog ochilmadi: {exc}"
            print(f"MATHOCR collector o'chirildi: {exc}", file=sys.stderr)
            return
        self._thread = threading.Thread(target=self._loop, name="mathocr-collector", daemon=True)
        self._thread.start()

    def _scan(self) -> None:
        """Qayta ishga tushganda mavjud hajm va hash'larni hisobga oladi."""
        for p in ROOT.rglob("*"):
            if p.is_file() and p.suffix != ".jsonl":
                self._bytes += p.stat().st_size
                self._seen.add(p.stem)

    # --- yozish ---

    def submit(self, data: bytes, result: dict[str, Any]) -> None:
        """So'rov oqimidan chaqiriladi. HECH QACHON bloklamaydi va xato bermaydi."""
        if self._thread is None:
            return
        try:
            if not self._wanted(result):
                return
            self._q.put_nowait((data, result))
        except queue.Full:
            self.dropped += 1
        except Exception:  # pragma: no cover — hech qanday holatda so'rov buzilmasin
            pass

    def _wanted(self, result: dict[str, Any]) -> bool:
        """Past ishonchli/yaroqsizlar DOIM, qolganidan ulush."""
        if not result.get("valid", False):
            return True
        if float(result.get("confidence", 0.0)) < LOW_CONF:
            return True
        return random.random() < SAMPLE

    def _loop(self) -> None:
        while True:
            data, result = self._q.get()
            try:
                self._write(data, result)
            except Exception as exc:  # pragma: no cover
                print(f"MATHOCR collector yozuv xatosi: {exc}", file=sys.stderr)
            finally:
                self._q.task_done()

    def _write(self, data: bytes, result: dict[str, Any]) -> None:
        # Chegara YOZISHDAN OLDIN, kelajakdagi hajm bo'yicha tekshiriladi —
        # aks holda oxirgi fayl chegaradan oshib ketardi.
        if self._bytes + len(data) > MAX_BYTES:
            self.skipped_full += 1
            return
        sha = hashlib.sha256(data).hexdigest()
        if sha in self._seen:
            return  # ayni rasm allaqachon bor (retry)
        valid = bool(result.get("valid", False))
        conf = float(result.get("confidence", 0.0))
        bucket = "review" if (not valid or conf < LOW_CONF) else "ok"

        day = time.strftime("%Y-%m-%d")
        folder = ROOT / bucket / day
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{sha}{_ext(data)}"

        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)  # atomik — yarim yozilgan fayl ko'rinmaydi

        self._seen.add(sha)
        self._bytes += len(data)
        self.written += 1

        line = json.dumps(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "sha256": sha,
                "file": str(path.relative_to(ROOT)),
                "bucket": bucket,
                "expression": result.get("expression"),
                "result": result.get("result"),
                "confidence": conf,
                "valid": valid,
                "chars": [c.get("char") for c in result.get("characters", [])],
            },
            ensure_ascii=False,
        )
        # O_APPEND bilan qisqa qator atomik yoziladi — bir nechta uvicorn
        # worker bir vaqtda yozsa ham qatorlar aralashmaydi.
        with open(ROOT / "index.jsonl", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # --- holat ---

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self._thread is not None,
            "reason": self._disabled_reason,
            "dir": str(ROOT),
            "written": self.written,
            "dropped_queue_full": self.dropped,
            "skipped_size_limit": self.skipped_full,
            "bytes": self._bytes,
            "max_bytes": MAX_BYTES,
            "low_conf_threshold": LOW_CONF,
            "sample_rate": SAMPLE,
        }


collector = SampleCollector()
