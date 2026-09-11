"""Bosqich 4b: belgi tanish (character recognition).

Usul: normallashtirilgan bitmap ustida kosinus o'xshashligi bo'yicha
eng yaqin qo'shni (template bank / 1-NN).

Nega aynan shu usul — README'da batafsil; qisqacha: shrift o'zgarmas,
belgilar to'plami 15 ta, segmentatsiyadan keyin glyph toza binar shakl
bo'lib chiqadi. Bunday sharoitda 1-NN deyarli xatosiz ishlaydi, CPU'da
mikrosoniyalarda hisoblanadi va eng muhimi — TUSHUNTIRILADIGAN: noto'g'ri
tanigan holatda qaysi template bilan chalkashgani ko'rinib turadi.

Interfeys `Recognizer` — keyinchalik CNN'ga almashtirish uchun shu
protokolni qondirish yetarli.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import CFG
from . import features

CHARSET = "0123456789+-x/="  # ichki alifbo; 'x' -> '×', '/' -> '÷'
DISPLAY = {"x": "×", "/": "÷"}


@dataclass
class Prediction:
    char: str
    confidence: float
    runner_up: str | None = None
    runner_up_score: float = 0.0


class Recognizer(Protocol):
    def predict(self, mask: np.ndarray) -> Prediction: ...


class TemplateRecognizer:
    """Template bank ustida 1-NN."""

    def __init__(self, vectors: np.ndarray, labels: list[str], meta: dict | None = None):
        self.vectors = vectors.astype(np.float32)
        self.labels = list(labels)
        self.meta = meta or {}
        self._reindex()

    def _reindex(self) -> None:
        """Har bir belgi uchun uning shablonlari indeksini oldindan hisoblaydi.

        `predict()` da har bir belgining eng yaxshi bali kerak. Buni 2004 ta
        shablon ustidan Python sikli bilan hisoblash 18 ms olardi (bitta
        glyph uchun!) — numpy skalyarini har safar Python obyektiga o'rash
        qimmat. Belgi bo'yicha guruhlangan indeks bilan bu 13 ta vektor
        amaliga aylanadi va ~100 barobar tezlashadi."""
        idx: dict[str, list[int]] = {}
        for i, lab in enumerate(self.labels):
            idx.setdefault(lab, []).append(i)
        self._label_index = {lab: np.asarray(v, dtype=np.intp) for lab, v in idx.items()}

    # --- yuklash / saqlash ---

    @classmethod
    def load(cls, path: str | Path) -> "TemplateRecognizer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"template bank topilmadi: {path}. Avval `python -m app.train` ishga tushiring."
            )
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"])) if "meta" in data else {}
        return cls(data["vectors"], [str(x) for x in data["labels"]], meta)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self.vectors,
            labels=np.array(self.labels),
            meta=json.dumps(self.meta, ensure_ascii=False),
        )

    @property
    def known_chars(self) -> list[str]:
        return sorted(set(self.labels))

    # --- tanish ---

    def predict(self, mask: np.ndarray) -> Prediction:
        if len(self.vectors) == 0:
            return Prediction(char="?", confidence=0.0)
        v = features.vector(mask)
        sims = self.vectors @ v  # kosinus o'xshashlik (ikkalasi ham L2-normal)

        # Har bir belgi uchun eng yaxshi ball. Belgi bo'yicha guruhlangan
        # indeks (`_reindex`) tufayli bu 13 ta numpy amali — shablonlar
        # ustidan Python sikli EMAS. Natija bitma-bit bir xil.
        best: dict[str, float] = {
            lab: float(sims[idx].max()) for lab, idx in self._label_index.items()
        }
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)

        top_char, top_score = ranked[0]
        second_char, second_score = ranked[1] if len(ranked) > 1 else (None, 0.0)
        margin = top_score - second_score

        # Ishonch = moslik sifati x ajralish darajasi.
        # Ikkala omil ham kerak: yuqori ball, lekin ikkinchi nomzod ham
        # shunchalik yaqin bo'lsa — bu ishonchli tanish emas.
        separation = float(np.clip(margin / 0.06, 0.0, 1.0)) ** 0.5
        conf = float(np.clip(top_score, 0.0, 1.0)) * (0.55 + 0.45 * separation)
        return Prediction(
            char=top_char,
            confidence=round(conf, 4),
            runner_up=second_char,
            runner_up_score=round(second_score, 4),
        )


class EqualsRecognizer:
    """'=' strukturaviy aniqlanadi, klassifikatsiya kerak emas."""

    @staticmethod
    def predict(bars: int) -> Prediction:
        return Prediction(char="=", confidence=0.99 if bars >= 2 else 0.75)
