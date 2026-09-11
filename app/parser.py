"""Bosqich 8-9: xavfsiz ifoda parseri va hisoblagich.

`eval()` ISHLATILMAYDI. Bu yerda rekursiv-tushuvchi (recursive descent)
parser bor: u faqat quyidagi alifboni qabul qiladi

    0-9  +  -  ×  ÷  (  )

va istalgan boshqa belgida darhol xato beradi. Parser AST quradi, AST esa
faqat to'rtta arifmetik amalni biladi — ya'ni kirish satri qanday bo'lishidan
qat'i nazar, kod bajarilishi mumkin emas.
"""
from __future__ import annotations

from dataclasses import dataclass

ALLOWED = set("0123456789+-x/()")
CANON = {"×": "x", "*": "x", "÷": "/", ":": "/", "−": "-", "–": "-"}


class ExpressionError(ValueError):
    pass


@dataclass
class Num:
    value: int


@dataclass
class BinOp:
    op: str
    left: object
    right: object


def canonical(expr: str) -> str:
    out = []
    for ch in expr:
        ch = CANON.get(ch, ch)
        if ch.isspace():
            continue
        if ch not in ALLOWED:
            raise ExpressionError(f"ruxsat etilmagan belgi: {ch!r}")
        out.append(ch)
    return "".join(out)


class _Parser:
    def __init__(self, s: str):
        self.s = s
        self.i = 0

    def peek(self) -> str | None:
        return self.s[self.i] if self.i < len(self.s) else None

    def parse(self):
        node = self.expr()
        if self.i != len(self.s):
            raise ExpressionError(f"kutilmagan belgi pozitsiya {self.i}: {self.s[self.i]!r}")
        return node

    def expr(self):
        node = self.term()
        while self.peek() in ("+", "-"):
            op = self.s[self.i]
            self.i += 1
            node = BinOp(op, node, self.term())
        return node

    def term(self):
        node = self.factor()
        while self.peek() in ("x", "/"):
            op = self.s[self.i]
            self.i += 1
            node = BinOp(op, node, self.factor())
        return node

    def factor(self):
        ch = self.peek()
        if ch == "(":
            self.i += 1
            node = self.expr()
            if self.peek() != ")":
                raise ExpressionError("yopilmagan qavs")
            self.i += 1
            return node
        if ch is not None and ch.isdigit():
            start = self.i
            while self.peek() is not None and self.s[self.i].isdigit():
                self.i += 1
            return Num(int(self.s[start:self.i]))
        raise ExpressionError(
            f"son kutilgan edi, pozitsiya {self.i}: {ch!r}" if ch else "ifoda tugallanmagan"
        )


def evaluate(node) -> int | float:
    if isinstance(node, Num):
        return node.value
    left, right = evaluate(node.left), evaluate(node.right)
    if node.op == "+":
        return left + right
    if node.op == "-":
        return left - right
    if node.op == "x":
        return left * right
    if node.op == "/":
        if right == 0:
            raise ExpressionError("nolga bo'lish")
        if left % right == 0:
            return left // right
        return left / right
    raise ExpressionError(f"noma'lum amal: {node.op!r}")


def compute(expr: str) -> int | float:
    """Satrni xavfsiz hisoblaydi. Faqat ALLOWED alifbosi qabul qilinadi."""
    return evaluate(_Parser(canonical(expr)).parse())
