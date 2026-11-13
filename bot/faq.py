"""Frequently asked questions, kept in a plain text file anyone can edit.

File format (data/faq.txt): blocks separated by an empty line. In each block,
lines starting with "س:" are different ways of asking the same question, and
the lines from "ج:" onward are the answer. Lines starting with # are ignored.

    س: بتقبلوا الدفع بالفيزا؟
    س: ينفع ادفع بالكارت؟
    ج: الحجز هنا تجريبي ومفيش دفع حقيقي.
"""
import re
from pathlib import Path

from .textutil import tokenize

_Q = ("س:", "س :", "Q:", "q:")
_A = ("ج:", "ج :", "A:", "a:")
_BOOKING_REF = re.compile(r"TB-[A-Z0-9]{6}", re.I)  # a real request, not a question


def parse(text: str) -> list[dict]:
    """Returns [{'questions': [...], 'answer': '...'}]; blocks missing either part are skipped."""
    entries, questions, answer, in_answer = [], [], [], False

    def flush():
        nonlocal questions, answer, in_answer
        if questions and answer:
            entries.append({"questions": questions, "answer": "\n".join(answer).strip()})
        questions, answer, in_answer = [], [], False

    for raw in text.replace("﻿", "").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        if not line:
            if in_answer and answer:
                answer.append("")  # keep paragraph breaks inside an answer
            elif not in_answer:
                questions = []     # a question block with no answer is dropped
            continue
        q = next((p for p in _Q if line.startswith(p)), None)
        a = next((p for p in _A if line.startswith(p)), None)
        if q:
            if in_answer:
                flush()
            if line[len(q):].strip():
                questions.append(line[len(q):].strip())
        elif a:
            if in_answer:
                flush()
            in_answer = True
            if line[len(a):].strip():
                answer.append(line[len(a):].strip())
        elif in_answer:
            answer.append(line)
    flush()
    return entries


class FAQ:
    MIN_SCORE = 0.6

    def __init__(self, path):
        path = Path(path)
        self.entries = parse(path.read_text(encoding="utf-8")) if path.exists() else []
        self._index = [(set(tokenize(q)), e) for e in self.entries for q in e["questions"]]

    def questions(self, limit: int | None = None) -> list[str]:
        """The first wording of each question, for showing as buttons."""
        first = [e["questions"][0] for e in self.entries]
        return first[:limit] if limit else first

    def match(self, text: str) -> str | None:
        """The stored answer when the text is close enough to a known question."""
        words = set(tokenize(text))
        if not words or _BOOKING_REF.search(text):
            return None
        best, answer = 0.0, None
        for known, entry in self._index:
            if not known:
                continue
            shared = len(words & known)
            score = shared / len(words | known)                     # overall similarity
            if len(known) >= 2 and shared == len(known):            # the whole stored question
                score = max(score, 0.5 + 0.5 * shared / len(words)) # appears inside the text
            if len(words) >= 2 and shared == len(words):            # the text is a short form
                score = max(score, 0.5 + 0.5 * shared / len(known)) # of the stored question
            if score > best:
                best, answer = score, entry["answer"]
        return answer if best >= self.MIN_SCORE else None
