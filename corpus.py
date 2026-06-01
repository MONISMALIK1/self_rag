"""A small, self-contained knowledge corpus + an evaluation set.

Just enough documents to exercise every branch of the pipeline without any
download: on-topic passages that answer a question, *distractor* passages on
unrelated subjects (so the relevance critic has something to reject), questions
whose answer is **not** in the corpus (so abstention can be measured), and a
trivial question that needs no retrieval at all.

Point the CLI at your own corpus with ``--corpus path.jsonl`` (one
``{"id","title","text"}`` object per line) to use this on real data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A retrievable passage. ``id`` is what citations refer back to."""
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class QAExample:
    """An eval question. ``answer=None`` means the correct behavior is to abstain."""
    question: str
    answer: str | None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""


# --- the corpus -----------------------------------------------------------

CORPUS: list[Document] = [
    Document("jupiter", "Jupiter",
             "Jupiter is the largest planet in the Solar System, a gas giant with a "
             "mass more than two and a half times that of all the other planets "
             "combined. Its most famous feature is the Great Red Spot, a giant storm. "
             "Jupiter has more than 90 known moons, the four largest being Io, Europa, "
             "Ganymede, and Callisto."),
    Document("saturn", "Saturn",
             "Saturn is the second-largest planet in the Solar System and is famous for "
             "its prominent ring system, made mostly of ice and rock. It is a gas giant "
             "with dozens of moons, the largest of which is Titan."),
    Document("mars", "Mars",
             "Mars is the fourth planet from the Sun and is often called the Red Planet "
             "because of iron oxide on its surface. Mars has two small moons, Phobos and "
             "Deimos. It is home to Olympus Mons, the tallest volcano in the Solar System."),
    Document("earth", "Earth",
             "Earth is the third planet from the Sun and the only known planet to support "
             "life. It has a single natural satellite, the Moon. About 71 percent of "
             "Earth's surface is covered by water."),
    Document("mercury", "Mercury",
             "Mercury is the smallest planet in the Solar System and the closest to the "
             "Sun. It has no moons and almost no atmosphere, leading to extreme swings in "
             "surface temperature."),
    Document("venus", "Venus",
             "Venus is the second planet from the Sun and the hottest planet in the Solar "
             "System, due to a thick carbon-dioxide atmosphere that traps heat in a "
             "runaway greenhouse effect."),
    Document("sun", "The Sun",
             "The Sun is the star at the center of the Solar System. It is about 4.6 "
             "billion years old and accounts for roughly 99.8 percent of the system's "
             "total mass. It is composed mainly of hydrogen and helium."),
    Document("pluto", "Pluto",
             "Pluto is a dwarf planet in the Kuiper belt. It was considered the ninth "
             "planet until 2006, when the International Astronomical Union reclassified "
             "it as a dwarf planet."),
    Document("apollo11", "Apollo 11",
             "Apollo 11 was the spaceflight that first landed humans on the Moon, in July "
             "1969. Neil Armstrong was the first person to walk on the lunar surface, "
             "followed by Buzz Aldrin, while Michael Collins orbited above."),
    # --- distractors: unrelated domains the relevance critic should reject ---
    Document("python_lang", "Python (programming language)",
             "Python is a high-level programming language created by Guido van Rossum and "
             "first released in 1991. It emphasizes code readability and is widely used "
             "for web development, data science, and automation."),
    Document("java_lang", "Java (programming language)",
             "Java is a programming language first released by Sun Microsystems in 1995. "
             "It was designed by James Gosling around the principle of 'write once, run "
             "anywhere' via the Java Virtual Machine."),
    Document("great_wall", "Great Wall of China",
             "The Great Wall of China is a series of fortifications built across the "
             "historical northern borders of China to protect against invasions. "
             "Construction spanned many centuries and dynasties."),
]


# --- the evaluation set ----------------------------------------------------

EVAL_QUESTIONS: list[QAExample] = [
    QAExample("Which planet is the largest in the Solar System?", "Jupiter",
              note="answerable from corpus"),
    QAExample("How many moons does Mars have?", "two", aliases=("2",),
              note="answerable; relevance must pick Mars over other planets"),
    QAExample("Who created the Python programming language?", "Guido van Rossum",
              note="answerable from a distractor-domain doc"),
    QAExample("In what year was Java first released?", "1995",
              note="answerable from corpus"),
    QAExample("Who was the first person to walk on the Moon?", "Neil Armstrong",
              note="answerable from corpus"),
    QAExample("Which planet is the hottest in the Solar System?", "Venus",
              note="answerable; tempting wrong answer is Mercury (closest to Sun)"),
    QAExample("What is the deepest ocean trench on Earth?", None,
              note="ABSTAIN: not covered by the corpus"),
    QAExample("Who painted the Mona Lisa?", None,
              note="ABSTAIN: not covered by the corpus"),
]


_PUNCT = re.compile(r"[^a-z0-9 ]+")


def _norm(s: str) -> str:
    return _PUNCT.sub(" ", s.lower()).strip()


def matches(predicted: str | None, example: QAExample) -> bool:
    """True if ``predicted`` contains the gold answer (or an alias).

    Only meaningful for answerable questions; abstain questions are graded on
    whether the pipeline declined to answer, handled by the caller.
    """
    if predicted is None or example.answer is None:
        return False
    p = f" {_norm(predicted)} "
    for target in (example.answer, *example.aliases):
        if target and f" {_norm(target)} " in p:
            return True
    return False


__all__ = ["CORPUS", "EVAL_QUESTIONS", "Document", "QAExample", "matches"]
