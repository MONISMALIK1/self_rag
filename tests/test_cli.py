"""CLI helpers — the JSONL corpus loader and its error handling (offline)."""

import tempfile
import unittest
from pathlib import Path

from self_rag.__main__ import _load_corpus


class LoadCorpusTests(unittest.TestCase):
    def _write(self, text: str) -> str:
        d = tempfile.mkdtemp()
        path = Path(d) / "corpus.jsonl"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_loads_valid_jsonl(self):
        path = self._write(
            '{"id": "a", "title": "Cats", "text": "Cats are felines."}\n'
            '\n'  # blank lines are skipped
            '{"id": "b", "title": "Dogs", "text": "Dogs are canines."}\n'
        )
        r = _load_corpus(path)
        self.assertEqual(len(r.documents), 2)
        self.assertEqual(r.search("felines", k=2)[0].doc.id, "a")

    def test_missing_file_is_friendly(self):
        with self.assertRaises(SystemExit) as cm:
            _load_corpus("/no/such/corpus.jsonl")
        self.assertIn("not found", str(cm.exception))

    def test_malformed_line_reports_line_number(self):
        path = self._write('{"id": "a", "title": "T", "text": "ok"}\nnot json\n')
        with self.assertRaises(SystemExit) as cm:
            _load_corpus(path)
        self.assertIn(":2:", str(cm.exception))

    def test_missing_required_field(self):
        path = self._write('{"id": "a", "title": "no text field"}\n')
        with self.assertRaises(SystemExit):
            _load_corpus(path)


if __name__ == "__main__":
    unittest.main()
