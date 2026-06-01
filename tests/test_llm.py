"""The LLM client — offline, by mocking urlopen.

Proves the backend-agnostic contract: a key is required for remote endpoints but
not for localhost, OpenRouter attribution headers are sent only to OpenRouter, and
the OpenAI-compatible payload/parse round-trips.
"""

import json
import unittest
from unittest.mock import patch

from self_rag import llm


class FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_urlopen_factory(captured, content="hi there"):
    def fake_urlopen(req, timeout=0):
        captured["req"] = req
        captured["body"] = json.loads(req.data.decode())
        return FakeResp({"choices": [{"message": {"content": content}}]})
    return fake_urlopen


class IsLocalTests(unittest.TestCase):
    def test_detects_localhost(self):
        self.assertTrue(llm._is_local("http://localhost:11434/v1/chat/completions"))
        self.assertTrue(llm._is_local("http://127.0.0.1:1234/v1/chat/completions"))

    def test_remote_is_not_local(self):
        self.assertFalse(llm._is_local("https://openrouter.ai/api/v1/chat/completions"))


class KeyRequirementTests(unittest.TestCase):
    def test_remote_without_key_raises(self):
        with patch.object(llm, "_API_KEY", ""), \
             patch.object(llm, "_API_URL", "https://openrouter.ai/api/v1/chat/completions"):
            with self.assertRaises(EnvironmentError):
                llm.chat("hello")

    def test_local_without_key_is_allowed(self):
        captured = {}
        with patch.object(llm, "_API_KEY", ""), \
             patch.object(llm, "_API_URL", "http://localhost:11434/v1/chat/completions"), \
             patch("urllib.request.urlopen", fake_urlopen_factory(captured)):
            out = llm.chat("hello")
        self.assertEqual(out, "hi there")
        self.assertFalse(captured["req"].has_header("Authorization"))


class RequestShapeTests(unittest.TestCase):
    def test_openrouter_headers_and_payload(self):
        captured = {}
        with patch.object(llm, "_API_KEY", "sk-test"), \
             patch.object(llm, "_API_URL", "https://openrouter.ai/api/v1/chat/completions"), \
             patch("urllib.request.urlopen", fake_urlopen_factory(captured)):
            out = llm.chat("hello", temperature=0.0)
        self.assertEqual(out, "hi there")
        req = captured["req"]
        self.assertEqual(req.get_header("Authorization"), "Bearer sk-test")
        self.assertTrue(req.has_header("Http-referer"))  # OpenRouter attribution
        self.assertEqual(captured["body"]["messages"][0]["content"], "hello")

    def test_no_openrouter_headers_for_local(self):
        captured = {}
        with patch.object(llm, "_API_KEY", ""), \
             patch.object(llm, "_API_URL", "http://localhost:11434/v1/chat/completions"), \
             patch("urllib.request.urlopen", fake_urlopen_factory(captured)):
            llm.chat("hello")
        self.assertFalse(captured["req"].has_header("Http-referer"))


if __name__ == "__main__":
    unittest.main()
