"""Tests for the readiness check (CLI + MCP `doctor`). Env is isolated so the
report is deterministic regardless of the developer's real .env."""
from __future__ import annotations

import json
import os
import unittest


class TestDoctor(unittest.TestCase):
    def setUp(self):
        from clippilot.brain import env as benv
        self._benv = benv
        self._orig_load = benv.load_dotenv
        benv.load_dotenv = lambda *a, **k: None  # don't let a real .env inject creds
        self._keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "LLM_API_KEY",
                      "YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN",
                      "UPLOAD_POST_API_KEY", "UPLOAD_POST_USERNAME")
        self._saved = {k: os.environ.pop(k, None) for k in self._keys}

    def tearDown(self):
        self._benv.load_dotenv = self._orig_load
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v

    def test_report_structure(self):
        from clippilot.doctor import check_readiness
        r = check_readiness()
        self.assertEqual(set(r.keys()), {"ready", "checks", "next_steps"})
        for c in r["checks"]:
            self.assertEqual(set(c.keys()), {"name", "ok", "detail", "unlocks", "required"})
        names = {c["name"] for c in r["checks"]}
        self.assertIn("ffmpeg", names)
        # ffmpeg is the only check marked required; its ok drives clip_sectionA
        ff = next(c for c in r["checks"] if c["name"] == "ffmpeg")
        self.assertTrue(ff["required"])
        self.assertEqual(r["ready"]["clip_sectionA"], ff["ok"])

    def test_no_creds_recommends_publishing_setup(self):
        from clippilot.doctor import check_readiness
        r = check_readiness()
        self.assertFalse(r["ready"]["publish_free"])
        self.assertFalse(r["ready"]["publish_paid"])
        self.assertFalse(r["ready"]["brain"])
        self.assertTrue(any("youtube_auth" in s for s in r["next_steps"]))

    def test_with_youtube_creds_enables_free_publish(self):
        os.environ["YOUTUBE_CLIENT_ID"] = "cid"
        os.environ["YOUTUBE_CLIENT_SECRET"] = "sec"
        os.environ["YOUTUBE_REFRESH_TOKEN"] = "ref"
        from clippilot.doctor import check_readiness
        r = check_readiness()
        # publish_free = ffmpeg AND youtube creds; ffmpeg is bundled
        self.assertEqual(r["ready"]["publish_free"], r["ready"]["clip_sectionA"])

    def test_format_report_is_ascii_string(self):
        from clippilot.doctor import check_readiness, format_report
        s = format_report(check_readiness())
        self.assertIn("ClipPilot readiness", s)
        self.assertIn("Capabilities", s)
        s.encode("ascii")  # must not raise (diagnostic stays console-safe)

    def test_mcp_doctor_tool(self):
        from clippilot.mcp_server import server as mcp
        resp = mcp.handle_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                               "params": {"name": "doctor", "arguments": {}}})
        data = json.loads(resp["result"]["content"][0]["text"])
        self.assertIn("checks", data)
        self.assertIn("ready", data)

    def test_cli_doctor_runs(self):
        from clippilot.cli import main
        self.assertEqual(main(["doctor"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
