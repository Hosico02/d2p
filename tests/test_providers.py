"""Unit tests for provider routing — no live API."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d2p.providers import (DEFAULT_ROLE_MODELS, ProviderSpec, RoleRouter,
                            build_router)
from d2p.providers._json import extract_json
from d2p.providers.base import LLMProvider


class _Stub:
    """Minimal fake LLMProvider."""
    def __init__(self, name): self.name = name
    def chat(self, *a, **k): return ""
    def chat_json(self, *a, **k): return {}


class TestRoleRouter(unittest.TestCase):
    def test_routes_by_role(self) -> None:
        p_exec = _Stub("haiku")
        p_smart = _Stub("opus")
        r = RoleRouter({
            "default": p_smart,
            "executor": p_exec,
            "qa": p_smart,
        })
        self.assertIs(r.for_role("executor"), p_exec)
        self.assertIs(r.for_role("analyzer"), p_smart)
        self.assertIs(r.for_role("nonexistent"), p_smart)

    def test_describe_lists_roles(self) -> None:
        r = RoleRouter({"default": _Stub("x"), "executor": _Stub("y")})
        d = r.describe()
        self.assertIn("executor", d)
        self.assertIn("default", d)


class TestProviderFactory(unittest.TestCase):
    def test_claude_default_role_models(self) -> None:
        m = DEFAULT_ROLE_MODELS["claude"]
        self.assertIn("haiku", m["executor"])
        self.assertIn("opus", m["planner"])
        self.assertIn("opus", m["analyzer"])
        self.assertIn("opus", m["qa"])

    def test_codex_default_role_models(self) -> None:
        m = DEFAULT_ROLE_MODELS["codex"]
        self.assertIn("mini", m["executor"])
        self.assertEqual(m["planner"], "gpt-4o")
        self.assertEqual(m["analyzer"], "gpt-4o")

    def test_build_router_minimax(self) -> None:
        spec = ProviderSpec(
            kind="minimax", api_key="sk-cp-test", base_url="https://x",
            default_model="MiniMax-test", role_models={},
        )
        r = build_router(spec)
        # MiniMax: same provider for every role
        self.assertIs(r.for_role("executor"), r.for_role("planner"))

    def test_build_router_claude_role_split(self) -> None:
        spec = ProviderSpec(
            kind="claude", api_key="sk-ant-test", base_url="",
            default_model="claude-haiku-4-5",
            role_models=DEFAULT_ROLE_MODELS["claude"],
        )
        r = build_router(spec)
        # Different models → different provider instances
        exec_name = r.for_role("executor").name
        plan_name = r.for_role("planner").name
        self.assertIn("haiku", exec_name)
        self.assertIn("opus", plan_name)
        self.assertNotEqual(exec_name, plan_name)

    def test_missing_key_raises(self) -> None:
        spec = ProviderSpec(kind="claude", api_key="", base_url="",
                            default_model="claude-haiku-4-5", role_models={})
        with self.assertRaises(RuntimeError):
            build_router(spec)


class TestExtractJson(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(extract_json('{"a":1}'), {"a": 1})

    def test_fenced(self) -> None:
        self.assertEqual(extract_json('```json\n{"a":2}\n```'), {"a": 2})


if __name__ == "__main__":
    unittest.main()
