"""HubClient — minimal, fail-safe client for MatrixOmnix Hub.

d2p calls into this from verifier (pull standards) and orchestrator
(push events). All operations degrade gracefully: hub-down never
blocks d2p.
"""
from __future__ import annotations
import json
import pathlib
import typing as t
import httpx

BAKED_STANDARDS_FALLBACK = """# baked fallback
- tests_run_and_pass
- error_envelope_present
- readme_cmd_matches_manifest
- missing_env_example
"""


class HubClient:
    def __init__(self, base_url: str, token: str, cache_dir: pathlib.Path):
        self.base = base_url.rstrip("/")
        self.token = token
        self.cache = pathlib.Path(cache_dir) / "hub_cache"
        self.cache.mkdir(parents=True, exist_ok=True)

    # ---- standards pull -----------------------------------------------

    def pull_standards(self, archetype: str) -> str:
        body_file = self.cache / f"{archetype}.md"
        etag_file = self.cache / f"{archetype}.etag"
        etag = etag_file.read_text().strip() if etag_file.exists() else None
        headers = {"Authorization": f"Bearer {self.token}"}
        if etag:
            headers["If-None-Match"] = etag
        try:
            # trust_env=False: ignore HTTP_PROXY / system proxy settings.
            # The Hub is typically on a private LAN or 127.0.0.1; routing
            # through a user's outbound proxy (Clash, Surge, corp proxy)
            # produces opaque 502s. Hub is also explicitly internal.
            with httpx.Client(timeout=5.0, trust_env=False) as client:
                resp = client.get(f"{self.base}/api/standards/{archetype}",
                                  headers=headers)
            if resp.status_code == 304:
                if body_file.exists():
                    return body_file.read_text()
                return BAKED_STANDARDS_FALLBACK
            if resp.status_code == 200:
                data = resp.json()
                body = data.get("body_md", "")
                body_file.write_text(body)
                if "etag" in data:
                    etag_file.write_text(str(data["etag"]))
                return body
            return self._fallback(body_file)
        except (httpx.HTTPError, OSError):
            return self._fallback(body_file)

    def _fallback(self, body_file: pathlib.Path) -> str:
        if body_file.exists():
            return body_file.read_text()
        return BAKED_STANDARDS_FALLBACK

    # ---- event push ---------------------------------------------------

    def push_event(self, event_type: str, run_id: str,
                   payload: dict[str, t.Any]) -> None:
        pending_file = self.cache / "pending_events.jsonl"
        events_to_send: list[dict[str, t.Any]] = []

        if pending_file.exists():
            for line in pending_file.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    events_to_send.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        events_to_send.append({
            "type": event_type, "run_id": run_id, "payload": payload,
        })

        unsent: list[dict[str, t.Any]] = []
        try:
            with httpx.Client(timeout=5.0, trust_env=False) as client:
                for evt in events_to_send:
                    try:
                        resp = client.post(
                            f"{self.base}/api/events",
                            headers={"Authorization": f"Bearer {self.token}"},
                            json=evt,
                        )
                        if resp.status_code not in (200, 202):
                            unsent.append(evt)
                    except httpx.HTTPError:
                        unsent.append(evt)
        except httpx.HTTPError:
            unsent.extend(events_to_send)

        if unsent:
            with pending_file.open("w") as f:
                for evt in unsent:
                    f.write(json.dumps(evt) + "\n")
        else:
            if pending_file.exists():
                pending_file.unlink()
