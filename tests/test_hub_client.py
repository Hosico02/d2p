import json
import pathlib
import pytest
import httpx
import respx
from d2p.hub_client import HubClient, BAKED_STANDARDS_FALLBACK


def make_client(tmp_path: pathlib.Path) -> HubClient:
    return HubClient(base_url="http://hub.local",
                     token="tok", cache_dir=tmp_path)


@respx.mock
def test_pull_standards_fetches_and_caches(tmp_path):
    respx.get("http://hub.local/api/standards/fastapi-api").mock(
        return_value=httpx.Response(200, json={
            "version": 3, "body_md": "- live body", "etag": "3",
        }, headers={"etag": "3"}),
    )
    c = make_client(tmp_path)
    out = c.pull_standards("fastapi-api")
    assert "live body" in out
    cached = (tmp_path / "hub_cache" / "fastapi-api.md").read_text()
    assert "live body" in cached


@respx.mock
def test_pull_standards_304_uses_cache(tmp_path):
    cache_dir = tmp_path / "hub_cache"
    cache_dir.mkdir()
    (cache_dir / "fastapi-api.md").write_text("- cached body")
    (cache_dir / "fastapi-api.etag").write_text("2")
    respx.get("http://hub.local/api/standards/fastapi-api").mock(
        return_value=httpx.Response(304),
    )
    c = make_client(tmp_path)
    out = c.pull_standards("fastapi-api")
    assert "cached body" in out


@respx.mock
def test_pull_standards_falls_back_to_cache_on_network_error(tmp_path):
    cache_dir = tmp_path / "hub_cache"
    cache_dir.mkdir()
    (cache_dir / "fastapi-api.md").write_text("- cached fallback")
    respx.get("http://hub.local/api/standards/fastapi-api").mock(
        side_effect=httpx.ConnectError("no network"),
    )
    c = make_client(tmp_path)
    out = c.pull_standards("fastapi-api")
    assert "cached fallback" in out


@respx.mock
def test_pull_standards_falls_back_to_baked_when_no_cache(tmp_path):
    respx.get("http://hub.local/api/standards/some-arche").mock(
        side_effect=httpx.ConnectError("no network"),
    )
    c = make_client(tmp_path)
    out = c.pull_standards("some-arche")
    assert out == BAKED_STANDARDS_FALLBACK


@respx.mock
def test_push_event_success(tmp_path):
    route = respx.post("http://hub.local/api/events").mock(
        return_value=httpx.Response(200, json={"event_id": "e1"}),
    )
    c = make_client(tmp_path)
    c.push_event("run_started", "r1", {"foo": "bar"})
    assert route.called


@respx.mock
def test_push_event_queues_to_pending_on_failure(tmp_path):
    respx.post("http://hub.local/api/events").mock(
        side_effect=httpx.ConnectError("down"),
    )
    c = make_client(tmp_path)
    c.push_event("run_started", "r1", {"foo": "bar"})
    pending = tmp_path / "hub_cache" / "pending_events.jsonl"
    assert pending.exists()
    lines = pending.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["type"] == "run_started"


@respx.mock
def test_push_event_flushes_pending_after_recovery(tmp_path):
    pending = tmp_path / "hub_cache"
    pending.mkdir()
    (pending / "pending_events.jsonl").write_text(
        json.dumps({"type": "run_started", "run_id": "r0", "payload": {}}) + "\n"
    )
    route = respx.post("http://hub.local/api/events").mock(
        return_value=httpx.Response(200, json={"event_id": "x"}),
    )
    c = make_client(tmp_path)
    c.push_event("iteration_complete", "r1", {"iter": 1})
    assert route.call_count == 2
    assert not (pending / "pending_events.jsonl").exists() or \
           (pending / "pending_events.jsonl").read_text().strip() == ""
