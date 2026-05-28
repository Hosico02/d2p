import pytest
from app import app


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()


def test_get_user_ok(client):
    r = client.get("/users/1")
    assert r.status_code == 200
    assert r.get_json() == {"data": {"name": "alice"}}


def test_get_user_404(client):
    r = client.get("/users/missing")
    assert r.status_code == 404
    body = r.get_json()
    assert body["error"] == "not_found"
