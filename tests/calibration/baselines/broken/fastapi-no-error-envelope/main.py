"""FastAPI user lookup service."""
from fastapi import FastAPI

app = FastAPI()

USERS = {"1": {"name": "alice"}, "2": {"name": "bob"}}


@app.get("/users/{user_id}")
def get_user(user_id: str):
    return USERS.get(user_id, {})
