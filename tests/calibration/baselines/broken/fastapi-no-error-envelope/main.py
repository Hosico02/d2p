"""FastAPI service that returns 200 OK for everything, including bad input.
No @app.exception_handler — production gap."""
from fastapi import FastAPI

app = FastAPI()

USERS = {"1": {"name": "alice"}, "2": {"name": "bob"}}


@app.get("/users/{user_id}")
def get_user(user_id: str):
    # Returns whatever is at USERS[user_id], or empty dict if missing.
    # No 404, no error shape — every response is 200 with whatever payload.
    return USERS.get(user_id, {})
