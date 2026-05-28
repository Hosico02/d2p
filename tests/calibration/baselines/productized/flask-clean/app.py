"""Productized Flask user service."""
from flask import Flask, jsonify

app = Flask(__name__)
USERS = {"1": {"name": "alice"}, "2": {"name": "bob"}}


@app.route("/users/<user_id>")
def get_user(user_id: str):
    if user_id not in USERS:
        return jsonify({"error": "not_found", "user_id": user_id}), 404
    return jsonify({"data": USERS[user_id]})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "not_found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "internal"}), 500


if __name__ == "__main__":
    app.run(port=5000)
