"""Simple greeter Flask app."""
from flask import Flask

app = Flask(__name__)


@app.route("/")
def index() -> str:
    return "hello"


if __name__ == "__main__":
    app.run(port=5000)
