from cli import build_message


def test_build_message_greets_by_name() -> None:
    assert build_message("World") == "Hello, World!"
