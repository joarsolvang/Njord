from example_pkg import hello


def test_example_pkg_says_hello() -> None:
    assert hello() == "Hello, world!"
