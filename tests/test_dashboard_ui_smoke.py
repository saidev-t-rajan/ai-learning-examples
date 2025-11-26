from streamlit.testing.v1 import AppTest


def test_dashboard_smoke():
    """Smoke test to ensure the dashboard app runs without error."""
    at = AppTest.from_file("app/web/main.py")
    at.run()

    assert not at.exception
    # Title in body is "Metrics Dashboard"
    assert "Metrics Dashboard" in at.title[0].value
