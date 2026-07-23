import subprocess
import sys

def test_import_catan():
    import catan

    assert catan is not None

def test_public_api():
    from catan import Tracking, SessionData

    assert Tracking is not None
    assert SessionData is not None

    

def test_headless_import():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import catan; print(catan.__version__)",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr