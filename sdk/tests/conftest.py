import logging
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _allow_extty_log_propagation():
    """Let pytest's caplog (attached to root) capture extty.* records.

    The package sets ``propagate=False`` on the ``extty`` logger so that a
    user's ``logging.basicConfig`` doesn't cause double output. Tests need
    propagation enabled to capture records via the root-attached handler.
    """
    log = logging.getLogger("extty")
    previous = log.propagate
    log.propagate = True
    try:
        yield
    finally:
        log.propagate = previous


@pytest.fixture(autouse=True)
def extty_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point extty at a per-test home so no test reads or writes ``~/.extty``.

    Checkpoints are written to the runs dir before any S3 upload, so without
    this a test exercising S3 checkpoints would write into the real home. It
    also hides a developer's S3 config (env or ``s3/config.toml``) from tests.
    """
    home = tmp_path / "extty-home"
    monkeypatch.setenv("EXTTY_HOME", str(home))
    for var in ("EXTTY_S3_BUCKET", "EXTTY_S3_PREFIX", "EXTTY_S3_ENDPOINT_URL"):
        monkeypatch.delenv(var, raising=False)
    return home
