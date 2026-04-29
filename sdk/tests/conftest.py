import logging

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
