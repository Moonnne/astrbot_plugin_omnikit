"""Keep host initialization and test artifacts outside the real installation."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["ASTRBOT_ROOT"] = tempfile.mkdtemp(prefix="omnikit-host-")
sys.dont_write_bytecode = True


def pytest_configure(config):
    config.option.basetemp = tempfile.mkdtemp(prefix="omnikit-test-")
