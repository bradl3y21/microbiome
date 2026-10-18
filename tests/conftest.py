"""Shared QA harness. Contract-independent: nothing here imports src/."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def digest_arrays(named):
    """sha256 over (name, dtype, shape, bytes) of each array, in sorted name order.

    dtype and shape are hashed so a silent float32->float64 change or a reshape
    is a digest change, not an invisible one.
    """
    h = hashlib.sha256()
    for name in sorted(named):
        a = np.ascontiguousarray(named[name])
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def run_python(code, hashseed="0", extra_env=None, timeout=300):
    """Run `code` in a fresh interpreter rooted at the project; return stdout.

    A fresh process has its own global RNGs, hash seed and import order, which is
    what an in-process repeat cannot vary.
    """
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hashseed)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(SRC), str(ROOT), str(ROOT / "tests"), env.get("PYTHONPATH", "")) if p
    )
    env.update(extra_env or {})
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess exit {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip()


@pytest.fixture
def digest():
    return digest_arrays


@pytest.fixture
def fresh_python():
    return run_python
