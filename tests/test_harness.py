"""The QA tooling must be able to fail before it is trusted to judge anyone else."""
import numpy as np

from conftest import digest_arrays, run_python


def _state():
    rng = np.random.default_rng(7)
    return {"x": rng.random(100), "alive": rng.random(100) > 0.5}


def test_digest_is_stable_for_identical_state():
    assert digest_arrays(_state()) == digest_arrays(_state())


def test_digest_sees_a_single_ulp_change():
    s = _state()
    t = {k: v.copy() for k, v in s.items()}
    t["x"][3] = np.nextafter(t["x"][3], 2.0)
    assert digest_arrays(s) != digest_arrays(t)


def test_digest_sees_dtype_and_shape_change():
    s = _state()
    assert digest_arrays(s) != digest_arrays({**s, "x": s["x"].astype(np.float32)})
    assert digest_arrays(s) != digest_arrays({**s, "x": s["x"].reshape(10, 10)})


def test_digest_ignores_dict_insertion_order():
    s = _state()
    assert digest_arrays(s) == digest_arrays(dict(reversed(list(s.items()))))


def test_fresh_subprocess_reproduces_seeded_numpy_stream():
    code = "import numpy as np;print(np.random.default_rng(7).random(5).tolist())"
    assert run_python(code, hashseed="0") == run_python(code, hashseed="12345")


def test_subprocess_detects_hash_order_dependence():
    # Control: a deliberately hash-order-dependent program MUST differ across
    # PYTHONHASHSEED values, otherwise the subprocess check could never catch it.
    code = "print(list({'a','b','c','d','e','f','g','h'}))"
    outs = {run_python(code, hashseed=s) for s in ("0", "1", "2", "3")}
    assert len(outs) > 1


def test_headless_pygame_surface():
    import pygame

    pygame.init()
    surf = pygame.display.set_mode((32, 32))
    surf.fill((9, 8, 7))
    assert tuple(surf.get_at((0, 0)))[:3] == (9, 8, 7)
    assert pygame.display.get_driver() == "dummy"
    pygame.quit()
