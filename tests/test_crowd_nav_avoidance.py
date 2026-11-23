import pytest
import gameplay.crowd_nav_avoidance as mod

CASES = [
    ("avoid_damping", (1)),
    ("nav_version", (1)),
    ("deterministic_side", (1)),
    ("gate_capacity", (1)),
    ("repath_throttle", (1)),
    ("layer_group", (1)),
    ("queue_order", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("density_grid", (1)),
    ("metrics", (1)),
    ("lost_agent", (1)),
    ("local_avoid", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
