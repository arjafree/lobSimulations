"""Pin the arm-parameter env overrides in AR_RL_Trainer.py.

All six of the 2026-09-07 runs launched on the wrong arm (exploration_bonus=0
when the plan called for 0.1) because the arm was hardcoded and nothing in the
job recorded it. The overrides exist so the arm travels with the submission
script. The failure mode worth pinning is EXP_APPROX's string->bool parse:
`bool("False")` is True, so a naive conversion would silently flip the whole
simulator regime for anyone who wrote EXP_APPROX=False.
"""
import os
import re
import sys

TRAINER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AR_RL_Trainer.py")

_NAMES = ("EXPLORATION_BONUS", "GAE_LAMBDA", "EXP_APPROX")


def _config_lines():
    """The three assignment lines, pulled verbatim out of the trainer source."""
    src = open(TRAINER).read()
    lines = {}
    for name in _NAMES:
        m = re.search(r"^%s = .*$" % name, src, re.M)
        assert m is not None, "%s assignment not found in AR_RL_Trainer.py" % name
        lines[name] = m.group(0)
    return lines


def _evaluate(env):
    """Exec the assignment lines under a given environment, return the values."""
    lines = _config_lines()
    saved = {k: os.environ.get(k) for k in _NAMES}
    try:
        for k in _NAMES:
            os.environ.pop(k, None)
        os.environ.update(env)
        ns = {"os": os}
        for name in _NAMES:
            exec(lines[name], ns)
        return {name: ns[name] for name in _NAMES}
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_defaults_are_the_combined_arm():
    """Unset env must give the explo_gae arm -- the one the plan asks for."""
    v = _evaluate({})
    assert v["EXPLORATION_BONUS"] == 0.1, v
    assert v["GAE_LAMBDA"] == 0.95, v
    assert v["EXP_APPROX"] is False, v


def test_gae_lambda_only_arm_is_reachable():
    v = _evaluate({"EXPLORATION_BONUS": "0"})
    assert v["EXPLORATION_BONUS"] == 0.0, v
    assert v["GAE_LAMBDA"] == 0.95, v


def test_exp_approx_truthy_spellings():
    for spelling in ("1", "true", "True", "TRUE", "yes", " true "):
        v = _evaluate({"EXP_APPROX": spelling})
        assert v["EXP_APPROX"] is True, (spelling, v)


def test_exp_approx_falsy_spellings_are_not_naively_truthy():
    """bool("False") is True. A naive parse here silently changes TAU 500->10."""
    for spelling in ("0", "false", "False", "FALSE", "no", ""):
        v = _evaluate({"EXP_APPROX": spelling})
        assert v["EXP_APPROX"] is False, (spelling, v)


def test_arm_params_are_actually_wired_into_the_agent():
    """An override that never reaches the constructor is worse than no override."""
    src = open(TRAINER).read()
    assert "exploration_bonus = EXPLORATION_BONUS" in src
    assert "gae_lambda=GAE_LAMBDA" in src
    assert "'expApprox' : EXP_APPROX" in src
    # and no stale hardcoded copy left behind
    assert "exploration_bonus = 0," not in src
    assert "gae_lambda=0.95" not in src
    assert "'expApprox' : False" not in src


def test_config_banner_reports_the_arm():
    """The .o file must state the arm, or runs are indistinguishable after the fact."""
    src = open(TRAINER).read()
    for name in _NAMES:
        assert ("{%s}" % name) in src, "%s missing from the config banner" % name


if __name__ == "__main__":
    mod = sys.modules[__name__]
    failures = 0
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        try:
            getattr(mod, name)()
            print("PASS %s" % name)
        except AssertionError as e:
            failures += 1
            print("FAIL %s: %s" % (name, e))
    print("all tests passed" if not failures else "%d FAILURES" % failures)
    sys.exit(1 if failures else 0)
