"""Pin the live per-episode metrics in AR_RL_Trainer.py.

Two things are being pinned:

1. `_twap_slippage_bps` must agree with AR_RL_runner.py's slippage formula
   (runner lines ~605-613) on both sides. Criterion 3 ("the agent raises the
   TWAP's transaction costs") is judged on this number, and a sign error would
   invert the conclusion rather than merely perturb it.

2. `start_midprices` must actually be appended to. It was initialised and
   assigned but never appended, so the arrival midprice -- the denominator of
   every slippage figure -- was absent from trainer output. That is why
   criterion 3 had never been measured from a training run.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(HERE, "..", "AR_RL_Trainer.py")
RUNNER = os.path.join(HERE, "..", "AR_RL_runner.py")


def _load_helper():
    """Pull `_twap_slippage_bps` out of the trainer source and exec it alone.

    Importing AR_RL_Trainer would launch a training run, so the function is
    extracted textually. If the def is renamed or moved this test fails loudly,
    which is the intent.
    """
    src = open(TRAINER).read()
    m = re.search(r"^def _twap_slippage_bps\(.*?(?=\n\n\n)", src, re.M | re.S)
    assert m is not None, "_twap_slippage_bps not found in AR_RL_Trainer.py"
    import numpy as np
    ns = {"np": np}
    exec(m.group(0), ns)
    return ns["_twap_slippage_bps"]


def _runner_slip(side, twap_final_cash, executed, start_mid):
    """The runner's formula, transcribed verbatim as the reference."""
    if side == "sell":
        cash_earned = twap_final_cash - 1000000
        benchmark_earned = start_mid * executed
        return (benchmark_earned - cash_earned) * 10000 / benchmark_earned
    cash_spent = 1000000 - twap_final_cash
    benchmark_spent = start_mid * executed
    return (cash_spent - benchmark_spent) * 10000 / benchmark_spent


def test_matches_runner_formula_both_sides():
    f = _load_helper()
    cases = [
        ("sell", 1000000 + 100.0 * 500 * (1 - 10e-4), 500, 100.0),
        ("sell", 1000000 + 99.5 * 480, 480, 100.02),
        ("buy", 1000000 - 100.0 * 500 * (1 + 7e-4), 500, 100.0),
        ("buy", 1000000 - 100.4 * 512, 512, 100.01),
    ]
    for side, cash, ex, mid in cases:
        got, want = f(side, cash, ex, mid), _runner_slip(side, cash, ex, mid)
        assert abs(got - want) < 1e-9, (side, got, want)


def test_sign_convention_positive_means_worse_for_the_twap():
    """Criterion 3 wants this number to go UP. A sign flip inverts the verdict."""
    f = _load_helper()
    # A seller who receives BELOW the arrival mid has suffered cost -> positive.
    assert f("sell", 1000000 + 99.0 * 500, 500, 100.0) > 0
    # A seller who receives ABOVE the arrival mid has done better -> negative.
    assert f("sell", 1000000 + 101.0 * 500, 500, 100.0) < 0
    # A buyer who pays ABOVE the arrival mid has suffered cost -> positive.
    assert f("buy", 1000000 - 101.0 * 500, 500, 100.0) > 0
    assert f("buy", 1000000 - 99.0 * 500, 500, 100.0) < 0


def test_executes_at_the_mid_is_zero_slippage():
    f = _load_helper()
    for side, sgn in (("sell", +1), ("buy", -1)):
        cash = 1000000 + sgn * 100.0 * 500
        assert abs(f(side, cash, 500, 100.0)) < 1e-9


def test_degenerate_inputs_return_none_not_inf():
    """TWAP-absent episodes execute nothing; dividing by zero would poison
    every downstream mean with inf/nan instead of being skipped."""
    f = _load_helper()
    assert f("sell", 1000000.0, 0, 100.0) is None
    assert f("sell", 1000000.0, float("nan"), 100.0) is None
    assert f("buy", 1000000.0, 500, 0) is None
    assert f("buy", 1000000.0, 500, None) is None


def test_known_magnitude():
    """A 10 bps adverse fill must read as ~10 bps, not 1 or 100."""
    f = _load_helper()
    got = f("sell", 1000000 + 100.0 * (1 - 10e-4) * 500, 500, 100.0)
    assert abs(got - 10.0) < 1e-6, got
    got = f("buy", 1000000 - 100.0 * (1 + 10e-4) * 500, 500, 100.0)
    assert abs(got - 10.0) < 1e-6, got


def test_start_midprices_is_no_longer_a_dead_list():
    src = open(TRAINER).read()
    assert "start_midprices.append(starting_midprice)" in src, \
        "start_midprices is assigned but never appended -- slippage has no denominator"
    assert 'slippages_"+label+"start_midprice.npy' in src, \
        "the arrival midprice is never saved, so slippage cannot be recomputed post-hoc"


def test_live_metrics_cover_all_three_criteria():
    src = open(TRAINER).read()
    for key in ("inv_level_in_window",   # criterion 1
                "terminal_pnl",          # criterion 2
                "twap_slippage_bps",     # criterion 3
                "twap_present", "side", "seed"):
        assert '"%s"' % key in src, "%s missing from episode_metrics" % key
    # and it must be flushed every episode, not only at the end
    assert "_save_episode_metrics()" in src


def test_in_window_bucketing_keys_on_actual_presence():
    """A TWAP-absent episode's [twap_start, twap_end] stretch is ordinary
    market-making time. Filing it as "with TWAP" would contaminate the exact
    comparison the metric exists to make."""
    src = open(TRAINER).read()
    assert re.search(r"if twap_present and \(twap_end_time > Simstate\['TimeCode'\] > twap_start_time\)", src), \
        "inventory bucketing no longer gates on twap_present"


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
