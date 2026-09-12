#!/usr/bin/env python3
"""Score training runs against the three criteria, from the live metrics files.

Reads every episode_metrics_<label>.json under the given roots and reports:

  Criterion 1 -- front-running. The headline number is the SIDE CONTRAST:

      mean(inventory level in window | TWAP buying)
    - mean(inventory level in window | TWAP selling)

    which must be POSITIVE and large. Inventory level on a single-side run
    cannot establish front-running on its own, because a run can carry a
    persistent directional inventory bias that has nothing to do with the
    meta-order -- and the 2026-09-07 alternating-side runs do exactly that:
    both_base sits near +12 on buy AND +13 on sell episodes, both_onoff near
    -8.5 on buy AND -7 on sell. A side-independent bias reproduces "correct"
    single-side levels on whichever side happens to match its sign, which is
    how buy runs and sell runs from different arms could both look right.
    The contrast is immune to that bias because it differences it out.

    Level and the paired shift (in-window minus out-of-window) are both still
    reported, per side, as secondary.

  Criterion 2 -- profitable standalone. Mean terminal PnL over TWAP-ABSENT
    episodes only. Episodes where the meta-order is present are excluded
    entirely rather than split by window, because PnL is a path quantity and
    the out-of-window part of a present episode is still downstream of the
    meta-order's impact.

  Criterion 3 -- raises the TWAP's costs. Mean slippage in bps, per side,
    against a TWAP-alone control run (--control) that must have been produced
    with RL_DISABLED=true on the same seeds and schedule. Reported as the
    paired per-episode difference where episode indices line up, which removes
    the between-seed variance that dominates the unpaired comparison.

Usage:
    python3 HawkesRLTrading/criteria_report.py <dir> [<dir> ...] [--control <dir>]
    python3 HawkesRLTrading/criteria_report.py <dir> --last 20
"""
import argparse
import glob
import json
import math
import os


def _mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else None


def _stderr(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var / len(xs))


def _fmt(m, se, width=9):
    if m is None:
        return " " * (width - 3) + "n/a"
    s = "%*.2f" % (width, m)
    return s + (" +-%.2f" % (1.96 * se) if se is not None else "")


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    return d


def find(roots):
    out = []
    for r in roots:
        if os.path.isfile(r):
            out.append(r)
            continue
        out.extend(sorted(glob.glob(os.path.join(r, "**", "episode_metrics_*.json"),
                                    recursive=True)))
    return out


def report(d, last=None, control=None):
    eps = d["episodes"]
    if last:
        eps = eps[-last:]
    present = [e for e in eps if e["twap_present"]]
    absent = [e for e in eps if not e["twap_present"]]
    buy = [e for e in present if e["side"] == "buy"]
    sell = [e for e in present if e["side"] == "sell"]

    cfg = ("bonus=%s explo=%s inv_pen=%s space=%s sym_mo=%s expApprox=%s%s"
           % (d.get("action_bonus"), d.get("exploration_bonus"),
              d.get("running_invpenalty"), d.get("action_space_config"),
              d.get("symmetric_mo_gating"), d.get("expApprox"),
              " RL_DISABLED" if d.get("rl_disabled") else ""))
    print("\n" + "=" * 78)
    print("%s   (%d episodes%s)" % (d["label"], len(eps),
                                    ", last %d" % last if last else ""))
    print("  " + cfg)
    print("  TWAP %s on / %s off, side=%s, seeds %s+%s"
          % (d.get("twap_on"), d.get("twap_off"), d.get("twap_side_mode"),
             d.get("seed_base"), d.get("seed_mode")))
    print("-" * 78)

    # --- criterion 1
    lb = [e["inv_level_in_window"] for e in buy]
    ls = [e["inv_level_in_window"] for e in sell]
    mb, ms = _mean(lb), _mean(ls)
    print("C1 front-running (want the CONTRAST positive and large)")
    print("     level | buy  n=%-3d %s" % (len(buy), _fmt(mb, _stderr(lb))))
    print("     level | sell n=%-3d %s" % (len(sell), _fmt(ms, _stderr(ls))))
    if mb is not None and ms is not None:
        contrast = mb - ms
        se = None
        if _stderr(lb) is not None and _stderr(ls) is not None:
            se = math.sqrt(_stderr(lb) ** 2 + _stderr(ls) ** 2)
        verdict = "PASS" if (contrast > 0 and se is not None and contrast > 1.96 * se) else "fail"
        print("     >>> SIDE CONTRAST  %s   %s" % (_fmt(contrast, se), verdict))
    elif mb is not None or ms is not None:
        print("     >>> SIDE CONTRAST  n/a  (single-side run: cannot separate")
        print("         front-running from a per-run directional inventory bias)")
    sh_b = [e["inv_level_in_window"] - e["inv_level_out_window"]
            for e in buy if e["inv_level_in_window"] is not None
            and e["inv_level_out_window"] is not None]
    sh_s = [e["inv_level_in_window"] - e["inv_level_out_window"]
            for e in sell if e["inv_level_in_window"] is not None
            and e["inv_level_out_window"] is not None]
    print("     shift | buy  %s      sell %s"
          % (_fmt(_mean(sh_b), None), _fmt(_mean(sh_s), None)))

    # --- criterion 2
    pa = [e["terminal_pnl"] for e in absent]
    pp = [e["terminal_pnl"] for e in present]
    print("C2 standalone PnL (want > 0, on TWAP-ABSENT episodes)")
    if absent:
        m = _mean(pa)
        verdict = "PASS" if (m is not None and m > 0 and _stderr(pa) is not None
                             and m > 1.96 * _stderr(pa)) else "fail"
        print("     absent n=%-3d %s   %s" % (len(absent), _fmt(m, _stderr(pa)), verdict))
    else:
        print("     absent n=0    n/a  (this run has no TWAP-off episodes;")
        print("       set TWAP_OFF>0 -- criterion 2 cannot be read from present episodes)")
    print("     present n=%-3d %s  (context only)" % (len(pp), _fmt(_mean(pp), _stderr(pp))))

    # --- criterion 3
    print("C3 TWAP slippage in bps (want ABOVE the TWAP-alone control)")
    for name, grp in (("buy", buy), ("sell", sell)):
        sl = [e["twap_slippage_bps"] for e in grp]
        line = "     %-4s n=%-3d %s" % (name, len(grp), _fmt(_mean(sl), _stderr(sl)))
        if control is not None:
            cmap = {e["episode"]: e["twap_slippage_bps"] for e in control["episodes"]
                    if e["twap_present"] and e["side"] == name}
            pairs = [(e["twap_slippage_bps"], cmap[e["episode"]]) for e in grp
                     if e["episode"] in cmap and e["twap_slippage_bps"] is not None
                     and cmap[e["episode"]] is not None]
            if pairs:
                diffs = [a - b for a, b in pairs]
                m, se = _mean(diffs), _stderr(diffs)
                verdict = "PASS" if (m is not None and se is not None
                                     and m > 1.96 * se) else "fail"
                line += "   paired vs control n=%-3d %s  %s" % (
                    len(diffs), _fmt(m, se), verdict)
            else:
                line += "   (no paired control episodes)"
        print(line)
    ex = [e["twap_total_executed"] for e in present]
    print("     twap executed (of 500): %s" % _fmt(_mean(ex), None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--last", type=int, default=None,
                    help="use only the last N episodes (policy at the end of training)")
    ap.add_argument("--control", default=None,
                    help="dir or json of a TWAP-alone run (RL_DISABLED=true) for criterion 3")
    a = ap.parse_args()

    control = None
    if a.control:
        paths = find([a.control])
        if not paths:
            raise SystemExit("no episode_metrics_*.json under %s" % a.control)
        control = load(paths[0])
        if not control.get("rl_disabled"):
            raise SystemExit("control run %s was NOT produced with RL_DISABLED=true; "
                             "it is not a TWAP-alone baseline" % paths[0])

    paths = find(a.roots)
    if not paths:
        raise SystemExit("no episode_metrics_*.json found")
    for p in paths:
        report(load(p), last=a.last, control=control)
    print()


if __name__ == "__main__":
    main()
