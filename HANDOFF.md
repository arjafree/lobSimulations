# Handoff — drift investigation, simulator fixes, and the front-running retrain

Written 2026-09-12. Repo `arjafree/lobSimulations`, branch `trying_to_avoid_localoptima`,
HEAD `4537d5c`. Supersedes the 2026-08-08 ablation handoff (that study's findings
are preserved in "Prior ablation" below).

---

# 0. THE GOAL (from the user, verbatim intent)

Build an RL agent that satisfies **all three** criteria:

1. **Front-runs correctly.** TWAP buying → agent goes **long** ahead of it.
   TWAP selling → agent goes **short** ahead of it.
2. **Stays profitable when TWAP is absent.** It must be a viable market maker
   standalone, not only a parasite on meta-order flow.
3. **Increases the TWAP's transaction costs when present.** Slippage against the
   meta-order should go **up** relative to the TWAP-alone baseline.

No agent to date satisfies all three. The current best (`explo_gae`) gets (1)
on both sides but has PnL ≈ 0. Criterion (3) has never been demonstrated — the
paper's earlier fRL results showed TWAP slippage going *down*, which is the
surprise that motivated this entire line of work.

**Your job: review the plan below, then work/research/test until an agent meets
all three criteria.** Do not treat the plan as fixed — it has been wrong before
(see §5).

---

# 1. WHAT WAS FIXED THIS SESSION (all validated, all pushed)

The user suspected a latent drift in the simulator's seed causing asymmetric
front-running results. There was a drift. It was **not** a seed artifact — it
was two real simulator bugs. Full detail in `PLAN_drift_study.md` (~1000 lines,
read it before re-deriving anything).

## Bug 1 — thinning bound violation biased event assignment toward Ask (`6e94941`)

`Arrival_Models.py` reused the thinning variate `D*lamb_bar` to choose which of
the 12 dimensions fired. That is valid only while `lamb_bar` upper-bounds the
intensity. It does not: 36 of the 144 kernel entries are **inhibitory**, so
intensity *rises* between events as inhibition decays, breaking the bound. When
breached, `D*lamb_bar` is truncated below `sum(decays)` and the walk — which
starts at index 0 — can never reach the tail. `cols` is Ask-first, so the
unreachable tail was always Bid.

Measured: **6.2% of all events** were assigned from a truncated distribution.
Fix: `sample_dimension()` draws a fresh uniform over the realised intensity.

## Bug 2 — stale `self.left` after history purge (`4294506`)

The purge re-based `timeseries` but never reset `self.left`, so the next window
advance started `self.left` points too far in and silently dropped kernel
contributions still inside TAU. The effective excitation window shrank with
every purge. **Inert at T=550** (purge only fires after t=500, ~50s to
compound) but wrong, and it would bite in longer simulations.

## Bug 3 — `TradingAgent.__init__` aliased the caller's Inventory dict (`8952420`)

`self.Inventory = Inventory` stored the caller's dict by reference and then
mutated it in place. Any caller reusing its kwargs carried one episode's closing
inventory into the next. **Verified this never corrupted a production run** —
all three callers reset inventory each episode — but it made seeding meaningless
and would have bitten the alternating-TWAP work. Fixed with `dict(Inventory)`.

Also seeded the stdlib RNG in `tradingEnv` (`Exchange.py:473` used unseeded
`random.randrange`). **Tested: not load-bearing** — the Inventory fix alone
restores reproducibility in the configuration tested. Kept on principle.

## Validation (n=48 seeds/arm unless noted)

| metric | before | after |
|---|---|---|
| agent-free midprice drift, kernels ON | **−4.48 bps** [−7.21,−1.74], p=0.002 | **+0.58** [−1.92,+3.08], p=0.65 |
| generator Ask excess, total | +1.77%, p=7e-06 | +0.48%, p=0.31 |
| `lo_deep` Ask excess | +6.38%, p=4e-09 | +0.47%, p=0.68 |

At **n=198** the generator is jointly symmetric: Hotelling T² F(6,192)=1.552,
**p=0.163**. `co_deep` (+1.49%, p=0.059 at n=48) resolved to noise (+0.33%,
p=0.38, 98/198 seeds positive). A `mo` residual (−1.86%, p=0.013) fails Holm
correction and the omnibus; documented as unresolved, not blocking.

## End-to-end confirmation — `PeggedMMAgent` (`0776d93`)

An exactly symmetric pegged market maker (the user's own proposed test). Round 1
at `order_size=100` **failed its control** — the agent was 25% of the touch and
damped the drift it was measuring (buggy-arm drift −4.48 → −0.63, p=1.9e-10).
Round 2 at `order_size=10`, n=150/arm, with a **buggy-code control arm** (clone
at `002bbf3`):

- inventory skew: buggy **+35.1** [+12.9,+57.2] p=0.002; fixed **−15.7** p=0.146;
  buggy−fixed **+50.7, p=0.0013**
- MtM PnL: fixed − buggy **+1.76, p=0.006** (the bug costs a symmetric MM money)
- mechanism: corr(drift, d_inv) **−0.455 pooled, p=2.5e-24, n=450**

The agent still damps drift ~46% at size 10, so effect sizes are **under**estimates.
That is conservative for detection but means don't quote them as magnitudes.

**Always run the buggy-code control arm.** Round 1's null would have read as
"market is symmetric" and been completely wrong.

---

# 2. THE SIX TRAINING RUNS — STATUS AND A PROBLEM

Launched 2026-09-07. Dirs: `~/LSTM_fRL/wout_expo/new_value_function/drift_fixed/<name>/`.
Config travels in each `trainer.sh` env block (see §3), so runs cannot
cross-contaminate.

| job | run | eps | TWAP cycle | side | status (09-12) |
|---|---|---|---|---|---|
| 7339143 | `buy_base` | 80 | every ep | buy | **finished** |
| 7339144 | `sell_base` | 80 | every ep | sell | **finished** |
| 7339145 | `both_base` | 120 | every ep | alternate | 69/120 |
| 7339146 | `buy_onoff` | 160 | 2 on, 1 off | buy | 69/160 |
| 7339147 | `sell_onoff` | 160 | 2 on, 1 off | sell | 84/160 |
| 7339148 | `both_onoff` | 160 | 2 on, 1 off | alternate | 82/160 |

All `SEED_MODE=vary SEED_BASE=1000`, `expApprox=False`. Zero errors. Both
finished runs wrote all five tail files, confirming the `dtype=object` fix
(`fbccd36`) works — that save crashed and truncated the previous run.
`h_rt` was raised to 600h on all six via `qalter` on running jobs (works, no
restart needed) after two jobs sharing `webb.local` were projected to be killed
before finishing.

## THE PROBLEM: all six run with `exploration_bonus = 0`

`AR_RL_Trainer.py:259`. The repo was checked out at `31f62c9` (the `gae_lambda`
arm) when the runs were launched. The plan called for the **combined** arm.
The two arms differ in exactly one parameter:

```
explo_gae  (0639c51):  exploration_bonus = 0.1
gae_lambda (31f62c9):  exploration_bonus = 0     <- what all six are running
```

This was my error — I made `log_dir`/`label`/episode count env-overridable but
never checked the hyperparameters matched the arm the plan specified.

**Per the prior ablation (§4), this breaks the BUY side specifically.** Sell is
fine on `gae_lambda` alone. So:
- `sell_base`, `sell_onoff` — on the correct arm for their side
- `buy_base`, `buy_onoff` — wrong arm; buy needs the exploration bonus
- `both_base`, `both_onoff` — half-wrong (the buy episodes)

**Decision pending from the user.** Options: let them run (clean drift-free
replication of `gae_lambda`), restart all six with `0.1`, or restart only the
buy/both runs. Nobody has decided. **Do not restart without asking.**

---

# 3. HOW TO RUN THINGS (read `.claude/skills/ucl-cluster.md` first)

`ssh peacock`. Repo at `~/lobSimulations`, sync is `git pull`.

The skill's "only one config queued at a time" rule **no longer applies** as of
`5388c69`: run identity is env-overridable, so config travels with the job.
Env vars now controlling `AR_RL_Trainer.py`:

| var | default | meaning |
|---|---|---|
| `RUN_LOG_DIR` / `RUN_MODEL_DIR` / `RUN_LABEL` | gae_lambda sell paths | run identity |
| `N_EPISODES` | 80 | episode count |
| `TWAP_ON` / `TWAP_OFF` | 1 / 0 | presence cycle; `(ep % (ON+OFF)) < ON` |
| `TWAP_SIDE_MODE` | `sell` | `buy` \| `sell` \| `alternate` \| `random` |
| `SEED_MODE` / `SEED_BASE` | `fixed` / 1000 | `vary` → seed = BASE + episode |

`exploration_bonus` is **not** yet env-overridable — make it so before the next
launch (`EXPLORATION_BONUS`, default 0.1) so the arm is explicit per job.

Gotchas:
- A `RUN_LABEL` containing `test` **skips training entirely** (`:536`).
- Episodes take ~4,200–6,300s at `expApprox=False`. Historical runs: 4,432s and
  6,047s. `expApprox=True` is ~13x faster (TAU=10 vs 500) but a different
  simulator regime — usable for fast iteration, not for final results.
- `qalter -l h_rt=...,tmem=64G,gpu=true <jobid>` works on **running** jobs.
- Seed hygiene: training uses 1000–1159. **Use 5000+ for evaluation** or you
  measure memorisation.

Tests: `python3 HawkesRLTrading/tests/*.py` — 17 tests, all passing. They pin
both simulator bugs' legacy behaviour, seed reproducibility, and the TWAP
presence/side logic.

---

# 4. PRIOR ABLATION (2026-08-08, still the best evidence on arms)

From reading `avg_inv_trajectories_ep76.png` plots:

- **explo_gae** (both on): correct on **both** sides. Best result so far.
- **exploration only, buy**: correct. **exploration only, sell**: WRONG (fades long).
- **gae_lambda only, sell**: correct (~−8). **gae_lambda only, buy**: "correct
  direction but slower/incomplete".

Conclusion: **exploration owns buy-side correctness, gae_lambda owns sell-side,
the combined arm needs both.** Evaluation used checkpoint **epoch 76** for both
explo_gae sides (the final save — models write at `episode % 4 == 0`).

---

# 5. MY ERRORS THIS SESSION — READ THIS, IT WILL SAVE YOU TIME

**(a) I called adverse selection "front-running".** I saw inventory sign flip
with TWAP side and reported it as the agent positioning ahead of the meta-order.
It was the opposite — short into a buying TWAP. The user caught it. If the agent
is short while a buying TWAP pushes price up, it is being **run over**, not
front-running.

**(b) I used the wrong primary metric.** I measured the *paired shift*
(mean inventory during the TWAP window minus outside). The prior ablation used
inventory **level**. These disagree, and **level is the right one** — it asks
whether the agent holds the profitable position, whereas the shift only asks
whether the position deepens during the window. Level and shift disagree on
three of five runs:

| run | side | bonus | sim | LEVEL (last 20 eps) | SHIFT |
|---|---|---|---|---|---|
| OLD explo_gae buy | buy | 0.1 | buggy | **+13.76 OK** | +0.78 OK |
| OLD gae_lambda sell | sell | 0 | buggy | **−4.37 OK** | +0.61 wrong |
| OLD new_vf buy | buy | ? | buggy | +0.47 OK | −0.43 wrong |
| NEW buy_base | buy | 0 | FIXED | **−20.42 WRONG** | −1.73 wrong |
| NEW sell_base | sell | 0 | FIXED | **−3.21 OK** | +0.78 wrong |

**Use LEVEL as primary.** Report shift only as secondary.

**(c) I claimed the exploration bonus was the whole story, then over-corrected.**
First I said "every run is on the wrong arm, that's why it regressed." Then the
user pushed back that front-running never emerged over training in explo_gae
either — and that is **true**: explo_gae buy's front-running trend is
+0.015/ep, **p=0.322**, with only 46/80 (57%) of episodes correctly signed. The
level is solidly correct (+13.76) but the behaviour is noisy episode-to-episode,
which matches the user's account of having to pick out specific episodes.

The defensible position: `exploration_bonus=0` breaks buy specifically (matches
§4), `sell_base` correctly reproduces old `gae_lambda` sell on the level metric
(−3.21 vs −4.37), and **`buy_base` is the one genuine anomaly** — level −20.42
and diverging (trend −0.17/ep, p=0.002, against `inventorylimit=25`).

**(d) Two independent-reviewer subagents went idle and delivered nothing.** All
verification here is self-checked. Compensated with controls (buggy-code arms,
kernels-nulled nulls, legacy-behaviour assertions in tests), which is not the
same thing. The user asked for an independent reviewer; that request was never
fulfilled.

---

# 6. WHERE THINGS ACTUALLY STAND AGAINST THE THREE CRITERIA

| criterion | best evidence | status |
|---|---|---|
| 1. correct front-running | explo_gae, level +13.76 buy / ~−8 sell | **partial** — level correct, but noisy per-episode and no learning trend |
| 2. profitable without TWAP | explo_gae buy PnL **+0.00**, declining last 10 eps | **NOT MET** |
| 3. raises TWAP slippage | never measured for explo_gae | **UNKNOWN** |

Criterion 3 has **never been tested**. That is a gap, not a negative result.

---

# 7. SUGGESTED DIRECTION — CHALLENGE IT

The plan below is a starting point. It has been wrong before.

**First, decide the arm question (§2) with the user.** Nothing else is worth
doing on the buy side until that is settled.

**Then, in rough priority:**

1. **Measure criterion 3.** Run an existing explo_gae checkpoint (epoch 76,
   `20260704_120739_train_new_vf_explo_gae_buy` / `20260630_214031_..._sell`)
   through `AR_RL_runner.py` with TWAP present and compare slippage to the
   TWAP-alone baseline (paper Table: 7.05/10.36 bps buy/sell for rPOV₁). The
   runner already computes per-episode slippage. Cheap, and it tells you whether
   the thing you are optimising toward is even reachable from here.

2. **Diagnose `buy_base`'s runaway short.** Level −20.42 against a limit of 25,
   still diverging at episode 80. Is this the missing exploration bonus, or
   something about the drift-fixed simulator? The discriminating experiment is
   an `exploration_bonus=0.1` buy run on the fixed simulator — one job.

3. **Reconsider whether the reward can express criterion 2 at all.** Every arm
   tried so far lands at PnL ≈ 0. `rewardpenalty=eta=5`, `tc=0.0001`,
   `inventorylimit=25`. If no hyperparameter arm has ever produced positive PnL,
   the reward may not reward what the goal wants. This is the hypothesis I would
   personally rank highest and it has not been tested.

4. **The `onoff` runs are the actual intervention** and nobody has seen their
   results yet. They are the only thing in flight aimed at criterion 2. Analyse
   them properly using the saved `twap_present` array to split TWAP-absent
   episodes cleanly — note `profit_without_twap` mixes absent episodes with the
   out-of-window portions of present episodes, so it is **not** a clean readout.

5. **Phase D/E from `PLAN_drift_study.md`** — paired evaluation on disjoint
   seeds, and Table 8 CIs. Table 8 numbers are computed and reproduce every
   published figure exactly, but `table8_ci.py` was never written and the LaTeX
   is untouched. The 20-bps-from-median trim is unstable (bootstrap CI of the
   trimmed mean for buy ckpt16: [−0.37, +63.55]) — recommend a fixed-fraction
   trimmed mean or Huber M-estimator.

**Still-open smaller items:** exchange-side `lo_deep` asymmetry (full sim shows
it with kernels nulled, isolated generator does not); event *timing* under the
still-invalid thinning bound (untouched, unmeasured — the §1 fix corrects *which*
event fires, not *when*); the `mo` residual; `src/backup/hawkes/simulate_optimized.py`
still carries bug 1 (left as archival).

---

# 8. ANALYSIS SNIPPETS

Per-episode inventory level (the primary metric):

```python
inw = np.load(f"inventorydists_{lbl}_inventory_with_twap_{side}.npy", allow_pickle=True)
out = np.load(f"inventorydists_{lbl}_inventory_without_twap.npy", allow_pickle=True)
level = np.array([np.mean(e) for e in inw if len(e)])   # want >0 for buy TWAP, <0 for sell
```

Per-episode terminal PnL (**RL agent starts with cash 2500, not 1000000** — that
is the TWAP agent's; getting this wrong makes every number look like −997500):

```python
a = np.load(f"sharpe_{lbl}_profit.npy", allow_pickle=True).astype(float)
t, p = a[0], a[1] - 2500
b = [0] + [i for i in range(1, len(t)) if t[i] < t[i-1]] + [len(t)]
term = np.array([p[b[i+1]-1] for i in range(len(b)-1) if b[i+1] > b[i]])
```

Working analysis scripts from this session are in the session scratchpad under
`train/` (`compare_old.py`, `paired_inv.py`, `base_res.py`) — rewrite rather
than hunt for them; they are short.

## Git discipline

Commit only files you deliberately changed. Long-standing untracked/modified and
**intentionally left alone**: `.gitignore`, `CLAUDE.md`,
`HawkesRLTrading/twap_alone_*.py`, `twap_alone_out/`, `twap_baseline_obs*`,
`analysis_twap_mechanism/`. `PLAN_drift_study.md` and this file are tracked and
should be kept current.
