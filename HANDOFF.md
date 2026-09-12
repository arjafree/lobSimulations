# Handoff — why no agent meets the three criteria, and what is now running

Written 2026-09-12 (second session that day). Repo `arjafree/lobSimulations`,
branch `trying_to_avoid_localoptima`. Supersedes the earlier 2026-09-12 handoff,
whose §1 (simulator fixes) and §4 (prior ablation) still stand and are preserved
below in condensed form. §2's "which arm are the six runs on" question is
settled: they are the gae_lambda arm, they were left running, and the combined
arm is now running alongside them.

---

# 0. THE GOAL (unchanged)

An RL agent that satisfies **all three**:

1. **Front-runs correctly.** TWAP buying → agent long ahead of it. TWAP selling
   → agent short ahead of it.
2. **Stays profitable when TWAP is absent.**
3. **Increases the TWAP's transaction costs when present.**

---

# 1. THE HEADLINE FINDING: no run has ever front-run, and "level" was measuring
     a confound

The previous handoff's primary metric was mean inventory **level** inside the
TWAP window, chosen over the paired **shift** because the two disagreed. Both
readings were wrong, for the same reason.

## 1a. Inventory level is ~90% a per-run directional bias

On the two finished drift-fixed runs, per episode, correlating the in-window
level against the out-of-window level:

| run | corr(in-window, out-of-window) | variance of level explained by the bias |
|---|---|---|
| `buy_base` (n=80) | **0.936** | **87.1%** |
| `sell_base` (n=80) | **0.961** | **92.3%** |

The agent sits at roughly the same inventory whether or not the meta-order is
there. The *response* to the TWAP is the residual, and it has the **wrong sign
in both runs**:

| run | TWAP side | response (in − out) | wanted |
|---|---|---|---|
| `buy_base` | buy | **−1.73** [95% CI ±1.05] | positive |
| `sell_base` | sell | **+0.78** [±0.75] | negative |

So `sell_base`'s "correct" level of −3.21, which the previous handoff recorded
as a success reproducing old `gae_lambda` sell, was the bias. Its actual
response to the meta-order points the wrong way.

## 1b. The alternating-side runs show the bias is side-independent

Decisive, and visible now in `dfx_both_base_avg_inv_trajectories_ep68.png` and
`dfx_both_onoff_avg_inv_trajectories_ep80.png` — the same policy, the same run,
sides alternating episode to episode:

| run | mean inventory, buy episodes | mean inventory, sell episodes |
|---|---|---|
| `both_base` (ep 68, n=35/34) | **+12** | **+13** |
| `both_onoff` (ep 80, n=27/27) | **−8.5** | **−7** |

Identical on both sides. In both runs the drift begins at t=100, when the RL
agent starts trading, ~150 s **before** the meta-order exists, and is flat
through the TWAP window. The sign differs between runs (long in one, short in
the other) but never within a run — it looks like arbitrary symmetry-breaking
of the policy, not a property of the market.

**This is how single-side runs produced apparently correct results.** A
side-independent bias reads as "correct" on whichever side matches its sign, so
a buy run from one arm and a sell run from another could both look right while
neither was front-running. That is exactly the shape of the prior ablation's
"exploration owns buy, gae_lambda owns sell" conclusion (§6 below) — it should
now be treated as unsupported until reproduced on an alternating-side run.

## 1c. The metric to use

**Side contrast** = mean(level | TWAP buying) − mean(level | TWAP selling),
within one alternating-side run. The bias is common to both sides so it cancels,
leaving exactly the response, in level units. Want it positive and large.

Implemented in `HawkesRLTrading/criteria_report.py`, which **refuses to score
criterion 1 on a single-side run at all**.

---

# 2. WHY CRITERION 2 HAS NEVER BEEN MET: the reward barely contains PnL

Measured per RL step across the six 2026-09-07 runs (~198k steps each):

| quantity | value |
|---|---|
| per-step PnL change, median | **exactly 0** — 85–88% of steps move no money |
| per-step PnL change, mean abs | 0.006 – 0.018 dollars |
| **terminal PnL per episode** | **±0.2 dollars** over ~2,483 steps |
| flat action bonus | **0.5** on every non-no-op step |
| exploration bonus | 0.2 first visit, then λ/√N (λ=0.1 on the explo arm) |
| running inventory penalty | **none** — the line is commented out |
| terminal inventory penalty | 25·inv², once, ~2,483 steps away, γ^2483 ≈ 0.08 |

The economic prize for comparison: the meta-order is **150 shares**
(`total_order_size=150`; the TWAP's 500 is its starting *inventory*, not its
order size). A successful front-run holds the 25-share inventory limit through a
~3 bps move, so it is worth about **$0.75 per episode**.

Per episode, against that $0.75:

| term | episode total | ratio to prize |
|---|---|---|
| action bonus 0.5 (30% action rate) | $372 | **497×** |
| exploration bonus, N≈100 | $25 | 33× |
| terminal inventory penalty at the limit | $15,625 | 20,833× |
| *the PnL the agent is supposed to maximise* | *$0.2* | *0.3×* |

**Every shaping term is one to four orders of magnitude above the signal.** The
action bonus in particular is deterministic and always positive where PnL is
near-zero-mean noise, so PPO has one large reliable gradient (act more) and one
tiny noisy one (make money). The objective actually being maximised is "act
often, visit novel states". That is a sufficient explanation for criterion 2
failing on every arm ever tried, and for it failing at PnL ≈ 0 specifically
rather than at a loss. It also explains criterion 1: the front-running prize is
~500× smaller than the action bonus, so it is invisible to the optimiser.

This was §7.3 of the previous handoff — its own highest-ranked untested
hypothesis. It is now tested and it holds.

**The obvious objection fails: there is no advantage normalisation.** Both
lines in `compute_gae` that would normalise the advantages are commented out
(`ICRLAgent.py:2013-2014`), so the shaping terms enter the PPO surrogate loss at
full magnitude rather than being rescaled to unit variance. Even with
normalisation the *ratio* of PnL to shaping in the gradient would be unchanged;
without it, the raw magnitudes are what the optimiser sees.

**Which network each term biases** — this matters for which criterion it
breaks, and the answer is not the same for all three:

| term | acts on | consequence |
|---|---|---|
| action bonus 0.5 | mainly the **decision** net (act vs no-op) | over-trading. At `tc=1e-4` an extra 100 shares/episode of churn costs $1 — more than the whole $0.75 prize. **Breaks criterion 2.** |
| exploration bonus λ/√N | the **utility** net — it is state *and action* dependent, so it differentiates *which side to quote* | at 33× the prize it swamps the directional signal. **This, not the action bonus, is the term that most directly breaks criterion 1.** |
| terminal inv penalty | both, but only near the episode end, through ~2,483 steps of γ=0.999 and 64-step LSTM chunks | pushes toward flat at the close; credit assignment back to the TWAP window is very weak |

Note the action bonus is constant across the four utility actions, so it does
not *directly* prefer one side — but it is not cancelled either, because GAE
computes the u-advantage from the same realised reward stream, so an action
followed by a high-activity stretch inherits that stretch's bonuses.

**Careful with the running inventory penalty.** 1e-4·inv² costs $42 to hold the
limit through the TWAP window — 56× the prize. It is not a mild "inventory
control"; it forbids precisely the inventory-holding criterion 1 requires. The
PnL-scale settings are action bonus 1e-3, running penalty **2e-7**, terminal
penalty 1.2e-3.

---

# 3. TWO STRUCTURAL CONSTRAINTS NOT PREVIOUSLY NOTED

## 3a. The agent has four actions, none of them aggressive

Every run to date uses `action_space_config=1`:

```
allowed_actions = ["lo_top_Ask", "co_top_Ask", "co_top_Bid", "lo_top_Bid"]
```

Post or cancel at the touch, either side. **No market order, no in-spread
order.** The policy cannot initiate a position — it can only choose which side
it is exposed on and wait for someone else's aggression to fill it. During a
buying TWAP the aggression in the book is buying, so a resting ask is lifted and
the agent ends up **short into a rising market**: run over, not in front. Going
long instead needs other participants to sell into its bid, which is what is
scarce while a buy meta-order works.

(CLAUDE.md documents a 13-choice action space. That is config 0 plus the no-op
— not what any run has used. Worth correcting.)

`ACTION_SPACE_CONFIG=0` opens the full 12-action space.

## 3b. The market-order gates are long-biased

```
mo_Ask (SELL) blocked when  inv < 1  OR  inv >= inventorylimit - 2
mo_Bid (BUY)  blocked when  inv <= 2 - inventorylimit
```

At `inventorylimit=25` that permits selling only on inv ∈ [1,22] but buying on
inv ∈ [−22, +∞). The `inv >= limit-2` clause blocks **selling while very long**,
which is backwards for a position limit, and its mirror — block buying while
very long — is absent, so nothing in this gate bounds inventory from above. The
exploration branch is looser still: no limit clause on the sell, no bid clause
at all.

**Dormant in every run to date**, because config 1 never emits u=4 or u=7, so it
explains nothing about current results. It would corrupt any config-0 run.
`SYMMETRIC_MO_GATING=true` fixes it; default is legacy so nothing in flight
changes. The separate "no shorting via market order" rule (`inv < 1`) is left
alone — it is a modelling choice, and **someone should confirm it is intended**,
because it means the agent can never short except by being filled passively,
which makes the sell side of criterion 1 structurally harder than the buy side.

---

# 4. CRITERION 3 WAS UNMEASURABLE, AND NOW ISN'T

`start_midprices` in `AR_RL_Trainer.py` was initialised and assigned but **never
appended to**. The arrival midprice — the denominator of every slippage figure —
never reached disk. That is the mechanical reason criterion 3 had never been
measured from a training run. Fixed.

Slippage is now computed per episode in the trainer using the same formula as
`AR_RL_runner.py:605-613` (positive = the meta-order paid more / received less =
higher transaction cost).

**The control was also missing.** Criterion 3 is a comparison and its
denominator — what the same meta-order costs with no RL agent — had never been
run on this simulator, these seeds, this timing. The paper's 7.05/10.36 bps are
rPOV₁ numbers from a different setup and cannot serve as the control.
`RL_DISABLED=true` runs the identical episode loop with the RL agent never
acting; `ctl_n` / `ctl_f` are those runs. `get_action` is skipped entirely
rather than overridden, because it draws from the stdlib RNG the exchange also
uses — calling and discarding would desynchronise the background flow and make
the control a different market.

First ever measurement, episode 0 of the smoke run: **1.96 bps** (buy side).
The TWAP completes its order (149–154 of 150 shares executed).

---

# 5. LIVE INSTRUMENTATION (this is the main usability change)

`episode_metrics_<label>.json`, rewritten **every episode**, in each run's
`logs/`. Carries per episode: `inv_level_in_window`, `inv_level_out_window`,
`terminal_pnl`, `twap_slippage_bps`, `twap_present`, `side`, `seed`,
`episode_seconds`, plus the full arm config in the header.

Previously the arrays holding the primary metric were written **only on run
completion**, which is why every in-flight run could be judged only by eyeballing
`avg_inv_trajectories_epNN.png`. That is also how the prior ablation was read.

```bash
scp "peacock:~/LSTM_fRL/wout_expo/new_value_function/*/*/logs/episode_metrics_*.json" ./j/
python3 HawkesRLTrading/criteria_report.py ./j/ --last 20 --control ./j/episode_metrics_ctl_normal.json
```

The report gives each criterion a CI and a pass/fail, reads C2 only from
TWAP-absent episodes, and pairs C3 per episode against the control.

Every run also now prints a **config banner** at startup, so a `.o` file states
its own arm. The six 2026-09-07 runs could not be told apart from their output,
which is how they all silently ran on the wrong arm.

---

# 6. WHAT IS RUNNING (2026-09-12, 19 jobs)

Config travels in each `trainer.sh` env block; runs cannot cross-contaminate.
All use `SEED_MODE=vary SEED_BASE=1000`, so **runs pair episode-by-episode**.

**Pre-existing, gae_lambda arm (`exploration_bonus=0`), left alone:**
`dfx_bot_b` (both_base), `dfx_buy_o`, `dfx_sel_o`, `dfx_bot_o`.

**`explo_gae_dfx/` — the combined arm the plan always called for** (bonus=0.1):
`eg_buy`, `eg_sell` (single-side, 80 eps, pair with `buy_base`/`sell_base`),
`eg_bot_o` (alternate 2on/1off, 160), `eg_bot_f` (same, expApprox), `eg_smoke`.

**`reward_scale/` — action bonus at the PnL scale:**
`rw_b` (bonus=1e-3, 120 eps, normal), `rw_bf` (fast), `rw_bif` (fast, with the
**56×-too-strong** 1e-4 inventory penalty — kept deliberately as an
"inventory crushed" contrast, do not read it as the intended arm).

**`pnl_scaled/` — every shaping term at the prize's scale** (bonus 1e-3,
running 2e-7, terminal 1.2e-3):
`ps_ls` (legacy 4-action space, 120 eps, normal — isolates reward scaling from
the action-space change), `ps_fs` (full 12-action space + symmetric gating,
120 eps, normal), `ps_ff` (same, fast), `ps_af` (same but exploration also
scaled to 3e-3 — tests whether exploration survives being put in dollar units),
`ps_as` (the same all-scaled arm in the normal regime, 120 eps — added once it
became clear the exploration bonus, not the action bonus, is the term that most
directly breaks criterion 1, so it deserves a full-regime run).

**`twap_alone_control/` — the criterion-3 denominator:** `ctl_n` (60 eps,
normal), `ctl_f` (90 eps, fast). `RL_DISABLED=true`.

Four jobs launched earlier this session were retired with `qdel` (7380749,
7380752, 7380770, 7380771): they carried `RUNNING_INVPENALTY=1e-4` before that
was recognised as 56× the prize. `pnl_scaled/` replaces them.

Timings: expApprox=False ≈ 4,200–6,300 s/episode; expApprox=True ≈ 550 s/episode
(~9 min) but a **different simulator regime** (TAU 10 vs 500) — fast iteration
only, never mix the two in one comparison.

---

# 7. WHAT TO DO NEXT

1. **Wait for the side contrast.** `eg_bot_f`, `ps_ff`, `ps_af` are the fast
   alternating runs and will have ~40 TWAP-present episodes within a day.
   Criterion 1 is decided by whether the contrast becomes positive and
   significant. Nothing before that is evidence.
2. **`ps_fs` vs `ps_ls`** isolates whether the action space or the reward scale
   is the binding constraint. If `ps_ls` (passive-only, scaled reward) still
   shows no contrast, §3a is confirmed as the obstacle.
3. **`ps_af` vs `ps_ff`** says whether the exploration bonus can be put in
   dollar units without exploration collapsing. If `ps_af` degenerates to
   no-op, the intrinsic reward is load-bearing and needs decaying rather than
   shrinking, or reward normalisation instead.
4. **Criterion 3 needs `ctl_n`/`ctl_f` finished** before any slippage number
   means anything. Do not compare against the paper's 7.05/10.36.
5. **Re-examine the prior ablation's arm conclusions (§8).** They rest on
   single-side runs and therefore on the metric §1 shows is ~90% confound.
6. Ask the user whether "no shorting via market order" (§3b) is intended.

---

# 8. PRESERVED FROM THE EARLIER HANDOFFS

**Simulator fixes (2026-09-12 morning, all validated and pushed).** Bug 1:
thinning-bound violation biased event assignment toward Ask (6.2% of events
drawn from a truncated distribution) — `sample_dimension()` now draws a fresh
uniform over realised intensity. Bug 2: stale `self.left` after history purge.
Bug 3: `TradingAgent.__init__` aliased the caller's Inventory dict. After the
fixes, agent-free midprice drift went from −4.48 bps (p=0.002) to +0.58
(p=0.65), and the generator is jointly symmetric at n=198 (Hotelling T²
F(6,192)=1.552, p=0.163). End-to-end `PeggedMMAgent` confirmation with a
buggy-code control arm: buggy−fixed inventory skew +50.7, p=0.0013.
Full detail in `PLAN_drift_study.md` (~950 lines). **Still open:** event
*timing* under the still-invalid thinning bound is untouched and unmeasured —
the fix corrects *which* event fires, not *when*; an `mo` residual (−1.86%,
p=0.013) fails Holm correction; `src/backup/hawkes/simulate_optimized.py` still
carries bug 1 (archival).

**Prior ablation (2026-08-08)** — read from `avg_inv_trajectories_ep76.png`:
explo_gae correct on both sides; exploration-only correct on buy, wrong on sell;
gae_lambda-only correct on sell, weak on buy. **Now suspect**: all single-side,
so §1 applies.

**Running things.** `ssh peacock`; repo `~/lobSimulations`, sync is `git pull`.
`qsub` from inside the run's own directory (`#$ -cwd`). A `RUN_LABEL` containing
`test` **skips training entirely**. `qalter -l h_rt=...,tmem=64G,gpu=true <id>`
works on running jobs. Training seeds are 1000+; **use 5000+ for evaluation** or
you measure memorisation. The "only one config queued at a time" rule no longer
applies — config travels with the job.

**Tests:** `python3 HawkesRLTrading/tests/*.py` — 8 files, 55 tests, all
passing. They pin both simulator bugs' legacy behaviour, seed reproducibility,
TWAP presence/side logic, the reward term magnitudes, the market-order gates
(legacy asymmetry *and* the symmetric fix), the action-space contents, and the
env-override defaults.

**Git discipline.** Commit only files you deliberately changed. Long-standing
untracked and intentionally left alone: `.gitignore`, `CLAUDE.md`,
`HawkesRLTrading/twap_alone_*.py`, `twap_alone_out/`, `twap_baseline_obs*`,
`analysis_twap_mechanism/`.

---

# 9. MY ERRORS THIS SESSION

**(a) I launched five jobs with `RUNNING_INVPENALTY=1e-4` before working out
what the prize was worth.** It is 56× the prize and suppresses the exact
behaviour the goal requires. Four were retired within the hour; one is kept as a
labelled contrast. The lesson is the one this session is otherwise about: work
out the scale of a reward term before choosing it, not after.

**(b) I first estimated the front-running prize at $2.50 by reading the TWAP's
starting inventory (500) as its order size.** It is 150 shares, so the prize is
$0.75 and every ratio in §2 is ~3× worse than I first wrote.

**(c) The side-contrast numbers in §1b are read off plots, not computed.** The
alternating runs only write `inventorydists_*` on completion, so the numerical
contrast for `both_base`/`both_onoff` is not available until they finish. The
correlation analysis in §1a *is* computed, on finished runs, and is the stronger
evidence. New runs write the numbers from episode 1.
