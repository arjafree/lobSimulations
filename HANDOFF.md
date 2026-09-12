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

# 1. THE HEADLINE FINDING: no run has ever front-run, and both metrics used
     so far were mis-baselined

The previous handoff's primary metric was mean inventory **level** inside the
TWAP window, chosen over the paired **shift** because the two disagreed. Both
readings were wrong, for the same reason.

## 1a. The baseline everyone has been subtracting contains the signal

`AR_RL_Trainer.py` pooled everything outside the TWAP window into
`inventory_without_twap`. On a TWAP-present episode that is
**(100,250] ∪ [400,550]** — and the pre-TWAP stretch is exactly where
front-running happens. A policy that builds its position before t=250 and holds
it through the window scores a paired shift of ~0 **by construction**. So the
"shift" metric the previous handoff set aside was not merely noisy, it was
broken; and "level" was never baselined at all.

The three windows are equal 150 s thirds, the out-of-window samples are appended
in time order, and kernel time is monotone (asserted at `Kernel.py:290`), so the
split is recoverable post hoc from the saved arrays: `pre = out[:len(inw)]`,
validated by `len(out)/len(inw) ≈ 2`. Measured on the finished runs that ratio
has median 2.02 and lies in [1.7,2.3] for **100%** of episodes, so the split is
sound. Recomputing criterion 1 against the **pre window only**:

| run | TWAP side | LEVEL | SHIFT (pooled baseline) | **response vs PRE** | wanted |
|---|---|---|---|---|---|
| `buy_base` | buy | −12.58 | −1.73 | **−4.01** [−5.58, −2.45] | positive |
| `sell_base` | sell | +3.58 | +0.78 | **+1.10** [−0.12, +2.32] | negative |

`buy_base` is **wrong-signed and significant**, and worse than the pooled shift
suggested (−4.01 vs −1.73). `sell_base` is wrong-signed but **not significant**,
and its last-20-episode value is −0.16, i.e. a null. Neither run front-runs; the
buy run is actively run over.

`inv_level_pre_window` is now recorded per episode, so this no longer has to be
inferred.

### Correction to my own first pass

I originally reported `corr(in-window level, out-of-window level)` = 0.936/0.961
as showing "87–92% of the level metric is a per-run directional bias". **That
inference was wrong**, and an independent reviewer caught it. Correlation is
computed on deviations from the run mean, so a per-run *constant* bias
contributes zero variance and is invisible to it. What the correlation actually
licenses is different and still useful: 88–92% of the *episode-to-episode
fluctuation* in the level is shared with the out-of-window level, i.e. a common
episode-level factor (training drift, the seed's market path, carried LSTM and
inventory state). So per-episode level carries almost no independent signal and
**n=80 episodes is closer to n≈10 effective** — which is its own reason to
distrust single-run readings.

The actual evidence that the level is dominated by a side-independent standing
position is simpler and does not need the correlation at all: the in-window and
out-of-window **means** are nearly equal (buy_base −12.58 vs −10.86; sell_base
+3.58 vs +2.80), and §1b below.

### Alignment hazard for anyone recomputing this

`inventories_without_twap.append(...)` runs once per episode unconditionally,
but `inventories_with_twap_buy/sell.append(...)` fire only when non-empty. On
the **alternating and on/off runs, index i of the buy array is not episode i**.
Map through `slippages_<label>sides.npy` and `slippages_<label>twap_present.npy`
(one entry per episode, both saved). `buy_base`/`sell_base` are single-side with
the TWAP present every episode, so their arrays do align and the numbers above
are safe. `episode_metrics_<label>.json` is immune — every record carries its
own episode index and side.

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

## 2a. The signal is small but NOT below the noise floor

A competing explanation would be that front-running is simply undetectable here.
It is not. Per-episode terminal PnL standard deviation, measured across the six
runs, is 0.75–1.37 dollars, so against the $0.75 prize:

| run | n | sd of episode PnL | SNR (prize/sd) | episodes to detect at 2σ |
|---|---|---|---|---|
| `buy_base` | 80 | 1.36 | 0.55 | 13 |
| `sell_base` | 80 | 0.75 | 1.00 | 4 |
| `both_base` | 70 | 1.37 | 0.55 | 13 |
| `both_onoff` | 84 | 1.10 | 0.68 | 9 |

Roughly 4–13 episodes of evidence, well inside a 120–160 episode run. So the
edge is learnable in principle, and the reason it has not been learned is the
shaping, not the measurability. This rules out "make the meta-order bigger" as a
necessary fix — though it would still help, and see §7.7.

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

Market orders *are* still submitted under config 1, but only by the
inventory-breach path (`mo = 4 if inv > 0 else 7` when `|inv| >= inventorylimit`),
which `return`s before any gating. Those are forced liquidations, symmetric by
construction, not position-taking — the policy still has no way to *choose* to
take a position. It also means the plots' many trajectories pinned at ±23–24 are
the agent riding the limit and being force-liquidated.

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

**Dormant in every run to date.** Two things have to hold for that and both do:
the policy path under config 1 can never emit u=4 or u=7, and the breach path,
which does emit them, returns before the gates are consulted. So the asymmetry
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
7. **Optional, a research-design question for the user, not an implementation
   detail:** the meta-order is 150 shares over 150 s. Making it larger or more
   aggressive (e.g. same size over 50 s) raises its price impact and therefore
   the prize, improving SNR roughly with impact. §2a says this is not *needed*,
   so it should be a deliberate choice about what market is being studied rather
   than a knob turned to make training easier.

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
gae_lambda-only correct on sell, weak on buy. **Discard it.** Two independent
reasons: it is all single-side and mis-baselined, so §1 applies; and an
independent review of `exploration_bonus` found **no side or sign asymmetry
anywhere in the implementation**, so the claimed "exploration owns buy,
gae_lambda owns sell" split has no mechanism either.

**Independent review (2026-09-12).** Commissioned this session; the previous
handoff noted two earlier reviewer attempts delivered nothing. Findings, grouped
by whether they confirm or correct.

*Confirmed.* Exactly **36 of 144** kernel entries inhibitory, verified from the
loaded pickle — and stronger than claimed, *every* parameter matrix is exactly
mirror-symmetric under i→11−i (`max|M − M[::-1,::-1]| = 0`). The bound-breach
rate survives: **6.53%** measured over 118,447 accepted events on 20 paths
(6.49% on an independent 8-seed run) against the handoff's 6.2%. The
`sample_dimension()` fix is correct (normalisation exact; negative per-dimension
intensities are clipped upstream at `Arrival_Models.py:330` before both the sum
and the draw) **and sufficient** — the residual event-*timing* bias is
side-symmetric by construction and leaves P(Ask)=0.49996. That closes the
previous handoff's open worry that timing could undermine the symmetry result.
Also confirmed: RL starting cash is 2500, the episode-boundary idiom is sound
(time monotonicity is asserted), and the with-TWAP window is exactly [250,400]
with no contamination.

*Corrected — the drift shift needs a different test.* The handoff argues
−4.48 → +0.58 bps from two separate one-sample tests (p=0.002 vs p=0.65), which
is the "significant versus not significant" fallacy, and the two reported CIs
overlap in [−1.92,−1.74]. Reconstructed as a proper two-sample test from those
CIs: **Δ = +5.06 bps [+1.40, +8.72], t≈2.75, p≈0.007**. The conclusion survives;
**report it as a difference with a CI**, not as two one-sample p-values.

*Corrected — RESOLVED BY EXPERIMENT. The magnitude is over-attributed.* An exact,
noise-free paired computation of the two assignment rules on identical realised
states gives a bias of **+0.243 pp in P(Ask) = +0.49% Ask excess**, identical on
all 20 seeds. The handoff attributes **+1.29 pp** of Ask-excess removal to the
fix (+1.77% → +0.48%) — about **2.6× more than the rule can produce per step**.
The residual ~0.8 pp needs an explanation. Either the near-critical cascade
amplifies it (spectral radius is forced to 0.99 at `Arrival_Models.py:278-280`)
or part of the improvement belongs to the still-unlocated exchange-side
asymmetry, which this fix cannot have touched. **The discriminating experiment was run — and it does NOT support the
over-attribution claim.** (`HAWKES_LEGACY_DIMENSION_RULE=1`, `44f2adc`, both
generators in isolation, n=200 paths per arm, 1,983,459 events.)

**Units first, because they caused two wrong write-ups before this one.** The
plan's `%` column is **`(A − B)/mean(A,B)` = 4·(P(Ask) − 0.5)**, not
`(A − B)/total`. Verified three independent ways: the six per-pair denominators
implied by the n=48 table sum to 12,243 against the TOTAL row's 12,226
(additive, so they are the same kind of quantity); the plan's own quoted counts
(`mo` 282, `lo_top` 4926) are **one-side** counts, since as pair totals they
would imply half the events; and the implied event rate settles it — factor 4
implies 44.5 events/s against the 49.7/s I measure in the same
isolated-generator, spread-pinned configuration, while factor 2 implies 22.2/s,
off by 55%. **Standardise on ΔP(Ask) in pp and convert once at the end.**

On the correct scale:

| quantity | plan % |
|---|---|
| legacy arm | +1.28% |
| fixed arm | +0.23% |
| **measured difference** | **+1.05%**, SE 0.65, CI [−0.23, +2.32] |
| exact per-step rule prediction | +0.97% |
| handoff's attributed difference | +1.29% |

* H0 = +0.97% (rule only): t=+0.12, **p=0.90 — cannot reject.**
* H0 = +1.29% (full attribution): t=−0.37, **p=0.71 — cannot reject.**

**Neither hypothesis is rejected, so the over-attribution claim is withdrawn.**
The residual is 1.29 − 0.97 = 0.32% against an SE of 0.65% — indistinguishable
from zero, and about a quarter of the attributed effect rather than 60% of it.
Resolving a 0.32% gap at 80% power needs SE ≈ 0.115%, roughly 6,400 paths per
arm. Not worth running.

What the experiment *does* establish, and these stand: the isolated-generator
arms reproduce the handoff's cluster arms to within noise (legacy +1.28% vs
+1.77%; fixed +0.23% vs +0.48%), and **the fixed arm is statistically
indistinguishable from zero**, independently corroborating "the generator is now
symmetric" and the reviewer's P(Ask)=0.49996.

**Consequence for what to work on: there is no orphaned residual, so the
exchange-side asymmetry is not implied by this evidence and should not be
chased on its account.** The predecessors' "`lo_deep` asymmetry visible with
kernels nulled" was reported in this same `%` convention — re-derive it as
ΔP(Ask) from raw counts before treating it as a lead.

*Corrected — the exploration-bonus arm story has no mechanism.* No side or sign
asymmetry exists anywhere in the `exploration_bonus` implementation, so the
prior ablation's "exploration owns buy, gae_lambda owns sell" has no candidate
cause, on top of resting on single-side mis-baselined runs.

*Corrected — `both_base`'s side contrast is confounded and cannot be read.* In a
1-on/0-off alternating run the side has period 2 while training fires on
`episode % 4 == 0`. Because 2 divides 4, side and training phase are **locked**:
every training session lands on a buy episode, so buy and sell episodes are
systematically generated by policies of different age. Pairing adjacent episodes
does not fix it — the buy member of every pair is always the just-updated one.
**Every run launched this session uses 2-on/1-off alternate**, giving side
period **3**, coprime with the training period 4 and the CEM period 8, so over
one cycle each side meets each training phase and each CEM state equally. That
is structurally safe, and it is pinned in `tests/test_schedule_aliasing.py`
alongside the failing schedule. Fixes for a period-2 run: `TWAP_SIDE_MODE=random`
or any side period coprime with 4 and 8.

*CEM: one claim withdrawn, one real defect found, one trap flagged.* The
reviewer initially reported that CEM pools elites with no split on side and then
**retracted it** — side balancing was already implemented. It also retracted its
account of the selector: `cem_full_episode=True` means the live path ranks
**whole episodes by total episode reward**, top 3 per side, and
`get_max_contiguous_rewards`'s subarray search is dead code in every run.

What is real:

* **TWAP-absent episodes could never be elites.** The pools were
  `episode_sides > 0` and `< 0` only; on an absent episode `TWAPPresent` is
  pinned to 0 all the way through, so `episode_sides[ep] == 0` and the episode
  fell into neither. In an alternating run both signed pools are always
  non-empty, so the global-top-5 fallback never fired either. **Self-imitation
  never reinforced a single standalone market-making episode — in exactly the
  on/off runs that exist to test criterion 2.** Fixed (`eccb871`): three
  regimes, top 3 of each non-empty pool.
* **Lowering `ACTION_BONUS` hands the elite criterion to the terminal penalty,
  not to PnL.** The ranking key is total episode reward, so at the default
  `terminal_invpenalty=25` one unit of terminal inventory costs −25 against an
  episode PnL of ±0.2: elites become "ended closest to flat". `TERMINAL_INVPENALTY`
  must come down with `ACTION_BONUS`. The `pnl_scaled` arms already do (1.2e-3);
  `reward_scale/bonus` did not and was corrected.
* **No elite floor.** `sorted(...)[:3]` returns three episodes however bad they
  are. Harmless while a deterministic action bonus dominated; once PnL
  dominates, episode totals are near zero-mean noise, so the top 3 of the ~40
  buffered episodes is roughly the +1.7σ tail — mostly luck, and CEM turns from
  a bias into a variance injector. `CEM_ELITE_FLOOR` (default 0 = unchanged)
  requires elites to clear the buffer mean by N SDs.

`USE_CEM` (default true) also exists now; `ps_nc` is the CEM-off arm.

*On the metrics.* SHIFT subtracts a baseline that contains the signal, so it is
**actively destructive**; LEVEL subtracts nothing, so it is merely uninformative
about ordering. That asymmetry is why the same runs read "correct" on LEVEL and
wrong-signed on SHIFT. The **side contrast differences across episodes rather
than across time windows within an episode, so the signal is never subtracted
from itself** — the reviewer calls it the best of the three and judged the
cancellation argument sound. It also refuted a worry of mine: nothing stateful
leaks across episodes except network weights, the visit counter and the
trajectory buffer — `AR_RL_Trainer.py:791-802` resets `last_state` (which
triggers `reset_hidden_state` on both actor-critics), `breach`, cash, inventory,
positions, profit and `statelog` every episode.

**Exchange defects found while chasing a residual that turned out not to exist.**
Recorded because they are real, not because they explain anything.

* **`Exchange.py:588` — a genuine non-mirror defect.** The ask branch of
  `regeneratequeuedepletion` assigns `self.askprice = self.askprices["Ask_L1"]`
  (scalar); the bid branch assigns `self.bidprice = [self.bidprices["Bid_L1"]]`
  — a **list**. It is in the cancel-depletion path. Most operations on it
  silently broadcast rather than raising (`np.float64 - list` → `array([0.03])`,
  `list + np.float64` → array, because `ticksize` is `np.float64` not a Python
  float), but **`:554`'s `del self.bids[self.bidprice]` raises
  `TypeError: unhashable type: 'list'`** — a hard crash. Since the six
  production runs had zero errors, that branch never fired; by generator
  symmetry neither did the ask branch, so the whole cancel-depletion promotion
  path is **dead in this configuration** and cannot have biased anything. It
  will crash on the first bid-L1 cancel depletion under any thinner-queue config
  (smaller `Pi_Q0`, fewer `numOrdersPerLevel`). Pinned in
  `tests/test_exchange_side_symmetry.py`, deliberately not fixed while 15 jobs
  are mid-flight.
* **`Exchange.py:601` — the spread-consistency check is disabled.**
  `condition5=(self.spread==np.round(abs(self.askprice-self.bidprice)), 2)` has
  the `, 2` outside the `np.round` call, so it builds the tuple `(bool, 2)`,
  which is always truthy. `checkLOBValidity` therefore never tests the spread.
  Symmetric, so not a bias source — but it is the guard that would have caught
  the item above. Even with the paren fixed it would be wrong: `np.round(0.02)`
  with no decimals is `0.0`.
* Ranked LOW with reasons, not merely asserted: price rounding (all prices sit
  on the 0.01 grid by repeated `±ticksize` from 100, float error ~1e-15, so
  round-half-to-even never has a tie to break); cancel selection at `:473`
  (side resolved into a local `queue` above, identical code both sides);
  `Pis`/`Pi_Q0` (symmetric in this config, though **by aliasing** — the trainer
  assigns the same list objects to both sides, so a future config that sets
  them independently could break it silently). The mirrored blocks — MO
  depletion, level promotion, in-spread placement, crossed-book checks — are
  exact reflections token for token.

**New issues from the review, with their status.**

1. **`rewardpenalty`/`eta` are dead code in the live objective.** The running
   quadratic inventory penalty is commented out; the only two other sites are
   guarded by `not alt_state` (the trainer passes `alt_state=True`) and
   `two_sided_reward` (passes `False`). `eta` reaches the reward **only** via
   `terminal_invpenalty = 5·eta = 25`. So mid-episode the agent carries ±25 at
   **zero cost**. The handoff, the plan and the run banner all cited it as a
   knob; the banner now prints both as INERT. Independently confirmed here.
2. **Reward double-count on duplicate kernel timestamps — SETTLED, does not
   fire.** `updatestatelog` refuses to append when `statelog[-1][0] ==
   current_time`, and `calculaterewards` would then return the previous step's
   reward verbatim. Measured across all six production runs: **0 ties in
   1,162,460 consecutive steps** (0 in each of 198688/198732/173876/206358/
   172971/211835). Fragility, not an active bug. No action.
3. **Breach steps are deleted from the learning signal — and this is the
   strongest mechanistic candidate for `buy_base`'s runaway.** `get_action`'s
   breach branch returns `(mo, (None, None))` **before setting `last_state`**,
   so `state_at_action` is the stale previous state — non-`None`, so the trainer
   calls `store_transition`, which early-returns on `d is None`. But
   `calculaterewards` has already run *as that call's argument* and advanced
   `statelog`. The breach step's PnL is therefore computed, printed and thrown
   away while the statelog pointer moves past it: **the forced liquidation's
   cost is credited to no stored transition at all**, rather than merely
   delayed. The only surviving in-episode pressure against the limit is
   `terminal_invpenalty` at the final step, so **the agent is never punished
   during an episode for hitting ±25**.

   Measured, fraction of steps at `|inv| >= 25`:

   | run | first 20 eps | last 20 eps | trend/ep | by 10-episode block |
   |---|---|---|---|---|
   | `buy_base` | 0.0150 | **0.0892** | **+0.0012** | .018 .012 .002 .032 .068 .018 .083 .095 |
   | `sell_base` | 0.0040 | 0.0000 | −0.0002 | .001 .007 .047 .000 .000 .003 .000 .000 |

   A ~6× growth over training in exactly the run the previous handoff called
   "the one genuine anomaly" (level −20.42, diverging at −0.17/ep against a
   limit of 25), and flat in the run that did not diverge. That is the predicted
   signature of a feedback loop: more limit touches → more deleted liquidation
   costs → less pressure against the limit. Note excursions are **not** one step
   each as a code-only bound suggests — 562 excursions of length > 1 in
   `buy_base` — because resting orders fill between RL steps, so the dropped
   fraction is close to the raw at-limit fraction.

   `frac_at_inventory_limit` is now recorded per episode.

   **DECISION NEEDED.** Fixing this touches reward attribution, and "the reward
   measures PnL change since the last statelog update, whichever event caused
   it" is on the supervisor-confirmed by-design list. Options: (a) leave it;
   (b) skip `calculaterewards` on dropped steps so the liquidation PnL accrues
   to the next stored transition; (c) store the breach transition with the
   policy's intended action, which is the convention already used for the other
   `get_action` overrides. Not changed unilaterally.

**A correction to the review itself.** It states CEM is on "for 3 of every 4
training sessions". It is on for **half**: `episode % 8 != 0` holds for 7 of
every 8 *episodes*, but training only runs on multiples of 4, and among those
`e % 8` alternates 0, 4 — so CEM-on sessions are e = 4, 12, 20, … CLAUDE.md says
the same. This does not change the pooling conclusion.

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

**(e) I published a rejection of the drift study's attribution three times, in
three different wrong units, and the correct answer is that nothing is
rejected.** First I tested the raw ΔP(Ask) figure against 0.49 and 1.29 as
though those were percentage points and reported **p<0.0001**. Then I decided
the plan's convention was `(A−B)/total` (factor 2) and reported **p=0.019**.
The convention is actually `(A−B)/mean(A,B)` — **factor 4** — under which the
measured difference is +1.05% ± 0.65 and **neither** the rule-only prediction
(+0.97%, p=0.90) nor the full attribution (+1.29%, p=0.71) can be rejected.
The over-attribution claim is withdrawn entirely.

Three lessons, in order of how much time each would have saved: (i) derive a
unit convention from the source's own numbers before using it — the plan's
quoted counts and its own additivity settle it in five minutes; (ii) a
cross-check that is independent of the convention, like the implied event rate,
catches this immediately; (iii) I took a collaborator's prose figure as
authoritative over their own script's output, and it was their slip. Both
earlier commit messages (`p<1e-4`, `p=0.019`) are wrong and stand only in
history.

**(f) I chased the exchange-side asymmetry on the strength of a residual that
does not exist.** The hunt was motivated by "0.8–1.0 pp attributed to nothing",
which was an artifact of (e). The two defects it turned up (§8, exchange
issues) are real and worth recording, but the campaign was not justified by the
evidence I had, and I started it before the unit question was settled.

**(d) I reported a correlation as evidence for a claim it cannot support.**
r=0.936/0.961 between in-window and out-of-window level does not show the level
is "90% bias" — a constant bias has zero variance and is invisible to a
correlation. An independent reviewer caught it. The conclusion survives on
different and simpler evidence (§1a), but the statistic was the wrong one and I
presented it as the headline.

**(c) The side-contrast numbers in §1b are read off plots, not computed.** The
alternating runs only write `inventorydists_*` on completion, so the numerical
contrast for `both_base`/`both_onoff` is not available until they finish. The
correlation analysis in §1a *is* computed, on finished runs, and is the stronger
evidence. New runs write the numbers from episode 1.
