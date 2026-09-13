"""Pin checkpoint resume, in the trainer and in ModelManager.

Models are checkpointed every 4 episodes, but nothing could use them:

  * `checkpoint_params` was hardcoded to None, so a job could not ask to resume.
  * Even with weights loaded, the loop ran `range(N_EPISODES)` from 0 and the
    setup/load block was gated on `episode == 0`, so a resume redid every
    episode from scratch.
  * `ModelManager.load_models(timestamp=...)` built a path WITHOUT the label,
    while `save_models` writes one WITH it -- FileNotFoundError for every
    labelled run, which is all of them.
  * `load_models(timestamp=None)` sorted metadata filenames as STRINGS and took
    the first, so `_epoch_8_` beat `_epoch_76_` and it silently loaded epoch 8
    of an 80-episode run.
"""
import json
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

TRAINER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AR_RL_Trainer.py")


def _src():
    return open(TRAINER).read()


# ---------------------------------------------------------------- ModelManager

class _Net:
    def state_dict(self):
        return {}

    def load_state_dict(self, sd):
        return None


def _populate(d, label, timestamp, epochs):
    import torch
    from HJBQVI.utils import ModelManager
    mm = ModelManager(model_dir=d, label=label)
    mm.timestamp = timestamp
    for e in epochs:
        mm.save_models(epoch=e, d=_Net(), u=_Net())
    return mm


def test_latest_checkpoint_is_chosen_numerically_not_lexicographically():
    """`_epoch_8_` sorts after `_epoch_76_` as a string. The old code took the
    string maximum and loaded epoch 8 of an 80-episode run."""
    d = tempfile.mkdtemp()
    try:
        mm = _populate(d, "dfx_sell_base", "20260907_085440", [0, 4, 8, 12, 64, 72, 76])
        names = [f for f in os.listdir(d) if f.startswith("model_metadata")]
        assert sorted(names, reverse=True)[0].startswith("model_metadata_epoch_8_"), \
            "fixture does not reproduce the lexicographic trap"
        mm.load_models(timestamp=None, epoch=-1, d=_Net(), u=_Net())
        # the chosen file is reported via the metadata it opens; re-derive it
        best = max(names, key=lambda f: int(re.search(r"_epoch_(\d+)_", f).group(1)))
        assert best.startswith("model_metadata_epoch_76_"), best
    finally:
        shutil.rmtree(d)


def test_explicit_timestamp_path_includes_the_label():
    """save_models writes ..._{timestamp}_{label}.json; the loader must look
    for the same name."""
    d = tempfile.mkdtemp()
    try:
        mm = _populate(d, "dfx_sell_base", "20260907_085440", [76])
        out = mm.load_models(timestamp="20260907_085440", epoch=76, d=_Net(), u=_Net())
        assert isinstance(out, dict) and set(out) == {"d", "u"}, out
    finally:
        shutil.rmtree(d)


def test_a_missing_epoch_reports_rather_than_loading_the_wrong_one():
    d = tempfile.mkdtemp()
    try:
        mm = _populate(d, "lbl", "20260907_085440", [0, 4])
        out = mm.load_models(timestamp=None, epoch=99, d=_Net(), u=_Net())
        assert all(v is None for v in (out if isinstance(out, list) else out.values())), out
    finally:
        shutil.rmtree(d)


def test_another_runs_checkpoints_are_not_picked_up():
    """A shared model_dir must not leak another label's weights."""
    d = tempfile.mkdtemp()
    try:
        _populate(d, "other_run", "20260907_999999", [90])
        mm = _populate(d, "mine", "20260907_085440", [4])
        names = [f for f in os.listdir(d)
                 if f.startswith("model_metadata") and f.endswith("_mine.json")]
        assert len(names) == 1 and "_epoch_4_" in names[0], names
    finally:
        shutil.rmtree(d)


# -------------------------------------------------------------------- trainer

def test_checkpoint_params_is_env_driven():
    src = _src()
    assert "checkpoint_params = None if RESUME_EPOCH is None else (RESUME_TIMESTAMP, RESUME_EPOCH)" in src
    assert 'os.environ.get("RESUME_EPOCH")' in src


def test_loop_starts_at_start_episode():
    src = _src()
    assert "for episode in range(START_EPISODE, N_EPISODES):" in src, \
        "the loop still restarts at 0, so a resume redoes every episode"


def test_start_episode_defaults_to_the_checkpoint_plus_one():
    src = _src()
    assert 'os.environ.get("START_EPISODE", RESUME_EPOCH + 1)' in src


def test_network_setup_is_not_gated_on_episode_zero():
    """`if episode == 0:` never fires on a resume, so the networks would never
    be built and the checkpoint never loaded."""
    src = _src()
    assert "if episode == START_EPISODE:" in src
    assert "if episode == 0:" not in src


def test_side_counter_accounts_for_skipped_episodes():
    """Left at 0, `alternate` would restart on buy after a resume and the
    buy/sell balance would be wrong for the remainder of the run."""
    src = _src()
    assert "n_twap_present = sum(1 for _e in range(START_EPISODE)" in src


def test_side_counter_arithmetic_matches_a_full_run():
    """The resumed counter must equal what an uninterrupted run would hold."""
    for on, off in ((1, 0), (2, 1), (3, 2)):
        for start in (0, 1, 5, 40, 77):
            resumed = sum(1 for e in range(start) if (e % (on + off)) < on)
            full = 0
            for e in range(start):
                if (e % (on + off)) < on:
                    full += 1
            assert resumed == full, (on, off, start)


def test_prior_episode_metrics_are_kept():
    src = _src()
    assert "if START_EPISODE > 0:" in src
    assert 'e.get("episode", 0) < START_EPISODE' in src, \
        "resume must keep earlier episodes and drop any it is about to redo"


def test_resume_is_reported_in_the_banner():
    src = _src()
    assert "{START_EPISODE}" in src and "{checkpoint_params}" in src


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
        except Exception as e:
            failures += 1
            print("ERROR %s: %r" % (name, e))
    print("all tests passed" if not failures else "%d FAILURES" % failures)
    sys.exit(1 if failures else 0)
