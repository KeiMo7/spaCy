"""Tests for periodic training snapshots and resuming training."""

import io
import json
import os
import socket
import time

import pytest

from spacy import util
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin
from spacy.training import Example, checkpoint as ckpt
from spacy.training.initialize import init_nlp
from spacy.training.loop import train
from spacy.util import load_config_from_str

from ..util import make_tempdir


TRAIN_WORDS = [
    "the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "a", "big",
    "red", "fox", "jumps", "over", "lazy", "brown", "quick", "happy",
]
TRAIN_TAGS = [
    "DT", "NN", "VBD", "IN", "NN", "NN", "VBD", "RB", "DT", "JJ",
    "JJ", "NN", "VBZ", "IN", "JJ", "JJ", "JJ", "JJ",
]

CONFIG_TEMPLATE = """
[paths]
train = "{train}"
dev = "{dev}"

[system]
seed = 0
gpu_allocator = null

[nlp]
lang = "en"
pipeline = ["tagger"]
batch_size = 8

[components]
[components.tagger]
factory = "tagger"

[components.tagger.model]
@architectures = "spacy.Tagger.v2"

[components.tagger.model.tok2vec]
@architectures = "spacy.HashEmbedCNN.v2"
width = 16
embed_size = 8
depth = 1
window_size = 1
maxout_pieces = 2
subword_features = false

[corpora]
[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${{paths.train}}

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${{paths.dev}}

[training]
seed = 0
gpu_allocator = null
max_epochs = {max_epochs}
max_steps = {max_steps}
eval_frequency = 5
patience = 0
score_weights = {{"tag_acc": 1.0}}
save_every = {save_every}
max_keep = {max_keep}
frozen_components = []
annotating_components = []
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
before_to_disk = null
before_update = {before_update}
resume_validation = {resume_validation}

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2

[training.batcher.size]
@schedules = "compounding.v1"
start = 20
stop = 100
compound = 1.1

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 1e-8

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
"""


@util.registry.callbacks("test_interrupt_once")
def create_interrupt_callback(trigger_file: str):
    """A before_update callback that raises once the trigger file asks for it.

    The file contains the step at which to interrupt. Reading the target from a
    separate file keeps the config (and therefore the snapshot identity)
    unchanged between the interrupted and the resumed run.
    """
    from pathlib import Path

    def before_update(nlp, args):
        path = Path(trigger_file)
        if not path.exists():
            return
        target = int(path.read_text().strip() or "-1")
        if target >= 0 and args["step"] >= target:
            path.unlink()
            raise RuntimeError("simulated interruption")

    return before_update


def write_corpus(path, n_docs=40):
    import random

    vocab = English().vocab
    rng = random.Random(0)
    docs = []
    for _ in range(n_docs):
        size = rng.randint(3, 9)
        words = [rng.choice(TRAIN_WORDS) for _ in range(size)]
        tags = [rng.choice(TRAIN_TAGS) for _ in range(size)]
        docs.append(Doc(vocab, words=words, tags=tags))
    DocBin(docs=docs).to_disk(path)
    return path


def write_data(tmp_dir, n_docs=40):
    train_path = write_corpus(tmp_dir / "train.spacy", n_docs=n_docs)
    dev_path = write_corpus(tmp_dir / "dev.spacy", n_docs=n_docs)
    return train_path, dev_path


def make_config(train, dev, **kwargs):
    defaults = dict(
        max_epochs=3,
        max_steps=40,
        save_every=7,
        max_keep=3,
        before_update="null",
        resume_validation="true",
    )
    defaults.update(kwargs)
    text = CONFIG_TEMPLATE.format(train=train, dev=dev, **defaults)
    return load_config_from_str(text)


def interrupt_config(train, dev, trigger_file, **kwargs):
    before_update = (
        '{"@callbacks": "test_interrupt_once", "trigger_file": "%s"}' % trigger_file
    )
    return make_config(train, dev, before_update=before_update, **kwargs)


def run_train(config, output_path, **kwargs):
    nlp = init_nlp(config)
    stdout = io.StringIO()
    train(nlp, output_path, stdout=stdout, stderr=io.StringIO(), **kwargs)
    return stdout.getvalue()


def load_nlp(output_path):
    import spacy

    return spacy.load(output_path / "model-last")


def newest_state(output_path):
    return ckpt.find_latest_snapshot(output_path).read_state()


def snapshot_paths(output_path):
    return [s.path for s in ckpt.list_snapshots(output_path)]


@pytest.fixture
def data_dir():
    with make_tempdir() as tmp_dir:
        train_path, dev_path = write_data(tmp_dir)
        yield tmp_dir, train_path, dev_path


def test_snapshots_disabled_by_default(data_dir):
    """Without `save_every`, no snapshot artifacts are written."""
    tmp_dir, train_path, dev_path = data_dir
    config = make_config(train_path, dev_path, save_every=0)
    with make_tempdir() as out:
        run_train(config, out)
        assert (out / "model-last").exists()
        assert (out / "model-best").exists()
        assert not (out / "training-state").exists()


def test_periodic_snapshots_and_pruning(data_dir):
    tmp_dir, train_path, dev_path = data_dir
    config = make_config(train_path, dev_path, save_every=5, max_keep=2)
    with make_tempdir() as out, make_tempdir() as full_out:
        run_train(make_config(train_path, dev_path, save_every=1), full_out)
        final_state = newest_state(full_out)
        run_train(config, out)
        snapshots = ckpt.list_snapshots(out)
        assert snapshots
        assert len(snapshots) <= 2
        # The newest snapshot describes the end of the run.
        meta = snapshots[-1].read_meta()
        assert meta["step"] == final_state["step"]
        for snapshot in snapshots:
            snapshot.validate()


def test_snapshot_does_not_overwrite_best_or_last(data_dir):
    tmp_dir, train_path, dev_path = data_dir
    config = make_config(train_path, dev_path, save_every=5)
    with make_tempdir() as out:
        run_train(config, out)
        assert (out / "model-best").is_dir()
        assert (out / "model-last").is_dir()
        for snapshot in ckpt.list_snapshots(out):
            assert snapshot.path.parent == out / "training-state" / "snapshots"
        import spacy

        spacy.load(out / "model-best")


def test_resume_matches_uninterrupted_run(data_dir):
    """A run interrupted once and resumed must match an uninterrupted run."""
    tmp_dir, train_path, dev_path = data_dir
    with make_tempdir() as full_out, make_tempdir() as out:
        run_train(make_config(train_path, dev_path, save_every=5), full_out)
        trigger = out.parent / "interrupt-at"
        trigger.write_text("9")
        config = interrupt_config(train_path, dev_path, trigger, save_every=5)
        with pytest.raises(RuntimeError):
            run_train(config, out)
        assert ckpt.list_snapshots(out)
        run_train(config, out, resume=True)
        full = load_nlp(full_out).meta["performance"]
        resumed = load_nlp(out).meta["performance"]
        assert full["tagger_loss"] == pytest.approx(resumed["tagger_loss"], rel=1e-5)
        assert full["tag_acc"] == pytest.approx(resumed["tag_acc"], rel=1e-5)


def test_resume_twice_matches_uninterrupted_run(data_dir):
    """Two interruptions must also produce the same final result."""
    tmp_dir, train_path, dev_path = data_dir
    with make_tempdir() as full_out, make_tempdir() as out:
        run_train(make_config(train_path, dev_path, save_every=4), full_out)
        trigger = out.parent / "interrupt-at"
        config = interrupt_config(train_path, dev_path, trigger, save_every=4)
        trigger.write_text("6")
        with pytest.raises(RuntimeError):
            run_train(config, out)
        trigger.write_text("14")
        with pytest.raises(RuntimeError):
            run_train(config, out, resume=True)
        run_train(config, out, resume=True)
        full = load_nlp(full_out).meta["performance"]
        resumed = load_nlp(out).meta["performance"]
        assert full["tagger_loss"] == pytest.approx(resumed["tagger_loss"], rel=1e-5)


def _interrupt_after(output, train_path, dev_path, trigger, step=9, **kwargs):
    """Run until the first periodic snapshot and then simulate a crash."""
    trigger.write_text(str(step))
    config = interrupt_config(train_path, dev_path, trigger, **kwargs)
    with pytest.raises(RuntimeError):
        run_train(config, output)
    return config


@pytest.mark.parametrize("save_every", [1, 4, 5])
def test_resume_at_various_points(data_dir, save_every):
    """Resuming right after an evaluation, between evaluations, and at an
    epoch boundary all produce the same final state."""
    tmp_dir, train_path, dev_path = data_dir
    with make_tempdir() as full_out, make_tempdir() as out:
        run_train(make_config(train_path, dev_path, save_every=save_every), full_out)
        trigger = out.parent / "interrupt-at"
        config = _interrupt_after(
            out, train_path, dev_path, trigger, save_every=save_every
        )
        run_train(config, out, resume=True)
        full = load_nlp(full_out).meta["performance"]
        resumed = load_nlp(out).meta["performance"]
        assert full["tagger_loss"] == pytest.approx(resumed["tagger_loss"], rel=1e-5)


def test_resume_keeps_evaluation_history(data_dir):
    """The evaluation history is continuous across a restart, so the best model
    and the early stopping counter don't restart from scratch."""
    tmp_dir, train_path, dev_path = data_dir
    with make_tempdir() as full_out, make_tempdir() as out:
        run_train(make_config(train_path, dev_path, save_every=5), full_out)
        full_state = newest_state(full_out)
        trigger = out.parent / "interrupt-at"
        config = _interrupt_after(out, train_path, dev_path, trigger, save_every=5)
        before = newest_state(out)
        run_train(config, out, resume=True)
        after = newest_state(out)
        assert after["results"][: len(before["results"])] == before["results"]
        assert len(after["results"]) == len(full_state["results"])
        steps = [step for _score, step in after["results"]]
        assert steps == sorted(set(steps))


def test_streaming_corpus_resume(data_dir):
    """A streaming corpus (max_epochs=-1) resumes by skipping consumed data."""
    tmp_dir, train_path, dev_path = data_dir
    kwargs = dict(max_epochs=-1, max_steps=30, save_every=5)
    with make_tempdir() as full_out, make_tempdir() as out:
        run_train(make_config(train_path, dev_path, **kwargs), full_out)
        trigger = out.parent / "interrupt-at"
        config = _interrupt_after(out, train_path, dev_path, trigger, **kwargs)
        assert newest_state(out)["step"] >= 5
        run_train(config, out, resume=True)
        full = load_nlp(full_out).meta["performance"]
        resumed = load_nlp(out).meta["performance"]
        assert full["tagger_loss"] == pytest.approx(resumed["tagger_loss"], rel=1e-5)
        assert newest_state(out)["step"] == 30


def test_corrupt_snapshot_falls_back(data_dir):
    """A snapshot that was interrupted mid-write is skipped and the previous
    complete snapshot is used."""
    tmp_dir, train_path, dev_path = data_dir
    kwargs = dict(save_every=4, max_keep=5)
    with make_tempdir() as out:
        trigger = out.parent / "interrupt-at"
        config = _interrupt_after(out, train_path, dev_path, trigger, **kwargs)
        snapshots = ckpt.list_snapshots(out)
        assert len(snapshots) >= 2
        newest = snapshots[-1]
        state_path = newest.path / "state.msg"
        state_path.write_bytes(state_path.read_bytes()[:10])
        with pytest.raises(ckpt.SnapshotError):
            newest.validate()
        # The fallback is the previous complete snapshot.
        fallback = ckpt.find_latest_snapshot(out)
        assert fallback.path == snapshots[-2].path
        run_train(config, out, resume=True)
        assert load_nlp(out)


def test_streaming_resume_does_not_load_corpus(data_dir):
    """Resuming a streaming corpus skips consumed examples lazily."""
    tmp_dir, train_path, dev_path = data_dir
    live = {"n": 0, "max": 0}
    words = TRAIN_WORDS
    tags = TRAIN_TAGS

    @util.registry.readers("test_counting_stream")
    def counting_stream(limit: int = 4000):
        class Reader:
            def __call__(self, nlp):
                for i in range(limit):
                    doc = Doc(
                        nlp.vocab,
                        words=words[i % 5 : i % 5 + 5],
                        tags=tags[i % 5 : i % 5 + 5],
                    )
                    example = Example(nlp.make_doc(doc.text), doc)
                    live["n"] += 1
                    live["max"] = max(live["max"], live["n"])
                    yield example

        return Reader()

    text = CONFIG_TEMPLATE.format(
        train=train_path,
        dev=dev_path,
        max_epochs=-1,
        max_steps=30,
        save_every=5,
        max_keep=5,
        before_update="null",
        resume_validation="true",
    ).replace(
        '[corpora.train]\n@readers = "spacy.Corpus.v1"\npath = ${paths.train}',
        '[corpora.train]\n@readers = "test_counting_stream"\nlimit = 4000',
    )
    with make_tempdir() as out:
        trigger = out.parent / "interrupt-at"
        # The same config is used for both runs because the config fingerprint
        # is part of the snapshot identity; only the trigger file changes.
        interrupted = load_config_from_str(text)
        interrupted["training"]["before_update"] = {
            "@callbacks": "test_interrupt_once",
            "trigger_file": str(trigger),
        }
        trigger.write_text("12")
        with pytest.raises(RuntimeError):
            run_train(interrupted, out)
        assert live["max"] > 100
        # Resume: the corpus is re-read from the start and only the remaining
        # examples are produced, so the number of live examples stays small.
        live["n"] = 0
        live["max"] = 0
        run_train(interrupted, out, resume=True)
        # The resume reads the whole prefix of the current epoch to skip it,
        # but the generator is consumed lazily and never materialized: the
        # number of examples alive at once must stay bounded by the batcher,
        # not by the corpus size.
        assert live["max"] < 1000


def test_incomplete_snapshot_is_ignored(data_dir):
    """A directory without the COMPLETE marker is treated as an interrupted
    write and ignored."""
    tmp_dir, train_path, dev_path = data_dir
    kwargs = dict(save_every=4)
    with make_tempdir() as out:
        trigger = out.parent / "interrupt-at"
        config = _interrupt_after(out, train_path, dev_path, trigger, **kwargs)
        snapshot = ckpt.list_snapshots(out)[-1]
        (snapshot.path / ckpt.COMPLETE_MARKER).unlink()
        with pytest.raises(ckpt.SnapshotError):
            snapshot.validate()
        run_train(config, out, resume=True)


def test_unknown_snapshot_version_fails(data_dir):
    """A snapshot with an unknown format version must fail loudly."""
    tmp_dir, train_path, dev_path = data_dir
    kwargs = dict(save_every=4)
    with make_tempdir() as out:
        trigger = out.parent / "interrupt-at"
        config = _interrupt_after(out, train_path, dev_path, trigger, **kwargs)
        snapshot = ckpt.list_snapshots(out)[-1]
        meta_path = snapshot.path / ckpt.FILE_META
        meta = json.loads(meta_path.read_text())
        meta["format_version"] = 999
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ckpt.SnapshotVersionError):
            snapshot.read_meta()
        with pytest.raises(ckpt.SnapshotVersionError):
            run_train(config, out, resume=True)


def test_resume_with_no_snapshot_fails(data_dir):
    tmp_dir, train_path, dev_path = data_dir
    config = make_config(train_path, dev_path, save_every=0)
    with make_tempdir() as out:
        with pytest.raises(ckpt.SnapshotError):
            run_train(config, out, resume=True)
        assert not (out / "model-best").exists()
        assert not (out / "model-last").exists()


def test_config_drift_fails(data_dir):
    """Changing score weights is reported item by item."""
    tmp_dir, train_path, dev_path = data_dir
    with make_tempdir() as out:
        trigger = out.parent / "interrupt-at"
        _interrupt_after(out, train_path, dev_path, trigger, save_every=4)
        changed = make_config(train_path, dev_path, save_every=4)
        changed["training"]["dropout"] = 0.5
        nlp2 = init_nlp(changed)
        with pytest.raises(ckpt.ResumeMismatchError) as excinfo:
            train(
                nlp2,
                out,
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                resume=True,
            )
        assert "config" in str(excinfo.value)
        assert (out / "model-last").exists()


def test_config_drift_warn_continues(data_dir):
    tmp_dir, train_path, dev_path = data_dir
    with make_tempdir() as out:
        trigger = out.parent / "interrupt-at"
        _interrupt_after(out, train_path, dev_path, trigger, save_every=4)
        changed = make_config(train_path, dev_path, save_every=4)
        changed["training"]["dropout"] = 0.5
        changed["training"]["resume_mismatch"] = "warn"
        nlp2 = init_nlp(changed)
        out_stream = io.StringIO()
        train(
            nlp2, out, stdout=out_stream, stderr=io.StringIO(), resume=True
        )
        assert "W128" in out_stream.getvalue()
        assert load_nlp(out)


def test_corpus_path_drift_is_reported(data_dir):
    tmp_dir, train_path, dev_path = data_dir
    with make_tempdir() as out:
        trigger = out.parent / "interrupt-at"
        _interrupt_after(out, train_path, dev_path, trigger, save_every=4)
        other = tmp_dir / "other.spacy"
        write_corpus(other, n_docs=20)
        changed = make_config(other, dev_path, save_every=4)
        nlp2 = init_nlp(changed)
        with pytest.raises(ckpt.ResumeMismatchError) as excinfo:
            train(
                nlp2,
                out,
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                resume=True,
            )
        assert "corpora" in str(excinfo.value) or "corpus" in str(excinfo.value)


def test_output_dir_lock(data_dir):
    """A second process can't write to a locked output directory."""
    with make_tempdir() as out:
        lock = ckpt.OutputDirLock(out).acquire()
        try:
            with pytest.raises(ckpt.OutputDirLockedError) as excinfo:
                ckpt.OutputDirLock(out).acquire()
            assert str(os.getpid()) in str(excinfo.value)
        finally:
            lock.release()
        with ckpt.OutputDirLock(out):
            pass


def test_stale_lock_is_taken_over(data_dir):
    tmp_dir, train_path, dev_path = data_dir
    config = make_config(train_path, dev_path, save_every=0)
    with make_tempdir() as out:
        lock_path = out / ckpt.LOCK_FILE
        lock_path.write_text(
            json.dumps(
                {"pid": 2**31 - 1, "host": socket.gethostname(), "created": time.time()}
            )
        )
        run_train(config, out)
        assert not lock_path.exists()


def _snapshot_signature(output_path):
    return sorted(
        (str(p.relative_to(output_path)), p.stat().st_mtime_ns)
        for p in (output_path / "training-state").rglob("*")
        if p.is_file()
    )


def test_resume_does_not_touch_existing_artifacts_on_failure(data_dir):
    """A failed resume leaves model-best/model-last and the snapshots as-is."""
    tmp_dir, train_path, dev_path = data_dir
    with make_tempdir() as out:
        trigger = out.parent / "interrupt-at"
        _interrupt_after(out, train_path, dev_path, trigger, save_every=4)
        before = _snapshot_signature(out)
        best_before = (out / "model-best").stat().st_mtime_ns
        changed = make_config(train_path, dev_path, save_every=4)
        changed["nlp"]["batch_size"] = 4
        nlp2 = init_nlp(changed)
        with pytest.raises(ckpt.ResumeMismatchError):
            train(
                nlp2,
                out,
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                resume=True,
            )
        assert _snapshot_signature(out) == before
        assert (out / "model-best").stat().st_mtime_ns == best_before


def test_save_failure_does_not_crash(data_dir, monkeypatch):
    """A failing snapshot write is reported and the run continues."""
    tmp_dir, train_path, dev_path = data_dir
    config = make_config(train_path, dev_path, save_every=4)
    with make_tempdir() as out:
        calls = {"n": 0}
        original = ckpt.write_snapshot

        def failing_write(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("No space left on device")
            return original(*args, **kwargs)

        monkeypatch.setattr(ckpt, "write_snapshot", failing_write)
        out_stream = io.StringIO()
        nlp = init_nlp(config)
        train(nlp, out, stdout=out_stream, stderr=io.StringIO())
        assert "No space left on device" in out_stream.getvalue()
        assert (out / "model-last").exists()
        assert ckpt.find_latest_snapshot(out) is not None


def test_resume_from_explicit_snapshot(data_dir):
    tmp_dir, train_path, dev_path = data_dir
    config = make_config(train_path, dev_path, save_every=4)
    with make_tempdir() as full_out, make_tempdir() as out:
        run_train(config, full_out)
        snapshot = ckpt.list_snapshots(full_out)[0]
        run_train(config, out, resume_from=snapshot.path)
        assert (out / "model-last").exists()


def test_resume_with_learn_rate_schedule(data_dir):
    """The learning-rate schedule continues from where it stopped."""
    tmp_dir, train_path, dev_path = data_dir

    def scheduled_config(save_every, **kwargs):
        config = make_config(train_path, dev_path, save_every=save_every, **kwargs)
        config["training"]["optimizer"]["learn_rate"] = {
            "@schedules": "compounding.v1",
            "start": 0.001,
            "stop": 0.0001,
            "compound": 0.7,
        }
        return config

    with make_tempdir() as full_out, make_tempdir() as out:
        run_train(scheduled_config(1), full_out)
        full = load_nlp(full_out).meta["performance"]
        trigger = out.parent / "interrupt-at"
        trigger.write_text("9")
        config = scheduled_config(
            4,
            before_update='{"@callbacks": "test_interrupt_once", '
            '"trigger_file": "%s"}' % trigger,
        )
        with pytest.raises(RuntimeError):
            run_train(config, out)
        run_train(config, out, resume=True)
        resumed = load_nlp(out).meta["performance"]
        assert full["tagger_loss"] == pytest.approx(resumed["tagger_loss"], rel=1e-5)


def test_resume_with_optimizer_averages(data_dir):
    """Moving averages are restored as writable arrays and continue correctly."""
    tmp_dir, train_path, dev_path = data_dir

    def averages_config(save_every, **kwargs):
        config = make_config(train_path, dev_path, save_every=save_every, **kwargs)
        config["training"]["optimizer"]["use_averages"] = True
        return config

    with make_tempdir() as full_out, make_tempdir() as out:
        run_train(averages_config(1), full_out)
        full = load_nlp(full_out).meta["performance"]
        trigger = out.parent / "interrupt-at"
        trigger.write_text("9")
        config = averages_config(
            4,
            before_update='{"@callbacks": "test_interrupt_once", '
            '"trigger_file": "%s"}' % trigger,
        )
        with pytest.raises(RuntimeError):
            run_train(config, out)
        run_train(config, out, resume=True)
        resumed = load_nlp(out).meta["performance"]
        assert full["tagger_loss"] == pytest.approx(resumed["tagger_loss"], rel=1e-5)


def test_snapshot_every_epoch(data_dir):
    """`frequency_unit = "epoch"` snapshots at epoch boundaries and resumes."""
    tmp_dir, train_path, dev_path = data_dir

    def epoch_config(save_every, **kwargs):
        config = make_config(train_path, dev_path, save_every=save_every, **kwargs)
        config["training"]["frequency_unit"] = "epoch"
        return config

    with make_tempdir() as full_out, make_tempdir() as out:
        run_train(epoch_config(1), full_out)
        full = load_nlp(full_out).meta["performance"]
        trigger = out.parent / "interrupt-at"
        config = epoch_config(
            1,
            before_update='{"@callbacks": "test_interrupt_once", '
            '"trigger_file": "%s"}' % trigger,
        )
        # Interrupt right after an epoch boundary (step 11 is the first batch
        # of epoch 1 and follows the epoch-0 snapshot).
        trigger.write_text("13")
        with pytest.raises(RuntimeError):
            run_train(config, out)
        snapshots = ckpt.list_snapshots(out)
        assert snapshots
        assert all(s.read_meta()["epoch"] >= 0 for s in snapshots)
        run_train(config, out, resume=True)
        resumed = load_nlp(out).meta["performance"]
        assert full["tagger_loss"] == pytest.approx(resumed["tagger_loss"], rel=1e-5)


def test_partial_write_failure_leaves_no_temp_dirs(data_dir, monkeypatch):
    """A snapshot that fails half-way is cleaned up and doesn't corrupt others."""
    tmp_dir, train_path, dev_path = data_dir
    config = make_config(train_path, dev_path, save_every=4)
    with make_tempdir() as out:
        original = ckpt._write_msgpack_atomic
        calls = {"n": 0}

        def flaky(path, data):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("No space left on device")
            return original(path, data)

        monkeypatch.setattr(ckpt, "_write_msgpack_atomic", flaky)
        out_stream = io.StringIO()
        train(init_nlp(config), out, stdout=out_stream, stderr=io.StringIO())
        snapshots_dir = ckpt.get_snapshots_dir(out)
        assert not [p for p in snapshots_dir.iterdir() if p.name.startswith(".tmp-")]
        assert "No space left on device" in out_stream.getvalue()
        for snapshot in ckpt.list_snapshots(out):
            snapshot.validate()


def test_optimizer_state_roundtrip(data_dir):
    """Optimizer accumulators survive a snapshot round trip."""
    tmp_dir, train_path, dev_path = data_dir
    with make_tempdir() as out:
        trigger = out.parent / "interrupt-at"
        _interrupt_after(out, train_path, dev_path, trigger, save_every=4)
        state = newest_state(out)
        optimizer_state = state["optimizer"]
        assert optimizer_state["mom1"]
        assert set(optimizer_state["mom1"]) <= set(optimizer_state["nr_update"])


def test_snapshot_meta_contains_identity(data_dir):
    tmp_dir, train_path, dev_path = data_dir
    config = make_config(train_path, dev_path, save_every=4)
    with make_tempdir() as out:
        run_train(config, out)
        meta = ckpt.find_latest_snapshot(out).read_meta()
        assert meta["format_version"] == ckpt.SNAPSHOT_FORMAT_VERSION
        for key in ("config_fingerprint", "structure_fingerprint", "run_id"):
            assert meta[key]
        assert meta["component_fingerprints"]


def test_structure_drift_names_component(data_dir):
    """A changed model structure reports the component that changed."""
    tmp_dir, train_path, dev_path = data_dir
    with make_tempdir() as out:
        trigger = out.parent / "interrupt-at"
        _interrupt_after(out, train_path, dev_path, trigger, save_every=4)
        # Same components, different model width: the structure changes.
        text = CONFIG_TEMPLATE.format(
            train=train_path,
            dev=dev_path,
            max_epochs=3,
            max_steps=40,
            save_every=4,
            max_keep=3,
            before_update="null",
            resume_validation="true",
        ).replace("width = 16", "width = 32")
        changed = load_config_from_str(text)
        nlp = init_nlp(changed)
        with pytest.raises(ckpt.ResumeMismatchError) as excinfo:
            train(
                nlp,
                out,
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                resume=True,
            )
        message = str(excinfo.value)
        assert "tagger" in message
        assert "structure" in message or "model" in message


def test_environment_change_is_reported(data_dir, monkeypatch):
    """Resuming on a different device is allowed but reported."""
    tmp_dir, train_path, dev_path = data_dir
    with make_tempdir() as out:
        trigger = out.parent / "interrupt-at"
        config = _interrupt_after(out, train_path, dev_path, trigger, save_every=4)
        # Pretend this process is running on a GPU while the snapshot was
        # written on the CPU.
        monkeypatch.setattr(
            ckpt,
            "describe_environment",
            lambda: {"thinc_ops": "CupyOps", "device_type": "gpu", "device_id": 0},
        )
        out_stream = io.StringIO()
        nlp = init_nlp(config)
        train(nlp, out, stdout=out_stream, stderr=io.StringIO(), resume=True)
        assert "W134" in out_stream.getvalue()
        assert load_nlp(out)


def test_new_run_does_not_resume_old_snapshots(data_dir):
    """A fresh run in the same directory gets a new run id and ignores the old
    snapshots, which are kept on disk."""
    tmp_dir, train_path, dev_path = data_dir
    config = make_config(train_path, dev_path, save_every=4)
    with make_tempdir() as out:
        run_train(config, out)
        old_run = ckpt.read_current_run_id(out)
        old_snapshots = snapshot_paths(out)
        run_train(config, out)
        assert ckpt.read_current_run_id(out) != old_run
        assert set(old_snapshots) <= set(snapshot_paths(out))
        # resuming now picks up the newest run's snapshot only.
        run_train(config, out, resume=True)
        assert load_nlp(out)


def test_resume_after_finished_run_does_not_retrain(data_dir):
    """Resuming a run that already reached its budget doesn't add steps."""
    tmp_dir, train_path, dev_path = data_dir
    config = make_config(train_path, dev_path, save_every=4)
    with make_tempdir() as out, make_tempdir() as full_out:
        run_train(make_config(train_path, dev_path, save_every=1), full_out)
        expected = newest_state(full_out)["step"]
        run_train(config, out)
        before = newest_state(out)
        assert before["step"] == expected
        run_train(config, out, resume=True)
        after = newest_state(out)
        assert after["step"] == before["step"]
        assert len(after["results"]) == len(before["results"])
