import functools
import itertools
import random
import shutil
import sys

import numpy
from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from thinc.api import Config, Optimizer, constant, fix_random_seed, set_gpu_allocator
from wasabi import Printer

from ..errors import Errors, Warnings
from ..schemas import ConfigSchemaTraining
from ..util import logger, registry, resolve_dot_names
from . import checkpoint as ckpt
from .example import Example

if TYPE_CHECKING:
    from ..language import Language  # noqa: F401


DIR_MODEL_BEST = "model-best"
DIR_MODEL_LAST = "model-last"

# Number of dev examples used for the small validation that runs right after
# restoring a snapshot.
PROBE_SIZE = 16
# A restored checkpoint is considered suspicious if the probe score is this
# much (in absolute terms) worse than the score stored in the snapshot.
PROBE_TOLERANCE = 0.2


@dataclass
class TrainingProgress:
    """The point in a training run that a snapshot captures.

    ``step`` is the index of the last completed step (``-1`` means that no step
    has been completed yet) and ``steps_in_epoch`` is the number of batches
    produced in the current epoch. The object is shared by the training loop
    and the batch generator, which records the shuffle state and the position
    of the batch-size schedule as the run advances so that a resumed run sees
    exactly the same data as an uninterrupted one.
    """

    step: int = -1
    epoch: int = 0
    steps_in_epoch: int = 0
    results: List[Tuple[float, int]] = field(default_factory=list)
    losses: Dict[str, float] = field(default_factory=dict)
    words_seen: int = 0
    seconds: float = 0.0
    # The permutation used for the current epoch and the Python RNG state right
    # after it was drawn. Recording the permutation itself makes resuming
    # independent of the RNG implementation, and recording the post-shuffle
    # state keeps later epochs identical to an uninterrupted run.
    epoch_shuffle: Optional[List[int]] = None
    epoch_rng_after_shuffle: Optional[Any] = None
    # Position of the batch-size schedule at the start of the current epoch.
    batcher_position_at_epoch_start: int = 0

    def to_state(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "steps_in_epoch": self.steps_in_epoch,
            "results": [list(r) for r in self.results],
            "losses": dict(self.losses),
            "words_seen": self.words_seen,
            "seconds": self.seconds,
            "epoch_shuffle": self.epoch_shuffle,
            "epoch_rng_after_shuffle": ckpt.python_rng_state_to_json(
                self.epoch_rng_after_shuffle
            ),
            "batcher_position_at_epoch_start": self.batcher_position_at_epoch_start,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TrainingProgress":
        shuffle = state.get("epoch_shuffle")
        return cls(
            step=int(state.get("step", -1)),
            epoch=int(state.get("epoch", 0)),
            steps_in_epoch=int(state.get("steps_in_epoch", 0)),
            results=[
                (float(score), int(step)) for score, step in state.get("results", [])
            ],
            losses={k: float(v) for k, v in state.get("losses", {}).items()},
            words_seen=int(state.get("words_seen", 0)),
            seconds=float(state.get("seconds", 0.0)),
            epoch_shuffle=[int(i) for i in shuffle] if shuffle is not None else None,
            epoch_rng_after_shuffle=ckpt.python_rng_state_from_json(
                state.get("epoch_rng_after_shuffle")
            ),
            batcher_position_at_epoch_start=int(
                state.get("batcher_position_at_epoch_start", 0)
            ),
        )


class BatcherSchedule:
    """Wraps a batcher whose batch size comes from a schedule.

    The position of the schedule is tracked by counting how many values were
    consumed, and ``seek`` replays them so a resumed run continues with the
    same batch sizes. Calling the wrapper behaves exactly like calling the
    wrapped batcher.
    """

    def __init__(self, batcher: Callable, size_schedule: Any):
        self.batcher = batcher
        self.size_schedule = size_schedule
        self.position = 0
        self._iterator: Optional[Any] = None
        self._last_value: Any = None

    def _next_value(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.size_schedule)
        try:
            value = next(self._iterator)
        except StopIteration:
            # Schedules are unbounded; an exhausted one keeps its last value,
            # mirroring how the optimizer treats exhausted schedules.
            value = self._last_value if self._last_value is not None else 1
        self._last_value = value
        self.position += 1
        return value

    def seek(self, position: int) -> None:
        """Advance the schedule to ``position`` consumed values."""
        while self.position < position:
            self._next_value()

    def __call__(self, examples: Iterable[Any]) -> Iterable[Any]:
        kwargs = dict(self.batcher.keywords)
        kwargs["size"] = self._size_generator()
        return functools.partial(self.batcher.func, **kwargs)(examples)

    def _size_generator(self):
        while True:
            yield self._next_value()


def prepare_batcher(batcher: Any) -> Any:
    """Wrap a batcher if its batch size comes from a schedule."""
    if not isinstance(batcher, functools.partial):
        return batcher
    size_schedule = batcher.keywords.get("size")
    if size_schedule is None or isinstance(size_schedule, (int, float)):
        return batcher
    if not hasattr(size_schedule, "__iter__"):
        return batcher
    return BatcherSchedule(batcher, size_schedule)


def train(
    nlp: "Language",
    output_path: Optional[Path] = None,
    *,
    use_gpu: int = -1,
    stdout: IO = sys.stdout,
    stderr: IO = sys.stderr,
    resume: bool = False,
    resume_from: Optional[Union[str, Path]] = None,
) -> Tuple["Language", Optional[Path]]:
    """Train a pipeline.

    nlp (Language): The initialized nlp object with the full config.
    output_path (Optional[Path]): Optional output path to save trained model to.
    use_gpu (int): Whether to train on GPU. Make sure to call require_gpu
        before calling this function.
    stdout (file): A file-like object to write output messages. To disable
        printing, set to io.StringIO.
    stderr (file): A second file-like object to write output messages. To disable
        printing, set to io.StringIO.
    resume (bool): Continue the run from the newest complete training snapshot
        in ``output_path``. Existing model-best/model-last directories are kept.
    resume_from (Optional[Path]): Continue from a specific snapshot directory
        instead of the newest one in ``output_path``.

    RETURNS (tuple): The final nlp object and the path to the exported model.
    """
    # We use no_print here so we can respect the stdout/stderr options.
    msg = Printer(no_print=True)
    # Create iterator, which yields out info after each optimization step.
    config = nlp.config.interpolate()
    if config["training"]["seed"] is not None:
        fix_random_seed(config["training"]["seed"])
    allocator = config["training"]["gpu_allocator"]
    if use_gpu >= 0 and allocator:
        set_gpu_allocator(allocator)
    T = registry.resolve(config["training"], schema=ConfigSchemaTraining)  # type: ignore[arg-type]
    dot_names = [T["train_corpus"], T["dev_corpus"]]
    train_corpus, dev_corpus = resolve_dot_names(config, dot_names)
    optimizer = T["optimizer"]
    score_weights = T["score_weights"]
    batcher = T["batcher"]
    train_logger = T["logger"]
    before_to_disk = create_before_to_disk_callback(T["before_to_disk"])
    before_update = T["before_update"]
    save_every = int(T.get("save_every", 0))
    frequency_unit = T.get("frequency_unit", "step")
    max_keep = int(T.get("max_keep", 2))
    resume_mismatch = T.get("resume_mismatch", "fail")
    resume_validation = bool(T.get("resume_validation", True))
    is_resume = bool(resume or resume_from is not None)
    # Snapshots are written when periodic saving is enabled, and always for a
    # resumed run so that a second interruption can be handled as well.
    snapshots_enabled = save_every > 0 or is_resume

    # Helper function to save checkpoints. This is a closure for convenience,
    # to avoid passing in all the args all the time.
    def save_checkpoint(is_best):
        with nlp.use_params(optimizer.averages):
            before_to_disk(nlp).to_disk(output_path / DIR_MODEL_LAST)
        if is_best:
            # Avoid saving twice (saving will be more expensive than
            # the dir copy)
            if (output_path / DIR_MODEL_BEST).exists():
                shutil.rmtree(output_path / DIR_MODEL_BEST)
            shutil.copytree(output_path / DIR_MODEL_LAST, output_path / DIR_MODEL_BEST)

    # Components that shouldn't be updated during training
    frozen_components = T["frozen_components"]
    # Components that should set annotations on update
    annotating_components = T["annotating_components"]

    progress = TrainingProgress()
    lock: Optional[ckpt.OutputDirLock] = None
    snapshot: Optional[ckpt.Snapshot] = None
    run_id: Optional[str] = None
    if output_path is not None:
        # Only one process may write to an output directory at a time. The
        # lock is taken before anything is removed or overwritten so that a
        # second process fails without touching existing artifacts.
        lock = ckpt.OutputDirLock(output_path).acquire()
    try:
        if output_path is not None:
            if is_resume:
                snapshot, progress, run_id = load_resume_state(
                    nlp,
                    output_path,
                    optimizer,
                    config,
                    T,
                    resume_from=resume_from,
                    mismatch=resume_mismatch,
                    stdout=stdout,
                    msg=msg,
                )
                # The validation runs against the restored model, with the
                # restored RNG state, before any new updates take place.
                if resume_validation:
                    validate_resumed_model(
                        nlp, dev_corpus, progress, snapshot, T, stdout=stdout, msg=msg
                    )
            else:
                clean_output_dir(output_path)
                if snapshots_enabled:
                    run_id = ckpt.start_new_run(output_path)

        # The batch-size schedule only needs to be tracked when its position
        # can be stored in a snapshot. Without snapshots the batcher is used
        # unchanged.
        if snapshots_enabled:
            batcher = prepare_batcher(batcher)
        if is_resume and isinstance(batcher, BatcherSchedule):
            batcher.seek(progress.batcher_position_at_epoch_start)
    except BaseException:
        if lock is not None:
            lock.release()
        raise
    hooks = create_snapshot_hooks(
        output_path=output_path,
        nlp=nlp,
        optimizer=optimizer,
        progress=progress,
        config=config,
        T=T,
        run_id=run_id,
        save_every=save_every,
        frequency_unit=frequency_unit,
        max_keep=max_keep,
        enabled=snapshots_enabled,
        stdout=stdout,
        msg=msg,
    )

    training_step_iterator = train_while_improving(
        nlp,
        optimizer,
        create_train_batches(
            nlp,
            train_corpus,
            batcher,
            T["max_epochs"],
            progress=progress,
            resume=is_resume,
            on_epoch_end=hooks.on_epoch_end if hooks else None,
        ),
        create_evaluation_callback(nlp, dev_corpus, score_weights),
        dropout=T["dropout"],
        accumulate_gradient=T["accumulate_gradient"],
        patience=T["patience"],
        max_steps=T["max_steps"],
        eval_frequency=T["eval_frequency"],
        exclude=frozen_components,
        annotating_components=annotating_components,
        before_update=before_update,
        progress=progress,
        resume=is_resume,
        on_step_end=hooks.on_step_end if hooks else None,
    )
    stdout.write(msg.info(f"Pipeline: {nlp.pipe_names}") + "\n")
    if frozen_components:
        stdout.write(msg.info(f"Frozen components: {frozen_components}") + "\n")
    if annotating_components:
        stdout.write(
            msg.info(f"Set annotations on update for: {annotating_components}") + "\n"
        )
    if is_resume:
        stdout.write(
            msg.info(
                f"Resuming from snapshot: {snapshot.path if snapshot else None} "
                f"(step {progress.step}, epoch {progress.epoch})"
            )
            + "\n"
        )
    stdout.write(msg.info(f"Initial learn rate: {optimizer.learn_rate}") + "\n")
    with nlp.select_pipes(disable=frozen_components):
        log_step, finalize_logger = train_logger(nlp, stdout, stderr)
    completed = False
    wrote_steps = False
    try:
        for batch, info, is_best_checkpoint in training_step_iterator:
            wrote_steps = True
            if is_best_checkpoint is not None:
                with nlp.select_pipes(disable=frozen_components):
                    update_meta(T, nlp, info)
                if output_path is not None:
                    save_checkpoint(is_best_checkpoint)
                    info["output_path"] = str(output_path / DIR_MODEL_LAST)
            log_step(info if is_best_checkpoint is not None else None)
        completed = True
    except Exception as e:
        if output_path is not None:
            stdout.write(
                msg.warn(
                    f"Aborting and saving the final best model. "
                    f"Encountered exception: {repr(e)}"
                )
                + "\n"
            )
        raise e
    finally:
        finalize_logger()
        if output_path is not None:
            save_checkpoint(False)
            # On a clean finish (including an early stop) record the final
            # state so that a later `--resume` continues where this run ended.
            # After an exception the last periodic snapshot is the safest
            # fallback: a snapshot taken mid-step could describe a step that
            # was only partially applied. A resumed run that had already
            # reached its budget doesn't rewrite an identical snapshot.
            if snapshots_enabled and completed and wrote_steps:
                write_snapshot_safely(
                    output_path,
                    nlp,
                    optimizer,
                    progress,
                    config,
                    T,
                    run_id=run_id,
                    max_keep=max_keep,
                    stdout=stdout,
                    msg=msg,
                )
            lock.release()
    # This will only run if we did't hit an error
    if optimizer.averages:
        nlp.use_params(optimizer.averages)
    if output_path is not None:
        stdout.write(
            msg.good("Saved pipeline to output directory", output_path / DIR_MODEL_LAST)
            + "\n"
        )
        return (nlp, output_path / DIR_MODEL_LAST)
    else:
        return (nlp, None)


# --- Snapshot creation -----------------------------------------------------


class SnapshotHooks:
    """Decides when to write periodic snapshots."""

    def __init__(
        self,
        *,
        output_path: Path,
        nlp: "Language",
        optimizer: Optimizer,
        progress: TrainingProgress,
        config: Any,
        T: Dict[str, Any],
        run_id: Optional[str],
        save_every: int,
        frequency_unit: str,
        max_keep: int,
        stdout: IO,
        msg: Printer,
    ):
        self.output_path = output_path
        self.nlp = nlp
        self.optimizer = optimizer
        self.config = config
        self.T = T
        self.run_id = run_id
        self.save_every = save_every
        self.frequency_unit = frequency_unit
        self.max_keep = max_keep
        self.stdout = stdout
        self.msg = msg
        self.identity = ckpt.compute_run_identity(nlp, config, T)

    def write(self, progress: TrainingProgress) -> None:
        write_snapshot_safely(
            self.output_path,
            self.nlp,
            self.optimizer,
            progress,
            self.config,
            self.T,
            run_id=self.run_id,
            max_keep=self.max_keep,
            stdout=self.stdout,
            msg=self.msg,
            identity=self.identity,
        )

    def on_step_end(self, progress: TrainingProgress) -> None:
        if self.save_every <= 0 or self.frequency_unit != "step":
            return
        if progress.step <= 0:
            return
        if progress.step % self.save_every == 0:
            self.write(progress)

    def on_epoch_end(self, progress: TrainingProgress) -> None:
        if self.save_every <= 0 or self.frequency_unit != "epoch":
            return
        # ``progress`` already describes the start of the next epoch.
        completed_epochs = progress.epoch
        if completed_epochs > 0 and completed_epochs % self.save_every == 0:
            self.write(progress)


def create_snapshot_hooks(
    *,
    output_path: Optional[Path],
    nlp: "Language",
    optimizer: Optimizer,
    progress: TrainingProgress,
    config: Any,
    T: Dict[str, Any],
    run_id: Optional[str],
    save_every: int,
    frequency_unit: str,
    max_keep: int,
    enabled: bool,
    stdout: IO,
    msg: Printer,
) -> Optional[SnapshotHooks]:
    if not enabled or output_path is None:
        return None
    return SnapshotHooks(
        output_path=output_path,
        nlp=nlp,
        optimizer=optimizer,
        progress=progress,
        config=config,
        T=T,
        run_id=run_id,
        save_every=save_every,
        frequency_unit=frequency_unit,
        max_keep=max_keep,
        stdout=stdout,
        msg=msg,
    )


def write_snapshot_safely(
    output_path: Path,
    nlp: "Language",
    optimizer: Optimizer,
    progress: TrainingProgress,
    config: Any,
    T: Dict[str, Any],
    *,
    run_id: Optional[str],
    max_keep: int,
    stdout: IO,
    msg: Printer,
    identity: Optional[Dict[str, Any]] = None,
) -> Optional[ckpt.Snapshot]:
    """Write a snapshot and report failures without interrupting training.

    A failed write (disk full, permissions, ...) must not take the run down:
    the previous complete snapshot is left in place and the run continues.
    """
    if identity is None:
        identity = ckpt.compute_run_identity(nlp, config, T)
    meta = {
        **identity,
        "run_id": run_id,
        "step": progress.step,
        "epoch": progress.epoch,
        "environment": ckpt.describe_environment(),
    }
    # Avoid writing an identical snapshot twice (e.g. a periodic snapshot at
    # the last step followed by the final snapshot). A corrupt newer snapshot
    # must not prevent the run from writing a good one.
    try:
        existing = ckpt.find_latest_snapshot(output_path, run_id=run_id)
    except ckpt.SnapshotVersionError:
        existing = None
    if existing is not None:
        try:
            existing_meta = existing.read_meta()
        except ckpt.SnapshotError:
            existing_meta = {}
        if existing_meta.get("step") == progress.step:
            return existing
    try:
        snapshot = ckpt.write_snapshot(
            output_path,
            nlp=nlp,
            optimizer=optimizer,
            state=progress.to_state(),
            meta=meta,
            run_id=run_id,
            keep=max_keep,
        )
    except Exception as e:
        stdout.write(
            msg.warn(
                Warnings.W130.format(
                    path=ckpt.get_snapshots_dir(output_path), error=repr(e)
                )
            )
            + "\n"
        )
        return None
    logger.debug("Wrote training snapshot: %s", snapshot.path)
    stdout.write(
        msg.info(f"Saved training snapshot at step {progress.step}: {snapshot.path}")
        + "\n"
    )
    return snapshot


# --- Resume ----------------------------------------------------------------


def load_resume_state(
    nlp: "Language",
    output_path: Path,
    optimizer: Optimizer,
    config: Any,
    T: Dict[str, Any],
    *,
    resume_from: Optional[Union[str, Path]],
    mismatch: str,
    stdout: IO,
    msg: Printer,
) -> Tuple[ckpt.Snapshot, TrainingProgress, Optional[str]]:
    """Find, validate and load a snapshot into ``nlp``.

    Loading is transactional with respect to the on-disk artifacts: the
    snapshot is validated (including checksums and identity) before any
    weights are copied into ``nlp``. A failed resume never modifies the
    snapshot directory or model-best/model-last.
    """
    run_id = None if resume_from is not None else ckpt.read_current_run_id(output_path)
    snapshot = find_resume_snapshot(output_path, resume_from, run_id)
    meta = snapshot.read_meta()
    run_id = meta.get("run_id", run_id)
    current_identity = ckpt.compute_run_identity(nlp, config, T)
    stored_identity = {
        key: meta.get(key) for key in ckpt.IDENTITY_LABELS if key in meta
    }
    diffs = ckpt.diff_identity(stored_identity, current_identity)
    if diffs:
        diff_text = "; ".join(diffs)
        if mismatch == "fail":
            raise ckpt.ResumeMismatchError(Errors.E1059.format(diff=diff_text))
        elif mismatch == "warn":
            stdout.write(msg.warn(Warnings.W128.format(diff=diff_text)) + "\n")
        # "none" skips the check entirely
    stored_env = meta.get("environment") or {}
    env_diffs = ckpt.diff_environment(stored_env, ckpt.describe_environment())
    if env_diffs:
        # Moving between CPU and GPUs is allowed; report it for the record.
        stdout.write(
            msg.info(
                Warnings.W134.format(
                    stored=stored_env,
                    current=ckpt.describe_environment(),
                )
            )
            + "\n"
        )
    state = snapshot.read_state()
    progress = TrainingProgress.from_state(state)
    try:
        _restore_into(nlp, optimizer, snapshot, state)
    except ckpt.ResumeMismatchError:
        raise
    except Exception as e:
        # The snapshot's weights couldn't be loaded into the current pipeline
        # (e.g. a changed architecture). Report it as a mismatch so the user
        # gets the same actionable message, with the underlying error.
        raise ckpt.ResumeMismatchError(
            Errors.E1059.format(
                diff=f"the snapshot model state can't be loaded into the current "
                f"pipeline ({type(e).__name__}: {e})"
            )
        ) from e
    ckpt.restore_rng_state(snapshot.read_rng())
    stdout.write(
        msg.info(
            f"Restored training state from {snapshot.path} "
            f"(step {progress.step}, epoch {progress.epoch})"
        )
        + "\n"
    )
    return snapshot, progress, run_id


def find_resume_snapshot(
    output_path: Path,
    resume_from: Optional[Union[str, Path]],
    run_id: Optional[str],
) -> ckpt.Snapshot:
    if resume_from is not None:
        path = Path(resume_from)
        if not path.is_dir():
            raise ckpt.SnapshotError(
                Errors.E1062.format(path=path, error="not a directory")
            )
        snapshot = ckpt.Snapshot(path)
        snapshot.validate()
        return snapshot
    snapshot = ckpt.find_latest_snapshot(output_path, run_id=run_id)
    if snapshot is None:
        raise ckpt.SnapshotError(Errors.E1058.format(path=output_path))
    return snapshot


def _restore_into(
    nlp: "Language",
    optimizer: Optimizer,
    snapshot: ckpt.Snapshot,
    state: Dict[str, Any],
) -> None:
    """Copy the snapshot's weights and optimizer state into ``nlp``."""
    current_ids = ckpt.collect_model_ids(nlp)
    stored_ids = state.get("model_ids", [])
    if stored_ids and len(stored_ids) != len(current_ids):
        raise ckpt.ResumeMismatchError(
            Errors.E1059.format(
                diff=(
                    "model structure: the snapshot has "
                    f"{len(stored_ids)} model nodes, the current pipeline has "
                    f"{len(current_ids)}"
                )
            )
        )
    # Load the weights in place. The snapshot's config.cfg and meta.json are
    # skipped: the current config and meta are authoritative for the
    # continuation, and the config was already compared above.
    nlp.from_disk(snapshot.model_path, exclude=["vocab", "config", "meta"])
    # `from_disk` records the snapshot directory as the model path, which would
    # make later component-sourcing operations resolve against the snapshot.
    nlp._path = None
    id_map = ckpt.model_id_map(stored_ids, current_ids) if stored_ids else {}
    shim_ids = ckpt.model_id_map(
        state.get("shim_ids", []), ckpt.collect_shim_ids(nlp)
    )
    id_map.update(shim_ids)
    ckpt.restore_optimizer(optimizer, state.get("optimizer", {}), id_map)
    # The schedules are re-created from the config; advance them to the value
    # that was in effect when the snapshot was taken so the next step uses the
    # same learning rate / momentum as an uninterrupted run.
    ckpt.restore_schedule_history(
        optimizer, state.get("optimizer", {}), progress_steps(state)
    )


def progress_steps(state: Dict[str, Any]) -> int:
    return int(state.get("step", -1)) + 1


def validate_resumed_model(
    nlp: "Language",
    dev_corpus: Callable,
    progress: TrainingProgress,
    snapshot: ckpt.Snapshot,
    T: Dict[str, Any],
    *,
    stdout: IO,
    msg: Printer,
) -> None:
    """Run a small evaluation to make sure the restored state is coherent.

    The stored score dates from the last evaluation, so a restored model should
    score roughly the same. A large drop usually means that the snapshot was
    restored into the wrong pipeline, or that the data changed.
    """
    if not progress.results:
        return
    stored_score = progress.results[-1][0]
    # The probe must not disturb the state that the run is about to continue
    # from: evaluation may update counters and consumes RNG values.
    rng_state = random.getstate()
    numpy_rng_state = numpy.random.get_state()
    try:
        examples = list(itertools.islice(dev_corpus(nlp), PROBE_SIZE))
        if not examples:
            return
        scores = nlp.evaluate(examples)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Could not validate the restored model: %s", repr(e))
        return
    finally:
        random.setstate(rng_state)
        numpy.random.set_state(numpy_rng_state)
    weights = {
        key: value
        for key, value in T.get("score_weights", {}).items()
        if value is not None and key in scores and isinstance(scores[key], (int, float))
    }
    if not weights:
        return
    probe_score = sum(scores[key] * weight for key, weight in weights.items())
    if probe_score < stored_score - PROBE_TOLERANCE:
        stdout.write(
            msg.warn(
                Warnings.W132.format(
                    path=snapshot.path,
                    step=progress.step,
                    epoch=progress.epoch,
                    resumed=round(probe_score, 4),
                    stored=round(stored_score, 4),
                )
            )
            + "\n"
        )


def train_while_improving(
    nlp: "Language",
    optimizer: Optimizer,
    train_data,
    evaluate,
    *,
    dropout: float,
    eval_frequency: int,
    accumulate_gradient: int,
    patience: int,
    max_steps: int,
    exclude: List[str],
    annotating_components: List[str],
    before_update: Optional[Callable[["Language", Dict[str, Any]], None]],
    progress: Optional[TrainingProgress] = None,
    resume: bool = False,
    on_step_end: Optional[Callable[[TrainingProgress], None]] = None,
):
    """Train until an evaluation stops improving. Works as a generator,
    with each iteration yielding a tuple `(batch, info, is_best_checkpoint)`,
    where info is a dict, and is_best_checkpoint is in [True, False, None] --
    None indicating that the iteration was not evaluated as a checkpoint.
    The evaluation is conducted by calling the evaluate callback.

    Positional arguments:
        nlp: The spaCy pipeline to evaluate.
        optimizer: The optimizer callable.
        train_data (Iterable[Batch]): A generator of batches, with the training
            data. Each batch should be a Sized[Tuple[Input, Annot]]. The training
            data iterable needs to take care of iterating over the epochs and
            shuffling.
        evaluate (Callable[[], Tuple[float, Any]]): A callback to perform evaluation.
            The callback should take no arguments and return a tuple
            `(main_score, other_scores)`. The main_score should be a float where
            higher is better. other_scores can be any object.

    Every iteration, the function yields out a tuple with:

    * batch: A list of Example objects.
    * info: A dict with various information about the last update (see below).
    * is_best_checkpoint: A value in None, False, True, indicating whether this
        was the best evaluation so far. You should use this to save the model
        checkpoints during training. If None, evaluation was not conducted on
        that iteration. False means evaluation was conducted, but a previous
        evaluation was better.

    The info dict provides the following information:

        epoch (int): How many passes over the data have been completed.
        step (int): How many steps have been completed.
        score (float): The main score from the last evaluation.
        other_scores: : The other scores from the last evaluation.
        losses: The accumulated losses throughout training.
        checkpoints: A list of previous results, where each result is a
            (score, step, epoch) tuple.

    When ``resume`` is True the counters and the evaluation history stored in
    ``progress`` are used instead of starting from zero, and the dropout
    schedule is advanced to the position reached before the interruption.
    ``on_step_end`` is called after every step (before the next batch is
    fetched) and can be used to snapshot the state.
    """
    if isinstance(dropout, float):
        dropouts = constant(dropout)
    else:
        dropouts = dropout
    if progress is None:
        progress = TrainingProgress()
    if resume and max_steps and progress.step >= max_steps:
        # The run had already reached its step budget when the snapshot was
        # written, so there is nothing left to train.
        return
    if resume:
        results = list(progress.results)
        losses = dict(progress.losses)
        words_seen = progress.words_seen
        step = progress.step
        start_time = timer() - progress.seconds
        # Schedules are restored from the config, so skip the values that have
        # already been consumed.
        for _ in range(max(0, progress.step + 1)):
            next(dropouts)  # type: ignore
    else:
        results = []
        losses = {}
        words_seen = 0
        step = -1
        start_time = timer()
    progress.results = results
    progress.losses = losses
    progress.words_seen = words_seen
    progress.step = step
    for epoch, batch in train_data:
        step += 1
        if before_update:
            before_update_args = {"step": step, "epoch": epoch}
            before_update(nlp, before_update_args)
        dropout = next(dropouts)  # type: ignore
        for subbatch in subdivide_batch(batch, accumulate_gradient):
            nlp.update(
                subbatch,
                drop=dropout,
                losses=losses,
                sgd=False,  # type: ignore[arg-type]
                exclude=exclude,
                annotates=annotating_components,
            )
        # TODO: refactor this so we don't have to run it separately in here
        for name, proc in nlp.pipeline:
            if (
                name not in exclude
                and hasattr(proc, "is_trainable")
                and proc.is_trainable
                and proc.model not in (True, False, None)  # type: ignore[attr-defined]
            ):
                proc.finish_update(optimizer)  # type: ignore[attr-defined]
        optimizer.step_schedules()
        if not (step % eval_frequency):
            if optimizer.averages:
                with nlp.use_params(optimizer.averages):
                    score, other_scores = evaluate()
            else:
                score, other_scores = evaluate()
            results.append((score, step))
            is_best_checkpoint = score == max(results)[0]
        else:
            score, other_scores = (None, None)
            is_best_checkpoint = None
        words_seen += sum(len(eg) for eg in batch)
        info = {
            "epoch": epoch,
            "step": step,
            "score": score,
            "other_scores": other_scores,
            "losses": losses,
            "checkpoints": results,
            "seconds": int(timer() - start_time),
            "words": words_seen,
        }
        progress.step = step
        progress.epoch = epoch
        progress.losses = losses
        progress.words_seen = words_seen
        progress.seconds = timer() - start_time
        yield batch, info, is_best_checkpoint
        if is_best_checkpoint is not None:
            losses = {}
            progress.losses = losses
        if on_step_end is not None:
            on_step_end(progress)
        # Stop if no improvement in `patience` updates (if specified)
        # Negate step value so that the earliest best step is chosen for the
        # same score, i.e. (1.0, 100) is chosen over (1.0, 200)
        best_result = max((r_score, -r_step) for r_score, r_step in results)
        best_step = -best_result[1]
        if patience and (step - best_step) >= patience:
            break
        # Stop if we've exhausted our max steps (if specified)
        if max_steps and step >= max_steps:
            break


def subdivide_batch(batch, accumulate_gradient):
    batch = list(batch)
    batch.sort(key=lambda eg: len(eg.predicted))
    sub_len = len(batch) // accumulate_gradient
    start = 0
    for i in range(accumulate_gradient):
        subbatch = batch[start : start + sub_len]
        if subbatch:
            yield subbatch
        start += len(subbatch)
    subbatch = batch[start:]
    if subbatch:
        yield subbatch


def create_evaluation_callback(
    nlp: "Language", dev_corpus: Callable, weights: Dict[str, float]
) -> Callable[[], Tuple[float, Dict[str, float]]]:
    weights = {key: value for key, value in weights.items() if value is not None}

    def evaluate() -> Tuple[float, Dict[str, float]]:
        nonlocal weights
        try:
            scores = nlp.evaluate(dev_corpus(nlp))
        except KeyError as e:
            raise KeyError(Errors.E900.format(pipeline=nlp.pipe_names)) from e
        # Calculate a weighted sum based on score_weights for the main score.
        # We can only consider scores that are ints/floats, not dicts like
        # entity scores per type etc.
        scores = {key: value for key, value in scores.items() if value is not None}
        weights = {key: value for key, value in weights.items() if key in scores}
        for key, value in scores.items():
            if key in weights and not isinstance(value, (int, float)):
                raise ValueError(Errors.E915.format(name=key, score_type=type(value)))
        try:
            weighted_score = sum(
                scores.get(s, 0.0) * weights.get(s, 0.0) for s in weights
            )
        except KeyError as e:
            keys = list(scores.keys())
            err = Errors.E983.format(dict="score_weights", key=str(e), keys=keys)
            raise KeyError(err) from None
        return weighted_score, scores

    return evaluate


def create_train_batches(
    nlp: "Language",
    corpus: Callable[["Language"], Iterable[Example]],
    batcher: Callable[[Iterable[Example]], Iterable[Example]],
    max_epochs: int,
    *,
    progress: Optional[TrainingProgress] = None,
    resume: bool = False,
    on_epoch_end: Optional[Callable[[TrainingProgress], None]] = None,
):
    """Yield ``(epoch, batch)`` pairs for the training loop.

    With a finite number of epochs the corpus is loaded once and shuffled every
    epoch. With ``max_epochs < 0`` the corpus is streamed and never loaded into
    memory.

    Resuming replays the current epoch from its start and drops the batches
    that were already consumed, so a streaming corpus is re-read sequentially
    instead of being loaded. ``progress`` carries the epoch, the RNG state used
    for the shuffle, the index of the current batch and the position of the
    batch-size schedule.
    """
    if progress is None:
        progress = TrainingProgress()
    if max_epochs >= 0:
        base_examples = list(corpus(nlp))  # type: List[Example]
        if not base_examples:
            # Raise error if no data
            raise ValueError(Errors.E986)
    epoch = progress.epoch if resume else 0
    progress.epoch = epoch
    # Only the first epoch after resuming reuses the stored permutation; all
    # later epochs are shuffled exactly as in an uninterrupted run.
    resumed_epoch = progress.epoch if resume else None
    replay_pending = resume
    while max_epochs < 1 or epoch != max_epochs:
        if max_epochs >= 0:
            if replay_pending and epoch == resumed_epoch:
                if progress.epoch_shuffle is not None:
                    order = progress.epoch_shuffle
                    examples = [base_examples[i] for i in order]
                else:
                    # The snapshot was taken at an epoch boundary, before the
                    # next shuffle. The live RNG was restored to that point, so
                    # drawing the permutation here reproduces the original run.
                    order = list(range(len(base_examples)))
                    random.shuffle(order)
                    examples = [base_examples[i] for i in order]
                    progress.epoch_shuffle = order
                    progress.epoch_rng_after_shuffle = random.getstate()
                # The live RNG already holds the post-shuffle state restored
                # from the snapshot, so later epochs continue where the
                # interrupted run left off.
            else:
                # The permutation is always drawn from the original corpus
                # order, never from the previous epoch's ordering, so it can be
                # replayed from a snapshot regardless of how many epochs have
                # been completed.
                order = list(range(len(base_examples)))
                random.shuffle(order)
                examples = [base_examples[i] for i in order]
                progress.epoch_shuffle = order
                progress.epoch_rng_after_shuffle = random.getstate()
        else:
            examples = corpus(nlp)
        skip = progress.steps_in_epoch if replay_pending else 0
        replay_pending = False
        produced = 0
        for batch in batcher(examples):
            produced += 1
            if produced <= skip:
                # This batch was already consumed before the interruption.
                continue
            progress.steps_in_epoch = produced
            progress.epoch = epoch
            yield epoch, batch
        # The epoch is exhausted. Record the state at the start of the next
        # epoch *before* the next iteration shuffles the data.
        epoch += 1
        progress.epoch = epoch
        progress.steps_in_epoch = 0
        progress.epoch_shuffle = None
        progress.epoch_rng_after_shuffle = None
        if isinstance(batcher, BatcherSchedule):
            progress.batcher_position_at_epoch_start = batcher.position
        if on_epoch_end is not None:
            on_epoch_end(progress)


def update_meta(
    training: Union[Dict[str, Any], Config], nlp: "Language", info: Dict[str, Any]
) -> None:
    nlp.meta["performance"] = {}
    for metric in training["score_weights"]:
        if metric is not None:
            nlp.meta["performance"][metric] = info["other_scores"].get(metric, 0.0)
    for pipe_name in nlp.pipe_names:
        if pipe_name in info["losses"]:
            nlp.meta["performance"][f"{pipe_name}_loss"] = info["losses"][pipe_name]


def create_before_to_disk_callback(
    callback: Optional[Callable[["Language"], "Language"]],
) -> Callable[["Language"], "Language"]:
    from ..language import Language  # noqa: F811

    def before_to_disk(nlp: Language) -> Language:
        if not callback:
            return nlp
        modified_nlp = callback(nlp)
        if not isinstance(modified_nlp, Language):
            err = Errors.E914.format(name="before_to_disk", value=type(modified_nlp))
            raise ValueError(err)
        return modified_nlp

    return before_to_disk


def clean_output_dir(path: Optional[Path]) -> None:
    """Remove an existing output directory. Typically used to ensure that that
    a directory like model-best and its contents aren't just being overwritten
    by nlp.to_disk, which could preserve existing subdirectories (e.g.
    components that don't exist anymore).

    Training snapshots are never removed here: a fresh run receives a new run
    id, so old snapshots can't be picked up by mistake but remain on disk as a
    fallback.
    """
    if path is not None and path.exists():
        for subdir in [path / DIR_MODEL_BEST, path / DIR_MODEL_LAST]:
            if subdir.exists():
                try:
                    shutil.rmtree(str(subdir))
                    logger.debug("Removed existing output directory: %s", subdir)
                except Exception as e:
                    raise IOError(Errors.E901.format(path=path)) from e
