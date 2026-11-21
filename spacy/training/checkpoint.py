"""Periodic, resumable training snapshots.

The functions in this module implement the on-disk format and the bookkeeping
needed to interrupt a training run and continue it later without repeating
completed work. Snapshots are written below ``<output>/training-state`` so that
they never overwrite ``model-best`` or ``model-last``.

A snapshot contains everything that is needed to continue a run:

* the current model state (``model/``),
* the optimizer accumulators (moments and moving averages),
* the number of completed steps, the position inside the current epoch, the
  state of the shuffling process and the position of the batch-size schedule,
* the Python and NumPy random number generator states, and
* the evaluation history, the accumulated losses and the probe score that is
  used to validate a restored checkpoint.

Snapshots are written atomically: the data is first written to a temporary
directory which is renamed into place once it is complete. A ``COMPLETE``
marker and a manifest of checksums make it possible to detect a snapshot that
was interrupted mid-write and fall back to the previous complete one.

The module also provides the lock that prevents two training processes from
writing to the same output directory at the same time.
"""

import hashlib
import json
import os
import random
import shutil
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy
import srsly
from thinc.api import Optimizer, get_current_ops

from ..errors import Errors, Warnings
from ..util import logger

# Bump this if the format of the snapshot changes in an incompatible way.
SNAPSHOT_FORMAT_VERSION = 1
SUPPORTED_SNAPSHOT_VERSIONS = (1,)

DIR_TRAINING_STATE = "training-state"
DIR_SNAPSHOTS = "snapshots"
DIR_MODEL = "model"
FILE_META = "meta.json"
FILE_STATE = "state.msg"
FILE_RNG = "rng.msg"
FILE_MANIFEST = "manifest.json"
FILE_RUN_ID = "run-id"
COMPLETE_MARKER = "COMPLETE"
LOCK_FILE = ".spacy-train.lock"
TMP_PREFIX = ".tmp-"


class SnapshotError(Exception):
    """Raised when a snapshot can't be read or validated."""


class SnapshotVersionError(SnapshotError):
    """Raised when a snapshot uses an unsupported format version."""


class ResumeMismatchError(Exception):
    """Raised when a snapshot doesn't belong to the current training run."""


class OutputDirLockedError(Exception):
    """Raised when the output directory is already locked by another process."""


def get_state_dir(output_path: Path) -> Path:
    return Path(output_path) / DIR_TRAINING_STATE


def get_snapshots_dir(output_path: Path) -> Path:
    return get_state_dir(output_path) / DIR_SNAPSHOTS


def _snapshot_name(step: int, epoch: int, run_id: Optional[str] = None) -> str:
    # The step is zero-padded so that lexical sorting matches numeric sorting.
    suffix = f"-{run_id}" if run_id else ""
    return f"snapshot-{step:09d}-e{epoch:06d}{suffix}"


def _unique_snapshot_path(
    snapshots_dir: Path, step: int, epoch: int, run_id: Optional[str] = None
) -> Path:
    base = _snapshot_name(step, epoch, run_id)
    path = snapshots_dir / base
    suffix = 1
    while path.exists():
        path = snapshots_dir / f"{base}-{suffix:02d}"
        suffix += 1
    return path


def start_new_run(output_path: Path) -> str:
    """Record a new run id in the output directory and return it."""
    state_dir = get_state_dir(output_path)
    state_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    existing = read_current_run_id(output_path)
    if existing == run_id:
        run_id = f"{run_id}-{time.time_ns() % 1_000_000:06d}"
    _write_json_atomic(state_dir / FILE_RUN_ID, {"run_id": run_id})
    return run_id


def read_current_run_id(output_path: Path) -> Optional[str]:
    path = get_state_dir(output_path) / FILE_RUN_ID
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("run_id")
    except (OSError, ValueError):
        return None


def _file_checksum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file_:
        for chunk in iter(lambda: file_.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json_atomic(path: Path, data: Any) -> None:
    tmp_path = path.with_name(path.name + TMP_PREFIX)
    with tmp_path.open("w", encoding="utf-8") as file_:
        json.dump(data, file_, ensure_ascii=False, indent=2, sort_keys=True)
        file_.flush()
        os.fsync(file_.fileno())
    os.replace(tmp_path, path)


def _write_msgpack_atomic(path: Path, data: Any) -> None:
    tmp_path = path.with_name(path.name + TMP_PREFIX)
    with tmp_path.open("wb") as file_:
        file_.write(srsly.msgpack_dumps(data))
        file_.flush()
        os.fsync(file_.fileno())
    os.replace(tmp_path, path)


def _rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:  # pragma: no cover - not supported on all platforms
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(fd)


class OutputDirLock:
    """A simple, cross-process lock for an output directory.

    The lock is implemented with an exclusive file creation (``O_EXCL``), which
    works on local filesystems. The lock records the PID and hostname of the
    owner so that the error message can tell the user what is going on. Locks
    left behind by a process that no longer exists on this host are considered
    stale and taken over.
    """

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.lock_path = self.output_path / LOCK_FILE
        self._acquired = False

    def acquire(self) -> "OutputDirLock":
        self.output_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "created": time.time(),
        }
        for _ in range(2):
            try:
                fd = os.open(
                    str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
                )
            except FileExistsError:
                if self._is_stale():
                    logger.warning("Removing stale training lock: %s", self.lock_path)
                    try:
                        self.lock_path.unlink()
                    except OSError:
                        pass
                    continue
                raise OutputDirLockedError(
                    Errors.E1061.format(
                        pid=self._read_pid(),
                        path=self.output_path,
                        lock_path=self.lock_path,
                    )
                ) from None
            else:
                with os.fdopen(fd, "w", encoding="utf-8") as file_:
                    json.dump(payload, file_)
                    file_.flush()
                    os.fsync(file_.fileno())
                self._acquired = True
                return self
        # If we get here, the lock disappeared and reappeared twice in a row,
        # which means that another process is racing us for it.
        raise OutputDirLockedError(
            Errors.E1061.format(
                pid=self._read_pid(),
                path=self.output_path,
                lock_path=self.lock_path,
            )
        )

    def _is_stale(self) -> bool:
        try:
            data = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # An unreadable lock is most likely a half-written file from a
            # process that died right after opening it.
            return True
        if data.get("host") != socket.gethostname():
            # We can't reliably check processes on other hosts, so assume the
            # lock is still held.
            return False
        pid = data.get("pid")
        if not isinstance(pid, int):
            return True
        if pid == os.getpid():
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except (PermissionError, OSError):
            return False
        return False

    def _read_pid(self) -> Optional[int]:
        try:
            return json.loads(self.lock_path.read_text(encoding="utf-8")).get("pid")
        except (OSError, ValueError):
            return None

    def release(self) -> None:
        if self._acquired:
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass
            self._acquired = False

    def __enter__(self) -> "OutputDirLock":
        return self.acquire()

    def __exit__(self, *args: Any) -> None:
        self.release()


# --- Optimizer state -------------------------------------------------------


def _key_to_str(key: Tuple[int, str]) -> str:
    return f"{key[0]}:{key[1]}"


def _key_from_str(key: str) -> Tuple[int, str]:
    node_id, name = key.split(":", 1)
    return int(node_id), name


def serialize_optimizer(optimizer: Optimizer) -> Dict[str, Any]:
    """Serialize the optimizer accumulators.

    The momentum/average arrays are keyed by ``(model_id, param_name)``. Model
    ids are process-local counters assigned when models are created, so they
    are stored together with the model id list and remapped when the snapshot
    is loaded into a freshly initialized pipeline (see
    :func:`restore_optimizer`).

    Samples from every active schedule are stored as well. The schedule
    generators themselves can't be pickled, but recording the values they have
    produced (including the one currently in effect) is enough to restore the
    exact learning-rate/beta trajectory. The sample count is bounded so a very
    long run doesn't store an unbounded list; once exceeded, the schedule is
    treated as converged and only its current value is kept.
    """
    ops = get_current_ops()

    def _pack(mapping: Optional[Dict[Tuple[int, str], Any]]) -> Dict[str, Any]:
        packed: Dict[str, Any] = {}
        if not mapping:
            return packed
        for key, value in mapping.items():
            packed[_key_to_str(key)] = ops.to_numpy(value)
        return packed

    schedules: Dict[str, Any] = {}
    for name in optimizer.schedules:
        schedules[name] = {"current": getattr(optimizer, name, None)}
    return {
        "nr_update": {_key_to_str(k): int(v) for k, v in optimizer.nr_update.items()},
        "last_seen": {_key_to_str(k): int(v) for k, v in optimizer.last_seen.items()},
        "mom1": _pack(optimizer.mom1),
        "mom2": _pack(optimizer.mom2),
        "averages": _pack(optimizer.averages),
        "attrs": {
            name: getattr(optimizer, name)
            for name in ("learn_rate", "b1", "b2", "eps", "L2", "grad_clip")
        },
        "schedules": schedules,
    }


def restore_optimizer(
    optimizer: Optimizer, state: Dict[str, Any], id_map: Dict[int, int]
) -> None:
    """Restore optimizer accumulators into a freshly created optimizer."""
    ops = get_current_ops()

    def _unpacked_array(value: Any) -> Any:
        # msgpack decodes arrays as read-only views; the optimizer updates the
        # accumulators in place, so they must be writable.
        array = numpy.array(numpy.asarray(value), copy=True)
        return ops.asarray(array)

    def _unpack(mapping: Dict[str, Any]) -> Dict[Tuple[int, str], Any]:
        unpacked: Dict[Tuple[int, str], Any] = {}
        for key, value in mapping.items():
            node_id, name = _key_from_str(key)
            if node_id not in id_map:
                # The parameter belongs to a model that no longer exists.
                continue
            unpacked[(id_map[node_id], name)] = _unpacked_array(value)
        return unpacked

    optimizer.mom1 = _unpack(state.get("mom1", {}))
    optimizer.mom2 = _unpack(state.get("mom2", {}))
    if state.get("averages") is not None and optimizer.averages is not None:
        optimizer.averages = _unpack(state["averages"])

    for target, key in (
        (optimizer.nr_update, "nr_update"),
        (optimizer.last_seen, "last_seen"),
    ):
        for raw_key, value in state.get(key, {}).items():
            node_id, name = _key_from_str(raw_key)
            if node_id in id_map:
                target[(id_map[node_id], name)] = int(value)

    for name, value in state.get("attrs", {}).items():
        if name in optimizer.schedules:
            continue
        try:
            setattr(optimizer, name, value)
        except AttributeError:  # pragma: no cover
            pass


def restore_schedule_history(
    optimizer: Optimizer, state: Dict[str, Any], steps_done: int
) -> None:
    """Advance config-created schedules to their state after ``steps_done`` steps.

    thinc's ``Optimizer`` advances its schedules at the end of every step
    (``step_schedules``), so on resume each schedule has to be advanced the
    same number of times. The value stored in the snapshot is then applied
    explicitly: schedules may stop producing new values or be custom, and the
    stored value is the one that was actually in effect.
    """
    for name, schedule in optimizer.schedules.items():
        for _ in range(max(0, steps_done)):
            try:
                next(schedule)
            except StopIteration:
                break
        schedule_state = state.get("schedules", {}).get(name)
        if schedule_state and schedule_state.get("current") is not None:
            setattr(optimizer, name, schedule_state["current"])


def make_rng_state() -> Dict[str, Any]:
    return {
        "python": python_rng_state_to_json(random.getstate()),
        "numpy": _numpy_rng_state_to_json(numpy.random.get_state()),
    }


def _numpy_rng_state_to_json(state: Any) -> Any:
    name, keys, pos, has_gauss, cached_gaussian = state
    return (
        str(name),
        numpy.asarray(keys, dtype=numpy.uint32),
        int(pos),
        int(has_gauss),
        float(cached_gaussian),
    )


def _numpy_rng_state_from_json(state: Any) -> Optional[Any]:
    if state is None:
        return None
    name, keys, pos, has_gauss, cached_gaussian = state
    return (
        str(name),
        numpy.asarray(keys, dtype=numpy.uint32),
        int(pos),
        int(has_gauss),
        float(cached_gaussian),
    )


def python_rng_state_to_json(state: Any) -> Any:
    """Convert a Python RNG state to a msgpack-friendly structure.

    ``random.getstate()`` returns a tuple containing a 625-element tuple of
    ints; msgpack decodes tuples as lists, so the shape is restored explicitly
    on load.
    """
    if state is None:
        return None
    version, internal, gauss_next = state
    return (int(version), list(internal), gauss_next)


def python_rng_state_from_json(state: Any) -> Optional[Any]:
    if state is None:
        return None
    if isinstance(state, (list, tuple)) and len(state) == 3:
        version, internal, gauss_next = state
        return (int(version), tuple(int(i) for i in internal), gauss_next)
    return None


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    python_state = python_rng_state_from_json(state.get("python"))
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = _numpy_rng_state_from_json(state.get("numpy"))
    if numpy_state is not None:
        numpy.random.set_state(numpy_state)


# --- Pipeline identity -----------------------------------------------------


def collect_model_ids(nlp: Any) -> List[int]:
    """Collect the node ids of all component models in a stable walk order."""
    ids: List[int] = []
    for _name, proc in nlp._components:
        model = getattr(proc, "model", None)
        if model is None or isinstance(model, bool):
            continue
        ids.extend(node.id for node in model.walk())
    return ids


def collect_shim_ids(nlp: Any) -> List[int]:
    """Collect the ids of the shims wrapping external models (e.g. torch)."""
    ids: List[int] = []
    for _name, proc in nlp._components:
        model = getattr(proc, "model", None)
        if model is None or isinstance(model, bool):
            continue
        for node in model.walk():
            ids.extend(shim.id for shim in node.shims)
    return ids


def model_id_map(stored_ids: List[int], current_ids: List[int]) -> Dict[int, int]:
    """Map the model ids stored in a snapshot to the current pipeline.

    Both lists are collected in the same deterministic walk order, which is the
    case when the pipeline was created from the same config. The mapping lets
    the optimizer accumulators follow their parameters.
    """
    return dict(zip(stored_ids, current_ids))


def compute_structure_fingerprint(nlp: Any) -> str:
    """A fingerprint of the components and their model trees."""
    return hashlib.sha256(
        "\n".join(
            f"{name}:{fingerprint}"
            for name, fingerprint in component_fingerprints(nlp).items()
        ).encode("utf-8")
    ).hexdigest()


def component_fingerprints(nlp: Any) -> Dict[str, str]:
    """A fingerprint per component, so a mismatch can name the culprit."""
    fingerprints: Dict[str, str] = {}
    for name, proc in nlp._components:
        parts = [f"component:{type(proc).__name__}"]
        model = getattr(proc, "model", None)
        if model is not None and not isinstance(model, bool):
            for node in model.walk():
                dims = ",".join(
                    f"{dim}={node.get_dim(dim) if node.has_dim(dim) else None}"
                    for dim in sorted(node.dim_names)
                )
                params = ",".join(
                    f"{param}:{tuple(node.get_param(param).shape)}"
                    for param in sorted(node.param_names)
                    if node.has_param(param)
                )
                parts.append(f"node:{node.name}:{dims}:{params}")
        fingerprints[name] = hashlib.sha256(
            "\n".join(parts).encode("utf-8")
        ).hexdigest()
    return fingerprints


# Settings that describe *how* to train rather than *what* is trained. They can
# change without invalidating a snapshot, so they are excluded from identity.
NON_IDENTITY_TRAINING_SETTINGS = (
    "save_every",
    "frequency_unit",
    "max_keep",
    "resume_mismatch",
    "resume_validation",
)


def normalize_config(config: Any) -> Dict[str, Any]:
    """Return a plain, JSON-serializable copy of a config for identity checks."""
    data = json.loads(srsly.json_dumps(dict(config), sort_keys=True))
    data.get("system", {}).pop("gpu_allocator", None)
    training = data.get("training", {})
    for key in NON_IDENTITY_TRAINING_SETTINGS:
        training.pop(key, None)
    return data


def compute_config_fingerprint(config: Any) -> str:
    text = json.dumps(normalize_config(config), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_run_identity(nlp: Any, config: Any, training: Dict[str, Any]) -> Dict[str, Any]:
    """Collect the settings that identify a training run."""
    normalized = normalize_config(config)
    return {
        "config_fingerprint": compute_config_fingerprint(config),
        "structure_fingerprint": compute_structure_fingerprint(nlp),
        "component_fingerprints": component_fingerprints(nlp),
        "components": [name for name, _ in nlp._components],
        "factories": {name: type(proc).__name__ for name, proc in nlp._components},
        "frozen_components": list(training.get("frozen_components", [])),
        "annotating_components": list(training.get("annotating_components", [])),
        "batch_size": nlp.batch_size,
        "score_weights": {
            key: value
            for key, value in training.get("score_weights", {}).items()
            if value is not None
        },
        "corpora": normalized.get("corpora", {}),
        "train_corpus": training.get("train_corpus"),
        "dev_corpus": training.get("dev_corpus"),
    }


# Human-readable names for the fields compared on resume.
IDENTITY_LABELS = {
    "config_fingerprint": "config fingerprint",
    "structure_fingerprint": "model structure fingerprint",
    "component_fingerprints": "component model structure",
    "components": "pipeline components",
    "factories": "component factories",
    "frozen_components": "training.frozen_components",
    "annotating_components": "training.annotating_components",
    "batch_size": "nlp.batch_size",
    "score_weights": "training.score_weights",
    "corpora": "corpora (paths and reader settings)",
    "train_corpus": "training.train_corpus",
    "dev_corpus": "training.dev_corpus",
}


def diff_identity(stored: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """List the identity fields that differ, with the stored and current values.

    The coarse ``config_fingerprint`` is only reported when none of the
    human-readable fields explains the difference, so the message always points
    at something actionable.
    """
    diffs: List[str] = []
    for key, label in IDENTITY_LABELS.items():
        if key == "config_fingerprint" or key not in stored:
            continue
        if key == "component_fingerprints":
            diffs.extend(
                _diff_component_fingerprints(stored[key], current.get(key, {}))
            )
            continue
        if stored[key] != current.get(key):
            diffs.append(f"{label}: {stored[key]!r} -> {current.get(key)!r}")
    if not diffs and stored.get("config_fingerprint") != current.get(
        "config_fingerprint"
    ):
        diffs.append(
            "config fingerprint: the config differs in a section that isn't "
            "reported individually ({} -> {})".format(
                stored.get("config_fingerprint"), current.get("config_fingerprint")
            )
        )
    return diffs


def _diff_component_fingerprints(
    stored: Dict[str, str], current: Dict[str, str]
) -> List[str]:
    diffs: List[str] = []
    stored_names = set(stored)
    current_names = set(current)
    for name in sorted(stored_names - current_names):
        diffs.append(f"component '{name}': present in the snapshot, missing now")
    for name in sorted(current_names - stored_names):
        diffs.append(f"component '{name}': added since the snapshot")
    for name in sorted(stored_names & current_names):
        if stored[name] != current[name]:
            diffs.append(
                f"component '{name}': the model structure (layers, dimensions "
                "or parameter shapes) changed"
            )
    return diffs


def describe_environment() -> Dict[str, Any]:
    ops = get_current_ops()
    return {
        "thinc_ops": type(ops).__name__,
        "device_type": getattr(ops, "device_type", "cpu"),
        "device_id": getattr(ops, "device_id", None),
    }


def diff_environment(stored: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """Differences that affect performance but not correctness.

    Moving a run between CPU and GPU (or between GPUs) is supported: the
    accumulators are moved to the active backend when the snapshot is loaded.
    The change is still reported so it shows up in the log.
    """
    diffs: List[str] = []
    if stored.get("device_type") != current.get("device_type"):
        diffs.append(
            f"device type: {stored.get('device_type')} -> {current.get('device_type')}"
        )
    elif stored.get("device_id") != current.get("device_id"):
        diffs.append(
            f"GPU id: {stored.get('device_id')} -> {current.get('device_id')}"
        )
    return diffs


# --- Snapshot persistence --------------------------------------------------


class Snapshot:
    """A single on-disk snapshot."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def is_complete(self) -> bool:
        return (self.path / COMPLETE_MARKER).exists()

    def read_meta(self) -> Dict[str, Any]:
        try:
            meta = json.loads((self.path / FILE_META).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise SnapshotError(Errors.E1062.format(path=self.path, error=repr(e))) from e
        version = meta.get("format_version")
        if version not in SUPPORTED_SNAPSHOT_VERSIONS:
            raise SnapshotVersionError(
                Errors.E1060.format(
                    version=version,
                    path=self.path,
                    supported=list(SUPPORTED_SNAPSHOT_VERSIONS),
                )
            )
        return meta

    def validate(self) -> None:
        """Check that the snapshot is complete and that all files are intact."""
        if not self.path.is_dir():
            raise SnapshotError(
                Errors.E1062.format(path=self.path, error="not a directory")
            )
        # Check the format version first: an unknown version must fail loudly
        # instead of being treated as a corrupt snapshot and skipped.
        self.read_meta()
        if not self.is_complete:
            raise SnapshotError(
                Errors.E1062.format(path=self.path, error="missing COMPLETE marker")
            )
        try:
            manifest = json.loads((self.path / FILE_MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise SnapshotError(Errors.E1062.format(path=self.path, error=repr(e))) from e
        for rel_path, expected in manifest.get("files", {}).items():
            file_path = self.path / rel_path
            if not file_path.exists():
                raise SnapshotError(
                    Errors.E1062.format(
                        path=self.path, error=f"missing file '{rel_path}'"
                    )
                )
            if _file_checksum(file_path) != expected:
                raise SnapshotError(
                    Errors.E1062.format(
                        path=self.path, error=f"checksum mismatch for '{rel_path}'"
                    )
                )

    def read_state(self) -> Dict[str, Any]:
        try:
            return srsly.msgpack_loads((self.path / FILE_STATE).read_bytes())
        except Exception as e:
            raise SnapshotError(Errors.E1062.format(path=self.path, error=repr(e))) from e

    def read_rng(self) -> Dict[str, Any]:
        try:
            return srsly.msgpack_loads((self.path / FILE_RNG).read_bytes())
        except Exception as e:
            raise SnapshotError(Errors.E1062.format(path=self.path, error=repr(e))) from e

    @property
    def model_path(self) -> Path:
        return self.path / DIR_MODEL

    def __repr__(self) -> str:
        return f"<Snapshot {self.path}>"


def list_snapshots(output_path: Path) -> List[Snapshot]:
    snapshots_dir = get_snapshots_dir(output_path)
    if not snapshots_dir.is_dir():
        return []
    snapshots = [
        Snapshot(p)
        for p in snapshots_dir.iterdir()
        if p.is_dir() and not p.name.startswith(TMP_PREFIX)
    ]
    snapshots.sort(key=lambda s: s.name)
    return snapshots


def find_latest_snapshot(
    output_path: Path, run_id: Optional[str] = None
) -> Optional[Snapshot]:
    """Return the newest complete snapshot, skipping corrupt ones.

    Snapshots using an unsupported format version raise immediately: silently
    falling back to an older snapshot would hide an incompatible checkpoint.
    Corrupt snapshots are reported and the search continues with the
    next-newest snapshot.

    When ``run_id`` is given, only snapshots that belong to that run are
    considered, so a fresh run in an existing output directory never picks up
    the snapshots of an earlier run.
    """
    candidates: List[Snapshot] = []
    for snapshot in reversed(list_snapshots(output_path)):
        if run_id is not None and not _snapshot_belongs_to_run(snapshot, run_id):
            continue
        candidates.append(snapshot)
    for index, snapshot in enumerate(candidates):
        try:
            snapshot.validate()
        except SnapshotVersionError:
            raise
        except SnapshotError as e:
            newer = candidates[:index]
            fallback = (
                f" {candidates[index + 1].path}" if index + 1 < len(candidates) else ""
            )
            newer_text = (
                f" (the newer snapshots {', '.join(str(s.path) for s in newer)} "
                "are also incomplete)"
                if newer
                else ""
            )
            logger.warning(
                Warnings.W129.format(
                    path=snapshot.path, error=e, fallback=fallback + newer_text
                )
            )
            continue
        return snapshot
    return None


def _snapshot_belongs_to_run(snapshot: Snapshot, run_id: str) -> bool:
    try:
        return snapshot.read_meta().get("run_id") == run_id
    except SnapshotVersionError:
        # An unknown format version must not be silently skipped, even while
        # filtering by run id.
        raise
    except SnapshotError:
        return False


def load_snapshot(path: Path) -> Snapshot:
    """Validate and return the snapshot at ``path``."""
    snapshot = Snapshot(path)
    snapshot.validate()
    return snapshot


def write_snapshot(
    output_path: Path,
    *,
    nlp: Any,
    optimizer: Optimizer,
    state: Dict[str, Any],
    meta: Dict[str, Any],
    run_id: Optional[str] = None,
    keep: int = 2,
) -> Snapshot:
    """Write a new snapshot atomically and prune old ones.

    The snapshot is first written to a temporary directory in the snapshots
    directory and only renamed into place once every file (including the
    manifest and the COMPLETE marker) has been flushed to disk. If anything
    goes wrong the temporary directory is removed and the exception is
    propagated; existing snapshots are never touched.
    """
    snapshots_dir = get_snapshots_dir(output_path)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    step = int(meta["step"])
    epoch = int(meta["epoch"])
    tmp_path = (
        snapshots_dir
        / f"{TMP_PREFIX}{_snapshot_name(step, epoch, run_id)}-{os.getpid()}"
    )
    _rmtree(tmp_path)
    tmp_path.mkdir(parents=True)
    try:
        model_path = tmp_path / DIR_MODEL
        nlp.to_disk(model_path, exclude=["vocab"])
        state = dict(state)
        state["optimizer"] = serialize_optimizer(optimizer)
        # Save the model and shim ids so the optimizer accumulators can be
        # mapped onto the freshly initialized pipeline when the snapshot is
        # restored.
        state["model_ids"] = collect_model_ids(nlp)
        state["shim_ids"] = collect_shim_ids(nlp)
        _write_msgpack_atomic(tmp_path / FILE_STATE, state)
        _write_msgpack_atomic(tmp_path / FILE_RNG, make_rng_state())
        files: Dict[str, str] = {}
        for file_path in sorted(tmp_path.rglob("*")):
            if file_path.is_file():
                rel = str(file_path.relative_to(tmp_path))
                files[rel] = _file_checksum(file_path)
        _write_json_atomic(
            tmp_path / FILE_META,
            {"format_version": SNAPSHOT_FORMAT_VERSION, "created": time.time(), **meta},
        )
        files[FILE_META] = _file_checksum(tmp_path / FILE_META)
        _write_json_atomic(
            tmp_path / FILE_MANIFEST, {"files": files, "created": time.time()}
        )
        # The COMPLETE marker is written last. A missing marker means that the
        # snapshot was interrupted before it could be finished.
        (tmp_path / COMPLETE_MARKER).write_text("ok\n", encoding="utf-8")
        _fsync_dir(tmp_path)
        final_path = _unique_snapshot_path(snapshots_dir, step, epoch, run_id)
        os.replace(tmp_path, final_path)
        _fsync_dir(snapshots_dir)
    except BaseException:
        _rmtree(tmp_path)
        raise
    _prune_snapshots(snapshots_dir, keep, run_id=run_id)
    return Snapshot(final_path)


def _prune_snapshots(snapshots_dir: Path, keep: int, run_id: Optional[str] = None) -> None:
    """Keep the newest ``keep`` complete snapshots and remove the rest.

    Pruning is scoped to ``run_id`` when one is given, so a new run in an
    existing output directory doesn't delete the snapshots of a previous run.
    Incomplete (temporary) snapshots are always removed - they can only be the
    leftovers of an interrupted write.
    """
    directories = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    complete: List[Path] = []
    for path in directories:
        if path.name.startswith(TMP_PREFIX):
            _rmtree(path)
        elif (path / COMPLETE_MARKER).exists():
            complete.append(path)
    complete.sort()
    if run_id is not None:
        complete = [
            path for path in complete if path.name.endswith(f"-{run_id}")
        ]
    if keep < 0:
        return
    for path in complete[: max(0, len(complete) - keep)]:
        _rmtree(path)
        logger.debug("Pruned old training snapshot: %s", path)


def format_snapshot_location(snapshot: Snapshot) -> str:
    try:
        meta = snapshot.read_meta()
    except SnapshotError:  # pragma: no cover
        return snapshot.name
    return f"{snapshot.path} (step {meta.get('step')}, epoch {meta.get('epoch')})"
