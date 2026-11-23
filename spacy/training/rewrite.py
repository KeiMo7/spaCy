"""Ledger-based remapping of named-entity offsets across text rewrites.

This module answers one question: given a source text, a list of rewrite
operations (character spans plus replacement strings) and entity annotations
on the *source* text, where do the entities live on the *rewritten* text?

The mapping is derived exclusively from the offset ledger built from the
operations. The rewritten text is never searched for the old fragments, so a
replacement that happens to contain a copy of the original substring cannot
confuse the mapping.

Terminology used throughout the module:

* **ledger** - the ordered, non-overlapping list of accepted rewrite
  operations together with their cumulative character delta.
* **disposition** - the fate of one input entity: ``kept``, ``replaced``,
  ``shrunk``, ``adjusted`` or ``dropped``.
* **crossing** - an entity that overlaps a rewrite span without being fully
  contained in it.

All per-entity and per-operation rows are frozen dataclasses emitted in a
deterministic order, so running the same input twice yields identical reports.
"""

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..errors import Errors
from ..tokens import Doc, Span

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CROSSING_POLICIES = ("shrink", "drop")
CONFLICT_POLICIES = ("raise", "keep_first", "keep_longest")
ALIGNMENT_MODES = ("none", "strict", "contract", "expand")
ERROR_POLICIES = ("raise", "skip")

#: Placeholder offsets used for dropped entities so that they can never be
#: mistaken for a valid zero-length span.
DROPPED_OFFSET = -1


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewriteOp:
    """A single rewrite operation on the source text.

    start (int): First character of the replaced span (inclusive).
    end (int): Character after the replaced span (exclusive).
    replacement (str): The text that replaces ``text[start:end]``.
    original (Optional[str]): Optional declared source fragment. When set, it
        must match ``text[start:end]`` and its length must equal ``end - start``.
    """

    start: int
    end: int
    replacement: str
    original: Optional[str] = None

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class EntityAnnotation:
    """One entity annotation on the source text."""

    start: int
    end: int
    label: str


@dataclass(frozen=True)
class EntityRewrite:
    """The remapping result for one source entity."""

    index: int
    label: str
    original_start: int
    original_end: int
    original_text: str
    new_start: int
    new_end: int
    disposition: str
    reason: Optional[str] = None
    ledger_start: Optional[int] = None
    ledger_end: Optional[int] = None

    @property
    def as_row(self) -> Dict[str, Any]:
        """The per-entity table row: original span, new span, disposition."""
        return {
            "index": self.index,
            "label": self.label,
            "original_start": self.original_start,
            "original_end": self.original_end,
            "original_text": self.original_text,
            "new_start": self.new_start,
            "new_end": self.new_end,
            "disposition": self.disposition,
            "reason": self.reason,
            "ledger_start": self.ledger_start,
            "ledger_end": self.ledger_end,
        }


@dataclass(frozen=True)
class RewriteProblem:
    """A rewrite operation that could not be applied as declared."""

    index: int
    kind: str
    message: str
    start: Optional[int] = None
    end: Optional[int] = None
    replacement: Optional[str] = None
    declared: Optional[str] = None
    found: Optional[str] = None
    skipped: bool = False

    @property
    def as_row(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "message": self.message,
            "start": self.start,
            "end": self.end,
            "replacement": self.replacement,
            "declared": self.declared,
            "found": self.found,
            "skipped": self.skipped,
        }


@dataclass
class RewriteResult:
    """The result of rewriting one text and its entity annotations."""

    text_before: str
    text_after: str
    entities: List[EntityRewrite] = field(default_factory=list)
    problems: List[RewriteProblem] = field(default_factory=list)
    operations: List[RewriteOp] = field(default_factory=list)
    entity_count_before: int = 0
    entity_count_after: int = 0
    kept_count: int = 0
    replaced_count: int = 0
    shrunk_count: int = 0
    adjusted_count: int = 0
    dropped_count: int = 0
    skipped_operation_count: int = 0
    char_delta: int = 0
    char_delta_declared: int = 0

    @property
    def char_count_before(self) -> int:
        return len(self.text_before)

    @property
    def char_count_after(self) -> int:
        return len(self.text_after)

    @property
    def reconciles(self) -> bool:
        """Whether all four summary quantities agree with each other."""
        counted_after = (
            self.kept_count
            + self.replaced_count
            + self.shrunk_count
            + self.adjusted_count
        )
        return (
            len(self.entities) == self.entity_count_before
            and self.entity_count_after == counted_after
            and self.entity_count_before == self.entity_count_after + self.dropped_count
            and self.char_count_after
            == self.char_count_before + self.char_delta_declared
            and self.char_delta == self.char_delta_declared
        )

    def summary(self) -> Dict[str, Any]:
        """A JSON-serializable reconciliation table."""
        return {
            "char_count_before": self.char_count_before,
            "char_count_after": self.char_count_after,
            "char_delta": self.char_delta,
            "char_delta_declared": self.char_delta_declared,
            "entity_count_before": self.entity_count_before,
            "entity_count_after": self.entity_count_after,
            "kept_count": self.kept_count,
            "replaced_count": self.replaced_count,
            "shrunk_count": self.shrunk_count,
            "adjusted_count": self.adjusted_count,
            "dropped_count": self.dropped_count,
            "skipped_operation_count": self.skipped_operation_count,
            "reconciles": self.reconciles,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_before": self.text_before,
            "text_after": self.text_after,
            "summary": self.summary(),
            "operations": [
                {
                    "start": op.start,
                    "end": op.end,
                    "replacement": op.replacement,
                    "original": op.original,
                }
                for op in self.operations
            ],
            "entities": [entity.as_row for entity in self.entities],
            "problems": [problem.as_row for problem in self.problems],
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_table(self) -> List[Dict[str, Any]]:
        """Return the per-entity table (original span, new span, disposition)."""
        return [entity.as_row for entity in self.entities]


@dataclass
class RewrittenDoc:
    """A rewritten Doc together with the offset reconciliation report."""

    doc: Doc
    report: RewriteResult


# ---------------------------------------------------------------------------
# Operation validation
# ---------------------------------------------------------------------------


def _coerce_operation(value: Any) -> RewriteOp:
    if isinstance(value, RewriteOp):
        return value
    if isinstance(value, dict):
        return RewriteOp(
            start=int(value["start"]),
            end=int(value["end"]),
            replacement=str(value["replacement"]),
            original=value.get("original"),
        )
    if isinstance(value, (tuple, list)):
        if len(value) == 3:
            return RewriteOp(
                start=int(value[0]), end=int(value[1]), replacement=str(value[2])
            )
        if len(value) == 4:
            return RewriteOp(
                start=int(value[0]),
                end=int(value[1]),
                replacement=str(value[2]),
                original=value[3],
            )
    raise TypeError(
        "Rewrite operations must be RewriteOp objects, "
        "(start, end, replacement[, original]) tuples or mappings with those keys."
    )


def _single_operation_problem(
    index: int, op: RewriteOp, text: str
) -> Optional[RewriteProblem]:
    """Return a problem for one operation, or ``None`` if it is well formed."""
    text_len = len(text)
    if op.start < 0 or op.end < 0 or op.start >= text_len or op.end > text_len:
        return RewriteProblem(
            index=index,
            kind="out_of_bounds",
            message=f"span {op.start}..{op.end} is outside text of length {text_len}",
            start=op.start,
            end=op.end,
            replacement=op.replacement,
            declared=op.original,
        )
    if op.start >= op.end:
        return RewriteProblem(
            index=index,
            kind="empty_span",
            message=f"span {op.start}..{op.end} is empty",
            start=op.start,
            end=op.end,
            replacement=op.replacement,
            declared=op.original,
        )
    if op.original is None:
        return None
    found = text[op.start : op.end]
    if op.length != len(op.original):
        return RewriteProblem(
            index=index,
            kind="declared_length_mismatch",
            message=(
                f"declared fragment has length {len(op.original)} but the "
                f"span length is {op.length}"
            ),
            start=op.start,
            end=op.end,
            replacement=op.replacement,
            declared=op.original,
            found=found,
        )
    if found != op.original:
        return RewriteProblem(
            index=index,
            kind="content_mismatch",
            message=(
                f"source fragment {found!r} does not match declared fragment "
                f"{op.original!r}"
            ),
            start=op.start,
            end=op.end,
            replacement=op.replacement,
            declared=op.original,
            found=found,
        )
    return None


def _raise_operation_problem(problem: RewriteProblem) -> None:
    raise ValueError(
        Errors.E1058.format(
            index=problem.index,
            message=problem.message,
            start=problem.start,
            end=problem.end,
            replacement=problem.replacement,
            found=problem.found,
            declared=problem.declared,
            skipped=False,
        )
    )


def _validate_operations(
    text: str,
    raw_operations: Sequence[Any],
    on_error: str,
) -> Tuple[List[Tuple[int, RewriteOp]], List[RewriteProblem]]:
    """Validate operations and return ``(index, op)`` pairs plus problems.

    Out-of-bounds spans, empty spans, declared-length mismatches and content
    mismatches are reported with the operation index, the source fragment and
    the declared value. With ``on_error="skip"`` they are dropped from the
    ledger and marked as skipped; with ``on_error="raise"`` the first problem
    raises ``ValueError``.
    """
    if on_error not in ERROR_POLICIES:
        raise ValueError(
            Errors.E1060.format(policy=on_error, options=", ".join(ERROR_POLICIES))
        )
    problems: List[RewriteProblem] = []
    candidates: List[Tuple[int, RewriteOp]] = []
    for index, raw in enumerate(raw_operations):
        op = _coerce_operation(raw)
        problem = _single_operation_problem(index, op, text)
        if problem is not None:
            if on_error == "raise":
                _raise_operation_problem(problem)
            problems.append(replace(problem, skipped=True))
            continue
        candidates.append((index, op))

    # Deterministic order: start, end, original index.
    candidates.sort(key=lambda pair: (pair[1].start, pair[1].end, pair[0]))
    accepted: List[Tuple[int, RewriteOp]] = []
    for index, op in candidates:
        if accepted and op.start < accepted[-1][1].end:
            previous_index, previous_op = accepted[-1]
            problem = RewriteProblem(
                index=index,
                kind="overlap",
                message=(
                    f"operation {index} ({op.start}, {op.end}) overlaps with "
                    f"operation {previous_index} "
                    f"({previous_op.start}, {previous_op.end})"
                ),
                start=op.start,
                end=op.end,
                replacement=op.replacement,
                declared=op.original,
            )
            if on_error == "raise":
                raise ValueError(
                    Errors.E1059.format(
                        index=index,
                        start=op.start,
                        end=op.end,
                        previous_index=previous_index,
                        previous_start=previous_op.start,
                        previous_end=previous_op.end,
                    )
                )
            problems.append(replace(problem, skipped=True))
            continue
        accepted.append((index, op))
    return accepted, problems


# ---------------------------------------------------------------------------
# Offset ledger
# ---------------------------------------------------------------------------


class _Ledger:
    """Monotonic piecewise-linear map from source offsets to rewritten offsets.

    The map is fully determined by the accepted operations; it never inspects
    the rewritten text. Offsets strictly inside a replaced span are clamped to
    the replacement boundaries, which is only used for diagnostics because
    entity boundaries inside a replaced span are handled by containment.
    """

    def __init__(self, text: str, accepted: Sequence[Tuple[int, RewriteOp]]) -> None:
        self.text = text
        self.accepted = list(accepted)
        self._shifts: List[Tuple[int, int, int, int, int]] = []
        shift = 0
        for _, op in self.accepted:
            new_start = op.start + shift
            new_end = new_start + len(op.replacement)
            self._shifts.append((op.start, op.end, new_start, new_end, shift))
            shift += len(op.replacement) - op.length
        self.total_delta = shift
        self.new_text = self._build_text()

    def _build_text(self) -> str:
        pieces: List[str] = []
        cursor = 0
        for _, op in self.accepted:
            pieces.append(self.text[cursor : op.start])
            pieces.append(op.replacement)
            cursor = op.end
        pieces.append(self.text[cursor:])
        return "".join(pieces)

    def map_offset(self, offset: int) -> int:
        if offset <= 0:
            return 0
        for start, end, new_start, new_end, shift in self._shifts:
            if offset < start:
                return offset + shift
            if offset < end:
                # Strictly inside a replaced span: clamp to the replacement.
                return new_start + min(offset - start, new_end - new_start)
        return offset + self.total_delta

    def overlapping_ops(self, start: int, end: int) -> List[Tuple[int, RewriteOp]]:
        return [
            (index, op)
            for index, op in self.accepted
            if start < op.end and end > op.start
        ]

    def replaced_span(self, op: RewriteOp) -> Optional[Tuple[int, int]]:
        for start, end, new_start, new_end, _ in self._shifts:
            if start == op.start and end == op.end:
                return new_start, new_end
        raise AssertionError("operation is not part of the ledger")

    def replacement_spans(self) -> List[Tuple[int, int]]:
        """Replacement regions in *new* text coordinates."""
        return [(new_start, new_end) for _, _, new_start, new_end, _ in self._shifts]

    def overlaps_replacement(self, start: int, end: int) -> bool:
        """Whether ``[start, end)`` overlaps any *non-empty* replacement region.

        Zero-length replacements (deletions) insert no new characters, so they
        can never carry synthetic text into an annotation.
        """
        return any(
            new_start < new_end and start < new_end and end > new_start
            for new_start, new_end in self.replacement_spans()
        )

    def unchanged_pieces(self, start: int, end: int) -> List[Tuple[int, int]]:
        """Source intervals inside ``[start, end)`` untouched by any operation."""
        pieces: List[Tuple[int, int]] = []
        cursor = start
        for _, op in self.accepted:
            if op.end <= start:
                continue
            if op.start >= end:
                break
            piece_end = min(op.start, end)
            if cursor < piece_end:
                pieces.append((cursor, piece_end))
            cursor = max(cursor, op.end)
            if cursor >= end:
                break
        if cursor < end:
            pieces.append((cursor, end))
        return pieces


# ---------------------------------------------------------------------------
# Token-boundary alignment
# ---------------------------------------------------------------------------


def _resolve_tokenizer(tokenizer: Any) -> Callable[[str], Doc]:
    if tokenizer is None:
        raise ValueError(Errors.E1063)
    if hasattr(tokenizer, "make_doc"):
        return tokenizer.make_doc
    if callable(tokenizer):
        return tokenizer
    raise TypeError(
        "tokenizer must be a Language, a Tokenizer or a callable taking text."
    )


def _token_index_at(doc: Doc, char: int) -> Optional[int]:
    for token in doc:
        if token.idx <= char < token.idx + len(token):
            return token.i
    return None


def _align_span(
    doc: Doc, start: int, end: int, mode: str, allow_widen: bool = True
) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
    """Snap a raw character span onto the Doc's token boundaries.

    Returns ``((new_start, new_end), note)`` or ``(None, reason)``. Every
    boundary comes from the supplied tokenizer's Doc; no independent boundary
    rules are invented here.

    ``contract`` first contracts to fully-contained tokens. When no token is
    fully inside the span but the span does fall inside a single token (the
    typical result of a replacement that splits or merges words) it widens to
    that token, because that is the only legal boundary that still carries the
    annotation. ``allow_widen=False`` is used for shrunk entities, where
    widening would pull replacement text into the "remaining part"; those are
    reported as unalignable so the caller can drop them with a reason.
    """
    if start >= end:
        return None, "empty_span"
    if mode == "none":
        return (start, end), None
    span = doc.char_span(start, end, alignment_mode=mode)
    if span is not None:
        aligned = (span.start_char, span.end_char)
        if aligned == (start, end):
            return aligned, None
        return aligned, "boundary_adjusted"
    if mode == "contract" and allow_widen:
        start_token = _token_index_at(doc, start)
        end_token = _token_index_at(doc, end - 1)
        if start_token is not None and start_token == end_token:
            token = doc[start_token]
            return (token.idx, token.idx + len(token)), "widened_to_token_boundary"
    return None, "no_token_alignment"


# ---------------------------------------------------------------------------
# Entity handling
# ---------------------------------------------------------------------------


def _coerce_entities(entities: Iterable[Any]) -> List[EntityAnnotation]:
    result: List[EntityAnnotation] = []
    for entity in entities:
        if isinstance(entity, EntityAnnotation):
            result.append(entity)
        elif isinstance(entity, dict):
            result.append(
                EntityAnnotation(
                    start=int(entity["start"]),
                    end=int(entity["end"]),
                    label=str(entity["label"]),
                )
            )
        elif isinstance(entity, (tuple, list)):
            if len(entity) < 3:
                raise ValueError("Entity tuples must be (start, end, label).")
            result.append(
                EntityAnnotation(
                    start=int(entity[0]), end=int(entity[1]), label=str(entity[2])
                )
            )
        else:
            raise TypeError(
                "Entities must be EntityAnnotation objects, mappings or "
                "(start, end, label) tuples."
            )
    return result


def _validate_entities(text: str, entities: Sequence[EntityAnnotation]) -> None:
    ordered = sorted(entities, key=lambda e: (e.start, e.end))
    previous: Optional[EntityAnnotation] = None
    for entity in ordered:
        if entity.start < 0 or entity.end > len(text) or entity.start >= entity.end:
            raise ValueError(
                Errors.E1065.format(
                    start=entity.start,
                    end=entity.end,
                    label=entity.label,
                    length=len(text),
                )
            )
        if previous is not None and entity.start < previous.end:
            raise ValueError(
                Errors.E1064.format(
                    start=entity.start,
                    end=entity.end,
                    label=entity.label,
                    other_start=previous.start,
                    other_end=previous.end,
                    other_label=previous.label,
                )
            )
        previous = entity


def _drop(
    index: int,
    annotation: EntityAnnotation,
    text: str,
    reason: str,
    ledger_span: Optional[Tuple[int, int]] = None,
) -> EntityRewrite:
    return EntityRewrite(
        index=index,
        label=annotation.label,
        original_start=annotation.start,
        original_end=annotation.end,
        original_text=text[annotation.start : annotation.end],
        new_start=DROPPED_OFFSET,
        new_end=DROPPED_OFFSET,
        disposition="dropped",
        reason=reason,
        ledger_start=ledger_span[0] if ledger_span else None,
        ledger_end=ledger_span[1] if ledger_span else None,
    )


def _resolve_conflicts(
    results: List[EntityRewrite], policy: str
) -> Tuple[List[EntityRewrite], List[RewriteProblem]]:
    """Resolve overlaps created by boundary snapping.

    ``raise`` refuses to produce overlapping output. ``keep_first`` keeps the
    entity that sorts first by ``(new_start, new_end, index)``.
    ``keep_longest`` keeps the longest entity in each connected overlap
    cluster, breaking ties by original index. Everything else is dropped and
    recorded as an ``overlap_conflict``.
    """
    ordered = sorted(results, key=lambda r: (r.new_start, r.new_end, r.index))
    clusters: List[List[EntityRewrite]] = []
    current: List[EntityRewrite] = []
    cluster_end = -1
    for result in ordered:
        if current and result.new_start >= cluster_end:
            clusters.append(current)
            current = []
            cluster_end = -1
        current.append(result)
        cluster_end = max(cluster_end, result.new_end)
    if current:
        clusters.append(current)

    kept_indices = set()
    problems: List[RewriteProblem] = []
    for cluster in clusters:
        if len(cluster) == 1:
            kept_indices.add(cluster[0].index)
            continue
        if policy == "raise":
            first, second = cluster[0], cluster[1]
            raise ValueError(
                Errors.E1062.format(
                    start=first.new_start,
                    end=first.new_end,
                    label=first.label,
                    other_start=second.new_start,
                    other_end=second.new_end,
                    other_label=second.label,
                )
            )
        if policy == "keep_first":
            winner = cluster[0]
        else:  # keep_longest
            winner = sorted(
                cluster, key=lambda r: (-(r.new_end - r.new_start), r.index)
            )[0]
        kept_indices.add(winner.index)
        for result in cluster:
            if result.index == winner.index:
                continue
            problems.append(
                RewriteProblem(
                    index=result.index,
                    kind="overlap_conflict",
                    message=(
                        f"entity {result.index} ({result.new_start}, "
                        f"{result.new_end}, {result.label!r}) overlaps kept "
                        f"entity {winner.index} and was dropped"
                    ),
                    start=result.new_start,
                    end=result.new_end,
                )
            )
    return [r for r in results if r.index in kept_indices], problems


def _ledger_span_for_entity(
    index: int,
    annotation: EntityAnnotation,
    text: str,
    ledger: "_Ledger",
    crossing: str,
) -> Tuple[Optional[Tuple[int, int]], str, Optional[str], Optional[EntityRewrite]]:
    """Decide the ledger span and disposition for one entity.

    Returns ``(ledger_span, disposition, reason, dropped)``. When ``dropped``
    is not ``None`` the caller must use it as the final row.
    """
    overlapping = ledger.overlapping_ops(annotation.start, annotation.end)
    if not overlapping:
        span = (
            ledger.map_offset(annotation.start),
            ledger.map_offset(annotation.end),
        )
        return span, "kept", None, None
    if (
        len(overlapping) == 1
        and annotation.start >= overlapping[0][1].start
        and annotation.end <= overlapping[0][1].end
    ):
        op_index, op = overlapping[0]
        span = ledger.replaced_span(op)
        if span is not None and span[0] == span[1]:
            return (
                span,
                "replaced",
                None,
                _drop(index, annotation, text, "empty_replacement", span),
            )
        return span, "replaced", f"contained_in_operation:{op_index}", None
    if crossing == "drop":
        return (
            None,
            "dropped",
            None,
            _drop(index, annotation, text, "crosses_rewrite_boundary"),
        )
    pieces = [
        piece
        for piece in ledger.unchanged_pieces(annotation.start, annotation.end)
        if piece[1] > piece[0]
    ]
    if not pieces:
        return (
            None,
            "dropped",
            None,
            _drop(index, annotation, text, "no_untouched_remainder"),
        )
    # Longest remaining piece; ties prefer the left-most piece.
    piece = sorted(pieces, key=lambda p: (-(p[1] - p[0]), p[0]))[0]
    span = (ledger.map_offset(piece[0]), ledger.map_offset(piece[1]))
    return span, "shrunk", "crosses_rewrite_boundary", None


def _apply_alignment(
    index: int,
    annotation: EntityAnnotation,
    text: str,
    span: Tuple[int, int],
    disposition: str,
    reason: Optional[str],
    token_doc: Optional[Doc],
    alignment: str,
) -> Tuple[Optional[Tuple[int, int]], str, Optional[str], Optional[EntityRewrite]]:
    """Align a ledger span to token boundaries and update the disposition."""
    if token_doc is None:
        return span, disposition, reason, None
    # A fully contained entity must cover the whole replacement, so it is
    # aligned with ``expand``; contracting would cut it back to the first
    # token of a multi-token replacement.
    align_mode = alignment
    if disposition == "replaced" and alignment == "contract":
        align_mode = "expand"
    aligned, note = _align_span(
        token_doc,
        span[0],
        span[1],
        align_mode,
        allow_widen=disposition != "shrunk",
    )
    if aligned is None:
        return None, disposition, reason, _drop(index, annotation, text, note, span)
    if aligned[0] >= aligned[1]:
        return (
            None,
            disposition,
            reason,
            _drop(index, annotation, text, "empty_after_alignment", span),
        )
    if aligned != span:
        if disposition == "kept":
            disposition = "adjusted"
        if note:
            if reason is None:
                reason = note
            elif note not in reason:
                reason = f"{reason};{note}"
    return aligned, disposition, reason, None


def _alignment_stays_within_original(
    ledger: "_Ledger", mapped: Tuple[int, int], disposition: str
) -> bool:
    """Whether an aligned span still consists solely of original characters.

    Only a ``replaced`` entity is allowed to cover replacement text. Any other
    entity that ends up overlapping a replacement region after token snapping
    has been widened into synthetic characters, which would silently change
    what the annotation denotes.
    """
    if disposition == "replaced":
        return True
    return not ledger.overlaps_replacement(*mapped)


def _map_entity(
    index: int,
    annotation: EntityAnnotation,
    text: str,
    ledger: "_Ledger",
    token_doc: Optional[Doc],
    *,
    crossing: str,
    alignment: str,
) -> EntityRewrite:
    """Map one source entity through the ledger and align it to tokens.

    This function is pure: it either returns a live ``EntityRewrite`` or one
    that is already marked as dropped with a reason. It never raises for data
    reasons, so the caller can keep processing the remaining entities.
    """
    span, disposition, reason, dropped = _ledger_span_for_entity(
        index, annotation, text, ledger, crossing
    )
    if dropped is not None:
        return dropped
    if span is None or span[0] >= span[1]:
        return _drop(index, annotation, text, "empty_span", span)
    if disposition == "shrunk" and ledger.overlaps_replacement(*span):
        # Shrinking must leave only original characters. A remainder that
        # still runs into a replacement would carry synthetic text into the
        # annotation, so it is dropped with an explicit reason.
        return _drop(index, annotation, text, "remainder_overlaps_replacement")

    mapped, disposition, reason, dropped = _apply_alignment(
        index,
        annotation,
        text,
        span,
        disposition,
        reason,
        token_doc,
        alignment,
    )
    if dropped is not None:
        return dropped
    assert mapped is not None
    if not _alignment_stays_within_original(ledger, mapped, disposition):
        return _drop(index, annotation, text, "alignment_overlaps_replacement")
    return EntityRewrite(
        index=index,
        label=annotation.label,
        original_start=annotation.start,
        original_end=annotation.end,
        original_text=text[annotation.start : annotation.end],
        new_start=mapped[0],
        new_end=mapped[1],
        disposition=disposition,
        reason=reason,
        ledger_start=span[0],
        ledger_end=span[1],
    )


def _identity_result(
    text: str, entity_list: Sequence[EntityAnnotation], result: RewriteResult
) -> None:
    for index, annotation in enumerate(entity_list):
        result.entities.append(
            EntityRewrite(
                index=index,
                label=annotation.label,
                original_start=annotation.start,
                original_end=annotation.end,
                original_text=text[annotation.start : annotation.end],
                new_start=annotation.start,
                new_end=annotation.end,
                disposition="kept",
                ledger_start=annotation.start,
                ledger_end=annotation.end,
            )
        )


def _apply_conflict_policy(result: RewriteResult, conflicts: str) -> None:
    surviving = [ent for ent in result.entities if ent.disposition != "dropped"]
    resolved, conflict_problems = _resolve_conflicts(surviving, conflicts)
    kept_indices = {ent.index for ent in resolved}
    final_entities: List[EntityRewrite] = []
    for entity in result.entities:
        if entity.disposition == "dropped" or entity.index in kept_indices:
            final_entities.append(entity)
        else:
            final_entities.append(
                replace(
                    entity,
                    new_start=DROPPED_OFFSET,
                    new_end=DROPPED_OFFSET,
                    disposition="dropped",
                    reason="overlap_conflict",
                )
            )
    result.entities = final_entities
    result.problems.extend(conflict_problems)


def _validate_policies(crossing: str, conflicts: str, alignment: str) -> None:
    if crossing not in CROSSING_POLICIES:
        raise ValueError(
            Errors.E1060.format(policy=crossing, options=", ".join(CROSSING_POLICIES))
        )
    if conflicts not in CONFLICT_POLICIES:
        raise ValueError(
            Errors.E1060.format(policy=conflicts, options=", ".join(CONFLICT_POLICIES))
        )
    if alignment not in ALIGNMENT_MODES:
        raise ValueError(
            Errors.E1061.format(mode=alignment, options=", ".join(ALIGNMENT_MODES))
        )


def rewrite_annotations(
    text: str,
    operations: Iterable[Any],
    entities: Iterable[Any],
    *,
    crossing: str = "shrink",
    conflicts: str = "raise",
    alignment: str = "contract",
    on_error: str = "raise",
    tokenizer: Any = None,
) -> RewriteResult:
    """Rewrite ``text`` and remap ``entities`` from the source to the new text.

    text (str): The source text.
    operations (Iterable): Rewrite operations as ``RewriteOp`` objects,
        ``(start, end, replacement[, original])`` tuples or mappings.
    entities (Iterable): Source entities as ``EntityAnnotation`` objects,
        ``(start, end, label)`` tuples or mappings.
    crossing (str): What to do with an entity that crosses an operation
        boundary: ``"shrink"`` (keep the longest untouched remainder) or
        ``"drop"``. Defaults to ``"shrink"``.
    conflicts (str): What to do when two remapped entities overlap:
        ``"raise"``, ``"keep_first"`` or ``"keep_longest"``. Defaults to
        ``"raise"``.
    alignment (str): How remapped offsets are snapped to token boundaries:
        ``"none"`` (raw ledger offsets), ``"strict"``, ``"contract"`` (default)
        or ``"expand"``. All modes except ``"none"`` use the provided
        tokenizer, so boundaries are exactly its token boundaries.
    on_error (str): ``"raise"`` or ``"skip"`` for malformed operations.
    tokenizer (Language/Tokenizer/callable): Used to tokenize the rewritten
        text. Required unless ``alignment="none"`` or no operation survives.

    RETURNS (RewriteResult): The rewritten text, one row per entity and a
        reconciliation summary.
    """
    _validate_policies(crossing, conflicts, alignment)
    text = str(text)
    raw_operations = list(operations)
    entity_list = _coerce_entities(entities)
    _validate_entities(text, entity_list)

    accepted, problems = _validate_operations(text, raw_operations, on_error)
    ledger = _Ledger(text, accepted)
    result = RewriteResult(
        text_before=text,
        text_after=ledger.new_text,
        problems=list(problems),
        operations=[op for _, op in accepted],
        entity_count_before=len(entity_list),
        char_delta=len(ledger.new_text) - len(text),
        char_delta_declared=ledger.total_delta,
        skipped_operation_count=sum(1 for problem in problems if problem.skipped),
    )

    # No operation survived: the annotations are already correct and must be
    # returned exactly as supplied (no tokenizer round trip).
    if not accepted:
        _identity_result(text, entity_list, result)
        _finalize_counts(result)
        return result

    token_doc: Optional[Doc] = None
    if alignment != "none":
        token_doc = _resolve_tokenizer(tokenizer)(ledger.new_text)

    for index, annotation in enumerate(entity_list):
        result.entities.append(
            _map_entity(
                index,
                annotation,
                text,
                ledger,
                token_doc,
                crossing=crossing,
                alignment=alignment,
            )
        )

    _apply_conflict_policy(result, conflicts)
    _finalize_counts(result)
    return result


def _finalize_counts(result: RewriteResult) -> None:
    counts = {"kept": 0, "replaced": 0, "shrunk": 0, "adjusted": 0, "dropped": 0}
    for entity in result.entities:
        counts[entity.disposition] += 1
    result.kept_count = counts["kept"]
    result.replaced_count = counts["replaced"]
    result.shrunk_count = counts["shrunk"]
    result.adjusted_count = counts["adjusted"]
    result.dropped_count = counts["dropped"]
    result.entity_count_after = (
        counts["kept"] + counts["replaced"] + counts["shrunk"] + counts["adjusted"]
    )


# ---------------------------------------------------------------------------
# Doc-level helpers and atomic corpus writing
# ---------------------------------------------------------------------------


def rewrite_doc(
    doc: Doc,
    operations: Iterable[Any],
    *,
    nlp: Any = None,
    tokenizer: Any = None,
    crossing: str = "shrink",
    conflicts: str = "raise",
    alignment: str = "contract",
    on_error: str = "raise",
) -> RewrittenDoc:
    """Rewrite a Doc's text and entity annotations using the offset ledger.

    Token-level annotations other than entities are not carried over because
    the text they referred to has changed. The returned Doc is tokenized with
    ``nlp.make_doc`` (or ``tokenizer``), so its tokenization always matches the
    boundaries used by the remapping.

    ``alignment="none"`` is rejected: a Doc's entity spans must align to its
    tokens, so raw character offsets cannot be stored on a Doc. Use
    :func:`rewrite_annotations` directly for unaligned offsets.
    """
    if alignment == "none":
        raise ValueError(Errors.E1066)
    tokenizer_obj = tokenizer if tokenizer is not None else nlp
    if tokenizer_obj is None:
        raise ValueError(Errors.E1063)
    entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
    result = rewrite_annotations(
        doc.text,
        operations,
        entities,
        crossing=crossing,
        conflicts=conflicts,
        alignment=alignment,
        on_error=on_error,
        tokenizer=tokenizer_obj,
    )
    if not result.operations:
        # No rewrite happened: return an exact copy so even a non-standard
        # (gold) tokenization and every Doc-level annotation survive untouched.
        return RewrittenDoc(doc=doc.copy(), report=result)
    new_doc = _resolve_tokenizer(tokenizer_obj)(result.text_after)
    spans: List[Span] = []
    for entity in result.entities:
        if entity.disposition == "dropped":
            continue
        span = new_doc.char_span(
            entity.new_start,
            entity.new_end,
            label=entity.label,
            alignment_mode="strict",
        )
        if span is None:
            raise AssertionError(
                "a non-dropped entity must align to the rewritten tokenization"
            )
        spans.append(span)
    new_doc.ents = spans
    return RewrittenDoc(doc=new_doc, report=result)


def rewrite_docbin(
    source: Union[str, Path],
    target: Union[str, Path],
    rewriter: Callable[[Doc, Any], RewrittenDoc],
    *,
    nlp: Any,
    store_user_data: bool = True,
) -> Dict[str, Any]:
    """Atomically rewrite a DocBin file and write the result to ``target``.

    The output is written to a ``.partial`` sibling first and moved into place
    with ``os.replace`` only after every document has been rewritten. A failed
    run therefore never leaves a half-written corpus behind, and a stale
    ``.partial`` file from an interrupted run is removed on the next attempt.
    """
    from ..tokens import DocBin

    source = Path(source)
    target = Path(target)
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        partial.unlink()
    doc_bin_in = DocBin().from_disk(source)
    docs = list(doc_bin_in.get_docs(nlp.vocab))
    reports: List[Dict[str, Any]] = []
    # Preserve the source corpus's serialized attribute set so the rewritten
    # file stays format-compatible with whatever produced the input.
    doc_bin_out = DocBin(
        attrs=doc_bin_in.attrs,
        store_user_data=store_user_data or doc_bin_in.store_user_data,
    )
    for doc in docs:
        rewritten = rewriter(doc, nlp)
        doc_bin_out.add(rewritten.doc)
        reports.append(rewritten.report.summary())
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(partial, "wb") as file_:
            file_.write(doc_bin_out.to_bytes())
            file_.flush()
            os.fsync(file_.fileno())
        os.replace(partial, target)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return {
        "source": str(source),
        "target": str(target),
        "document_count": len(docs),
        "summaries": reports,
        "reconciles": all(summary["reconciles"] for summary in reports),
    }


def rewrite_docbin_from_manifest(
    source: Union[str, Path],
    target: Union[str, Path],
    manifest: Union[str, Path, Dict[str, Any]],
    *,
    nlp: Any,
    crossing: str = "shrink",
    conflicts: str = "keep_first",
    alignment: str = "contract",
    on_error: str = "skip",
    store_user_data: bool = True,
) -> Dict[str, Any]:
    """Rewrite a DocBin using a JSON manifest of per-document operations.

    The manifest maps a document's position (as a string key, ``"0"`` for the
    first document) to a list of operations:
    ``{"0": [[10, 20, "[NAME]"]], "1": []}``. Documents without an entry are
    copied through unchanged.
    """
    if isinstance(manifest, dict):
        data = manifest
    else:
        with open(manifest, encoding="utf-8") as file_:
            data = json.load(file_)
    operations_by_index: Dict[int, List[Any]] = {}
    for key, ops in data.items():
        operations_by_index[int(key)] = list(ops)

    counter = {"value": 0}

    def rewriter(doc: Doc, nlp_obj: Any) -> RewrittenDoc:
        doc_index = counter["value"]
        counter["value"] += 1
        operations = operations_by_index.get(doc_index, [])
        return rewrite_doc(
            doc,
            operations,
            nlp=nlp_obj,
            crossing=crossing,
            conflicts=conflicts,
            alignment=alignment,
            on_error=on_error,
        )

    return rewrite_docbin(
        source,
        target,
        rewriter,
        nlp=nlp,
        store_user_data=store_user_data,
    )
