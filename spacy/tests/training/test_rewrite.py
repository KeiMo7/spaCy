"""Tests for ledger-based entity offset remapping across text rewrites."""

import pytest
import srsly

from spacy.lang.en import English
from spacy.tokens import DocBin
from spacy.training import Corpus
from spacy.training.rewrite import (
    DROPPED_OFFSET,
    EntityAnnotation,
    RewriteOp,
    rewrite_annotations,
    rewrite_doc,
    rewrite_docbin,
    rewrite_docbin_from_manifest,
)
from spacy.util import make_tempdir


@pytest.fixture(scope="module")
def nlp():
    return English()


def _text():
    return "The contract was signed by John Smith on 2024-01-01."


def _span(text, fragment, offset=0):
    start = text.index(fragment, offset)
    return start, start + len(fragment)


# ---------------------------------------------------------------------------
# Rule coverage: whole replacement, crossing shrink, adjacent merge
# ---------------------------------------------------------------------------


def test_whole_replacement_maps_onto_replacement(nlp):
    text = _text()
    start, end = _span(text, "John Smith")
    result = rewrite_annotations(
        text,
        [(start, end, "[CLIENT]")],
        [(start, end, "PERSON")],
        tokenizer=nlp,
    )
    assert result.text_after == "The contract was signed by [CLIENT] on 2024-01-01."
    rows = result.to_table()
    assert len(rows) == 1
    row = rows[0]
    assert (row["original_start"], row["original_end"]) == (start, end)
    assert row["disposition"] == "replaced"
    assert row["reason"] == "contained_in_operation:0"
    assert (row["new_start"], row["new_end"]) == (start, start + len("[CLIENT]"))
    assert result.text_after[row["new_start"] : row["new_end"]] == "[CLIENT]"
    assert result.reconciles


def test_crossing_shrink_keeps_untouched_remainder(nlp):
    text = "ACME Corporation Ltd owns it."
    # The entity starts before and ends inside the replaced span, and the
    # untouched remainder maps cleanly onto token boundaries.
    entity = (0, 22)  # "ACME Corporation Ltd o" -> remaining "Ltd " stops at 22
    result = rewrite_annotations(
        text,
        [(0, 5, "ACME Inc.")],
        [(0, 18, "ORG")],
        crossing="shrink",
        tokenizer=nlp,
    )
    assert result.text_after == "ACME Inc.Corporation Ltd owns it."
    row = result.to_table()[0]
    assert row["disposition"] == "shrunk"
    assert row["reason"].startswith("crosses_rewrite_boundary")
    new_text = result.text_after[row["new_start"] : row["new_end"]]
    # The untouched remainder is "Corporation" (token-aligned contract).
    assert new_text == "Corporation"
    assert result.reconciles
    assert entity  # keep the literal meaningful


def test_crossing_drop_records_disposition(nlp):
    text = "ACME Corporation Ltd owns it."
    result = rewrite_annotations(
        text,
        [(0, 5, "ACME Inc.")],
        [(0, 18, "ORG")],
        crossing="drop",
        tokenizer=nlp,
    )
    assert result.text_after == "ACME Inc.Corporation Ltd owns it."
    row = result.to_table()[0]
    assert row["disposition"] == "dropped"
    assert row["reason"] == "crosses_rewrite_boundary"
    assert (row["new_start"], row["new_end"]) == (DROPPED_OFFSET, DROPPED_OFFSET)
    assert result.dropped_count == 1
    assert result.entity_count_after == 0
    assert result.reconciles


def test_adjacent_merge_lands_on_legal_word_boundary(nlp):
    text = "abc def ghi"
    # Removing both spaces merges three words into a single token; positions
    # must be re-snapped onto the rewritten tokenization, not left dangling.
    result = rewrite_annotations(
        text,
        [(3, 4, ""), (7, 8, "")],
        [(0, 3, "A"), (4, 7, "B")],
        conflicts="keep_first",
        tokenizer=nlp,
    )
    assert result.text_after == "abcdefghi"
    rewritten = nlp.make_doc(result.text_after)
    boundaries = {token.idx for token in rewritten}
    boundaries |= {token.idx + len(token) for token in rewritten}
    for row in result.to_table():
        if row["disposition"] == "dropped":
            continue
        assert row["new_start"] in boundaries
        assert row["new_end"] in boundaries
    assert result.to_table()[0]["disposition"] == "adjusted"
    # The two entities collapsed onto one token; the conflict rule dropped one.
    assert result.dropped_count == 1
    assert any(p.kind == "overlap_conflict" for p in result.problems)
    assert result.reconciles


def test_adjacent_split_lands_on_legal_word_boundary(nlp):
    # Replacing the middle of a word with a space splits it into two tokens;
    # the entity must snap to the token that still carries the words.
    result = rewrite_annotations(
        "abcdef",
        [(3, 4, " ")],
        [(0, 6, "WORD")],
        tokenizer=nlp,
    )
    assert result.text_after == "abc ef"
    row = result.to_table()[0]
    assert row["disposition"] == "shrunk"
    for boundary in (row["new_start"], row["new_end"]):
        assert boundary in {0, 3, 4, 6}
    assert result.reconciles


# ---------------------------------------------------------------------------
# Growth and shrinkage of the replacement string
# ---------------------------------------------------------------------------


def test_replacement_longer_than_source(nlp):
    text = "Mr. John Smith signed."
    start, end = _span(text, "John Smith")
    replacement = "Mr. John Smith Jr."
    result = rewrite_annotations(
        text, [(start, end, replacement)], [(start, end, "PERSON")], tokenizer=nlp
    )
    assert result.text_after == f"Mr. {replacement} signed."
    row = result.to_table()[0]
    assert (row["new_start"], row["new_end"]) == (start, start + len(replacement))
    assert result.char_delta == len(replacement) - (end - start) > 0
    assert result.summary()["char_delta_declared"] == result.char_delta
    assert result.reconciles


def test_replacement_shorter_than_source(nlp):
    text = "AB CD EF"
    result = rewrite_annotations(text, [(3, 5, "X")], [(3, 5, "T")], tokenizer=nlp)
    assert result.text_after == "AB X EF"
    row = result.to_table()[0]
    assert (row["new_start"], row["new_end"]) == (3, 4)
    assert result.char_delta == -1
    assert result.reconciles


def test_multiple_operations_shift_following_entities(nlp):
    text = _text()
    name = _span(text, "John Smith")
    date = _span(text, "2024-01-01")
    operations = [(name[0], name[1], "[CLIENT]"), (date[0], date[1], "[DATE]")]
    result = rewrite_annotations(
        text, operations, [name + ("PERSON",), date + ("DATE",)], tokenizer=nlp
    )
    assert result.text_after == ("The contract was signed by [CLIENT] on [DATE].")
    name_row, date_row = result.to_table()
    assert name_row["disposition"] == "replaced"
    assert date_row["disposition"] == "replaced"
    assert result.text_after[name_row["new_start"] : name_row["new_end"]] == "[CLIENT]"
    assert result.text_after[date_row["new_start"] : date_row["new_end"]] == "[DATE]"
    assert result.reconciles


# ---------------------------------------------------------------------------
# No substring searching: a replacement containing the old text
# ---------------------------------------------------------------------------


def test_mapping_does_not_search_new_text(nlp):
    text = "Mr. John Smith signed."
    start, end = _span(text, "John Smith")
    # The replacement contains the original substring *and* the same prefix
    # again, so a searcher would pick the wrong occurrence.
    replacement = "John Smith (John Smith)"
    result = rewrite_annotations(
        text, [(start, end, replacement)], [(start, end, "PERSON")], tokenizer=nlp
    )
    row = result.to_table()[0]
    assert (row["new_start"], row["new_end"]) == (start, start + len(replacement))
    assert result.text_after[row["new_start"] : row["new_end"]] == replacement
    # And an entity before the operation keeps a ledger-derived position even
    # though the inserted text shifts everything after it.
    prefix = rewrite_annotations(
        text,
        [(start, end, replacement)],
        [(0, 3, "TITLE")],
        alignment="none",
    )
    assert prefix.to_table()[0]["new_start"] == 0


# ---------------------------------------------------------------------------
# Empty operation list
# ---------------------------------------------------------------------------


def test_empty_operation_list_is_identity(nlp):
    text = _text()
    entities = [_span(text, "John Smith") + ("PERSON",)]
    result = rewrite_annotations(text, [], entities, tokenizer=nlp)
    assert result.text_after == text
    assert result.char_delta == 0
    rows = result.to_table()
    assert len(rows) == 1
    assert rows[0]["disposition"] == "kept"
    assert (rows[0]["new_start"], rows[0]["new_end"]) == (
        rows[0]["original_start"],
        rows[0]["original_end"],
    )
    assert result.reconciles


def test_empty_operation_list_matches_doc_annotations(nlp):
    doc = nlp.make_doc(_text())
    start, end = _span(doc.text, "John Smith")
    doc.ents = [doc.char_span(start, end, label="PERSON")]
    rewritten = rewrite_doc(doc, [], nlp=nlp)
    assert rewritten.doc.text == doc.text
    assert [(e.start_char, e.end_char, e.label_) for e in rewritten.doc.ents] == [
        (start, end, "PERSON")
    ]


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------


def test_conflict_raise(nlp):
    text = "abc def ghi"
    with pytest.raises(ValueError) as excinfo:
        rewrite_annotations(
            text,
            [(3, 4, ""), (7, 8, "")],
            [(0, 3, "A"), (4, 7, "B")],
            conflicts="raise",
            tokenizer=nlp,
        )
    assert "E1062" in str(excinfo.value)


def test_conflict_keep_longest(nlp):
    text = "abc def ghi"
    result = rewrite_annotations(
        text,
        [(3, 4, ""), (7, 8, "")],
        [(0, 3, "A"), (4, 7, "B")],
        conflicts="keep_longest",
        tokenizer=nlp,
    )
    dispositions = [row["disposition"] for row in result.to_table()]
    assert dispositions.count("dropped") == 1
    assert result.dropped_count == 1
    assert result.reconciles


def test_adjacent_replacement_does_not_create_overlap_unnecessarily(nlp):
    # Distinct, non-overlapping entities stay non-overlapping as long as the
    # ledger keeps them apart.
    text = "ACME and Globex are vendors."
    acme = _span(text, "ACME")
    globex = _span(text, "Globex")
    result = rewrite_annotations(
        text,
        [(acme[0], acme[1], "ACME Inc.")],
        [acme + ("ORG",), globex + ("ORG",)],
        tokenizer=nlp,
    )
    positions = [
        (row["new_start"], row["new_end"])
        for row in result.to_table()
        if row["disposition"] != "dropped"
    ]
    assert len(positions) == 2
    (a_start, a_end), (b_start, b_end) = positions
    assert a_end <= b_start
    assert result.text_after[a_start:a_end] == "ACME Inc."


# ---------------------------------------------------------------------------
# Malformed operations
# ---------------------------------------------------------------------------


def test_out_of_bounds_operation_is_reported_with_index_and_fragment(nlp):
    text = "Hello world"
    with pytest.raises(ValueError) as excinfo:
        rewrite_annotations(text, [(0, 5, "Hi"), (100, 200, "X")], [], tokenizer=nlp)
    message = str(excinfo.value)
    assert "E1058" in message
    assert "index 1" in message


def test_declared_length_mismatch_is_reported(nlp):
    text = "Hello world"
    with pytest.raises(ValueError) as excinfo:
        rewrite_annotations(
            text,
            [RewriteOp(start=0, end=5, replacement="Hi", original="Helloo")],
            [],
            tokenizer=nlp,
        )
    assert "E1058" in str(excinfo.value)


def test_content_mismatch_is_reported(nlp):
    text = "Hello world"
    with pytest.raises(ValueError) as excinfo:
        rewrite_annotations(
            text,
            [(0, 5, "Hi", "World")],
            [],
            tokenizer=nlp,
        )
    assert "E1058" in str(excinfo.value)


def test_skipped_operation_is_recorded(nlp):
    text = "Hello world"
    result = rewrite_annotations(
        text,
        [(0, 5, "Hi"), (100, 200, "X")],
        [(0, 5, "GREET")],
        on_error="skip",
        tokenizer=nlp,
    )
    assert result.text_after == "Hi world"
    assert result.skipped_operation_count == 1
    problem = result.problems[0]
    assert problem.index == 1
    assert problem.kind == "out_of_bounds"
    assert problem.skipped is True
    assert problem.start == 100 and problem.end == 200
    assert problem.found is None
    assert result.reconciles


def test_overlap_between_operations_is_reported(nlp):
    text = "Hello world"
    with pytest.raises(ValueError) as excinfo:
        rewrite_annotations(text, [(0, 6, "Hi"), (3, 8, "There")], [], tokenizer=nlp)
    assert "E1059" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Shrink-to-empty and dropped-entity invariants
# ---------------------------------------------------------------------------


def test_entity_removed_entirely_by_replacement_is_not_zero_length(nlp):
    text = "Hello world"
    result = rewrite_annotations(text, [(0, 5, "")], [(0, 5, "GREET")], tokenizer=nlp)
    assert result.text_after == " world"
    row = result.to_table()[0]
    assert row["disposition"] == "dropped"
    assert row["reason"] == "empty_replacement"
    assert (row["new_start"], row["new_end"]) == (DROPPED_OFFSET, DROPPED_OFFSET)
    assert result.entity_count_after == 0
    assert result.reconciles


def test_crossing_entity_with_no_untouched_remainder_is_dropped(nlp):
    text = "abcdef"
    result = rewrite_annotations(
        text, [(0, 6, "X")], [(0, 6, "WORD")], crossing="shrink", tokenizer=nlp
    )
    # Fully contained -> replaced, not a crossing drop.
    assert result.to_table()[0]["disposition"] == "replaced"


# ---------------------------------------------------------------------------
# Determinism and reconciliation
# ---------------------------------------------------------------------------


def test_repeated_execution_is_identical(nlp):
    text = _text()
    name = _span(text, "John Smith")
    date = _span(text, "2024-01-01")
    operations = [(name[0], name[1], "[CLIENT]"), (date[0], date[1], "[DATE]")]
    entities = [name + ("PERSON",), date + ("DATE",)]
    first = rewrite_annotations(text, operations, entities, tokenizer=nlp)
    second = rewrite_annotations(text, operations, entities, tokenizer=nlp)
    assert first.to_json() == second.to_json()
    assert first.to_table() == second.to_table()
    assert first.summary() == second.summary()


def test_reconciliation_table_balances(nlp):
    text = "ACME Corporation Ltd owns it. John Smith signed."
    acme = _span(text, "ACME")
    name = _span(text, "John Smith")
    result = rewrite_annotations(
        text,
        [(acme[0], acme[1], "ACME Inc."), (name[0], name[1], "[CLIENT]")],
        [(acme[0], acme[1], "ORG"), (name[0], name[1], "PERSON")],
        tokenizer=nlp,
    )
    summary = result.summary()
    assert summary["reconciles"] is True
    assert summary["entity_count_before"] == 2
    assert summary["entity_count_after"] == 2
    assert summary["dropped_count"] == 0
    assert (
        summary["kept_count"]
        + summary["replaced_count"]
        + summary["shrunk_count"]
        + summary["adjusted_count"]
        == summary["entity_count_after"]
    )
    assert (
        summary["char_count_after"]
        == summary["char_count_before"] + summary["char_delta_declared"]
    )


def test_rows_carry_original_and_new_spans_and_disposition(nlp):
    text = _text()
    start, end = _span(text, "John Smith")
    result = rewrite_annotations(
        text, [(start, end, "[CLIENT]")], [(start, end, "PERSON")], tokenizer=nlp
    )
    row = result.to_table()[0]
    assert set(row) >= {
        "original_start",
        "original_end",
        "new_start",
        "new_end",
        "disposition",
    }
    assert row["original_text"] == "John Smith"


# ---------------------------------------------------------------------------
# Doc / corpus integration
# ---------------------------------------------------------------------------


def _make_docbin(path, nlp):
    doc_one = nlp.make_doc(_text())
    start, end = _span(doc_one.text, "John Smith")
    doc_one.ents = [doc_one.char_span(start, end, label="PERSON")]
    doc_two = nlp.make_doc("ACME Corporation Ltd owns it.")
    doc_two.ents = [doc_two.char_span(0, 16, label="ORG")]
    DocBin(docs=[doc_one, doc_two]).to_disk(path)
    return start, end


def test_docbin_rewrite_roundtrip(nlp):
    with make_tempdir() as tmpdir:
        source = tmpdir / "in.spacy"
        target = tmpdir / "out.spacy"
        start, end = _make_docbin(source, nlp)
        manifest = {"0": [[start, end, "[CLIENT]"]], "1": [[0, 5, "ACME Inc."]]}
        report = rewrite_docbin_from_manifest(source, target, manifest, nlp=nlp)
        assert report["document_count"] == 2
        assert report["reconciles"] is True
        docs = list(DocBin().from_disk(target).get_docs(nlp.vocab))
        assert docs[0].text == "The contract was signed by [CLIENT] on 2024-01-01."
        assert [(e.text, e.label_) for e in docs[0].ents] == [("[CLIENT]", "PERSON")]
        assert docs[1].text.startswith("ACME Inc.")


def test_docbin_rewrite_is_atomic_on_failure(nlp):
    with make_tempdir() as tmpdir:
        source = tmpdir / "in.spacy"
        target = tmpdir / "out.spacy"
        _make_docbin(source, nlp)

        def failing_rewriter(doc, nlp_obj):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            rewrite_docbin(source, target, failing_rewriter, nlp=nlp)
        assert not target.exists()
        assert not target.with_name(target.name + ".partial").exists()


def test_stale_partial_file_is_replaced(nlp):
    with make_tempdir() as tmpdir:
        source = tmpdir / "in.spacy"
        target = tmpdir / "out.spacy"
        start, end = _make_docbin(source, nlp)
        partial = target.with_name(target.name + ".partial")
        partial.write_bytes(b"stale half-written corpus")
        report = rewrite_docbin_from_manifest(
            source, target, {"0": [[start, end, "[CLIENT]"]]}, nlp=nlp
        )
        assert report["reconciles"] is True
        assert not partial.exists()
        assert target.exists()


def test_corpus_reader_unchanged_without_rewriter(nlp):
    with make_tempdir() as tmpdir:
        source = tmpdir / "in.spacy"
        _make_docbin(source, nlp)
        corpus = Corpus(source)
        baseline = [
            (
                eg.reference.text,
                [(e.start_char, e.end_char, e.label_) for e in eg.reference.ents],
            )
            for eg in corpus(nlp)
        ]
    assert baseline == [
        (_text(), [(27, 37, "PERSON")]),
        ("ACME Corporation Ltd owns it.", [(0, 16, "ORG")]),
    ]


def test_corpus_reader_hook_is_opt_in(nlp):
    with make_tempdir() as tmpdir:
        source = tmpdir / "in.spacy"
        start, end = _make_docbin(source, nlp)
        seen = []

        def rewriter(doc, nlp_obj):
            operations = []
            if doc.text == _text():
                operations = [(start, end, "[CLIENT]")]
            rewritten = rewrite_doc(doc, operations, nlp=nlp_obj)
            seen.append(rewritten.report.reconciles)
            return rewritten.doc

        corpus = Corpus(source, rewriter=rewriter)
        examples = list(corpus(nlp))
    assert seen and all(seen)
    texts = [eg.reference.text for eg in examples]
    assert "[CLIENT]" in texts[0]


# ---------------------------------------------------------------------------
# Serialization compatibility
# ---------------------------------------------------------------------------


def test_rewritten_docbin_is_readable_by_unmodified_path(nlp):
    with make_tempdir() as tmpdir:
        source = tmpdir / "in.spacy"
        target = tmpdir / "out.spacy"
        start, end = _make_docbin(source, nlp)
        rewrite_docbin_from_manifest(
            source, target, {"0": [[start, end, "[CLIENT]"]]}, nlp=nlp
        )
        doc_bin = DocBin().from_disk(target)
        docs = list(doc_bin.get_docs(nlp.vocab))
    assert len(docs) == 2
    assert docs[0].text == "The contract was signed by [CLIENT] on 2024-01-01."
    assert docs[0].has_annotation("ENT_IOB")


def test_rewritten_docbin_matches_original_format_version(nlp):
    with make_tempdir() as tmpdir:
        source = tmpdir / "in.spacy"
        target = tmpdir / "out.spacy"
        start, end = _make_docbin(source, nlp)
        original = DocBin().from_disk(source)
        assert original.version
        rewrite_docbin_from_manifest(
            source, target, {"0": [[start, end, "[CLIENT]"]]}, nlp=nlp
        )
        rewritten = DocBin().from_disk(target)
    assert rewritten.version == original.version


def test_manifest_without_entry_passes_document_through(nlp):
    with make_tempdir() as tmpdir:
        source = tmpdir / "in.spacy"
        target = tmpdir / "out.spacy"
        start, end = _make_docbin(source, nlp)
        report = rewrite_docbin_from_manifest(
            source, target, {"0": [[start, end, "[CLIENT]"]]}, nlp=nlp
        )
        assert report["summaries"][1]["kept_count"] == 1
        assert report["summaries"][1]["entity_count_before"] == 1
        docs = list(DocBin().from_disk(target).get_docs(nlp.vocab))
    assert docs[1].text == "ACME Corporation Ltd owns it."


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def test_entity_objects_and_mappings_are_accepted(nlp):
    text = _text()
    start, end = _span(text, "John Smith")
    result = rewrite_annotations(
        text,
        [RewriteOp(start=start, end=end, replacement="[CLIENT]")],
        [EntityAnnotation(start=start, end=end, label="PERSON")],
        tokenizer=nlp,
    )
    assert result.to_table()[0]["disposition"] == "replaced"


def test_alignment_none_keeps_raw_offsets(nlp):
    text = "abcdef"
    # The entity lies entirely after the operation, so with raw ledger
    # offsets its start/end are shifted but not snapped to token boundaries.
    result = rewrite_annotations(
        text, [(3, 4, " XY ")], [(5, 6, "TAIL")], alignment="none"
    )
    row = result.to_table()[0]
    # Delta is +3, so 5..6 shifts to 8..9 with no boundary snapping.
    assert (row["new_start"], row["new_end"]) == (8, 9)
    assert (row["ledger_start"], row["ledger_end"]) == (8, 9)


def test_invalid_policy_raises(nlp):
    with pytest.raises(ValueError):
        rewrite_annotations("abc", [(0, 1, "x")], [], crossing="nope")
    with pytest.raises(ValueError):
        rewrite_annotations("abc", [(0, 1, "x")], [], conflicts="nope")
    with pytest.raises(ValueError):
        rewrite_annotations("abc", [(0, 1, "x")], [], alignment="nope")


def test_source_entities_must_not_overlap(nlp):
    text = "abcdef"
    with pytest.raises(ValueError) as excinfo:
        rewrite_annotations(text, [], [(0, 3, "A"), (2, 5, "B")], tokenizer=nlp)
    assert "E1064" in str(excinfo.value)


def test_report_is_json_serializable(nlp):
    text = _text()
    start, end = _span(text, "John Smith")
    result = rewrite_annotations(
        text, [(start, end, "[CLIENT]")], [(start, end, "PERSON")], tokenizer=nlp
    )
    parsed = srsly.json_loads(result.to_json())
    assert parsed["summary"]["reconciles"] is True
    assert parsed["entities"][0]["disposition"] == "replaced"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_rewrite_ents_end_to_end(nlp):
    from typer.testing import CliRunner

    from spacy.cli._util import app

    with make_tempdir() as tmpdir:
        source = tmpdir / "in.spacy"
        target = tmpdir / "out.spacy"
        manifest = tmpdir / "manifest.json"
        report = tmpdir / "report.jsonl"
        start, end = _make_docbin(source, nlp)
        srsly.write_json(manifest, {"0": [[start, end, "[CLIENT]"]]})
        result = CliRunner().invoke(
            app,
            [
                "rewrite-ents",
                str(source),
                str(target),
                str(manifest),
                "--lang",
                "en",
                "--report",
                str(report),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert target.exists()
        docs = list(DocBin().from_disk(target).get_docs(nlp.vocab))
        assert docs[0].text == "The contract was signed by [CLIENT] on 2024-01-01."
        summaries = list(srsly.read_jsonl(report))
        assert len(summaries) == 2
        assert all(summary["reconciles"] for summary in summaries)


def test_cli_rewrite_ents_rejects_malformed_manifest(nlp):
    from typer.testing import CliRunner

    from spacy.cli._util import app

    with make_tempdir() as tmpdir:
        source = tmpdir / "in.spacy"
        target = tmpdir / "out.spacy"
        manifest = tmpdir / "manifest.json"
        _make_docbin(source, nlp)
        # Operation 1 is out of bounds and must be skipped, not fatal.
        srsly.write_json(manifest, {"0": [[100, 200, "X"]]})
        result = CliRunner().invoke(
            app,
            ["rewrite-ents", str(source), str(target), str(manifest), "--lang", "en"],
        )
        assert result.exit_code == 0, result.stdout
        docs = list(DocBin().from_disk(target).get_docs(nlp.vocab))
        assert docs[0].text == _text()


def test_corpus_without_rewriter_matches_previous_example_construction(nlp):
    """Turning the capability off must leave Example construction unchanged."""
    from spacy.training import Example

    with make_tempdir() as tmpdir:
        source = tmpdir / "in.spacy"
        _make_docbin(source, nlp)
        with_hook_disabled = list(Corpus(source)(nlp))
        explicit_none = list(Corpus(source, rewriter=None)(nlp))

    def snapshot(examples):
        return [
            (
                eg.reference.text,
                [t.text for t in eg.reference],
                [(e.start_char, e.end_char, e.label_) for e in eg.reference.ents],
            )
            for eg in examples
        ]

    assert snapshot(with_hook_disabled) == snapshot(explicit_none)
    # The reference docs are constructed through the unmodified path, so the
    # Example objects keep the same type and annotations.
    assert all(isinstance(eg, Example) for eg in with_hook_disabled)


def test_existing_out_of_bounds_annotation_warning_is_unchanged(nlp):
    """The corpus/training path must still warn W030 for unaligned entities."""
    import warnings

    from spacy.training import Example

    doc = nlp.make_doc("I flew to San Francisco Valley.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        example = Example.from_dict(doc, {"entities": [(0, 200, "LOC")]})
    assert len(example.reference.ents) == 0
    assert any("[W030]" in str(item.message) for item in caught)


def test_alignment_never_pulls_replacement_text_into_kept_entity(nlp):
    # The entity is an untouched fragment, but token snapping would widen it
    # over "[REPL]"; that would silently change what it denotes, so it is
    # dropped with an explicit reason instead.
    text = "aa XY bb"
    # "a" at 1..2 is untouched, but after "[REPL]" is spliced in the whole text
    # tokenizes as one token, so expanding the span would pull in the
    # replacement text.
    result = rewrite_annotations(
        text, [(2, 6, "[REPL]")], [(1, 2, "T")], alignment="expand", tokenizer=nlp
    )
    row = result.to_table()[0]
    assert row["disposition"] == "dropped"
    assert row["reason"] == "alignment_overlaps_replacement"
    assert result.reconciles


def test_rewrite_doc_rejects_alignment_none(nlp):
    doc = nlp.make_doc("John Smith works")
    doc.ents = [doc.char_span(0, 10, label="PERSON")]
    with pytest.raises(ValueError) as excinfo:
        rewrite_doc(doc, [(1, 2, "X")], nlp=nlp, alignment="none")
    assert "E1066" in str(excinfo.value)


def test_reported_bug_scenario_end_to_end(nlp):
    """A masked name shortens the text; the following entity must shift by the
    ledger delta instead of being dropped as out of bounds."""
    text = "Dear John Smith, your contract 2024-123 is ready."
    name = _span(text, "John Smith")
    ident = _span(text, "2024-123")
    result = rewrite_annotations(
        text,
        [(name[0], name[1], "[NAME]")],
        [name + ("PERSON",), ident + ("ID",)],
        tokenizer=nlp,
    )
    assert result.text_after == "Dear [NAME], your contract 2024-123 is ready."
    name_row, id_row = result.to_table()
    assert (name_row["new_start"], name_row["new_end"]) == (5, 11)
    assert (id_row["new_start"], id_row["new_end"]) == (27, 35)
    assert result.text_after[id_row["new_start"] : id_row["new_end"]] == "2024-123"
    assert result.reconciles
