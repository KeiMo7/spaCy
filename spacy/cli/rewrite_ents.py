from pathlib import Path
from typing import Optional

import srsly
import typer
from wasabi import Printer

from ..training.rewrite import (
    ALIGNMENT_MODES,
    CONFLICT_POLICIES,
    CROSSING_POLICIES,
    ERROR_POLICIES,
    rewrite_docbin_from_manifest,
)
from ..util import get_lang_class
from ._util import Arg, Opt, app

# fmt: off
_reports_help = (
    "Write a per-document reconciliation report (JSONL) to this path. "
    "Defaults to no report."
)
# fmt: on


@app.command("rewrite-ents")
def rewrite_ents_cli(
    # fmt: off
    source: Path = Arg(..., help="Corpus in DocBin (.spacy) format", exists=True),
    target: Path = Arg(..., help="Output DocBin (.spacy) path"),
    manifest: Path = Arg(
        ...,
        help="JSON manifest mapping document index to rewrite operations",
        exists=True,
    ),
    lang: str = Opt("xx", "--lang", "-l", help="Language for tokenization"),
    crossing: str = Opt(
        "shrink", "--crossing", help=f"Crossing policy: {', '.join(CROSSING_POLICIES)}"
    ),
    conflicts: str = Opt(
        "keep_first",
        "--conflicts",
        help=f"Conflict policy: {', '.join(CONFLICT_POLICIES)}",
    ),
    alignment: str = Opt(
        "contract",
        "--alignment",
        help=f"Alignment mode: {', '.join(ALIGNMENT_MODES)}",
    ),
    on_error: str = Opt(
        "skip",
        "--on-error",
        help=f"Malformed operation policy: {', '.join(ERROR_POLICIES)}",
    ),
    report_path: Optional[Path] = Opt(None, "--report", "-r", help=_reports_help),
    # fmt: on
):
    """
    Rewrite entity offsets in a DocBin corpus using a manifest of character
    rewrites. The rewrite ledger is the only source of the new positions; the
    new text is never searched for the old fragments.

    The manifest maps a document index (as a string key, ``"0"`` for the first
    document) to a list of ``[start, end, replacement]`` operations:

    \b
    {"0": [[27, 37, "[CLIENT]"]], "1": [[0, 5, "ACME Inc."]]}

    The output is written atomically: a failed run never leaves a half-written
    corpus behind, and a stale ``.partial`` file is replaced on the next run.

    DOCS: https://spacy.io/api/cli#rewrite-ents
    """
    msg = Printer()
    nlp = get_lang_class(lang)()
    manifest_data = srsly.read_json(manifest)
    result = rewrite_docbin_from_manifest(
        source,
        target,
        manifest_data,
        nlp=nlp,
        crossing=crossing,
        conflicts=conflicts,
        alignment=alignment,
        on_error=on_error,
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        srsly.write_jsonl(report_path, result["summaries"])
    for index, summary in enumerate(result["summaries"]):
        if not summary["reconciles"]:
            msg.fail(
                f"Document {index} does not reconcile: "
                f"entities {summary['entity_count_before']} -> "
                f"{summary['entity_count_after']} + "
                f"{summary['dropped_count']} dropped"
            )
            raise typer.Exit(code=1)
    msg.good(
        f"Rewrote {result['document_count']} document(s) from {source} to {target}"
    )
