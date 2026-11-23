"""Generate reports/ artifacts for the rewrite-offset capability."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

from spacy.lang.en import English  # noqa: E402
from spacy.tokens import DocBin  # noqa: E402
from spacy.training.rewrite import (  # noqa: E402
    rewrite_annotations,
    rewrite_docbin_from_manifest,
)

nlp = English()


def span(text, fragment):
    i = text.index(fragment)
    return i, i + len(fragment)


def dump(name, payload):
    (REPORTS / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


# --- Sample A: whole replacement, shorter replacement -----------------------
text_a = "The contract was signed by John Smith on 2024-01-01."
name_a = span(text_a, "John Smith")
res_a = rewrite_annotations(
    text_a, [(name_a[0], name_a[1], "[CLIENT]")], [name_a + ("PERSON",)], tokenizer=nlp
)

# --- Sample B: crossing shrink and drop -------------------------------------
text_b = "ACME Corporation Ltd owns it."
res_b_shrink = rewrite_annotations(
    text_b, [(0, 5, "ACME Inc.")], [(0, 18, "ORG")], crossing="shrink", tokenizer=nlp
)
res_b_drop = rewrite_annotations(
    text_b, [(0, 5, "ACME Inc.")], [(0, 18, "ORG")], crossing="drop", tokenizer=nlp
)

# --- Sample C: adjacent replacements merging words --------------------------
text_c = "abc def ghi"
res_c = rewrite_annotations(
    text_c,
    [(3, 4, ""), (7, 8, "")],
    [(0, 3, "A"), (4, 7, "B")],
    conflicts="keep_first",
    tokenizer=nlp,
)

# --- Sample D: replacement grows, multiple operations -----------------------
text_d = "John Smith signed it on 2024-01-01 in Paris."
name_d = span(text_d, "John Smith")
date_d = span(text_d, "2024-01-01")
res_d = rewrite_annotations(
    text_d,
    [(name_d[0], name_d[1], "Mr. John Smith Jr."), (date_d[0], date_d[1], "[DATE]")],
    [name_d + ("PERSON",), date_d + ("DATE",)],
    tokenizer=nlp,
)

# --- Sample E: empty operation list -----------------------------------------
res_e = rewrite_annotations(text_a, [], [name_a + ("PERSON",)], tokenizer=nlp)

# --- Sample F: malformed operation (out of bounds, skipped) -----------------
res_f = rewrite_annotations(
    "Hello world",
    [(0, 5, "Hi"), (100, 200, "X")],
    [(0, 5, "GREET")],
    on_error="skip",
    tokenizer=nlp,
)

samples = {
    "whole_replacement": res_a,
    "crossing_shrink": res_b_shrink,
    "crossing_drop": res_b_drop,
    "adjacent_merge": res_c,
    "grow_and_multiple": res_d,
    "empty_operations": res_e,
    "malformed_skip": res_f,
}

dump(
    "rewrite_samples.json",
    {name: result.to_dict() for name, result in samples.items()},
)

# --- Determinism: run the same input twice ----------------------------------
repeat = [
    rewrite_annotations(
        text_d,
        [(name_d[0], name_d[1], "X"), (date_d[0], date_d[1], "Y")],
        [name_d + ("PERSON",), date_d + ("DATE",)],
        tokenizer=nlp,
    ).to_json()
    for _ in range(5)
]
dump("determinism.json", {"runs": len(repeat), "identical": len(set(repeat)) == 1})

# --- Corpus-level round trip + reconciliation -------------------------------
corpus_dir = REPORTS / "corpus"
if corpus_dir.exists():
    shutil.rmtree(corpus_dir)
corpus_dir.mkdir()
doc_one = nlp.make_doc(text_a)
s = text_a.index("John Smith")
doc_one.ents = [doc_one.char_span(s, s + len("John Smith"), label="PERSON")]
doc_two = nlp.make_doc(text_b)
doc_two.ents = [doc_two.char_span(0, 16, label="ORG")]
source = corpus_dir / "source.spacy"
target = corpus_dir / "rewritten.spacy"
DocBin(docs=[doc_one, doc_two]).to_disk(source)
manifest = {"0": [[s, s + len("John Smith"), "[CLIENT]"]], "1": [[0, 5, "ACME Inc."]]}
corpus_report = rewrite_docbin_from_manifest(source, target, manifest, nlp=nlp)
corpus_report["source"] = relative(source)
corpus_report["target"] = relative(target)
dump("corpus_rewrite.json", corpus_report)

docs = list(DocBin().from_disk(target).get_docs(nlp.vocab))
dump(
    "corpus_documents.json",
    [
        {
            "text": doc.text,
            "entities": [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
                for ent in doc.ents
            ],
        }
        for doc in docs
    ],
)


# --- Existing corpus read/write tests: capture actual output ----------------
def run_pytest(args):
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args, "-q"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
    )
    return {
        "command": "python -m pytest " + " ".join(args) + " -q",
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


existing = run_pytest(
    [
        "spacy/tests/training/test_corpus.py",
        "spacy/tests/training/test_readers.py",
        "spacy/tests/training/test_training.py",
        "spacy/tests/serialize/test_serialize_docbin.py",
        "spacy/tests/serialize/test_serialize_doc.py",
    ]
)
dump("existing_corpus_tests.json", existing)

new_tests = run_pytest(["spacy/tests/training/test_rewrite.py"])
dump("new_rewrite_tests.json", new_tests)

print("existing exit:", existing["exit_code"])
print(existing["stdout"].strip().splitlines()[-1])
print("new exit:", new_tests["exit_code"])
print(new_tests["stdout"].strip().splitlines()[-1])
