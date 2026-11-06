"""Traditional non-neural baseline: binary BIO CRF + TF-IDF Linear SVM.

The CRF follows the sklearn-crfsuite NER tutorial pattern: token-level feature
dicts, BIO labels, and L-BFGS training. The SVM follows the standard
scikit-learn text-classification pipeline: TF-IDF features plus a linear SVM.
Predictions use the same JSONL span format as the neural baselines.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.common.utils import get_labels, load_yaml, read_jsonl, sort_spans, write_json, write_jsonl


TOKEN_RE = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]"
    r"|[^\W\d_]+(?:[-'][^\W\d_]+)*"
    r"|\d+(?:[.,]\d+)*"
    r"|[^\w\s]",
    re.UNICODE,
)


def missing_dependency_error(exc: ImportError) -> SystemExit:
    return SystemExit(
        "Missing traditional baseline dependencies. Install them with:\n"
        "  python -m pip install scikit-learn sklearn-crfsuite joblib tqdm\n"
    )


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def token_shape(token: str) -> str:
    out = []
    for ch in token:
        if ch.isupper():
            out.append("X")
        elif ch.islower():
            out.append("x")
        elif ch.isdigit():
            out.append("d")
        else:
            out.append(ch)
    return "".join(out)


def token_features(tokens: list[tuple[str, int, int]], index: int) -> dict[str, Any]:
    token = tokens[index][0]
    lower = token.lower()
    features: dict[str, Any] = {
        "bias": 1.0,
        "tok.lower": lower,
        "tok[-3:]": lower[-3:],
        "tok[-2:]": lower[-2:],
        "tok.isupper": token.isupper(),
        "tok.istitle": token.istitle(),
        "tok.isdigit": token.isdigit(),
        "tok.shape": token_shape(token),
        "tok.len": min(len(token), 20),
        "has_digit": any(ch.isdigit() for ch in token),
        "has_punct": any(not ch.isalnum() for ch in token),
    }
    if index > 0:
        prev = tokens[index - 1][0]
        features.update(
            {
                "-1:tok.lower": prev.lower(),
                "-1:tok.istitle": prev.istitle(),
                "-1:tok.isupper": prev.isupper(),
                "-1:tok.shape": token_shape(prev),
            }
        )
    else:
        features["BOS"] = True
    if index + 1 < len(tokens):
        nxt = tokens[index + 1][0]
        features.update(
            {
                "+1:tok.lower": nxt.lower(),
                "+1:tok.istitle": nxt.istitle(),
                "+1:tok.isupper": nxt.isupper(),
                "+1:tok.shape": token_shape(nxt),
            }
        )
    else:
        features["EOS"] = True
    return features


def merge_location_spans(spans: list[dict[str, Any]]) -> list[dict[str, int]]:
    intervals = []
    for span in spans:
        try:
            start = int(span["start"])
            end = int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < end:
            intervals.append((start, end))
    intervals.sort()
    merged: list[dict[str, int]] = []
    for start, end in intervals:
        if not merged or start >= merged[-1]["end"]:
            merged.append({"start": start, "end": end})
        else:
            merged[-1]["end"] = max(merged[-1]["end"], end)
    return merged


def bio_labels(tokens: list[tuple[str, int, int]], spans: list[dict[str, Any]]) -> list[str]:
    labels = ["O"] * len(tokens)
    for span in merge_location_spans(spans):
        overlapping = [
            i
            for i, (_, start, end) in enumerate(tokens)
            if max(start, int(span["start"])) < min(end, int(span["end"]))
        ]
        if not overlapping:
            continue
        labels[overlapping[0]] = "B-PROP"
        for i in overlapping[1:]:
            labels[i] = "I-PROP"
    return labels


def doc_to_crf_item(doc: dict[str, Any], with_labels: bool) -> tuple[list[dict[str, Any]], list[str], list[tuple[str, int, int]]]:
    tokens = tokenize_with_offsets(doc["text"])
    features = [token_features(tokens, i) for i in range(len(tokens))]
    labels = bio_labels(tokens, doc.get("spans", [])) if with_labels else []
    return features, labels, tokens


def tags_to_binary_spans(tags: list[str], tokens: list[tuple[str, int, int]]) -> list[dict[str, int]]:
    spans = []
    current: list[int] | None = None
    for tag, (_, start, end) in zip(tags, tokens):
        if tag == "B-PROP":
            if current is not None:
                spans.append({"start": current[0], "end": current[1]})
            current = [start, end]
        elif tag == "I-PROP":
            if current is None:
                current = [start, end]
            else:
                current[1] = end
        else:
            if current is not None:
                spans.append({"start": current[0], "end": current[1]})
                current = None
    if current is not None:
        spans.append({"start": current[0], "end": current[1]})
    return spans


def span_context_text(doc: dict[str, Any], span: dict[str, Any], context_chars: int) -> str:
    text = doc["text"]
    start = max(0, int(span["start"]) - context_chars)
    end = min(len(text), int(span["end"]) + context_chars)
    left = text[start : int(span["start"])]
    mid = text[int(span["start"]) : int(span["end"])]
    right = text[int(span["end"]) : end]
    return f"{left} <SPAN> {mid} </SPAN> {right}"


def train_crf(train_docs: list[dict[str, Any]], max_iterations: int, c1: float, c2: float, verbose: bool):
    try:
        import sklearn_crfsuite
    except ImportError as exc:  # pragma: no cover - environment-specific guidance.
        raise missing_dependency_error(exc) from exc

    started = time.perf_counter()
    x_train = []
    y_train = []
    for doc in tqdm(train_docs, desc="CRF feature extraction", dynamic_ncols=True):
        features, labels, _ = doc_to_crf_item(doc, with_labels=True)
        if features:
            x_train.append(features)
            y_train.append(labels)
    print(f"CRF feature extraction complete in {time.perf_counter() - started:.1f}s", flush=True)
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=True,
        verbose=verbose,
    )
    print(f"CRF fitting {len(x_train)} documents", flush=True)
    started = time.perf_counter()
    crf.fit(x_train, y_train)
    print(f"CRF fit complete in {time.perf_counter() - started:.1f}s", flush=True)
    return crf


def span_overlap_score(span: dict[str, Any], gold: dict[str, Any], mode: str) -> float:
    start = int(span["start"])
    end = int(span["end"])
    gold_start = int(gold["start"])
    gold_end = int(gold["end"])
    inter = max(0, min(end, gold_end) - max(start, gold_start))
    if inter <= 0:
        return 0.0
    if mode == "iou":
        union = max(end, gold_end) - min(start, gold_start)
        return inter / max(1, union)
    if mode == "gold_coverage":
        return inter / max(1, gold_end - gold_start)
    if mode == "candidate_coverage":
        return inter / max(1, end - start)
    if mode == "either_coverage":
        return max(inter / max(1, gold_end - gold_start), inter / max(1, end - start))
    raise ValueError(f"Unsupported overlap mode: {mode}")


def labels_for_candidate_span(
    candidate: dict[str, Any],
    gold_spans: list[dict[str, Any]],
    allowed: set[str],
    overlap_threshold: float,
    overlap_mode: str,
) -> tuple[str, ...]:
    labels = {
        str(span["technique"])
        for span in gold_spans
        if span.get("technique") in allowed
        and span_overlap_score(candidate, span, overlap_mode) >= overlap_threshold
    }
    return tuple(sorted(labels))


def add_svm_example(
    examples: list[tuple[str, tuple[str, ...]]],
    seen: set[tuple[str, int, int, tuple[str, ...]]],
    doc: dict[str, Any],
    span: dict[str, Any],
    label_set: tuple[str, ...],
    context_chars: int,
) -> None:
    if not label_set:
        return
    start = int(span["start"])
    end = int(span["end"])
    if not (0 <= start < end <= len(doc["text"])):
        return
    key = (str(doc["id"]), start, end, label_set)
    if key in seen:
        return
    seen.add(key)
    examples.append((span_context_text(doc, {"start": start, "end": end}, context_chars), label_set))


def iter_gold_label_spans(doc: dict[str, Any], allowed: set[str]) -> list[tuple[dict[str, int], tuple[str, ...]]]:
    by_boundary: dict[tuple[int, int], set[str]] = {}
    for span in doc.get("spans", []):
        technique = span.get("technique")
        if technique not in allowed:
            continue
        start = int(span["start"])
        end = int(span["end"])
        if start < end:
            by_boundary.setdefault((start, end), set()).add(str(technique))
    return [
        ({"start": start, "end": end}, tuple(sorted(label_set)))
        for (start, end), label_set in sorted(by_boundary.items())
    ]


def perturb_span(span: dict[str, int], text_len: int, radius: int) -> list[dict[str, int]]:
    if radius <= 0:
        return []
    out = []
    for start_delta in range(-radius, radius + 1):
        for end_delta in range(-radius, radius + 1):
            if start_delta == 0 and end_delta == 0:
                continue
            start = int(span["start"]) + start_delta
            end = int(span["end"]) + end_delta
            if 0 <= start < end <= text_len:
                out.append({"start": start, "end": end})
    return out


def crf_proposals_for_doc(doc: dict[str, Any], crf) -> list[dict[str, int]]:
    features, _, tokens = doc_to_crf_item(doc, with_labels=False)
    if not features:
        return []
    tags = crf.predict_single(features)
    return tags_to_binary_spans(tags, tokens)


def build_svm_training_examples(
    train_docs: list[dict[str, Any]],
    labels: list[str],
    context_chars: int,
    crf=None,
    include_gold_spans: bool = True,
    include_crf_proposals: bool = True,
    boundary_perturb_radius: int = 0,
    overlap_threshold: float = 0.5,
    overlap_mode: str = "gold_coverage",
    max_examples: int | None = None,
) -> tuple[list[str], list[tuple[str, ...]]]:
    allowed = set(labels)
    examples: list[tuple[str, tuple[str, ...]]] = []
    seen: set[tuple[str, int, int, tuple[str, ...]]] = set()
    for doc in tqdm(train_docs, desc="SVM span extraction", dynamic_ncols=True):
        gold_spans = [span for span in doc.get("spans", []) if span.get("technique") in allowed]
        if include_gold_spans:
            for span, label_set in iter_gold_label_spans(doc, allowed):
                add_svm_example(examples, seen, doc, span, label_set, context_chars)
                for noisy_span in perturb_span(span, len(doc["text"]), boundary_perturb_radius):
                    noisy_labels = labels_for_candidate_span(
                        noisy_span,
                        gold_spans,
                        allowed,
                        overlap_threshold,
                        overlap_mode,
                    )
                    add_svm_example(examples, seen, doc, noisy_span, noisy_labels, context_chars)
        if include_crf_proposals and crf is not None:
            for span in crf_proposals_for_doc(doc, crf):
                label_set = labels_for_candidate_span(span, gold_spans, allowed, overlap_threshold, overlap_mode)
                add_svm_example(examples, seen, doc, span, label_set, context_chars)
        if max_examples is not None and len(examples) >= max_examples:
            examples = examples[:max_examples]
            break
    return [text for text, _ in examples], [label_set for _, label_set in examples]


def train_svm(
    train_docs: list[dict[str, Any]],
    labels: list[str],
    context_chars: int,
    max_iter: int,
    crf=None,
    include_gold_spans: bool = True,
    include_crf_proposals: bool = True,
    boundary_perturb_radius: int = 0,
    overlap_threshold: float = 0.5,
    overlap_mode: str = "gold_coverage",
    max_examples: int | None = None,
):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.svm import LinearSVC
    except ImportError as exc:  # pragma: no cover - environment-specific guidance.
        raise missing_dependency_error(exc) from exc

    started = time.perf_counter()
    x_train, y_sets = build_svm_training_examples(
        train_docs,
        labels,
        context_chars,
        crf=crf,
        include_gold_spans=include_gold_spans,
        include_crf_proposals=include_crf_proposals,
        boundary_perturb_radius=boundary_perturb_radius,
        overlap_threshold=overlap_threshold,
        overlap_mode=overlap_mode,
        max_examples=max_examples,
    )
    if not x_train:
        raise ValueError("No gold spans available to train the SVM technique classifier.")
    print(f"SVM span extraction complete in {time.perf_counter() - started:.1f}s", flush=True)
    active_labels = [label for label in labels if any(label in label_set for label_set in y_sets)]
    if not active_labels:
        raise ValueError("No active technique labels available to train the SVM technique classifier.")
    print(
        f"SVM fitting {len(x_train)} span examples across {len(active_labels)} active labels "
        f"(include_crf_proposals={include_crf_proposals}, perturb_radius={boundary_perturb_radius})",
        flush=True,
    )

    word_tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    mlb = MultiLabelBinarizer(classes=active_labels)
    y_train = mlb.fit_transform(y_sets)
    clf = Pipeline(
        [
            ("features", FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])),
            ("svm", OneVsRestClassifier(LinearSVC(class_weight="balanced", max_iter=max_iter))),
        ]
    )
    print("SVM vectorizing spans and fitting OneVsRest LinearSVC; sklearn does not provide a tqdm bar here", flush=True)
    started = time.perf_counter()
    clf.fit(x_train, y_train)
    print(f"SVM fit complete in {time.perf_counter() - started:.1f}s", flush=True)
    return {"model": clf, "label_binarizer": mlb, "active_labels": active_labels, "multi_label": True}


def predict_svm_labels(svm_bundle, texts: list[str], threshold: float) -> list[tuple[str, ...]]:
    model = svm_bundle["model"] if isinstance(svm_bundle, dict) else svm_bundle
    if not isinstance(svm_bundle, dict):
        return [(str(label),) for label in model.predict(texts)]
    mlb = svm_bundle["label_binarizer"]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(texts)
        import numpy as np

        scores = np.asarray(scores)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        pred = scores >= threshold
    else:
        pred = model.predict(texts)
    out = []
    for row in mlb.inverse_transform(pred):
        out.append(tuple(str(label) for label in row))
    return out


def predict_docs(
    docs: list[dict[str, Any]],
    crf,
    svm,
    context_chars: int,
    svm_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    predictions = []
    for doc in tqdm(docs, desc="traditional predict", dynamic_ncols=True):
        features, _, tokens = doc_to_crf_item(doc, with_labels=False)
        if not features:
            predictions.append({"id": doc["id"], "spans": []})
            continue
        tags = crf.predict_single(features)
        spans = tags_to_binary_spans(tags, tokens)
        if spans:
            texts = [span_context_text(doc, span, context_chars) for span in spans]
            label_sets = predict_svm_labels(svm, texts, svm_threshold)
            classified = []
            for span, label_set in zip(spans, label_sets):
                if int(span["start"]) >= int(span["end"]):
                    continue
                for technique in label_set:
                    classified.append(
                        {"start": int(span["start"]), "end": int(span["end"]), "technique": technique}
                    )
            spans = classified
        predictions.append({"id": doc["id"], "spans": sort_spans(spans)})
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Traditional CRF + TF-IDF Linear SVM propaganda baseline.")
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--splits", nargs="+", default=["id_eval", "eval"])
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--out_root", default="outputs", help="Root output directory; predictions go under outputs/<dataset>/<run>/predictions.")
    parser.add_argument("--model_root", default="outputs", help="Root output directory; model goes under outputs/<dataset>/<run>/traditional_model.")
    parser.add_argument("--context_chars", type=int, default=160)
    parser.add_argument("--crf_max_iterations", type=int, default=100)
    parser.add_argument("--crf_c1", type=float, default=0.1)
    parser.add_argument("--crf_c2", type=float, default=0.1)
    parser.add_argument("--crf_verbose", action="store_true", help="Print sklearn-crfsuite L-BFGS iteration logs.")
    parser.add_argument("--svm_max_iter", type=int, default=5000)
    parser.add_argument("--max_train_docs", type=int, default=None, help="Limit training documents for large-dataset smoke runs.")
    parser.add_argument("--no_svm_gold_spans", action="store_true", help="Do not include exact gold spans as SVM training examples.")
    parser.add_argument(
        "--no_svm_crf_proposals",
        action="store_true",
        help="Do not add CRF-predicted train spans to SVM training examples.",
    )
    parser.add_argument("--svm_boundary_perturb_radius", type=int, default=0)
    parser.add_argument("--svm_overlap_threshold", type=float, default=0.5)
    parser.add_argument(
        "--svm_overlap_mode",
        choices=["iou", "gold_coverage", "candidate_coverage", "either_coverage"],
        default="gold_coverage",
    )
    parser.add_argument("--svm_threshold", type=float, default=0.0, help="Decision threshold for multi-label SVM prediction.")
    parser.add_argument("--max_svm_examples", type=int, default=None, help="Cap SVM span examples after extraction.")
    args = parser.parse_args()

    cfg = load_yaml(args.dataset_config)
    dataset = str(cfg["name"])
    run_name = args.run_name or "crf_svm"
    processed_dir = Path(cfg["processed_dir"])
    labels = get_labels(cfg)
    train_docs = read_jsonl(processed_dir / "train.jsonl")
    if args.max_train_docs:
        train_docs = train_docs[: args.max_train_docs]
    out_base = Path(args.out_root)
    if out_base.name != dataset:
        out_base = out_base / dataset
    model_base = Path(args.model_root)
    if model_base.name != dataset:
        model_base = model_base / dataset
    run_root = out_base / run_name

    print(f"training CRF span detector on {len(train_docs)} docs", flush=True)
    crf = train_crf(train_docs, args.crf_max_iterations, args.crf_c1, args.crf_c2, args.crf_verbose)
    print("training multi-label TF-IDF LinearSVM technique classifier", flush=True)
    svm = train_svm(
        train_docs,
        labels,
        args.context_chars,
        args.svm_max_iter,
        crf=crf,
        include_gold_spans=not args.no_svm_gold_spans,
        include_crf_proposals=not args.no_svm_crf_proposals,
        boundary_perturb_radius=args.svm_boundary_perturb_radius,
        overlap_threshold=args.svm_overlap_threshold,
        overlap_mode=args.svm_overlap_mode,
        max_examples=args.max_svm_examples,
    )

    model_dir = model_base / run_name / "traditional_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving traditional model to {model_dir}", flush=True)
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - environment-specific guidance.
        raise missing_dependency_error(exc) from exc

    joblib.dump(
        {
            "crf": crf,
            "svm": svm,
            "labels": labels,
            "context_chars": args.context_chars,
            "svm_threshold": args.svm_threshold,
        },
        model_dir / "model.joblib",
    )
    write_json(
        model_dir / "meta.json",
        {
            "run_name": run_name,
            "dataset": dataset,
            "method": "crf_svm",
            "context_chars": args.context_chars,
            "crf_max_iterations": args.crf_max_iterations,
            "crf_c1": args.crf_c1,
            "crf_c2": args.crf_c2,
            "svm_max_iter": args.svm_max_iter,
            "svm_multi_label": True,
            "svm_include_gold_spans": not args.no_svm_gold_spans,
            "svm_include_crf_proposals": not args.no_svm_crf_proposals,
            "svm_boundary_perturb_radius": args.svm_boundary_perturb_radius,
            "svm_overlap_threshold": args.svm_overlap_threshold,
            "svm_overlap_mode": args.svm_overlap_mode,
            "svm_threshold": args.svm_threshold,
            "max_train_docs": args.max_train_docs,
            "max_svm_examples": args.max_svm_examples,
            "labels": labels,
            "active_labels": svm.get("active_labels", labels) if isinstance(svm, dict) else labels,
        },
    )

    for split in args.splits:
        docs = read_jsonl(processed_dir / f"{split}.jsonl")
        preds = predict_docs(docs, crf, svm, args.context_chars, svm_threshold=args.svm_threshold)
        out_path = run_root / "predictions" / f"{split}.jsonl"
        write_jsonl(out_path, preds)
        n_spans = sum(len(row["spans"]) for row in preds)
        print(f"{split}: {len(preds)} docs, {n_spans} spans -> {out_path}", flush=True)


if __name__ == "__main__":
    from src.common.process import set_process_title

    set_process_title("GLH")
    main()
