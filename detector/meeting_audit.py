"""Inventory reported metrics/feature sets without guessing a meeting result."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from detector.common import save_json, sha256_file
from detector.io_utils import load_frozen_bundle
from detector.transfer_detector import fresh_output


def inventory(roots):
    records, errors = [], []
    for root in roots:
        root = Path(root).resolve()
        if not root.exists():
            errors.append({"path": str(root), "error": "missing input"})
            continue
        paths = sorted(root.rglob("*.json")) if root.is_dir() else [root]
        for path in paths:
            if path.name not in (
                "metrics.json",
                "fit_metrics.json",
                "evaluation_metrics.json",
                "run_config.json",
            ):
                continue
            try:
                value = json.loads(path.read_text())
                base = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "protocol": value.get("feature_protocol", value.get("protocol")),
                    "feature_set": value.get("feature_set"),
                    "feature_columns": value.get("feature_columns"),
                    "threshold": value.get("threshold"),
                    "kind": "metadata",
                }
                found = False
                for split in ("validation", "test", "sample_metrics"):
                    result = value.get(split)
                    if isinstance(result, dict) and "poison_tpr" in result:
                        row = dict(
                            base,
                            kind="sample_metrics",
                            split=split,
                            **{
                                key: result.get(key)
                                for key in (
                                    "roc_auc",
                                    "poison_tpr",
                                    "clean_fpr",
                                    "threshold",
                                )
                            },
                        )
                        records.append(row)
                        found = True
                if not found:
                    records.append(base)
            except (ValueError, OSError) as exc:
                errors.append({"path": str(path), "error": str(exc)})
        if root.is_dir():
            for path in sorted(root.rglob("feature_comparison_validation.csv")):
                try:
                    for row in pd.read_csv(path).to_dict("records"):
                        records.append(
                            {
                                "path": str(path),
                                "sha256": sha256_file(path),
                                "kind": "validation_ablation",
                                "feature_set": row.get("feature_set"),
                                "feature_columns": row.get("feature_columns"),
                                "poison_tpr": row.get("validation_poison_tpr"),
                                "clean_fpr": row.get("validation_clean_fpr"),
                                "roc_auc": row.get("validation_roc_auc"),
                            }
                        )
                except (ValueError, OSError) as exc:
                    errors.append({"path": str(path), "error": str(exc)})
            for path in sorted(root.rglob("*.joblib")):
                try:
                    value = load_frozen_bundle(path)
                    if isinstance(value, dict) and "feature_columns" in value:
                        records.append(
                            {
                                "path": str(path),
                                "sha256": sha256_file(path),
                                "kind": "verified_bundle",
                                "feature_set": value.get("feature_set"),
                                "feature_columns": value["feature_columns"],
                                "threshold": value.get("threshold"),
                            }
                        )
                except (ValueError, OSError) as exc:
                    errors.append({"path": str(path), "error": str(exc)})
    matches = [
        r
        for r in records
        if r.get("poison_tpr") is not None
        and np.isclose(r["poison_tpr"], 0.864, atol=0.0005)
    ]
    return {
        "records": records,
        "errors": errors,
        "candidate_86_4_percent_records": matches,
        "claim_status": "candidate_matches_need_context"
        if matches
        else "86.4 percent not substantiated by supplied artifacts",
        "caution": "A metric match alone does not identify the slide, split, feature set or frozen classifier. No historical result is relabelled.",
    }


def main(args):
    output = fresh_output(args.output_dir)
    result = inventory(args.roots)
    output.mkdir(parents=True)
    save_json(result, output / "audit.json")
    lines = [
        "# Meeting 3 result inventory",
        "",
        result["claim_status"],
        "",
        result["caution"],
        "",
        "| File | Kind/split | Feature set | Poison TPR | Clean FPR |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in result["records"]:

        def fmt(key):
            return "unavailable" if row.get(key) is None else "{:.2%}".format(row[key])

        lines.append(
            "| {} | {} / {} | {} | {} | {} |".format(
                row["path"],
                row["kind"],
                row.get("split", ""),
                row.get("feature_set") or "see JSON",
                fmt("poison_tpr"),
                fmt("clean_fpr"),
            )
        )
    lines += [
        "",
        "The configured extended set contains activation_norm_l2 and historical gradient similarities.",
        "Meeting claims that those were removed require the exact corresponding bundle/configuration.",
        "Existing MMD already compares detector descriptors, not full backbone embeddings.",
        "The three-feature legacy results are not the meeting's extended-feature result.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved result inventory:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    main(parser.parse_args())
