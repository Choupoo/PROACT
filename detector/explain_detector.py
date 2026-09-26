"""Exact interventional linear SHAP on logistic log-odds, with train background.

No SHAP package is required: phi_j = beta_j * (z_j - mean_train(z_j)).
This is an additive explanation of the fitted linear logit, not probability,
not conditional/correlation-aware SHAP, and not evidence of causal importance.
"""

import argparse
import os

import numpy as np
import pandas as pd

from detector.common import save_json, sha256_file
from detector.io_utils import (
    DETECTOR_ROOT,
    assert_compatible_provenance,
    feature_provenance,
    load_frozen_bundle,
    read_feature_table,
)
from detector.train_detector import compare_feature_sets, validate_feature_table
from detector.transfer_detector import fresh_output, prepare_features

SHAP_SOURCE = (
    "https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html"
)


def linear_shap(bundle, background, samples):
    background = prepare_features(background, bundle.get("feature_set"))
    samples = prepare_features(samples, bundle.get("feature_set"))
    columns = bundle["feature_columns"]
    classifier, scaler = bundle["classifier"], bundle["scaler"]
    if list(classifier.classes_) != [0, 1] or classifier.coef_.shape != (
        1,
        len(columns),
    ):
        raise ValueError("Need binary logistic clean=0 / poison=1 with a linear logit.")
    if background.empty or samples.empty:
        raise ValueError("Background and explained rows must be nonempty.")
    z_background = scaler.transform(background[columns].to_numpy(dtype=float))
    z = scaler.transform(samples[columns].to_numpy(dtype=float))
    mean = z_background.mean(axis=0)
    beta = classifier.coef_[0]
    phi = (z - mean) * beta
    base = float(classifier.intercept_[0] + mean.dot(beta))
    logits = classifier.decision_function(z)
    error = float(np.max(np.abs(phi.sum(axis=1) + base - logits)))
    if not np.isfinite(phi).all() or not np.allclose(
        phi.sum(axis=1) + base, logits, rtol=1e-10, atol=1e-10
    ):
        raise ValueError("SHAP additivity failed.")
    return phi, base, logits, error


def feature_family(name):
    if name.startswith("grad_shape_stage_"):
        return name[len("grad_shape_stage_") :]
    if name.startswith("grad_norm_stage_"):
        return name[len("grad_norm_stage_") :]
    if name.startswith(("grad_norm_param__", "grad_norm_layer__")):
        group = name.split("__", 1)[1].split(".", 1)[0]
        return "stem" if group in ("conv1", "bn1") else group
    if name.startswith("grad_cosine"):
        return "historical_gradient_similarity"
    if name in ("entropy", "confidence", "true_class_probability", "margin"):
        return "uncertainty"
    return {
        "loss": "loss",
        "grad_norm_l2": "global_gradient",
        "activation_norm_l2": "activation",
    }.get(name, name)


def explain(table, metadata, bundle, output, split="validation", ablations=False):
    validate_feature_table(table)
    if bundle.get("negative_policy", "clean_only") != "clean_only" and ablations:
        raise ValueError(
            "Legacy ablations omit random-control negatives; use the registered negative-policy comparison."
        )
    if bundle.get("feature_set") == "shape" and ablations:
        raise ValueError(
            "Legacy raw-feature ablations do not apply to the shape model."
        )
    table = prepare_features(table, bundle.get("feature_set"))
    assert_compatible_provenance(bundle["provenance"], feature_provenance(metadata))
    expected_hash = bundle.get("features_sha256", bundle.get("input_file_sha256"))
    if expected_hash is not None and expected_hash != metadata["features_sha256"]:
        raise ValueError(
            "Explanation requires the original frozen training feature file."
        )
    fit_ids = set(bundle["fit_original_indices"])
    background = table.loc[
        table.original_index.isin(fit_ids)
        & table.view.isin(bundle.get("fit_views", ["clean", "poison"]))
    ]
    if set(background.original_index) != fit_ids or set(background.split) != {"train"}:
        raise ValueError(
            "Background must exactly reproduce frozen training identities."
        )
    samples = table.loc[table.split == split].copy()
    if set(samples.original_index) & fit_ids:
        raise ValueError("Explanation rows overlap training originals.")
    phi, base, logits, error = linear_shap(bundle, background, samples)
    columns = bundle["feature_columns"]
    z = bundle["scaler"].transform(samples[columns].to_numpy(dtype=float))
    scores = bundle["classifier"].predict_proba(z)[:, 1]
    positive = scores >= bundle["threshold"]
    labels = samples.detector_label.to_numpy()
    outcomes = np.where(
        labels == -1,
        "random_control",
        np.where(
            labels == 1, np.where(positive, "TP", "FN"), np.where(positive, "FP", "TN")
        ),
    )
    ids = samples[["original_index", "view", "detector_label"]].reset_index(drop=True)
    ids["outcome"] = outcomes
    ids["score"] = scores
    ids["logit"] = logits
    ids["base_logit"] = base
    contributions = pd.DataFrame(phi, columns=columns)
    pd.concat([ids, contributions.add_prefix("shap_logit__")], axis=1).to_csv(
        output / "sample_shap.csv", index=False
    )
    paired = labels != -1
    importance = pd.DataFrame(
        {
            "feature": columns,
            "family": [feature_family(c) for c in columns],
            "mean_abs_shap_logit": np.abs(phi[paired]).mean(axis=0),
            "standardized_coefficient": bundle["classifier"].coef_[0],
        }
    )
    importance = importance.sort_values("mean_abs_shap_logit", ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    grouped = (
        importance.groupby("family", as_index=False)
        .mean_abs_shap_logit.sum()
        .sort_values("mean_abs_shap_logit", ascending=False)
    )
    grouped.to_csv(output / "family_importance.csv", index=False)
    correlation = background[columns].corr()
    correlation.to_csv(output / "train_feature_correlations.csv")
    examples = []
    for outcome in ("TP", "FN", "FP", "TN", "random_control"):
        indices = np.flatnonzero(outcomes == outcome)
        if not len(indices):
            continue
        # Deterministic representative: first original ID, not maximum attribution.
        i = int(indices[0])
        strongest = np.argsort(-np.abs(phi[i]))[:5]
        examples.append(
            {
                "outcome": outcome,
                "original_index": int(ids.iloc[i].original_index),
                "view": str(ids.iloc[i]["view"]),
                "score": float(scores[i]),
                "top_contributions": [
                    {"feature": columns[j], "shap_logit": float(phi[i, j])}
                    for j in strongest
                ],
            }
        )
    save_json(examples, output / "examples.json")
    if ablations:
        if split != "validation":
            raise ValueError("Feature comparisons are restricted to validation.")
        validation = table.loc[
            (table.split == "validation") & table.view.isin(["clean", "poison"])
        ]
        comparison = compare_feature_sets(background, validation, 20260723, 0.05)
        comparison.to_csv(output / "feature_comparison_validation.csv", index=False)
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    os.environ["FONTCONFIG_FILE"] = str(DETECTOR_ROOT / "plot-fontconfig.xml")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, frame, label in (
        ("feature_importance", importance.head(25), "feature"),
        ("family_importance", grouped, "family"),
    ):
        frame = frame.iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(frame) + 1)))
        ax.barh(frame[label], frame.mean_abs_shap_logit)
        ax.set_xlabel("Mean absolute interventional SHAP value (log-odds)")
        title = "{} clean/poison; training background".format(split)
        if metadata.get("synthetic", False):
            title = "SYNTHETIC FIXTURE ONLY\n" + title
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output / (name + ".png"), dpi=160)
        plt.close(fig)
    report = {
        "method": "exact_interventional_linear_shap_logit",
        "split": split,
        "background_split": "train",
        "background_originals": len(fit_ids),
        "background_rows": len(background),
        "explained_rows": len(samples),
        "base_logit": base,
        "max_additivity_error": error,
        "source": SHAP_SOURCE,
        "test_used_for_selection": False,
        "feature_columns": columns,
        "negative_policy": bundle.get("negative_policy", "clean_only"),
        "background_views": bundle.get("fit_views", ["clean", "poison"]),
        "examples": examples,
        "synthetic": bool(metadata.get("synthetic", False)),
        "limitations": "Marginal/interventional attribution ignores conditional feature dependence; correlated gradient features may share or substitute importance. Not causal, not probability contributions.",
    }
    save_json(report, output / "explanation.json")
    lines = [
        "# Feature explanation",
        "",
        "SYNTHETIC FIXTURE ONLY"
        if report["synthetic"]
        else "Explanation of the supplied frozen model.",
        "",
        "Exact interventional SHAP for the logistic logit; training rows supply the background.",
        "Additivity error: {:.3g}. Explained split: {}.".format(error, split),
        "",
        "| Feature | Mean absolute SHAP (log-odds) |",
        "| --- | ---: |",
    ]
    lines += [
        "| {} | {:.6g} |".format(row.feature, row.mean_abs_shap_logit)
        for row in importance.itertuples()
    ]
    lines += [
        "",
        report["limitations"],
        "",
        "A zero coefficient is not evidence that a feature cannot transfer to another task.",
        "Test explanations are descriptive only; no automatic feature selection.",
        "",
        "Reference: " + SHAP_SOURCE,
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(args):
    if args.ablations and args.split != "validation":
        raise ValueError("Ablations are validation-only.")
    output = fresh_output(args.output_dir)
    table, metadata = read_feature_table(args.features)
    bundle = load_frozen_bundle(args.bundle)
    before = sha256_file(args.bundle)
    output.mkdir(parents=True)
    report = explain(table, metadata, bundle, output, args.split, args.ablations)
    if sha256_file(args.bundle) != before:
        raise RuntimeError("Frozen bundle changed during explanation.")
    save_json(
        {
            "bundle_sha256": before,
            "features_sha256": metadata["features_sha256"],
            "status": "complete",
            "method": report["method"],
        },
        output / "run_state.json",
    )
    print("Saved explanations:", output)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--ablations", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
