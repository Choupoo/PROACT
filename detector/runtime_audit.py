"""Read-only runtime accounting for an existing study; no CUDA or retraining."""

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

from detector.advisor_imagenet import stage_category
from detector.common import save_json
from detector.io_utils import ensure_output_path


def audit(run_dir):
    root = Path(run_dir).expanduser().resolve()
    paths = sorted(root.glob("*/seed*/run_state.json"))
    if not paths:
        raise FileNotFoundError("No method/seed run_state.json files under " + str(root))
    cells = []
    for path in paths:
        state = json.loads(path.read_text())
        groups = defaultdict(list)
        missing_timing = []
        for name, record in state.items():
            if record.get("status") != "complete":
                continue
            elapsed = record.get("elapsed_seconds")
            if elapsed is None and record.get("started") and record.get("finished"):
                elapsed = (datetime.fromisoformat(record["finished"]) -
                           datetime.fromisoformat(record["started"])).total_seconds()
            if elapsed is None:
                missing_timing.append(name)
                continue
            if not 0 <= float(elapsed) < float("inf"):
                raise ValueError("Invalid duration for " + name)
            groups[stage_category(name)].append(float(elapsed))
        categories = {k: dict(stages=len(v), hours=sum(v) / 3600) for k, v in groups.items()}
        total = sum(r["hours"] for r in categories.values())
        cells.append(dict(method=path.parent.parent.name, seed=path.parent.name,
                          total_stage_wall_hours=total, categories=categories,
                          missing_timing=missing_timing))
    return dict(cells=cells, input_run=str(root),
                interpretation="Sum of recorded subprocess wall durations, not measured GPU kernel time. Parallel cells overlap.",
                repeated_work=[
                    "Legacy target clean continuation runs four times per cell; the new plan runs it once.",
                    "Legacy uniform continuation runs twice per budget; the new plan runs it once per budget.",
                    "Legacy clean/random feature measurements repeat across attacks; the new plan reuses matching frozen-model controls.",
                    "ImageNet attack evaluation now loads and partitions prepared arrays once per subprocess, with fresh dataset wrappers and unchanged RNG resets.",
                ],
                retained_work=[
                    "Six distinct BrainWash artifacts remain: Task 1 and Task 4 reckless/0.3, plus four Task 9 attacks.",
                    "Different stage checkpoints require their own inversions and feature references.",
                    "Attack epochs (5000), victim epochs (100), full images and inversion iterations (2000) are unchanged.",
                    "Pseudo-gradient model creation remains in the unmodified upstream attack loop; no unvalidated functional rewrite is enabled.",
                ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = ensure_output_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = audit(args.run_dir)
    save_json(result, output / "runtime_audit.json")
    lines = ["# Runtime audit", "", result["interpretation"], "",
             "| Method | Seed | Total stage wall hours | Attack hours | Continuation hours |",
             "| --- | --- | ---: | ---: | ---: |"]
    for cell in result["cells"]:
        groups = cell["categories"]
        lines.append("| {} | {} | {:.2f} | {:.2f} | {:.2f} |".format(cell["method"], cell["seed"],
            cell["total_stage_wall_hours"], groups.get("attack", {}).get("hours", 0),
            groups.get("continuation_training", {}).get("hours", 0)))
    lines += ["", "## Repeated work removed from the new plan", ""] + ["- " + s for s in result["repeated_work"]]
    lines += ["", "## Work retained", ""] + ["- " + s for s in result["retained_work"]]
    (output / "runtime_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved runtime audit:", output)


if __name__ == "__main__":
    main()
