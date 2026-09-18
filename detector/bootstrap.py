import argparse
import shlex
import sys

import numpy as np
import torch

from detector.common import save_json, sha256_file
from detector.io_utils import DETECTOR_ROOT, ensure_output_path
from detector.pipeline import child_environment, execute


def build_commands(args):
    root = ensure_output_path(args.output_dir)
    victim = root / "victim" / "checkpoint.pkl"
    artifact = root / "attack" / "noise.pkl"
    common = [
        "--experiment",
        "split_cifar100",
        "--approach",
        "ewc",
        "--lasttask",
        "9",
        "--tasknum",
        "10",
        "--nepochs",
        str(args.victim_epochs),
        "--batch-size",
        "16",
        "--lamb",
        "500000",
        "--clip",
        "100.0",
        "--lr",
        "0.01",
        "--seed",
        str(args.seed),
    ]
    definitions = [
        (
            "victim",
            "main_baselines.py",
            common + ["--output_dir", root / "victim"],
            victim,
        ),
        (
            "inversion",
            "main_inv.py",
            [
                "--pretrained_model_add",
                victim,
                "--num_samples",
                "128",
                # Upstream duplicates save_dir in filenames: it MUST be a basename.
                "--save_dir",
                "inversions",
                "--task_lst",
                "0,1,2,3,4,5,6,7,8",
                "--save_every",
                "1000",
                "--batch_reg",
                "--init_acc",
                "--n_iters",
                str(args.inversion_iters),
            ],
            root / "inversions" / "inversions_tid_08.npz",
        ),
        (
            "attack",
            "main_brainwash.py",
            [
                "--pretrained_model_add",
                victim,
                "--mode",
                "reckless",
                "--target_task_for_eval",
                "0",
                "--delta",
                str(args.delta),
                "--seed",
                str(args.seed),
                "--eval_every",
                "10",
                "--distill_folder",
                root / "inversions",
                "--init_acc",
                "--noise_norm",
                "inf",
                "--cont_learner_lr",
                "0.001",
                "--n_epochs",
                str(args.attack_epochs),
                "--n_iters",
                "1",
                "--reset_head_every",
                "1",
                "--save_every",
                "100",
                "--output_dir",
                root / "attack",
                "--grads_track_every",
                str(args.attack_epochs + 1),
            ],
            artifact,
        ),
        (
            "clean_training",
            "main_baselines.py",
            common
            + [
                "--checkpoint",
                artifact,
                "--init_acc",
                "--output_dir",
                root / "effectiveness" / "clean",
            ],
            root / "effectiveness" / "clean" / "acc_mat_clean.npy",
        ),
        (
            "poison_training",
            "main_baselines.py",
            common
            + [
                "--checkpoint",
                artifact,
                "--init_acc",
                "--addnoise",
                "--output_dir",
                root / "effectiveness" / "poison",
            ],
            root / "effectiveness" / "poison" / "acc_mat_ours.npy",
        ),
    ]
    return [
        (
            name,
            [sys.executable, "-B", "-u", str(DETECTOR_ROOT.parent / script)]
            + [str(arg) for arg in argv],
            output,
        )
        for name, script, argv, output in definitions
    ]


def attack_effectiveness(clean_path, poison_path):
    """Compute backward transfer from the actually observed accuracy matrices."""
    result = {}
    for label, path in (("clean", clean_path), ("poison", poison_path)):
        matrix = np.load(path, allow_pickle=False)
        if (
            matrix.shape != (10, 10)
            or not np.isfinite(matrix).all()
            or np.any((matrix < 0) | (matrix > 1))
        ):
            raise ValueError("Expected a finite 10 x 10 accuracy matrix in [0,1].")
        result[label] = {
            "past_task_mean_accuracy": float(matrix[9, :9].mean()),
            "incoming_task_accuracy": float(matrix[9, 9]),
            "backward_transfer": float((matrix[9, :9] - np.diag(matrix)[:9]).mean()),
            "sha256": sha256_file(path),
        }
    result["past_accuracy_drop_clean_minus_poison"] = (
        result["clean"]["past_task_mean_accuracy"]
        - result["poison"]["past_task_mean_accuracy"]
    )
    result["note"] = (
        "One paired attack-effectiveness experiment, not a multi-seed significance test. Positive drop means poisoning worsened mean historical accuracy in this run."
    )
    return result


def main(args):
    if args.attack_epochs < 100 or args.attack_epochs % 100:
        raise ValueError(
            "attack_epochs must be a positive multiple of 100 so upstream saves the final noise."
        )
    if (
        args.victim_epochs < 1
        or args.inversion_iters < 1
        or not np.isfinite(args.delta)
        or args.delta <= 0
    ):
        raise ValueError("Epochs, inversion iterations and delta must be positive.")
    plan = build_commands(args)
    if args.stage != "all":
        selected = (
            {"clean_training", "poison_training"}
            if args.stage == "effectiveness"
            else {args.stage}
        )
        plan = [item for item in plan if item[0] in selected]
    root = ensure_output_path(args.output_dir)
    if args.dry_run:
        print("Working directory (CIFAR will be stored in its data/):", root)
        for name, command, _ in plan:
            print("\n[{}]\n{}".format(name, shlex.join(command)))
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Upstream PROACT training requires CUDA. Use a GPU environment, or run detector.demo for CPU integration verification."
        )
    root.mkdir(parents=True, exist_ok=True)
    env = child_environment(root)
    for name, command, expected_output in plan:
        if expected_output.exists():
            raise FileExistsError(
                "Refusing to overwrite {}. Choose another output-dir or --stage for unfinished work.".format(
                    expected_output
                )
            )
        execute(command, cwd=root, log_path=root / "logs" / (name + ".log"), env=env)
        if not expected_output.is_file():
            raise RuntimeError(
                "Upstream exited without expected artifact: {}".format(expected_output)
            )
        save_json(
            {
                "command": command,
                "cwd": str(root),
                "output_sha256": sha256_file(expected_output),
            },
            root / (name + ".run.json"),
        )
    if args.stage in ("all", "effectiveness"):
        result = attack_effectiveness(
            root / "effectiveness" / "clean" / "acc_mat_clean.npy",
            root / "effectiveness" / "poison" / "acc_mat_ours.npy",
        )
        save_json(result, root / "effectiveness" / "metrics.json")
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default=str(DETECTOR_ROOT / "work" / "artifacts")
    )
    parser.add_argument(
        "--stage",
        choices=("all", "victim", "inversion", "attack", "effectiveness"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--victim-epochs", type=int, default=20)
    parser.add_argument("--inversion-iters", type=int, default=10000)
    parser.add_argument("--attack-epochs", type=int, default=5000)
    parser.add_argument("--delta", type=float, default=0.3)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
