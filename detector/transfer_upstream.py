"""Run unmodified PROACT scripts with a registered fixed-partition adapter."""

import argparse
import runpy
import sys
import json
from pathlib import Path

from detector.io_utils import DETECTOR_ROOT, ensure_output_path
from detector.transfer_core import (
    fixed_dataset_specs,
    task_index,
    truncate_training_tasks,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming-task", type=int, required=True)
    parser.add_argument("--data-manifest", help="Full prepared ImageNet data registration JSON.")
    parser.add_argument(
        "script", choices=("main_baselines.py", "main_inv.py", "main_brainwash.py")
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    ensure_output_path(Path.cwd())
    arguments = args.arguments
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    prepared = None
    if args.data_manifest:
        from detector.image_data import PreparedData
        prepared = PreparedData(json.loads(Path(args.data_manifest).read_text()))
    from detector.image_data import dataset_spec
    tasks = dataset_spec(prepared.record["dataset"])["tasks"] if prepared is not None else 10
    task = task_index(args.incoming_task, tasks)
    if args.script == "main_baselines.py":
        if (
            "--tasknum" not in arguments
            or arguments[arguments.index("--tasknum") + 1] != str(tasks)
        ):
            raise ValueError("--tasknum must match the registered {}-task partition.".format(tasks))
        if (
            "--lasttask" not in arguments
            or int(arguments[arguments.index("--lasttask") + 1]) != task
        ):
            raise ValueError("--lasttask must match incoming-task.")
        if prepared is not None:
            import approaches.data_utils as data
            name = prepared.record["dataset"]
            if arguments[arguments.index("--experiment") + 1] != name:
                raise ValueError("Training experiment differs from data registration.")
            function = "generate_" + name + "_tasks"
            # Upstream's ImageNet call omits seed and would reset model-training
            # RNGs to zero even when --seed is 6/7/8. Make the registered seed
            # effective, as it already is for upstream's CIFAR loader call.
            def image_training(*a, **kw):
                kw["seed"] = int(arguments[arguments.index("--seed") + 1])
                return prepared.training(task, "--checkpoint" in arguments, *a, **kw)

            setattr(data, function, image_training)
        elif "--checkpoint" in arguments:
            import approaches.data_utils as data

            original = data.generate_split_cifar100_tasks

            def limited(*positional, **keywords):
                return truncate_training_tasks(original(*positional, **keywords), task)

            data.generate_split_cifar100_tasks = limited
    else:
        import data_utils

        data_utils.get_dataset_specs = prepared.specs if prepared is not None else fixed_dataset_specs
        from detector.common import load_pickle
        from detector.transfer_core import validate_checkpoint

        flag = "--pretrained_model_add"
        checkpoint = load_pickle(arguments[arguments.index(flag) + 1])
        validate_checkpoint(checkpoint, task)
        if prepared is not None and checkpoint["dataset"] != prepared.record["dataset"]:
            raise ValueError("Checkpoint differs from the registered dataset.")
    sys.argv = [args.script] + arguments
    runpy.run_path(str(DETECTOR_ROOT.parent / args.script), run_name="__main__")


if __name__ == "__main__":
    main()
