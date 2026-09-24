"""Run unmodified PROACT scripts with a fixed-ten-task dataset adapter."""

import argparse
import runpy
import sys
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
    parser.add_argument(
        "script", choices=("main_baselines.py", "main_inv.py", "main_brainwash.py")
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    task = task_index(args.incoming_task)
    ensure_output_path(Path.cwd())
    arguments = args.arguments
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if args.script == "main_baselines.py":
        if (
            "--tasknum" not in arguments
            or arguments[arguments.index("--tasknum") + 1] != "10"
        ):
            raise ValueError("Always pass --tasknum 10; do not repartition CIFAR.")
        if (
            "--lasttask" not in arguments
            or int(arguments[arguments.index("--lasttask") + 1]) != task
        ):
            raise ValueError("--lasttask must match incoming-task.")
        if "--checkpoint" in arguments:
            import approaches.data_utils as data

            original = data.generate_split_cifar100_tasks

            def limited(*positional, **keywords):
                return truncate_training_tasks(original(*positional, **keywords), task)

            data.generate_split_cifar100_tasks = limited
    else:
        import data_utils

        data_utils.get_dataset_specs = fixed_dataset_specs
        from detector.common import load_pickle
        from detector.transfer_core import validate_checkpoint

        flag = "--pretrained_model_add"
        validate_checkpoint(load_pickle(arguments[arguments.index(flag) + 1]), task)
    sys.argv = [args.script] + arguments
    runpy.run_path(str(DETECTOR_ROOT.parent / args.script), run_name="__main__")


if __name__ == "__main__":
    main()
