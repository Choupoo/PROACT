import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def diagnose(require_cuda=False):
    distributions = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "joblib",
        "torch",
        "torchvision",
        "PyYAML",
        "matplotlib",
        "seaborn",
        "tqdm",
    ]
    versions = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    programs = {
        "numerical": "import numpy,pandas,scipy,sklearn,joblib; print('ok')",
        "torch": "import torch,json; print(json.dumps({'cuda':torch.cuda.is_available(),'version':torch.__version__}))",
        "upstream_imports": "import torchvision, yaml; print('ok')",
    }
    checks = {}
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    root = Path(__file__).resolve().parent
    env["PYTHONPATH"] = str(root.parent) + os.pathsep + env.get("PYTHONPATH", "")
    # Do not import pyplot/approaches.utils: they may write a font cache.
    for name, source in programs.items():
        try:
            result = subprocess.run(
                [sys.executable, "-B", "-u", "-c", source],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            checks[name] = {
                "ok": result.returncode == 0,
                "returncode": result.returncode,
                "output": (result.stdout + result.stderr).strip()[-2000:],
            }
        except subprocess.TimeoutExpired:
            checks[name] = {"ok": False, "output": "Import timed out after 30 seconds."}
    cuda = False
    if checks["torch"]["ok"]:
        cuda = json.loads(checks["torch"]["output"])["cuda"]
    success = (
        all(value is not None for value in versions.values())
        and all(item["ok"] for item in checks.values())
        and (cuda or not require_cuda)
    )
    return {
        "python": sys.version,
        "executable": sys.executable,
        "versions": versions,
        "checks": checks,
        "cuda_available": cuda,
        "ready": success,
        "note": "CUDA is required for upstream bootstrap. CPU tests/demo do not require torchvision. Matching torch/torchvision binaries must import successfully for real feature extraction.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    result = diagnose(parser.parse_args().require_cuda)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result["ready"] else 1)
